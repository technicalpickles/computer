"""Pure protocol tests for `AcpServerConnection` (Tier 0: no network, no DB).

Drives a real `AcpServerConnection` over `InMemoryTransport.connected_pair()` against
a stub `SessionBackend`, exercising handshake/dispatch/error-code behavior in
isolation. The final class also dogfoods cptr's own `AcpClient` as the peer, so the
server and client sides of the protocol are validated against each other directly.
"""

from __future__ import annotations

import asyncio

import pytest

from cptr.utils.acp_server import AcpServerConnection
from cptr.utils.agents.acp import AcpClient
from cptr.utils.agents.acp_transport import TransportClosed, connected_pair


class StubBackend:
    """Records calls; session id / load result / errors are configurable."""

    def __init__(self, *, session_id: str = "server-sess-1", load_result: bool = True) -> None:
        self.session_id = session_id
        self.load_result = load_result
        self.new_session_calls: list[tuple[str, dict]] = []
        self.load_session_calls: list[tuple[str, str]] = []
        self.prompt_calls: list[tuple[str, list]] = []
        self.cancel_calls: list[str] = []
        self.new_session_error: Exception | None = None

    async def new_session(self, cwd, params):
        self.new_session_calls.append((cwd, params))
        if self.new_session_error is not None:
            raise self.new_session_error
        return self.session_id

    async def load_session(self, session_id, cwd):
        self.load_session_calls.append((session_id, cwd))
        return self.load_result

    async def prompt(self, session_id, prompt_items):
        self.prompt_calls.append((session_id, prompt_items))
        raise NotImplementedError

    async def cancel(self, session_id):
        self.cancel_calls.append(session_id)


class RecordingTransport:
    """Wraps an `AcpTransport`, recording every message that passes through `receive()`."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.received: list[dict] = []

    async def start(self):
        await self.inner.start()

    async def send(self, message):
        await self.inner.send(message)

    async def receive(self):
        message = await self.inner.receive()
        self.received.append(message)
        return message

    async def close(self):
        await self.inner.close()


@pytest.fixture
async def wired():
    """(client_transport, server_transport, backend, connection, serve_task)."""
    client_transport, server_transport = connected_pair()
    backend = StubBackend()
    connection = AcpServerConnection(server_transport, backend)
    task = asyncio.create_task(connection.serve())
    yield client_transport, server_transport, backend, connection, task
    if not task.done():
        task.cancel()


async def _teardown(client_transport, task):
    await client_transport.close()
    with contextlib_suppress():
        await asyncio.wait_for(task, timeout=2)
    _assert_task_finished_cleanly(task)


def contextlib_suppress():
    import contextlib

    return contextlib.suppress(asyncio.CancelledError, TransportClosed)


def _assert_task_finished_cleanly(task: asyncio.Task) -> None:
    """The narrowed `contextlib_suppress()` above only swallows expected shutdown
    exceptions, so a `serve()` crash or a `wait_for` timeout now propagates instead of
    silently passing the teardown. This asserts what "finished cleanly" actually means:
    the task is done, and if it wasn't cancelled, it didn't end on an exception either.
    """
    assert task.done()
    if not task.cancelled():
        assert task.exception() is None


async def _request(transport, request_id, method, params=None):
    await transport.send(
        {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
    )
    return await transport.receive()


async def _notify(transport, method, params=None):
    await transport.send({"jsonrpc": "2.0", "method": method, "params": params or {}})


async def _assert_no_message(transport, timeout=0.05):
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(transport.receive(), timeout=timeout)


class TestInitializeHandshake:
    async def test_initialize_result_exact_shape(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        response = await _request(client_transport, 1, "initialize", {"protocolVersion": 1})

        assert response == {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": 1,
                "agentCapabilities": {"loadSession": True},
                "authMethods": [],
                "agentInfo": {"name": "cptr", "version": "0"},
            },
        }
        await _teardown(client_transport, task)

    async def test_request_before_initialize_is_rejected(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        response = await _request(client_transport, 1, "session/new", {"cwd": "/tmp"})

        assert response["error"]["code"] == -32002
        assert "not initialized" in response["error"]["message"]
        await _teardown(client_transport, task)

    async def test_double_initialize_is_idempotent(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        first = await _request(client_transport, 1, "initialize", {})
        second = await _request(client_transport, 2, "initialize", {})

        assert first["result"] == second["result"]
        await _teardown(client_transport, task)

    async def test_authenticate_returns_empty_result_despite_no_advertised_methods(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        response = await _request(client_transport, 2, "authenticate", {"methodId": "whatever"})

        assert response == {"jsonrpc": "2.0", "id": 2, "result": {}}
        await _teardown(client_transport, task)


class TestSessionNew:
    async def test_happy_path_round_trips_cwd_and_params(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        response = await _request(
            client_transport, 2, "session/new", {"cwd": "/workspace/foo", "mcpServers": []}
        )

        assert response["result"] == {"sessionId": "server-sess-1"}
        assert len(backend.new_session_calls) == 1
        cwd, params = backend.new_session_calls[0]
        assert cwd == "/workspace/foo"
        assert params["cwd"] == "/workspace/foo"
        await _teardown(client_transport, task)

    async def test_missing_cwd_is_invalid_params(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        response = await _request(client_transport, 2, "session/new", {})

        assert response["error"]["code"] == -32602
        assert backend.new_session_calls == []
        await _teardown(client_transport, task)

    async def test_non_string_cwd_is_invalid_params(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        response = await _request(client_transport, 2, "session/new", {"cwd": 123})

        assert response["error"]["code"] == -32602
        assert backend.new_session_calls == []
        await _teardown(client_transport, task)

    async def test_relative_cwd_is_invalid_params(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        response = await _request(client_transport, 2, "session/new", {"cwd": "relative/path"})

        assert response["error"]["code"] == -32602
        assert backend.new_session_calls == []
        await _teardown(client_transport, task)

    async def test_cwd_with_dotdot_segment_is_invalid_params(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        response = await _request(client_transport, 2, "session/new", {"cwd": "/workspace/../etc"})

        assert response["error"]["code"] == -32602
        assert backend.new_session_calls == []
        await _teardown(client_transport, task)

    async def test_oversized_cwd_is_invalid_params(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        oversized_cwd = "/" + ("a" * 4096)
        response = await _request(client_transport, 2, "session/new", {"cwd": oversized_cwd})

        assert response["error"]["code"] == -32602
        assert backend.new_session_calls == []
        await _teardown(client_transport, task)

    async def test_session_new_cap_enforced(self):
        client_transport, server_transport = connected_pair()
        backend = StubBackend()
        connection = AcpServerConnection(server_transport, backend, session_new_cap=2)
        task = asyncio.create_task(connection.serve())

        await _request(client_transport, 1, "initialize", {})
        first = await _request(client_transport, 2, "session/new", {"cwd": "/tmp"})
        second = await _request(client_transport, 3, "session/new", {"cwd": "/tmp"})
        third = await _request(client_transport, 4, "session/new", {"cwd": "/tmp"})

        assert first["result"]["sessionId"] == "server-sess-1"
        assert second["result"]["sessionId"] == "server-sess-1"
        assert third["error"]["code"] == -32000
        assert len(backend.new_session_calls) == 2

        await _teardown(client_transport, task)


class TestSessionLoad:
    async def test_found_returns_empty_object(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired
        backend.load_result = True

        await _request(client_transport, 1, "initialize", {})
        response = await _request(
            client_transport, 2, "session/load", {"sessionId": "sess-1", "cwd": "/tmp"}
        )

        assert response["result"] == {}
        assert backend.load_session_calls == [("sess-1", "/tmp")]
        await _teardown(client_transport, task)

    async def test_not_found_returns_session_not_found_error(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired
        backend.load_result = False

        await _request(client_transport, 1, "initialize", {})
        response = await _request(
            client_transport, 2, "session/load", {"sessionId": "missing", "cwd": "/tmp"}
        )

        assert response["error"]["code"] == -32001
        await _teardown(client_transport, task)


class TestSessionPromptStub:
    async def test_prompt_returns_method_not_found_and_connection_survives(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        await _request(client_transport, 2, "session/new", {"cwd": "/tmp"})
        response = await _request(
            client_transport,
            3,
            "session/prompt",
            {"sessionId": "server-sess-1", "prompt": []},
        )

        assert response["error"]["code"] == -32601
        assert "not supported" in response["error"]["message"]

        # Connection must still be usable afterwards.
        follow_up = await _request(client_transport, 4, "session/new", {"cwd": "/tmp2"})
        assert follow_up["result"] == {"sessionId": "server-sess-1"}

        await _teardown(client_transport, task)


class TestAuthorizedSessionGate:
    async def test_prompt_against_unauthorized_session_id_is_rejected_without_backend_call(
        self, wired
    ):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        response = await _request(
            client_transport, 2, "session/prompt", {"sessionId": "not-mine", "prompt": []}
        )

        assert response["error"]["code"] == -32001
        assert backend.prompt_calls == []
        await _teardown(client_transport, task)

    async def test_cancel_for_unauthorized_session_id_does_not_reach_backend(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        await _notify(client_transport, "session/cancel", {"sessionId": "not-mine"})

        await _assert_no_message(client_transport)
        assert backend.cancel_calls == []
        await _teardown(client_transport, task)

    async def test_cancel_for_authorized_session_id_reaches_backend(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        new_response = await _request(client_transport, 2, "session/new", {"cwd": "/tmp"})
        session_id = new_response["result"]["sessionId"]

        await _notify(client_transport, "session/cancel", {"sessionId": session_id})

        await _assert_no_message(client_transport)
        assert backend.cancel_calls == [session_id]
        await _teardown(client_transport, task)


class TestBackendExceptions:
    async def test_new_session_runtime_error_becomes_internal_error(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired
        backend.new_session_error = RuntimeError("boom")

        await _request(client_transport, 1, "initialize", {})
        response = await _request(client_transport, 2, "session/new", {"cwd": "/tmp"})

        assert response["error"]["code"] == -32603
        # The exception text must never reach the client -- it can carry SQL/schema
        # details or other server internals. Only the fixed, generic message goes out.
        assert "boom" not in response["error"]["message"]
        assert response["error"]["message"] == "internal error"

        # Connection survives: reset the error and retry successfully.
        backend.new_session_error = None
        follow_up = await _request(client_transport, 3, "session/new", {"cwd": "/tmp"})
        assert follow_up["result"] == {"sessionId": "server-sess-1"}

        await _teardown(client_transport, task)


class TestDispatchEdgeCases:
    async def test_unknown_method_returns_method_not_found(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        response = await _request(client_transport, 2, "totally/unknown", {})

        assert response["error"]["code"] == -32601
        await _teardown(client_transport, task)

    async def test_string_request_id_round_trips_byte_identical(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        response = await _request(client_transport, "req-abc-123", "initialize", {})

        assert response["id"] == "req-abc-123"
        await _teardown(client_transport, task)

    async def test_notification_gets_no_reply(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        await _request(client_transport, 1, "initialize", {})
        await _notify(client_transport, "session/cancel", {"sessionId": "server-sess-1"})

        await _assert_no_message(client_transport)
        await _teardown(client_transport, task)

    async def test_malformed_message_with_id_gets_invalid_request_error(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        await client_transport.send({"jsonrpc": "2.0", "id": 7})
        response = await client_transport.receive()

        assert response["id"] == 7
        assert response["error"]["code"] == -32600
        await _teardown(client_transport, task)

    async def test_malformed_message_without_id_gets_null_id_error(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        await client_transport.send({"jsonrpc": "2.0"})
        response = await client_transport.receive()

        # A message missing `method` is invalid regardless of whether it looks like a
        # notification -- there's no method to know it was even meant as one, so it
        # always gets answered, with `id: null` since none was supplied.
        assert response["id"] is None
        assert response["error"]["code"] == -32600
        await _teardown(client_transport, task)

    async def test_batch_array_message_gets_single_invalid_request_error(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        await client_transport.send(
            [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}]
        )
        response = await client_transport.receive()

        assert response["id"] is None
        assert response["error"]["code"] == -32600
        await _assert_no_message(client_transport)
        await _teardown(client_transport, task)

    async def test_zero_request_id_round_trips_exact_type(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        response = await _request(client_transport, 0, "initialize", {})

        assert response["id"] == 0
        assert type(response["id"]) is int
        await _teardown(client_transport, task)

    async def test_null_request_id_present_gets_response_with_null_id(self, wired):
        client_transport, _server_transport, _backend, _connection, task = wired

        await client_transport.send(
            {"jsonrpc": "2.0", "id": None, "method": "initialize", "params": {}}
        )
        response = await client_transport.receive()

        assert response["id"] is None
        assert response["result"]["agentInfo"]["name"] == "cptr"
        await _teardown(client_transport, task)


class TestDogfoodWithRealAcpClient:
    """Run cptr's own `AcpClient` against `AcpServerConnection` directly."""

    async def test_client_start_completes_without_authenticate(self):
        client_transport, server_transport = connected_pair()
        recording_server_transport = RecordingTransport(server_transport)
        backend = StubBackend(session_id="dogfood-sess")
        connection = AcpServerConnection(recording_server_transport, backend)
        server_task = asyncio.create_task(connection.serve())

        client = AcpClient(
            command="dummy",
            args=[],
            cwd="/tmp",
            env={},
            auth_method_id=None,
            transport=client_transport,
        )
        await client.start()

        assert client.session_id == "dogfood-sess"
        methods_seen = [m.get("method") for m in recording_server_transport.received]
        assert "authenticate" not in methods_seen
        assert "initialize" in methods_seen
        assert "session/new" in methods_seen

        await client.close()
        if not server_task.done():
            server_task.cancel()
        with contextlib_suppress():
            await asyncio.wait_for(server_task, timeout=2)
        _assert_task_finished_cleanly(server_task)

    async def test_resume_path_keeps_session_id_when_load_succeeds(self):
        client_transport, server_transport = connected_pair()
        backend = StubBackend(load_result=True)
        connection = AcpServerConnection(server_transport, backend)
        server_task = asyncio.create_task(connection.serve())

        client = AcpClient(
            command="dummy",
            args=[],
            cwd="/tmp",
            env={},
            auth_method_id=None,
            transport=client_transport,
            resume_session_id="resume-me",
        )
        await client.start()

        assert client.session_id == "resume-me"
        assert backend.load_session_calls == [("resume-me", "/tmp")]
        assert backend.new_session_calls == []

        await client.close()
        if not server_task.done():
            server_task.cancel()
        with contextlib_suppress():
            await asyncio.wait_for(server_task, timeout=2)
        _assert_task_finished_cleanly(server_task)

    async def test_resume_path_falls_back_to_session_new_when_load_fails(self):
        client_transport, server_transport = connected_pair()
        backend = StubBackend(load_result=False, session_id="fresh-session")
        connection = AcpServerConnection(server_transport, backend)
        server_task = asyncio.create_task(connection.serve())

        client = AcpClient(
            command="dummy",
            args=[],
            cwd="/tmp",
            env={},
            auth_method_id=None,
            transport=client_transport,
            resume_session_id="missing-session",
        )
        await client.start()

        assert client.session_id == "fresh-session"
        assert backend.load_session_calls == [("missing-session", "/tmp")]
        assert len(backend.new_session_calls) == 1

        await client.close()
        if not server_task.done():
            server_task.cancel()
        with contextlib_suppress():
            await asyncio.wait_for(server_task, timeout=2)
        _assert_task_finished_cleanly(server_task)
