"""Pure protocol tests for `AcpServerConnection` (Tier 0: no network, no DB).

Drives a real `AcpServerConnection` over `InMemoryTransport.connected_pair()` against
a stub `SessionBackend`, exercising handshake/dispatch/error-code behavior in
isolation. The final class also dogfoods cptr's own `AcpClient` as the peer, so the
server and client sides of the protocol are validated against each other directly.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

import pytest

from cptr.utils.acp_server import AcpServerConnection, PromptRejected, acp_updates_from_queue_item
from cptr.utils.agents.acp import (
    AcpClient,
    acp_text_from_update,
    acp_tool_from_update,
    acp_turn_event_stream,
)
from cptr.utils.agents.acp_transport import TransportClosed, connected_pair


class StubBackend:
    """Records calls; session id / load result / errors are configurable.

    `prompt_impl`, when given, is an `async def (session_id, prompt_items, notify)`
    that replaces the default "raise NotImplementedError" prompt behavior -- lets
    individual tests script exactly what a turn does (stream updates, block, raise
    `PromptRejected`, etc.) without a new backend class each time.
    """

    def __init__(
        self,
        *,
        session_id: str = "server-sess-1",
        load_result: bool = True,
        prompt_impl: Callable[[str, list, Callable[[dict], Awaitable[None]]], Awaitable[dict]]
        | None = None,
    ) -> None:
        self.session_id = session_id
        self.load_result = load_result
        self.new_session_calls: list[tuple[str, dict]] = []
        self.load_session_calls: list[tuple[str, str]] = []
        self.prompt_calls: list[tuple[str, list]] = []
        self.cancel_calls: list[str] = []
        self.new_session_error: Exception | None = None
        self.prompt_impl = prompt_impl

    async def new_session(self, cwd, params):
        self.new_session_calls.append((cwd, params))
        if self.new_session_error is not None:
            raise self.new_session_error
        return self.session_id

    async def load_session(self, session_id, cwd):
        self.load_session_calls.append((session_id, cwd))
        return self.load_result

    async def prompt(self, session_id, prompt_items, notify):
        self.prompt_calls.append((session_id, prompt_items))
        if self.prompt_impl is not None:
            return await self.prompt_impl(session_id, prompt_items, notify)
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


class TestAcpUpdatesFromQueueItem:
    """Unit tests for the translation layer: `output_queue` items -> ACP updates.

    Each mapping is also checked for a round trip through the client-side parsers
    in `cptr/utils/agents/acp.py` where applicable -- these are the mirror image of
    each other, so a shape mismatch here is exactly the kind of bug that would
    otherwise only surface as a silently-dropped update in a real client.
    """

    def test_delta_maps_to_agent_message_chunk_and_round_trips(self):
        updates = acp_updates_from_queue_item({"type": "delta", "content": "hello"})

        assert updates == [
            {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "hello"}}
        ]
        assert acp_text_from_update({"update": updates[0]}) == "hello"

    def test_empty_delta_yields_no_updates(self):
        assert acp_updates_from_queue_item({"type": "delta", "content": ""}) == []

    def test_pending_function_call_maps_to_tool_call_and_round_trips(self):
        item = {
            "type": "output",
            "item": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "run_command",
                "arguments": {"command": "ls"},
                "status": "pending",
            },
        }

        updates = acp_updates_from_queue_item(item)

        assert len(updates) == 1
        update = updates[0]
        assert update["sessionUpdate"] == "tool_call"
        assert update["toolCallId"] == "call_1"
        assert update["status"] == "pending"
        assert update["rawInput"] == {"command": "ls"}

        parsed = acp_tool_from_update({"update": update})
        assert parsed["call_id"] == "call_1"
        assert parsed["name"] == "run_command"
        assert parsed["arguments"] == {"command": "ls"}
        assert parsed["status"] == "pending"

    def test_in_progress_function_call_maps_to_tool_call(self):
        item = {
            "type": "output",
            "item": {
                "type": "function_call",
                "call_id": "call_2",
                "name": "agent_tool",
                "arguments": {},
                "status": "in_progress",
            },
        }

        updates = acp_updates_from_queue_item(item)

        assert updates[0]["sessionUpdate"] == "tool_call"

    def test_completed_function_call_maps_to_tool_call_update_and_round_trips(self):
        item = {
            "type": "output",
            "item": {
                "type": "function_call",
                "call_id": "call_3",
                "name": "run_command",
                "arguments": {"command": "pwd"},
                "status": "completed",
            },
        }

        updates = acp_updates_from_queue_item(item)

        assert updates[0]["sessionUpdate"] == "tool_call_update"
        parsed = acp_tool_from_update({"update": updates[0]})
        assert parsed["status"] == "completed"
        assert parsed["arguments"] == {"command": "pwd"}

    def test_failed_function_call_maps_to_tool_call_update(self):
        item = {
            "type": "output",
            "item": {
                "type": "function_call",
                "call_id": "call_4",
                "name": "agent_tool",
                "arguments": {},
                "status": "failed",
            },
        }

        updates = acp_updates_from_queue_item(item)

        assert updates[0]["sessionUpdate"] == "tool_call_update"
        parsed = acp_tool_from_update({"update": updates[0]})
        assert parsed["status"] == "failed"

    def test_function_call_missing_call_id_yields_no_updates(self):
        item = {
            "type": "output",
            "item": {"type": "function_call", "name": "x", "status": "pending"},
        }

        assert acp_updates_from_queue_item(item) == []

    def test_function_call_output_maps_to_tool_call_update_with_nested_content_and_round_trips(
        self,
    ):
        item = {
            "type": "output",
            "item": {"type": "function_call_output", "call_id": "call_1", "output": "file1\nfile2"},
        }

        updates = acp_updates_from_queue_item(item)

        assert updates == [
            {
                "sessionUpdate": "tool_call_update",
                "toolCallId": "call_1",
                "content": [
                    {"type": "content", "content": {"type": "text", "text": "file1\nfile2"}}
                ],
            }
        ]
        parsed = acp_tool_from_update({"update": updates[0]})
        assert parsed["call_id"] == "call_1"
        assert parsed["output"] == "file1\nfile2"

    def test_function_call_output_missing_call_id_yields_no_updates(self):
        item = {"type": "output", "item": {"type": "function_call_output", "output": "x"}}

        assert acp_updates_from_queue_item(item) == []

    def test_message_item_yields_no_updates_its_text_already_streamed_as_deltas(self):
        item = {
            "type": "output",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "already streamed"}],
            },
        }

        assert acp_updates_from_queue_item(item) == []

    def test_reasoning_item_with_text_maps_to_agent_thought_chunk(self):
        item = {
            "type": "output",
            "item": {
                "type": "reasoning",
                "status": "in_progress",
                "content": [{"type": "reasoning_text", "text": "thinking about it"}],
            },
        }

        updates = acp_updates_from_queue_item(item)

        assert updates == [
            {
                "sessionUpdate": "agent_thought_chunk",
                "content": {"type": "text", "text": "thinking about it"},
            }
        ]

    def test_reasoning_item_without_text_yields_no_updates(self):
        item = {
            "type": "output",
            "item": {"type": "reasoning", "status": "in_progress", "content": []},
        }

        assert acp_updates_from_queue_item(item) == []

    def test_unknown_output_item_type_yields_no_updates(self):
        assert (
            acp_updates_from_queue_item({"type": "output", "item": {"type": "something_else"}})
            == []
        )

    def test_done_and_error_queue_items_yield_no_updates(self):
        assert acp_updates_from_queue_item({"type": "done", "finish_reason": "stop"}) == []
        assert acp_updates_from_queue_item({"type": "error", "message": "boom"}) == []

    def test_unknown_top_level_type_yields_no_updates(self):
        assert acp_updates_from_queue_item({"type": "context_usage", "tokens": 10}) == []


class BlockingPromptBackend(StubBackend):
    """Streams two updates, then blocks until `cancel()` (or the test) unblocks it.

    `entered` fires once both updates are out and the block has begun, so a test can
    `await` it instead of racing a fixed sleep before sending `session/cancel`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.unblock = asyncio.Event()
        self.entered = asyncio.Event()

    async def prompt(self, session_id, prompt_items, notify):
        self.prompt_calls.append((session_id, prompt_items))
        await notify(
            {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "one"}}
        )
        await notify(
            {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "two"}}
        )
        self.entered.set()
        await self.unblock.wait()
        return {"stopReason": "cancelled"}

    async def cancel(self, session_id):
        self.cancel_calls.append(session_id)
        self.unblock.set()


async def _new_session(client_transport, request_id: int, cwd: str = "/tmp") -> str:
    response = await _request(client_transport, request_id, "session/new", {"cwd": cwd})
    return response["result"]["sessionId"]


class TestPromptConcurrencyAndCancel:
    """The concurrency contract from step 3: `session/prompt` must not block the
    serve loop, so `session/cancel` (and other requests) can still be handled while
    a turn is in flight.
    """

    async def test_cancel_mid_prompt_completes_with_cancelled_stop_reason(self):
        client_transport, server_transport = connected_pair()
        backend = BlockingPromptBackend()
        connection = AcpServerConnection(server_transport, backend)
        task = asyncio.create_task(connection.serve())

        await _request(client_transport, 1, "initialize", {})
        session_id = await _new_session(client_transport, 2)

        await client_transport.send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": []},
            }
        )
        first_update = await client_transport.receive()
        second_update = await client_transport.receive()
        assert first_update["method"] == "session/update"
        assert second_update["method"] == "session/update"

        await asyncio.wait_for(backend.entered.wait(), timeout=2)
        await _notify(client_transport, "session/cancel", {"sessionId": session_id})

        response = await asyncio.wait_for(client_transport.receive(), timeout=2)
        assert response["id"] == 3
        assert response["result"] == {"stopReason": "cancelled"}
        assert backend.cancel_calls == [session_id]

        await _teardown(client_transport, task)

    async def test_second_prompt_same_session_rejected_while_first_in_progress(self):
        client_transport, server_transport = connected_pair()
        backend = BlockingPromptBackend()
        connection = AcpServerConnection(server_transport, backend)
        task = asyncio.create_task(connection.serve())

        await _request(client_transport, 1, "initialize", {})
        session_id = await _new_session(client_transport, 2)

        await client_transport.send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": []},
            }
        )
        await client_transport.receive()
        await client_transport.receive()
        await asyncio.wait_for(backend.entered.wait(), timeout=2)

        second = await _request(
            client_transport, 4, "session/prompt", {"sessionId": session_id, "prompt": []}
        )
        assert second["error"]["code"] == -32000
        assert "already in progress" in second["error"]["message"]
        # The rejected duplicate must never even reach the backend.
        assert len(backend.prompt_calls) == 1

        backend.unblock.set()
        first_response = await asyncio.wait_for(client_transport.receive(), timeout=2)
        assert first_response["id"] == 3
        assert first_response["result"] == {"stopReason": "cancelled"}

        await _teardown(client_transport, task)

    async def test_prompts_on_two_different_sessions_run_concurrently(self):
        client_transport, server_transport = connected_pair()

        class TwoSessionBackend:
            def __init__(self) -> None:
                self._next_id = 0
                self.cancel_calls: list[str] = []
                self.events: dict[str, asyncio.Event] = {}

            async def new_session(self, cwd, params):
                self._next_id += 1
                session_id = f"sess-{self._next_id}"
                self.events[session_id] = asyncio.Event()
                return session_id

            async def load_session(self, session_id, cwd):
                return True

            async def prompt(self, session_id, prompt_items, notify):
                await notify(
                    {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": f"hi from {session_id}"},
                    }
                )
                await self.events[session_id].wait()
                return {"stopReason": "end_turn"}

            async def cancel(self, session_id):
                self.cancel_calls.append(session_id)
                event = self.events.get(session_id)
                if event:
                    event.set()

        backend = TwoSessionBackend()
        connection = AcpServerConnection(server_transport, backend)
        task = asyncio.create_task(connection.serve())

        await _request(client_transport, 1, "initialize", {})
        session_a = await _new_session(client_transport, 2, "/tmp/a")
        session_b = await _new_session(client_transport, 3, "/tmp/b")

        await client_transport.send(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "session/prompt",
                "params": {"sessionId": session_a, "prompt": []},
            }
        )
        await client_transport.send(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "session/prompt",
                "params": {"sessionId": session_b, "prompt": []},
            }
        )

        update_1 = await client_transport.receive()
        update_2 = await client_transport.receive()
        texts = {
            update_1["params"]["update"]["content"]["text"],
            update_2["params"]["update"]["content"]["text"],
        }
        assert texts == {f"hi from {session_a}", f"hi from {session_b}"}

        backend.events[session_a].set()
        backend.events[session_b].set()

        response_1 = await asyncio.wait_for(client_transport.receive(), timeout=2)
        response_2 = await asyncio.wait_for(client_transport.receive(), timeout=2)
        by_id = {response_1["id"]: response_1, response_2["id"]: response_2}
        assert by_id[4]["result"] == {"stopReason": "end_turn"}
        assert by_id[5]["result"] == {"stopReason": "end_turn"}

        await _teardown(client_transport, task)


class TestPromptExceptionMapping:
    async def test_prompt_rejected_becomes_dash32000_with_the_safe_message(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        async def reject_impl(session_id, prompt_items, notify):
            raise PromptRejected("no model configured")

        backend.prompt_impl = reject_impl

        await _request(client_transport, 1, "initialize", {})
        session_id = await _new_session(client_transport, 2)
        response = await _request(
            client_transport, 3, "session/prompt", {"sessionId": session_id, "prompt": []}
        )

        assert response["error"]["code"] == -32000
        assert response["error"]["message"] == "no model configured"
        await _teardown(client_transport, task)

    async def test_generic_exception_in_prompt_becomes_internal_error_only(self, wired):
        client_transport, _server_transport, backend, _connection, task = wired

        async def boom_impl(session_id, prompt_items, notify):
            raise RuntimeError("sql error: leaked schema detail")

        backend.prompt_impl = boom_impl

        await _request(client_transport, 1, "initialize", {})
        session_id = await _new_session(client_transport, 2)
        response = await _request(
            client_transport, 3, "session/prompt", {"sessionId": session_id, "prompt": []}
        )

        assert response["error"]["code"] == -32603
        assert response["error"]["message"] == "internal error"
        assert "sql error" not in response["error"]["message"]
        await _teardown(client_transport, task)


class TestDisconnectMidPrompt:
    async def test_transport_close_cancels_in_flight_prompt_and_calls_backend_cancel(self):
        client_transport, server_transport = connected_pair()
        backend = BlockingPromptBackend()
        connection = AcpServerConnection(server_transport, backend)
        task = asyncio.create_task(connection.serve())

        await _request(client_transport, 1, "initialize", {})
        session_id = await _new_session(client_transport, 2)

        await client_transport.send(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {"sessionId": session_id, "prompt": []},
            }
        )
        await client_transport.receive()
        await client_transport.receive()
        await asyncio.wait_for(backend.entered.wait(), timeout=2)

        await client_transport.close()

        await asyncio.wait_for(task, timeout=2)
        assert task.done()
        assert task.exception() is None
        assert backend.cancel_calls == [session_id]


class TestDogfoodPromptRoundTrip:
    """Runs a real `AcpClient.prompt()` against the server (step 3's payoff): the
    server-side translation and the client-side parsers must agree on every shape.
    """

    async def test_client_prompt_parses_streamed_text_and_tool_updates(self):
        client_transport, server_transport = connected_pair()

        async def prompt_impl(session_id, prompt_items, notify):
            await notify(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Hello "},
                }
            )
            await notify(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "world"},
                }
            )
            await notify(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "call_1",
                    "title": "run_command",
                    "status": "in_progress",
                    "rawInput": {"command": "ls"},
                }
            )
            await notify(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "call_1",
                    "content": [
                        {"type": "content", "content": {"type": "text", "text": "file1\nfile2"}}
                    ],
                }
            )
            return {"stopReason": "end_turn"}

        backend = StubBackend(session_id="dogfood-turn", prompt_impl=prompt_impl)
        connection = AcpServerConnection(server_transport, backend)
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
        assert client.session_id == "dogfood-turn"

        prompt_task = asyncio.create_task(client.prompt("hi"))
        texts: list[str] = []
        tools: list[dict] = []
        async for event in acp_turn_event_stream(client, prompt_task):
            params = event.get("params") if isinstance(event.get("params"), dict) else {}
            text = acp_text_from_update(params)
            if text:
                texts.append(text)
                continue
            tool = acp_tool_from_update(params)
            if tool:
                tools.append(tool)

        result = await prompt_task
        assert result["stopReason"] == "end_turn"
        assert "".join(texts) == "Hello world"
        assert len(tools) == 2
        assert tools[0]["call_id"] == "call_1"
        assert tools[0]["name"] == "run_command"
        assert tools[0]["arguments"] == {"command": "ls"}
        assert tools[1]["call_id"] == "call_1"
        assert tools[1]["output"] == "file1\nfile2"

        await client.close()
        if not server_task.done():
            server_task.cancel()
        with contextlib_suppress():
            await asyncio.wait_for(server_task, timeout=2)
        _assert_task_finished_cleanly(server_task)
