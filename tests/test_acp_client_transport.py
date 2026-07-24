"""Drive a real `AcpClient` over `InMemoryTransport.connected_pair()`.

No subprocess, no network: a scripted "fake agent" coroutine plays the other side of
the JSON-RPC conversation directly against the in-memory queues, exercising all the
protocol logic that lives in `cptr/utils/agents/acp.py` (handshake, session
setup/resume, permission replies, event delivery, error handling) without touching
`StdioSubprocessTransport` at all.
"""

from __future__ import annotations

import asyncio
import contextlib

import pytest

from cptr.utils.agents.acp import AcpClient
from cptr.utils.agents.acp_transport import connected_pair


def make_client(agent_transport, **kwargs) -> AcpClient:
    return AcpClient(
        command="dummy",
        args=[],
        cwd="/tmp",
        env={},
        auth_method_id=kwargs.pop("auth_method_id", None),
        transport=agent_transport,
        **kwargs,
    )


class FakeAgent:
    """Reads requests off its transport and replies according to scripted handlers."""

    def __init__(self, transport):
        self.transport = transport
        self.received: list[dict] = []
        self.handlers: dict[str, object] = {}
        self._task: asyncio.Task | None = None

    def on(self, method, handler):
        self.handlers[method] = handler

    def start(self):
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self):
        while True:
            try:
                message = await self.transport.receive()
            except Exception:
                break
            self.received.append(message)
            method = message.get("method")
            if method is None:
                continue
            handler = self.handlers.get(method)
            if handler is None:
                continue
            await handler(message)

    async def reply_result(self, message, result):
        await self.transport.send({"jsonrpc": "2.0", "id": message["id"], "result": result})

    async def reply_error(self, message, error):
        await self.transport.send({"jsonrpc": "2.0", "id": message["id"], "error": error})


@pytest.fixture
def transports():
    client_side, agent_side = connected_pair()
    return client_side, agent_side


async def _default_handshake(agent: FakeAgent, *, auth_methods=None, session_id="sess-1"):
    auth_methods = auth_methods if auth_methods is not None else [{"id": "api-key"}]

    async def handle_initialize(message):
        await agent.reply_result(
            message,
            {"authMethods": auth_methods, "protocolVersion": 1},
        )

    async def handle_authenticate(message):
        await agent.reply_result(message, {})

    async def handle_session_new(message):
        await agent.reply_result(
            message,
            {
                "sessionId": session_id,
                "configOptions": [{"id": "model-config", "category": "model"}],
            },
        )

    agent.on("initialize", handle_initialize)
    agent.on("authenticate", handle_authenticate)
    agent.on("session/new", handle_session_new)


class TestHandshake:
    async def test_start_completes_and_session_established(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent, auth_methods=[{"id": "method-a"}, {"id": "method-b"}])
        agent.start()

        client = make_client(client_transport)
        await client.start()

        assert client.session_id == "sess-1"
        assert client.model_config_id == "model-config"

        auth_calls = [m for m in agent.received if m.get("method") == "authenticate"]
        assert len(auth_calls) == 1
        assert auth_calls[0]["params"]["methodId"] == "method-a"

        await client.close()
        await agent.stop()

    async def test_session_load_resume_path(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)

        async def handle_session_load(message):
            assert message["params"]["sessionId"] == "resume-me"
            await agent.reply_result(
                message,
                {"configOptions": [{"id": "model-config", "category": "model"}]},
            )

        agent.on("session/load", handle_session_load)
        agent.start()

        client = make_client(client_transport, resume_session_id="resume-me")
        await client.start()

        assert client.session_id == "resume-me"
        load_calls = [m for m in agent.received if m.get("method") == "session/load"]
        assert len(load_calls) == 1
        new_calls = [m for m in agent.received if m.get("method") == "session/new"]
        assert len(new_calls) == 0

        await client.close()
        await agent.stop()

    async def test_session_load_error_falls_back_to_session_new(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent, session_id="new-session")

        async def handle_session_load(message):
            await agent.reply_error(message, {"code": -1, "message": "no such session"})

        agent.on("session/load", handle_session_load)
        agent.start()

        client = make_client(client_transport, resume_session_id="missing-session")
        await client.start()

        assert client.session_id == "new-session"
        load_calls = [m for m in agent.received if m.get("method") == "session/load"]
        new_calls = [m for m in agent.received if m.get("method") == "session/new"]
        assert len(load_calls) == 1
        assert len(new_calls) == 1

        await client.close()
        await agent.stop()


class TestRequestErrors:
    async def test_request_raises_runtime_error_on_jsonrpc_error(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)
        agent.start()

        client = make_client(client_transport)
        await client.start()

        async def handle_boom(message):
            await agent.reply_error(message, {"code": 123, "message": "boom"})

        agent.on("custom/boom", handle_boom)

        with pytest.raises(RuntimeError):
            await client.request("custom/boom", {})

        await client.close()
        await agent.stop()


class TestPermissionRequest:
    async def _send_permission_request(self, agent: FakeAgent, request_id="req-abc"):
        await agent.transport.send(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session/request_permission",
                "params": {
                    "sessionId": "sess-1",
                    "options": [{"kind": "allow_once", "optionId": "ok"}],
                },
            }
        )

    async def _wait_for_reply(self, agent: FakeAgent, request_id):
        # `agent.start()` already runs a background loop draining `agent.transport`,
        # so we must not issue a second, competing `receive()` call on the same
        # queue -- instead poll the messages that loop has already recorded.
        for _ in range(200):
            for message in agent.received:
                if message.get("id") == request_id and "result" in message:
                    return message
            await asyncio.sleep(0.01)
        raise AssertionError(f"no reply for id={request_id!r} within timeout")

    async def test_non_integer_id_auto_approve_selects_option(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)
        agent.start()

        client = make_client(client_transport, auto_approve_permissions=True)
        await client.start()

        await self._send_permission_request(agent, request_id="req-abc")
        reply = await self._wait_for_reply(agent, "req-abc")

        assert reply["result"]["outcome"] == {"outcome": "selected", "optionId": "ok"}

        await client.close()
        await agent.stop()

    async def test_non_integer_id_no_auto_approve_cancels(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)
        agent.start()

        client = make_client(client_transport, auto_approve_permissions=False)
        await client.start()

        await self._send_permission_request(agent, request_id="req-xyz")
        reply = await self._wait_for_reply(agent, "req-xyz")

        assert reply["result"]["outcome"] == {"outcome": "cancelled"}

        await client.close()
        await agent.stop()


class TestEvents:
    async def test_session_update_notification_lands_in_events_queue(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)
        agent.start()

        client = make_client(client_transport)
        await client.start()

        await agent.transport.send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "sess-1",
                    "update": {"sessionUpdate": "agent_message_chunk"},
                },
            }
        )

        event = await asyncio.wait_for(client.events.get(), timeout=2)
        assert event["method"] == "session/update"
        assert event["params"]["update"]["sessionUpdate"] == "agent_message_chunk"

        await client.close()
        await agent.stop()


class TestTransportClose:
    async def test_peer_close_mid_session_does_not_explode(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)
        agent.start()

        client = make_client(client_transport)
        await client.start()

        pending = asyncio.ensure_future(client.request("custom/never_answered", {}))
        # Let the request actually get sent before we close the peer.
        await asyncio.sleep(0.01)

        await agent_transport.close()

        # Give the reader loop a moment to observe TransportClosed and exit cleanly.
        await asyncio.sleep(0.01)
        assert client.reader_task is not None
        # The reader task should have finished (not hung), and not raised.
        for _ in range(20):
            if client.reader_task.done():
                break
            await asyncio.sleep(0.01)
        assert client.reader_task.done()
        assert client.reader_task.exception() is None

        # close() must clean up the still-pending future without hanging.
        await asyncio.wait_for(client.close(), timeout=2)

        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=2)
        assert pending.done()

        await agent.stop()

    async def test_close_after_peer_close_is_clean(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)
        agent.start()

        client = make_client(client_transport)
        await client.start()

        await agent_transport.close()
        await asyncio.sleep(0.01)

        # Calling close() twice / after the peer already closed must not raise.
        await asyncio.wait_for(client.close(), timeout=2)
        await asyncio.wait_for(client.close(), timeout=2)

        await agent.stop()


class TestPrompt:
    async def test_prompt_builds_text_and_image_blocks(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)

        captured = {}

        async def handle_prompt(message):
            captured["params"] = message["params"]
            await agent.reply_result(message, {"stopReason": "end_turn"})

        agent.on("session/prompt", handle_prompt)
        agent.start()

        client = make_client(client_transport)
        await client.start()

        result = await client.prompt(
            "hello there",
            images=[{"data": "base64data", "mimeType": "image/png"}],
        )

        assert result == {"stopReason": "end_turn"}
        assert captured["params"]["sessionId"] == "sess-1"
        assert captured["params"]["prompt"] == [
            {"type": "text", "text": "hello there"},
            {"type": "image", "data": "base64data", "mimeType": "image/png"},
        ]

        await client.close()
        await agent.stop()

    async def test_prompt_with_blank_text_and_no_images(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)

        captured = {}

        async def handle_prompt(message):
            captured["params"] = message["params"]
            await agent.reply_result(message, {"stopReason": "end_turn"})

        agent.on("session/prompt", handle_prompt)
        agent.start()

        client = make_client(client_transport)
        await client.start()

        await client.prompt("   ")

        assert captured["params"]["prompt"] == []

        await client.close()
        await agent.stop()


class TestPendingIdMatching:
    """`_handle_message` pops `self.pending` using `message["id"]` as-is. `AcpClient`
    itself only ever generates integer ids via `self.next_id`, so a real agent
    conversation never exercises the string-id branch of `pending.pop(...)` -- this
    test seeds `client.pending` directly to pin that non-integer ids resolve their
    matching future (and don't disturb an int-keyed sibling), which would fail if the
    pop ever coerced the id with e.g. `int(message["id"])`.
    """

    async def test_response_with_string_id_resolves_matching_future_only(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)
        agent.start()

        client = make_client(client_transport)
        await client.start()

        loop = asyncio.get_running_loop()
        string_future: asyncio.Future = loop.create_future()
        int_future: asyncio.Future = loop.create_future()
        client.pending["req-string-id"] = string_future
        client.pending[999] = int_future

        await agent.transport.send(
            {"jsonrpc": "2.0", "id": "req-string-id", "result": {"ok": True}}
        )

        response = await asyncio.wait_for(string_future, timeout=2)
        assert response == {"jsonrpc": "2.0", "id": "req-string-id", "result": {"ok": True}}

        # The int-keyed sibling must be untouched: still registered, still pending.
        assert client.pending.get(999) is int_future
        assert not int_future.done()
        assert "req-string-id" not in client.pending

        await client.close()
        await agent.stop()


class TestCloseCancelsReaderAndPending:
    async def test_close_cancels_reader_task_and_pending_request_while_peer_alive(self, transports):
        client_transport, agent_transport = transports
        agent = FakeAgent(agent_transport)
        await _default_handshake(agent)
        agent.start()

        client = make_client(client_transport)
        await client.start()

        # Peer stays alive throughout -- this pins that `close()` itself cancels the
        # reader task and pending futures, not that they resolve via peer closure.
        pending = asyncio.ensure_future(client.request("custom/never_answered", {}))
        await asyncio.sleep(0.01)  # let the request actually reach `self.pending`

        assert client.reader_task is not None
        assert not client.reader_task.done()

        await asyncio.wait_for(client.close(), timeout=2)

        assert client.reader_task.done()
        assert client.reader_task.cancelled()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=2)
        assert pending.cancelled()

        await agent.stop()
