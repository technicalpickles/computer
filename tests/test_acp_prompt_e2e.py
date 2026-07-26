"""Tier 1 end-to-end test for step 3 (docs/acp-server-validation-harness.md): a real
`session/prompt` turn, bridged through the real `run_chat_task` machinery, against a
scripted OpenAI-compatible model server -- no real LLM anywhere.

Two real local HTTP/WebSocket servers, both started with `uvicorn.Server` on an
ephemeral port (`port=0`, bound port discovered via `server.servers[0].sockets`) as
background `asyncio.Task`s on the *same* event loop as the test:

  1. The fake model server (`FakeOpenAIServer`): a tiny FastAPI app exposing
     `POST /v1/chat/completions` that replays a scripted OpenAI chat-completions SSE
     stream (role chunk, two content chunks, a `finish_reason: stop` chunk with
     `usage`, then `[DONE]`). `cptr/utils/ai.py:stream_openai_completions` is
     exercised for real over a real socket -- nothing about the model call is
     mocked. A connection with `data.models` pre-set (see
     `cptr/utils/model_targets.py:first_api_model_target`) skips live model
     discovery entirely, so no `GET /v1/models` endpoint is needed.
  2. The app under test: a slim `FastAPI()` mounting only `cptr.routers.acp.router`
     (same pattern as `test_acp_ws_endpoint.py`), against a per-test isolated SQLite
     DB.

Both run on the test's own event loop (not a separate thread, unlike
`TestClient.websocket_connect`), so a real async WebSocket client (`websockets`) can
drive the ACP conversation with proper `asyncio.wait_for` timeouts throughout --
the thing a synchronous `TestClient` can't give us for a multi-second real network
round trip.
"""

from __future__ import annotations

import asyncio
import json

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import cptr.utils.config as config_module
import cptr.utils.db as db_module
import cptr.utils.skills as skills_module
from cptr.models import Chat, ChatMessage, Config, User
from cptr.routers.acp import router as acp_router
from cptr.utils import chat_task as chat_task_module
from cptr.utils.acp_server import ChatSessionBackend, PromptBusy, PromptRejected
from cptr.utils.config import create_token, now_ms
from cptr.utils.db import init_db

WS_RECV_TIMEOUT = 20  # generous bound for a real (if tiny) local network round trip


# ── DB isolation (same mechanism as test_acp_ws_endpoint.py) ─────────────────


@pytest.fixture
async def isolated_db(tmp_path, monkeypatch):
    data_dir = tmp_path / "cptr-data"
    data_dir.mkdir()
    db_file = data_dir / "app.db"
    config_file = data_dir / "config.toml"

    monkeypatch.setattr(db_module, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(db_module, "DB_FILE", db_file, raising=True)
    monkeypatch.setattr(config_module, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(config_module, "CONFIG_FILE", config_file, raising=True)
    monkeypatch.setattr(config_module, "_config_cache", None, raising=True)
    # Defensive: discover_skills() falls back to this global dir (bound to the real
    # `Path.home()` at import time). It's read-only in this test's scenario (no
    # `$skill` mention in the prompt, so nothing ever writes into it) but pointing
    # it at a tmp dir means a real run_chat_task turn can never so much as *list*
    # the real ~/.cptr/skills, regardless.
    monkeypatch.setattr(
        skills_module, "MANAGED_GLOBAL_SKILL_DIR", data_dir / "global-skills", raising=True
    )

    if db_module._engine is not None:
        await db_module._engine.dispose()
    db_module._engine = None
    db_module._async_session = None

    await init_db()

    yield

    if db_module._engine is not None:
        await db_module._engine.dispose()
    db_module._engine = None
    db_module._async_session = None
    config_module._config_cache = None


@pytest.fixture
def slim_app():
    app = FastAPI()
    app.include_router(acp_router)
    return app


@pytest.fixture
async def authed_user(isolated_db):
    """Create a real user + return (user_id, jwt_token)."""
    user_id = await User.create(
        username="acp-prompt-tester",
        password_hash="unused-in-this-test",
        role="user",
        created_at=now_ms(),
    )
    token = create_token(user_id, "acp-prompt-tester", role="user")
    return user_id, token


# ── Fake OpenAI-compatible model server ──────────────────────────────────────


class FakeOpenAIServer:
    """Scripted `POST /v1/chat/completions`: replays a canned SSE chat-completion
    stream and records every request body it receives, so tests can assert the
    prompt text actually made it into the upstream call.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        # When set, the generator awaits this event after the first content chunk
        # ("Hello ") before sending the second ("world") -- lets a test hold the
        # stream open to exercise mid-turn `session/cancel`.
        self.hold_before_second_chunk: asyncio.Event | None = None
        self.app = FastAPI()

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            self.requests.append(body)
            return StreamingResponse(self._stream(), media_type="text/event-stream")

    async def _stream(self):
        def sse(delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> str:
            payload = {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "fake-model",
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            if usage is not None:
                payload["usage"] = usage
            return f"data: {json.dumps(payload)}\n\n"

        yield sse({"role": "assistant"})
        yield sse({"content": "Hello "})
        if self.hold_before_second_chunk is not None:
            await self.hold_before_second_chunk.wait()
        yield sse({"content": "world"})
        yield sse(
            {},
            finish_reason="stop",
            usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        )
        yield "data: [DONE]\n\n"


class FakeOpenAIToolCallServer:
    """Scripted `POST /v1/chat/completions` for the tool-call e2e scenario
    (MIN-8): the *first* request (no `tool` role message yet in `body["messages"]`)
    gets a streamed `run_command` tool call (`finish_reason: "tool_calls"`, the
    `cptr/utils/ai.py:stream_openai_completions` shape -- one `tool_calls` delta
    carrying `id`+`function.name`+full `function.arguments` in one chunk, same as a
    provider that doesn't fragment small argument strings across chunks); every
    request *after* that (i.e. once `run_chat_task` has looped back with the tool
    result appended) gets a normal final-text stream.
    """

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.app = FastAPI()

        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            body = await request.json()
            self.requests.append(body)
            has_tool_result = any(m.get("role") == "tool" for m in body.get("messages", []))
            stream = self._final_text_stream() if has_tool_result else self._tool_call_stream()
            return StreamingResponse(stream, media_type="text/event-stream")

    @staticmethod
    def _sse(delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> str:
        payload = {
            "id": "chatcmpl-fake",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "fake-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload)}\n\n"

    async def _tool_call_stream(self):
        yield self._sse({"role": "assistant"})
        yield self._sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_acp_e2e",
                        "type": "function",
                        "function": {
                            "name": "run_command",
                            "arguments": json.dumps({"command": "echo acp-e2e-ok", "wait": 10}),
                        },
                    }
                ]
            }
        )
        yield self._sse(
            {},
            finish_reason="tool_calls",
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        )
        yield "data: [DONE]\n\n"

    async def _final_text_stream(self):
        yield self._sse({"role": "assistant"})
        yield self._sse({"content": "ran it"})
        yield self._sse(
            {},
            finish_reason="stop",
            usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        )
        yield "data: [DONE]\n\n"


async def _run_uvicorn(app: FastAPI) -> tuple[uvicorn.Server, asyncio.Task, int]:
    """Start `app` with uvicorn on an ephemeral port, as a background task on the
    *current* event loop. Returns (server, task, bound_port); caller tears down with
    `server.should_exit = True; await task`.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    # MIN-8: bounded -- a startup that never flips `server.started` (e.g. a port
    # bind failure that doesn't otherwise raise here) must fail this test loudly
    # instead of hanging it forever.
    deadline = asyncio.get_event_loop().time() + 10
    while not server.started:
        if asyncio.get_event_loop().time() > deadline:
            task.cancel()
            raise TimeoutError("uvicorn server did not start within 10s")
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, task, port


async def _register_fake_connection(base_url: str) -> None:
    """Register a `chat.connections` entry with `data.models` pre-set, so model
    resolution (`first_api_model_target`) never needs live discovery -- see
    `cptr/utils/model_targets.py` / `cptr/routers/chat.py:_resolve_connection`.
    """
    await Config.upsert(
        {
            "chat.connections": [
                {
                    "id": "fake-conn",
                    "provider": "openai",
                    "base_url": base_url,
                    # Non-empty: httpx/h11 rejects a trailing-space-only
                    # "Authorization: Bearer " header value outright, and the fake
                    # server doesn't check this anyway.
                    "api_key": "fake-key-unused",
                    "api_type": "chat_completions",
                    "enabled": True,
                    "data": {"models": ["fake-model"]},
                }
            ]
        }
    )


async def _ws_json_send(ws, message: dict) -> None:
    await ws.send(json.dumps(message))


async def _ws_json_recv(ws) -> dict:
    raw = await asyncio.wait_for(ws.recv(), timeout=WS_RECV_TIMEOUT)
    return json.loads(raw)


class TestPromptBridgeEndToEnd:
    async def test_prompt_streams_real_completion_and_persists_message(
        self, slim_app, authed_user, tmp_path
    ):
        import websockets

        user_id, token = authed_user
        fake = FakeOpenAIServer()
        fake_server, fake_task, fake_port = await _run_uvicorn(fake.app)
        acp_server, acp_task, acp_port = await _run_uvicorn(slim_app)
        try:
            await _register_fake_connection(f"http://127.0.0.1:{fake_port}/v1")

            workspace = tmp_path / "workspace"
            workspace.mkdir()

            url = f"ws://127.0.0.1:{acp_port}/api/acp/ws?token={token}"
            async with websockets.connect(url, open_timeout=WS_RECV_TIMEOUT) as ws:
                await _ws_json_send(
                    ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                )
                init_response = await _ws_json_recv(ws)
                assert init_response["result"]["agentInfo"]["name"] == "cptr"

                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": str(workspace), "mcpServers": []},
                    },
                )
                new_response = await _ws_json_recv(ws)
                session_id = new_response["result"]["sessionId"]

                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "session/prompt",
                        "params": {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "hi"}],
                        },
                    },
                )

                texts: list[str] = []
                prompt_response = None
                while prompt_response is None:
                    message = await _ws_json_recv(ws)
                    if message.get("id") == 3:
                        prompt_response = message
                        break
                    update = message.get("params", {}).get("update", {})
                    if update.get("sessionUpdate") == "agent_message_chunk":
                        texts.append(update["content"]["text"])

            assert prompt_response["result"] == {"stopReason": "end_turn"}
            assert "".join(texts) == "Hello world"

            chat = await Chat.get_by_id(session_id)
            assert chat is not None
            assert chat.user_id == user_id

            messages = await ChatMessage.get_all_by_chat(session_id)
            user_messages = [m for m in messages if m.role == "user"]
            assistant_messages = [m for m in messages if m.role == "assistant"]
            assert len(user_messages) == 1
            assert user_messages[0].content == "hi"
            assert len(assistant_messages) == 1
            assistant_msg = assistant_messages[0]
            assert assistant_msg.done is True
            assert "Hello world" in assistant_msg.content

            assert len(fake.requests) == 1
            assert "hi" in json.dumps(fake.requests[0])
        finally:
            acp_server.should_exit = True
            fake_server.should_exit = True
            await asyncio.wait_for(acp_task, timeout=5)
            await asyncio.wait_for(fake_task, timeout=5)

    async def test_cancel_mid_stream_stops_the_turn_with_cancelled_stop_reason(
        self, slim_app, authed_user, tmp_path
    ):
        """`session/cancel` sent while the fake model server is mid-stream (holding
        after the first content chunk) must stop the turn with `stopReason:
        "cancelled"` instead of hanging or erroring -- see
        `ChatSessionBackend.cancel`/`prompt` in `cptr/utils/acp_server.py`.

        This exercises the same cancellation path end to end (`cancel_task` ->
        `run_chat_task`'s `except asyncio.CancelledError` -> `done` pushed to the
        `output_queue` -> `ChatSessionBackend.prompt`'s consumer loop) that
        `test_acp_server.py::TestPromptConcurrencyAndCancel` covers at the stub
        level; here the actual HTTP stream is what gets interrupted.
        """
        import websockets

        _user_id, token = authed_user
        fake = FakeOpenAIServer()
        fake.hold_before_second_chunk = asyncio.Event()
        fake_server, fake_task, fake_port = await _run_uvicorn(fake.app)
        acp_server, acp_task, acp_port = await _run_uvicorn(slim_app)
        try:
            await _register_fake_connection(f"http://127.0.0.1:{fake_port}/v1")

            workspace = tmp_path / "workspace"
            workspace.mkdir()

            url = f"ws://127.0.0.1:{acp_port}/api/acp/ws?token={token}"
            async with websockets.connect(url, open_timeout=WS_RECV_TIMEOUT) as ws:
                await _ws_json_send(
                    ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                )
                await _ws_json_recv(ws)

                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": str(workspace), "mcpServers": []},
                    },
                )
                session_id = (await _ws_json_recv(ws))["result"]["sessionId"]

                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "session/prompt",
                        "params": {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "hi"}],
                        },
                    },
                )

                # Wait for the first streamed chunk before cancelling -- proves the
                # turn was genuinely in flight, not cancelled before it started.
                first_text = None
                while first_text is None:
                    message = await _ws_json_recv(ws)
                    update = message.get("params", {}).get("update", {})
                    if update.get("sessionUpdate") == "agent_message_chunk":
                        first_text = update["content"]["text"]
                assert first_text == "Hello "

                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": {"sessionId": session_id},
                    },
                )

                prompt_response = None
                while prompt_response is None:
                    message = await _ws_json_recv(ws)
                    if message.get("id") == 3:
                        prompt_response = message

            assert prompt_response["result"] == {"stopReason": "cancelled"}

            messages = await ChatMessage.get_all_by_chat(session_id)
            assistant_messages = [m for m in messages if m.role == "assistant"]
            assert len(assistant_messages) == 1
            assert assistant_messages[0].done is True
        finally:
            # Unblock the fake server's generator regardless of outcome so its
            # background task can actually finish during teardown.
            fake.hold_before_second_chunk.set()
            acp_server.should_exit = True
            fake_server.should_exit = True
            await asyncio.wait_for(acp_task, timeout=5)
            await asyncio.wait_for(fake_task, timeout=5)


async def _assert_no_leaked_tasks(timeout: float = 3.0) -> None:
    """Poll for `chat_task._tasks` to empty out (BL-2): a surviving entry means
    some `run_chat_task` is still registered as running when it shouldn't be.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while chat_task_module._tasks and asyncio.get_event_loop().time() < deadline:
        await asyncio.sleep(0.05)
    assert not chat_task_module._tasks, (
        f"chat_task._tasks still has entries: {sorted(chat_task_module._tasks)}"
    )


class TestCancelDoesNotPoisonNextTurn:
    """MJ-1 regression (adapted from the reviewer's probe A): a `session/cancel`
    that arrives after a turn has already ended must not leave a stale flag that
    poisons the *next* turn on that session into falsely reporting "cancelled".
    """

    async def test_late_cancel_after_turn_ends_does_not_poison_next_prompt(
        self, slim_app, authed_user, tmp_path
    ):
        import websockets

        _user_id, token = authed_user
        fake = FakeOpenAIServer()
        fake_server, fake_task, fake_port = await _run_uvicorn(fake.app)
        acp_server, acp_task, acp_port = await _run_uvicorn(slim_app)
        try:
            await _register_fake_connection(f"http://127.0.0.1:{fake_port}/v1")
            workspace = tmp_path / "workspace"
            workspace.mkdir()

            url = f"ws://127.0.0.1:{acp_port}/api/acp/ws?token={token}"
            async with websockets.connect(url, open_timeout=WS_RECV_TIMEOUT) as ws:
                await _ws_json_send(
                    ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                )
                await _ws_json_recv(ws)
                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": str(workspace), "mcpServers": []},
                    },
                )
                session_id = (await _ws_json_recv(ws))["result"]["sessionId"]

                # Turn 1: runs to a normal completion.
                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "session/prompt",
                        "params": {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "hi"}],
                        },
                    },
                )
                first = None
                while first is None:
                    message = await _ws_json_recv(ws)
                    if message.get("id") == 3:
                        first = message
                assert first["result"] == {"stopReason": "end_turn"}

                # A stale/late cancel: nothing is in flight anymore -- a real ACP
                # client racing ESC against the turn ending sends exactly this.
                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": {"sessionId": session_id},
                    },
                )
                await asyncio.sleep(0.3)

                # Turn 2: a brand new prompt, nobody cancels it.
                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 4,
                        "method": "session/prompt",
                        "params": {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "again"}],
                        },
                    },
                )
                second = None
                while second is None:
                    message = await _ws_json_recv(ws)
                    if message.get("id") == 4:
                        second = message

            assert second["result"] == {"stopReason": "end_turn"}, (
                f"stale cancel poisoned the next turn: {second['result']}"
            )
            assert len(fake.requests) == 2
        finally:
            acp_server.should_exit = True
            fake_server.should_exit = True
            await asyncio.wait_for(acp_task, timeout=5)
            await asyncio.wait_for(fake_task, timeout=5)
        await _assert_no_leaked_tasks()


class TestCancelRacingPromptSetup:
    """BL-1 regression (adapted from the reviewer's probe B): a `session/cancel`
    that arrives while `ChatSessionBackend.prompt` is still in its pre-`start_task`
    setup (chat fetch, model resolution, message creation) must actually stop the
    turn instead of letting it run to completion (with tool_approval_mode="full",
    auto-executing tools!) while the client is told "cancelled".
    """

    async def test_cancel_immediately_after_prompt_is_never_silently_lost(
        self, slim_app, authed_user, tmp_path
    ):
        import websockets

        _user_id, token = authed_user
        fake = FakeOpenAIServer()
        fake_server, fake_task, fake_port = await _run_uvicorn(fake.app)
        acp_server, acp_task, acp_port = await _run_uvicorn(slim_app)
        try:
            await _register_fake_connection(f"http://127.0.0.1:{fake_port}/v1")
            workspace = tmp_path / "workspace"
            workspace.mkdir()

            url = f"ws://127.0.0.1:{acp_port}/api/acp/ws?token={token}"
            async with websockets.connect(url, open_timeout=WS_RECV_TIMEOUT) as ws:
                await _ws_json_send(
                    ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                )
                await _ws_json_recv(ws)
                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": str(workspace), "mcpServers": []},
                    },
                )
                session_id = (await _ws_json_recv(ws))["result"]["sessionId"]

                # Prompt immediately followed by cancel -- the user hits ESC right
                # away, before `ChatSessionBackend.prompt` has had time to do
                # anything (chat fetch, model resolution, message creation).
                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "session/prompt",
                        "params": {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "hi"}],
                        },
                    },
                )
                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": {"sessionId": session_id},
                    },
                )

                response = None
                while response is None:
                    message = await _ws_json_recv(ws)
                    if message.get("id") == 3:
                        response = message

            msgs = await ChatMessage.get_all_by_chat(session_id)
            assistant_messages = [m for m in msgs if m.role == "assistant"]
            ran_to_completion = bool(assistant_messages) and (
                assistant_messages[0].content == "Hello world"
            )

            # Whatever the outcome (the exact race between the cancel and
            # `prompt()`'s setup is not something this test controls), it must be
            # HONEST: "cancelled" and "ran to completion anyway" together is
            # exactly the bug (BL-1) -- the turn ran fully, upstream model call and
            # all, while the client was told it had been stopped.
            if response["result"] == {"stopReason": "cancelled"}:
                assert not ran_to_completion, (
                    "CANCEL LOST: reported 'cancelled' but the turn ran to "
                    f"completion anyway ({len(fake.requests)} upstream call(s))"
                )
            else:
                assert response["result"] == {"stopReason": "end_turn"}
        finally:
            acp_server.should_exit = True
            fake_server.should_exit = True
            await asyncio.wait_for(acp_task, timeout=5)
            await asyncio.wait_for(fake_task, timeout=5)
        await _assert_no_leaked_tasks()


class TestDisconnectDuringPromptSetupDoesNotOrphanTurn:
    """BL-2(a) regression (adapted from the reviewer's probe C): a client that
    disconnects immediately after sending `session/prompt` -- before
    `ChatSessionBackend.prompt` has necessarily registered a real message id in
    `_running` -- must not leave the underlying `run_chat_task` running forever.
    """

    async def test_disconnect_right_after_prompt_does_not_leave_turn_running(
        self, slim_app, authed_user, tmp_path
    ):
        import websockets

        _user_id, token = authed_user
        fake = FakeOpenAIServer()
        fake.hold_before_second_chunk = asyncio.Event()
        fake_server, fake_task, fake_port = await _run_uvicorn(fake.app)
        acp_server, acp_task, acp_port = await _run_uvicorn(slim_app)
        try:
            await _register_fake_connection(f"http://127.0.0.1:{fake_port}/v1")
            workspace = tmp_path / "workspace"
            workspace.mkdir()

            url = f"ws://127.0.0.1:{acp_port}/api/acp/ws?token={token}"
            ws = await websockets.connect(url, open_timeout=WS_RECV_TIMEOUT)
            await _ws_json_send(
                ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
            )
            await _ws_json_recv(ws)
            await _ws_json_send(
                ws,
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/new",
                    "params": {"cwd": str(workspace), "mcpServers": []},
                },
            )
            session_id = (await _ws_json_recv(ws))["result"]["sessionId"]

            await _ws_json_send(
                ws,
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/prompt",
                    "params": {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]},
                },
            )
            # Vanish immediately -- before the backend has necessarily registered
            # the underlying run_chat_task's real message id in `_running`.
            await ws.close()

            await _assert_no_leaked_tasks()

            # Release the held model stream: if the turn was truly stopped nothing
            # more happens; if it was orphaned, it now runs to completion.
            fake.hold_before_second_chunk.set()
            await asyncio.sleep(1.0)

            msgs = await ChatMessage.get_all_by_chat(session_id)
            assistant_messages = [m for m in msgs if m.role == "assistant"]
            content = assistant_messages[0].content if assistant_messages else ""
            assert content != "Hello world", "ORPHANED: turn ran to completion after disconnect"
        finally:
            fake.hold_before_second_chunk.set()
            acp_server.should_exit = True
            fake_server.should_exit = True
            await asyncio.wait_for(acp_task, timeout=15)
            await asyncio.wait_for(fake_task, timeout=15)


class TestNotifyFailureDoesNotOrphanTurn:
    """BL-2(b) regression (adapted from the reviewer's probe D): if `notify()`
    raises mid-turn (the transport died while sending a `session/update`),
    `ChatSessionBackend.prompt` must still stop the underlying `run_chat_task`
    instead of leaving it running with nobody left able to cancel it.
    """

    async def test_notify_failure_stops_the_underlying_task(self, authed_user, tmp_path):
        user_id, _token = authed_user
        fake = FakeOpenAIServer()
        fake.hold_before_second_chunk = asyncio.Event()
        fake_server, fake_task, fake_port = await _run_uvicorn(fake.app)
        try:
            await _register_fake_connection(f"http://127.0.0.1:{fake_port}/v1")
            workspace = tmp_path / "workspace"
            workspace.mkdir()

            backend = ChatSessionBackend(user_id)
            session_id = await backend.new_session(str(workspace), {})

            async def dead_notify(update):
                raise ConnectionResetError("transport is closed")

            with pytest.raises(ConnectionResetError):
                await backend.prompt(session_id, [{"type": "text", "text": "hi"}], dead_notify)

            await _assert_no_leaked_tasks()

            fake.hold_before_second_chunk.set()
            await asyncio.sleep(1.0)
            msgs = await ChatMessage.get_all_by_chat(session_id)
            assistant_messages = [m for m in msgs if m.role == "assistant"]
            content = assistant_messages[0].content if assistant_messages else ""
            assert content != "Hello world", "ORPHANED: notify() failure left the turn running"
        finally:
            fake.hold_before_second_chunk.set()
            fake_server.should_exit = True
            await asyncio.wait_for(fake_task, timeout=15)


class TestConcurrentPromptsSameSessionAcrossConnections:
    """MJ-2 regression (adapted from the reviewer's probe F): the one-prompt-per-
    session gate must be process-wide, not just per-connection -- a second
    connection `session/load`-ing the same chat must not be able to run a
    concurrent turn on it.
    """

    async def test_two_connections_same_session_one_runs_one_is_rejected_busy(
        self, slim_app, authed_user, tmp_path
    ):
        import websockets

        _user_id, token = authed_user
        fake = FakeOpenAIServer()
        fake.hold_before_second_chunk = asyncio.Event()
        fake_server, fake_task, fake_port = await _run_uvicorn(fake.app)
        acp_server, acp_task, acp_port = await _run_uvicorn(slim_app)
        try:
            await _register_fake_connection(f"http://127.0.0.1:{fake_port}/v1")
            workspace = tmp_path / "workspace"
            workspace.mkdir()
            url = f"ws://127.0.0.1:{acp_port}/api/acp/ws?token={token}"

            ws_a = await websockets.connect(url, open_timeout=WS_RECV_TIMEOUT)
            ws_b = await websockets.connect(url, open_timeout=WS_RECV_TIMEOUT)
            try:
                for ws in (ws_a, ws_b):
                    await _ws_json_send(
                        ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                    )
                    await _ws_json_recv(ws)

                await _ws_json_send(
                    ws_a,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": str(workspace), "mcpServers": []},
                    },
                )
                session_id = (await _ws_json_recv(ws_a))["result"]["sessionId"]

                # Connection B loads the SAME session (same user -> allowed).
                await _ws_json_send(
                    ws_b,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/load",
                        "params": {"sessionId": session_id, "cwd": str(workspace)},
                    },
                )
                assert (await _ws_json_recv(ws_b))["result"] == {}

                for ws, text in ((ws_a, "from-a"), (ws_b, "from-b")):
                    await _ws_json_send(
                        ws,
                        {
                            "jsonrpc": "2.0",
                            "id": 3,
                            "method": "session/prompt",
                            "params": {
                                "sessionId": session_id,
                                "prompt": [{"type": "text", "text": text}],
                            },
                        },
                    )

                await asyncio.sleep(0.5)
                fake.hold_before_second_chunk.set()

                results = {}
                for ws, name in ((ws_a, "A"), (ws_b, "B")):
                    while True:
                        m = await _ws_json_recv(ws)
                        if m.get("id") == 3:
                            results[name] = m
                            break
            finally:
                await ws_a.close()
                await ws_b.close()

            # Exactly one of the two must have actually run; the other must be
            # rejected as busy (-32004, MIN-9) without ever reaching the model.
            outcomes = [
                "ok" if "result" in results[name] else results[name]["error"]["code"]
                for name in ("A", "B")
            ]
            assert outcomes.count("ok") == 1 and outcomes.count(-32004) == 1, (
                f"expected exactly one ok and one busy(-32004): {results}"
            )
            assert len(fake.requests) == 1, (
                f"CONCURRENT TURNS ON ONE CHAT: {len(fake.requests)} upstream calls"
            )

            # The message tree has exactly one coherent branch from this turn --
            # not two concurrent branches racing each other.
            chat = await Chat.get_by_id(session_id)
            msgs = await ChatMessage.get_all_by_chat(session_id)
            assert len([m for m in msgs if m.role == "user"]) == 1
            assistant_messages = [m for m in msgs if m.role == "assistant"]
            assert len(assistant_messages) == 1
            assert chat.current_message_id == assistant_messages[0].id
        finally:
            fake.hold_before_second_chunk.set()
            acp_server.should_exit = True
            fake_server.should_exit = True
            await asyncio.wait_for(acp_task, timeout=15)
            await asyncio.wait_for(fake_task, timeout=15)
        await _assert_no_leaked_tasks()


class TestPromptRejectionEdgeCases:
    """Direct unit tests of `ChatSessionBackend.prompt`'s own rejection gates,
    below the ACP protocol layer -- so a dropped check here is caught even though
    `AcpServerConnection`'s authorized-session gate would normally prevent it from
    being reachable at all in a real client (defense in depth, mutation-tested: see
    docs/acp-server-validation-harness.md's mutation log for M6/M11).
    """

    async def test_prompt_for_a_chat_owned_by_another_user_is_rejected(self, authed_user, tmp_path):
        """M6: `ChatSessionBackend.prompt`'s own owner recheck (`chat.user_id !=
        self.user_id`) must independently reject a prompt for a chat it doesn't
        own. Must fail if that recheck is ever dropped.
        """
        owner_id, _owner_token = authed_user
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        owner_backend = ChatSessionBackend(owner_id)
        session_id = await owner_backend.new_session(str(workspace), {})

        other_user_id = await User.create(
            username="acp-other-user", password_hash="unused", role="user", created_at=now_ms()
        )
        other_backend = ChatSessionBackend(other_user_id)

        async def notify(update):
            raise AssertionError("must never stream an update for a rejected prompt")

        with pytest.raises(PromptRejected) as exc_info:
            await other_backend.prompt(session_id, [{"type": "text", "text": "hi"}], notify)
        assert exc_info.value.message == "session not found"
        assert not isinstance(exc_info.value, PromptBusy)

        # Never touched: no message rows, no claim left behind.
        assert await ChatMessage.get_all_by_chat(session_id) == []
        assert session_id not in ChatSessionBackend._inflight_chats

    async def test_empty_prompt_text_is_rejected(self, authed_user, tmp_path):
        """M11: a whitespace-only/no-text prompt must be rejected before ever
        touching model resolution or creating any message rows. Must fail if this
        gate is ever dropped.
        """
        user_id, _token = authed_user
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        backend = ChatSessionBackend(user_id)
        session_id = await backend.new_session(str(workspace), {})

        async def notify(update):
            raise AssertionError("must never stream an update for a rejected prompt")

        with pytest.raises(PromptRejected) as exc_info:
            await backend.prompt(session_id, [{"type": "text", "text": "   \n  "}], notify)
        assert exc_info.value.message == "empty prompt"

        assert await ChatMessage.get_all_by_chat(session_id) == []
        assert session_id not in ChatSessionBackend._inflight_chats


class TestPromptToolCallEndToEnd:
    """MIN-8(iv): a full tool-call turn end to end -- the fake model's first
    response emits a `run_command` tool call, ACP chats run with
    `tool_approval_mode="full"` (see `ChatSessionBackend.new_session`) so it
    auto-executes for real (a real subprocess via `cptr/utils/tools.py:run_command`,
    workspace-scoped to `tmp_path`), then the second response (after the tool
    result comes back) streams the final text. Asserts the client sees the tool
    call, a completed tool_call_update carrying the real output, and the final
    text -- exercising `acp_updates_from_queue_item` against every shape a real
    tool-using turn produces, not just the hand-scripted ones above.

    `run_command` was chosen over any read-only tool because it is always
    available (no feature flags), has a trivially safe, deterministic invocation
    (`echo acp-e2e-ok`), and is exactly the tool most likely to be gated behind
    approval in the first place -- so it's the most meaningful one to prove
    `tool_approval_mode="full"` actually auto-executes over ACP.
    """

    async def test_run_command_tool_call_streams_call_output_and_final_text(
        self, slim_app, authed_user, tmp_path
    ):
        import websockets

        _user_id, token = authed_user
        fake = FakeOpenAIToolCallServer()
        fake_server, fake_task, fake_port = await _run_uvicorn(fake.app)
        acp_server, acp_task, acp_port = await _run_uvicorn(slim_app)
        try:
            await _register_fake_connection(f"http://127.0.0.1:{fake_port}/v1")
            workspace = tmp_path / "workspace"
            workspace.mkdir()

            url = f"ws://127.0.0.1:{acp_port}/api/acp/ws?token={token}"
            async with websockets.connect(url, open_timeout=WS_RECV_TIMEOUT) as ws:
                await _ws_json_send(
                    ws, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                )
                await _ws_json_recv(ws)
                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "session/new",
                        "params": {"cwd": str(workspace), "mcpServers": []},
                    },
                )
                session_id = (await _ws_json_recv(ws))["result"]["sessionId"]

                await _ws_json_send(
                    ws,
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "session/prompt",
                        "params": {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "run the command"}],
                        },
                    },
                )

                tool_updates: list[dict] = []
                texts: list[str] = []
                prompt_response = None
                while prompt_response is None:
                    message = await _ws_json_recv(ws)
                    if message.get("id") == 3:
                        prompt_response = message
                        break
                    update = message.get("params", {}).get("update", {})
                    if update.get("sessionUpdate") in ("tool_call", "tool_call_update"):
                        tool_updates.append(update)
                    elif update.get("sessionUpdate") == "agent_message_chunk":
                        texts.append(update["content"]["text"])

            assert prompt_response["result"] == {"stopReason": "end_turn"}
            assert "".join(texts) == "ran it"

            call_ids = {u["toolCallId"] for u in tool_updates}
            assert call_ids == {"call_acp_e2e"}

            first_call = tool_updates[0]
            assert first_call["sessionUpdate"] == "tool_call"
            assert first_call["status"] == "in_progress"
            assert first_call["rawInput"]["command"] == "echo acp-e2e-ok"

            completed_updates = [u for u in tool_updates if u.get("status") == "completed"]
            assert completed_updates, "no tool_call_update ever reported status=completed"

            output_texts = [
                block["content"]["text"]
                for u in tool_updates
                for block in u.get("content") or []
                if isinstance(block, dict) and block.get("content", {}).get("type") == "text"
            ]
            assert any("acp-e2e-ok" in text for text in output_texts), (
                f"tool output never reached the client: {output_texts}"
            )

            # Two model calls: the tool-call turn, then the post-tool-result turn.
            assert len(fake.requests) == 2

            messages = await ChatMessage.get_all_by_chat(session_id)
            assistant_messages = [m for m in messages if m.role == "assistant"]
            assert len(assistant_messages) == 1
            assert assistant_messages[0].done is True
            output_item_types = [item.get("type") for item in assistant_messages[0].output or []]
            assert "function_call" in output_item_types
            assert "function_call_output" in output_item_types
        finally:
            acp_server.should_exit = True
            fake_server.should_exit = True
            await asyncio.wait_for(acp_task, timeout=10)
            await asyncio.wait_for(fake_task, timeout=10)
