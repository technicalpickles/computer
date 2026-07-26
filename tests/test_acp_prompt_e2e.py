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


async def _run_uvicorn(app: FastAPI) -> tuple[uvicorn.Server, asyncio.Task, int]:
    """Start `app` with uvicorn on an ephemeral port, as a background task on the
    *current* event loop. Returns (server, task, bound_port); caller tears down with
    `server.should_exit = True; await task`.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning", lifespan="off")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    while not server.started:
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
