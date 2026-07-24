"""In-process integration tests for the ACP WebSocket endpoint (Tier 1).

Builds a slim `FastAPI()` app mounting only `cptr.routers.acp.router` (no full
`cptr.app` lifespan -- no bot manager, scheduler, timers, browser cleanup) against a
temp SQLite database, and drives it with Starlette's synchronous `TestClient`.

DB isolation mechanism (investigated in `cptr/utils/db.py` / `cptr/env.py`): `DB_FILE`
/ `DATA_DIR` are read from the environment once, at import time, and then bound as
plain module-level names in `cptr.utils.db` and `cptr.utils.config` (`from cptr.env
import DATA_DIR, DB_FILE`). Because those names are rebound into each importing
module's own namespace, patching `cptr.env` after import has no effect on already
loaded modules, and test file import order relative to other test modules in the
suite is not something we control. The reliable fix is to monkeypatch the *consuming*
modules' own module-level attributes directly (`cptr.utils.db.DATA_DIR/DB_FILE`,
`cptr.utils.config.DATA_DIR/CONFIG_FILE`) and reset the lazily created engine/session
globals, so `get_engine()` builds a fresh `AsyncEngine` pointed at the tmp file
regardless of what ran before.

Auth mechanism: `check_access` defaults to `AuthMode.PASSWORD` when `config.toml` has
no `[auth]` section (the common case for a fresh tmp `CONFIG_FILE`), which verifies a
JWT cookie/`?token=` via `verify_token`. We mint a token with `create_token` -- the
exact helper `routers/auth.py:login` uses -- against the same `_get_jwt_secret()`
(auto-generated into the isolated `CONFIG_FILE` on first use), after creating a real
`User` row so `Chat.user_id` foreign-keys resolve to something real.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient, WebSocketDisconnect

import cptr.utils.config as config_module
import cptr.utils.db as db_module
from cptr.models import Chat, User
from cptr.routers.acp import router as acp_router
from cptr.utils.config import create_token, now_ms
from cptr.utils.db import init_db


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

    db_module._engine = None
    db_module._async_session = None

    await init_db()

    yield

    engine = db_module.get_engine()
    await engine.dispose()
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
    """Create a real user + return (user_id, auth_cookies)."""
    user_id = await User.create(
        username="acp-tester",
        password_hash="unused-in-this-test",
        role="user",
        created_at=now_ms(),
    )
    token = create_token(user_id, "acp-tester", role="user")
    return user_id, {"cptr_session": token}


class TestAuth:
    def test_connect_without_token_closes_4001(self, slim_app, isolated_db):
        client = TestClient(slim_app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/acp"):
                pass
        assert exc_info.value.code == 4001

    async def test_connect_with_bad_token_closes_4001(self, slim_app, isolated_db):
        client = TestClient(slim_app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/acp?token=not-a-real-token"):
                pass
        assert exc_info.value.code == 4001


class TestHandshakeAndSessionMapping:
    async def test_session_new_creates_real_chat_row(self, slim_app, authed_user):
        user_id, cookies = authed_user
        client = TestClient(slim_app, cookies=cookies)

        with client.websocket_connect("/ws/acp") as ws:
            ws.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            init_response = ws.receive_json()
            assert init_response["result"]["authMethods"] == []

            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/new",
                    "params": {"cwd": "/workspace/demo", "mcpServers": []},
                }
            )
            new_response = ws.receive_json()
            session_id = new_response["result"]["sessionId"]
            assert isinstance(session_id, str) and session_id

        chat = await Chat.get_by_id(session_id)
        assert chat is not None
        assert chat.user_id == user_id
        assert chat.meta["workspace"] == "/workspace/demo"

    async def test_session_load_found_and_not_found(self, slim_app, authed_user):
        user_id, cookies = authed_user
        client = TestClient(slim_app, cookies=cookies)

        with client.websocket_connect("/ws/acp") as ws:
            ws.send_json({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            ws.receive_json()
            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "session/new",
                    "params": {"cwd": "/workspace/demo", "mcpServers": []},
                }
            )
            session_id = ws.receive_json()["result"]["sessionId"]

            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "session/load",
                    "params": {"sessionId": session_id, "cwd": "/workspace/demo"},
                }
            )
            load_response = ws.receive_json()
            assert load_response["result"] == {}

            ws.send_json(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "session/load",
                    "params": {"sessionId": "does-not-exist", "cwd": "/workspace/demo"},
                }
            )
            missing_response = ws.receive_json()
            assert missing_response["error"]["code"] == -32001
