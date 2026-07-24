"""ACP (Agent Client Protocol) server endpoint: exposes cptr as a network-reachable
ACP agent over WebSocket. Auth mirrors `cptr/routers/events.py:events_ws` -- token
from the `cptr_session` cookie or a `?token=` query param, checked with the same
`check_access` used by the HTTP auth middleware, closing unauthenticated connections
with code 4001 before ever accepting them -- plus one requirement `events_ws` doesn't
have: `auth.user_id` must actually be set (not just `auth is not None`), since every
ACP session is scoped to a user id via `ChatSessionBackend`.

Note this is a WebSocket route: Starlette's HTTP-scoped middleware (including
`cptr.app`'s `auth_middleware`, a `@app.middleware("http")` function) never runs for
websocket-scope connections, so the `check_access` call below is the *only* auth
check this endpoint gets -- it is not a redundant second check on top of the HTTP
middleware.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from cptr.utils.acp_server import AcpServerConnection, ChatSessionBackend
from cptr.utils.agents.acp_transport import WebSocketTransport
from cptr.utils.config import check_access

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/acp", tags=["acp"])


@router.websocket("/ws")
async def acp_ws(websocket: WebSocket):
    """ACP agent endpoint (`/api/acp/ws`): initialize / authenticate / session/new /
    session/load, with `session/prompt`/`session/cancel` gated by a per-connection
    authorized-session set (see `cptr/utils/acp_server.py`). Real `session/prompt`
    bridging into `run_chat_task` lands in step 3; for now it is stubbed by
    `ChatSessionBackend`.
    """
    client_host = websocket.client.host if websocket.client else "127.0.0.1"
    token = websocket.cookies.get("cptr_session") or websocket.query_params.get("token")
    auth = check_access(client_host=client_host, jwt_token=token)
    if auth is None or not auth.user_id:
        await websocket.close(code=4001, reason="unauthorized")
        return

    await websocket.accept()
    logger.info("ACP WS connected for user %s", auth.user_id)

    transport = WebSocketTransport(websocket)
    backend = ChatSessionBackend(auth.user_id)
    connection = AcpServerConnection(transport, backend)
    try:
        await connection.serve()
    finally:
        await transport.close()
