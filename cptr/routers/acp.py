"""ACP (Agent Client Protocol) server endpoint: exposes cptr as a network-reachable
ACP agent over WebSocket. Auth mirrors `cptr/routers/events.py:events_ws` exactly --
token from the `cptr_session` cookie or a `?token=` query param, checked with the
same `check_access` used by the HTTP auth middleware, closing unauthenticated
connections with code 4001 before ever accepting them.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from cptr.utils.acp_server import AcpServerConnection, ChatSessionBackend
from cptr.utils.agents.acp_transport import WebSocketTransport
from cptr.utils.config import check_access

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/acp")
async def acp_ws(websocket: WebSocket):
    """ACP agent endpoint: initialize / authenticate / session/new / session/load.

    `session/prompt` bridging and `session/cancel` land in step 3; for now they are
    stubbed by `ChatSessionBackend` (see `cptr/utils/acp_server.py`).
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
