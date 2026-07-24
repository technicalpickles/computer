"""Server-side ACP (Agent Client Protocol) handshake + session mapping.

Mirrors `cptr/utils/agents/acp.py` (the client side) on the same transport seam
(`cptr/utils/agents/acp_transport.py`): `AcpServerConnection` speaks the agent side of
the JSON-RPC conversation over any `AcpTransport`, dispatching onto an injectable
`SessionBackend` so the protocol logic here is fully unit-testable without a database
or a real WebSocket (see `tests/test_acp_server.py`).

Step 2 of the build order (docs/acp-server-validation-harness.md): `initialize`,
`authenticate`, `session/new`, `session/load`. `session/prompt` bridging into
`run_chat_task` and `session/cancel` semantics land in step 3 -- here they are stubbed
(`SessionBackend.prompt` raises `NotImplementedError`, `cancel` is a no-op hook).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from cptr.utils.agents.acp_transport import AcpTransport, TransportClosed

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

# JSON-RPC / ACP error codes used by this connection.
_ERR_SESSION_NOT_FOUND = -32001
_ERR_NOT_INITIALIZED = -32002
_ERR_INVALID_PARAMS = -32602
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INTERNAL = -32603


class SessionBackend(Protocol):
    """The seam between ACP session lifecycle and whatever owns chat state."""

    async def new_session(self, cwd: str, params: dict[str, Any]) -> str:
        """Create a new session for `cwd`; return its session id."""
        ...

    async def load_session(self, session_id: str, cwd: str) -> bool:
        """Resume an existing session. False means unknown/forbidden."""
        ...

    async def prompt(self, session_id: str, prompt_items: list[dict[str, Any]]) -> dict[str, Any]:
        """Run one prompt turn. Step 2 stub: raise `NotImplementedError`."""
        ...

    async def cancel(self, session_id: str) -> None:
        """Cancel an in-flight turn for `session_id`."""
        ...


class _JsonRpcError(Exception):
    """Internal control-flow exception: carries a JSON-RPC error code + message."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class AcpServerConnection:
    """Drives one ACP connection over an `AcpTransport` against a `SessionBackend`.

    `serve()` loops `transport.receive()` until `TransportClosed`, dispatching each
    message sequentially. Dispatch lives in `_handle_request` as a single method so
    step 3 can move prompt handling onto a background task without rewriting the loop.
    """

    def __init__(self, transport: AcpTransport, backend: SessionBackend) -> None:
        self.transport = transport
        self.backend = backend
        self._initialized = False

    async def serve(self) -> None:
        while True:
            try:
                message = await self.transport.receive()
            except TransportClosed:
                return
            await self._handle_message(message)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        if not isinstance(message, dict):
            return

        has_id = "id" in message
        request_id = message.get("id")
        method = message.get("method")

        if not isinstance(method, str) or not method:
            # Malformed inbound: no usable method. Error it back if it claims to be a
            # request (has an id); otherwise there is nothing sensible to reply to.
            if has_id:
                await self._send_error(
                    request_id, _ERR_METHOD_NOT_FOUND, "invalid request: missing method"
                )
            return

        params = message.get("params")
        if not isinstance(params, dict):
            params = {}

        try:
            result = await self._handle_request(method, params)
        except _JsonRpcError as exc:
            if has_id:
                await self._send_error(request_id, exc.code, exc.message)
            return
        except Exception as exc:  # noqa: BLE001 - handler errors must not kill the loop
            logger.exception("ACP server: unhandled error in %s", method)
            if has_id:
                await self._send_error(request_id, _ERR_INTERNAL, str(exc))
            return

        if has_id:
            await self.transport.send({"jsonrpc": "2.0", "id": request_id, "result": result})
        # Notifications (no id) never get a response, even on success.

    async def _handle_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            self._initialized = True
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "agentCapabilities": {"loadSession": True},
                # MUST stay empty: auth happens at the transport layer (WS token)
                # before the ACP protocol ever starts. Advertising a method we don't
                # implement breaks real clients that eagerly call the first
                # advertised method (our own AcpClient does this) and it is exactly
                # the interop bug hit testing against Zed's claude-code-acp.
                "authMethods": [],
                "agentInfo": {"name": "cptr", "version": "0"},
            }

        if method == "authenticate":
            # Tolerated unconditionally: never an error, even though we never
            # advertise a method for a client to pick.
            return {}

        if not self._initialized:
            raise _JsonRpcError(_ERR_NOT_INITIALIZED, "not initialized")

        if method == "session/new":
            cwd = params.get("cwd")
            if not isinstance(cwd, str) or not cwd:
                raise _JsonRpcError(_ERR_INVALID_PARAMS, "invalid params: cwd (string) required")
            session_id = await self.backend.new_session(cwd, params)
            return {"sessionId": session_id}

        if method == "session/load":
            session_id = params.get("sessionId")
            cwd = params.get("cwd")
            if (
                not isinstance(session_id, str)
                or not session_id
                or not isinstance(cwd, str)
                or not cwd
            ):
                raise _JsonRpcError(
                    _ERR_INVALID_PARAMS, "invalid params: sessionId and cwd (strings) required"
                )
            found = await self.backend.load_session(session_id, cwd)
            if not found:
                raise _JsonRpcError(_ERR_SESSION_NOT_FOUND, "session not found")
            return {}

        if method == "session/prompt":
            session_id = params.get("sessionId")
            prompt_items = params.get("prompt")
            if not isinstance(prompt_items, list):
                prompt_items = []
            try:
                return await self.backend.prompt(session_id, prompt_items)
            except NotImplementedError:
                raise _JsonRpcError(
                    _ERR_METHOD_NOT_FOUND, "session/prompt not supported yet"
                ) from None

        if method == "session/cancel":
            session_id = params.get("sessionId")
            if isinstance(session_id, str) and session_id:
                await self.backend.cancel(session_id)
            return {}

        raise _JsonRpcError(_ERR_METHOD_NOT_FOUND, f"method not found: {method}")

    async def _send_error(self, request_id: Any, code: int, message: str) -> None:
        await self.transport.send(
            {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
        )


class ChatSessionBackend:
    """Maps ACP sessions onto `Chat` rows scoped to one authenticated user."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id

    async def new_session(self, cwd: str, params: dict[str, Any]) -> str:
        from cptr.models import Chat
        from cptr.utils.config import now_ms

        chat = await Chat.create(
            user_id=self.user_id,
            title="ACP session",
            meta={
                "workspace": cwd,
                "params": dict(params.get("params") or {}),
                "acp": {"created_via": "acp"},
            },
            created_at=now_ms(),
        )
        # step 3: mirror routers/chat.py's chats-dir + gitignore setup for workspace
        # chats (chats_dir.mkdir + ensure_cptr_gitignored) once prompt bridging
        # actually persists messages into that directory.
        return chat.id

    async def load_session(self, session_id: str, cwd: str) -> bool:
        from cptr.models import Chat

        chat = await Chat.get_by_id(session_id)
        if chat is None or chat.user_id != self.user_id:
            return False
        return True

    async def prompt(self, session_id: str, prompt_items: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    async def cancel(self, session_id: str) -> None:
        return None
