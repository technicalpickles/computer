"""Server-side ACP (Agent Client Protocol) handshake + session mapping.

Mirrors `cptr/utils/agents/acp.py` (the client side) on the same transport seam
(`cptr/utils/agents/acp_transport.py`): `AcpServerConnection` speaks the agent side of
the JSON-RPC conversation over any `AcpTransport`, dispatching onto an injectable
`SessionBackend` so the protocol logic here is fully unit-testable without a database
or a real WebSocket (see `tests/test_acp_server.py`).

Step 2 of the build order (docs/acp-server-validation-harness.md): `initialize`,
`authenticate`, `session/new` (validated: absolute `cwd`, no `..` segments, bounded
length, capped call count per connection), `session/load` (ownership + workspace
match). `session/prompt`/`session/cancel` are gated by a per-connection
authorized-session set so a connection can only ever act on sessions it created or
loaded itself; unauthorized ids are indistinguishable from nonexistent (`-32001`).
`session/prompt` bridging into `run_chat_task` and real `session/cancel` semantics
land in step 3 -- here they are stubbed (`SessionBackend.prompt` raises
`NotImplementedError`, `cancel` is a no-op hook).
"""

from __future__ import annotations

import logging
import posixpath
from typing import Any, Protocol

from cptr.utils.agents.acp_transport import AcpParseError, AcpTransport, TransportClosed

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = 1

# JSON-RPC / ACP error codes used by this connection.
_ERR_PARSE_ERROR = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INVALID_PARAMS = -32602
_ERR_INTERNAL = -32603
_ERR_SESSION_LIMIT = -32000
_ERR_SESSION_NOT_FOUND = -32001
_ERR_NOT_INITIALIZED = -32002

# Bounds for `session/new`, independent of anything the DB layer would enforce.
_MAX_CWD_LENGTH = 4096
_DEFAULT_SESSION_NEW_CAP = 64


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

    Tracks two pieces of per-connection state that are *not* the backend's concern:
    a count of `session/new` calls (bounded by `session_new_cap`, protecting the
    backend from an unbounded number of session-creation side effects on one
    connection) and the set of session ids this connection has actually created or
    successfully loaded. `session/prompt`/`session/cancel` are gated against that set
    so a connection can only ever act on sessions it is authorized for on itself --
    the backend's per-user check (`ChatSessionBackend.load_session`) still applies on
    top of this, but this set also stops a connection from touching a session it
    never loaded/created *this* connection, even if the backend would allow it.
    """

    def __init__(
        self,
        transport: AcpTransport,
        backend: SessionBackend,
        *,
        session_new_cap: int = _DEFAULT_SESSION_NEW_CAP,
    ) -> None:
        self.transport = transport
        self.backend = backend
        self._initialized = False
        self._session_new_cap = session_new_cap
        self._session_new_count = 0
        self._authorized_sessions: set[str] = set()

    async def serve(self) -> None:
        while True:
            try:
                message = await self.transport.receive()
            except TransportClosed:
                return
            except AcpParseError:
                await self.transport.send(
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": _ERR_PARSE_ERROR, "message": "parse error"},
                    }
                )
                continue
            await self._handle_message(message)

    async def _handle_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            # Non-object top-level payload (e.g. a JSON-RPC batch array). We can't
            # recover an id from it, so per spec it gets `id: null`.
            await self._send_error(
                None, _ERR_INVALID_REQUEST, "invalid request: expected a JSON object"
            )
            return

        has_id = "id" in message
        request_id = message.get("id")
        method = message.get("method")

        if not isinstance(method, str) or not method:
            # Malformed inbound: no usable method, so we can't tell whether the sender
            # meant this as a notification. Always answer, using the id when we have
            # one and `null` otherwise.
            await self._send_error(
                request_id if has_id else None,
                _ERR_INVALID_REQUEST,
                "invalid request: missing method",
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
        except Exception:  # noqa: BLE001 - handler errors must not kill the loop
            logger.exception("ACP server: unhandled error in %s", method)
            if has_id:
                # Never leak exception text to the client -- it can carry SQL/schema
                # details or other internals. Full detail stays server-side above.
                await self._send_error(request_id, _ERR_INTERNAL, "internal error")
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
            if self._session_new_count >= self._session_new_cap:
                raise _JsonRpcError(_ERR_SESSION_LIMIT, "session limit reached")
            cwd = self._validate_cwd(params.get("cwd"))
            self._session_new_count += 1
            session_id = await self.backend.new_session(cwd, params)
            self._authorized_sessions.add(session_id)
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
            self._authorized_sessions.add(session_id)
            return {}

        if method == "session/prompt":
            session_id = params.get("sessionId")
            prompt_items = params.get("prompt")
            if not isinstance(prompt_items, list):
                prompt_items = []
            # Same error as "not found": an id this connection never created/loaded is
            # indistinguishable from one that doesn't exist at all, whether or not the
            # backend would actually recognize it.
            if not isinstance(session_id, str) or session_id not in self._authorized_sessions:
                raise _JsonRpcError(_ERR_SESSION_NOT_FOUND, "session not found")
            try:
                return await self.backend.prompt(session_id, prompt_items)
            except NotImplementedError:
                raise _JsonRpcError(
                    _ERR_METHOD_NOT_FOUND, "session/prompt not supported yet"
                ) from None

        if method == "session/cancel":
            session_id = params.get("sessionId")
            # Unauthorized/unknown ids are silently ignored (never reach the backend)
            # rather than errored: `session/cancel` is a notification-style, best-effort
            # call in the ACP spec, and erroring here would reveal whether an id exists.
            if isinstance(session_id, str) and session_id in self._authorized_sessions:
                await self.backend.cancel(session_id)
            return {}

        raise _JsonRpcError(_ERR_METHOD_NOT_FOUND, f"method not found: {method}")

    @staticmethod
    def _validate_cwd(cwd: Any) -> str:
        if not isinstance(cwd, str) or not cwd:
            raise _JsonRpcError(_ERR_INVALID_PARAMS, "invalid params: cwd (string) required")
        if len(cwd) > _MAX_CWD_LENGTH:
            raise _JsonRpcError(_ERR_INVALID_PARAMS, "invalid params: cwd too long")
        if not posixpath.isabs(cwd):
            raise _JsonRpcError(_ERR_INVALID_PARAMS, "invalid params: cwd must be an absolute path")
        if ".." in cwd.split("/"):
            raise _JsonRpcError(
                _ERR_INVALID_PARAMS, "invalid params: cwd must not contain '..' segments"
            )
        return cwd

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
                # step 3: chat params (e.g. tool_approval_mode) will be sourced
                # explicitly -- `params` here is the raw `session/new` JSON-RPC params
                # dict, which has no nested "params" key of its own.
                "params": {},
                "acp": {"created_via": "acp"},
            },
            created_at=now_ms(),
        )
        # step 3: mirror routers/chat.py's chats-dir + gitignore setup for workspace
        # chats (chats_dir.mkdir + ensure_cptr_gitignored) once prompt bridging
        # actually persists messages into that directory.
        return chat.id

    async def load_session(self, session_id: str, cwd: str) -> bool:
        """Resume an existing session: True iff `session_id` names a `Chat` owned by
        this connection's user AND (when the chat recorded a workspace at
        `session/new` time) the requested `cwd` matches it exactly. Returning False
        for either failure is intentional and lets the caller answer with the same
        `-32001 session not found` either way -- ownership and workspace mismatches
        must be indistinguishable from a nonexistent session.

        This is the full extent of "load" today: no resume-state mapping (message
        replay) happens here yet -- that lands in step 3.
        """
        from cptr.models import Chat

        chat = await Chat.get_by_id(session_id)
        if chat is None or chat.user_id != self.user_id:
            return False
        workspace = (chat.meta or {}).get("workspace")
        if workspace and workspace != cwd:
            return False
        return True

    async def prompt(self, session_id: str, prompt_items: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError

    async def cancel(self, session_id: str) -> None:
        return None
