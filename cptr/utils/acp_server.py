"""Server-side ACP (Agent Client Protocol) handshake + session mapping.

Mirrors `cptr/utils/agents/acp.py` (the client side) on the same transport seam
(`cptr/utils/agents/acp_transport.py`): `AcpServerConnection` speaks the agent side of
the JSON-RPC conversation over any `AcpTransport`, dispatching onto an injectable
`SessionBackend` so the protocol logic here is fully unit-testable without a database
or a real WebSocket (see `tests/test_acp_server.py`).

Step 2 of the build order (docs/acp-server-validation-harness.md) landed: `initialize`,
`authenticate`, `session/new` (validated: absolute `cwd`, no `..` segments, bounded
length, capped call count per connection), `session/load` (ownership + workspace
match). `session/prompt`/`session/cancel` are gated by a per-connection
authorized-session set so a connection can only ever act on sessions it created or
loaded itself; unauthorized ids are indistinguishable from nonexistent (`-32001`).

Step 3 lands here: `session/prompt` bridges into `run_chat_task` (`ChatSessionBackend`)
and real `session/cancel` semantics. Prompt handling runs on a background
`asyncio.Task` per session so the serve loop stays live to receive `session/cancel`
mid-turn -- see `AcpServerConnection._dispatch_prompt` / `_cancel_all_prompts`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import posixpath
from contextlib import suppress
from typing import Any, Awaitable, Callable, Protocol

from cptr.utils.agents.acp_transport import AcpParseError, AcpTransport, TransportClosed

logger = logging.getLogger(__name__)

NotifyFn = Callable[[dict[str, Any]], Awaitable[None]]

PROTOCOL_VERSION = 1

# JSON-RPC / ACP error codes used by this connection.
_ERR_PARSE_ERROR = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INVALID_PARAMS = -32602
_ERR_INTERNAL = -32603
_ERR_PROMPT_REJECTED = -32000  # PromptRejected: safe, static rejection reasons
_ERR_SESSION_NOT_FOUND = -32001
_ERR_NOT_INITIALIZED = -32002
_ERR_SESSION_LIMIT_REACHED = -32003  # `session/new` cap only (MIN-9: split from -32000)
_ERR_PROMPT_BUSY = -32004  # "prompt already in progress", connection- or process-wide

# Bounds for `session/new`, independent of anything the DB layer would enforce.
_MAX_CWD_LENGTH = 4096
_DEFAULT_SESSION_NEW_CAP = 64

# Cancellation timing (`ChatSessionBackend.prompt`/`AcpServerConnection._cancel_all_prompts`).
# `_PROMPT_TASK_GRACE_SECONDS` (used by `_cancel_all_prompts` to bound how long it
# waits for an in-flight `session/prompt` task to finish on its own before force-
# cancelling it) must stay comfortably larger than `prompt()`'s own worst-case
# cancelled-turn timeline below it, or a disconnect can force-cancel the ACP task
# while `prompt()` is still legitimately winding down -- delivering a second,
# overlapping cancellation into the same `run_chat_task` that can leave it stuck
# mid-cleanup (a real regression hit while tightening this bound during review).
_CANCEL_QUEUE_GRACE_SECONDS = 5.0  # prompt()'s bounded wait on the queue post-cancel
_CANCEL_SETTLE_GRACE_SECONDS = 2.0  # MIN-1: settle-wait for chat_task._tasks to clear
_PROMPT_TASK_GRACE_SECONDS = _CANCEL_QUEUE_GRACE_SECONDS + _CANCEL_SETTLE_GRACE_SECONDS + 1.0


class SessionBackend(Protocol):
    """The seam between ACP session lifecycle and whatever owns chat state."""

    async def new_session(self, cwd: str, params: dict[str, Any]) -> str:
        """Create a new session for `cwd`; return its session id."""
        ...

    async def load_session(self, session_id: str, cwd: str) -> bool:
        """Resume an existing session. False means unknown/forbidden."""
        ...

    async def prompt(
        self, session_id: str, prompt_items: list[dict[str, Any]], notify: NotifyFn
    ) -> dict[str, Any]:
        """Run one prompt turn, calling `notify(update)` for each `session/update`
        the connection should send while the turn is in flight. Returns the
        `session/prompt` JSON-RPC result (e.g. `{"stopReason": "end_turn"}`).

        Raise `PromptRejected(message)` for a safe, user-facing rejection (e.g. no
        model configured), or its subclass `PromptBusy(message)` when a turn for
        this session is already in flight elsewhere; any other exception becomes a
        generic internal error to the client (never `str(exc)` -- full detail stays
        server-side).
        """
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


class PromptRejected(Exception):
    """Raised by `SessionBackend.prompt` for a safe, static, user-facing rejection
    reason (e.g. "no model configured", "empty prompt"). Never wrap arbitrary
    exception text in this -- `message` is sent to the client verbatim as the
    JSON-RPC error message.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PromptBusy(PromptRejected):
    """Raised by `SessionBackend.prompt` when this session already has a turn in
    flight -- process-wide, not just on this connection (MJ-2: `ChatSessionBackend`
    claims a chat id in a class-level registry before doing anything else). Maps to
    its own `-32004` (MIN-9) rather than sharing `-32000` with every other
    `PromptRejected` reason, so a client can tell "busy, retry" apart from a static
    rejection like "no model configured".
    """


class AcpServerConnection:
    """Drives one ACP connection over an `AcpTransport` against a `SessionBackend`.

    `serve()` loops `transport.receive()` until `TransportClosed`, dispatching each
    message sequentially -- except `session/prompt`, which `_dispatch_prompt` hands
    off to a background `asyncio.Task` (see `_prompt_tasks`) so a long-running turn
    never blocks the loop from handling `session/cancel` (or anything else) in the
    meantime. Every other method still goes through `_handle_request` synchronously.

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
        # One in-flight `session/prompt` task per session id. Tracked here (not just
        # inside the backend) so the connection itself can reject a second concurrent
        # prompt for the same session without ever reaching the backend, and so a
        # disconnect can find and clean up every still-running prompt.
        self._prompt_tasks: dict[str, asyncio.Task] = {}

    async def serve(self) -> None:
        try:
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
        finally:
            # A disconnected client must not leave a turn running forever: cancel
            # every in-flight prompt for this connection and make sure the backend
            # actually stops the underlying work (not just our local bookkeeping).
            await self._cancel_all_prompts()

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

        if method == "session/prompt":
            # Dispatched onto a background task (never awaited here) so the serve
            # loop stays live to receive `session/cancel` for this same session (or
            # `session/prompt`/anything else for a different one) while this turn
            # runs. The task itself sends the eventual JSON-RPC response.
            await self._dispatch_prompt(request_id, has_id, params)
            return

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

    async def _dispatch_prompt(self, request_id: Any, has_id: bool, params: dict[str, Any]) -> None:
        if not self._initialized:
            if has_id:
                await self._send_error(request_id, _ERR_NOT_INITIALIZED, "not initialized")
            return

        session_id = params.get("sessionId")
        prompt_items = params.get("prompt")
        if not isinstance(prompt_items, list):
            prompt_items = []
        # Same error as "not found": an id this connection never created/loaded is
        # indistinguishable from one that doesn't exist at all, whether or not the
        # backend would actually recognize it.
        if not isinstance(session_id, str) or session_id not in self._authorized_sessions:
            if has_id:
                await self._send_error(request_id, _ERR_SESSION_NOT_FOUND, "session not found")
            return

        if session_id in self._prompt_tasks:
            if has_id:
                await self._send_error(request_id, _ERR_PROMPT_BUSY, "prompt already in progress")
            return

        async def notify(update: dict[str, Any]) -> None:
            await self.transport.send(
                {
                    "jsonrpc": "2.0",
                    "method": "session/update",
                    "params": {"sessionId": session_id, "update": update},
                }
            )

        async def run() -> None:
            error: tuple[int, str] | None = None
            result: dict[str, Any] = {}
            try:
                result = await self.backend.prompt(session_id, prompt_items, notify)
            except NotImplementedError:
                error = (_ERR_METHOD_NOT_FOUND, "session/prompt not supported yet")
            except PromptBusy as exc:
                # MIN-9: PromptBusy is a PromptRejected subclass -- must be checked
                # first, and gets its own error code instead of sharing -32000.
                error = (_ERR_PROMPT_BUSY, exc.message)
            except PromptRejected as exc:
                error = (_ERR_PROMPT_REJECTED, exc.message)
            except Exception:  # noqa: BLE001 - never let a prompt task die unhandled
                logger.exception(
                    "ACP server: unhandled error in session/prompt for session %s", session_id
                )
                error = (_ERR_INTERNAL, "internal error")
            finally:
                self._prompt_tasks.pop(session_id, None)

            if not has_id:
                return
            try:
                if error is not None:
                    await self._send_error(request_id, *error)
                else:
                    await self.transport.send(
                        {"jsonrpc": "2.0", "id": request_id, "result": result}
                    )
            except Exception:
                # The connection is gone by the time we tried to answer -- there's
                # nothing left to report back to, and this must never surface as an
                # unretrieved task exception.
                logger.debug(
                    "ACP server: failed to deliver session/prompt response for %s",
                    session_id,
                    exc_info=True,
                )

        self._prompt_tasks[session_id] = asyncio.create_task(run())

    async def _cancel_all_prompts(self) -> None:
        """Best-effort, bounded shutdown of every in-flight `session/prompt` on this
        connection. Called from `serve()`'s `finally` on disconnect, so it must run
        to completion even when the caller races a cancellation against it (e.g. a
        test harness cancelling the `serve()` task directly, or a slow shutdown):
        `contextlib.suppress(Exception)` does **not** catch `asyncio.CancelledError`
        (it's a `BaseException`, not an `Exception`), so every await below is
        individually guarded against both regular exceptions and cancellation, and
        bounded to a couple of seconds each so a stuck backend or task never blocks
        shutdown for long (BL-2).
        """
        session_ids = list(self._prompt_tasks.keys())

        for session_id in session_ids:
            # Ask the backend to actually stop the underlying work first: this is
            # what makes the prompt task's own consumption loop unwind and return
            # normally (with a `cancelled` stopReason) instead of needing to be
            # force-cancelled below.
            with suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(
                    asyncio.shield(self.backend.cancel(session_id)),
                    timeout=_CANCEL_SETTLE_GRACE_SECONDS,
                )

        for session_id in session_ids:
            task = self._prompt_tasks.get(session_id)
            if task is None:
                continue
            try:
                # Bounded, but wide enough to comfortably outlast
                # `ChatSessionBackend.prompt`'s own worst-case cancelled-turn
                # timeline (its 5s bounded queue wait, MIN-1's settle-wait after
                # that) -- a *tighter* bound here than that would force-cancel the
                # ACP task while `prompt()` is still legitimately winding down on
                # its own, which delivers a second, overlapping cancellation into
                # the same `run_chat_task` and can leave it stuck mid-cleanup
                # (observed while tightening this bound during review: a stray
                # `chat_task._tasks` entry never got popped).
                await asyncio.wait_for(asyncio.shield(task), timeout=_PROMPT_TASK_GRACE_SECONDS)
            except Exception:
                if not task.done():
                    task.cancel()
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.shield(task), timeout=_CANCEL_SETTLE_GRACE_SECONDS
                    )
                # Re-assert: force-cancelling the ACP-side task only cancels
                # whatever it's currently awaiting -- ask the backend again so the
                # underlying `run_chat_task` is not left running regardless of
                # exactly where that cancellation landed.
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(
                        asyncio.shield(self.backend.cancel(session_id)),
                        timeout=_CANCEL_SETTLE_GRACE_SECONDS,
                    )
        self._prompt_tasks.clear()

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
                raise _JsonRpcError(_ERR_SESSION_LIMIT_REACHED, "session limit reached")
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

        # `session/prompt` is dispatched in `_handle_message`/`_dispatch_prompt`
        # before it ever reaches here (it runs on a background task so the serve
        # loop stays live for `session/cancel`); it is never routed through this
        # method.

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


# ── Translation: internal `run_chat_task` output_queue items → ACP updates ──
#
# Pure, unit-testable mirror of the client-side parsers in `cptr/utils/agents/acp.py`
# (`acp_text_from_update`/`acp_tool_from_update`): those consume `session/update`
# `update` objects, these produce them. `ChatSessionBackend.prompt` calls
# `acp_updates_from_queue_item` for every item pulled off the `output_queue` that
# `run_chat_task.emit()` feeds (see `cptr/utils/chat_task.py`), same shape the
# OpenAI-compat gateway consumes in `cptr/routers/gateway.py:_stream`/`_collect`.

_TOOL_CALL_STATUSES = {"pending", "in_progress", "completed", "failed"}
_PENDING_TOOL_CALL_STATUSES = {"pending", "in_progress"}


def acp_updates_from_queue_item(
    item: dict[str, Any], *, call_meta: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Map one `output_queue` item (as pushed by `run_chat_task.emit()`) onto zero or
    more ACP `session/update` `update` objects.

    `call_meta`, when given, is a caller-owned `call_id -> {"name", "arguments"}` dict
    that this function both reads and writes across the calls for one turn (see
    `ChatSessionBackend.prompt`): recorded from each `function_call` item so the later
    `function_call_output` update for the same `call_id` (which carries no name/
    arguments of its own) can carry them too (MIN-3) instead of leaving the client
    parser (`acp_tool_from_update` in `cptr/utils/agents/acp.py`) to re-derive a
    generic "agent_tool" name on that update and clobber whatever the call was really
    named. Omit it (the default) for the pure, single-item unit tests below.
    """
    item_type = item.get("type")

    if item_type == "delta":
        text = item.get("content")
        if not isinstance(text, str) or not text:
            return []
        return [{"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": text}}]

    if item_type == "output":
        inner = item.get("item")
        if not isinstance(inner, dict):
            return []
        return _acp_updates_from_output_item(inner, call_meta)

    # "done"/"error" are turn-lifecycle signals consumed directly by
    # `ChatSessionBackend.prompt`, never translated into a `session/update`.
    return []


def _acp_updates_from_output_item(
    inner: dict[str, Any], call_meta: dict[str, dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    inner_type = inner.get("type")

    if inner_type == "function_call":
        call_id = inner.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return []
        status = _acp_tool_status(inner.get("status"))
        session_update = (
            "tool_call" if status in _PENDING_TOOL_CALL_STATUSES else "tool_call_update"
        )
        if call_meta is not None:
            call_meta[call_id] = {"name": inner.get("name"), "arguments": inner.get("arguments")}
        return [
            {
                "sessionUpdate": session_update,
                "toolCallId": call_id,
                "title": inner.get("name"),
                "status": status,
                "rawInput": inner.get("arguments"),
            }
        ]

    if inner_type == "function_call_output":
        call_id = inner.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return []
        output = inner.get("output")
        if isinstance(output, str):
            output_text = output
        elif output is None:
            output_text = ""
        else:
            # MIN-10: match the client's own `rawOutput` convention
            # (`_tool_output` in `cptr/utils/agents/acp.py`) instead of `str()`,
            # which produces Python repr syntax (single quotes, `None`/`True`) that
            # isn't valid JSON and reads oddly to a user.
            output_text = json.dumps(output, indent=2)
        update: dict[str, Any] = {
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "content": [{"type": "content", "content": {"type": "text", "text": output_text}}],
            # MIN-3: the output arriving means this call is done -- say so
            # explicitly. Without it, the client's last-write-wins status parsing
            # (`acp_tool_from_update`: no `status` key -> "in_progress") resets an
            # already-completed tool call back to in-progress.
            "status": "completed",
        }
        meta = call_meta.get(call_id) if call_meta is not None else None
        if meta:
            # MIN-3: reproduce the same title/rawInput the `function_call` update
            # carried, so `acp_tool_from_update` re-derives the identical name
            # instead of falling back to a generic one on this update.
            update["title"] = meta.get("name")
            update["rawInput"] = meta.get("arguments")
        return [update]

    if inner_type == "message":
        # Its text already streamed as "delta" items -- emitting it again here would
        # double-send the same text, exactly what the gateway's `_stream`/`_collect`
        # avoid by only handling "delta"/"output"-with-tool-call-content and treating
        # everything else as DB-only.
        return []

    if inner_type == "reasoning":
        text = _reasoning_item_text(inner)
        if not text:
            return []
        return [{"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": text}}]

    return []


def _acp_tool_status(value: Any) -> str:
    normalized = str(value or "").strip().lower()
    return normalized if normalized in _TOOL_CALL_STATUSES else "in_progress"


def _reasoning_item_text(item: dict[str, Any]) -> str:
    """Extract displayable text from a "reasoning" output item (see `chat_task.py` /
    `_reasoning_output_item` in `cptr/utils/ai.py`): `content` is a list of blocks,
    each `{"type": "reasoning_text"|"text"|"output_text", "text": ...}`.
    """
    blocks = item.get("content")
    if not isinstance(blocks, list):
        return ""
    return "".join(
        block.get("text") or ""
        for block in blocks
        if isinstance(block, dict)
        and block.get("type") in ("reasoning_text", "text", "output_text")
    )


class ChatSessionBackend:
    """Maps ACP sessions onto `Chat` rows scoped to one authenticated user.

    One instance per WebSocket connection (see `cptr/routers/acp.py`) -- so two
    connections (even for the same user, e.g. a second client `session/load`-ing the
    same chat) each get their own `ChatSessionBackend`. `_inflight_chats`/
    `_claim_lock` below are declared at class level specifically so that claim is
    process-wide (MJ-2), shared by every instance, instead of only stopping a second
    concurrent prompt on the *same* connection (which `_prompt_tasks` in
    `AcpServerConnection` already did).
    """

    # MJ-2: process-wide "one turn per chat" claim, shared by every
    # `ChatSessionBackend` instance (i.e. every connection) in this process --
    # `_prompt_tasks` in `AcpServerConnection` only ever protected against a second
    # concurrent `session/prompt` on the *same* connection; a second connection that
    # `session/load`s the same chat bypassed it entirely.  `chat.id == session.id`
    # (see module docstring), so this is keyed on session id directly.
    _inflight_chats: set[str] = set()
    _claim_lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        # session_id -> assistant ChatMessage.id for the turn currently running, so
        # `cancel()` can find the real background task to stop. `None` between the
        # moment `prompt()` claims the turn and the moment `start_task` actually
        # creates the assistant message (BL-1): a session being a *key* in this dict
        # at all is what "this connection has a claimed turn for it" means, even
        # before there's a real message id to cancel yet.
        self._running: dict[str, str | None] = {}
        # session_id -> True once `cancel()` has been called for it, consulted by
        # the queue-consumption loop in `prompt()` to know whether a "done"/"error"
        # item (or a consumer-side timeout) means "cancelled" rather than a normal
        # end-of-turn/failure.
        self._cancelled: dict[str, bool] = {}

    async def new_session(self, cwd: str, params: dict[str, Any]) -> str:
        from cptr.models import Chat
        from cptr.utils.config import now_ms
        from cptr.utils.chat_export import chat_directory
        from cptr.utils.workspace import ensure_cptr_gitignored

        chat = await Chat.create(
            user_id=self.user_id,
            title="ACP session",
            meta={
                "workspace": cwd,
                # step 4: permission bridge makes ask/auto work over ACP -- until
                # then, "full" is the only mode that can complete a turn without a
                # `session/request_permission` round trip nothing here answers yet.
                "params": {"tool_approval_mode": "full"},
                "acp": {"created_via": "acp"},
            },
            created_at=now_ms(),
        )
        # Mirror routers/chat.py:send_message's chats-dir + gitignore setup for a
        # brand new workspace chat, now that `prompt()` actually persists messages
        # (and `run_chat_task`'s export pass) into that directory.
        chats_dir = chat_directory(cwd)
        await asyncio.to_thread(lambda: chats_dir.mkdir(parents=True, exist_ok=True))
        await asyncio.to_thread(ensure_cptr_gitignored, cwd)
        return chat.id

    async def load_session(self, session_id: str, cwd: str) -> bool:
        """Resume an existing session: True iff `session_id` names a `Chat` owned by
        this connection's user AND (when the chat recorded a workspace at
        `session/new` time) the requested `cwd` matches it exactly. Returning False
        for either failure is intentional and lets the caller answer with the same
        `-32001 session not found` either way -- ownership and workspace mismatches
        must be indistinguishable from a nonexistent session.

        This is the full extent of "load" today: no resume-state mapping (message
        replay) happens here yet -- that lands in a later step.
        """
        from cptr.models import Chat

        chat = await Chat.get_by_id(session_id)
        if chat is None or chat.user_id != self.user_id:
            return False
        workspace = (chat.meta or {}).get("workspace")
        if workspace and workspace != cwd:
            return False
        return True

    async def prompt(
        self, session_id: str, prompt_items: list[dict[str, Any]], notify: NotifyFn
    ) -> dict[str, Any]:
        """Run one real `run_chat_task` turn and stream it as ACP `session/update`s.

        Mirrors `routers/chat.py:send_message`'s message-row creation pattern and
        `routers/gateway.py`'s `output_queue` consumption, translating each queued
        item through `acp_updates_from_queue_item` instead of OpenAI SSE chunks.

        Claims `session_id` process-wide (MJ-2) and in `_running` (BL-1) atomically,
        before any `await` -- both a concurrent `prompt()` for the same chat on
        another connection and a `cancel()` racing this call's own setup (chat
        fetch, model resolution, message creation) now have something to act on
        immediately, instead of the multi-await gap where both used to be lost.
        """
        from cptr.models import Chat, ChatMessage
        from cptr.utils.chat_task import cancel_task, get_pending_input_lock, start_task
        from cptr.utils.config import now_ms
        from cptr.utils.model_targets import first_api_model_target, resolve_model_target

        async with ChatSessionBackend._claim_lock:
            if session_id in ChatSessionBackend._inflight_chats:
                raise PromptBusy("prompt already in progress")
            ChatSessionBackend._inflight_chats.add(session_id)
        self._running[session_id] = None
        # MJ-1: a `cancel()` for a *previous* turn on this session that arrived
        # after that turn already finished must never poison this new one. Safe to
        # clear unconditionally right here: the claim above (plus the post-
        # `start_task` recheck below) means any cancel racing *this* turn can only
        # be observed from this point on.
        self._cancelled.pop(session_id, None)

        assistant_msg = None
        normal_exit = False
        try:
            chat = await Chat.get_by_id(session_id)
            if chat is None or chat.user_id != self.user_id:
                # Defense in depth: the connection's authorized-session set already
                # gates this, but a session could in principle vanish/change owner
                # between `session/new`/`session/load` and this call.
                raise PromptRejected("session not found")

            # images: later -- only "text" prompt items are used for this turn.
            text = "\n\n".join(
                item.get("text", "")
                for item in prompt_items
                if isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ).strip()
            if not text:
                raise PromptRejected("empty prompt")

            meta = dict(chat.meta or {})
            workspace = meta.get("workspace") or ""
            last_model = meta.get("last_model")

            target = None
            if isinstance(last_model, str) and last_model:
                try:
                    target = await resolve_model_target(last_model)
                except Exception:
                    target = None
            if target is None:
                try:
                    target = await first_api_model_target()
                except Exception as exc:
                    raise PromptRejected("no model configured") from exc

            if meta.get("last_model") != target.full_model_id:
                meta["last_model"] = target.full_model_id
                await Chat.update_meta(session_id, meta, now_ms())

            # MJ-2: hold the same per-chat lock `routers/chat.py:send_message` uses
            # (see its `get_pending_input_lock` usage around line 780) around the
            # message-row creation + current-message update, so an ACP turn can
            # never race the web UI's own `send_message` writing this chat's
            # message tree at the same time.
            async with get_pending_input_lock(session_id):
                user_msg = await ChatMessage.create(
                    chat_id=session_id,
                    role="user",
                    content=text,
                    parent_id=chat.current_message_id,
                    created_at=now_ms(),
                )
                assistant_msg = await ChatMessage.create(
                    chat_id=session_id,
                    role="assistant",
                    content="",
                    parent_id=user_msg.id,
                    model=target.full_model_id,
                    done=False,
                    created_at=now_ms(),
                )
                await Chat.update_current_message(session_id, assistant_msg.id, now_ms())

            output_queue: asyncio.Queue = asyncio.Queue()
            start_task(
                message_id=assistant_msg.id,
                chat_id=session_id,
                user_id=self.user_id,
                workspace=workspace,
                output_queue=output_queue,
                target=target,
            )
            self._running[session_id] = assistant_msg.id
            if self._cancelled.get(session_id, False):
                # BL-1: a `cancel()` arrived between the claim at the top of this
                # method and here (mid-setup) -- it had nothing to act on yet, so
                # stop the turn we just started right now instead of letting the
                # rest of this function run it to completion while telling the
                # client it was cancelled.
                await cancel_task(assistant_msg.id)

            call_meta: dict[str, dict[str, Any]] = {}
            while True:
                cancelled = self._cancelled.get(session_id, False)
                try:
                    if cancelled:
                        # A cancelled `run_chat_task` may die without ever pushing a
                        # "done"/"error" item (e.g. cancelled mid-`finally`); don't
                        # let that hang this turn's response forever.
                        item = await asyncio.wait_for(
                            output_queue.get(), timeout=_CANCEL_QUEUE_GRACE_SECONDS
                        )
                    else:
                        item = await output_queue.get()
                except asyncio.TimeoutError:
                    # BL-2/MIN-1: the turn never got here; make sure it's actually
                    # being stopped (belt and braces -- `cancel()` already called
                    # `cancel_task` once, but this is cheap and safe to repeat) and
                    # give it a brief, bounded chance to actually finish writing
                    # before we release this chat's claim in `finally`.
                    with suppress(Exception):
                        await cancel_task(assistant_msg.id)
                    await self._await_task_settled(assistant_msg.id)
                    return {"stopReason": "cancelled"}

                if item is None:
                    # `run_chat_task`'s unconditional end-of-task sentinel (see its
                    # `finally`); only reached if no "done"/"error" arrived first --
                    # not a normal exit, so `finally` below still re-asserts
                    # `cancel_task` as a safety net.
                    was_cancelled = self._cancelled.pop(session_id, False)
                    if was_cancelled:
                        await self._await_task_settled(assistant_msg.id)
                    return {"stopReason": "cancelled" if was_cancelled else "end_turn"}

                item_type = item.get("type")
                if item_type == "done":
                    # A normal exit: `run_chat_task` already persisted its final DB
                    # state before pushing this (see its `_emit_done`/`_save_message`
                    # ordering), and its own task is finishing on its own -- no
                    # `cancel_task` needed in `finally`.
                    normal_exit = True
                    return {
                        "stopReason": "cancelled"
                        if self._cancelled.pop(session_id, False)
                        else "end_turn"
                    }
                if item_type == "error":
                    # Also a normal exit: the task already reached its own terminal
                    # error-handling path (persisted + emitted) before this arrived.
                    normal_exit = True
                    if self._cancelled.get(session_id, False):
                        return {"stopReason": "cancelled"}
                    # Detail stays server-side (logged); the connection layer turns
                    # this into a fixed `-32603 internal error` for the client.
                    raise RuntimeError(str(item.get("message", "chat task error")))

                for update in acp_updates_from_queue_item(item, call_meta=call_meta):
                    await notify(update)
        finally:
            if not normal_exit and assistant_msg is not None:
                # BL-2: every abnormal exit from the block above (an exception --
                # including `CancelledError` propagating from a dead `notify()` or
                # a force-cancelled ACP task -- or the bounded-wait timeout path)
                # must stop the underlying `run_chat_task`, not just this method's
                # own bookkeeping. `cancel_task` on an already-finished task is a
                # harmless no-op, so this is safe to call redundantly.
                with suppress(Exception):
                    await cancel_task(assistant_msg.id)
            self._running.pop(session_id, None)
            self._cancelled.pop(session_id, None)
            ChatSessionBackend._inflight_chats.discard(session_id)

    async def _await_task_settled(
        self, message_id: str, timeout: float = _CANCEL_SETTLE_GRACE_SECONDS
    ) -> None:
        """Best-effort: after deciding a turn is "cancelled", wait briefly for
        `chat_task._tasks` to actually drop `message_id` before returning (MIN-1).
        The underlying `run_chat_task` may still be mid-write when the bounded-wait
        timeout fires (or when the unconditional `None` sentinel is all we ever saw,
        e.g. cancelled mid-`finally`); returning immediately risks releasing this
        chat's process-wide claim (`_inflight_chats`) while that write is still in
        flight, letting an immediate follow-up prompt race it. Never raises, never
        blocks past `timeout` -- purely best-effort.
        """
        from cptr.utils.chat_task import is_running

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while is_running(message_id) and loop.time() < deadline:
            await asyncio.sleep(0.05)

    async def cancel(self, session_id: str) -> None:
        """Cancel the in-flight turn for `session_id`, if this connection's backend
        actually has one claimed (MJ-1: a `cancel()` for a session with nothing
        claimed -- already finished, or never started -- is a no-op beyond this
        bookkeeping; it must never arm `_cancelled` for a *future* turn to
        mistakenly observe).
        """
        from cptr.utils.chat_task import cancel_task

        if session_id not in self._running:
            return
        self._cancelled[session_id] = True
        message_id = self._running.get(session_id)
        if message_id:
            await cancel_task(message_id)
