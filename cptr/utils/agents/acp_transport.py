"""Transport abstraction for ACP JSON-RPC clients.

`AcpClient` (see `cptr/utils/agents/acp.py`) originally spoke JSON-RPC directly over a
stdio subprocess. This module extracts that I/O boundary into an `AcpTransport`
protocol so the protocol logic in `AcpClient` can be exercised without spawning a real
process: `StdioSubprocessTransport` reproduces the original subprocess behavior exactly,
while `InMemoryTransport` (via `connected_pair()`) lets tests wire a real `AcpClient`
against a scripted "fake agent" coroutine using nothing but `asyncio.Queue`s.

Framing behavior (`extract_json_message`) is moved here byte-for-byte from the old
`acp.py::_extract_json_message` and must keep its quirks: leading whitespace is
stripped (via `bytes.lstrip()`, which also treats embedded `\n`/`\r` as whitespace)
before sniffing the frame kind, and a `Content-Length` header present but missing a
valid integer value raises `RuntimeError` rather than returning `None`.

The code has an `if not line: return {}, ...` branch that looks like it is meant to turn
a "blank NDJSON line" into an empty-dict message. In practice this branch is unreachable
through the public function: the unconditional `buffer.lstrip()` at the top of the
function always consumes a full run of leading whitespace, and `\n` is itself
whitespace, so any blank line sitting at the front of `buffer` is absorbed by that
lstrip before the "first line" scan ever runs (post-lstrip, `buffer[0]` is always
non-whitespace unless the buffer is now empty). A stray blank line between two NDJSON
messages therefore never surfaces as a distinct `{}` message on its own extraction pass
-- it is silently skipped, and the *next* real JSON message is returned directly.
`tests/test_acp_framing.py` pins this exact (if slightly surprising) behavior instead of
asserting the unreachable branch. The dead branch itself is left in place verbatim since
removing it is outside the scope of this refactor, and downstream code (the reader
loops in this module) still forwards empty-dict messages unchanged in the unlikely event
some other code path ever produces one.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from typing import Any, Protocol


class TransportClosed(Exception):
    """Raised by `AcpTransport.receive()` when the peer/stream has closed."""


class AcpTransport(Protocol):
    async def send(self, message: dict[str, Any]) -> None: ...

    async def receive(self) -> dict[str, Any]:
        """Return one parsed JSON-RPC message; raise `TransportClosed` on EOF/close."""
        ...

    async def start(self) -> None: ...

    async def close(self) -> None: ...


def extract_json_message(buffer: bytes) -> tuple[dict[str, Any], bytes] | None:
    stripped = buffer.lstrip()
    skipped = len(buffer) - len(stripped)
    if skipped:
        buffer = stripped

    lower = buffer[:32].lower()
    if lower.startswith(b"content-length:"):
        header_end = buffer.find(b"\r\n\r\n")
        sep_len = 4
        if header_end < 0:
            header_end = buffer.find(b"\n\n")
            sep_len = 2
        if header_end < 0:
            return None
        header = buffer[:header_end].decode(errors="replace")
        length = None
        for line in header.splitlines():
            if line.lower().startswith("content-length:"):
                with suppress(ValueError):
                    length = int(line.split(":", 1)[1].strip())
        if length is None:
            raise RuntimeError("ACP message missing Content-Length")
        start = header_end + sep_len
        end = start + length
        if len(buffer) < end:
            return None
        return json.loads(buffer[start:end].decode()), buffer[end:]

    line_end = buffer.find(b"\n")
    if line_end < 0:
        return None
    line = buffer[:line_end].strip()
    if not line:
        return {}, buffer[line_end + 1 :]
    return json.loads(line.decode()), buffer[line_end + 1 :]


_CLOSED_SENTINEL = object()


class StdioSubprocessTransport:
    """ACP transport backed by a stdio subprocess (the original `AcpClient` behavior)."""

    def __init__(self, command: str, args: list[str], cwd: str, env: dict[str, str]) -> None:
        self.command = command
        self.args = args
        self.cwd = cwd
        self.env = env
        self.proc: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task | None = None
        self._stderr_task: asyncio.Task | None = None
        self._queue: asyncio.Queue[Any] = asyncio.Queue()

    async def start(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd or os.getcwd(),
            env=self.env,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        self._stderr_task = asyncio.create_task(self._stderr_loop())

    async def send(self, message: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        data = json.dumps(message, separators=(",", ":")).encode() + b"\n"
        self.proc.stdin.write(data)
        await self.proc.stdin.drain()

    async def receive(self) -> dict[str, Any]:
        item = await self._queue.get()
        if item is _CLOSED_SENTINEL:
            # Keep raising for any further callers too.
            self._queue.put_nowait(_CLOSED_SENTINEL)
            raise TransportClosed("ACP subprocess stdout closed")
        return item

    async def close(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.proc.wait(), timeout=3)
            if self.proc.returncode is None:
                self.proc.kill()
                await self.proc.wait()
        for task in (self._reader_task, self._stderr_task):
            if task:
                task.cancel()
                # `.cancel()` only requests cancellation; awaiting the task can still
                # surface whatever exception it was in the middle of raising (e.g. a
                # `ValueError` from an oversized `readline()`), not just
                # `CancelledError`. `close()` must complete regardless so the caller's
                # pending-future cleanup always runs.
                with suppress(BaseException):
                    await task

    async def _reader_loop(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        buffer = b""
        try:
            while True:
                chunk = await self.proc.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    extracted = extract_json_message(buffer)
                    if extracted is None:
                        break
                    message, buffer = extracted
                    await self._queue.put(message)
        finally:
            # Deliver the EOF sentinel even if the loop above dies on an exception
            # (e.g. malformed JSON from the agent), so `receive()` never hangs.
            await self._queue.put(_CLOSED_SENTINEL)

    async def _stderr_loop(self) -> None:
        assert self.proc is not None and self.proc.stderr is not None
        # Drain with bounded reads rather than `readline()`: a single stderr line
        # longer than the stream's line-length limit makes `readline()` raise
        # `ValueError`, which used to propagate out of `close()` and skip
        # `AcpClient`'s pending-future cleanup.
        while await self.proc.stderr.read(65536):
            pass


class InMemoryTransport:
    """In-memory `AcpTransport` for tests: cross-wire two via `connected_pair()`."""

    def __init__(
        self,
        outgoing: asyncio.Queue[Any],
        incoming: asyncio.Queue[Any],
        closed: list[bool],
    ) -> None:
        self._outgoing = outgoing
        self._incoming = incoming
        # Shared with the peer transport created by the same `connected_pair()` call,
        # so closing either side is immediately visible to `send()` on both sides.
        self._closed = closed

    async def start(self) -> None:
        pass

    async def send(self, message: dict[str, Any]) -> None:
        if self._closed[0]:
            raise ConnectionResetError("transport is closed")
        await self._outgoing.put(message)

    async def receive(self) -> dict[str, Any]:
        item = await self._incoming.get()
        if item is _CLOSED_SENTINEL:
            self._incoming.put_nowait(_CLOSED_SENTINEL)
            raise TransportClosed("peer closed the transport")
        return item

    async def close(self) -> None:
        if self._closed[0]:
            return
        self._closed[0] = True
        # Unblock a peer parked in `receive()` *and* our own side, in case we're the
        # one with a pending `receive()` call: put the sentinel on both queues.
        await self._outgoing.put(_CLOSED_SENTINEL)
        await self._incoming.put(_CLOSED_SENTINEL)


def connected_pair() -> tuple[InMemoryTransport, InMemoryTransport]:
    queue_a: asyncio.Queue[Any] = asyncio.Queue()
    queue_b: asyncio.Queue[Any] = asyncio.Queue()
    closed = [False]
    side_a = InMemoryTransport(outgoing=queue_a, incoming=queue_b, closed=closed)
    side_b = InMemoryTransport(outgoing=queue_b, incoming=queue_a, closed=closed)
    return side_a, side_b
