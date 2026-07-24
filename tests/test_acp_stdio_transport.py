"""Real-subprocess coverage for `StdioSubprocessTransport`.

Every other ACP test drives `AcpClient` (or raw framing) purely in-memory, so
`StdioSubprocessTransport` -- the transport that actually spawns and talks to a real
agent process -- had zero coverage before this file. These tests spawn trivial
`sys.executable -c "..."` "echo agents" that speak the real NDJSON stdio protocol, so
they exercise process spawn/pipe wiring, line framing (including the `.strip()` on
each extracted line), stderr draining, EOF handling, and `close()`'s
terminate-then-reap and background-task-cleanup logic -- none of which
`InMemoryTransport` touches at all.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

from cptr.utils.agents.acp_transport import StdioSubprocessTransport, TransportClosed

pytestmark = pytest.mark.asyncio


def _transport(script: str) -> StdioSubprocessTransport:
    return StdioSubprocessTransport(
        command=sys.executable,
        args=["-u", "-c", script],
        cwd=os.getcwd(),
        env=dict(os.environ),
    )


# Reads NDJSON requests off stdin and echoes each back as a JSON-RPC result, padded
# with trailing spaces before the newline -- this specifically exercises the
# `.strip()` call in `extract_json_message` on the extracted line.
ECHO_AGENT = """
import json
import sys

for raw_line in sys.stdin:
    raw_line = raw_line.strip()
    if not raw_line:
        continue
    message = json.loads(raw_line)
    response = {
        "jsonrpc": "2.0",
        "id": message["id"],
        "result": {"echo": message.get("params")},
    }
    sys.stdout.write(json.dumps(response) + "   \\n")
    sys.stdout.flush()
"""

# Replies once with a CRLF-terminated line instead of a bare "\\n" (Windows-agent
# fidelity), then exits.
CRLF_AGENT = """
import json
import sys

raw_line = sys.stdin.readline().strip()
message = json.loads(raw_line)
response = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {}})
sys.stdout.buffer.write(response.encode() + b"\\r\\n")
sys.stdout.buffer.flush()
"""

# Exits immediately without reading or writing anything.
EXITS_IMMEDIATELY_AGENT = "pass"

# Loops forever without ever exiting on its own; used to prove close() actually
# terminates a still-running child instead of just cancelling our own tasks.
LONG_RUNNING_AGENT = """
import time

while True:
    time.sleep(0.05)
"""

# Writes a normal-sized line to stderr and then replies on stdout, so we can confirm
# stderr traffic doesn't block or corrupt the stdout response path.
STDERR_AND_ECHO_AGENT = """
import json
import sys

sys.stderr.write("warning: something happened\\n")
sys.stderr.flush()
raw_line = sys.stdin.readline().strip()
message = json.loads(raw_line)
response = json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": {}})
sys.stdout.write(response + "\\n")
sys.stdout.flush()
"""

# Writes a single stderr line well over 64KiB with no embedded newline, which makes
# asyncio's default StreamReader.readline() raise ValueError once its internal
# 64KiB limit is exceeded before a separator is found.
OVERSIZED_STDERR_AGENT = """
import sys

sys.stderr.write("x" * 200000 + "\\n")
sys.stderr.flush()
"""

# Writes a line that isn't valid JSON at all, so the reader loop's extraction dies
# on a JSONDecodeError instead of a clean EOF.
INVALID_JSON_AGENT = """
import sys

sys.stdout.write("not valid json\\n")
sys.stdout.flush()
"""


async def test_round_trip_send_and_receive():
    transport = _transport(ECHO_AGENT)
    await transport.start()
    try:
        await transport.send({"jsonrpc": "2.0", "id": 1, "method": "ping", "params": {"x": 1}})
        message = await asyncio.wait_for(transport.receive(), timeout=5)
        assert message == {"jsonrpc": "2.0", "id": 1, "result": {"echo": {"x": 1}}}
    finally:
        await asyncio.wait_for(transport.close(), timeout=5)


async def test_crlf_terminated_response_line_is_parsed():
    transport = _transport(CRLF_AGENT)
    await transport.start()
    try:
        await transport.send({"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}})
        message = await asyncio.wait_for(transport.receive(), timeout=5)
        assert message == {"jsonrpc": "2.0", "id": 2, "result": {}}
    finally:
        await asyncio.wait_for(transport.close(), timeout=5)


async def test_agent_exit_raises_transport_closed_and_keeps_raising():
    transport = _transport(EXITS_IMMEDIATELY_AGENT)
    await transport.start()
    try:
        with pytest.raises(TransportClosed):
            await asyncio.wait_for(transport.receive(), timeout=5)
        # The sentinel is re-queued so subsequent callers also observe closure
        # instead of hanging forever.
        with pytest.raises(TransportClosed):
            await asyncio.wait_for(transport.receive(), timeout=5)
    finally:
        await asyncio.wait_for(transport.close(), timeout=5)


async def test_close_terminates_still_running_subprocess():
    transport = _transport(LONG_RUNNING_AGENT)
    await transport.start()
    assert transport.proc is not None
    assert transport.proc.returncode is None

    await asyncio.wait_for(transport.close(), timeout=5)

    assert transport.proc.returncode is not None


async def test_stderr_output_does_not_block_stdout_and_close_is_clean():
    transport = _transport(STDERR_AND_ECHO_AGENT)
    await transport.start()
    try:
        await transport.send({"jsonrpc": "2.0", "id": 3, "method": "ping", "params": {}})
        message = await asyncio.wait_for(transport.receive(), timeout=5)
        assert message == {"jsonrpc": "2.0", "id": 3, "result": {}}
    finally:
        await asyncio.wait_for(transport.close(), timeout=5)


async def test_oversized_stderr_line_does_not_break_close():
    transport = _transport(OVERSIZED_STDERR_AGENT)
    await transport.start()
    # Give the child time to write and for our stderr drain loop to actually hit
    # the oversized line before we ask close() to tear things down.
    await asyncio.sleep(0.5)
    # Must complete without raising -- a bare `readline()`-based drain loop would
    # raise ValueError here, and an old close() would let that escape.
    await asyncio.wait_for(transport.close(), timeout=5)


async def test_malformed_json_from_agent_surfaces_as_transport_closed():
    transport = _transport(INVALID_JSON_AGENT)
    await transport.start()
    try:
        with pytest.raises(TransportClosed):
            await asyncio.wait_for(transport.receive(), timeout=5)
    finally:
        await asyncio.wait_for(transport.close(), timeout=5)
