"""Scripted ACP agent used as a real subprocess in adapter smoke tests.

Speaks just enough of the Agent Client Protocol to drive the production adapters
(`run_cursor_agent`, `run_grok_agent`) end to end: initialize/authenticate/session/new,
then on session/prompt streams text chunks and a tool call, optionally raises a
server-initiated ``session/request_permission`` (with a *string* JSON-RPC id), and
finally answers the prompt with ``stopReason: end_turn``.

Configuration via environment (adapters copy ``os.environ``):
- ``FAKE_ACP_FRAMING``: ``ndjson`` (default) or ``content-length`` for outbound frames.
  Inbound is always parsed as NDJSON, which is what ``AcpClient`` writes.
- ``FAKE_ACP_PERMISSION``: when ``1``, request permission mid-turn and report the
  outcome as a text chunk (``perm:granted`` / ``perm:denied``).

Positional argv (e.g. the adapters' ``acp`` / ``agent stdio``) is ignored.
"""

from __future__ import annotations

import json
import os
import sys


def send(message: dict) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode()
    if os.environ.get("FAKE_ACP_FRAMING") == "content-length":
        sys.stdout.buffer.write(b"Content-Length: %d\r\n\r\n" % len(payload))
        sys.stdout.buffer.write(payload)
    else:
        sys.stdout.buffer.write(payload + b"\n")
    sys.stdout.buffer.flush()


def read_message() -> dict | None:
    line = sys.stdin.buffer.readline()
    if not line:
        return None
    line = line.strip()
    if not line:
        return {}
    return json.loads(line.decode())


def notify(method: str, params: dict) -> None:
    send({"jsonrpc": "2.0", "method": method, "params": params})


def session_update(session_id: str, update: dict) -> None:
    notify("session/update", {"sessionId": session_id, "update": update})


def handle_prompt(request_id, session_id: str) -> None:
    session_update(
        session_id,
        {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "Hello "}},
    )
    session_update(
        session_id,
        {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "world"}},
    )
    session_update(
        session_id,
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "tc-1",
            "title": "Run command",
            "status": "pending",
            "rawInput": {"command": "echo hi"},
        },
    )

    if os.environ.get("FAKE_ACP_PERMISSION") == "1":
        # String id on purpose: regression coverage for non-integer JSON-RPC ids.
        send(
            {
                "jsonrpc": "2.0",
                "id": "perm-1",
                "method": "session/request_permission",
                "params": {
                    "sessionId": session_id,
                    "toolCall": {"toolCallId": "tc-1"},
                    "options": [
                        {"kind": "allow_once", "optionId": "approve-once"},
                        {"kind": "reject_once", "optionId": "reject-once"},
                    ],
                },
            }
        )
        reply = read_message()
        outcome = ((reply or {}).get("result") or {}).get("outcome") or {}
        granted = outcome.get("outcome") == "selected" and outcome.get("optionId") in {
            "approve-once",
            "allow-always",
        }
        session_update(
            session_id,
            {
                "sessionUpdate": "agent_message_chunk",
                "content": {
                    "type": "text",
                    "text": " perm:granted" if granted else " perm:denied",
                },
            },
        )

    session_update(
        session_id,
        {
            "sessionUpdate": "tool_call_update",
            "toolCallId": "tc-1",
            "status": "completed",
            "rawOutput": "hi",
        },
    )
    send({"jsonrpc": "2.0", "id": request_id, "result": {"stopReason": "end_turn"}})


def main() -> None:
    session_id = "sess-fake-1"
    while True:
        message = read_message()
        if message is None:
            return
        if not message:
            continue
        method = message.get("method")
        request_id = message.get("id")
        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": 1,
                        "authMethods": [{"id": "cursor_login", "name": "Login"}],
                        "agentCapabilities": {"loadSession": True},
                    },
                }
            )
        elif method == "authenticate":
            send({"jsonrpc": "2.0", "id": request_id, "result": {}})
        elif method == "session/new":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "sessionId": session_id,
                        "configOptions": [{"id": "model-opt", "category": "model"}],
                    },
                }
            )
        elif method == "session/prompt":
            handle_prompt(request_id, session_id)
        elif method == "session/cancel":
            pass
        elif request_id is not None:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"method not found: {method}"},
                }
            )


if __name__ == "__main__":
    main()
