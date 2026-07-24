# Validation Harness for the ACP Server (`cptr` as ACP agent)

Companion to the "Expose `cptr` as a Network-Reachable ACP Agent" proposal. That doc
sketches *what* to build; this one designs *how to build and validate it autonomously* —
no human at an editor, no real coding-agent CLI, and no real LLM API calls in the loop.

The harness is intended to exist **before** the feature: the scenario suite below is an
executable spec that an implementing agent (or human) codes against until green.

---

## 1. The core insight: two fakes, one real system under test

The system under test is the new ACP endpoint plus the existing `run_chat_task`
machinery it bridges into. Everything *outside* that boundary is one of two external
parties, and both are fully scriptable:

```
┌─────────────────┐   WS/JSON-RPC   ┌──────────────────────────────┐   HTTP/SSE   ┌──────────────────┐
│ FAKE ACP CLIENT │ ◄─────────────► │  REAL cptr                   │ ◄──────────► │ FAKE MODEL SERVER │
│ (scripted, in   │                 │  acp router → run_chat_task  │              │ (scripted OpenAI- │
│  pytest)        │                 │  → approval pause → tools    │              │  compatible SSE)  │
└─────────────────┘                 └──────────────────────────────┘              └──────────────────┘
```

**Fake ACP client (replaces Zed/JetBrains).** `cptr/utils/agents/acp.py` already
contains ~90% of a working ACP client — JSON-RPC request/response with
`pending: dict[id, Future]`, dual NDJSON/Content-Length framing
(`_extract_json_message`), and the client side of `session/request_permission`
(`_reply_permission`). It is coupled to a stdio subprocess only in `start()`/`_send()`/
`_reader_loop()`. Extracting a transport interface gives us a scriptable WebSocket ACP
client for free — and dogfoods the same protocol code the product already ships.

**Fake model backend (replaces Anthropic/OpenAI/Claude Code).** Two seams, both useful:

1. *HTTP seam (preferred for E2E, zero code changes needed):* connections come from DB
   config (`Config.get("chat.connections")`, see `cptr/routers/chat.py:_get_connections`),
   and a connection with `data.models` pre-set skips live model discovery. So the harness
   registers `http://127.0.0.1:<port>/v1` as a connection and runs a tiny scripted
   OpenAI-compatible server that replays canned SSE — including deterministic
   `tool_calls` for tools *not* on the auto-approve list, which is exactly what forces
   the `session/request_permission` path. All of `stream_openai_completions` /
   `stream_openai_responses` in `cptr/utils/ai.py` is exercised for real.
2. *In-process seam (for design (b) / passthrough mode):* a fake backend adapter that
   emits a scripted sequence of `AgentEvent`s (`cptr/utils/agents/events.py` — the union
   every real adapter already normalizes into). This tests the AgentEvent →
   `session/update` bridge without any subprocess.

Both fakes are deterministic: each scenario is a list of canned responses the fake pops
per request, and the fake also *records* what it received, so tests assert both
directions (e.g. that a tool result actually round-tripped back into the next model
call).

---

## 2. Test tiers

### Tier 0 — pure protocol unit tests (no network, no DB)

Design constraint on the feature itself: the ACP server handler must be written
transport-abstracted —

```python
class AcpTransport(Protocol):
    async def send(self, msg: dict) -> None: ...
    async def receive(self) -> dict: ...   # raises on close

class AcpServerConnection:
    def __init__(self, transport: AcpTransport, session_backend: SessionBackend): ...
```

with a WebSocket transport in production and an in-memory (`asyncio.Queue`-pair)
transport in tests. `SessionBackend` is the seam to `run_chat_task`/`Chat` — injectable,
so Tier 0 stubs it. This tier covers: initialize/version negotiation, method dispatch,
error objects for unknown methods/bad params, id bookkeeping (including **non-integer
JSON-RPC ids** — regression for upstream issue #108), permission request/reply
correlation, cancellation races.

The same refactor applies to `AcpClient` (stdio transport extracted) so client and
server share framing code — issue #84's NDJSON/Content-Length ambiguity gets one
canonical, tested implementation instead of two.

### Tier 1 — in-process app integration

A slim app factory (`create_test_app()`) mounting the ACP router + chat router with:
a temp SQLite DB (`init_db` against a tmpdir), auth configured for
localhost + a known token, and the heavy lifespan branches disabled (bot manager,
automation scheduler, timers, browser cleanup — `cptr/app.py` lifespan needs env flags
for these; `ENABLE_CHAT_RECONCILE_ON_STARTUP` shows the pattern already exists).
Client side: Starlette `TestClient.websocket_connect` for sync-style tests, or uvicorn
on an ephemeral port + `websockets`/`httpx-ws` for async ones. Model calls go to the
fake model server (httpx `MockTransport` where injectable, real localhost server where
not).

### Tier 2 — black-box E2E ("the autonomous build validation")

Real `cptr` process launched as a subprocess with a scratch `HOME`/config dir, fake
model server as a second local process, scripted ACP client connecting over a real
WebSocket. No network egress, fully CI-able, single `pytest -m e2e` entry point.

Assertions at this tier are **golden transcripts**: the harness records the complete
ordered JSON-RPC message sequence, normalizes volatile fields (ids, uuids, timestamps,
durations, absolute paths), and diffs against a checked-in expectation per scenario.
A failing diff is readable; updating expectations is `--update-golden`.

### Cross-cutting: schema conformance without an editor

Pin the ACP schema (`schema.json` from the `agentclientprotocol` repo, at the commit of
the draft transport RFD — PR #721 — the proposal says to pin anyway) and validate
**every outbound message** in every tier against it. This substitutes for "does Zed
accept it" to a first approximation, and localizes breakage when the pinned RFD is
bumped later.

---

## 3. Scenario matrix (the executable spec)

Lifecycle / auth:
- `initialize` handshake: capabilities echo, protocol version negotiation, `clientInfo`
- WS auth failures: missing/bad token → close `4001` (same `check_access` pattern as
  `routers/events.py:events_ws`); valid `cptr_session` cookie; valid `?token=` param
- `session/new` creates a Chat with the right workspace; `session/load` maps onto
  existing `Chat` + `meta.agent_sessions` resume state

Streaming turns:
- prompt → text-only answer → `agent_message_chunk` updates → `session/prompt` result
  with `stopReason: end_turn`
- prompt → auto-approved tool (`tool_approval_mode: full`, or auto-listed tool in
  `auto` mode) → `tool_call` / `tool_call_update` notifications, **no** permission
  request, output content mapped

The payoff path (what the OpenAI gateway structurally cannot do):
- `ask` mode → non-auto tool → server sends `session/request_permission` → client
  replies `allow_once` → tool executes → turn **continues in the same prompt** →
  completes. Assert the fake model server then received the tool result.
- reply `reject`/cancelled outcome → tool marked rejected, turn ends
- client never answers → what happens? (timeout? held forever?) — the harness forces
  this design decision to be made explicitly
- cross-channel consistency: permission pending over ACP *and* answered via the
  existing REST `POST /{chat}/messages/{id}/approve` — exactly one execution, the ACP
  waiter resolved, no double-run. (The REST path executes the tool then calls
  `start_task` again — `routers/chat.py:approve_tool` — so the bridge must not race it.)

Robustness:
- `session/cancel` mid-stream: partial output persisted, `stopReason: cancelled`
- WS disconnect mid-turn → task keeps running or pauses per design → reconnect +
  `session/load` → state consistent with what the REST API reports
- disconnect while a permission request is in flight → falls back to persisted
  pending-approval state (the existing `run_chat_task` behavior at the pause point)
- two concurrent sessions (one connection and two connections)
- framing: both NDJSON and Content-Length accepted inbound (issue #84 regression)
- malformed JSON-RPC, oversized messages, unknown session ids

Every scenario doubles as a golden transcript in Tier 2 and a direct-assertion test in
Tier 0/1.

---

## 4. How the harness constrains the feature design (build-order)

Because the harness comes first, it dictates testable structure:

1. **Transport abstraction** (`AcpTransport`, shared framing) + refactor `AcpClient`
   onto it. Validated by Tier 0 + existing Cursor/Grok paths still working.
2. **Server handshake + session mapping** (`initialize`, `session/new`/`load` onto
   `Chat`). Tier 0/1.
3. **Prompt bridging**: `session/prompt` → `run_chat_task` with an `output_queue`-style
   consumer (the gateway's queue in `run_chat_task.emit` is the precedent) → ACP
   `session/update` mapping (mirror of `acp_text_from_update`/`acp_tool_from_update`).
4. **Permission bridge**: a small standalone module — a registry of pending permission
   futures keyed by `(chat_id, message_id, call_id)` that the ACP connection awaits and
   that both the ACP reply handler and the REST `/approve` endpoint resolve. Small
   enough to unit-test exhaustively (timeout, double-resolve, disconnect cleanup).
5. **Cancel/resume/hardening** against the robustness scenarios.

Each step lands with its slice of the matrix green; Tier 2 golden transcripts flip from
`xfail` to passing as coverage grows — that's the autonomous progress signal.

## 5. Tooling / CI

- Add `pytest`, `pytest-asyncio` (and `websockets` for the E2E client) to the `dev`
  dependency group; the repo currently has **no test suite at all** (dev deps = ruff
  only), so this bootstraps `tests/` generally, not just for ACP.
- GitHub Actions job: `uv run pytest` — Tier 0/1 on every push, Tier 2 behind
  `-m e2e` but still in CI (all-local, no egress, seconds not minutes).
- Later, human-in-the-loop only for the last mile: a record/replay proxy (tee the
  stdio/WS traffic from one real Zed session once, replay it as a Tier 2 scenario
  forever) turns even editor validation into a regression test after a single manual
  capture.
