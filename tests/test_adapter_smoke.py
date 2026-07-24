"""Smoke tests: the production Cursor/Grok adapters against a real fake-agent subprocess.

Unlike the transport/client unit tests, these exercise the full, unmodified adapter
code paths (`run_cursor_agent`, `run_grok_agent`) end to end: real subprocess, real
pipes, real framing, AgentEvent stream out the top.
"""

from __future__ import annotations

import asyncio
import stat
import sys
from pathlib import Path

import pytest

from cptr.utils.agents.attachments import PreparedAgentAttachments
from cptr.utils.agents.cursor import run_cursor_agent
from cptr.utils.agents.events import AgentDone, AgentError, AgentTextDelta, AgentToolUpdate
from cptr.utils.agents.grok import run_grok_agent

FAKE_AGENT = Path(__file__).parent / "fake_acp_agent.py"


@pytest.fixture
def fake_agent_command(tmp_path: Path) -> str:
    """An executable wrapper for the fake agent that ignores adapter argv."""
    wrapper = tmp_path / "fake-agent"
    wrapper.write_text(
        f'#!{sys.executable}\nimport runpy\nrunpy.run_path("{FAKE_AGENT}", run_name="__main__")\n'
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return str(wrapper)


async def collect_events(adapter, command: str, workspace: str, chat_params: dict):
    events = []
    agen = adapter(
        profile={"id": "profile-1", "command": command},
        model="default",
        workspace=workspace,
        messages=[{"role": "user", "content": "hi"}],
        system_prompt="be brief",
        chat_params=chat_params,
        resume_state=None,
        attachments=PreparedAgentAttachments(images=[], files=[], prompt_suffix=""),
    )
    async for event in agen:
        events.append(event)
    return events


def assert_happy_path(events: list, expect_perm: str | None = None) -> None:
    errors = [e for e in events if isinstance(e, AgentError)]
    assert not errors, f"adapter surfaced errors: {errors}"

    text = "".join(e.text for e in events if isinstance(e, AgentTextDelta))
    assert text.startswith("Hello world")
    if expect_perm:
        assert f"perm:{expect_perm}" in text

    tool_updates = [e for e in events if isinstance(e, AgentToolUpdate)]
    # The initial tool_call carries rawInput.command and maps to run_command; the
    # completion update has no rawInput, so it maps to the generic agent_tool name.
    assert any(
        t.call_id == "tc-1" and t.name == "run_command" and t.status == "pending"
        for t in tool_updates
    )
    assert any(
        t.call_id == "tc-1" and t.status == "completed" and t.output == "hi" for t in tool_updates
    )

    done = [e for e in events if isinstance(e, AgentDone)]
    assert len(done) == 1
    assert done[0].resume_state["session_id"] == "sess-fake-1"


@pytest.mark.parametrize("adapter", [run_cursor_agent, run_grok_agent])
async def test_adapter_end_to_end_ndjson(adapter, fake_agent_command, tmp_path, monkeypatch):
    monkeypatch.delenv("FAKE_ACP_FRAMING", raising=False)
    monkeypatch.delenv("FAKE_ACP_PERMISSION", raising=False)
    events = await asyncio.wait_for(
        collect_events(adapter, fake_agent_command, str(tmp_path), {"tool_approval_mode": "full"}),
        timeout=15,
    )
    assert_happy_path(events)


async def test_cursor_adapter_content_length_framing(fake_agent_command, tmp_path, monkeypatch):
    """Issue #84 end to end: agent responds with Content-Length frames, adapter still works."""
    monkeypatch.setenv("FAKE_ACP_FRAMING", "content-length")
    events = await asyncio.wait_for(
        collect_events(
            run_cursor_agent, fake_agent_command, str(tmp_path), {"tool_approval_mode": "full"}
        ),
        timeout=15,
    )
    assert_happy_path(events)


async def test_cursor_adapter_permission_granted_when_full(
    fake_agent_command, tmp_path, monkeypatch
):
    """Server-initiated permission request (string JSON-RPC id) auto-approved in full mode."""
    monkeypatch.setenv("FAKE_ACP_PERMISSION", "1")
    events = await asyncio.wait_for(
        collect_events(
            run_cursor_agent, fake_agent_command, str(tmp_path), {"tool_approval_mode": "full"}
        ),
        timeout=15,
    )
    assert_happy_path(events, expect_perm="granted")


async def test_cursor_adapter_permission_denied_when_ask(fake_agent_command, tmp_path, monkeypatch):
    """Without auto-approve the client answers 'cancelled' and the turn still completes."""
    monkeypatch.setenv("FAKE_ACP_PERMISSION", "1")
    events = await asyncio.wait_for(
        collect_events(
            run_cursor_agent, fake_agent_command, str(tmp_path), {"tool_approval_mode": "ask"}
        ),
        timeout=15,
    )
    assert_happy_path(events, expect_perm="denied")
