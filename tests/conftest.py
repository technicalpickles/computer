"""Suite-wide pytest fixtures.

Currently just one autouse guard: nothing in this suite should ever read or write
the real `~/.cptr` (the default `DATA_DIR`, see `cptr/env.py`). Tests that exercise
DB-backed code (`run_chat_task`, `Chat`/`ChatMessage`, `Config`, ...) are expected to
monkeypatch `cptr.utils.db.DATA_DIR`/`DB_FILE` and `cptr.utils.config.DATA_DIR`/
`CONFIG_FILE` onto a tmp dir (see `isolated_db` fixtures in `test_acp_ws_endpoint.py`
and `test_acp_prompt_e2e.py`), and workspace-scoped paths (`.cptr/chats`, skills,
memory, ...) are always relative to whatever tmp workspace dir a test passes as
`cwd`/`workspace` -- never the real filesystem.

This fixture doesn't stop a rogue test from doing the wrong thing; it *notices*
after the fact, so a regression here fails loudly in CI instead of silently
littering (or corrupting) the machine running the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _real_home_cptr_dir() -> Path:
    return Path.home() / ".cptr"


def _snapshot(path: Path) -> set[str]:
    """Relative paths of every file/dir under `path` (empty if it doesn't exist)."""
    if not path.exists():
        return set()
    entries: set[str] = set()
    for root, dirs, files in os.walk(path):
        rel_root = Path(root).relative_to(path)
        for name in dirs:
            entries.add(str(rel_root / name) + "/")
        for name in files:
            entries.add(str(rel_root / name))
    return entries


@pytest.fixture(scope="session", autouse=True)
def guard_real_home_cptr_dir_untouched():
    home_cptr = _real_home_cptr_dir()
    existed_before = home_cptr.exists()
    before = _snapshot(home_cptr)

    yield

    if not existed_before:
        assert not home_cptr.exists(), (
            f"Test suite created {home_cptr} -- some test wrote real user data "
            "instead of using an isolated tmp DATA_DIR/workspace. Every test that "
            "touches DB-backed or filesystem-backed cptr code must monkeypatch "
            "DATA_DIR/CONFIG_FILE/DB_FILE (see isolated_db fixtures) and pass a "
            "tmp dir as cwd/workspace."
        )
        return

    after = _snapshot(home_cptr)
    added = after - before
    removed = before - after
    assert not added and not removed, (
        f"Test suite modified {home_cptr}: added={sorted(added)} removed={sorted(removed)}"
    )
