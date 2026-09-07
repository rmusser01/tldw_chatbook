"""PROBE (task-15860 Task 0, throwaway): prove which checkout is imported.

The venv's editable install resolves `tldw_chatbook` to a FOREIGN worktree
(`.worktrees/task-2512-mcp-unified`). Every probe in this arc is worthless
unless the code under test is THIS worktree's origin/dev checkout.
"""
from __future__ import annotations

from pathlib import Path

import tldw_chatbook


def test_probe_imports_this_worktree():
    here = Path(__file__).resolve().parents[1]
    imported = Path(tldw_chatbook.__file__).resolve()
    print(f"\nPROBE imported tldw_chatbook from: {imported}")
    print(f"PROBE worktree root: {here}")
    assert imported.is_relative_to(here), (
        f"imported {imported} is NOT under {here} -- the editable install's "
        "foreign-worktree finder won"
    )
