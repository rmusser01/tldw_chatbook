"""A size ratchet for the largest ungoverned modules — stop them growing.

**Why this test exists.** The core-runtime code review (2026-09-17,
`qa/core-code-review-2026-09-17/report.md`) found that several of the
repository's largest modules had no size ratchet row at all, across
directories that neither `test_screen_size_ratchet.py` (screens) nor
`test_library_modules_size_ratchet.py` (`Library_Modules/*_controller.py`)
covers. Without a row, nothing stops them growing — the same silent creep
those two files were written to catch, on the modules that need it most.

**This is a ratchet, not a limit.** Each budget below is pinned at the
module's exact current line count (`len(path.read_text().splitlines())`,
the expression `_measure` uses). The numbers may only ever go DOWN. When
you shrink one of these files, lower its row to the new measurement in the
same commit. If you are here because CI failed, the fix is to put your new
code somewhere else — a controller, a widget, a helper module — never to
raise the number, which re-opens the hole this test exists to close.

**Scope.** These are hand-picked god modules, not a directory family, so
(unlike the Library controller ratchet) there is no glob that auto-adds new
files. `settings_screen.py` is deliberately absent: its split and its
missing ratchet row are already tracked by task-1378 / task-31202 — linked,
not duplicated here (core-review TASK-32809.2 AC#2).

First recorded 2026-09-19 by core-review TASK-32809.2, each row at its exact
measured size as of `origin/dev`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: path -> max line count. LOWER these when a module shrinks. Never raise
#: them to silence a failure — see the module docstring.
_BUDGETS: dict[str, int] = {
    "tldw_chatbook/app.py": 21415,
    "tldw_chatbook/Chat/console_chat_controller.py": 29367,
    "tldw_chatbook/Chat/console_chat_store.py": 22344,
    "tldw_chatbook/UI/Screens/personas_screen.py": 16436,
    "tldw_chatbook/Widgets/Console/console_transcript.py": 8353,
    "tldw_chatbook/Widgets/Console/console_settings_modal.py": 7807,
    "tldw_chatbook/UI/MCP_Modules/mcp_workbench.py": 6760,
}

#: Same tolerance as the Library controller ratchet: loose enough that
#: ordinary in-file edits do not fail CI, tight enough that a real shrink
#: which forgot to lower its row is still caught.
_SLACK_TOLERANCE_LINES = 50


@lru_cache(maxsize=None)
def _measure(rel_path: str) -> int:
    """Line count of a module via ``str.splitlines()``.

    Cached because both tests below parametrize over the same paths.

    Raises:
        AssertionError: If the module is missing — the budget entry is
            stale and must be updated deliberately, not silently skipped.
    """
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} not found; the budget entry is stale."
    return len(path.read_text(encoding="utf-8").splitlines())


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_module_does_not_grow_past_its_budget(rel_path: str) -> None:
    """The ceiling itself: a budgeted module may not exceed its pin."""
    max_lines = _BUDGETS[rel_path]
    lines = _measure(rel_path)

    assert lines <= max_lines, (
        f"{rel_path} grew to {lines} lines (budget {max_lines}, "
        f"+{lines - max_lines}).\n\n"
        f"{rel_path} is under a size ratchet "
        f"(Tests/Architecture/test_module_size_ratchet.py). Put new code in "
        f"a controller/widget/helper module — do NOT raise the budget to "
        f"make this pass. Lower it when the file shrinks."
    )


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_budget_is_not_left_slack(rel_path: str) -> None:
    """The recorded budget should track reality, not drift above it."""
    max_lines = _BUDGETS[rel_path]
    lines = _measure(rel_path)

    assert max_lines - lines <= _SLACK_TOLERANCE_LINES, (
        f"{rel_path} is {max_lines - lines} lines under its budget "
        f"({lines} vs {max_lines}). Set it to {lines} so the real "
        f"measurement is what's pinned."
    )
