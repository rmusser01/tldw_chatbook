"""TASK-25825: six actions should not cost thirty rows.

The composer menu gave each row `height: 3` plus `margin-bottom: 1` -- four
rows per item, ~30 for six actions -- and centred every label, so the list
scanned badly and occluded the transcript to present very little. Worse, the
disabled rows' explanatory reason was separated from the row it explains by
that trailing margin, weakening the association the reason exists to make.

The rule has to live in the APP stylesheet, not the modal's DEFAULT_CSS: the
sibling contract test (test_master_shell_design_system_contract) records that
`Button`'s own rules outrank a modal's DEFAULT_CSS, where "an identical rule
inside the modal measured no change".
"""

from __future__ import annotations

import re
from pathlib import Path

_SHEET = Path("tldw_chatbook/css/components/_agentic_terminal.tcss")


def _rule(selector: str) -> str:
    match = re.search(
        rf"{re.escape(selector)}\s*\{{([^}}]*)\}}", _SHEET.read_text(), re.S
    )
    assert match, f"{selector} must be declared in the app stylesheet"
    return match.group(1)


def test_menu_rows_are_left_aligned() -> None:
    assert "text-align: left" in _rule(".console-composer-menu-item")


def test_menu_rows_do_not_carry_a_trailing_gap() -> None:
    """The gap is what separated a disabled row from its own reason."""
    body = _rule(".console-composer-menu-item")
    margin = re.search(r"margin:\s*([^;]+);", body)
    assert margin, "the row rule must state its margin explicitly"
    assert margin.group(1).strip() == "0", (
        f"rows should not add a trailing gap, got margin: {margin.group(1)}"
    )
