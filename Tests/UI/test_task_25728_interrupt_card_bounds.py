"""TASK-25728: a decision card must not consume the whole screen.

At 80x24 -- the default terminal, and the size this user runs -- the
trace-recovery card's `height: auto` grew until no transcript was visible at
all. The user was asked to choose between sending, retrying and cancelling
without being able to see the message the decision applied to.
"""

from __future__ import annotations

import re

from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)


def _rule(selector: str) -> str:
    match = re.search(
        rf"{re.escape(selector)}\s*\{{([^}}]*)\}}",
        TraceCallRecoveryCallout.BUNDLED_CSS,
        re.S,
    )
    assert match, f"{selector} must be declared"
    return match.group(1)


def test_card_is_height_bounded() -> None:
    body = _rule("TraceCallRecoveryCallout")
    assert "max-height:" in body, (
        "an unbounded height: auto lets the card grow until the transcript is "
        "gone at 80x24"
    )


def test_card_scrolls_inside_its_own_bound() -> None:
    body = _rule("TraceCallRecoveryCallout")
    assert "overflow-y: auto" in body, (
        "bounding the card without letting it scroll would hide its own actions"
    )


def test_actions_do_not_waste_a_row_between_every_button() -> None:
    body = _rule("TraceCallRecoveryCallout Button")
    margin = re.search(r"margin-bottom:\s*([^;]+);", body)
    assert margin is None or margin.group(1).strip() == "0", (
        "a trailing row under each of four buttons is four rows of a 24-row "
        f"screen, got margin-bottom: {margin.group(1) if margin else None}"
    )
