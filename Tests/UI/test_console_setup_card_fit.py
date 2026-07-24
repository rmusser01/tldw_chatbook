"""TASK-389: the first-run setup card must fit its step-3 line (or ellipsize).

The honest step-3 line "3. ○ Send your first message  Composer unlocks after
setup" is wider than the old 62-cell card gave in content, so its last word
wrapped to a hidden second line with no ellipsis. This ties the card width to the
actual step text so the copy can't silently outgrow the card again.
"""

import re
from pathlib import Path

from rich.cells import cell_len

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_STEP_THREE_DETAIL,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"


def _rule_body(css: str, selector: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    return match.group(1) if match else ""


def test_setup_card_is_wide_enough_for_the_step_three_line():
    """AC#1: the full step-3 sentence fits inside the card's content width."""
    step_three = ConsoleSetupStep(
        state="pending",
        label="Send your first message",
        detail=CONSOLE_SETUP_STEP_THREE_DETAIL,
    )
    line = ConsoleSetupModal._step_text(3, step_three)

    card = _rule_body(AGENTIC.read_text(encoding="utf-8"), ".console-setup-modal-card")
    width = int(re.search(r"\bwidth\s*:\s*(\d+)", card).group(1))
    # padding: 1 2 -> 2 left + 2 right; border: solid -> 1 each side.
    content_width = width - 4 - 2

    assert cell_len(line) <= content_width, (
        f"step-3 line is {cell_len(line)} cells but the card content is only "
        f"{content_width}: {line!r}"
    )


def test_setup_step_ellipsizes_on_unavoidable_overflow():
    """AC#2: a step too wide for a capped card is marked, not silently clipped."""
    step = _rule_body(AGENTIC.read_text(encoding="utf-8"), ".console-setup-step")
    assert "nowrap" in step
    assert "ellipsis" in step
