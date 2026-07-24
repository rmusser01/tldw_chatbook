"""TASK-389/441: the first-run setup card must fit its step-3 line.

The honest step-3 line "3. ○ Send your first message  Composer unlocks after
setup" is wider than the old 62-cell card gave in content, so its last word
wrapped to a hidden second line with no ellipsis. TASK-389 first fixed this by
widening the card (AC#1 pin below) and ellipsizing any remaining overflow on a
single ``height: 1`` line. TASK-441 replaces the ellipsis strategy: at default
terminal sizes the full sentence already fits on one line (AC#1), but on a
narrow terminal that caps ``.console-setup-modal-card`` below its declared
width, the copy must now *wrap* onto additional rows instead of truncating
(AC#2) -- ``.console-setup-step`` moved from a fixed ``height: 1`` +
``text-wrap: nowrap`` + ``text-overflow: ellipsis`` to ``height: auto`` with
Static's default wrapping restored.
"""

import re
from pathlib import Path

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_STEP_THREE_DETAIL,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
_BUNDLED_STYLESHEET = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _rule_body(css: str, selector: str) -> str:
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    match = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
    return match.group(1) if match else ""


def _card_state_with_step_three() -> ConsoleSetupCardState:
    """Card state carrying the real production step-3 label + detail."""
    return ConsoleSetupCardState(
        mode="card",
        steps=(
            ConsoleSetupStep(state="active", label="Add an API key"),
            ConsoleSetupStep(state="done", label="Pick a model"),
            ConsoleSetupStep(
                state="pending",
                label="Send your first message",
                detail=CONSOLE_SETUP_STEP_THREE_DETAIL,
            ),
        ),
    )


def test_setup_card_is_wide_enough_for_the_step_three_line():
    """AC#1: the full step-3 sentence fits inside the card's content width."""
    step_three = ConsoleSetupStep(
        state="pending",
        label="Send your first message",
        detail=CONSOLE_SETUP_STEP_THREE_DETAIL,
    )
    line = ConsoleSetupModal._step_text(3, step_three)

    card = _rule_body(AGENTIC.read_text(encoding="utf-8"), ".console-setup-modal-card")
    # Match the `width:` property specifically -- not `max-width:` (the `-`
    # before "width" is excluded so the lookbehind can't be fooled by ordering).
    width = int(re.search(r"(?<![-\w])width\s*:\s*(\d+)", card).group(1))
    # padding: 1 2 -> 2 left + 2 right; border: solid -> 1 each side.
    content_width = width - 4 - 2

    assert cell_len(line) <= content_width, (
        f"step-3 line is {cell_len(line)} cells but the card content is only "
        f"{content_width}: {line!r}"
    )


def test_setup_step_wraps_instead_of_ellipsizing_on_overflow():
    """AC#2: overflow wraps onto another row instead of being truncated.

    TASK-389's ``text-wrap: nowrap`` + ``text-overflow: ellipsis`` guaranteed
    a *marked* truncation on unavoidable overflow, but still dropped words off
    the end of the sentence. TASK-441 asks for the words to survive by
    wrapping, so the step rule must no longer force a single line, and must
    grow (``height: auto``) rather than stay pinned at ``height: 1`` (which is
    what silently hid the wrapped second row before TASK-389's ellipsis fix
    existed at all).
    """
    step = _rule_body(AGENTIC.read_text(encoding="utf-8"), ".console-setup-step")
    assert "nowrap" not in step
    assert "ellipsis" not in step
    assert "height: auto" in step


class _SetupModalGeometryApp(App[None]):
    """Mount the setup modal with the production stylesheet for geometry checks."""

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        yield ConsoleSetupModal(id="console-setup-modal")

    async def on_mount(self) -> None:
        modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.sync_card_state(
            _card_state_with_step_three(),
            action_label="Configure API",
            action_tooltip="Open provider settings.",
        )


@pytest.mark.asyncio
async def test_step_three_is_one_complete_line_at_default_terminal_size():
    """AC#1, rendered: an 80-column default terminal shows the full sentence
    on a single row -- not clipped, and not wrapped (the card is wide enough)."""
    app = _SetupModalGeometryApp()
    async with app.run_test(size=(80, 24)):
        step3 = app.query_one("#console-setup-step-3", Static)
        assert step3.size.height == 1
        rendered = step3.render_line(0).text
        assert rendered.strip() == (
            "3. ○ Send your first message  Composer unlocks after setup"
        )


@pytest.mark.asyncio
async def test_step_three_wraps_onto_a_second_row_on_a_narrow_terminal():
    """AC#2, rendered: once the card's max-width caps it below the full
    sentence's width, the widget grows to a second row and every word --
    including the "setup" that TASK-389's evidence showed getting dropped --
    is still painted onscreen instead of being cut."""
    app = _SetupModalGeometryApp()
    async with app.run_test(size=(50, 30)):
        step3 = app.query_one("#console-setup-step-3", Static)
        # A fixed height: 1 (the pre-fix rule) would clamp this at 1 no matter
        # how much text wraps -- the second row growing in is the geometry
        # proof that overflow is now handled by wrapping, not by clipping.
        assert step3.size.height >= 2
        rendered_rows = [
            step3.render_line(y).text for y in range(step3.size.height)
        ]
        combined = " ".join(row.strip() for row in rendered_rows if row.strip())
        for word in (
            "Send",
            "your",
            "first",
            "message",
            "Composer",
            "unlocks",
            "after",
            "setup",
        ):
            assert word in combined, (
                f"{word!r} missing from wrapped rows {rendered_rows!r}"
            )
