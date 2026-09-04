"""Unit tests for the Console cost chip (PR3 task-4).

Mirrors the mount/assert idioms in ``Tests/UI/test_console_status_chips.py``
and the message-capture idiom in ``Tests/UI/test_console_shell_chip_actions.py``
/``Tests/UI/test_console_status_chips.py::test_approvals_chip_posts_review_requested``.
"""

import re
from pathlib import Path

import pytest
from textual import on
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleCostChip,
    ConsoleStatusChips,
)

ROOT = Path(__file__).resolve().parents[2]
AGENTIC = ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BUNDLE = ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _control_state(**overrides) -> ConsoleControlState:
    base = dict(
        provider_label="Provider: Anthropic",
        model_label="Model: claude-3-haiku",
        assistant_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0 staged",
        tools_label="Tools: 0 ready",
        approvals_label="Approvals: 0 pending",
        sources_active=False,
        tools_active=False,
        approvals_active=False,
    )
    base.update(overrides)
    return ConsoleControlState(**base)


def _cost_state(**overrides) -> ConsoleCostState:
    base = dict(
        label="$0.48 ● ~+$0.13",
        compact_label="$0.48 ●",
        tooltip="Total: $0.48\nTokens: 1.2k\nCache: warm (4:00 remaining)",
        alert=False,
        cold=False,
    )
    base.update(overrides)
    return ConsoleCostState(**base)


class _ChipsApp(App):
    CSS = ".console-control-chip { width: auto; }"

    def __init__(self, state: ConsoleControlState, *, cost_state=None) -> None:
        super().__init__()
        self._state = state
        self._cost_state = cost_state
        self.captured_pressed: list[object] = []

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(
            self._state,
            cost_state=self._cost_state,
            id="console-status-chips",
        )

    @on(ConsoleCostChip.ConsoleCostChipPressed)
    def _capture_cost_chip_pressed(
        self, event: ConsoleCostChip.ConsoleCostChipPressed
    ) -> None:
        self.captured_pressed.append(event)


class _ProductionChipsApp(_ChipsApp):
    """Status-chip harness using the shipped stylesheet and hierarchy."""

    CSS = ""
    CSS_PATH = str(BUNDLE)


@pytest.mark.asyncio
async def test_cost_chip_is_composed_last():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        visible_ids = [
            chip.id for chip in chips.query(".console-control-chip") if chip.display
        ]
        assert visible_ids[-1] == "console-cost-chip"
        assert isinstance(
            app.query_one("#console-cost-chip"), ConsoleCostChip
        )


@pytest.mark.asyncio
async def test_cost_chip_renders_label_from_state():
    app = _ChipsApp(
        _control_state(), cost_state=_cost_state(label="$1.23 ● ~+$0.05")
    )
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        assert chip.display is True
        assert "$1.23" in str(chip.render())


@pytest.mark.asyncio
async def test_cost_chip_hidden_when_no_cost_state_at_compose():
    app = _ChipsApp(_control_state(), cost_state=None)
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        assert chip.display is False


@pytest.mark.asyncio
async def test_sync_cost_state_hides_chip_when_state_becomes_none():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        assert chip.display is True

        chips.sync_cost_state(None)
        await pilot.pause()
        assert chip.display is False


@pytest.mark.asyncio
async def test_sync_cost_state_reveals_chip_from_hidden():
    app = _ChipsApp(_control_state(), cost_state=None)
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        assert chip.display is False

        chips.sync_cost_state(_cost_state(label="$0.02 ●"))
        await pilot.pause()
        assert chip.display is True
        assert "$0.02" in str(chip.render())


@pytest.mark.asyncio
async def test_neutral_state_is_dim_not_alert_or_cold():
    app = _ChipsApp(_control_state(), cost_state=_cost_state(alert=False, cold=False))
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        assert chip.has_class("console-chip-dim")
        assert not chip.has_class("console-chip-alert")
        assert not chip.has_class("console-chip-cold")


@pytest.mark.asyncio
async def test_sync_cost_state_toggles_alert_class():
    app = _ChipsApp(_control_state(), cost_state=_cost_state(alert=False, cold=False))
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)

        chips.sync_cost_state(
            _cost_state(alert=True, cold=False, label="$0.48 ⚠ ~+$0.13")
        )
        await pilot.pause()
        assert chip.has_class("console-chip-alert")
        assert not chip.has_class("console-chip-dim")
        assert not chip.has_class("console-chip-cold")
        assert "$0.48" in str(chip.render())

        # And back to neutral -- classes must not stick around.
        chips.sync_cost_state(_cost_state(alert=False, cold=False, label="$0.50 ●"))
        await pilot.pause()
        assert chip.has_class("console-chip-dim")
        assert not chip.has_class("console-chip-alert")
        assert not chip.has_class("console-chip-cold")


@pytest.mark.asyncio
async def test_sync_cost_state_toggles_cold_class():
    app = _ChipsApp(_control_state(), cost_state=_cost_state(alert=False, cold=False))
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)

        chips.sync_cost_state(_cost_state(alert=False, cold=True, label="$0.48 ○"))
        await pilot.pause()
        assert chip.has_class("console-chip-cold")
        assert not chip.has_class("console-chip-dim")
        assert not chip.has_class("console-chip-alert")


@pytest.mark.asyncio
async def test_cold_state_at_compose_time():
    """The cold class must also apply on the very first frame (F1 precedent)."""
    app = _ChipsApp(
        _control_state(), cost_state=_cost_state(alert=False, cold=True)
    )
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        assert chip.has_class("console-chip-cold")
        assert not chip.has_class("console-chip-dim")
        assert not chip.has_class("console-chip-alert")


@pytest.mark.asyncio
async def test_sync_cost_state_equality_guard_skips_redundant_render():
    """An equal-by-value (but not identical) state must not re-render the chip."""
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)

        calls: list[object] = []
        original_update = chip.update
        chip.update = lambda *a, **k: calls.append((a, k)) or original_update(*a, **k)
        try:
            same_by_value = _cost_state()
            assert same_by_value is not chips._cost_state
            assert same_by_value == chips._cost_state
            chips.sync_cost_state(same_by_value)
            await pilot.pause()
        finally:
            chip.update = original_update

        assert calls == [], "equal cost state must not re-render the chip"


@pytest.mark.asyncio
async def test_sync_cost_state_uses_compact_label_when_narrow():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(80, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        assert chips.size.width < 120
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)

        chips.sync_cost_state(
            _cost_state(
                label="$0.48 ⚠ ~+$0.13",
                compact_label="$0.48 ⚠",
                alert=True,
            )
        )
        await pilot.pause()
        rendered = str(chip.render())
        assert "$0.48" in rendered
        assert "~+$0.13" not in rendered


@pytest.mark.asyncio
async def test_sync_cost_state_uses_full_label_when_wide():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        assert chips.size.width >= 120
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)

        chips.sync_cost_state(
            _cost_state(
                label="$0.48 ⚠ ~+$0.13",
                compact_label="$0.48 ⚠",
                alert=True,
            )
        )
        await pilot.pause()
        rendered = str(chip.render())
        assert "~+$0.13" in rendered


@pytest.mark.parametrize(
    ("width", "expected_prefix"),
    ((80, "Ctx 45%"), (200, "Context 45%")),
)
@pytest.mark.asyncio
async def test_context_cost_label_fits_shipped_status_strip(
    width: int, expected_prefix: str
) -> None:
    app = _ProductionChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(width, 6)) as pilot:
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chips.sync_cost_state(
            _cost_state(
                label="Context 45% · $0.48 ●",
                compact_label="Ctx 45% · $0.48 ●",
            )
        )
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        scroll = app.query_one("#console-status-chip-scroll")

        rendered = str(chip.render())
        assert rendered.startswith(expected_prefix)
        assert chip.region.height == 1
        assert chip.region.y == scroll.content_region.y
        assert chip.content_region.width >= len(rendered)


@pytest.mark.asyncio
async def test_cost_chip_click_posts_pressed_message():
    """A real ``pilot.click`` must dispatch through the normal message pump.

    Unlike the keyboard-activation test below, this must NOT monkeypatch
    ``chip.post_message`` before the click: Textual posts the internal
    ``Click`` event to the widget through that same method, so stubbing it
    out first would swallow the click event itself and never reach
    ``_on_click``. Instead this captures the bubbled
    ``ConsoleCostChipPressed`` at the App via a real ``@on`` handler.
    """
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        clicked = await pilot.click("#console-cost-chip")
        await pilot.pause()
        assert clicked
        assert any(
            isinstance(message, ConsoleCostChip.ConsoleCostChipPressed)
            for message in app.captured_pressed
        )


@pytest.mark.asyncio
async def test_cost_chip_keyboard_activation_posts_pressed_message():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        posted: list[object] = []
        original_post_message = chip.post_message
        chip.post_message = lambda message: posted.append(message)  # type: ignore[assignment]
        try:
            chip.action_open_cost_breakdown()
        finally:
            chip.post_message = original_post_message
        assert any(
            isinstance(message, ConsoleCostChip.ConsoleCostChipPressed)
            for message in posted
        )


def test_cost_chip_uses_markup_false():
    """Cost labels are machine-generated, but the house rule is uniform
    (dollar amounts don't need it, but a stray ``[`` should never raise)."""
    import inspect

    source = inspect.getsource(ConsoleStatusChips._cost_chip)
    assert "chip_class=ConsoleCostChip" in source


def _chip_cold_body(css_text: str) -> str:
    uncommented = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    match = re.search(r"\.console-chip-cold\s*\{([^}]*)\}", uncommented)
    return match.group(1) if match else ""


def test_console_chip_cold_css_exists_in_both_source_and_bundle():
    """Dossier §9c dual-file contract: the source module and the generated
    bundle must both carry the new class (the bundle is regenerated from
    source, never hand-edited -- this just proves the rebuild happened)."""
    for css_path in (AGENTIC, BUNDLE):
        body = _chip_cold_body(css_path.read_text(encoding="utf-8"))
        assert body, f"{css_path.name}: no .console-chip-cold rule"
        assert "$ds-status-info" in body or "color" in body
