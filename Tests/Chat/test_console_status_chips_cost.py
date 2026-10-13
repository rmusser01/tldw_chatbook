"""Unit tests for the Console cost chip (PR3 task-4).

Mirrors the mount/assert idioms in ``Tests/UI/test_console_status_chips.py``
and the message-capture idiom in ``Tests/UI/test_console_shell_chip_actions.py``
/``Tests/UI/test_console_status_chips.py::test_approvals_chip_posts_review_requested``.
"""

import re
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest
from rich.cells import cell_len
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import ModalScreen
from textual.widgets import Input, Static

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleCostChip,
    ConsoleStatusChips,
)

ROOT = Path(__file__).resolve().parents[2]
_CSS_ROOT = ROOT / "tldw_chatbook" / "css"
# ADR-161 task 10: the console-owned chip rules live in the console
# source sheets (union: each rule exists in exactly one of the pair) and
# ride the boot bundle.
CONSOLE_SOURCES = (
    _CSS_ROOT / "features" / "_console.tcss",
    _CSS_ROOT / "features" / "_console_panels.tcss",
)
BUNDLE = _CSS_ROOT / "tldw_cli_modular.tcss"


def _console_sources_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in CONSOLE_SOURCES)


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
    # ADR-161 task 10: the console vocabulary rides the boot bundle via
    # features/_console{,_panels}.tcss (the always-boot-parsed console
    # split sheet was dissolved), so the bundle alone is the app-tier set.
    CSS_PATH = [str(BUNDLE)]


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
        assert isinstance(app.query_one("#console-cost-chip"), ConsoleCostChip)


@pytest.mark.asyncio
async def test_cost_chip_renders_label_from_state():
    app = _ChipsApp(_control_state(), cost_state=_cost_state(label="$1.23 ● ~+$0.05"))
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
    app = _ChipsApp(_control_state(), cost_state=_cost_state(alert=False, cold=True))
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


@pytest.mark.asyncio
async def test_context_cost_label_uses_compact_copy_on_initial_narrow_mount():
    state = _cost_state(
        label="Context 45% · Current $0.48 · On next send ~+$0.13",
        compact_label="Ctx 45% · Now $0.48 · Next ~+$0.13",
    )
    app = _ChipsApp(_control_state(), cost_state=state)

    async with app.run_test(size=(80, 6)) as pilot:
        await pilot.pause()
        rendered = str(app.query_one("#console-cost-chip", ConsoleCostChip).render())

        assert "Ctx 45% · Now $0.48 · Next ~+$0.13" in rendered
        assert "On next send" not in rendered


@pytest.mark.asyncio
async def test_context_cost_label_tracks_live_resize_without_state_change():
    state = _cost_state(
        label="Context 45% · Current $0.48 · On next send ~+$0.13",
        compact_label="Ctx 45% · Now $0.48 · Next ~+$0.13",
    )
    app = _ChipsApp(_control_state(), cost_state=state)

    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        assert "On next send" in str(chip.render())

        await pilot.resize_terminal(80, 6)
        await pilot.pause()
        assert "Ctx 45% · Now $0.48 · Next ~+$0.13" in str(chip.render())

        await pilot.resize_terminal(200, 6)
        await pilot.pause()
        assert "Context 45% · Current $0.48 · On next send ~+$0.13" in str(
            chip.render()
        )


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    (
        (80, 24, "Ctx 100%+ · Now $12,345.67 · Next ~+$987.65"),
        (
            120,
            35,
            "Context 100%+ · Current $12,345.67 · On next send ~+$987.65",
        ),
    ),
)
@pytest.mark.asyncio
async def test_context_cost_label_fits_shipped_status_strip(
    width: int, height: int, expected: str
) -> None:
    app = _ProductionChipsApp(
        _control_state(),
        cost_state=_cost_state(
            label="Context 100%+ · Current $12,345.67 · On next send ~+$987.65",
            compact_label="Ctx 100%+ · Now $12,345.67 · Next ~+$987.65",
        ),
    )
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        scroll = app.query_one("#console-status-chip-scroll")

        rendered = str(chip.render())
        assert rendered == expected
        assert "…" not in rendered and "..." not in rendered
        assert chip.region.height == 1
        assert chip.region.y == scroll.content_region.y
        assert chip.content_region.width >= cell_len(expected)


#: The status row of captures 03f/05c (OpenAI, nothing sent yet).
_CAPTURE_LABELS = {
    "label": "Context 0% · Current $0.00 · On next send —",
    "compact_label": "Ctx 0% · Now $0.00 · Next —",
}


def _capture_control_state(**overrides) -> ConsoleControlState:
    return _control_state(
        **{
            "rag_label": "Library · Auto off · Agent access off",
            "provider_label": "Provider: OpenAI",
            "model_label": "Model: gpt-5.6-terra",
            "system_prompt_label": "System Prompt: off",
            "sources_label": "Sources: 0",
            **overrides,
        }
    )


def _cost_chip_drawn_whole(app) -> str:
    """The chip's label, after asserting the strip shows all of the chip."""
    chip = app.query_one("#console-cost-chip", ConsoleCostChip)
    scroll = app.query_one("#console-status-chip-scroll")
    assert chip.region.right <= scroll.content_region.right, (
        str(chip.render()),
        chip.region,
        scroll.content_region,
    )
    return str(chip.render())


@pytest.mark.parametrize(
    ("width", "expected"),
    (
        (211, "Ctx 0% · Now $0.00"),
        (235, _CAPTURE_LABELS["label"]),
    ),
)
@pytest.mark.asyncio
async def test_the_cost_chip_drops_whole_segments_to_fit_the_strip(
    width: int, expected: str
) -> None:
    """Capture fix 3 finding 4 (03f, 05c): at 211 columns the strip cut the
    chip at "Current $0.", because the label is chosen by screen width alone
    and the chips before it fill the row. The chip shows the longest form
    whose whole " · " segments fit the room left; the tooltip keeps all."""
    app = _ProductionChipsApp(
        _capture_control_state(), cost_state=_cost_state(**_CAPTURE_LABELS)
    )
    async with app.run_test(size=(width, 8)) as pilot:
        await pilot.pause()
        await pilot.pause()
        assert _cost_chip_drawn_whole(app) == expected


@pytest.mark.asyncio
async def test_the_cost_chip_refits_when_a_chip_before_it_changes() -> None:
    """Capture fix 3 finding 4: a longer model name moves the cost chip right
    with no cost change; the chip refits, and fits again when it shortens."""
    app = _ProductionChipsApp(
        _capture_control_state(), cost_state=_cost_state(**_CAPTURE_LABELS)
    )
    async with app.run_test(size=(235, 8)) as pilot:
        await pilot.pause()
        assert _cost_chip_drawn_whole(app) == _CAPTURE_LABELS["label"]
        chips = app.query_one(ConsoleStatusChips)

        longer = _capture_control_state(model_label="Model: gpt-5.6-terra-mini")
        chips.sync_state(longer)
        await pilot.pause()
        await pilot.pause()
        assert _cost_chip_drawn_whole(app) == _CAPTURE_LABELS["compact_label"]

        chips.sync_state(_capture_control_state())
        await pilot.pause()
        await pilot.pause()
        assert _cost_chip_drawn_whole(app) == _CAPTURE_LABELS["label"]


@pytest.mark.parametrize(
    "overrides",
    (
        {"tools_label": "Tools: 12 ready", "tools_active": True},
        {
            "provider_label": "Provider: OpenRouter",
            "model_label": "Model: meta-llama/llama-3.3-70b-instruct",
            "assistant_label": "Assistant: Research helper",
        },
    ),
    ids=("tools-chip", "openrouter-long-model"),
)
@pytest.mark.asyncio
async def test_the_cost_chip_shows_an_ellipsis_when_no_form_fits(overrides) -> None:
    """Capture fix 3 round 1 finding 1: with the Tools chip shown (or a long
    provider, model and assistant) the room left at 211 columns is under the
    shortest form, and the full label was drawn cut ("  Cont"). The chip
    shows "…" whole instead (its 12-cell box is clipped at the edge like any
    chip there, its text is not); focus shows the full label in view (the
    strip scrolls to it), and leaving it puts "…" back."""
    app = _ProductionChipsApp(
        _capture_control_state(**overrides),
        cost_state=_cost_state(**_CAPTURE_LABELS),
    )
    async with app.run_test(size=(211, 8)) as pilot:
        await pilot.pause()
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        edge = app.query_one("#console-status-chip-scroll").content_region.right
        assert str(chip.render()) == "…"
        assert chip.content_region.x + cell_len("…") <= edge, chip.region

        chip.focus()
        for _ in range(5):
            await pilot.pause()
        assert _cost_chip_drawn_whole(app) == _CAPTURE_LABELS["label"]

        app.query_one("#console-provider-chip").focus()
        for _ in range(5):
            await pilot.pause()
        assert str(chip.render()) == "…"


@pytest.mark.asyncio
async def test_the_cost_chip_fits_when_it_is_shown_after_layout() -> None:
    """Capture fix 3 round 1 finding 3: a chip hidden at the strip's first
    layout (no cost state, e.g. after the last tab closed) was labelled at
    x 0 when a state arrived, so it got the full label and was cut at 211."""
    app = _ProductionChipsApp(_capture_control_state(), cost_state=None)
    async with app.run_test(size=(211, 8)) as pilot:
        await pilot.pause()
        app.query_one(ConsoleStatusChips).sync_cost_state(
            _cost_state(**_CAPTURE_LABELS)
        )
        for _ in range(5):
            await pilot.pause()
        assert _cost_chip_drawn_whole(app) == "Ctx 0% · Now $0.00"


class _SwitchModelStandIn(ModalScreen[None]):
    """A dialog over the strip: Enter applies a pair while it is still the
    active screen, as Switch model's live commit does; Esc closes it."""

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("enter", "apply", "Apply"),
        Binding("escape", "dismiss", "Close"),
    ]

    def __init__(self, apply: Callable[[], None]) -> None:
        super().__init__()
        self._apply = apply

    def compose(self) -> ComposeResult:
        yield Static("Switch model")

    def action_apply(self) -> None:
        self._apply()


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    (
        (
            {},
            {
                "provider_label": "Provider: llama.cpp",
                "model_label": "Model: qwen3-8b-instruct-q4_k_m",
            },
            "Ctx 0%",
        ),
        (
            {
                "provider_label": "Provider: DeepSeek",
                "model_label": "Model: deepseek-v4-flash",
            },
            {},
            "Ctx 0% · Now $0.00",
        ),
    ),
    ids=("wider-pair-02c", "narrower-pair"),
)
@pytest.mark.asyncio
async def test_a_pair_applied_under_a_dialog_refits_the_cost_chip(
    before: dict, after: dict, expected: str
) -> None:
    """Capture findings 4 (02c at 211x44): a model applied from Switch model
    changed the Provider and Model chips' widths, and the cost chip kept the
    fit for the old widths until a resize ("Ctx 0% · No", cut mid-figure;
    "Context 0%" where "Ctx 0% · Now $0.00" fit). The refit the change
    scheduled ran after the dialog's refresh, before the Console under it
    laid the new widths out."""
    app = _ProductionChipsApp(
        _capture_control_state(**before), cost_state=_cost_state(**_CAPTURE_LABELS)
    )
    async with app.run_test(size=(211, 8)) as pilot:
        for _ in range(3):
            await pilot.pause()
        chips = app.query_one(ConsoleStatusChips)
        assert _cost_chip_drawn_whole(app) != expected
        app.push_screen(
            _SwitchModelStandIn(
                lambda: chips.sync_state(_capture_control_state(**after))
            )
        )
        await pilot.pause()
        await pilot.press("enter")
        for _ in range(3):
            await pilot.pause()
        await pilot.press("escape")
        for _ in range(5):
            await pilot.pause()
        assert not isinstance(app.screen, _SwitchModelStandIn)
        assert _cost_chip_drawn_whole(app) == expected


class _ChipsAndComposerApp(_ProductionChipsApp):
    """The production strip with an Input outside it, like the composer."""

    def compose(self) -> ComposeResult:
        yield from super().compose()
        yield Input(id="composer")


@pytest.mark.parametrize("leave", ("composer", "shift+tab"))
@pytest.mark.asyncio
async def test_leaving_the_cost_chip_scrolls_the_strip_home(leave: str) -> None:
    """Capture fix 3 round 2 finding 1: focus scrolled the strip to the
    chip's full label; when focus left for the composer (or Shift+Tab to an
    earlier chip) the chip went back to "…" but the strip stayed scrolled,
    so its first chip was cut at the left edge."""
    app = _ChipsAndComposerApp(
        _capture_control_state(tools_label="Tools: 12 ready", tools_active=True),
        cost_state=_cost_state(**_CAPTURE_LABELS),
    )
    async with app.run_test(size=(211, 8)) as pilot:
        await pilot.pause()
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        scroll = app.query_one("#console-status-chip-scroll")
        first = app.query_one("#console-provider-chip")
        before = (str(chip.render()), first.region.x)
        assert before[0] == "…"

        chip.focus()
        for _ in range(5):
            await pilot.pause()
        assert scroll.scroll_x > 0

        if leave == "composer":
            app.query_one("#composer", Input).focus()
        else:
            await pilot.press("shift+tab")
        for _ in range(5):
            await pilot.pause()
        assert app.focused is not chip
        assert scroll.scroll_x == 0
        assert (str(chip.render()), first.region.x) == before


@pytest.mark.asyncio
async def test_a_focused_cost_chip_relabelled_longer_stays_in_view() -> None:
    """Capture fix 3 round 2 finding 2: the strip scrolled to the focused
    chip only once, on focus; a reply finishing while it had focus gave it a
    longer label whose end the strip cut at the edge."""
    app = _ProductionChipsApp(
        _capture_control_state(tools_label="Tools: 12 ready", tools_active=True),
        cost_state=_cost_state(**_CAPTURE_LABELS),
    )
    async with app.run_test(size=(211, 8)) as pilot:
        await pilot.pause()
        app.query_one("#console-cost-chip", ConsoleCostChip).focus()
        for _ in range(5):
            await pilot.pause()
        assert _cost_chip_drawn_whole(app) == _CAPTURE_LABELS["label"]

        longer = "Context 4% · Current $12,345.67 ● · On next send —"
        app.query_one(ConsoleStatusChips).sync_cost_state(
            _cost_state(label=longer, compact_label=_CAPTURE_LABELS["compact_label"])
        )
        for _ in range(5):
            await pilot.pause()
        assert _cost_chip_drawn_whole(app) == longer


@pytest.mark.parametrize("width", (211, 235))
@pytest.mark.asyncio
async def test_a_refit_that_keeps_the_focused_chips_label_keeps_the_scroll(
    width: int,
) -> None:
    """Capture fix 3 round 3: every refit of a focused cost chip scrolled
    the strip back to it, so a count change elsewhere on the row undid the
    user's Shift+wheel or swipe back to its start."""
    app = _ProductionChipsApp(
        _capture_control_state(tools_label="Tools: 12 ready", tools_active=True),
        cost_state=_cost_state(**_CAPTURE_LABELS),
    )
    async with app.run_test(size=(width, 8)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        scroll = app.query_one("#console-status-chip-scroll")
        chip.focus()
        for _ in range(5):
            await pilot.pause()
        assert scroll.scroll_x > 0

        scroll.scroll_home(animate=False)
        for _ in range(5):
            await pilot.pause()
        assert (scroll.scroll_x, app.focused) == (0, chip)

        app.query_one(ConsoleStatusChips).sync_state(
            _capture_control_state(
                tools_label="Tools: 12 ready",
                tools_active=True,
                approvals_label="Approvals: 1 pending",
                approvals_active=True,
            )
        )
        for _ in range(5):
            await pilot.pause()
        assert str(chip.render()) == _CAPTURE_LABELS["label"]
        assert scroll.scroll_x == 0


@pytest.mark.parametrize(
    ("width", "change"), ((235, "narrower-terminal"), (211, "wider-chip-before-it"))
)
@pytest.mark.asyncio
async def test_a_focused_cost_chip_in_view_stays_in_view_when_the_row_moves(
    width: int, change: str
) -> None:
    """Capture fix 3 round 4: with the label unchanged, a narrower terminal or
    a wider chip before it pushed the focused chip past the right edge (13
    of its 47 cells showed at 200 columns): the strip only followed a
    relabel. A chip the user has not scrolled away from stays in view."""
    app = _ProductionChipsApp(
        _capture_control_state(tools_label="Tools: 12 ready", tools_active=True),
        cost_state=_cost_state(**_CAPTURE_LABELS),
    )
    async with app.run_test(size=(width, 8)) as pilot:
        await pilot.pause()
        app.query_one("#console-cost-chip", ConsoleCostChip).focus()
        for _ in range(5):
            await pilot.pause()
        assert _cost_chip_drawn_whole(app) == _CAPTURE_LABELS["label"]

        if change == "narrower-terminal":
            await pilot.resize_terminal(200, 8)
        else:
            app.query_one(ConsoleStatusChips).sync_state(
                _capture_control_state(
                    tools_label="Tools: 12 ready",
                    tools_active=True,
                    approvals_label="Approvals: 12 pending, 3 queued",
                    approvals_active=True,
                )
            )
        for _ in range(5):
            await pilot.pause()
        assert _cost_chip_drawn_whole(app) == _CAPTURE_LABELS["label"]


@pytest.mark.parametrize("width", (211, 235))
@pytest.mark.asyncio
async def test_a_same_width_relabel_does_not_undo_a_later_swipe(width: int) -> None:
    """Capture fix 3 round 5: a focused relabel that kept the chip's width
    ("$0.00" to "$0.01") laid nothing out, so the mark it left was still set
    after the user swiped the row home; the next wider chip before it
    scrolled the row back to the cost chip (scroll_x 0 to 52 at 211)."""
    app = _ProductionChipsApp(
        _capture_control_state(tools_label="Tools: 12 ready", tools_active=True),
        cost_state=_cost_state(**_CAPTURE_LABELS),
    )
    async with app.run_test(size=(width, 8)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        scroll = app.query_one("#console-status-chip-scroll")
        chips = app.query_one(ConsoleStatusChips)
        chip.focus()
        for _ in range(5):
            await pilot.pause()

        same_width = "Context 1% · Current $0.01 · On next send —"
        chips.sync_cost_state(
            _cost_state(
                label=same_width, compact_label=_CAPTURE_LABELS["compact_label"]
            )
        )
        for _ in range(5):
            await pilot.pause()
        assert str(chip.render()) == same_width

        scroll.scroll_home(animate=False)
        for _ in range(5):
            await pilot.pause()
        assert (scroll.scroll_x, app.focused) == (0, chip)

        chips.sync_state(
            _capture_control_state(
                tools_label="Tools: 12 ready",
                tools_active=True,
                approvals_label="Approvals: 12 pending, 3 queued",
                approvals_active=True,
            )
        )
        for _ in range(5):
            await pilot.pause()
        assert scroll.scroll_x == 0


@pytest.mark.parametrize(("start", "end"), ((60, 56), (52, 50)))
@pytest.mark.asyncio
async def test_a_capped_cost_chip_does_not_undo_a_swipe_on_resize(
    start: int, end: int
) -> None:
    """Capture fix 3 round 6: a focused label wider than the strip is capped
    at the strip's width, so a narrower terminal changed the chip's width
    with no relabel and the check read that as a relabel: the strip the user
    had swiped home scrolled back to the cost chip (scroll_x 0 to 194)."""
    app = _ProductionChipsApp(
        _capture_control_state(tools_label="Tools: 12 ready", tools_active=True),
        cost_state=_cost_state(
            label="Context 100%+ · Current $12,345.67 · On next send ~+$987.65",
            compact_label="Ctx 100%+ · Now $12,345.67 · Next ~+$987.65",
        ),
    )
    async with app.run_test(size=(start, 8)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-cost-chip", ConsoleCostChip)
        scroll = app.query_one("#console-status-chip-scroll")
        chip.focus()
        for _ in range(5):
            await pilot.pause()

        scroll.scroll_home(animate=False)
        for _ in range(5):
            await pilot.pause()
        assert (scroll.scroll_x, app.focused) == (0, chip)

        width = chip.size.width
        await pilot.resize_terminal(end, 8)
        for _ in range(5):
            await pilot.pause()
        assert chip.size.width < width
        assert scroll.scroll_x == 0


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


def test_console_chip_cold_css_exists_in_source_and_generated_screen_sheet():
    """The chip rule exists in the console sources and the shipped bundle.

    ADR-161 task 10: the TASK-25812 console split sheet was dissolved --
    the console vocabulary rides the boot bundle via the console source
    sheets -- so the pair is (console sources union, bundle).
    """
    for label, css_text in (
        ("console sources", _console_sources_text()),
        ("bundle", BUNDLE.read_text(encoding="utf-8")),
    ):
        body = _chip_cold_body(css_text)
        assert body, f"{label}: no .console-chip-cold rule"
        assert "$ds-status-info" in body or "color" in body
