"""Unit tests for the Console cost segment in the status strip (PR3 task-4).

TASK-35160 collapsed the strip into one text line; the cost chip's contracts
carry over to the cost SEGMENT: composed last, label from state (compact vs
full by viewport width, live on resize), hidden without a state, alert/dim
emphasis, whole-line equality guarding, and activation posting
``ConsoleCostChip.ConsoleCostChipPressed``. The chip era's distinct "cold"
styling flattened into dim (a markup span cannot carry an info color
token-legally); the cold FACT stays on the cost state (tooltip/breakdown).
"""

import pytest
from rich.cells import cell_len
from textual import on
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostState
from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleCostChip,
    ConsoleStatusChips,
    ConsoleStatusText,
)

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
    CSS = ".console-status-text { width: auto; }"

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


def _line(app: App) -> ConsoleStatusText:
    return app.query_one("#console-status-text", ConsoleStatusText)


def _line_text(app: App) -> str:
    renderable = _line(app).renderable
    return (
        renderable.plain if hasattr(renderable, "plain") else str(_line(app).render())
    )


def _visible_names(app: App) -> list[str]:
    return [name for name, _s, _e in _line(app).segment_spans]


def _cost_segment(app: App):
    strip = app.query_one("#console-status-chips", ConsoleStatusChips)
    for segment in strip._build_segments():
        if segment.name == "cost":
            return segment
    raise AssertionError("cost segment missing")


@pytest.mark.asyncio
async def test_cost_segment_is_composed_last():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        assert _visible_names(app)[-1] == "cost"


@pytest.mark.asyncio
async def test_cost_segment_renders_label_from_state():
    app = _ChipsApp(_control_state(), cost_state=_cost_state(label="$1.23 ● ~+$0.05"))
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        assert "$1.23" in _line_text(app)


@pytest.mark.asyncio
async def test_cost_segment_hidden_when_no_cost_state_at_compose():
    app = _ChipsApp(_control_state(), cost_state=None)
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        assert "cost" not in _visible_names(app)


@pytest.mark.asyncio
async def test_sync_cost_state_hides_segment_when_state_becomes_none():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        assert "cost" in _visible_names(app)

        chips.sync_cost_state(None)
        await pilot.pause()
        assert "cost" not in _visible_names(app)


@pytest.mark.asyncio
async def test_sync_cost_state_reveals_segment_from_hidden():
    app = _ChipsApp(_control_state(), cost_state=None)
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        assert "cost" not in _visible_names(app)

        chips.sync_cost_state(_cost_state(label="$0.02 ●"))
        await pilot.pause()
        assert "cost" in _visible_names(app)
        assert "$0.02" in _line_text(app)


@pytest.mark.asyncio
async def test_neutral_state_is_dim_not_alert():
    app = _ChipsApp(_control_state(), cost_state=_cost_state(alert=False, cold=False))
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        segment = _cost_segment(app)
        assert segment.dim is True
        assert segment.alert is False


@pytest.mark.asyncio
async def test_sync_cost_state_toggles_alert_emphasis():
    app = _ChipsApp(_control_state(), cost_state=_cost_state(alert=False, cold=False))
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)

        chips.sync_cost_state(
            _cost_state(alert=True, cold=False, label="$0.48 ⚠ ~+$0.13")
        )
        await pilot.pause()
        segment = _cost_segment(app)
        assert segment.alert is True
        assert segment.dim is False
        assert "$0.48" in _line_text(app)

        # And back to neutral -- emphasis must not stick around.
        chips.sync_cost_state(_cost_state(alert=False, cold=False, label="$0.50 ●"))
        await pilot.pause()
        segment = _cost_segment(app)
        assert segment.dim is True
        assert segment.alert is False


@pytest.mark.asyncio
async def test_sync_cost_state_cold_renders_like_neutral_without_alert():
    """The chip era's info-colored "cold" state flattened into dim on the
    text line (a markup span cannot carry a token-legal info color); the
    load-bearing parts are alert=False and the segment staying visible."""
    app = _ChipsApp(_control_state(), cost_state=_cost_state(alert=False, cold=True))
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        segment = _cost_segment(app)
        assert segment.alert is False
        assert segment.dim is True
        assert "cost" in _visible_names(app)


@pytest.mark.asyncio
async def test_sync_cost_state_equality_guard_skips_redundant_render():
    """An equal-by-value (but not identical) state must not re-render the line."""
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        line = _line(app)

        calls: list[object] = []
        original_update = line.update
        line.update = lambda *a, **k: calls.append((a, k)) or original_update(*a, **k)
        try:
            same_by_value = _cost_state()
            assert same_by_value is not chips._cost_state
            assert same_by_value == chips._cost_state
            chips.sync_cost_state(same_by_value)
            await pilot.pause()
        finally:
            line.update = original_update

        assert calls == [], "equal cost state must not re-render the line"


@pytest.mark.asyncio
async def test_sync_cost_state_uses_compact_label_when_narrow():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(80, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        assert chips.size.width < 120

        chips.sync_cost_state(
            _cost_state(
                label="$0.48 ⚠ ~+$0.13",
                compact_label="$0.48 ⚠",
                alert=True,
            )
        )
        await pilot.pause()
        rendered = _line_text(app)
        assert "$0.48" in rendered
        assert "~+$0.13" not in rendered


@pytest.mark.asyncio
async def test_sync_cost_state_uses_full_label_when_wide():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        assert chips.size.width >= 120

        chips.sync_cost_state(
            _cost_state(
                label="$0.48 ⚠ ~+$0.13",
                compact_label="$0.48 ⚠",
                alert=True,
            )
        )
        await pilot.pause()
        assert "~+$0.13" in _line_text(app)


@pytest.mark.asyncio
async def test_context_cost_label_uses_compact_copy_on_initial_narrow_mount():
    state = _cost_state(
        label="Context 45% · Current $0.48 · On next send ~+$0.13",
        compact_label="Ctx 45% · Now $0.48 · Next ~+$0.13",
    )
    app = _ChipsApp(_control_state(), cost_state=state)

    async with app.run_test(size=(80, 6)) as pilot:
        await pilot.pause()
        rendered = _line_text(app)
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
        assert "On next send" in _line_text(app)

        await pilot.resize_terminal(80, 6)
        await pilot.pause()
        assert "Ctx 45% · Now $0.48 · Next ~+$0.13" in _line_text(app)

        await pilot.resize_terminal(200, 6)
        await pilot.pause()
        assert "Context 45% · Current $0.48 · On next send ~+$0.13" in _line_text(app)


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
    app = _ChipsApp(
        _control_state(),
        cost_state=_cost_state(
            label="Context 100%+ · Current $12,345.67 · On next send ~+$987.65",
            compact_label="Ctx 100%+ · Now $12,345.67 · Next ~+$987.65",
        ),
    )
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        line = _line(app)
        scroll = app.query_one("#console-status-chip-scroll")

        rendered = _line_text(app)
        assert expected in rendered
        assert "…" not in rendered and "..." not in rendered
        assert line.region.height == 1
        assert line.region.y == scroll.content_region.y


@pytest.mark.asyncio
async def test_cost_segment_click_posts_pressed_message():
    """A real ``pilot.click`` on the cost segment must dispatch through the
    normal message pump, captured at the App via a real ``@on`` handler."""
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        spans = {
            name: start for name, start, _end in _line(app).segment_spans
        }
        clicked = await pilot.click(
            "#console-status-text", offset=(spans["cost"] + 2, 0)
        )
        await pilot.pause()
        assert clicked
        assert any(
            isinstance(message, ConsoleCostChip.ConsoleCostChipPressed)
            for message in app.captured_pressed
        )


@pytest.mark.asyncio
async def test_cost_segment_keyboard_activation_posts_pressed_message():
    app = _ChipsApp(_control_state(), cost_state=_cost_state())
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()
        _line(app).action_open_segment("cost")
        await pilot.pause()
        assert any(
            isinstance(message, ConsoleCostChip.ConsoleCostChipPressed)
            for message in app.captured_pressed
        )
