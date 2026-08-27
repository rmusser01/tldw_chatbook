"""Unit tests for the extracted Console status-chips strip."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Tooltip

from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleApprovalsChip,
    ConsoleLibraryChip,
    ConsoleStatusChips,
)


def _state(**overrides) -> ConsoleControlState:
    base = dict(
        provider_label="Provider: Anthropic",
        model_label="Model: claude-3-haiku",
        assistant_label="Assistant: General",
        rag_label="Library · Auto off · Agent blocked",
        sources_label="Sources: 0 staged",
        tools_label="Tools: 0 ready",
        approvals_label="Approvals: 0 pending",
        sources_active=False,
        tools_active=False,
        approvals_active=False,
    )
    base.update(overrides)
    return ConsoleControlState(**base)


class _ChipsApp(App):
    CSS = ".console-control-chip { width: auto; }"

    def __init__(self, state: ConsoleControlState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(self._state, id="console-status-chips")


class _ProductionChipsApp(ConsolidatedCSSApp):
    def __init__(self, state: ConsoleControlState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(self._state, id="console-status-chips")


@pytest.mark.asyncio
async def test_status_chips_render_one_assistant_identity_label():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        for selector, expected in (
            ("#console-provider-chip", "Provider:"),
            ("#console-model-chip", "Model:"),
            ("#console-assistant-chip", "Assistant: General"),
            ("#console-library-chip", "Library · Auto off · Agent blocked"),
            ("#console-sources-chip", "Sources:"),
            ("#console-tools-chip", "Tools:"),
            ("#console-approvals-chip", "Approvals:"),
        ):
            chip = app.query_one(selector)
            assert expected in str(chip.render())
        assert not app.query("#console-character-chip")
        assert not app.query("#console-persona-chip")
        assert len(app.query(ConsoleLibraryChip)) == 1
        assert not app.query("#console-rag-chip")


@pytest.mark.asyncio
async def test_library_chip_keyboard_and_click_post_the_same_open_request() -> None:
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-library-chip", ConsoleLibraryChip)
        posted: list[object] = []
        original = chip.post_message
        chip.post_message = lambda message: posted.append(message)  # type: ignore[assignment]
        try:
            chip.action_open_library_access()
            chip._on_click(object())  # type: ignore[arg-type]
        finally:
            chip.post_message = original

        assert len(posted) == 2
        assert all(isinstance(item, ConsoleLibraryChip.OpenRequested) for item in posted)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "label",
    (
        "Library · Auto off · Agent blocked",
        "Library: blocked · policy unavailable",
    ),
)
async def test_library_policy_is_first_stable_chip_and_fully_painted_at_120_columns(
    label: str,
) -> None:
    app = _ProductionChipsApp(_state(rag_label=label))
    async with app.run_test(size=(120, 6)) as pilot:
        await pilot.pause()
        scroll = app.query_one("#console-status-chip-scroll")
        library = app.query_one("#console-library-chip")
        provider = app.query_one("#console-provider-chip")

        assert library.region.x < provider.region.x
        assert library.region.right <= scroll.region.right
        assert library.region.width >= len(label) + 4
        assert str(library.render()) == label


@pytest.mark.asyncio
async def test_status_chips_sync_updates_labels_and_emphasis():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chips.sync_state(
            _state(
                model_label="Model: gpt-4o",
                sources_label="Sources: 3 staged",
                sources_active=True,
            )
        )
        await pilot.pause()
        assert "gpt-4o" in str(app.query_one("#console-model-chip").render())
        sources = app.query_one("#console-sources-chip")
        assert sources.has_class("console-chip-alert")
        assert not sources.has_class("console-chip-dim")
        # A zero counter stays dim.
        assert app.query_one("#console-tools-chip").has_class("console-chip-dim")


@pytest.mark.asyncio
async def test_tools_chip_hides_at_zero_count_and_shows_when_tools_ready():
    """TX-04 (TASK-2154.12): the tools chip must not render the lazy-loading
    detail -- it hides entirely at a zero tool count (same posture as the
    unscoped scope chip) and appears once tools are actually counted."""
    app = _ChipsApp(_state(tools_label="Tools: —", tools_active=False))
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        tools_chip = app.query_one("#console-tools-chip")
        assert tools_chip.display is False

        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chips.sync_state(_state(tools_label="Tools: 3 ready", tools_active=True))
        await pilot.pause()
        assert tools_chip.display is True
        assert "3 ready" in str(tools_chip.render())

        chips.sync_state(_state(tools_label="Tools: —", tools_active=False))
        await pilot.pause()
        assert tools_chip.display is False


@pytest.mark.asyncio
async def test_tools_chip_shown_at_compose_when_tools_ready():
    app = _ChipsApp(_state(tools_label="Tools: 10 ready", tools_active=True))
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        assert app.query_one("#console-tools-chip").display is True


@pytest.mark.asyncio
async def test_approvals_chip_posts_review_requested():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-approvals-chip", ConsoleApprovalsChip)
        posted: list[object] = []
        original_post_message = chip.post_message
        chip.post_message = lambda message: posted.append(message)  # type: ignore[assignment]
        try:
            chip.action_review_approval()
        finally:
            # Restore before teardown — Textual's prune cascade calls
            # post_message(Prune()) on exit and a swallowing stub hangs it.
            chip.post_message = original_post_message
        assert any(isinstance(m, ConsoleApprovalsChip.ReviewRequested) for m in posted)


@pytest.mark.asyncio
async def test_status_chips_sync_updates_assistant_identity_chip():
    """The assistant chip must refresh on sync, not stay at its compose value.

    Live repro: starting a chat from a character rendered "Character: none"
    forever, because the chip was painted once at compose (before any character
    existed) and `sync_state` never touched it again.
    """
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chips.sync_state(
            ConsoleControlState.from_values(
                provider="llama_cpp", model="m", character="Seraphina"
            )
        )
        await pilot.pause()

        chip = app.query_one("#console-assistant-chip")
        assert "Seraphina" in str(chip.render())


@pytest.mark.asyncio
async def test_status_chips_do_not_parse_markup_in_assistant_names():
    """A character or Persona name must never be parsed as Rich markup.

    Names are user data (imported cards included). Rendering them through a
    markup-enabled Static lets `[red]...[/]` restyle the chip strip, or raise
    MarkupError on an unbalanced tag, from nothing more than a character name.
    """
    app = _ChipsApp(
        ConsoleControlState.from_values(
            provider="llama_cpp",
            model="m",
            assistant_kind="persona",
            assistant_name="[bold]Guide[/]",
            assistant_id="persona-7",
        )
    )
    async with app.run_test(size=(200, 6)) as pilot:
        await pilot.pause()

        assistant_chip = app.query_one("#console-assistant-chip")

        assert "[bold]Guide[/]" in str(assistant_chip.render())
        assert "Persona:" in str(assistant_chip.render())


@pytest.mark.parametrize("after_sync", [False, True], ids=["compose", "sync"])
@pytest.mark.asyncio
async def test_assistant_tooltip_renders_malformed_markup_literally(after_sync):
    """Textual's separate Tooltip widget must not parse assistant names."""
    malformed_state = ConsoleControlState.from_values(
        provider="llama_cpp",
        model="m",
        assistant_kind="persona",
        assistant_name="[bold]Guide[/red]",
        assistant_id="persona-7",
    )
    app = _ChipsApp(_state() if after_sync else malformed_state)
    app.TOOLTIP_DELAY = 0.01

    async with app.run_test(size=(200, 6), tooltips=True) as pilot:
        await pilot.pause()
        if after_sync:
            app.query_one("#console-status-chips", ConsoleStatusChips).sync_state(
                malformed_state
            )
            await pilot.pause()

        assert await pilot.hover("#console-assistant-chip")
        await pilot.pause(0.05)

        tooltip = app.screen.get_child_by_type(Tooltip)
        assert tooltip.display is True
        assert "Persona: [bold]Guide[/red]" in str(tooltip.render())


class _RunChipsApp(_ChipsApp):
    """Chips app variant that passes the FB-08 run-copy constructor arg."""

    def __init__(self, state: ConsoleControlState, *, run_copy: str = "") -> None:
        super().__init__(state)
        self._run_copy = run_copy

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(
            self._state, run_copy=self._run_copy, id="console-status-chips"
        )


@pytest.mark.asyncio
async def test_run_chip_hidden_until_run_active_then_shows_and_hides():
    """FB-08 (TASK-2154.18): the run chip is the persistent visible home
    for active-run copy -- hidden at idle, "Run: {copy}" while active,
    hidden again once the run leaves an active status; repeat poll ticks
    with unchanged state are equality-guarded no-ops."""
    app = _RunChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-run-chip")
        assert chip.display is False

        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chips.sync_run_chip(True, "Streaming response.")
        await pilot.pause()
        assert chip.display is True
        assert str(chip.render()) == "Run: Streaming response."

        chips.sync_run_chip(True, "Streaming response.")
        await pilot.pause()
        assert chip.display is True

        chips.sync_run_chip(False, "")
        await pilot.pause()
        assert chip.display is False


@pytest.mark.asyncio
async def test_run_chip_renders_on_first_frame_when_run_already_active():
    """FB-08 (TASK-2154.18): returning to Console while a background run
    is still active must render the chip at compose time (the F1
    first-frame precedent), not only after a post-mount sync tick."""
    app = _RunChipsApp(_state(), run_copy="Agent running.")
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-run-chip")
        assert chip.display is True
        assert str(chip.render()) == "Run: Agent running."


@pytest.mark.asyncio
async def test_run_chip_tracks_active_run_state_via_mode_bar_sync():
    """FB-08 (TASK-2154.18) integration: with a live Console, an active
    run status renders the chip through ``_sync_console_mode_bar`` (the
    sync that runs on send/stop transitions and every 0.2s poll tick); a
    terminal status hides it again even though terminal copy lingers --
    the active-only contract (terminal outcomes toast instead)."""
    from Tests.UI.test_console_native_chat_flow import (
        _configure_native_ready_console,
    )
    from Tests.UI.test_destination_shells import (
        _build_test_app,
        _wait_for_selector,
    )
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleRunState,
        ConsoleRunStatus,
    )

    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-status-chips")
        store = console._ensure_console_chat_store()
        store.ensure_session()
        chip = console.query_one("#console-run-chip")
        assert chip.display is False

        controller = console._ensure_console_chat_controller()
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.STREAMING, "Streaming response.")
        )
        console._sync_console_mode_bar()
        await pilot.pause()
        assert chip.display is True
        assert str(chip.render()) == "Run: Streaming response."

        # Terminal statuses hide the chip even with lingering copy --
        # their ambient signal is the task-2154.16/.17 toast pair.
        controller._set_run_state(
            ConsoleRunState(ConsoleRunStatus.COMPLETED, "Response complete.")
        )
        console._sync_console_mode_bar()
        await pilot.pause()
        assert chip.display is False
