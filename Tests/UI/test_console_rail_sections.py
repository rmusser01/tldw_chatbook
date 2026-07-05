"""Console rail section header widget contracts."""

from __future__ import annotations

import random

import pytest
from textual.app import App
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    CONSOLE_READY_EMPTY_COPY,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Widgets.Console.console_rail_section import (
    CONSOLE_RAIL_SECTION_TOGGLE_PREFIX,
    ConsoleRailSectionHeader,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.Console.console_setup_modal import (
    CONSOLE_SETUP_MODAL_BACKDROP_ID,
    ConsoleSetupBackdrop,
    ConsoleSetupModal,
)
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscriptEmptyPanel
from tldw_chatbook.Widgets.Console.console_workspace_context import (
    ConsoleWorkspaceContextTray,
)
from tldw_chatbook.Widgets.Console.console_workspace_details import (
    ConsoleWorkspaceDetailsTray,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


class _HeaderApp(App):
    def compose(self):
        yield ConsoleRailSectionHeader(
            "Details",
            section_id="details",
            open=False,
            id="header-under-test",
        )


@pytest.mark.asyncio
async def test_rail_section_header_renders_title_and_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        title = app.query_one("#console-rail-section-title-details", Static)
        assert str(getattr(title.renderable, "plain", title.renderable)) == "Details"
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "+"
        assert toggle.tooltip == "Expand Details"


@pytest.mark.asyncio
async def test_rail_section_header_sync_open_flips_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        header = app.query_one("#header-under-test", ConsoleRailSectionHeader)
        header.sync_open(True)
        toggle = app.query_one(f"#{CONSOLE_RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "-"
        assert toggle.tooltip == "Collapse Details"


def _workspace_state() -> ConsoleWorkspaceContextState:
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label="Workspace: Default",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Runtime: none, file tools disabled",
        conversation_rows=(),
        conversation_empty_copy="No conversations yet.",
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=False,
        new_conversation_recovery="",
        recovery_copy="",
    )


class _DetailsApp(App):
    def compose(self):
        yield ConsoleWorkspaceDetailsTray(_workspace_state(), id="details-tray")


@pytest.mark.asyncio
async def test_details_tray_renders_status_and_handoff_rows():
    app = _DetailsApp()
    async with app.run_test(size=(60, 30)):
        assert app.query_one("#console-workspace-authority-label")
        assert app.query_one("#console-workspace-sync-label")
        assert app.query_one("#console-workspace-runtime-label")
        assert app.query_one("#console-workspace-server-readiness-label")
        assert app.query_one("#console-workspace-handoff-title")
        assert app.query_one("#console-workspace-acp-handoff-audit")


class _ContextTrayApp(App):
    def compose(self):
        yield ConsoleWorkspaceContextTray(
            _workspace_state(),
            show_heading=False,
            id="context-tray",
        )


@pytest.mark.asyncio
async def test_context_tray_without_heading_omits_status_rows():
    app = _ContextTrayApp()
    async with app.run_test(size=(60, 30)):
        assert not list(app.query("#console-workspace-context-title"))
        assert not list(app.query("#console-workspace-authority-label"))
        assert not list(app.query("#console-workspace-handoff-title"))
        assert app.query_one("#console-workspace-selected-conversation")


def _card_state() -> ConsoleSetupCardState:
    return ConsoleSetupCardState(
        mode="card",
        steps=(
            ConsoleSetupStep(state="active", label="Add an API key"),
            ConsoleSetupStep(state="done", label="Pick a model"),
            ConsoleSetupStep(
                state="pending",
                label="Send your first message",
                detail="Type below, Enter to send",
            ),
        ),
    )


class _SetupPanelApp(App):
    def __init__(self, state: ConsoleSetupCardState) -> None:
        super().__init__()
        self._state = state

    def compose(self):
        yield ConsoleTranscriptEmptyPanel(
            self._state,
            provider_action_label="Configure API",
            provider_action_tooltip="Open provider settings.",
        )


@pytest.mark.asyncio
async def test_setup_panel_card_mode_shows_quiet_line_without_steps_or_actions():
    # The numbered setup card (title + steps + primary action) moved to the
    # blocking ``ConsoleSetupModal``; while setup is incomplete the in-transcript
    # panel shows only the quiet line, dimmed under the overlay.
    app = _SetupPanelApp(_card_state())
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_QUIET_EMPTY_COPY in str(
            getattr(body.renderable, "plain", body.renderable)
        )
        assert not list(app.query("#console-setup-step-1"))
        assert app.query_one("#console-empty-title").styles.display == "none"
        assert app.query_one("#console-empty-action-row").styles.display == "none"


class _SetupModalApp(App):
    def __init__(self, state: ConsoleSetupCardState) -> None:
        super().__init__()
        self._state = state
        self.workbench_actions: list[str] = []

    def compose(self):
        modal = ConsoleSetupModal(id="console-setup-modal")
        yield modal

    async def on_mount(self) -> None:
        modal = self.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.sync_card_state(
            self._state,
            action_label="Configure API",
            action_tooltip="Open provider settings.",
        )

    def on_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
        event.stop()
        self.workbench_actions.append(event.action_id)


@pytest.mark.asyncio
async def test_setup_modal_card_mode_renders_title_steps_and_primary_action():
    app = _SetupModalApp(_card_state())
    async with app.run_test(size=(100, 30)):
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.display is True
        assert modal.is_blocking
        title = app.query_one("#console-setup-modal-title", Static)
        assert "Get started" in str(getattr(title.renderable, "plain", title.renderable))
        step1 = app.query_one("#console-setup-step-1", Static)
        assert "1. ● Add an API key" in str(getattr(step1.renderable, "plain", step1.renderable))
        step2 = app.query_one("#console-setup-step-2", Static)
        assert "2. ✓ Pick a model" in str(getattr(step2.renderable, "plain", step2.renderable))
        step3 = app.query_one("#console-setup-step-3", Static)
        text3 = str(getattr(step3.renderable, "plain", step3.renderable))
        assert "3. ○ Send your first message" in text3
        assert "Type below, Enter to send" in text3
        action = app.query_one("#console-setup-modal-action", Button)
        assert str(action.label) == "Configure API"
        # No attach/RAG controls on the modal.
        assert not list(app.query("#console-empty-attach-context"))
        assert not list(app.query("#console-empty-run-library-rag"))


@pytest.mark.asyncio
async def test_setup_modal_primary_action_routes_provider_recovery():
    app = _SetupModalApp(_card_state())
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.click("#console-setup-modal-action")
        await pilot.pause()
        assert app.workbench_actions == ["provider-recovery"]


@pytest.mark.asyncio
async def test_setup_modal_hides_when_state_leaves_card_mode():
    app = _SetupModalApp(_card_state())
    async with app.run_test(size=(100, 30)) as pilot:
        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        assert modal.display is True
        modal.sync_card_state(
            ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY),
            action_label="Choose model",
            action_tooltip="Pick a model.",
        )
        await pilot.pause()
        assert modal.display is False
        assert modal.is_blocking is False


@pytest.mark.asyncio
async def test_setup_panel_ready_line_hides_steps_and_actions():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_READY_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
        assert not list(app.query("#console-setup-step-1"))
        assert app.query_one("#console-empty-action-row").styles.display == "none"
        assert app.query_one("#console-empty-title").styles.display == "none"


@pytest.mark.asyncio
async def test_setup_panel_quiet_mode_shows_only_quiet_copy():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_QUIET_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
        assert not list(app.query("#console-setup-step-1"))
        assert app.query_one("#console-empty-action-row").styles.display == "none"


@pytest.mark.asyncio
async def test_setup_panel_sync_card_state_transitions_modes():
    app = _SetupPanelApp(_card_state())
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(ConsoleTranscriptEmptyPanel)
        panel.sync_card_state(
            ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY),
            provider_action_label="Choose model",
            provider_action_tooltip="Pick a model.",
        )
        await pilot.pause()
        assert not list(app.query("#console-setup-step-1"))
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_READY_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))


@pytest.mark.asyncio
async def test_setup_panel_coerces_non_card_state_to_quiet_copy():
    # Regression guard: a flaky resume race can transiently hand the panel a
    # bare value instead of a ``ConsoleSetupCardState``. It must not raise and
    # should fall back to rendering the quiet empty-state copy.
    app = _SetupPanelApp("not-a-card-state")
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_QUIET_EMPTY_COPY in str(getattr(body.renderable, "plain", body.renderable))
        assert not list(app.query("#console-setup-step-1"))
        assert app.query_one("#console-empty-action-row").styles.display == "none"


# ---------------------------------------------------------------------------
# Setup-modal snow backdrop (ZSNES-style falling glyphs behind the card).
# ---------------------------------------------------------------------------

_SNOW_GLYPHS = ("·", "•", "*")


def _snow_glyph_count(text: str) -> int:
    return sum(text.count(glyph) for glyph in _SNOW_GLYPHS)


class _SnowBackdropApp(App):
    def __init__(self, rng: random.Random) -> None:
        super().__init__()
        self._rng = rng

    def compose(self):
        yield ConsoleSetupBackdrop(id="backdrop-under-test", rng=self._rng)


@pytest.mark.asyncio
async def test_setup_backdrop_seeded_rng_renders_flake_glyphs():
    # Seeded rng + fixed size => fully deterministic flake field: 40x10 cells
    # at ~1 flake per 40 cells yields exactly 10 non-overlapping flakes.
    app = _SnowBackdropApp(random.Random(42))
    async with app.run_test(size=(40, 10)):
        backdrop = app.query_one("#backdrop-under-test", ConsoleSetupBackdrop)
        assert backdrop.flake_count == 10
        text = str(backdrop.renderable)
        assert _snow_glyph_count(text) >= 5


@pytest.mark.asyncio
async def test_setup_backdrop_tick_advances_positions_and_repaints():
    app = _SnowBackdropApp(random.Random(42))
    async with app.run_test(size=(40, 10)):
        backdrop = app.query_one("#backdrop-under-test", ConsoleSetupBackdrop)
        positions_before = [(flake.x, flake.y) for flake in backdrop._flakes]
        text_before = str(backdrop.renderable)

        backdrop._tick()

        positions_after = [(flake.x, flake.y) for flake in backdrop._flakes]
        text_after = str(backdrop.renderable)
        assert positions_after != positions_before
        assert text_after != text_before


@pytest.mark.asyncio
async def test_setup_backdrop_tick_wraps_flake_past_bottom_to_top():
    app = _SnowBackdropApp(random.Random(42))
    async with app.run_test(size=(40, 10)):
        backdrop = app.query_one("#backdrop-under-test", ConsoleSetupBackdrop)
        flake = backdrop._flakes[0]
        flake.y = backdrop._field_height - 0.05
        flake.speed = 1.0

        backdrop._tick()

        assert flake.y == 0.0


@pytest.mark.asyncio
async def test_setup_backdrop_resize_safe_at_tiny_size():
    app = _SnowBackdropApp(random.Random(42))
    async with app.run_test(size=(40, 10)) as pilot:
        backdrop = app.query_one("#backdrop-under-test", ConsoleSetupBackdrop)
        await pilot.resize_terminal(1, 1)
        await pilot.pause()
        assert backdrop.flake_count >= 1
        # Must not raise even at the smallest possible field.
        backdrop._tick()
        await pilot.resize_terminal(40, 10)
        await pilot.pause()
        assert backdrop.flake_count == 10


@pytest.mark.asyncio
async def test_setup_modal_snow_timer_paused_until_blocking():
    app = _SetupModalApp(_card_state())
    async with app.run_test(size=(100, 30)) as pilot:
        backdrop = app.query_one(
            f"#{CONSOLE_SETUP_MODAL_BACKDROP_ID}", ConsoleSetupBackdrop
        )
        # _SetupModalApp.on_mount() immediately syncs card-mode (blocking).
        assert backdrop.timer_paused is False

        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.sync_card_state(
            ConsoleSetupCardState(mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY),
            action_label="Choose model",
            action_tooltip="Pick a model.",
        )
        await pilot.pause()
        assert backdrop.timer_paused is True

        modal.sync_card_state(
            _card_state(),
            action_label="Configure API",
            action_tooltip="Open provider settings.",
        )
        await pilot.pause()
        assert backdrop.timer_paused is False


@pytest.mark.asyncio
async def test_setup_backdrop_resume_before_mount_starts_timer_running():
    # Regression: resume_snow() called before on_mount() creates the interval
    # timer used to be a lost intent -- on_mount() unconditionally created the
    # timer paused, so a resume() issued against the not-yet-existing timer
    # never took effect. The widget must remember the intent and apply it once
    # the timer exists.
    backdrop = ConsoleSetupBackdrop(rng=random.Random(42))
    backdrop.resume_snow()

    class _ResumeBeforeMountApp(App):
        def compose(self):
            yield backdrop

    app = _ResumeBeforeMountApp()
    async with app.run_test(size=(40, 10)):
        assert backdrop._snow_timer is not None
        assert backdrop._snow_timer._active.is_set() is True
        assert backdrop.timer_paused is False


@pytest.mark.asyncio
async def test_setup_backdrop_no_resume_intent_stays_paused_after_mount():
    backdrop = ConsoleSetupBackdrop(rng=random.Random(42))

    class _NoResumeApp(App):
        def compose(self):
            yield backdrop

    app = _NoResumeApp()
    async with app.run_test(size=(40, 10)):
        assert backdrop._snow_timer is not None
        assert backdrop._snow_timer._active.is_set() is False
        assert backdrop.timer_paused is True


# ---------------------------------------------------------------------------
# Console session switcher modal (Ctrl+K).
# ---------------------------------------------------------------------------

from tldw_chatbook.Chat.console_switcher_state import ConsoleSwitcherEntry
from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)


def _switcher_rows() -> tuple[ConsoleConversationBrowserInputRow, ...]:
    def row(key, title, native=None, **kw):
        return ConsoleConversationBrowserInputRow(
            row_key=key, conversation_id=None if native else key,
            native_session_id=native, title=title, scope_type="workspace",
            workspace_id="ws-1", workspace_label="Workspace 1",
            updated_sort="2026-07-04T10:00:00+00:00", **kw,
        )
    return (
        row("native-1", "Groq testing", native="sess-1", selected=True),
        row("conv-2", "API refactor plan"),
        row("conv-3", "Tides explainer"),
    )


class _SwitcherApp(App):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    async def on_mount(self) -> None:
        def _capture(choice):
            self.result = choice
        await self.push_screen(
            ConsoleSessionSwitcherModal(rows=_switcher_rows()), callback=_capture
        )


@pytest.mark.asyncio
async def test_switcher_lists_recent_first_and_filters_on_typing():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        first = app.screen.query_one("#console-switcher-result-0", Button)
        assert "Groq testing" in str(first.label)
        await pilot.click("#console-switcher-query")
        await pilot.press(*"refactor")
        await pilot.pause()
        first = app.screen.query_one("#console-switcher-result-0", Button)
        assert "API refactor plan" in str(first.label)
        assert not list(app.screen.query("#console-switcher-result-1"))


@pytest.mark.asyncio
async def test_switcher_enter_activates_first_result():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-switcher-query")
        await pilot.press(*"tides")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSwitcherChoice)
        assert app.result.kind == "activate"
        assert app.result.entry.title == "Tides explainer"


@pytest.mark.asyncio
async def test_switcher_f2_requests_rename_for_native_entry():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSwitcherChoice)
        assert app.result.kind == "rename"
        assert app.result.entry.native_session_id == "sess-1"


@pytest.mark.asyncio
async def test_switcher_escape_dismisses_none_and_empty_query_shows_no_matches():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-switcher-query")
        await pilot.press(*"zzzz")
        await pilot.pause()
        assert list(app.screen.query("#console-switcher-empty"))
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is None
