"""Console rail section header widget contracts."""

from __future__ import annotations

import asyncio
import random
from dataclasses import replace

import pytest
from textual.app import App

# Harness apps load the consolidated widget CSS the real app loads
# (TASK-15450); without it the widgets under test mount unstyled.
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    CONSOLE_READY_EMPTY_COPY,
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsReadiness,
    ConsoleSettingsSummaryState,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsCommittedSubmission,
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsLiveCommit,
    ConsoleSettingsOrigin,
    ConsoleSettingsSubmission,
    ConsoleSettingsTransfer,
)
from tldw_chatbook.Widgets.Console.console_model_popover import (
    ConsoleModelPopover,
)
from tldw_chatbook.Widgets.Console.console_settings_summary import (
    ConsoleSettingsSummary,
    build_console_readiness_presentation,
)
from tldw_chatbook.Widgets.destination_rail import (
    RAIL_SECTION_TOGGLE_PREFIX,
    DestinationRailSectionHeader,
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
from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker
from tldw_chatbook.Workspaces.conversation_browser_state import (
    CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT,
    ConsoleConversationBrowserInputRow,
    build_console_conversation_browser_state,
)
from tldw_chatbook.Workspaces.display_state import ConsoleWorkspaceContextState


class _HeaderApp(ConsolidatedCSSApp):
    def compose(self):
        yield DestinationRailSectionHeader(
            "Details",
            section_id="details",
            open=False,
            id="header-under-test",
        )


def _typed_unreachable_readiness() -> ConsoleSettingsReadiness:
    return ConsoleSettingsReadiness(
        label="READY legacy poison",
        detail="PRIVATE http://127.0.0.1:9876 exception",
        native_send_supported=False,
        operability="not_ready",
        blocker="endpoint_unreachable",
        recovery_action="retry_connection",
        provider_display_name="Ollama",
        configuration="configured",
        credential="not_required",
        endpoint="unreachable",
        endpoint_category="timeout",
        model="unconfirmed",
        generation="failed",
        generation_category="timeout",
    )


class _ReadinessSummaryApp(ConsolidatedCSSApp):
    def compose(self):
        yield ConsoleSettingsSummary(
            ConsoleSettingsSummaryState(
                provider_row="Provider: Ollama",
                model_row="Model: llama3",
                context_row="Context: unavailable",
                sampling_row="Sampling: T 0.70, P 0.95",
                identity_row="Assistant: General",
                readiness_label="READY legacy poison",
                readiness=_typed_unreachable_readiness(),
            )
        )


@pytest.mark.asyncio
async def test_console_rail_summary_renders_typed_operability_and_verification_evidence():
    app = _ReadinessSummaryApp()
    async with app.run_test(size=(72, 22)) as pilot:
        await pilot.pause()
        text = "\n".join(
            str(getattr(item.renderable, "plain", item.renderable))
            for item in app.query(Static)
        )
        assert "Not ready — endpoint unreachable" in text
        assert "Endpoint · Unreachable — timed out" in text
        assert "Generation · Failed — timed out" in text
        assert "Retry connection" in str(
            app.query_one("#console-settings-open", Button).label
        )
        assert "PRIVATE" not in text


@pytest.mark.asyncio
async def test_console_rail_summary_uses_canonical_provider_and_honest_empty_copy():
    """Raw provider keys and unavailable estimates must not leak into the rail."""
    readiness = ConsoleSettingsReadiness(
        label="Ready",
        detail="Ready",
        native_send_supported=True,
        operability="ready_to_send",
        provider_display_name="llama.cpp",
        configuration="configured",
        credential="not_required",
        endpoint="not_tested",
        model="unconfirmed",
        generation="not_tested",
    )
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: llama_cpp",
        model_row="Model: model-a",
        context_row="Context: unavailable",
        endpoint_row="Endpoint: provider default",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Assistant: General",
        readiness=readiness,
    )
    class _HonestCopyApp(ConsolidatedCSSApp):
        def compose(self):
            yield ConsoleSettingsSummary(state)

    app = _HonestCopyApp()

    async with app.run_test(size=(72, 22)):
        provider = app.query_one("#console-settings-provider-row", Static)
        context = app.query_one("#console-settings-context-row", Static)
        endpoint = app.query_one("#console-settings-endpoint-row", Static)
        assert str(provider.renderable) == "Provider: llama.cpp"
        assert str(context.renderable) == "Context: Not estimated"
        assert str(endpoint.renderable) == "Endpoint · Not tested"


@pytest.mark.asyncio
async def test_console_rail_summary_labels_genuine_provider_default_inheritance():
    """An inherited endpoint must be named as a provider default, not raw copy."""
    state = ConsoleSettingsSummaryState(
        provider_row="Provider: OpenAI",
        model_row="Model: gpt-5.6-terra",
        context_row="Context: 10 / 4k",
        endpoint_row="Endpoint: provider default",
        sampling_row="Sampling: T 0.70, P 0.95",
        identity_row="Assistant: General",
    )

    class _ProviderDefaultApp(ConsolidatedCSSApp):
        def compose(self):
            yield ConsoleSettingsSummary(state)

    app = _ProviderDefaultApp()
    async with app.run_test(size=(72, 22)) as pilot:
        endpoint = app.query_one("#console-settings-endpoint-row", Static)
        assert str(endpoint.renderable) == "Endpoint: Provider default"

        app.query_one(ConsoleSettingsSummary).sync_state(
            ConsoleSettingsSummaryState(
                provider_row="Provider: Anthropic",
                model_row="Model: claude-sonnet",
                context_row="Context: 20 / 8k",
                endpoint_row="Endpoint: provider default",
                sampling_row="Sampling: T 0.50, P 0.90",
                identity_row="Assistant: General",
            )
        )
        await pilot.pause()
        assert str(endpoint.renderable) == "Endpoint: Provider default"


@pytest.mark.parametrize(
    ("readiness", "expected_rows"),
    (
        (
            ConsoleSettingsReadiness(
                "ignored",
                "ignored",
                True,
                operability="ready_to_send",
                provider_display_name="OpenAI",
                configuration="configured",
                credential="authenticated",
                credential_source="stored",
                endpoint="reachable",
                model="confirmed",
                generation="succeeded",
            ),
            (
                "Credential · Authenticated",
                "Endpoint · Reachable",
                "Model · Confirmed",
                "Generation · Succeeded",
            ),
        ),
        (
            ConsoleSettingsReadiness(
                "ignored",
                "ignored",
                True,
                operability="ready_to_send",
                provider_display_name="Ollama",
                configuration="configured",
                credential="not_required",
                endpoint="model_listing_unavailable",
                model="unconfirmed",
                generation="not_tested",
            ),
            (
                "Credential · Not required",
                "Endpoint · Reachable — model listing unavailable",
                "Model · Listing unavailable",
                "Generation · Not tested",
            ),
        ),
        (
            ConsoleSettingsReadiness(
                "ignored",
                "ignored",
                True,
                operability="ready_to_send",
                provider_display_name="Ollama",
                configuration="configured",
                credential="not_required",
                endpoint="changed_since_test",
                model="unconfirmed",
                generation="changed_since_test",
            ),
            (
                "Credential · Not required",
                "Endpoint · Changed since test",
                "Model · Changed since test",
                "Generation · Changed since test",
            ),
        ),
    ),
)
def test_console_verification_evidence_rows_are_independent_of_operability(
    readiness,
    expected_rows,
) -> None:
    presentation = build_console_readiness_presentation(readiness)
    assert (
        presentation.credential_row,
        presentation.endpoint_row,
        presentation.model_row,
        presentation.generation_row,
    ) == expected_rows


@pytest.mark.asyncio
async def test_rail_section_header_renders_title_and_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        title = app.query_one("#console-rail-section-title-details", Static)
        assert str(getattr(title.renderable, "plain", title.renderable)) == "Details"
        toggle = app.query_one(f"#{RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "▸"
        assert toggle.tooltip == "Expand Details"


@pytest.mark.asyncio
async def test_rail_section_header_sync_open_flips_toggle():
    app = _HeaderApp()
    async with app.run_test(size=(60, 10)):
        header = app.query_one("#header-under-test", DestinationRailSectionHeader)
        header.sync_open(True)
        toggle = app.query_one(f"#{RAIL_SECTION_TOGGLE_PREFIX}details", Button)
        assert str(toggle.label) == "▾"
        assert toggle.tooltip == "Collapse Details"


def test_section_header_allows_border_height():
    header = DestinationRailSectionHeader("Session", section_id="session", open=True)
    # Inline height constraints should be gone so CSS can set min-height 2.
    assert header.styles.height is None or header.styles.height.value != 1
    assert header.styles.max_height is None


def test_console_glyph_constants():
    from tldw_chatbook.Chat.console_glyphs import (
        GLYPH_ACTIVE,
        GLYPH_CLOSE,
        GLYPH_COLLAPSED,
        GLYPH_COLLAPSE_LEFT,
        GLYPH_DONE,
        GLYPH_EXPANDED,
        GLYPH_IN_PROGRESS,
    )

    assert (GLYPH_EXPANDED, GLYPH_COLLAPSED) == ("▾", "▸")
    assert (GLYPH_ACTIVE, GLYPH_IN_PROGRESS, GLYPH_DONE) == ("▸", "●", "✓")
    assert (GLYPH_CLOSE, GLYPH_COLLAPSE_LEFT) == ("✕", "◂")


def test_console_active_row_marker_and_close_glyphs():
    from tldw_chatbook.Widgets.Console import (
        console_workspace_context,
        console_session_surface,
    )
    import inspect

    assert '"> "' not in inspect.getsource(console_workspace_context)
    assert '"x"' not in inspect.getsource(console_session_surface)


def _workspace_state() -> ConsoleWorkspaceContextState:
    # TASK-1190: production always attaches a real (possibly empty) grouped
    # conversation browser -- the transitional legacy compose path (taken
    # only when conversation_browser is None) was retired, so this fixture
    # carries an empty browser to match the one real production shape.
    return ConsoleWorkspaceContextState(
        heading="Convos & Workspaces",
        workspace_label="Workspace: Default",
        authority_label="Authority: local registry ready",
        sync_label="Sync: not configured",
        runtime_label="Local file tools: Private scratch",
        conversation_rows=(),
        conversation_empty_copy="No conversations yet.",
        conversation_browser=build_console_conversation_browser_state(
            rows=(), active_workspace_id=None
        ),
        change_workspace_enabled=False,
        change_workspace_recovery="",
        new_conversation_enabled=False,
        new_conversation_recovery="",
        recovery_copy="",
    )


class _DetailsApp(ConsolidatedCSSApp):
    def compose(self):
        yield ConsoleWorkspaceDetailsTray(_workspace_state(), id="details-tray")


@pytest.mark.asyncio
async def test_details_tray_renders_status_and_handoff_rows():
    app = _DetailsApp()
    async with app.run_test(size=(60, 30)):
        assert app.query_one("#console-workspace-authority-label")
        assert app.query_one("#console-workspace-runtime-label")
        assert app.query_one("#console-workspace-handoff-title")
        # TASK-715: sync/server/ACP rows are factory defaults here, so they
        # collapse into a single plain not-configured line.
        assert app.query_one("#console-workspace-server-features-collapsed")
        assert not list(app.query("#console-workspace-sync-label"))
        assert not list(app.query("#console-workspace-server-readiness-label"))
        assert not list(app.query("#console-workspace-acp-handoff-audit"))


class _ContextTrayApp(ConsolidatedCSSApp):
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


class _SetupPanelApp(ConsolidatedCSSApp):
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
        assert not list(app.query("#console-empty-title"))
        assert not list(app.query("#console-empty-action-row"))


@pytest.mark.asyncio
async def test_empty_panel_has_no_legacy_shim_widgets():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        assert not list(app.query("#console-empty-title"))
        assert not list(app.query("#console-empty-action-row"))
        assert not list(app.query("#console-empty-choose-model"))
        assert list(app.query("#console-empty-body"))


class _SetupModalApp(ConsolidatedCSSApp):
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
        assert "Get started" in str(
            getattr(title.renderable, "plain", title.renderable)
        )
        step1 = app.query_one("#console-setup-step-1", Static)
        assert "1. ● Add an API key" in str(
            getattr(step1.renderable, "plain", step1.renderable)
        )
        step2 = app.query_one("#console-setup-step-2", Static)
        assert "2. ✓ Pick a model" in str(
            getattr(step2.renderable, "plain", step2.renderable)
        )
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
            ConsoleSetupCardState(
                mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY
            ),
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
        assert CONSOLE_READY_EMPTY_COPY in str(
            getattr(body.renderable, "plain", body.renderable)
        )
        assert not list(app.query("#console-setup-step-1"))
        assert not list(app.query("#console-empty-action-row"))
        assert not list(app.query("#console-empty-title"))


@pytest.mark.asyncio
async def test_setup_panel_quiet_mode_shows_only_quiet_copy():
    app = _SetupPanelApp(
        ConsoleSetupCardState(mode="quiet", body_copy=CONSOLE_QUIET_EMPTY_COPY)
    )
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_QUIET_EMPTY_COPY in str(
            getattr(body.renderable, "plain", body.renderable)
        )
        assert not list(app.query("#console-setup-step-1"))
        assert not list(app.query("#console-empty-action-row"))


@pytest.mark.asyncio
async def test_setup_panel_sync_card_state_transitions_modes():
    app = _SetupPanelApp(_card_state())
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one(ConsoleTranscriptEmptyPanel)
        panel.sync_card_state(
            ConsoleSetupCardState(
                mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY
            ),
            provider_action_label="Choose model",
            provider_action_tooltip="Pick a model.",
        )
        await pilot.pause()
        assert not list(app.query("#console-setup-step-1"))
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_READY_EMPTY_COPY in str(
            getattr(body.renderable, "plain", body.renderable)
        )


@pytest.mark.asyncio
async def test_setup_panel_coerces_non_card_state_to_quiet_copy():
    # Regression guard: a flaky resume race can transiently hand the panel a
    # bare value instead of a ``ConsoleSetupCardState``. It must not raise and
    # should fall back to rendering the quiet empty-state copy.
    app = _SetupPanelApp("not-a-card-state")
    async with app.run_test(size=(100, 30)):
        body = app.query_one("#console-empty-body", Static)
        assert CONSOLE_QUIET_EMPTY_COPY in str(
            getattr(body.renderable, "plain", body.renderable)
        )
        assert not list(app.query("#console-setup-step-1"))
        assert not list(app.query("#console-empty-action-row"))


# ---------------------------------------------------------------------------
# Setup-modal snow backdrop (ZSNES-style falling glyphs behind the card).
# ---------------------------------------------------------------------------

_SNOW_GLYPHS = ("·", "•", "*")


def _snow_glyph_count(text: str) -> int:
    return sum(text.count(glyph) for glyph in _SNOW_GLYPHS)


class _SnowBackdropApp(ConsolidatedCSSApp):
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
async def test_setup_backdrop_field_is_still_between_resizes():
    """TASK-23021: the snow is a still frame -- positions and rendered text
    must not change while the widget merely sits mounted."""
    app = _SnowBackdropApp(random.Random(42))
    async with app.run_test(size=(40, 10)):
        backdrop = app.query_one("#backdrop-under-test", ConsoleSetupBackdrop)
        positions_before = [(flake.x, flake.y) for flake in backdrop._flakes]
        text_before = str(backdrop.renderable)

        # Longer than several of the retired animation's 0.4 s intervals.
        await asyncio.sleep(1.0)

        positions_after = [(flake.x, flake.y) for flake in backdrop._flakes]
        text_after = str(backdrop.renderable)
        assert positions_after == positions_before
        assert text_after == text_before


@pytest.mark.asyncio
async def test_setup_backdrop_resize_safe_at_tiny_size():
    app = _SnowBackdropApp(random.Random(42))
    async with app.run_test(size=(40, 10)) as pilot:
        backdrop = app.query_one("#backdrop-under-test", ConsoleSetupBackdrop)
        await pilot.resize_terminal(1, 1)
        await pilot.pause()
        assert backdrop.flake_count >= 1
        await pilot.resize_terminal(40, 10)
        await pilot.pause()
        assert backdrop.flake_count == 10


@pytest.mark.asyncio
async def test_setup_modal_backdrop_never_arms_a_timer_in_any_block_state():
    """TASK-23021 retired the snow tick outright: blocking, unblocked, and
    re-blocked states must all leave the backdrop with zero timers (the old
    contract paused/resumed a real interval timer across these transitions)."""
    app = _SetupModalApp(_card_state())
    async with app.run_test(size=(100, 30)) as pilot:
        backdrop = app.query_one(
            f"#{CONSOLE_SETUP_MODAL_BACKDROP_ID}", ConsoleSetupBackdrop
        )
        # _SetupModalApp.on_mount() immediately syncs card-mode (blocking).
        assert len(backdrop._timers) == 0

        modal = app.query_one("#console-setup-modal", ConsoleSetupModal)
        modal.sync_card_state(
            ConsoleSetupCardState(
                mode="ready_line", body_copy=CONSOLE_READY_EMPTY_COPY
            ),
            action_label="Choose model",
            action_tooltip="Pick a model.",
        )
        await pilot.pause()
        assert len(backdrop._timers) == 0

        modal.sync_card_state(
            _card_state(),
            action_label="Configure API",
            action_tooltip="Open provider settings.",
        )
        await pilot.pause()
        assert len(backdrop._timers) == 0


# ---------------------------------------------------------------------------
# Console session switcher modal (Ctrl+K).
# ---------------------------------------------------------------------------

from tldw_chatbook.Widgets.Console.console_session_switcher_modal import (  # noqa: E402
    SEARCH_DEBOUNCE_SECONDS,
    ConsoleSessionSwitcherModal,
    ConsoleSwitcherChoice,
)


def _switcher_rows() -> tuple[ConsoleConversationBrowserInputRow, ...]:
    def row(key, title, native=None, **kw):
        return ConsoleConversationBrowserInputRow(
            row_key=key,
            conversation_id=None if native else key,
            native_session_id=native,
            title=title,
            scope_type="workspace",
            workspace_id="ws-1",
            workspace_label="Workspace 1",
            updated_sort="2026-07-04T10:00:00+00:00",
            **kw,
        )

    return (
        row("native-1", "Groq testing", native="sess-1", selected=True),
        row("conv-2", "API refactor plan"),
        row("conv-3", "Tides explainer"),
    )


class _SwitcherApp(ConsolidatedCSSApp):
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
        # Debounced (task-15476): the result list only re-renders once the
        # filter settles, not on every keystroke.
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        first = app.screen.query_one("#console-switcher-result-0", Button)
        assert "API refactor plan" in str(first.label)
        assert not list(app.screen.query("#console-switcher-result-1"))


@pytest.mark.asyncio
async def test_switcher_title_cannot_add_a_forged_result_line() -> None:
    raw_title = "Chat with Nyx\n\tAdmin\x00[/bold]"
    row = ConsoleConversationBrowserInputRow(
        row_key="native-unsafe",
        conversation_id=None,
        native_session_id="session-unsafe",
        title=raw_title,
        scope_type="global",
        workspace_id=None,
        workspace_label="Chats",
        updated_sort="2026-07-04T10:00:00+00:00",
    )

    class _UnsafeSwitcherApp(App):
        async def on_mount(self) -> None:
            await self.push_screen(ConsoleSessionSwitcherModal(rows=(row,)))

    app = _UnsafeSwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        result = app.screen.query_one("#console-switcher-result-0", Button)
        rendered = str(result.label)
        assert rendered.count("\n") == 1  # only the intentional subtitle line
        assert "\t" not in rendered
        assert "Chat with Nyx Admin?[/bold]" in rendered
        assert app.screen._entries[0].title == raw_title


@pytest.mark.asyncio
async def test_switcher_enter_activates_first_result():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-switcher-query")
        await pilot.press(*"tides")
        # Debounced (task-15476): let the filter settle before Enter, or it
        # would activate the still-unfiltered first result instead.
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSwitcherChoice)
        assert app.result.kind == "activate"
        assert app.result.entry.title == "Tides explainer"


@pytest.mark.asyncio
async def test_switcher_f2_does_not_fall_back_from_search_to_native_entry():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.press("f2")
        await pilot.pause()
        assert app.result == "unset"
        feedback = app.screen.query_one("#console-switcher-feedback", Static)
        assert "focus an open agent result" in str(feedback.renderable).lower()


def _two_native_switcher_rows() -> tuple[ConsoleConversationBrowserInputRow, ...]:
    def row(key, title, native, **kw):
        return ConsoleConversationBrowserInputRow(
            row_key=key,
            conversation_id=None,
            native_session_id=native,
            title=title,
            scope_type="workspace",
            workspace_id="ws-1",
            workspace_label="Workspace 1",
            updated_sort="2026-07-04T10:00:00+00:00",
            **kw,
        )

    return (
        row("native-1", "Groq testing", "sess-1", selected=True),
        row("native-2", "Claude testing", "sess-2"),
    )


class _TwoNativeSwitcherApp(ConsolidatedCSSApp):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    async def on_mount(self) -> None:
        def _capture(choice):
            self.result = choice

        await self.push_screen(
            ConsoleSessionSwitcherModal(rows=_two_native_switcher_rows()),
            callback=_capture,
        )


@pytest.mark.asyncio
async def test_switcher_f2_renames_focused_result_not_always_first():
    app = _TwoNativeSwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        second_button = app.screen.query_one("#console-switcher-result-1", Button)
        second_button.focus()
        await pilot.pause()
        await pilot.press("f2")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSwitcherChoice)
        assert app.result.kind == "rename"
        assert app.result.entry.native_session_id == "sess-2"


@pytest.mark.asyncio
async def test_switcher_escape_dismisses_none_and_empty_query_shows_no_matches():
    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-switcher-query")
        await pilot.press(*"zzzz")
        # Debounced (task-15476): the empty state only appears once the
        # filter settles.
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        assert list(app.screen.query("#console-switcher-empty"))
        await pilot.press("escape")
        await pilot.pause()
        assert app.result is None


@pytest.mark.asyncio
async def test_switcher_rapid_refresh_does_not_duplicate_ids():
    from textual.widgets import Input

    app = _SwitcherApp()
    async with app.run_test(size=(90, 30)) as pilot:
        # Two back-to-back Input.Changed posts with no settling between them —
        # simulates paste/fast typing faster than pilot.press's per-key
        # wait_for_idle can produce.
        query_input = app.screen.query_one("#console-switcher-query", Input)
        query_input.value = "r"
        query_input.value = "refactor"
        # Debounced (task-15476): the second Input.Changed re-arms the timer
        # and cancels the first, so only "refactor" is ever applied.
        await pilot.pause(SEARCH_DEBOUNCE_SECONDS + 0.1)
        first = app.screen.query_one("#console-switcher-result-0", Button)
        assert "API refactor plan" in str(first.label)
        assert not list(app.screen.query("#console-switcher-result-1"))


_POPOVER_PROVIDERS = {"llama_cpp": ["model-a", "model-b"], "openai": ["gpt-4o"]}


def _test_popover(
    settings: ConsoleSessionSettings,
    providers_models,
) -> ConsoleModelPopover:
    origin = ConsoleSettingsOrigin("popover-session", None, 0)
    draft = ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        field_drafts=tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in ("temperature", "streaming")
        ),
        model_drafts=(),
        endpoint_draft=None,
    )

    def rebase(state, **kwargs):
        return replace(
            state,
            settings=replace(
                state.settings,
                provider=kwargs["provider"],
                model=kwargs["model"],
            ),
        )

    def commit(submission: ConsoleSettingsSubmission) -> ConsoleSettingsLiveCommit:
        return ConsoleSettingsLiveCommit(
            submission_id=submission.submission_id,
            session_id=origin.session_id,
            persisted_conversation_id=None,
            conversation_binding_revision=0,
            generation_revision=1,
            context_policy_revision=1,
            settings=submission.draft.settings,
            context_policy_overrides=submission.draft.context_policy_overrides,
        )

    return ConsoleModelPopover(
        origin=origin,
        app_config={
            "api_settings": {
                "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
                "openai": {"api_key": "test-key"},
                "openrouter": {"api_key": "test-key"},
            }
        },
        initial_draft=draft,
        providers_models=providers_models,
        scope_copy="Applies to this conversation",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=rebase,
        live_committer=commit,
        default_readiness_resolver=lambda _provider, _model: ConsoleSettingsReadiness(
            "Ready", "Ready.", True
        ),
    )


class _PopoverApp(ConsolidatedCSSApp):
    def __init__(self):
        super().__init__()
        self.result = "unset"

    async def on_mount(self) -> None:
        settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")

        def _capture(result):
            self.result = result

        await self.push_screen(
            _test_popover(settings, _POPOVER_PROVIDERS),
            callback=_capture,
        )


@pytest.mark.asyncio
async def test_popover_apply_returns_replaced_settings():
    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#model-search-picker-input")
        await pilot.press(*"model-b", "enter")
        await pilot.pause()
        streaming = app.screen.query_one("#console-popover-streaming", Button)
        streaming.scroll_visible(animate=False, force=True)
        await pilot.pause()
        assert await pilot.click("#console-popover-streaming") is True
        await pilot.pause()
        await pilot.click("#console-popover-apply")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
        committed = app.result.live_commit.settings
        assert committed.model == "model-b"
        assert committed.provider == "llama_cpp"
        # ConsoleSessionSettings defaults streaming True; one toggle flips it.
        assert committed.streaming is False


@pytest.mark.asyncio
async def test_popover_full_settings_returns_sentinel_and_escape_cancels():
    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#console-popover-full-settings")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSettingsTransfer)
        assert app.result.origin.session_id == "popover-session"
    app2 = _PopoverApp()
    async with app2.run_test(size=(90, 30)) as pilot:
        await pilot.press("escape")
        await pilot.pause()
        assert app2.result is None


@pytest.mark.asyncio
async def test_popover_apply_with_blank_temperature_clears_it():
    from textual.widgets import Input

    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        temperature_input = app.screen.query_one("#console-popover-temperature", Input)
        temperature_input.value = ""
        await pilot.click("#console-popover-apply")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
        assert app.result.live_commit.settings.temperature is None


@pytest.mark.parametrize("invalid_text", ["nan", "5.5", "-1"])
@pytest.mark.asyncio
async def test_popover_apply_rejects_nan_and_out_of_range_temperature(invalid_text):
    from textual.widgets import Input

    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        temperature_input = app.screen.query_one("#console-popover-temperature", Input)
        temperature_input.value = invalid_text
        await pilot.click("#console-popover-apply")
        await pilot.pause()
        assert app.result == "unset"


@pytest.mark.asyncio
async def test_popover_apply_accepts_in_range_temperature():
    from textual.widgets import Input

    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        temperature_input = app.screen.query_one("#console-popover-temperature", Input)
        temperature_input.value = "1.2"
        await pilot.click("#console-popover-apply")
        await pilot.pause()
        assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
        assert app.result.live_commit.settings.temperature == 1.2


class _PopoverSearchScope:
    """Minimal llm_provider_catalog_scope_service stand-in for search tests."""

    def __init__(self, entries):
        self._entries = entries

    async def merge_saved_and_discovered_models(self, *, mode, provider):
        return self._entries


_POPOVER_SEARCH_PROVIDERS = {"openrouter": ["saved-model"]}
_POPOVER_SEARCH_MODEL_IDS = ["anthropic/claude-x", "openai/gpt-y"]


def _popover_search_entries():
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
        MergedModelEntry,
    )

    return tuple(
        MergedModelEntry(
            provider="openrouter",
            provider_list_key="openrouter",
            model_id=m,
            display_name=m,
            source="runtime_discovered",
            capability_status="unknown",
            persisted=False,
        )
        for m in _POPOVER_SEARCH_MODEL_IDS
    )


class _PopoverSearchApp(ConsolidatedCSSApp):
    """Popover host app exposing the catalog scope the search picker reads."""

    def __init__(self):
        super().__init__()
        self.result = "unset"
        self.providers_models = _POPOVER_SEARCH_PROVIDERS
        self.llm_provider_catalog_scope_service = _PopoverSearchScope(
            _popover_search_entries()
        )

    async def on_mount(self) -> None:
        settings = ConsoleSessionSettings(provider="openrouter", model="saved-model")

        def _capture(result):
            self.result = result

        await self.push_screen(
            _test_popover(settings, self.providers_models),
            callback=_capture,
        )


@pytest.mark.asyncio
async def test_popover_model_search_inserts_transient_option():
    """Picking a search result inserts it as a transient option and selects it."""
    from textual.widgets import Input, OptionList, Select

    app = _PopoverSearchApp()
    async with app.run_test(size=(90, 30)) as pilot:
        search_input = app.screen.query_one("#model-search-picker-input", Input)
        search_input.value = "claude"
        await pilot.pause()
        results = app.screen.query_one("#model-search-picker-results", OptionList)
        assert results.display
        option = results.get_option_at_index(0)
        results.post_message(OptionList.OptionSelected(results, option, 0))
        await pilot.pause()
        model_select = app.screen.query_one("#console-popover-model", Select)
        picker = app.screen.query_one(
            "#console-popover-model-search", ModelSearchPicker
        )
        option_values = [value for _, value in model_select._options]
        assert picker.display is True
        assert model_select.display is False
        assert "anthropic/claude-x" in option_values
        assert model_select.value == "anthropic/claude-x"


@pytest.mark.asyncio
async def test_popover_search_control_fits_compact_terminal_geometry():
    """The shared picker stays operable at the popover's minimum width."""
    from textual.widgets import Input, OptionList

    app = _PopoverSearchApp()
    async with app.run_test(size=(60, 24)) as pilot:
        search_input = app.screen.query_one("#model-search-picker-input", Input)
        search_input.focus()
        await pilot.pause()
        results = app.screen.query_one("#model-search-picker-results", OptionList)
        popover = app.screen.query_one("#console-model-popover")

        assert results.display is True
        for widget in (popover, search_input, results):
            assert widget.region.x >= 0
            assert widget.region.y >= 0
            assert widget.region.right <= app.size.width
            assert widget.region.bottom <= app.size.height


@pytest.mark.asyncio
async def test_popover_preserves_prefilled_model_after_mount():
    """TASK-364: the model Select must still show the session's current model
    after mount — the provider Select's mount-time Select.Changed must not wipe
    the prefill to blank (a user cannot confirm/Apply a model they can't see)."""
    from textual.widgets import Select

    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        model_select = app.screen.query_one("#console-popover-model", Select)
        assert model_select.value == "model-a"


@pytest.mark.asyncio
async def test_popover_changing_provider_still_resets_the_model():
    """TASK-364 guard must not over-fire: a REAL provider change (to one whose
    models differ) must still clear the stale model selection."""
    from textual.widgets import Select

    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        provider_select = app.screen.query_one("#console-popover-provider", Select)
        provider_select.value = "openai"
        await pilot.pause()
        model_select = app.screen.query_one("#console-popover-model", Select)
        # The stale llama.cpp model must not linger under the new provider.
        assert model_select.value != "model-a"
        picker = app.screen.query_one(
            "#console-popover-model-search", ModelSearchPicker
        )
        assert picker.value is None


@pytest.mark.asyncio
async def test_popover_custom_model_uses_shared_picker_escape_hatch():
    from textual.widgets import Input

    app = _PopoverApp()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.click("#model-search-picker-custom")
        custom_input = app.screen.query_one("#model-search-picker-input", Input)
        custom_input.value = "private/model-id"
        await pilot.pause()
        await pilot.click("#console-popover-apply")
        await pilot.pause()

    assert app.result is not None
    assert isinstance(app.result, ConsoleSettingsCommittedSubmission)
    assert app.result.live_commit.settings.model == "private/model-id"


@pytest.mark.asyncio
async def test_popover_provider_options_use_display_names():
    """TASK-364: the provider Select must use the same catalog display names as
    the full settings modal ('llama.cpp'), not the raw 'llama_cpp' key."""
    from textual.widgets import Select

    app = _PopoverApp()
    async with app.run_test(size=(90, 30)):
        provider_select = app.screen.query_one("#console-popover-provider", Select)
        labels = {label: value for label, value in provider_select._options}
        assert "llama.cpp" in labels
        assert labels["llama.cpp"] == "llama_cpp"
        assert "llama_cpp" not in labels


@pytest.mark.asyncio
async def test_popover_labels_temperature_input():
    """TASK-364: the temperature Input needs a visible label — its placeholder
    disappears once a value is present, leaving a bare cryptic number."""
    from textual.widgets import Static

    app = _PopoverApp()
    async with app.run_test(size=(90, 30)):
        texts = [
            str(getattr(w.renderable, "plain", w.renderable))
            for w in app.screen.query(Static)
        ]
        assert any("Temperature" in text for text in texts)


@pytest.mark.asyncio
async def test_switcher_result_shows_saved_chat_vocabulary_not_in_progress():
    """TASK-356 end-to-end: a saved conversation with a membership role
    renders in the switcher as 'saved chat' (the rail's vocabulary), never
    the raw 'in-progress', with a recency label derived from updated_sort."""

    class _App(ConsolidatedCSSApp):
        async def on_mount(self) -> None:
            row = ConsoleConversationBrowserInputRow(
                row_key="conv-9",
                conversation_id="conv-9",
                native_session_id=None,
                title="Websocket reconnect strategy",
                scope_type="workspace",
                workspace_id="ws-1",
                workspace_label="Chats",
                status="in-progress",
                updated_sort="2026-07-04T10:00:00+00:00",
            )
            await self.push_screen(ConsoleSessionSwitcherModal(rows=(row,)))

    app = _App()
    async with app.run_test(size=(90, 30)) as pilot:
        await pilot.pause()
        result = app.screen.query_one("#console-switcher-result-0", Button)
        label = str(result.label)
        assert "saved chat" in label
        assert "in-progress" not in label


def _overflow_conversation_browser():
    rows = tuple(
        ConsoleConversationBrowserInputRow(
            row_key=f"c{i}",
            conversation_id=f"c{i}",
            native_session_id=None,
            title=f"Chat {i}",
            scope_type="global",
            workspace_id=None,
            workspace_label="Chats",
            status="workspace-thread",
            updated_label="1d",
        )
        for i in range(CONSOLE_CONVERSATION_BROWSER_GROUP_ROW_LIMIT + 3)
    )
    return build_console_conversation_browser_state(
        rows=rows, active_workspace_id="ws-a"
    )


class _OverflowTrayApp(ConsolidatedCSSApp):
    def compose(self):
        import dataclasses

        state = dataclasses.replace(
            _workspace_state(), conversation_browser=_overflow_conversation_browser()
        )
        yield ConsoleWorkspaceContextTray(state, id="overflow-tray")


@pytest.mark.asyncio
async def test_rail_discloses_conversations_hidden_by_the_cap_in_no_query_view():
    """TASK-354: with more conversations than the per-group cap and no search
    active, the rail must render an explicit overflow disclosure pointing at
    Ctrl+K, instead of silently dropping the oldest with no affordance."""
    app = _OverflowTrayApp()
    async with app.run_test(size=(70, 40)):
        status = app.query_one("#console-workspace-conversation-search-status", Static)
        text = str(getattr(status.renderable, "plain", status.renderable))
        assert "3 more" in text
        assert "Ctrl+K" in text
