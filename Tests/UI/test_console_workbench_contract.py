import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical

from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from tldw_chatbook.Chat.console_chat_models import ConsoleRunState, ConsoleRunStatus
from tldw_chatbook.Chat.console_display_state import (
    ConsoleControlState,
    build_console_disabled_reason,
)
from tldw_chatbook.Chat.console_onboarding_state import (
    ConsoleSetupCardState,
    ConsoleSetupStep,
)
from tldw_chatbook.UI.Screens.chat_screen import (
    CONSOLE_FOCUS_TARGETS_BY_PANE,
    CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
    ChatScreen,
)
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript
from tldw_chatbook.Widgets.Console.console_workbench_state import (
    build_console_workbench_state,
)


class ConsoleHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


class ConsoleFooterHarness(App):
    def __init__(self, app_instance):
        super().__init__()
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        yield AppFooterStatus(id="app-footer-status")

    async def on_mount(self) -> None:
        await self.push_screen(ChatScreen(self.app_instance))


class EmptyTranscriptActionHarness(App):
    def __init__(self):
        super().__init__()
        self.workbench_actions: list[str] = []

    def compose(self) -> ComposeResult:
        yield ConsoleTranscript(id="console-native-transcript")

    def on_workbench_action_requested(self, event: WorkbenchActionRequested) -> None:
        event.stop()
        self.workbench_actions.append(event.action_id)


def _configure_native_ready_console(app, model: str = "local-model") -> None:
    app.app_config = {
        "chat_defaults": {"provider": "llama_cpp", "model": model},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:9099",
                "model": model,
            },
        },
    }
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = model


def _is_displayed(widget) -> bool:
    current = widget
    while current is not None:
        if current.display is False or current.styles.display == "none":
            return False
        current = getattr(current, "parent", None)
    return True


def _visible_workbench_focus_ids(console: ChatScreen) -> set[str]:
    visible: set[str] = set()
    for pane_id, target_ids in CONSOLE_FOCUS_TARGETS_BY_PANE.items():
        pane = console.query_one(f"#{pane_id}")
        if not _is_displayed(pane):
            continue
        for widget_id in target_ids:
            widget = console.query_one(f"#{widget_id}")
            if _is_displayed(widget):
                visible.add(widget_id)
                break
    return visible


def _widget_text(widget) -> str:
    label = getattr(widget, "label", None)
    if label is not None:
        plain_label = getattr(label, "plain", None)
        return str(plain_label if plain_label is not None else label)
    if getattr(widget, "id", "") == "console-transcript-empty-state":
        child_text = [
            _widget_text(child)
            for child in widget.walk_children()
            if _is_displayed(child)
        ]
        return " ".join(part for part in child_text if part)
    renderable = getattr(widget, "renderable", "")
    plain = getattr(renderable, "plain", None)
    text = str(plain if plain is not None else renderable)
    if text:
        return text
    return ""


def _children_in_display_order(widget) -> list[str]:
    ids: list[str] = []
    for child in widget.walk_children():
        child_id = getattr(child, "id", None)
        if child_id:
            ids.append(child_id)
    return ids


def _style_scalar_value(value) -> float:
    return float(getattr(value, "value", value))


def _control_state() -> ConsoleControlState:
    return ConsoleControlState(
        provider_label="Provider: llama.cpp",
        model_label="Model: local-model",
        persona_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0",
        tools_label="Tools: 0",
        approvals_label="Approvals: 0",
    )


@pytest.mark.asyncio
async def test_console_workbench_header_seam_has_no_visible_layout_cost():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        header = console.query_one("#console-workbench-header")
        assert not _is_displayed(header)
        assert _style_scalar_value(header.styles.height) == 0
        assert _style_scalar_value(header.styles.min_height) == 0
        assert not _is_displayed(console.query_one("#console-workbench-mode-strip"))
        assert not _is_displayed(console.query_one("#console-workbench-command-strip"))
        assert _is_displayed(console.query_one("#console-control-bar"))


@pytest.mark.asyncio
async def test_console_has_one_canonical_visible_state_action_strip():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        assert not _is_displayed(console.query_one("#console-workbench-mode-strip"))
        assert not _is_displayed(console.query_one("#console-workbench-command-strip"))
        control_bar = console.query_one("#console-control-bar")
        assert _is_displayed(control_bar)

        visible_text = " ".join(
            _widget_text(child)
            for child in control_bar.walk_children()
            if _is_displayed(child)
        )
        assert visible_text.count("Provider:") == 1
        assert visible_text.count("Model:") == 1
        assert visible_text.count("Settings") == 1
        assert visible_text.count("Library RAG") == 1


@pytest.mark.asyncio
async def test_console_control_bar_renders_visible_state_chips():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        control_bar = console.query_one("#console-control-bar")
        assert _is_displayed(control_bar)

        expected_selectors = (
            "#console-provider-chip",
            "#console-model-chip",
            "#console-persona-chip",
            "#console-rag-chip",
            "#console-sources-chip",
            "#console-tools-chip",
            "#console-approvals-chip",
        )
        visible_chip_text = []
        for selector in expected_selectors:
            chip = console.query_one(selector)
            assert _is_displayed(chip), selector
            visible_chip_text.append(_widget_text(chip))

        assert any("Provider:" in text for text in visible_chip_text)
        assert any("Model:" in text for text in visible_chip_text)
        assert any("Assistant:" in text or "Persona:" in text for text in visible_chip_text)
        assert any("RAG:" in text for text in visible_chip_text)
        assert any("Sources:" in text for text in visible_chip_text)
        assert any("Tools:" in text for text in visible_chip_text)
        assert any("Approvals:" in text for text in visible_chip_text)


@pytest.mark.asyncio
async def test_console_control_bar_exposes_compact_visible_actions():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        for selector in (
            "#console-control-new-tab",
            "#console-control-settings",
            "#console-control-attach-context",
            "#console-control-run-library-rag",
            "#console-control-save-chatbook",
            "#console-control-help",
        ):
            action = console.query_one(selector)
            assert _is_displayed(action), selector
        assert console.query_one("#console-control-settings").disabled is False
        assert console.query_one("#console-control-attach-context").disabled is False
        assert console.query_one("#console-control-run-library-rag").disabled is False
        assert console.query_one("#console-control-help").disabled is False


@pytest.mark.asyncio
async def test_console_left_rail_prioritizes_attach_and_active_conversation():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        rail = console.query_one("#console-left-rail")
        visible_text = " ".join(
            _widget_text(child)
            for child in rail.walk_children()
            if _is_displayed(child)
        )

        assert visible_text.index("Attach") < visible_text.index("Conversations")
        assert "No sources attached." in visible_text
        assert "Chat 1" in visible_text


@pytest.mark.asyncio
async def test_console_inspector_prioritizes_actionable_status_before_secondary_groups():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.console_pending_approval_count = 1
    app.console_tool_count = 1
    app.console_chatbook_artifact_available = True
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        inspector = console.query_one("#console-run-inspector-state")
        ordered_ids = _children_in_display_order(inspector)

        status_index = ordered_ids.index("console-inspector-run-status-summary")
        approval_action_index = ordered_ids.index("console-inspector-review-approval")
        tool_action_index = ordered_ids.index("console-inspector-review-tool-call")
        save_action_index = ordered_ids.index("console-inspector-save-chatbook")
        first_secondary_heading_index = ordered_ids.index(
            "console-inspector-selected-conversation-heading"
        )

        assert status_index < approval_action_index < first_secondary_heading_index
        assert status_index < tool_action_index < first_secondary_heading_index
        assert status_index < save_action_index < first_secondary_heading_index
        assert console.query_one("#console-inspector-review-approval").disabled is False
        assert console.query_one("#console-inspector-review-tool-call").disabled is False
        assert console.query_one("#console-inspector-save-chatbook").disabled is False


@pytest.mark.asyncio
async def test_console_blocked_inspector_explains_impact_and_next_action():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        inspector = console.query_one("#console-run-inspector-state")
        visible_text = " ".join(
            _widget_text(child)
            for child in inspector.walk_children()
            if _is_displayed(child)
        )
        assert "Blocked" in visible_text
        assert "Send is blocked" in visible_text
        assert "Choose model" in visible_text


@pytest.mark.asyncio
async def test_console_ready_inspector_shows_run_recipe_and_operational_groups():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        inspector = console.query_one("#console-run-inspector-state")
        text = " ".join(
            _widget_text(child)
            for child in inspector.walk_children()
            if _is_displayed(child)
        )
        assert "Run recipe" in text
        assert "Sources" in text
        assert "Tools" in text
        assert "Approvals" in text
        assert "Artifacts" in text


@pytest.mark.asyncio
async def test_console_composer_keeps_primary_actions_and_setup_card_recovery_visible():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        composer = console.query_one("#console-native-composer")
        assert _is_displayed(composer)
        assert _is_displayed(console.query_one("#console-attach-context"))
        assert _is_displayed(console.query_one("#console-send-message"))
        assert _is_displayed(console.query_one("#console-stop-generation"))
        assert _is_displayed(console.query_one("#console-save-chatbook"))
        assert not _is_displayed(console.query_one("#console-composer-recovery"))
        # The shared Workbench recovery banner must stay hidden — the setup
        # card in the empty transcript owns first-run recovery guidance now.
        assert not _is_displayed(console.query_one("#workbench-recovery-callout"))
        setup_card = console.query_one("#console-transcript-empty-state")
        assert _is_displayed(setup_card)
        card_action = console.query_one("#console-empty-choose-model")
        assert _is_displayed(card_action)
        assert "Choose model" in _widget_text(card_action)
        visible_draft = console.query_one("#console-command-visible-text")
        assert visible_draft.region.width >= 32


@pytest.mark.asyncio
async def test_console_composer_keeps_disabled_reason_outside_input_row():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        reason = console.query_one("#console-send-disabled-reason")
        assert not _is_displayed(reason)
        visible_draft = console.query_one("#console-command-visible-text")
        assert visible_draft.region.width >= 32
        actions = console.query_one("#console-composer-actions")
        send = console.query_one("#console-send-message")
        assert actions.region.y == send.region.y
        assert visible_draft.region.x < actions.region.x
        assert visible_draft.region.x + visible_draft.region.width <= actions.region.x
        assert "model" in (send.tooltip or "").lower()
        assert "nowrap" in str(reason.styles.text_wrap)


@pytest.mark.asyncio
async def test_console_empty_transcript_exposes_beginner_activation_actions():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        empty_panel = console.query_one("#console-transcript-empty-state")
        assert _is_displayed(empty_panel)
        # Setup-incomplete transcripts show the numbered setup card, not a banner.
        assert _widget_text(console.query_one("#console-empty-title")) == "Get started"
        assert "Pick a model" in _widget_text(empty_panel)
        assert "Send your first message" in _widget_text(empty_panel)
        assert "Choose model" in _widget_text(empty_panel)
        assert _is_displayed(console.query_one("#console-empty-choose-model"))
        assert _is_displayed(console.query_one("#console-empty-attach-context"))
        assert _is_displayed(console.query_one("#console-empty-run-library-rag"))


@pytest.mark.asyncio
async def test_console_ready_empty_transcript_exposes_activation_panel_copy():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        empty_panel = console.query_one("#console-transcript-empty-state")
        assert isinstance(empty_panel, Vertical)
        assert _is_displayed(empty_panel)
        # Ready state shows one ready line and hides the setup card + action row.
        assert _widget_text(console.query_one("#console-empty-body")) == (
            "Ready — type a message to begin."
        )
        assert not _is_displayed(console.query_one("#console-empty-title"))
        assert not _is_displayed(console.query_one("#console-empty-action-row"))


@pytest.mark.asyncio
async def test_console_empty_transcript_choose_model_opens_settings():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        await _wait_for_selector(console, pilot, "#console-empty-choose-model")

        await pilot.click("#console-empty-choose-model")
        await pilot.pause()

        assert host.screen.query("#console-settings-modal") or host.screen.query(
            "#settings-screen"
        )


@pytest.mark.asyncio
async def test_console_ready_empty_transcript_omits_setup_action_row():
    # A ready Console no longer surfaces the empty-state choose-model action;
    # the spec routes ready users straight to typing (no setup action row).
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        await _wait_for_selector(console, pilot, "#console-transcript-empty-state")

        assert not _is_displayed(console.query_one("#console-empty-action-row"))
        assert not _is_displayed(console.query_one("#console-empty-choose-model"))
        assert _widget_text(console.query_one("#console-empty-body")) == (
            "Ready — type a message to begin."
        )


@pytest.mark.asyncio
async def test_console_empty_transcript_actions_post_workbench_messages():
    app = EmptyTranscriptActionHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)
        transcript.sync_empty_state(
            ConsoleSetupCardState(
                mode="card",
                steps=(ConsoleSetupStep(state="active", label="Add an API key"),),
            ),
            provider_action_label="Configure API",
            provider_action_tooltip="Open provider settings.",
        )
        await pilot.pause()

        await pilot.click("#console-empty-attach-context")
        await pilot.click("#console-empty-run-library-rag")
        await pilot.pause()

        assert app.workbench_actions == ["attach-context", "run-library-rag"]


@pytest.mark.asyncio
async def test_console_transcript_empty_state_renders_ready_activation_copy():
    app = EmptyTranscriptActionHarness()

    async with app.run_test(size=(80, 24)) as pilot:
        transcript = app.query_one("#console-native-transcript", ConsoleTranscript)

        transcript.sync_empty_state(
            ConsoleSetupCardState(
                mode="ready_line",
                body_copy="Type in Composer, attach sources, or run Library RAG before sending.",
            )
        )
        await pilot.pause()

        assert _widget_text(app.query_one("#console-empty-body")) == (
            "Type in Composer, attach sources, or run Library RAG before sending."
        )


# NOTE: The static ``_console_empty_transcript_copy`` / blocked-copy adapters were
# removed with the setup-card rewire; their setup-blocker mapping now lives in
# ``build_console_setup_card_state`` and is covered by
# ``Tests/Chat/test_console_onboarding_state.py``.


@pytest.mark.parametrize(
    ("blocker_copy", "expected_label", "expected_tooltip"),
    (
        (
            "Provider setup needed: choose a model",
            "Choose model",
            "Choose a model for this Console session",
        ),
        (
            "Provider setup needed: choose a provider",
            "Choose provider",
            "Choose a provider for this Console session",
        ),
        (
            "Provider setup needed: OpenAI missing API key",
            CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL,
            "Configure API and API key before sending",
        ),
        (
            "Provider setup needed: save the endpoint in settings",
            "Configure endpoint",
            "Configure the provider endpoint before sending",
        ),
        (
            "Provider setup needed: verify local runtime",
            "Review settings",
            "Review Console provider settings before sending",
        ),
    ),
)
def test_console_empty_recovery_action_copy_matches_setup_blocker(
    blocker_copy: str,
    expected_label: str,
    expected_tooltip: str,
):
    assert ChatScreen._console_empty_recovery_action_copy(blocker_copy) == (
        expected_label,
        expected_tooltip,
    )


@pytest.mark.asyncio
async def test_console_empty_transcript_provider_recovery_label_matches_setup_blocker(
    monkeypatch,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": "gpt-4.1-2025-04-14"},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        action = console.query_one("#console-empty-choose-model")
        assert _widget_text(action) == CONSOLE_PROVIDER_CONFIGURE_API_KEY_LABEL
        assert str(action.tooltip or "") == "Configure OpenAI API and API key in Settings"


def test_console_empty_recovery_action_keeps_provider_label_with_empty_tooltip():
    assert ChatScreen._console_empty_recovery_action_copy(
        "Provider setup needed: OpenAI missing API key",
        provider_action_label="Localized recovery label",
        provider_action_tooltip="",
    ) == ("Localized recovery label", "")


def test_console_workbench_state_exposes_core_actions_visibly():
    state = build_console_workbench_state(
        control_state=_control_state(),
        provider_blocker_copy="",
        can_send=True,
        can_stop=False,
        can_save_chatbook=True,
    )

    actions = {action.id: action for action in state.actions}
    action_labels = {action.label for action in state.actions}

    assert {
        "Settings",
        "Attach context",
        "Run Library RAG",
        "Save Chatbook",
        "Help",
    } <= action_labels
    assert tuple(actions) == (
        "new-tab",
        "settings",
        "attach-context",
        "run-library-rag",
        "save-chatbook",
        "send",
        "stop",
        "help",
    )
    assert actions["save-chatbook"].disabled is False
    assert actions["send"].disabled is False
    assert actions["send"].primary is True
    assert actions["stop"].disabled is True
    assert state.route_id == "chat"
    assert state.density == "normal"
    assert state.header.title == "Console"
    assert tuple(pane.id for pane in state.panes) == (
        "context",
        "transcript",
        "inspector",
        "composer",
    )
    assert state.recovery is None


def test_console_workbench_state_hides_recovery_banner_when_provider_blocked():
    """The setup card + composer own first-run guidance now (Phase 2 spec
    section 2); the shared Workbench recovery banner must stay absent even
    while a provider blocker keeps Send disabled."""
    state = build_console_workbench_state(
        control_state=ConsoleControlState(
            provider_label="Provider: OpenAI",
            model_label="Model: --",
            persona_label="Assistant: General",
            rag_label="RAG: off",
            sources_label="Sources: 0",
            tools_label="Tools: 0",
            approvals_label="Approvals: 0",
        ),
        provider_blocker_copy="Provider setup needed: choose a model",
        provider_action_label="Choose model",
        can_send=False,
        can_stop=False,
        can_save_chatbook=False,
    )

    assert state.recovery is None
    assert state.header.status == "blocked"

    modes = {mode.id: mode for mode in state.modes}
    assert modes["provider"].status == "blocked"
    assert modes["model"].status == "blocked"


def test_console_workbench_state_never_resurfaces_recovery_for_named_setup_action():
    """A blocker copy that names a specific recovery action (e.g. "Choose
    model") must still not leak into a duplicate workbench recovery banner;
    that copy belongs to the setup card and composer disabled-reason only."""
    state = build_console_workbench_state(
        control_state=_control_state(),
        provider_blocker_copy="Provider setup needed: choose a model",
        provider_action_label="Choose model",
        can_send=False,
    )

    assert state.recovery is None
    assert state.header.status == "blocked"


def test_console_disabled_reason_copy_prefers_setup_blocker():
    reason = build_console_disabled_reason(
        action_id="send",
        has_draft=False,
        send_blocked=True,
        setup_blocked_reason="Provider setup needed: choose a model",
    )

    assert reason == "Send blocked — choose a model to continue"


@pytest.mark.parametrize(
    ("setup_blocked_reason", "expected_reason"),
    (
        (
            "Provider setup needed: choose a provider",
            "Send blocked — choose a provider to continue",
        ),
        (
            "Provider setup needed: OpenAI missing API key",
            "Send blocked — add an API key to continue",
        ),
        (
            "Provider setup needed: configure endpoint",
            "Send blocked — configure the endpoint to continue",
        ),
        (
            "Provider setup needed: llama.cpp endpoint unavailable",
            "Send blocked — configure the endpoint to continue",
        ),
        (
            "Provider setup needed: verify local runtime",
            "Send blocked — finish provider setup to continue",
        ),
    ),
)
def test_console_disabled_reason_copy_maps_setup_blockers(
    setup_blocked_reason, expected_reason
):
    for has_draft in (False, True):
        reason = build_console_disabled_reason(
            action_id="send",
            has_draft=has_draft,
            send_blocked=True,
            setup_blocked_reason=setup_blocked_reason,
        )

        assert reason == expected_reason


def test_console_disabled_reason_copy_handles_draft_and_ready_states():
    assert (
        build_console_disabled_reason(
            action_id="send",
            has_draft=False,
            send_blocked=False,
        )
        == "Send disabled: type a message"
    )
    assert (
        build_console_disabled_reason(
            action_id="send",
            has_draft=True,
            send_blocked=False,
        )
        == ""
    )


def test_console_workbench_state_disables_send_when_provider_is_blocked():
    state = build_console_workbench_state(
        control_state=_control_state(),
        provider_blocker_copy="Provider setup needed: choose a model",
        can_send=True,
        can_stop=True,
        can_save_chatbook=True,
        density="compact",
    )

    actions = {action.id: action for action in state.actions}

    assert state.density == "compact"
    assert state.header.status == "blocked"
    assert actions["send"].disabled is True
    assert actions["send"].primary is False
    assert actions["stop"].disabled is False


@pytest.mark.asyncio
async def test_console_core_controls_are_visible_without_command_palette():
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        assert not _is_displayed(console.query_one("#console-workbench-command-strip"))
        for selector in (
            "#console-control-settings",
            "#console-control-attach-context",
            "#console-control-run-library-rag",
            "#console-control-help",
            "#console-native-composer",
        ):
            widget = console.query_one(selector)
            assert _is_displayed(widget), selector


@pytest.mark.asyncio
async def test_console_setup_card_carries_blocked_send_guidance_instead_of_banner():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        # The shared Workbench recovery banner must stay hidden while the
        # setup card and the composer's Send tooltip carry the "blocked"
        # guidance instead (Phase 2 spec, section 2).
        recovery = console.query_one("#workbench-recovery-callout")
        assert not _is_displayed(recovery)
        setup_card = console.query_one("#console-transcript-empty-state")
        assert _is_displayed(setup_card)
        send_button = console.query_one("#console-send-message")
        assert "model" in (send_button.tooltip or "").lower()


@pytest.mark.asyncio
async def test_console_setup_card_choose_model_action_is_visible_and_primary_recovery():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        recovery = console.query_one("#workbench-recovery-callout")
        assert not _is_displayed(recovery)
        action = console.query_one("#console-empty-choose-model")
        assert _is_displayed(action)
        assert str(action.label) == "Choose model"
        assert action.disabled is False


@pytest.mark.asyncio
async def test_console_setup_card_recovery_action_button_is_visible_and_actionable():
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {"provider": "OpenAI", "model": ""},
        "api_settings": {"openai": {"api_key": ""}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = ""
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        await _wait_for_selector(console, pilot, "#console-empty-choose-model")

        assert not _is_displayed(console.query_one("#workbench-recovery-callout"))
        action = console.query_one("#console-empty-choose-model")
        assert _is_displayed(action)
        assert not action.disabled
        await pilot.click("#console-empty-choose-model")
        await pilot.pause()

        assert host.screen.query("#console-settings-modal") or host.screen.query(
            "#settings-screen"
        )


@pytest.mark.asyncio
async def test_console_workbench_send_action_enables_after_typing_draft():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-send-message")

        assert not _is_displayed(console.query_one("#console-workbench-command-strip"))
        send_action = console.query_one("#console-send-message")
        assert _is_displayed(send_action)
        assert send_action.has_class("console-send-ready") is False

        await pilot.press("h")
        await pilot.pause()

        send_action = console.query_one("#console-send-message")
        assert _is_displayed(send_action)
        assert send_action.has_class("console-send-ready") is True


@pytest.mark.asyncio
async def test_console_workbench_send_action_disables_during_active_run():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-send-message")
        await _wait_for_selector(console, pilot, "#console-stop-generation")

        await pilot.press("h")
        await pilot.pause()

        controller = console._ensure_console_chat_controller()
        controller.run_state = ConsoleRunState(
            ConsoleRunStatus.STREAMING,
            "Streaming response.",
        )
        console._sync_console_control_bar()
        await pilot.pause()

        assert not _is_displayed(console.query_one("#console-workbench-command-strip"))
        send_action = console.query_one("#console-send-message")
        stop_action = console.query_one("#console-stop-generation")
        assert _is_displayed(send_action)
        assert _is_displayed(stop_action)
        assert send_action.has_class("console-send-ready") is False
        assert send_action.has_class("console-send-blocked") is True
        assert stop_action.has_class("console-stop-active") is True


@pytest.mark.asyncio
async def test_console_active_stream_sync_skips_unchanged_chrome_and_inspector(monkeypatch):
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        controller = console._ensure_console_chat_controller()
        controller.run_state = ConsoleRunState(
            ConsoleRunStatus.STREAMING,
            "Streaming response.",
        )
        control_state = console._build_console_control_state(
            console._pending_console_launch_context
        )
        console._last_console_control_state = control_state
        console._last_console_workbench_state = console._build_console_workbench_state(
            control_state
        )
        console._last_console_rail_state = console._current_console_rail_state()

        workbench_syncs = 0
        inspector_refreshes = 0

        def count_workbench_sync(*_args, **_kwargs):
            nonlocal workbench_syncs
            workbench_syncs += 1

        inspector = console.query_one("#console-run-inspector-state")

        def count_inspector_refresh(*_args, **_kwargs):
            nonlocal inspector_refreshes
            inspector_refreshes += 1

        monkeypatch.setattr(console, "_sync_console_workbench_state", count_workbench_sync)
        monkeypatch.setattr(inspector, "refresh", count_inspector_refresh)

        for _ in range(5):
            console._sync_console_control_bar()
        await pilot.pause()

        assert workbench_syncs == 0
        assert inspector_refreshes == 0


@pytest.mark.asyncio
async def test_console_f6_cycles_visible_workbench_panes():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        visible_focus_ids = _visible_workbench_focus_ids(console)
        assert "console-native-transcript" in visible_focus_ids
        assert "console-native-composer" in visible_focus_ids

        await pilot.press("f6")
        await pilot.pause()
        first_focus = pilot.app.focused
        assert first_focus is not None
        assert first_focus.id in visible_focus_ids

        await pilot.press("f6")
        await pilot.pause()
        second_focus = pilot.app.focused
        assert second_focus is not None
        assert second_focus.id in visible_focus_ids
        assert second_focus.id != first_focus.id


@pytest.mark.asyncio
async def test_console_route_switch_restores_workbench_focus():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        console.query_one("#console-native-composer").focus()
        await pilot.pause()
        assert pilot.app.focused is not None
        assert pilot.app.focused.id == "console-native-composer"

        await console.remove()
        await pilot.pause()
        await host.push_screen(ChatScreen(app))
        await pilot.pause()

        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        visible_focus_ids = _visible_workbench_focus_ids(console)
        assert pilot.app.focused is not None
        assert pilot.app.focused.id in visible_focus_ids


@pytest.mark.asyncio
async def test_console_f1_help_lists_visible_actions():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        await pilot.press("f1")
        await pilot.pause()

        panel = host.screen.query_one("#workbench-help-panel")
        assert _is_displayed(panel)
        body = str(host.screen.query_one("#workbench-help-body").renderable)
        assert "Settings" in body
        assert "Attach context" in body
        assert "Run Library RAG" in body
        assert "F6" in body
        assert "next pane" in body
        assert "Ctrl+P" in body


@pytest.mark.asyncio
async def test_console_registers_footer_workbench_shortcuts():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleFooterHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")
        footer = host.query_one(AppFooterStatus)

        assert footer.shortcut_text == (
            "F6 next pane | Shift+F6 previous pane | F1 help | Enter send | Ctrl+P palette"
        )

        await console.remove()
        await pilot.pause()

        assert footer.shortcut_text == AppFooterStatus.DEFAULT_SHORTCUT_TEXT


@pytest.mark.asyncio
async def test_console_shell_uses_configured_workbench_density():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config.setdefault("appearance", {})["ui_density"] = "compact"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        shell = console.query_one("#console-shell")
        assert shell.has_class("density-compact")
        assert not shell.has_class("density-normal")


@pytest.mark.asyncio
async def test_console_shell_uses_normal_workbench_density_by_default():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        shell = console.query_one("#console-shell")
        assert shell.has_class("density-normal")
        assert not shell.has_class("density-compact")


@pytest.mark.asyncio
async def test_console_shell_uses_legacy_density_setting_fallback():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config.setdefault("appearance", {})["density"] = "compact"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        shell = console.query_one("#console-shell")
        assert shell.has_class("density-compact")
        assert not shell.has_class("density-normal")


@pytest.mark.asyncio
async def test_console_shell_invalid_workbench_density_falls_back_to_normal():
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.app_config.setdefault("appearance", {})["ui_density"] = "spacious"
    host = ConsoleHarness(app)

    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-shell")

        shell = console.query_one("#console-shell")
        assert shell.has_class("density-normal")
        assert not shell.has_class("density-compact")
