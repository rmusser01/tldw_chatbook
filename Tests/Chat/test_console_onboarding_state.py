"""Pure Console setup-card state contracts."""

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    CONSOLE_READY_EMPTY_COPY,
    CONSOLE_SETUP_CARD_TITLE,
    CONSOLE_SETUP_STEP_THREE_DETAIL,
    ConsoleSetupCardState,
    build_console_setup_card_state,
    coerce_console_first_send_completed,
)
from tldw_chatbook.Chat.console_session_settings import ConsoleSettingsReadiness


def _readiness(label: str, *, ready: bool = False, detail: str = "") -> ConsoleSettingsReadiness:
    return ConsoleSettingsReadiness(
        label=label,
        detail=detail,
        native_send_supported=ready,
    )


def _build(**overrides) -> ConsoleSetupCardState:
    defaults = dict(
        readiness=_readiness("Missing key"),
        provider_label="OpenAI",
        has_model=True,
        first_send_completed=False,
        has_messages=False,
        guidance_dismissed=False,
    )
    defaults.update(overrides)
    return build_console_setup_card_state(**defaults)


def test_missing_key_renders_card_with_provider_step_active():
    state = _build()
    assert state.mode == "card"
    assert CONSOLE_SETUP_CARD_TITLE == "Get started"
    # Step 2 must not be pre-checked by a template-default model while the
    # provider is still blocked (virgin-profile gpt-4o default, task-183).
    assert [step.state for step in state.steps] == ["active", "pending", "pending"]
    assert state.steps[0].label == "Connect a provider (API key or local server)"
    assert state.steps[0].glyph == "●"
    assert state.steps[1].label == "Pick a model"
    assert state.steps[2].label == "Send your first message"
    assert state.steps[2].glyph == "○"
    # The composer is blocked by the setup modal while the card shows, so the
    # detail must not claim typing/Enter works yet.
    assert state.steps[2].detail == CONSOLE_SETUP_STEP_THREE_DETAIL
    assert state.steps[2].detail == "Composer unlocks after setup"


def test_template_default_model_does_not_precheck_step_two():
    blocked_with_default_model = _build(has_model=True)
    assert blocked_with_default_model.steps[1].state == "pending"

    ready_without_model = _build(readiness=_readiness("Ready", ready=True), has_model=False)
    assert ready_without_model.steps[1].state == "active"


def test_endpoint_problems_relabel_step_one():
    assert _build(readiness=_readiness("Invalid URL")).steps[0].label == "Save the provider endpoint"
    assert _build(readiness=_readiness("Endpoint not saved")).steps[0].label == "Save the provider endpoint"
    assert _build(readiness=_readiness("Unknown")).steps[0].label == "Choose a supported provider"
    assert _build(readiness=_readiness("Pending")).steps[0].label == "Choose a send-capable provider"


def test_provider_ready_without_model_activates_model_step():
    state = _build(readiness=_readiness("Ready", ready=True), has_model=False)
    assert state.mode == "card"
    assert [step.state for step in state.steps] == ["done", "active", "pending"]
    assert state.steps[0].glyph == "✓"
    assert state.steps[0].detail == "OpenAI ready"


def test_setup_complete_collapses_to_ready_line():
    state = _build(readiness=_readiness("Ready", ready=True), has_model=True)
    assert state.mode == "ready_line"
    assert state.body_copy == CONSOLE_READY_EMPTY_COPY
    assert state.steps == ()


def test_first_send_completed_is_quiet_forever():
    state = _build(
        readiness=_readiness("Ready", ready=True),
        first_send_completed=True,
    )
    assert state.mode == "quiet"
    assert state.body_copy == CONSOLE_QUIET_EMPTY_COPY
    # Quiet wins even when setup is incomplete on a fresh scope.
    assert _build(first_send_completed=True).mode == "quiet"


def test_messages_present_is_quiet():
    assert _build(has_messages=True).mode == "quiet"


def test_dismissal_hides_ready_line_but_not_setup_card():
    ready = _build(
        readiness=_readiness("Ready", ready=True),
        guidance_dismissed=True,
    )
    assert ready.mode == "quiet"
    blocked = _build(guidance_dismissed=True)
    assert blocked.mode == "card"


def test_coerce_first_send_completed():
    assert coerce_console_first_send_completed(True) is True
    assert coerce_console_first_send_completed("true") is True
    assert coerce_console_first_send_completed(1) is True
    assert coerce_console_first_send_completed(None) is False
    assert coerce_console_first_send_completed("no") is False
    assert coerce_console_first_send_completed({}) is False
