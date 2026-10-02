"""Pure Console setup-card state contracts."""

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_QUIET_EMPTY_COPY,
    CONSOLE_READY_EMPTY_COPY,
    CONSOLE_SETUP_CARD_TITLE,
    CONSOLE_SETUP_STEP_THREE_DETAIL,
    ConsoleSetupCardState,
    build_console_detected_server_action,
    build_console_setup_card_state,
    coerce_console_first_send_completed,
    console_setup_is_blocking,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSettingsReadiness,
    readiness_words,
)
from tldw_chatbook.Chat.local_server_discovery import DiscoveredLocalServer


def _readiness(
    label: str, *, ready: bool = False, detail: str = ""
) -> ConsoleSettingsReadiness:
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

    ready_without_model = _build(
        readiness=_readiness("Ready", ready=True), has_model=False
    )
    assert ready_without_model.steps[1].state == "active"


def test_endpoint_problems_relabel_step_one():
    assert (
        _build(readiness=_readiness("Invalid URL")).steps[0].label
        == "Save the provider's server address (endpoint)"
    )
    assert (
        _build(readiness=_readiness("Endpoint not saved")).steps[0].label
        == "Save the provider's server address (endpoint)"
    )
    assert (
        _build(readiness=_readiness("Unknown")).steps[0].label
        == "Choose a supported provider"
    )
    assert (
        _build(readiness=_readiness("Pending")).steps[0].label
        == "Choose a provider that works in the Console"
    )


def test_provider_ready_without_model_activates_model_step():
    state = _build(readiness=_readiness("Ready", ready=True), has_model=False)
    assert state.mode == "card"
    assert [step.state for step in state.steps] == ["done", "active", "pending"]
    assert state.steps[0].glyph == "✓"
    # TASK-33005.3 (rewritten on purpose): was "OpenAI ready", a readiness
    # claim outside the one vocabulary; the done step now names the provider.
    assert state.steps[0].detail == "OpenAI"


def test_the_active_step_carries_the_one_readiness_word():
    """TASK-33005.3 (AC#8): the card shows the word every other surface shows."""
    refused = ConsoleSettingsReadiness(
        "Not ready",
        "",
        False,
        operability="not_ready",
        blocker="endpoint_unreachable",
        recovery_action="retry_connection",
        configuration="configured",
        credential="not_required",
        endpoint="unreachable",
        endpoint_category="connection_refused",
        model="unconfirmed",
    )
    no_model = ConsoleSettingsReadiness(
        "Missing model",
        "",
        False,
        operability="not_ready",
        blocker="model_missing",
        recovery_action="select_model",
        configuration="configured",
        credential="not_required",
        model="missing",
    )

    blocked = _build(readiness=refused)
    assert blocked.steps[0].label == "Reconnect the provider server"
    assert blocked.steps[0].detail == readiness_words(refused)
    assert blocked.steps[0].detail == "Not ready · refused"
    pick = _build(readiness=no_model, has_model=False)
    assert [step.detail for step in pick.steps[:2]] == [
        "OpenAI",
        "Not ready · no model",
    ]


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


def test_detected_server_action_offers_labeled_affordance_with_model():
    action = build_console_detected_server_action(
        DiscoveredLocalServer(
            provider_key="llama_cpp",
            base_url="http://127.0.0.1:8080",
            model_ids=("qwen-3", "phi-4"),
        ),
        card_mode="card",
    )

    assert action is not None
    assert action.label == "Use detected llama.cpp (127.0.0.1:8080)"
    assert (
        action.tooltip
        == "Sets provider to llama.cpp at 127.0.0.1:8080 and model to qwen-3."
    )
    assert action.provider_key == "llama_cpp"
    assert action.base_url == "http://127.0.0.1:8080"
    assert action.model_id == "qwen-3"


def test_detected_server_action_without_models_asks_for_model_next():
    action = build_console_detected_server_action(
        DiscoveredLocalServer(
            provider_key="ollama",
            base_url="http://localhost:11434",
            model_ids=(),
        ),
        card_mode="card",
    )

    assert action is not None
    assert action.label == "Use detected Ollama (localhost:11434)"
    assert (
        action.tooltip
        == "Sets provider to Ollama at localhost:11434. Pick a model next."
    )
    assert action.model_id is None


def test_detected_server_action_only_exists_in_card_mode():
    server = DiscoveredLocalServer(
        provider_key="llama_cpp",
        base_url="http://127.0.0.1:8080",
    )

    assert build_console_detected_server_action(server, card_mode="ready_line") is None
    assert build_console_detected_server_action(server, card_mode="quiet") is None
    assert build_console_detected_server_action(None, card_mode="card") is None


def test_detected_server_action_drops_non_loopback_and_malformed_servers():
    remote = DiscoveredLocalServer(
        provider_key="vllm",
        base_url="http://192.168.1.5:8000",
    )
    blank = DiscoveredLocalServer(provider_key="", base_url="http://127.0.0.1:8080")

    assert build_console_detected_server_action(remote, card_mode="card") is None
    assert build_console_detected_server_action(blank, card_mode="card") is None


# --- console_setup_is_blocking (task-2852) -----------------------------
#
# Library's "Use in Console" pre-navigation check has no mounted ChatScreen
# to ask "would landing on Console right now show the blocking setup card".
# `console_setup_is_blocking` is the one source of truth both that check and
# (indirectly, via `build_console_setup_card_state`) the real Console screen
# use -- these tests pin its branch outcomes against the exact same fixtures
# `build_console_setup_card_state`'s own tests above use, so the two can
# never silently drift apart.


def test_setup_is_blocking_when_provider_missing():
    assert (
        console_setup_is_blocking(
            readiness=_readiness("Missing key"),
            has_model=True,
            first_send_completed=False,
        )
        is True
    )


def test_setup_is_blocking_when_ready_but_no_model():
    assert (
        console_setup_is_blocking(
            readiness=_readiness("Ready", ready=True),
            has_model=False,
            first_send_completed=False,
        )
        is True
    )


def test_setup_not_blocking_once_provider_and_model_ready():
    assert (
        console_setup_is_blocking(
            readiness=_readiness("Ready", ready=True),
            has_model=True,
            first_send_completed=False,
        )
        is False
    )


def test_setup_not_blocking_once_first_send_completed():
    """Mirrors `test_first_send_completed_is_quiet_forever`: once the
    persisted global flag is set, an otherwise-unconfigured provider must
    not report as blocking -- the real Console screen would show the quiet
    empty state here, not the setup card, so a caller outside an active
    session must agree."""
    assert (
        console_setup_is_blocking(
            readiness=_readiness("Missing key"),
            has_model=False,
            first_send_completed=True,
        )
        is False
    )
