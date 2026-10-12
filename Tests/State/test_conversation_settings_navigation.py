from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
    ConsoleProviderReturnIntent,
    ConsoleProviderReturnTarget,
    ConsoleSettingsReturnTarget,
    ConversationSettingsReturnIntent,
    ConversationSettingsReturnOutcome,
    ProviderSettingsNavigationTarget,
)


def test_provider_settings_target_valid_round_trip() -> None:
    context = {
        "category": "providers-models",
        "provider": "OpenAI",
        "model": "gpt-4o-mini",
        "field": "api_key",
        "return_revision": 4,
    }
    target = ProviderSettingsNavigationTarget.from_context(context)
    assert target is not None
    assert target.provider == "openai"
    assert target.to_context() == {**context, "provider": "openai"}


def test_provider_settings_target_allows_missing_model_explicitly() -> None:
    context = {
        "category": "providers-models",
        "provider": "openai",
        "model": None,
        "field": "api_key",
        "return_revision": 4,
    }
    target = ProviderSettingsNavigationTarget.from_context(context)
    assert target is not None
    assert target.model is None
    assert target.to_context() == context


def test_provider_settings_target_rejects_unknown_context_key() -> None:
    assert (
        ProviderSettingsNavigationTarget.from_context(
            {
                "category": "providers-models",
                "provider": "openai",
                "model": "gpt-4o",
                "field": "api_key",
                "return_revision": 4,
                "unexpected": "value",
            }
        )
        is None
    )


@pytest.mark.parametrize("field", ["password", "api_key_env_var", ""])
def test_provider_settings_target_rejects_invalid_field(field: str) -> None:
    assert (
        ProviderSettingsNavigationTarget.from_context(
            {
                "category": "providers-models",
                "provider": "openai",
                "model": "gpt-4o",
                "field": field,
                "return_revision": 4,
            }
        )
        is None
    )


@pytest.mark.parametrize("revision", [0, -1, True, "4"])
def test_provider_settings_target_rejects_invalid_revision(revision: object) -> None:
    assert (
        ProviderSettingsNavigationTarget.from_context(
            {
                "category": "providers-models",
                "provider": "openai",
                "model": "gpt-4o",
                "field": "api_key",
                "return_revision": revision,
            }
        )
        is None
    )


@pytest.mark.parametrize("field", ["endpoint", "model", None])
def test_console_kind_provider_target_names_its_chat_and_any_field(
    field: str | None,
) -> None:
    """TASK-33008.1 (R34): a Console recovery names its chat; the modal's
    context stays byte-identical because the key appears only when set."""
    context = {
        "category": "providers-models",
        "provider": "openai",
        "model": None,
        "field": field,
        "return_revision": 3,
        "session_id": "session-1",
    }
    target = ProviderSettingsNavigationTarget.from_context(context)
    assert target is not None
    assert target.session_id == "session-1"
    assert target.destination == "Console"
    assert target.to_context() == context
    modal = ProviderSettingsNavigationTarget(
        "providers-models", "openai", None, "api_key", 3
    )
    assert modal.destination == "Chat settings"
    assert "session_id" not in modal.to_context()
    for bad in ("", " session-1", "x" * 257, 7):
        assert (
            ProviderSettingsNavigationTarget.from_context(
                {**context, "session_id": bad}
            )
            is None
        )


def test_provider_target_matches_only_its_own_return_kind() -> None:
    """A Console claim for another chat, or the other kind, is stale (R34)."""
    modal_intent = ConversationSettingsReturnIntent("session-1", 1, "model", None)
    console_target = ProviderSettingsNavigationTarget(
        "providers-models", "openai", "gpt-5", "api_key", 2, "session-1"
    )
    modal_target = ProviderSettingsNavigationTarget(
        "providers-models", "openai", "gpt-5", "api_key", 2
    )
    assert console_target.matches(ConsoleProviderReturnIntent("session-1"))
    assert not console_target.matches(ConsoleProviderReturnIntent("session-2"))
    assert not console_target.matches(modal_intent)
    assert modal_target.matches(modal_intent)
    assert not modal_target.matches(ConsoleProviderReturnIntent("session-1"))


def test_a_console_recovery_pins_the_model_only_when_it_names_one() -> None:
    """A route with no model (a provider that lists none) leaves Settings'
    own model in the field; Chat settings routes its draft's, even none."""

    def target(model: str | None, session_id: str | None):
        return ProviderSettingsNavigationTarget(
            "providers-models", "openai", model, "api_key", 2, session_id
        )

    assert target("gpt-5", "session-1").pins_model is True
    assert target(None, "session-1").pins_model is False
    assert target("gpt-5", None).pins_model is True
    assert target(None, None).pins_model is True


def test_console_provider_return_target_is_strict_and_disjoint_from_the_modal() -> None:
    """R36: ``{session_id, return_revision, outcome}`` and nothing else."""
    target = ConsoleProviderReturnIntent("session-1").return_target(
        5, ConversationSettingsReturnOutcome.CREDENTIAL_SAVED
    )
    assert isinstance(target, ConsoleProviderReturnTarget)
    context = target.to_context()
    assert context == {
        "session_id": "session-1",
        "return_revision": 5,
        "outcome": "credential_saved",
    }
    assert ConsoleProviderReturnTarget.from_context(context) == target
    assert ConsoleSettingsReturnTarget.from_context(context) is None
    assert ConsoleProviderReturnTarget.from_context({**context, "extra": 1}) is None
    assert (
        ConsoleProviderReturnTarget.from_context({**context, "return_revision": 0})
        is None
    )
    modal = ConversationSettingsReturnIntent(
        "session-1", 1, "model", None
    ).return_target(5, ConversationSettingsReturnOutcome.WITHOUT_SAVING)
    assert isinstance(modal, ConsoleSettingsReturnTarget)
    assert ConsoleProviderReturnTarget.from_context(modal.to_context()) is None
    with pytest.raises(ValueError):
        ConsoleProviderReturnIntent("")


def test_console_provider_return_carries_the_pair_switch_model_reopens_on() -> None:
    """TASK-33008.2 (R43): a NEEDS SETUP row's return also names the provider
    and the model Switch model reopens on. The keys are sent only when set,
    so a recovery's context is unchanged, and they are held to the navigation
    contract's provider key and text rules."""
    saved = ConversationSettingsReturnOutcome.CREDENTIAL_SAVED
    intent = ConsoleProviderReturnIntent("session-1", "OpenAI", "gpt-5")
    assert (intent.switcher_provider, intent.switcher_model) == ("openai", "gpt-5")
    target = intent.return_target(5, saved)
    context = target.to_context()
    assert context == {
        "session_id": "session-1",
        "return_revision": 5,
        "outcome": "credential_saved",
        "switcher_provider": "openai",
        "switcher_model": "gpt-5",
    }
    assert ConsoleProviderReturnTarget.from_context(context) == target
    assert ConsoleSettingsReturnTarget.from_context(context) is None
    # A provider that lists no model reopens on the provider alone.
    bare = ConsoleProviderReturnIntent("session-1", "openai").return_target(5, saved)
    assert bare.to_context() == {
        "session_id": "session-1",
        "return_revision": 5,
        "outcome": "credential_saved",
        "switcher_provider": "openai",
    }
    assert ConsoleProviderReturnTarget.from_context(bare.to_context()) == bare
    # Not the same return: the Console's consumer compares the two whole.
    assert target != ConsoleProviderReturnIntent("session-1").return_target(5, saved)
    for provider, model in (
        ("custom-ep:lab", "lab-model"),  # R35: no return can name one.
        ("", "gpt-5"),
        ("openai", ""),
        ("openai", "gpt-5\nsk-secret"),
    ):
        with pytest.raises(ValueError):
            ConsoleProviderReturnIntent("session-1", provider, model)
        assert (
            ConsoleProviderReturnTarget.from_context(
                {**context, "switcher_provider": provider, "switcher_model": model}
            )
            is None
        )


def test_console_return_target_round_trip_and_outcome_allowlist() -> None:
    context = {
        "session_id": "session-1",
        "settings_revision": 2,
        "active_view": "model",
        "focus_control_id": "console-settings-model-change",
        "return_revision": 7,
        "outcome": "credential_saved",
    }
    target = ConsoleSettingsReturnTarget.from_context(context)
    assert target is not None
    assert target.outcome is ConversationSettingsReturnOutcome.CREDENTIAL_SAVED
    assert target.to_context() == context
    assert (
        ConsoleSettingsReturnTarget.from_context({**context, "outcome": "arbitrary"})
        is None
    )


def test_return_contracts_are_frozen_and_validate_shape() -> None:
    with pytest.raises(FrozenInstanceError):
        intent = ConversationSettingsReturnIntent("session-1", 1, "model", None)
        intent.session_id = "other"  # type: ignore[misc]
    assert ConversationSettingsReturnIntent.from_context(
        {
            "session_id": "session-1",
            "settings_revision": 1,
            "active_view": "context",
            "focus_control_id": None,
        }
    ) == ConversationSettingsReturnIntent("session-1", 1, "context", None)
    assert (
        ConversationSettingsReturnIntent(
            "session-1", 0, "model", None
        ).settings_revision
        == 0
    )


@pytest.mark.parametrize(
    "focus",
    [
        "console-settings-secret",
        "console-settings-api-key",
        "console-settings-context-view",
        "console-settings-model",
        # TASK-33006.4 deleted the provider picker and the model search.
        "console-settings-provider",
        "console-settings-provider-picker",
        "console-settings-model-picker",
        "bad",
        "x" * 129,
    ],
)
def test_return_target_rejects_focus_outside_bounded_allowlist(focus: str) -> None:
    context = {
        "session_id": "session-1",
        "settings_revision": 2,
        "active_view": "model",
        "focus_control_id": focus,
        "return_revision": 7,
        "outcome": "without_saving",
    }
    assert ConsoleSettingsReturnTarget.from_context(context) is None


@pytest.mark.parametrize(
    "focus",
    [
        "console-settings-model-change",
        "console-settings-base-url",
        "console-settings-temperature",
        "console-settings-max-tokens",
        "console-settings-streaming",
        "console-settings-view-model",
        "console-settings-view-context",
        "console-context-budget-mode",
    ],
)
def test_return_target_accepts_real_focusable_modal_targets(focus: str) -> None:
    context = {
        "session_id": "session-1",
        "settings_revision": 2,
        "active_view": "model",
        "focus_control_id": focus,
        "return_revision": 7,
        "outcome": "without_saving",
    }
    target = ConsoleSettingsReturnTarget.from_context(context)
    assert target is not None
    assert target.to_context() == context
