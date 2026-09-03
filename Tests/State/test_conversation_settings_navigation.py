from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
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


def test_provider_settings_target_rejects_unknown_context_key() -> None:
    assert ProviderSettingsNavigationTarget.from_context({
        "category": "providers-models", "provider": "openai", "model": "gpt-4o",
        "field": "api_key", "return_revision": 4, "unexpected": "value",
    }) is None


@pytest.mark.parametrize("field", ["password", "api_key_env_var", ""])
def test_provider_settings_target_rejects_invalid_field(field: str) -> None:
    assert ProviderSettingsNavigationTarget.from_context({
        "category": "providers-models", "provider": "openai", "model": "gpt-4o",
        "field": field, "return_revision": 4,
    }) is None


@pytest.mark.parametrize("revision", [0, -1, True, "4"])
def test_provider_settings_target_rejects_invalid_revision(revision: object) -> None:
    assert ProviderSettingsNavigationTarget.from_context({
        "category": "providers-models", "provider": "openai", "model": "gpt-4o",
        "field": "api_key", "return_revision": revision,
    }) is None


def test_console_return_target_round_trip_and_outcome_allowlist() -> None:
    context = {
        "session_id": "session-1", "settings_revision": 2, "active_view": "model",
        "focus_control_id": "console-settings-model", "return_revision": 7,
        "outcome": "credential_saved",
    }
    target = ConsoleSettingsReturnTarget.from_context(context)
    assert target is not None
    assert target.outcome is ConversationSettingsReturnOutcome.CREDENTIAL_SAVED
    assert target.to_context() == context
    assert ConsoleSettingsReturnTarget.from_context({**context, "outcome": "arbitrary"}) is None


def test_return_contracts_are_frozen_and_validate_shape() -> None:
    with pytest.raises(FrozenInstanceError):
        intent = ConversationSettingsReturnIntent("session-1", 1, "model", None)
        intent.session_id = "other"  # type: ignore[misc]
    assert ConversationSettingsReturnIntent.from_context({
        "session_id": "session-1", "settings_revision": 1,
        "active_view": "context", "focus_control_id": None,
    }) == ConversationSettingsReturnIntent("session-1", 1, "context", None)
    assert ConversationSettingsReturnIntent("session-1", 0, "model", None).settings_revision == 0


@pytest.mark.parametrize("focus", ["console-settings-secret", "bad", "x" * 129])
def test_return_target_rejects_focus_outside_bounded_allowlist(focus: str) -> None:
    context = {
        "session_id": "session-1", "settings_revision": 2, "active_view": "model",
        "focus_control_id": focus, "return_revision": 7, "outcome": "without_saving",
    }
    assert ConsoleSettingsReturnTarget.from_context(context) is None
