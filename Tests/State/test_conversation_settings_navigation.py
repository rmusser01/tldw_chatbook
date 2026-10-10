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


@pytest.mark.parametrize("field", ["endpoint", None])
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
