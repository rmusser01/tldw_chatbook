"""Unit tests for the inline Chat-Defaults API-key field state helpers."""

from tldw_chatbook.Chat.provider_readiness import (
    ProviderReadiness,
    ChatApiKeyFieldState,
    chat_api_key_field_state,
    chat_api_key_value_to_persist,
)


def _readiness(
    *,
    requires_api_key=True,
    ready=False,
    api_key=None,
    api_key_source=None,
    env_var=None,
    provider="OpenAI",
    provider_key="openai",
):
    return ProviderReadiness(
        provider=provider,
        provider_key=provider_key,
        requires_api_key=requires_api_key,
        ready=ready,
        api_key=api_key,
        api_key_source=api_key_source,
        env_var=env_var,
        reason="test",
        recovery=None,
    )


def test_keyless_provider_is_disabled_and_not_persistable():
    state = chat_api_key_field_state(
        _readiness(requires_api_key=False, ready=True, provider="Ollama", provider_key="ollama"),
        locked=False,
    )
    assert state.disabled is True
    assert state.can_persist is False
    assert state.value == ""
    assert "No API key needed" in state.placeholder


def test_locked_config_is_disabled_with_unlock_hint():
    state = chat_api_key_field_state(
        _readiness(ready=True, api_key="test-secret-key", api_key_source="config:api_settings.openai.api_key"),
        locked=True,
    )
    assert state.disabled is True
    assert state.can_persist is False
    assert state.value == ""
    assert "Unlock config" in state.placeholder


def test_config_key_is_prefilled():
    state = chat_api_key_field_state(
        _readiness(ready=True, api_key="test-key-abc123", api_key_source="config:api_settings.openai.api_key"),
        locked=False,
    )
    assert state.disabled is False
    assert state.value == "test-key-abc123"
    assert state.can_persist is True


def test_env_key_shows_hint_and_empty_value():
    state = chat_api_key_field_state(
        _readiness(ready=True, api_key="test-env-key", api_key_source="env:OPENAI_API_KEY", env_var="OPENAI_API_KEY"),
        locked=False,
    )
    assert state.value == ""
    assert state.disabled is False
    assert state.can_persist is True
    assert "OPENAI_API_KEY" in state.placeholder


def test_missing_key_is_empty_and_persistable():
    state = chat_api_key_field_state(_readiness(ready=False), locked=False)
    assert state.value == ""
    assert state.disabled is False
    assert state.can_persist is True


def test_persist_skips_when_not_persistable():
    state = ChatApiKeyFieldState(value="", disabled=True, placeholder="", can_persist=False)
    assert chat_api_key_value_to_persist("test-new-key", state) is None


def test_persist_skips_blank_and_placeholder():
    state = ChatApiKeyFieldState(value="", disabled=False, placeholder="", can_persist=True)
    assert chat_api_key_value_to_persist("   ", state) is None
    assert chat_api_key_value_to_persist("<API_KEY_HERE>", state) is None


def test_persist_skips_unchanged_config_value():
    state = ChatApiKeyFieldState(value="test-key-abc123", disabled=False, placeholder="", can_persist=True)
    assert chat_api_key_value_to_persist("test-key-abc123", state) is None


def test_persist_returns_stripped_new_value():
    state = ChatApiKeyFieldState(value="test-old-key", disabled=False, placeholder="", can_persist=True)
    assert chat_api_key_value_to_persist("  test-new-key  ", state) == "test-new-key"
