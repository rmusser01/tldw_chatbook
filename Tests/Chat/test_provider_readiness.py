"""Provider readiness tests for first-run Chat guidance."""

import pytest

from tldw_chatbook.Chat import provider_readiness as provider_readiness_module
from tldw_chatbook.Chat.provider_readiness import (
    ProviderReadiness,
    get_provider_readiness,
)


def test_key_required_provider_reports_missing_key_without_value_leakage():
    readiness = get_provider_readiness(
        "OpenAI",
        {
            "api_settings": {
                "openai": {
                    "api_key": "",
                    "api_key_env_var": "OPENAI_API_KEY",
                }
            }
        },
        environ={},
    )

    assert readiness == ProviderReadiness(
        provider="OpenAI",
        provider_key="openai",
        requires_api_key=True,
        ready=False,
        api_key=None,
        api_key_source=None,
        env_var="OPENAI_API_KEY",
        reason="Missing API key",
        recovery="Set OPENAI_API_KEY or add api_key under [api_settings.openai].",
    )
    assert "OPENAI_API_KEY" in readiness.user_message
    assert "api_settings.openai" in readiness.user_message
    assert "sk-" not in readiness.user_message


def test_key_required_provider_uses_environment_key_without_displaying_it():
    readiness = get_provider_readiness(
        "Anthropic",
        {"api_settings": {"anthropic": {"api_key_env_var": "ANTHROPIC_API_KEY"}}},
        environ={"ANTHROPIC_API_KEY": "sk-ant-secret"},
    )

    assert readiness.ready is True
    assert readiness.requires_api_key is True
    assert readiness.api_key == "sk-ant-secret"
    assert readiness.api_key_source == "env:ANTHROPIC_API_KEY"
    assert "sk-ant-secret" not in readiness.user_message


def test_key_required_provider_uses_standard_environment_key_when_config_only_has_model():
    readiness = get_provider_readiness(
        "Mistral",
        {"api_settings": {"mistral": {"model": "open-mistral-nemo"}}},
        environ={"MISTRAL_API_KEY": "mistral-secret"},
    )

    assert readiness.ready is True
    assert readiness.requires_api_key is True
    assert readiness.api_key == "mistral-secret"
    assert readiness.api_key_source == "env:MISTRAL_API_KEY"
    assert "mistral-secret" not in readiness.user_message


def test_mistralai_defaults_to_mistral_environment_key():
    readiness = get_provider_readiness(
        "MistralAI",
        {"api_settings": {"mistralai": {"model": "open-mistral-nemo"}}},
        environ={"MISTRAL_API_KEY": "mistral-secret"},
    )

    assert readiness.ready is True
    assert readiness.api_key == "mistral-secret"
    assert readiness.env_var == "MISTRAL_API_KEY"


def test_placeholder_config_key_is_not_ready():
    readiness = get_provider_readiness(
        "OpenRouter",
        {
            "api_settings": {
                "openrouter": {
                    "api_key": "<API_KEY_HERE>",
                    "api_key_env_var": "OPENROUTER_API_KEY",
                }
            }
        },
        environ={},
    )

    assert readiness.ready is False
    assert readiness.api_key is None
    assert readiness.recovery == "Set OPENROUTER_API_KEY or add api_key under [api_settings.openrouter]."


@pytest.mark.parametrize(
    "value",
    ["", "<API_KEY_HERE>", "YOUR_KEY", "your_key", "your-api-key"],
)
def test_public_provider_api_key_validator_rejects_placeholder_values(value):
    assert provider_readiness_module.is_valid_provider_api_key(value) is False


def test_public_provider_api_key_validator_accepts_real_trimmed_key():
    assert provider_readiness_module.is_valid_provider_api_key("  sk-real-key  ") is True


def test_key_required_provider_names_are_case_insensitive():
    readiness = get_provider_readiness(
        "openai",
        {"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        environ={},
    )

    assert readiness.requires_api_key is True
    assert readiness.ready is False
    assert readiness.recovery == "Set OPENAI_API_KEY or add api_key under [api_settings.openai]."


def test_provider_settings_lookup_uses_normalized_config_key():
    readiness = get_provider_readiness(
        "Custom-2",
        {"api_settings": {"Custom-2": {"api_key": "configured-custom-key"}}},
        environ={},
    )

    assert readiness.provider_key == "custom_2"
    assert readiness.ready is True
    assert readiness.requires_api_key is False
    assert readiness.api_key == "configured-custom-key"
    assert readiness.api_key_source == "config:api_settings.custom_2.api_key"


def test_keyless_local_provider_is_ready_without_api_key():
    readiness = get_provider_readiness(
        "Ollama",
        {"api_settings": {"ollama": {"api_url": "http://localhost:11434"}}},
        environ={},
    )

    assert readiness.ready is True
    assert readiness.requires_api_key is False
    assert readiness.api_key is None
    assert readiness.user_message == "Ollama is ready. No API key is required."


@pytest.mark.parametrize("provider", ["vLLM", "Custom-2", "local-llm"])
def test_known_keyless_provider_aliases_are_ready_without_api_key(provider):
    readiness = get_provider_readiness(
        provider,
        {"api_settings": {}},
        environ={},
    )

    assert readiness.ready is True
    assert readiness.requires_api_key is False
    assert readiness.api_key is None


def test_unknown_provider_without_key_is_not_ready():
    readiness = get_provider_readiness(
        "OpenAi Typo",
        {"api_settings": {}},
        environ={},
    )

    assert readiness.ready is False
    assert readiness.requires_api_key is True
    assert readiness.api_key is None
    assert readiness.reason == "Unknown provider"
    assert readiness.recovery == (
        "Choose a supported provider or add api_key under [api_settings.openai_typo]."
    )
