from __future__ import annotations

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_provider_identity import (
    resolve_provider_list_key,
)


def test_resolves_exact_top_level_provider_key_for_openrouter():
    providers = {"OpenRouter": ["openrouter/auto"], "OpenAI": ["gpt-4.1"]}

    result = resolve_provider_list_key("openrouter", providers)

    assert result.status == "resolved"
    assert result.provider_list_key == "OpenRouter"


def test_preserves_custom_2_key_spelling():
    providers = {"Custom_2": ["existing-model"]}

    result = resolve_provider_list_key("custom_2", providers)

    assert result.status == "resolved"
    assert result.provider_list_key == "Custom_2"


def test_ambiguous_normalized_provider_keys_refuse_persistence():
    providers = {"Custom": ["a"], "custom": ["b"]}

    result = resolve_provider_list_key("custom", providers)

    assert result.status == "ambiguous"
    assert sorted(result.matches) == ["Custom", "custom"]


def test_missing_provider_key_reports_missing():
    result = resolve_provider_list_key("openrouter", {"OpenAI": ["gpt-4.1"]})

    assert result.status == "missing"
    assert result.provider_list_key is None


def test_resolves_custom_openai_execution_alias_to_saved_custom_key():
    providers = {"Custom": ["existing-model"]}

    result = resolve_provider_list_key("custom-openai-api", providers)

    assert result.status == "resolved"
    assert result.provider_list_key == "Custom"


def test_resolves_direct_llama_key_without_synthesizing_alias():
    providers = {"local_llamacpp": ["llama-model"]}

    result = resolve_provider_list_key("local-llamacpp", providers)

    assert result.status == "resolved"
    assert result.provider_list_key == "local_llamacpp"
