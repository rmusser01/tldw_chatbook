from __future__ import annotations

import builtins
import sys

import pytest

from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import DiscoveredModel
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


def test_none_provider_reports_missing_without_crashing():
    result = resolve_provider_list_key(None, {"OpenAI": ["gpt-4.1"]})

    assert result.status == "missing"
    assert result.requested_provider == ""
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


def test_resolving_non_direct_provider_does_not_import_chat_functions(monkeypatch):
    sys.modules.pop("tldw_chatbook.Chat.Chat_Functions", None)
    original_import = builtins.__import__

    def rejecting_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_chatbook.Chat.Chat_Functions":
            raise AssertionError("Chat_Functions import is not allowed")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)

    result = resolve_provider_list_key(
        "custom-openai-api",
        {"Custom": ["existing-model"]},
    )

    assert result.status == "resolved"
    assert result.provider_list_key == "Custom"
    assert "tldw_chatbook.Chat.Chat_Functions" not in sys.modules


def test_discovered_model_metadata_is_copied_from_caller_mapping():
    metadata = {
        "owned": False,
        "nested": {"modalities": ["text"]},
    }

    model = DiscoveredModel(
        provider="openrouter",
        provider_list_key="OpenRouter",
        model_id="openrouter/auto",
        display_name="openrouter/auto",
        source="runtime_discovered",
        endpoint_fingerprint="endpoint",
        discovered_at="2026-06-04T00:00:00Z",
        metadata_raw_safe=metadata,
    )
    metadata["owned"] = True
    metadata["nested"]["modalities"].append("vision")

    assert model.metadata_raw_safe["owned"] is False
    assert model.metadata_raw_safe["nested"]["modalities"] == ("text",)


def test_discovered_model_metadata_rejects_direct_mutation():
    model = DiscoveredModel(
        provider="openrouter",
        provider_list_key="OpenRouter",
        model_id="openrouter/auto",
        display_name="openrouter/auto",
        source="runtime_discovered",
        endpoint_fingerprint="endpoint",
        discovered_at="2026-06-04T00:00:00Z",
        metadata_raw_safe={"nested": {"owned": False}},
    )

    with pytest.raises(TypeError):
        model.metadata_raw_safe["new"] = "value"

    with pytest.raises(TypeError):
        model.metadata_raw_safe["nested"]["owned"] = True
