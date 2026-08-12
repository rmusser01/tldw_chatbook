from dataclasses import fields

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat import provider_setup_persistence as persistence_module
from tldw_chatbook.Chat.provider_setup_persistence import (
    ProviderSetupDraft,
    build_provider_setup_mutation,
    persist_provider_setup,
    provider_credential_keys,
    provider_endpoint_key,
    provider_model_key,
    provider_setup_is_explicitly_configured,
    resolve_remembered_provider_model,
)
from tldw_chatbook.config import ConfigMutationResult


def _draft(**overrides) -> ProviderSetupDraft:
    values = {
        "provider": "llama_cpp",
        "model": "qwen",
        "endpoint": "http://127.0.0.1:8080/v1/models",
        "credential_source": "none",
        "credential_revision": 0,
        "draft_generation": 1,
    }
    values.update(overrides)
    return ProviderSetupDraft(**values)


@pytest.mark.parametrize(
    ("provider", "endpoint_key", "model_key"),
    [
        ("llama_cpp", "api_url", "model"),
        ("llama.cpp", "api_url", "model"),
        ("local_llamacpp", "api_url", "model"),
        ("custom", "api_url", "model"),
        ("custom-openai-api", "api_url", "model"),
        ("custom_openai_api", "api_url", "model"),
        ("custom_2", "api_url", "model"),
        ("custom-openai-api-2", "api_url", "model"),
        ("custom_openai_api_2", "api_url", "model"),
        ("ollama", "api_url", "model"),
        ("local_ollama", "api_url", "model"),
        ("openai", "api_base_url", "model"),
        ("OpenAI", "api_base_url", "model"),
        ("qwencloud", "api_base_url", "model"),
        ("mistral", "api_base_url", "model"),
        ("mistralai", "api_base_url", "model"),
    ],
)
def test_provider_ownership_tables_cover_established_aliases(
    provider, endpoint_key, model_key
):
    assert provider_endpoint_key(provider) == endpoint_key
    assert provider_model_key(provider) == model_key
    assert provider_credential_keys(provider) == ("api_key", "api_key_env_var")


@pytest.mark.parametrize(
    "provider",
    ["", "unknown", "custom---openai", "open ai", "../openai", "openai\u202e"],
)
def test_provider_ownership_tables_reject_unknown_or_malformed_keys(provider):
    with pytest.raises(ValueError, match="Provider is not supported"):
        provider_endpoint_key(provider)
    with pytest.raises(ValueError, match="Provider is not supported"):
        provider_model_key(provider)
    with pytest.raises(ValueError, match="Provider is not supported"):
        provider_credential_keys(provider)


def test_matching_chat_default_wins_over_legacy_provider_model():
    config = {
        "chat_defaults": {"provider": "custom", "model": "new-model"},
        "api_settings": {"custom": {"model": "old-model"}},
    }

    assert resolve_remembered_provider_model(config, "custom") == "new-model"


def test_cross_provider_chat_default_never_leaks_into_selected_provider():
    config = {
        "chat_defaults": {"provider": "openai", "model": "gpt-model"},
        "api_settings": {"custom": {"model": "custom-model"}},
    }

    assert resolve_remembered_provider_model(config, "custom") == "custom-model"


def test_cross_provider_chat_default_without_legacy_model_returns_none():
    config = {
        "chat_defaults": {"provider": "openai", "model": "gpt-model"},
        "api_settings": {"custom": {}},
    }

    assert resolve_remembered_provider_model(config, "custom") is None


@pytest.mark.parametrize(
    "model",
    ["x" * 121, "line\nbreak", "bidi\u202evalue", "\ud800"],
)
def test_remembered_model_rejects_unbounded_or_unsafe_values(model):
    config = {
        "chat_defaults": {"provider": "custom", "model": model},
        "api_settings": {"custom": {"model": model}},
    }

    assert resolve_remembered_provider_model(config, "custom") is None


def test_setup_mutation_updates_llama_pair_endpoint_model_and_confirmation():
    mutation = build_provider_setup_mutation(_draft(), {})

    assert mutation.section_values["chat_defaults"] == {
        "provider": "llama_cpp",
        "model": "qwen",
    }
    assert mutation.section_values["provider_setup.confirmed"] == {
        "llama_cpp": True
    }
    assert mutation.section_values["api_settings.llama_cpp"] == {
        "api_url": "http://127.0.0.1:8080",
        "model": "qwen",
    }
    assert mutation.delete_keys == {}
    assert mutation.semantic_identity is not None
    assert mutation.semantic_identity.provider_key == "llama_cpp"
    assert mutation.semantic_identity.connection_identity == (
        "llama_cpp",
        "http://127.0.0.1:8080",
    )


@pytest.mark.parametrize(
    ("provider", "endpoint", "section", "confirmation"),
    [
        (
            "custom",
            "https://example.test/proxy",
            "api_settings.custom",
            "custom",
        ),
        (
            "custom-openai-api",
            "https://example.test/proxy/v1",
            "api_settings.custom",
            "custom",
        ),
        (
            "custom_2",
            "https://example.test/proxy/v1/chat/completions",
            "api_settings.custom_2",
            "custom_2",
        ),
        (
            "custom_openai_api_2",
            "https://example.test/proxy/v1/models",
            "api_settings.custom_2",
            "custom_2",
        ),
    ],
)
def test_custom_setup_mutation_persists_full_chat_url_for_all_input_forms(
    provider, endpoint, section, confirmation
):
    mutation = build_provider_setup_mutation(
        _draft(provider=provider, endpoint=endpoint, model="model-a"), {}
    )

    assert mutation.section_values[section] == {
        "api_url": "https://example.test/proxy/v1/chat/completions",
        "model": "model-a",
    }
    assert mutation.section_values["chat_defaults"] == {
        "provider": confirmation,
        "model": "model-a",
    }
    assert mutation.section_values["provider_setup.confirmed"] == {
        confirmation: True
    }


def test_setup_mutation_preserves_an_established_config_section_alias():
    config = {
        "api_settings": {
            "OpenAI": {
                "api_base_url": "https://old.example.test/v1",
                "model": "old-model",
            }
        }
    }

    mutation = build_provider_setup_mutation(
        _draft(
            provider="OpenAI",
            endpoint="https://new.example.test/v1",
            model="new-model",
        ),
        config,
    )

    assert "api_settings.OpenAI" in mutation.section_values
    assert "api_settings.openai" not in mutation.section_values


def test_semantic_identity_collapses_default_https_port():
    mutation = build_provider_setup_mutation(
        _draft(
            provider="custom",
            endpoint="https://example.test:443/v1/models",
        ),
        {},
    )

    assert mutation.semantic_identity is not None
    assert mutation.semantic_identity.connection_identity == (
        "custom",
        "https://example.test/v1/chat/completions",
    )


def test_endpoint_clear_deletes_only_owned_endpoint_and_confirmation():
    config = {
        "provider_setup": {"confirmed": {"llama_cpp": True, "custom": True}},
        "api_settings": {
            "llama_cpp": {
                "api_url": "http://127.0.0.1:8080",
                "model": "old-model",
                "timeout": 90,
            }
        },
    }

    mutation = build_provider_setup_mutation(_draft(endpoint=""), config)

    assert mutation.section_values == {
        "api_settings.llama_cpp": {"model": "qwen"},
        "chat_defaults": {"provider": "llama_cpp", "model": "qwen"},
    }
    assert mutation.delete_keys == {
        "api_settings.llama_cpp": ("api_url",),
        "provider_setup.confirmed": ("llama_cpp",),
    }
    assert mutation.semantic_identity is None
    assert "custom" not in mutation.delete_keys["provider_setup.confirmed"]


def test_credential_replacement_and_clear_are_sparse_and_secret_safe():
    secret = "sk-super-secret-value"
    replacement = build_provider_setup_mutation(
        _draft(
            provider="openai",
            endpoint="https://api.example.test/v1",
            credential_source="draft",
            credential_revision=4,
            credential_value=secret,
        ),
        {},
    )

    assert replacement.section_values["api_settings.openai"]["api_key"] == secret
    assert secret not in repr(replacement)
    assert secret not in repr(replacement.semantic_identity)
    assert secret not in repr(_draft(credential_value=secret))

    cleared = build_provider_setup_mutation(
        _draft(
            provider="openai",
            endpoint="https://api.example.test/v1",
            credential_source="none",
            credential_revision=5,
            credential_value="",
        ),
        {},
    )
    assert "api_key" not in cleared.section_values["api_settings.openai"]
    assert cleared.delete_keys["api_settings.openai"] == ("api_key",)


def test_environment_credential_persists_only_variable_name_not_value(monkeypatch):
    monkeypatch.setenv("PRIVATE_PROVIDER_KEY", "environment-secret")

    mutation = build_provider_setup_mutation(
        _draft(
            provider="openai",
            endpoint="https://api.example.test/v1",
            credential_source="environment",
            credential_revision=3,
            credential_env_var="PRIVATE_PROVIDER_KEY",
        ),
        {},
    )

    assert mutation.section_values["api_settings.openai"]["api_key_env_var"] == (
        "PRIVATE_PROVIDER_KEY"
    )
    assert "environment-secret" not in repr(mutation)
    assert "environment-secret" not in str(mutation.section_values)


def test_credential_clear_can_remove_stored_and_environment_keys_together():
    mutation = build_provider_setup_mutation(
        _draft(
            provider="openai",
            endpoint="https://api.example.test/v1",
            credential_source="none",
            credential_revision=6,
            credential_value="",
            credential_env_var="",
        ),
        {},
    )

    assert mutation.delete_keys["api_settings.openai"] == (
        "api_key",
        "api_key_env_var",
    )


def test_setup_types_are_frozen_slotted_and_have_secret_free_repr():
    secret = "sk-private-draft-value"
    draft = _draft(credential_value=secret)
    mutation = build_provider_setup_mutation(draft, {})

    assert [item.name for item in fields(ProviderSetupDraft)] == [
        "provider",
        "model",
        "endpoint",
        "credential_source",
        "credential_revision",
        "draft_generation",
        "credential_value",
        "credential_env_var",
    ]
    assert not hasattr(draft, "__dict__")
    assert not hasattr(mutation, "__dict__")
    assert secret not in repr(draft)
    assert secret not in repr(mutation)
    with pytest.raises(AttributeError):
        draft.model = "changed"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:password@example.test/v1",
        "https://example.test/v1?token=secret",
        "https://example.test/v1#secret",
    ],
)
def test_setup_mutation_rejects_credential_bearing_or_ambiguous_endpoint(endpoint):
    with pytest.raises(ValueError, match="Endpoint is invalid") as error:
        build_provider_setup_mutation(_draft(endpoint=endpoint), {})

    assert "password" not in str(error.value)
    assert "secret" not in str(error.value)


def test_persist_provider_setup_delegates_once_and_preserves_typed_result(monkeypatch):
    calls = []
    expected = ConfigMutationResult(True, False, "cache_reload")

    def fake_writer(section_values, *, delete_keys=None):
        calls.append((section_values, delete_keys))
        return expected

    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        fake_writer,
    )
    mutation = build_provider_setup_mutation(_draft(), {})

    assert persist_provider_setup(mutation) is expected
    assert calls == [(mutation.section_values, mutation.delete_keys)]


def test_explicit_confirmation_is_authoritative_and_sparse():
    config = {
        "provider_setup": {"confirmed": {"llama_cpp": True, "custom": False}},
        "api_settings": {
            "llama_cpp": {"api_url": "http://127.0.0.1:8080"},
            "custom": {"api_url": "https://example.test/v1/chat/completions"},
        },
    }

    assert provider_setup_is_explicitly_configured(config, "llama_cpp") is True
    assert provider_setup_is_explicitly_configured(config, "custom") is False


@pytest.mark.parametrize(
    "provider_setup",
    [None, [], {"confirmed": []}, {"confirmed": {"llama_cpp": "true"}}],
)
def test_malformed_confirmation_tables_fail_closed(provider_setup):
    config = {
        "provider_setup": provider_setup,
        "api_settings": {"llama_cpp": {"api_url": "http://other.test:8080"}},
    }

    assert provider_setup_is_explicitly_configured(config, "llama_cpp") is False


def test_config_without_confirmation_uses_legacy_readiness_heuristic():
    config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "my-model"},
        "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}},
    }

    assert provider_setup_is_explicitly_configured(config, "llama_cpp") is True


def test_template_endpoint_without_user_acceptance_is_not_explicitly_configured():
    assert (
        provider_setup_is_explicitly_configured(
            config_module.DEFAULT_CONFIG_FROM_TOML,
            "llama_cpp",
        )
        is False
    )
