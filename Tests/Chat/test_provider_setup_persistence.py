import threading
from dataclasses import fields, replace
from types import MappingProxyType

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat import provider_setup_persistence as persistence_module
from tldw_chatbook.Chat.provider_setup_persistence import (
    ProviderSetupDraft,
    ProviderSetupMutation,
    build_provider_setup_mutation,
    canonical_provider_key,
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


def _bind_atomic_expectation(
    mutation,
    *,
    snapshot,
    identity,
):
    guard = persistence_module.ProviderSetupWriteGuard()
    expectation = guard.arm(identity)
    expected_state = persistence_module.capture_expected_provider_setup_state(
        snapshot,
        identity=identity,
    )
    persistence_module.bind_provider_setup_write_expectation(
        mutation,
        guard=guard,
        expectation=expectation,
        expected_state=expected_state,
    )
    return guard, expected_state


def test_console_endpoint_save_uses_bound_provider_setup_transaction(monkeypatch):
    """Console's endpoint action inherits the provider writer's atomic CAS guard."""
    snapshot = config_module.AtomicConfigSnapshot(
        4,
        {
            "api_settings": {
                "ollama": {
                    "api_url": "http://127.0.0.1:11434",
                    "model": "qwen-old",
                    "credential_source": "none",
                }
            },
            "chat_defaults": {"provider": "ollama", "model": "qwen-old"},
        },
    )
    draft = _draft(
        provider="ollama",
        model="qwen-new",
        endpoint="http://127.0.0.1:22434",
    )
    mutation = build_provider_setup_mutation(draft, snapshot.values)
    identity = persistence_module.ProviderSetupWriteIdentity(
        provider_key="ollama",
        connection_identity=persistence_module.canonical_connection_identity(
            "ollama", "http://127.0.0.1:22434"
        ),
        credential_source="none",
        credential_revision=0,
        model_id="qwen-new",
        model_provenance="manual",
    )
    guard = persistence_module.ProviderSetupWriteGuard()
    expectation = guard.arm(identity)
    expected_state = persistence_module.bind_provider_setup_precondition(
        persistence_module.capture_provider_setup_precondition(
            snapshot,
            provider="ollama",
        ),
        identity=identity,
    )
    persistence_module.bind_provider_setup_write_expectation(
        mutation,
        guard=guard,
        expectation=expectation,
        expected_state=expected_state,
    )
    calls = 0

    def writer(*_args, locked_snapshot_precondition=None, **_kwargs):
        nonlocal calls
        calls += 1
        assert locked_snapshot_precondition(snapshot)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        writer,
    )

    result = persist_provider_setup(mutation)

    assert result.fully_applied is True
    assert calls == 1
    assert mutation.section_values["api_settings.ollama"] == {
        "model": "qwen-new",
        # Ollama's provider owner stores the exact chat-completions route,
        # while its canonical connection identity remains the server root.
        "api_url": "http://127.0.0.1:22434/v1/chat/completions",
        "credential_source": "none",
    }
    assert mutation.section_values["chat_defaults"] == {
        "provider": "ollama",
        "model": "qwen-new",
    }


def test_provider_setup_precondition_rebinds_original_locked_state():
    original = config_module.AtomicConfigSnapshot(
        7,
        {
            "api_settings": {
                "moonshot": {
                    "api_region": "china",
                    "api_base_url": "https://api.moonshot.cn/v1",
                }
            }
        },
    )
    changed = config_module.AtomicConfigSnapshot(
        8,
        {
            "api_settings": {
                "moonshot": {
                    "api_region": "global",
                    "api_base_url": "https://api.moonshot.ai/v1",
                }
            }
        },
    )
    identity = persistence_module.ProviderSetupWriteIdentity(
        provider_key="moonshot",
        connection_identity=persistence_module.canonical_connection_identity(
            "moonshot", "https://api.moonshot.cn/v1"
        ),
        credential_source="none",
        credential_revision=0,
        model_id="moonshot-cn-model",
        model_provenance="discovered",
    )

    precondition = persistence_module.capture_provider_setup_precondition(
        original,
        provider="moonshot",
    )
    expected = persistence_module.bind_provider_setup_precondition(
        precondition,
        identity=identity,
    )

    assert expected._matches_snapshot(original) is True
    assert expected._matches_snapshot(changed) is False


def test_provider_setup_precondition_allows_unrelated_locked_change():
    original = config_module.AtomicConfigSnapshot(
        3,
        {
            "api_settings": {"custom": {"api_url": "https://a.example/v1"}},
            "general": {"users_name": "Before"},
        },
    )
    unrelated = config_module.AtomicConfigSnapshot(
        4,
        {
            "api_settings": {"custom": {"api_url": "https://a.example/v1"}},
            "general": {"users_name": "After"},
        },
    )
    identity = persistence_module.ProviderSetupWriteIdentity(
        provider_key="custom",
        connection_identity=persistence_module.canonical_connection_identity(
            "custom", "https://a.example/v1"
        ),
        credential_source="none",
        credential_revision=0,
        model_id="manual-model",
        model_provenance="manual",
    )

    precondition = persistence_module.capture_provider_setup_precondition(
        original,
        provider="custom",
    )
    expected = persistence_module.bind_provider_setup_precondition(
        precondition,
        identity=identity,
    )

    assert expected._matches_snapshot(unrelated) is True


def test_provider_setup_postcondition_projects_exact_atomic_mutation():
    before = config_module.AtomicConfigSnapshot(
        11,
        {
            "api_settings": {
                "custom": {
                    "api_url": "https://before.example/v1/chat/completions",
                    "api_key": "postcondition-key-a",
                    "model": "before-model",
                }
            },
            "chat_defaults": {"provider": "custom", "model": "before-model"},
            "general": {"users_name": "Unchanged"},
        },
    )
    draft = _draft(
        provider="custom",
        endpoint="https://after.example/v1/chat/completions",
        model="after-model",
        credential_source="draft",
        credential_revision=12,
        credential_value="postcondition-key-b",
    )
    mutation = build_provider_setup_mutation(draft, before.values)
    identity = persistence_module.ProviderSetupWriteIdentity(
        provider_key="custom",
        connection_identity=persistence_module.canonical_connection_identity(
            "custom", "https://after.example/v1/chat/completions"
        ),
        credential_source="stored",
        credential_revision=12,
        model_id="after-model",
        model_provenance="manual",
    )

    expected = persistence_module.project_provider_setup_expected_state(
        before,
        mutation=mutation,
        identity=identity,
    )
    after = config_module.AtomicConfigSnapshot(
        12,
        {
            "api_settings": {
                "custom": {
                    "api_url": "https://after.example/v1/chat/completions",
                    "api_key": "postcondition-key-b",
                    "credential_source": "stored",
                    "model": "after-model",
                }
            },
            "chat_defaults": {"provider": "custom", "model": "after-model"},
            "provider_setup": {"confirmed": {"custom": True}},
            "general": {"users_name": "Unchanged"},
        },
    )
    unrelated = config_module.AtomicConfigSnapshot(
        13,
        {
            **after.values,
            "general": {"users_name": "Changed"},
        },
    )
    changed_credential = config_module.AtomicConfigSnapshot(
        14,
        {
            **after.values,
            "api_settings": {
                "custom": {
                    **after.values["api_settings"]["custom"],
                    "api_key": "postcondition-key-c",
                }
            },
        },
    )

    assert expected._matches_snapshot(after) is True
    assert expected._matches_snapshot(unrelated) is True
    assert expected._matches_snapshot(changed_credential) is False
    rendered = repr(expected)
    assert "postcondition-key-a" not in rendered
    assert "postcondition-key-b" not in rendered
    assert "postcondition-key-c" not in rendered


def test_provider_setup_precondition_repr_never_exposes_credential():
    canary = "selection-precondition-secret-canary"
    snapshot = config_module.AtomicConfigSnapshot(
        2,
        {"api_settings": {"custom": {"api_key": canary}}},
    )

    precondition = persistence_module.capture_provider_setup_precondition(
        snapshot,
        provider="custom",
    )

    assert canary not in repr(precondition)
    assert not hasattr(precondition, "__dict__")


def _build_bound_first_run_mutation(
    *,
    snapshot,
    provider: str,
    endpoint: str,
    model: str = "selected-model",
):
    from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state

    draft = wizard_state.FirstRunProviderDraft(
        provider,
        endpoint,
        wizard_state.ProviderCredentialDraft("stored", "", 7),
    )
    effective = wizard_state.resolve_first_run_provider_draft(draft, snapshot.values)
    discovery_key = wizard_state.build_first_run_model_discovery_key(effective)
    identity = persistence_module.ProviderSetupWriteIdentity(
        provider_key=discovery_key.provider_key,
        connection_identity=discovery_key.connection_identity,
        credential_source=discovery_key.credential_source,
        credential_revision=discovery_key.credential_revision,
        model_id=model,
        model_provenance="discovered",
    )
    mutation = wizard_state.build_first_run_provider_commit(
        draft,
        model,
        snapshot.values,
    )
    guard, expected_state = _bind_atomic_expectation(
        mutation,
        snapshot=snapshot,
        identity=identity,
    )
    return mutation, guard, expected_state


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
    ("provider", "expected"),
    [
        ("Anthropic", "anthropic"),
        ("Aphrodite", "aphrodite"),
        ("Cohere", "cohere"),
        ("Custom", "custom"),
        ("Custom OpenAI", "custom"),
        ("Custom OpenAI API", "custom"),
        ("custom-openai", "custom"),
        ("custom_openai", "custom"),
        ("custom-openai-api", "custom"),
        ("custom_openai_api", "custom"),
        ("Custom 2", "custom_2"),
        ("Custom-2", "custom_2"),
        ("Custom OpenAI 2", "custom_2"),
        ("Custom OpenAI API 2", "custom_2"),
        ("Custom OpenAI API-2", "custom_2"),
        ("custom-2", "custom_2"),
        ("custom-openai-2", "custom_2"),
        ("custom_openai_2", "custom_2"),
        ("custom-openai-api-2", "custom_2"),
        ("custom_openai_api_2", "custom_2"),
        ("DeepSeek", "deepseek"),
        ("Google", "google"),
        ("Groq", "groq"),
        ("HuggingFace", "huggingface"),
        ("Hugging Face", "huggingface"),
        ("Llama_cpp", "llama_cpp"),
        ("llama.cpp", "llama_cpp"),
        ("local llama.cpp", "local_llamacpp"),
        ("local-llamacpp", "local_llamacpp"),
        ("local-llm", "local_llm"),
        ("mlx_lm", "local_mlx_lm"),
        ("Mistral", "mistral"),
        ("MistralAI", "mistralai"),
        ("Moonshot", "moonshot"),
        ("Ollama", "ollama"),
        ("Oobabooga", "oobabooga"),
        ("OpenAI", "openai"),
        ("OpenRouter", "openrouter"),
        ("QwenCloud", "qwencloud"),
        ("TabbyAPI", "tabbyapi"),
        ("vLLM", "vllm"),
        ("ZAI", "zai"),
    ],
)
def test_canonical_provider_key_is_the_public_alias_authority(provider, expected):
    assert canonical_provider_key(provider) == expected


@pytest.mark.parametrize("provider", ["mistral", "mistralai"])
def test_mistral_provider_entries_keep_distinct_config_owners(provider):
    mutation = build_provider_setup_mutation(
        _draft(
            provider=provider,
            model=f"{provider}-model",
            endpoint="https://api.mistral.ai/v1",
        ),
        {
            "api_settings": {
                "mistral": {"model": "legacy-model"},
                "mistralai": {"model": "catalog-model"},
            }
        },
    )

    assert f"api_settings.{provider}" in mutation.section_values
    assert mutation.section_values[f"api_settings.{provider}"]["model"] == (
        f"{provider}-model"
    )
    assert mutation.section_values["chat_defaults"] == {
        "provider": provider,
        "model": f"{provider}-model",
    }
    assert mutation.section_values["provider_setup.confirmed"] == {provider: True}


@pytest.mark.parametrize(
    "provider",
    ["", "unknown", "custom---openai", "open ai", "../openai", "openai\u202e"],
)
def test_provider_ownership_tables_reject_unknown_or_malformed_keys(provider):
    with pytest.raises(ValueError, match="Provider is not supported"):
        canonical_provider_key(provider)
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
    assert mutation.section_values["provider_setup.confirmed"] == {"llama_cpp": True}
    assert mutation.section_values["api_settings.llama_cpp"] == {
        "api_url": "http://127.0.0.1:8080",
        "model": "qwen",
        "credential_source": "none",
    }
    assert mutation.delete_keys == {
        "api_settings.llama_cpp": ("api_key", "api_key_env_var")
    }
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
        "credential_source": "none",
    }
    assert mutation.section_values["chat_defaults"] == {
        "provider": confirmation,
        "model": "model-a",
    }
    assert mutation.section_values["provider_setup.confirmed"] == {confirmation: True}


@pytest.mark.parametrize(
    ("provider", "config", "endpoint", "expected_key", "expected_endpoint"),
    [
        ("llama_cpp", {}, "https://example.test", "api_url", "https://example.test"),
        ("llama_cpp", {}, "https://example.test/v1", "api_url", "https://example.test"),
        (
            "llama_cpp",
            {},
            "https://example.test/v1/chat/completions",
            "api_url",
            "https://example.test",
        ),
        (
            "llama_cpp",
            {},
            "https://example.test/v1/models",
            "api_url",
            "https://example.test",
        ),
        (
            "local_llamacpp",
            {},
            "https://example.test/proxy/v1/models",
            "api_url",
            "https://example.test/proxy",
        ),
        (
            "custom",
            {},
            "https://example.test",
            "api_url",
            "https://example.test/v1/chat/completions",
        ),
        (
            "custom",
            {},
            "https://example.test/v1",
            "api_url",
            "https://example.test/v1/chat/completions",
        ),
        (
            "custom_2",
            {},
            "https://example.test/v1/chat/completions",
            "api_url",
            "https://example.test/v1/chat/completions",
        ),
        (
            "custom_openai_api",
            {},
            "https://example.test/v1/models",
            "api_url",
            "https://example.test/v1/chat/completions",
        ),
        (
            "custom_openai_api_2",
            {},
            "https://example.test/proxy/v1/models",
            "api_url",
            "https://example.test/proxy/v1/chat/completions",
        ),
        (
            "openai",
            {"api_settings": {"openai": {"api_base_url": "https://old.test/v1"}}},
            "https://example.test",
            "api_base_url",
            "https://example.test/v1",
        ),
        (
            "openai",
            {"api_settings": {"openai": {"api_base_url": "https://old.test/v1"}}},
            "https://example.test/v1",
            "api_base_url",
            "https://example.test/v1",
        ),
        (
            "openai",
            {"api_settings": {"openai": {"api_base_url": "https://old.test/v1"}}},
            "https://example.test/v1/chat/completions",
            "api_base_url",
            "https://example.test/v1",
        ),
        (
            "openai",
            {"api_settings": {"openai": {"api_base_url": "https://old.test/v1"}}},
            "https://example.test/v1/models",
            "api_base_url",
            "https://example.test/v1",
        ),
        (
            "qwencloud",
            {"api_settings": {"qwencloud": {"api_base_url": "https://old.test/v1"}}},
            "https://example.test/proxy/v1/models",
            "api_base_url",
            "https://example.test/proxy/v1",
        ),
        (
            "ollama",
            {},
            "https://example.test",
            "api_url",
            "https://example.test/v1/chat/completions",
        ),
        (
            "ollama",
            {},
            "https://example.test/v1",
            "api_url",
            "https://example.test/v1/chat/completions",
        ),
        (
            "ollama",
            {},
            "https://example.test/v1/chat/completions",
            "api_url",
            "https://example.test/v1/chat/completions",
        ),
        (
            "ollama",
            {},
            "https://example.test/v1/models",
            "api_url",
            "https://example.test/v1/chat/completions",
        ),
        (
            "ollama",
            {},
            "https://example.test/proxy/v1/models",
            "api_url",
            "https://example.test/proxy/v1/chat/completions",
        ),
    ],
)
def test_persisted_endpoint_shape_matches_provider_and_owned_key(
    provider, config, endpoint, expected_key, expected_endpoint
):
    mutation = build_provider_setup_mutation(
        _draft(provider=provider, endpoint=endpoint), config
    )
    provider_values = next(
        values
        for section, values in mutation.section_values.items()
        if section.startswith("api_settings.")
    )

    assert provider_values[expected_key] == expected_endpoint


@pytest.mark.parametrize(
    ("endpoint_key", "expected_endpoint"),
    [
        ("api_base_url", "https://new.example.test/v1"),
        ("api_base", "https://new.example.test/v1"),
        ("base_url", "https://new.example.test/v1"),
        ("api_url", "https://new.example.test/v1/chat/completions"),
        ("endpoint", "https://new.example.test/v1/chat/completions"),
    ],
)
def test_setup_mutation_preserves_each_existing_endpoint_key(
    endpoint_key, expected_endpoint
):
    config = {"api_settings": {"openai": {endpoint_key: "https://old.example.test/v1"}}}

    mutation = build_provider_setup_mutation(
        _draft(provider="openai", endpoint="https://new.example.test/v1/models"),
        config,
    )
    provider_values = mutation.section_values["api_settings.openai"]

    assert provider_values[endpoint_key] == expected_endpoint
    assert (
        not (
            {"api_base_url", "api_base", "base_url", "api_url", "endpoint"}
            - {endpoint_key}
        )
        & provider_values.keys()
    )


def test_existing_endpoint_key_uses_settings_read_precedence_without_shadow_key():
    config = {
        "api_settings": {
            "openai": {
                "api_url": "https://shadowed.example.test/v1/chat/completions",
                "base_url": "https://selected.example.test/v1",
                "endpoint": "https://also-shadowed.example.test/v1/chat/completions",
            }
        }
    }

    mutation = build_provider_setup_mutation(
        _draft(provider="openai", endpoint="https://new.example.test/proxy/v1/models"),
        config,
    )

    assert mutation.section_values["api_settings.openai"] == {
        "base_url": "https://new.example.test/proxy/v1",
        "model": "qwen",
        "credential_source": "none",
    }


def test_endpoint_clear_deletes_the_existing_owned_alias_only():
    config = {
        "api_settings": {
            "openai": {
                "api_base": "https://selected.example.test/v1",
                "api_url": "https://shadowed.example.test/v1/chat/completions",
            }
        }
    }

    mutation = build_provider_setup_mutation(
        _draft(provider="openai", endpoint=""), config
    )

    assert mutation.delete_keys["api_settings.openai"] == (
        "api_base",
        "api_key",
        "api_key_env_var",
    )


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
        "api_settings.llama_cpp": {
            "model": "qwen",
            "credential_source": "none",
        },
        "chat_defaults": {"provider": "llama_cpp", "model": "qwen"},
    }
    assert mutation.delete_keys == {
        "api_settings.llama_cpp": (
            "api_url",
            "api_key",
            "api_key_env_var",
        ),
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
    assert replacement.semantic_identity.credential_source == "stored"
    assert replacement.delete_keys["api_settings.openai"] == ("api_key_env_var",)
    assert secret not in repr(replacement)
    assert secret not in repr(replacement.semantic_identity)
    assert secret not in repr(
        _draft(credential_source="draft", credential_value=secret)
    )

    cleared = build_provider_setup_mutation(
        _draft(
            provider="openai",
            endpoint="https://api.example.test/v1",
            credential_source="none",
            credential_revision=5,
        ),
        {},
    )
    assert "api_key" not in cleared.section_values["api_settings.openai"]
    assert (
        cleared.section_values["api_settings.openai"]["credential_source"]
        == "none"
    )
    assert cleared.delete_keys["api_settings.openai"] == (
        "api_key",
        "api_key_env_var",
    )


def test_credential_source_is_persisted_for_each_authoritative_auth_decision():
    config = {
        "api_settings": {
            "custom": {
                "api_key": "saved-key-canary",
                "api_key_env_var": "CUSTOM_API_KEY",
            }
        }
    }

    cleared = build_provider_setup_mutation(
        _draft(
            provider="custom",
            endpoint="https://keyless.example.test/v1/chat/completions",
            credential_source="none",
        ),
        config,
    )
    kept = build_provider_setup_mutation(
        _draft(
            provider="custom",
            endpoint="https://keyless.example.test/v1/chat/completions",
            credential_source="stored",
        ),
        config,
    )
    environment = build_provider_setup_mutation(
        _draft(
            provider="custom",
            endpoint="https://keyless.example.test/v1/chat/completions",
            credential_source="environment",
            credential_env_var="CUSTOM_API_KEY",
        ),
        config,
    )

    assert cleared.section_values["api_settings.custom"]["credential_source"] == (
        "none"
    )
    assert kept.section_values["api_settings.custom"]["credential_source"] == (
        "stored"
    )
    assert environment.section_values["api_settings.custom"][
        "credential_source"
    ] == "environment"


def test_unset_environment_declaration_changes_credential_routing_identity(
    monkeypatch,
):
    monkeypatch.delenv("CUSTOM_API_KEY", raising=False)
    original = config_module.AtomicConfigSnapshot(
        1,
        {
            "api_settings": {
                "custom": {
                    "api_url": "https://keyless.example.test/v1/chat/completions",
                    "api_key_env_var": "CUSTOM_API_KEY",
                }
            }
        },
    )
    sparse = config_module.AtomicConfigSnapshot(
        2,
        {
            "api_settings": {
                "custom": {
                    "api_url": "https://keyless.example.test/v1/chat/completions",
                }
            }
        },
    )
    identity = persistence_module.ProviderSetupWriteIdentity(
        provider_key="custom",
        connection_identity=persistence_module.canonical_connection_identity(
            "custom", "https://keyless.example.test/v1/chat/completions"
        ),
        credential_source="none",
        credential_revision=0,
        model_id="manual-model",
        model_provenance="manual",
    )
    expected = persistence_module.bind_provider_setup_precondition(
        persistence_module.capture_provider_setup_precondition(
            original,
            provider="custom",
        ),
        identity=identity,
    )

    assert expected._matches_snapshot(sparse) is False

    monkeypatch.setenv("CUSTOM_API_KEY", "appeared-environment-canary")
    assert expected._matches_snapshot(original) is False


def test_unset_environment_variable_name_change_fails_provider_setup_cas(
    monkeypatch,
):
    """An unset ENV_A -> ENV_B config race must not evade the provider CAS."""
    monkeypatch.delenv("CUSTOM_ENV_A", raising=False)
    monkeypatch.delenv("CUSTOM_ENV_B", raising=False)
    original = config_module.AtomicConfigSnapshot(
        1,
        {
            "api_settings": {
                "custom": {
                    "api_url": "https://keyless.example.test/v1/chat/completions",
                    "credential_source": "environment",
                    "api_key_env_var": "CUSTOM_ENV_A",
                }
            }
        },
    )
    raced = config_module.AtomicConfigSnapshot(
        2,
        {
            "api_settings": {
                "custom": {
                    "api_url": "https://keyless.example.test/v1/chat/completions",
                    "credential_source": "environment",
                    "api_key_env_var": "CUSTOM_ENV_B",
                }
            }
        },
    )
    identity = persistence_module.ProviderSetupWriteIdentity(
        provider_key="custom",
        connection_identity=persistence_module.canonical_connection_identity(
            "custom", "https://keyless.example.test/v1/chat/completions"
        ),
        credential_source="environment",
        credential_revision=0,
        model_id="manual-model",
        model_provenance="manual",
    )
    expected = persistence_module.bind_provider_setup_precondition(
        persistence_module.capture_provider_setup_precondition(
            original,
            provider="custom",
        ),
        identity=identity,
    )

    assert expected._matches_snapshot(original) is True
    assert expected._matches_snapshot(raced) is False
    assert "CUSTOM_ENV_A" not in repr(expected)
    assert "CUSTOM_ENV_B" not in repr(expected)


def test_present_environment_rotation_changes_credential_identity(monkeypatch):
    settings = {
        "api_settings": {
            "custom": {
                "api_url": "https://keyless.example.test/v1/chat/completions",
                "credential_source": "environment",
                "api_key_env_var": "CUSTOM_API_KEY",
            }
        }
    }
    snapshot = config_module.AtomicConfigSnapshot(1, settings)
    identity = persistence_module.ProviderSetupWriteIdentity(
        provider_key="custom",
        connection_identity=persistence_module.canonical_connection_identity(
            "custom", "https://keyless.example.test/v1/chat/completions"
        ),
        credential_source="environment",
        credential_revision=1,
        model_id="manual-model",
        model_provenance="manual",
    )
    monkeypatch.setenv("CUSTOM_API_KEY", "environment-key-a")
    expected = persistence_module.bind_provider_setup_precondition(
        persistence_module.capture_provider_setup_precondition(
            snapshot,
            provider="custom",
        ),
        identity=identity,
    )

    assert expected._matches_snapshot(snapshot) is True
    monkeypatch.setenv("CUSTOM_API_KEY", "environment-key-b")
    assert expected._matches_snapshot(snapshot) is False
    rendered = repr(expected)
    assert "environment-key-a" not in rendered
    assert "environment-key-b" not in rendered


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
    assert mutation.delete_keys["api_settings.openai"] == ("api_key",)
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


def test_stored_credential_can_reuse_existing_value_and_deletes_environment_source():
    config = {
        "api_settings": {
            "openai": {
                "api_key": "existing-secret",
                "api_key_env_var": "OLD_ENV",
            }
        }
    }

    mutation = build_provider_setup_mutation(
        _draft(provider="openai", credential_source="stored"), config
    )

    assert (
        mutation.section_values["api_settings.openai"]["api_key"] == "existing-secret"
    )
    assert mutation.delete_keys["api_settings.openai"] == ("api_key_env_var",)
    assert "existing-secret" not in repr(mutation)


def test_environment_credential_can_reuse_existing_name_and_deletes_stored_source():
    config = {
        "api_settings": {
            "openai": {
                "api_key": "existing-secret",
                "api_key_env_var": "EXISTING_ENV",
            }
        }
    }

    mutation = build_provider_setup_mutation(
        _draft(provider="openai", credential_source="environment"), config
    )

    assert (
        mutation.section_values["api_settings.openai"]["api_key_env_var"]
        == "EXISTING_ENV"
    )
    assert mutation.delete_keys["api_settings.openai"] == ("api_key",)
    assert "existing-secret" not in repr(mutation)


@pytest.mark.parametrize(
    "overrides",
    [
        {"credential_source": "draft", "credential_value": None},
        {
            "credential_source": "draft",
            "credential_value": "new",
            "credential_env_var": "ENV",
        },
        {"credential_source": "stored", "credential_value": None},
        {
            "credential_source": "stored",
            "credential_env_var": "ENV",
            "credential_value": "new",
        },
        {"credential_source": "environment", "credential_env_var": None},
        {
            "credential_source": "environment",
            "credential_env_var": "ENV",
            "credential_value": "new",
        },
        {"credential_source": "none", "credential_value": "new"},
        {"credential_source": "none", "credential_env_var": "ENV"},
    ],
)
def test_credential_source_rejects_contradictory_or_incomplete_states(overrides):
    with pytest.raises(ValueError, match="Credential"):
        build_provider_setup_mutation(_draft(provider="openai", **overrides), {})


def test_setup_types_are_frozen_slotted_and_have_secret_free_repr():
    secret = "sk-private-draft-value"
    draft = _draft(credential_source="draft", credential_value=secret)
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
    assert [item.name for item in fields(ProviderSetupMutation)] == [
        "section_values",
        "delete_keys",
        "semantic_identity",
    ]
    assert not hasattr(draft, "__dict__")
    assert not hasattr(mutation, "__dict__")
    assert secret not in repr(draft)
    assert secret not in repr(mutation)
    with pytest.raises(AttributeError):
        draft.model = "changed"


def test_setup_mutation_requires_exact_immutable_mapping_shapes():
    valid = build_provider_setup_mutation(_draft(), {})

    structurally_valid_but_unissued = ProviderSetupMutation(
        section_values=MappingProxyType(
            {
                section: MappingProxyType(dict(values))
                for section, values in valid.section_values.items()
            }
        ),
        delete_keys=MappingProxyType(dict(valid.delete_keys)),
        semantic_identity=valid.semantic_identity,
    )
    with pytest.raises(ValueError, match="mutation"):
        persist_provider_setup(structurally_valid_but_unissued)

    with pytest.raises(ValueError, match="mutation"):
        ProviderSetupMutation(
            section_values=dict(valid.section_values),
            delete_keys=valid.delete_keys,
            semantic_identity=valid.semantic_identity,
        )

    with pytest.raises(TypeError):
        valid.section_values["new"] = MappingProxyType({"key": "value"})
    with pytest.raises(TypeError):
        valid.section_values["api_settings.llama_cpp"]["model"] = "changed"
    with pytest.raises(ValueError, match="mutation"):
        ProviderSetupMutation(
            section_values=MappingProxyType(
                {
                    section: dict(values)
                    for section, values in valid.section_values.items()
                }
            ),
            delete_keys=valid.delete_keys,
            semantic_identity=valid.semantic_identity,
        )
    with pytest.raises(ValueError, match="mutation"):
        ProviderSetupMutation(
            section_values=valid.section_values,
            delete_keys=MappingProxyType({"api_settings.llama_cpp": ["api_key"]}),
            semantic_identity=valid.semantic_identity,
        )


def test_setup_mutation_rejects_overlapping_or_incoherent_owned_keys():
    valid = build_provider_setup_mutation(_draft(), {})
    overlapping_deletes = MappingProxyType({"api_settings.llama_cpp": ("api_url",)})

    with pytest.raises(ValueError, match="overlapping"):
        ProviderSetupMutation(
            section_values=valid.section_values,
            delete_keys=overlapping_deletes,
            semantic_identity=valid.semantic_identity,
        )

    forged_sections = MappingProxyType(
        {
            **{
                section: values
                for section, values in valid.section_values.items()
                if section != "provider_setup.confirmed"
            },
            "unrelated.section": MappingProxyType({"secret": "canary-secret"}),
        }
    )
    assert len(forged_sections) == len(valid.section_values) == 3
    with pytest.raises(ValueError, match="ownership") as error:
        ProviderSetupMutation(
            section_values=forged_sections,
            delete_keys=valid.delete_keys,
            semantic_identity=valid.semantic_identity,
        )
    assert "canary-secret" not in str(error.value)

    with pytest.raises(ValueError, match="identity"):
        ProviderSetupMutation(
            section_values=valid.section_values,
            delete_keys=valid.delete_keys,
            semantic_identity=None,
        )


def test_setup_mutation_rejects_unbounded_or_unsafe_section_shapes():
    valid = build_provider_setup_mutation(_draft(), {})
    too_many_sections = MappingProxyType(
        {
            **dict(valid.section_values),
            "extra.one": MappingProxyType({"key": "value"}),
        }
    )
    unsafe_section = MappingProxyType(
        {
            **{
                section: values
                for section, values in valid.section_values.items()
                if section != "api_settings.llama_cpp"
            },
            "api_settings.llama_cpp\nsecret": valid.section_values[
                "api_settings.llama_cpp"
            ],
        }
    )

    for section_values in (too_many_sections, unsafe_section):
        with pytest.raises(ValueError, match="mutation"):
            ProviderSetupMutation(
                section_values=section_values,
                delete_keys=valid.delete_keys,
                semantic_identity=valid.semantic_identity,
            )


def test_persist_revalidates_instances_that_bypass_construction(monkeypatch):
    forged = build_provider_setup_mutation(_draft(), {})
    object.__setattr__(forged, "section_values", {"bad": {"api_key": "secret-canary"}})
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: pytest.fail("forged mutation reached persistence"),
    )

    with pytest.raises(ValueError, match="mutation") as error:
        persist_provider_setup(forged)
    assert "secret-canary" not in str(error.value)


def test_persist_rejects_post_issuance_overlap_without_calling_writer(monkeypatch):
    forged = build_provider_setup_mutation(_draft(), {})
    provider_deletes = forged.delete_keys["api_settings.llama_cpp"]
    object.__setattr__(
        forged,
        "delete_keys",
        MappingProxyType(
            {
                **dict(forged.delete_keys),
                "api_settings.llama_cpp": (*provider_deletes, "api_url"),
            }
        ),
    )
    calls = []
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="overlapping"):
        persist_provider_setup(forged)
    assert calls == []


def test_persist_rejects_post_issuance_ownership_without_calling_writer(monkeypatch):
    forged = build_provider_setup_mutation(_draft(), {})
    forged_sections = MappingProxyType(
        {
            **{
                section: values
                for section, values in forged.section_values.items()
                if section != "provider_setup.confirmed"
            },
            "unrelated.section": MappingProxyType({"marker": "safe-value"}),
        }
    )
    assert len(forged_sections) == len(forged.section_values) == 3
    object.__setattr__(forged, "section_values", forged_sections)
    calls = []
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="ownership"):
        persist_provider_setup(forged)
    assert calls == []


def test_persist_rejects_post_issuance_identity_without_calling_writer(monkeypatch):
    forged = build_provider_setup_mutation(_draft(), {})
    other = build_provider_setup_mutation(
        _draft(endpoint="http://127.0.0.1:9090/v1/models"),
        {},
    )
    object.__setattr__(forged, "semantic_identity", other.semantic_identity)
    calls = []
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="identity"):
        persist_provider_setup(forged)
    assert calls == []


@pytest.mark.parametrize(
    ("field_name", "tampered_value"),
    [
        ("credential_revision", -1),
        ("credential_revision", 2**63),
        ("credential_revision", True),
        ("credential_revision", "1"),
        ("draft_generation", -1),
        ("draft_generation", 2**63),
        ("draft_generation", True),
        ("draft_generation", "1"),
    ],
)
def test_persist_revalidates_all_semantic_identity_counters(
    monkeypatch, field_name, tampered_value
):
    forged = build_provider_setup_mutation(_draft(), {})
    assert forged.semantic_identity is not None
    object.__setattr__(forged.semantic_identity, field_name, tampered_value)
    calls = []
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="identity"):
        persist_provider_setup(forged)
    assert calls == []


def test_provider_alias_after_bounded_scan_fails_closed_without_writer(monkeypatch):
    api_settings = {f"unknown_{index}": {} for index in range(256)}
    api_settings["OpenAI"] = {
        "api_base_url": "https://existing.example.test/v1",
        "model": "existing-model",
    }
    calls = []
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="Provider settings"):
        mutation = build_provider_setup_mutation(
            _draft(
                provider="openai",
                endpoint="https://new.example.test/v1",
            ),
            {"api_settings": api_settings},
        )
        persist_provider_setup(mutation)

    assert calls == []


def test_dataclass_replace_cannot_copy_issuance_or_persist_oversized_credential(
    monkeypatch,
):
    valid = build_provider_setup_mutation(
        _draft(
            provider="openai",
            credential_source="draft",
            credential_value="original-test-secret",
        ),
        {},
    )
    provider_values = dict(valid.section_values["api_settings.openai"])
    provider_values["api_key"] = "x" * 8193
    forged = replace(
        valid,
        section_values=MappingProxyType(
            {
                **dict(valid.section_values),
                "api_settings.openai": MappingProxyType(provider_values),
            }
        ),
    )
    calls = []
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="mutation"):
        persist_provider_setup(forged)
    assert calls == []


@pytest.mark.parametrize(
    ("credential_key", "credential_value"),
    [
        ("api_key", "x" * 8193),
        ("api_key_env_var", "INVALID ENV NAME"),
    ],
)
def test_persist_revalidates_issued_credential_values(
    monkeypatch, credential_key, credential_value
):
    if credential_key == "api_key":
        valid = build_provider_setup_mutation(
            _draft(
                provider="openai",
                credential_source="draft",
                credential_value="original-test-secret",
            ),
            {},
        )
    else:
        valid = build_provider_setup_mutation(
            _draft(
                provider="openai",
                credential_source="environment",
                credential_env_var="VALID_ENV_NAME",
            ),
            {},
        )
    provider_values = dict(valid.section_values["api_settings.openai"])
    provider_values[credential_key] = credential_value
    object.__setattr__(
        valid,
        "section_values",
        MappingProxyType(
            {
                **dict(valid.section_values),
                "api_settings.openai": MappingProxyType(provider_values),
            }
        ),
    )
    calls = []
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="mutation"):
        persist_provider_setup(valid)
    assert calls == []


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


def _write_identity(
    *,
    endpoint: str = "http://127.0.0.1:8080",
    credential_revision: int = 0,
    model: str = "qwen",
    provenance: str = "discovered",
):
    return persistence_module.ProviderSetupWriteIdentity(
        provider_key="llama_cpp",
        connection_identity=("llama_cpp", endpoint),
        credential_source="none",
        credential_revision=credential_revision,
        model_id=model,
        model_provenance=provenance,
    )


def _mock_atomic_snapshot():
    return config_module.AtomicConfigSnapshot(
        0,
        {"api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:8080"}}},
    )


@pytest.mark.parametrize(
    "changed_identity",
    [
        _write_identity(endpoint="http://127.0.0.1:8081"),
        _write_identity(credential_revision=1),
        replace(_write_identity(), credential_source="stored"),
        _write_identity(model="other-model"),
        _write_identity(provenance="manual"),
    ],
    ids=["endpoint", "credential-revision", "auth-source", "model", "provenance"],
)
def test_guarded_provider_setup_rejects_changed_identity_before_atomic_writer(
    monkeypatch,
    changed_identity,
):
    writes = []

    snapshot = _mock_atomic_snapshot()

    def writer(*_args, locked_snapshot_precondition=None, **_kwargs):
        assert callable(locked_snapshot_precondition)
        if not locked_snapshot_precondition(snapshot):
            return ConfigMutationResult(
                False,
                False,
                None,
                conflict=True,
                conflict_reason="identity_changed",
            )
        writes.append(True)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        writer,
    )
    mutation = build_provider_setup_mutation(_draft(), {})
    guard = persistence_module.ProviderSetupWriteGuard()
    identity = _write_identity()
    expected = guard.arm(identity)
    expected_state = persistence_module.capture_expected_provider_setup_state(
        snapshot,
        identity=identity,
    )
    persistence_module.bind_provider_setup_write_expectation(
        mutation,
        guard=guard,
        expectation=expected,
        expected_state=expected_state,
    )
    guard.arm(changed_identity)

    result = persist_provider_setup(mutation)

    assert result == ConfigMutationResult(
        False,
        False,
        None,
        conflict=True,
        conflict_reason="identity_changed",
    )
    assert writes == []


def test_guarded_provider_setup_rejects_invalidated_generation(monkeypatch):
    writes = []
    snapshot = _mock_atomic_snapshot()

    def writer(*_args, locked_snapshot_precondition=None, **_kwargs):
        assert callable(locked_snapshot_precondition)
        if not locked_snapshot_precondition(snapshot):
            return ConfigMutationResult(
                False,
                False,
                None,
                conflict=True,
                conflict_reason="identity_changed",
            )
        writes.append(True)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        writer,
    )
    mutation = build_provider_setup_mutation(_draft(), {})
    identity = _write_identity()
    guard = persistence_module.ProviderSetupWriteGuard()
    expected = guard.arm(identity)
    expected_state = persistence_module.capture_expected_provider_setup_state(
        snapshot,
        identity=identity,
    )
    persistence_module.bind_provider_setup_write_expectation(
        mutation,
        guard=guard,
        expectation=expected,
        expected_state=expected_state,
    )
    guard.invalidate()

    result = persist_provider_setup(mutation)

    assert result.conflict is True
    assert result.conflict_reason == "identity_changed"
    assert writes == []


def test_guarded_provider_setup_unchanged_identity_writes_once(monkeypatch):
    expected_result = ConfigMutationResult(True, True, None)
    writes = []
    snapshot = _mock_atomic_snapshot()

    def writer(
        section_values,
        *,
        delete_keys=None,
        locked_snapshot_precondition=None,
        **_kwargs,
    ):
        assert callable(locked_snapshot_precondition)
        assert locked_snapshot_precondition(snapshot)
        writes.append((section_values, delete_keys))
        return expected_result

    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        writer,
    )
    mutation = build_provider_setup_mutation(_draft(), {})
    identity = _write_identity()
    guard = persistence_module.ProviderSetupWriteGuard()
    expected = guard.arm(identity)
    expected_state = persistence_module.capture_expected_provider_setup_state(
        snapshot,
        identity=identity,
    )
    persistence_module.bind_provider_setup_write_expectation(
        mutation,
        guard=guard,
        expectation=expected,
        expected_state=expected_state,
    )

    result = persist_provider_setup(mutation)

    assert result is expected_result
    assert writes == [(mutation.section_values, mutation.delete_keys)]


def test_guarded_provider_setup_holds_identity_lease_through_atomic_writer(
    monkeypatch,
):
    writer_entered = threading.Event()
    release_writer = threading.Event()
    invalidation_finished = threading.Event()
    snapshot = _mock_atomic_snapshot()

    def writer(
        _section_values,
        *,
        delete_keys=None,
        locked_snapshot_precondition=None,
        **_kwargs,
    ):
        del delete_keys
        assert callable(locked_snapshot_precondition)
        assert locked_snapshot_precondition(snapshot)
        writer_entered.set()
        assert release_writer.wait(timeout=2)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        writer,
    )
    mutation = build_provider_setup_mutation(_draft(), {})
    identity = _write_identity()
    guard = persistence_module.ProviderSetupWriteGuard()
    expected = guard.arm(identity)
    expected_state = persistence_module.capture_expected_provider_setup_state(
        snapshot,
        identity=identity,
    )
    persistence_module.bind_provider_setup_write_expectation(
        mutation,
        guard=guard,
        expectation=expected,
        expected_state=expected_state,
    )

    result_holder = []
    writer_thread = threading.Thread(
        target=lambda: result_holder.append(persist_provider_setup(mutation))
    )
    writer_thread.start()
    assert writer_entered.wait(timeout=2)
    invalidator = threading.Thread(
        target=lambda: (
            guard.invalidate(),
            invalidation_finished.set(),
        )
    )
    invalidator.start()
    assert not invalidation_finished.wait(timeout=0.05)

    release_writer.set()
    writer_thread.join(timeout=2)
    invalidator.join(timeout=2)

    assert not writer_thread.is_alive()
    assert not invalidator.is_alive()
    assert invalidation_finished.is_set()
    assert result_holder == [ConfigMutationResult(True, True, None)]


@pytest.mark.parametrize(
    ("provider", "endpoint", "initial_settings", "changed_values", "changed_key"),
    [
        (
            "moonshot",
            "",
            {"api_region": "china", "api_key": "locked-route-key"},
            {"api_region": "global"},
            "api_region",
        ),
        (
            "huggingface",
            "",
            {
                "use_router_url_format": "true",
                "api_key": "locked-route-key",
            },
            {"use_router_url_format": "false"},
            "use_router_url_format",
        ),
        (
            "custom",
            "https://first.example/v1/chat/completions",
            {
                "api_url": "https://first.example/v1/chat/completions",
                "api_key": "locked-route-key",
            },
            {"api_url": "https://second.example/v1/chat/completions"},
            "api_url",
        ),
    ],
    ids=["moonshot-region", "huggingface-router", "custom-endpoint"],
)
def test_guarded_setup_rejects_completed_relevant_config_write(
    tmp_path,
    monkeypatch,
    provider,
    endpoint,
    initial_settings,
    changed_values,
    changed_key,
):
    import tomllib

    import toml

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps({"api_settings": {provider: initial_settings}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    snapshot = config_module.get_atomic_config_snapshot()
    mutation, _guard, _expected_state = _build_bound_first_run_mutation(
        snapshot=snapshot,
        provider=provider,
        endpoint=endpoint,
    )
    assert config_module.apply_settings_mutation_to_cli_config(
        {f"api_settings.{provider}": changed_values}
    ).fully_applied

    result = persist_provider_setup(mutation)

    assert result.conflict is True
    assert result.conflict_reason == "identity_changed"
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["api_settings"][provider][changed_key] == changed_values[changed_key]
    assert saved.get("chat_defaults", {}).get("model") != "selected-model"


def test_guarded_setup_rejects_completed_stored_credential_replacement(
    tmp_path,
    monkeypatch,
):
    import tomllib

    import toml
    from loguru import logger as loguru_logger

    first_secret = "locked-stored-credential-a"
    second_secret = "locked-stored-credential-b"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps(
            {
                "api_settings": {
                    "custom": {
                        "api_url": "https://credential.example/v1/chat/completions",
                        "api_key": first_secret,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    messages = []
    sink_id = loguru_logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        snapshot = config_module.get_atomic_config_snapshot()
        mutation, _guard, expected_state = _build_bound_first_run_mutation(
            snapshot=snapshot,
            provider="custom",
            endpoint="https://credential.example/v1/chat/completions",
        )
        assert config_module.apply_settings_mutation_to_cli_config(
            {"api_settings.custom": {"api_key": second_secret}}
        ).fully_applied
        result = persist_provider_setup(mutation)
    finally:
        loguru_logger.remove(sink_id)

    assert result.conflict is True
    rendered = "\n".join(
        (
            repr(expected_state),
            repr(mutation),
            repr(result),
            *(str(item) for item in messages),
        )
    )
    assert first_secret not in rendered
    assert second_secret not in rendered
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["api_settings"]["custom"]["api_key"] == second_secret
    assert first_secret not in config_path.read_text(encoding="utf-8")
    assert saved.get("chat_defaults", {}).get("model") != "selected-model"


def test_guarded_setup_allows_unrelated_generation_advance(
    tmp_path,
    monkeypatch,
):
    import tomllib

    import toml

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps(
            {
                "api_settings": {
                    "custom": {
                        "api_url": "https://stable.example/v1/chat/completions",
                        "api_key": "stable-credential",
                    }
                },
                "general": {"users_name": "before"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    snapshot = config_module.get_atomic_config_snapshot()
    mutation, _guard, _expected_state = _build_bound_first_run_mutation(
        snapshot=snapshot,
        provider="custom",
        endpoint="https://stable.example/v1/chat/completions",
    )
    assert config_module.apply_settings_mutation_to_cli_config(
        {"general": {"users_name": "after"}}
    ).fully_applied

    result = persist_provider_setup(mutation)

    assert result.fully_applied is True
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["general"]["users_name"] == "after"
    assert saved["chat_defaults"]["model"] == "selected-model"


def test_combined_provider_settings_boundary_validates_setup_and_writes_once(
    monkeypatch,
):
    setup = build_provider_setup_mutation(
        _draft(
            provider="openai",
            model="gpt-4.1",
            endpoint="https://api.openai.com/v1",
            credential_source="environment",
            credential_env_var="OPENAI_API_KEY",
        ),
        {},
    )
    section_values = {
        section: dict(values) for section, values in setup.section_values.items()
    }
    section_values["api_settings.openai"]["model_defaults"] = {
        "gpt-4.1": {"temperature": 0.2}
    }
    delete_keys = {section: tuple(keys) for section, keys in setup.delete_keys.items()}
    expected = ConfigMutationResult(True, True, None)
    calls = []

    def writer(values, *, delete_keys=None):
        calls.append((values, delete_keys))
        return expected

    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        writer,
    )

    result = persistence_module.persist_provider_settings_atomic(
        setup,
        provider="openai",
        model="gpt-4.1",
        section_values=section_values,
        delete_keys=delete_keys,
    )

    assert result is expected
    assert calls == [(section_values, delete_keys)]


def test_combined_provider_settings_boundary_rejects_connection_without_setup(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        persistence_module,
        "apply_settings_mutation_to_cli_config",
        lambda *_args, **_kwargs: calls.append(True),
    )

    with pytest.raises(ValueError, match="connection"):
        persistence_module.persist_provider_settings_atomic(
            None,
            provider="openai",
            model="gpt-4.1",
            section_values={
                "chat_defaults": {"provider": "openai", "model": "gpt-4.1"}
            },
            delete_keys={},
        )

    assert calls == []


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
