import asyncio
import builtins
from dataclasses import FrozenInstanceError
import inspect
from types import SimpleNamespace

import pytest
from textual.app import App
from textual.containers import Vertical
from textual.widgets import Button, Select, Static

import tldw_chatbook.Chat.console_session_settings as session_settings
from tldw_chatbook.Chat.console_context_repository import ConsoleMemoryRecord
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    ConsoleSettingsContextEstimate,
    ConsoleSessionSettings,
    ConsoleSettingsSummaryState,
    _estimate_tokens_locally,
    build_console_context_estimate,
    build_console_settings_summary_state,
    build_console_settings_readiness,
    build_default_console_session_settings,
    build_console_model_options,
    build_console_provider_options,
    console_settings_warnings,
    reasoning_effort_hint_for_model,
    validate_console_session_settings,
)
from tldw_chatbook.Utils.token_counter import count_tokens_messages
from tldw_chatbook.Widgets.Console.console_context_controls import (
    build_console_context_control_state,
)
from tldw_chatbook.Widgets.Console.console_settings_modal import ConsoleSettingsModal


def test_console_settings_exclude_presentation_while_session_owns_identity() -> None:
    settings_fields = set(ConsoleSessionSettings.__dataclass_fields__)
    session_fields = set(ConsoleChatSession.__dataclass_fields__)

    assert "user_profile_label" not in settings_fields
    assert "persona_label" not in settings_fields
    assert {"assistant_kind", "assistant_name", "assistant_id"}.isdisjoint(
        settings_fields
    )
    assert "assistant_name" not in session_fields
    assert {
        "runtime_backend",
        "assistant_kind",
        "assistant_id",
        "assistant_authority_id",
        "character_id",
        "character_name",
    } <= session_fields


def test_session_settings_keeps_gateway_runtime_dependencies_out() -> None:
    source = inspect.getsource(session_settings)

    forbidden_dependencies = {
        "console_provider_gateway",
        "httpx",
        "custom_tokenizers",
        "count_tokens_chat_history",
    }

    assert not forbidden_dependencies.intersection(source.split())
    for forbidden_dependency in forbidden_dependencies:
        assert forbidden_dependency not in source


def test_readiness_does_not_import_gateway_or_config_runtime_modules(
    monkeypatch,
) -> None:
    real_import = builtins.__import__
    forbidden_modules = {
        "tldw_chatbook.Chat.Chat_Functions",
        "tldw_chatbook.config",
    }

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name in forbidden_modules:
            raise AssertionError(f"unexpected import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        app_config={"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        environ={},
    )

    assert readiness.label == "Missing key"


def test_settings_execution_provider_keys_match_chat_api_handlers() -> None:
    from tldw_chatbook.Chat.Chat_Functions import API_CALL_HANDLERS

    assert CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS == frozenset(API_CALL_HANDLERS)


def test_default_settings_prefers_chat_defaults_and_provider_config() -> None:
    config = {
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "chat-default",
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 2048,
        },
        "api_settings": {
            "llama_cpp": {
                "api_url": "127.0.0.1:9099/v1",
                "model": "configured-model",
                "top_k": 40,
                "min_p": 0.05,
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config=config,
        provider="llama_cpp",
        model=None,
    )

    assert settings.provider == "llama_cpp"
    assert settings.model == "chat-default"
    assert settings.base_url == "http://127.0.0.1:9099"
    assert settings.temperature == 0.2
    assert settings.top_p == 0.8
    assert settings.min_p == 0.05
    assert settings.top_k == 40
    assert settings.max_tokens == 2048


def test_chat_defaults_model_outranks_provider_fallback() -> None:
    config = {
        "chat_defaults": {"provider": "openai", "model": "chosen-model"},
        "api_settings": {"openai": {"model": "legacy-model"}},
    }

    effective = session_settings.resolve_effective_chat_configuration(config)

    assert effective.provider == "openai"
    assert effective.model == "chosen-model"
    assert effective.model_source == "chat_defaults"


def test_explicit_model_outranks_chat_defaults_and_provider_fallback() -> None:
    config = {
        "chat_defaults": {"provider": "openai", "model": "global-model"},
        "api_settings": {"openai": {"model": "legacy-model"}},
    }

    effective = session_settings.resolve_effective_chat_configuration(
        config,
        model="session-model",
    )

    assert effective.model == "session-model"
    assert effective.model_source == "session"


def test_provider_model_fallback_order_is_model_then_api_model_then_default_model() -> (
    None
):
    provider_settings = {
        "model": "model-value",
        "api_model": "api-model-value",
        "default_model": "default-model-value",
    }
    config = {
        "chat_defaults": {"provider": "openai"},
        "api_settings": {"openai": provider_settings},
    }

    effective = session_settings.resolve_effective_chat_configuration(config)
    assert effective.model == "model-value"
    assert effective.model_source == "provider_fallback"

    del provider_settings["model"]
    effective = session_settings.resolve_effective_chat_configuration(config)
    assert effective.model == "api-model-value"
    assert effective.model_source == "provider_fallback"

    del provider_settings["api_model"]
    effective = session_settings.resolve_effective_chat_configuration(config)
    assert effective.model == "default-model-value"
    assert effective.model_source == "provider_fallback"


def test_missing_model_reports_none_provenance() -> None:
    effective = session_settings.resolve_effective_chat_configuration(
        {"chat_defaults": {"provider": "openai"}}
    )

    assert effective.model is None
    assert effective.model_source == "none"


def test_legacy_provider_alias_reads_without_rewrite_and_save_is_canonical() -> None:
    config = {
        "chat_defaults": {
            "provider": "OpenAI-Compatible",
            "model": "pocket-tts",
        }
    }

    effective = session_settings.resolve_effective_chat_configuration(config)

    assert effective.provider == "openai"
    assert config["chat_defaults"]["provider"] == "OpenAI-Compatible"
    assert session_settings.build_canonical_chat_defaults_mutation(effective) == {
        "chat_defaults": {"provider": "openai", "model": "pocket-tts"}
    }


def test_explicit_provider_alias_is_canonicalized_for_read_only_resolution() -> None:
    config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "chat-model"},
        "api_settings": {"custom": {"api_url": "https://example.test/v1"}},
    }

    effective = session_settings.resolve_effective_chat_configuration(
        config,
        provider="custom-openai-api",
    )

    assert effective.provider == "custom"
    assert effective.base_url == "https://example.test/v1"
    assert config["chat_defaults"]["provider"] == "llama_cpp"


def test_effective_chat_configuration_is_frozen_and_slotted() -> None:
    effective = session_settings.resolve_effective_chat_configuration(
        {"chat_defaults": {"provider": "openai", "model": "gpt-test"}}
    )

    assert not hasattr(effective, "__dict__")
    with pytest.raises(FrozenInstanceError):
        effective.model = "replacement"


def test_canonical_chat_defaults_mutation_omits_missing_values() -> None:
    effective = session_settings.EffectiveChatConfiguration(
        provider="",
        model=None,
        base_url=None,
        model_source="none",
    )

    assert session_settings.build_canonical_chat_defaults_mutation(effective) == {
        "chat_defaults": {}
    }


def test_qwencloud_default_settings_use_canonical_fields_with_alias_fallbacks() -> None:
    settings = build_default_console_session_settings(
        {
            "api_settings": {
                "QwenCloud": {
                    "model": "alias-model",
                    "api_base_url": ("https://alias.example.test/compatible-mode/v1"),
                },
                "qwencloud": {"model": "canonical-model"},
            }
        },
        provider="QwenCloud",
    )

    assert settings.model == "canonical-model"
    assert settings.base_url == "https://alias.example.test/compatible-mode/v1"


def test_qwencloud_public_builders_fail_closed_for_malformed_canonical_settings() -> (
    None
):
    config = {
        "chat_defaults": {"model": "safe-default"},
        "api_settings": {
            "qwencloud": ["not", "a", "table"],
            "QwenCloud": {
                "model": "alias-canary-model",
                "api_key": "alias-canary-key",
            },
        },
    }

    defaults = build_default_console_session_settings(config, provider="QwenCloud")
    errors = validate_console_session_settings(defaults, app_config=config)
    readiness = build_console_settings_readiness(
        defaults,
        app_config=config,
        environ={},
    )

    assert defaults.model == "safe-default"
    assert errors == []
    assert readiness.label == "Not ready"
    assert "Invalid provider settings" in readiness.detail
    assert "alias-canary" not in readiness.detail


def test_qwencloud_public_builders_fail_closed_for_malformed_alias_only() -> None:
    config = {
        "chat_defaults": {"model": "safe-default"},
        "api_settings": {"QwenCloud": "not-a-table"},
    }

    defaults = build_default_console_session_settings(config, provider="qwencloud")
    errors = validate_console_session_settings(defaults, app_config=config)
    readiness = build_console_settings_readiness(
        defaults,
        app_config=config,
        environ={},
    )

    assert defaults.model == "safe-default"
    assert errors == []
    assert readiness.label == "Not ready"
    assert "Invalid provider settings" in readiness.detail


def test_qwencloud_public_builders_ignore_malformed_alias_when_exact_is_valid() -> None:
    config = {
        "api_settings": {
            "QwenCloud": ["not", "a", "table"],
            "qwencloud": {
                "model": "canonical-model",
                "api_base_url": "https://canonical.example.test/v1",
                "api_key_env_var": "QWENCLOUD_TEST_API_KEY",
            },
        }
    }

    defaults = build_default_console_session_settings(config, provider="QwenCloud")
    errors = validate_console_session_settings(defaults, app_config=config)
    readiness = build_console_settings_readiness(
        defaults,
        app_config=config,
        environ={"QWENCLOUD_TEST_API_KEY": "available"},
    )

    assert defaults.model == "canonical-model"
    assert defaults.base_url == "https://canonical.example.test/v1"
    assert errors == []
    assert readiness.label == "Ready"


def test_console_session_settings_system_prompt_defaults_to_none() -> None:
    """Native Console session settings carry no system prompt by default."""
    settings = ConsoleSessionSettings(provider="llama_cpp")

    assert settings.system_prompt is None


def test_default_settings_never_seeds_system_prompt_from_chat_defaults() -> None:
    """``build_default_console_session_settings`` must never seed a system prompt.

    This is an explicit product decision (not an oversight): the native
    Console sends no system message until a user sets one for a session, even
    when ``[chat_defaults]`` carries a ``system_prompt`` key used by other
    (non-Console) chat surfaces.
    """
    config = {
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "chat-default",
            "system_prompt": "You are a helpful assistant.",
        },
    }

    settings = build_default_console_session_settings(
        app_config=config,
        provider="llama_cpp",
        model=None,
    )

    assert settings.system_prompt is None


def test_default_settings_uses_api_base_for_llamacpp_base_url() -> None:
    settings = build_default_console_session_settings(
        {
            "chat_defaults": {"provider": "llama_cpp"},
            "api_settings": {"llama_cpp": {"api_base": "127.0.0.1:9191/v1"}},
        },
        provider="llama_cpp",
    )

    assert settings.base_url == "http://127.0.0.1:9191"


def test_default_settings_uses_api_base_url_for_llamacpp_base_url() -> None:
    settings = build_default_console_session_settings(
        {
            "chat_defaults": {"provider": "llama_cpp"},
            "api_settings": {"llama_cpp": {"api_base_url": "127.0.0.1:9292/v1"}},
        },
        provider="llama_cpp",
    )

    assert settings.base_url == "http://127.0.0.1:9292"


def test_default_settings_prefers_api_base_url_over_merged_llamacpp_api_url() -> None:
    settings = build_default_console_session_settings(
        {
            "chat_defaults": {"provider": "llama_cpp"},
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://localhost:8080/completion",
                    "api_base_url": "127.0.0.1:9292/v1",
                }
            },
        },
        provider="llama_cpp",
    )

    assert settings.base_url == "http://127.0.0.1:9292"


def test_public_helpers_accept_planned_positional_call_forms() -> None:
    config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "chat-default"},
        "api_settings": {"llama_cpp": {"api_url": "127.0.0.1:9099/v1"}},
    }

    settings = build_default_console_session_settings(config, "llama_cpp", None)
    provider_options = build_console_provider_options({"llama_cpp": ["m"]})
    model_options = build_console_model_options(
        "llama_cpp", {"llama_cpp": ["m"]}, "current"
    )
    estimate = build_console_context_estimate(
        [{"role": "user", "content": "hello"}],
        "openai",
        "gpt-3.5-turbo",
        1,
        "1 staged source",
        128,
        "You are concise.",
    )

    assert settings.provider == "llama_cpp"
    assert "llama_cpp" in [option.value for option in provider_options]
    assert [option.value for option in model_options] == ["current", "m"]
    assert estimate.used_tokens is not None
    assert estimate.staged_source_count == 1
    assert estimate.staged_context_summary == "1 staged source"


def test_model_options_include_current_model_missing_from_registry() -> None:
    options = build_console_model_options(
        provider="llama_cpp",
        providers_models={"llama_cpp": ["listed-model"]},
        current_model="configured-model",
    )

    assert [option.value for option in options] == ["configured-model", "listed-model"]


def test_model_options_use_normalized_provider_keys() -> None:
    options = build_console_model_options(
        provider="local_llamacpp",
        providers_models={"local-llamacpp": ["local-model"]},
        current_model=None,
    )

    assert [option.value for option in options] == ["local-model"]


def test_model_options_ignore_none_sentinel_values() -> None:
    options = build_console_model_options(
        provider="llama_cpp",
        providers_models={
            "Llama_cpp": ["None", "", " "],
            "llama_cpp": ["gemma-model"],
        },
        current_model=None,
    )

    assert [option.value for option in options] == ["gemma-model"]


def test_model_options_preserve_current_model_even_when_registry_has_none_sentinel() -> (
    None
):
    options = build_console_model_options(
        provider="llama_cpp",
        providers_models={"Llama_cpp": ["None"], "llama_cpp": ["gemma-model"]},
        current_model="manual-model",
    )

    assert [option.value for option in options] == ["manual-model", "gemma-model"]


def test_provider_options_include_all_configured_providers() -> None:
    options = build_console_provider_options(
        providers_models={
            "llama_cpp": ["local-model"],
            "openai": ["gpt-4.1"],
            "anthropic": ["claude-sonnet"],
        }
    )
    option_values = [option.value for option in options]

    assert option_values == sorted(option_values)
    assert {"anthropic", "llama_cpp", "openai"}.issubset(option_values)


def test_provider_options_include_console_sendable_handlers_missing_from_model_registry() -> (
    None
):
    options = build_console_provider_options({"openai": ["gpt-4.1"]})
    option_values = [option.value for option in options]

    assert "mistral" in option_values
    assert "mistralai" in option_values


def test_qwencloud_is_a_first_class_normalized_console_provider_option() -> None:
    options = build_console_provider_options({"QwenCloud": ["qwen3.8-max"]})
    options_by_value = {option.value: option for option in options}

    assert options_by_value["qwencloud"].label == "qwencloud"
    assert "qwencloud" in CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS


def test_provider_options_label_configured_unsupported_providers_as_wip() -> None:
    options = build_console_provider_options({"local_onnx": ["manual-model"]})
    options_by_value = {option.value: option for option in options}

    assert options_by_value["local_onnx"].label == "local_onnx (WIP)"
    assert options_by_value["local_ollama"].label == "local_ollama"


def test_validation_rejects_out_of_range_temperature() -> None:
    settings = ConsoleSessionSettings(provider="llama_cpp", model="m", temperature=2.1)

    errors = validate_console_session_settings(settings, app_config={})

    assert "Temperature must be between 0 and 2." in errors


def test_validation_allows_blank_optional_numeric_fields() -> None:
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="m",
        min_p="",  # type: ignore[arg-type]
        top_k="",  # type: ignore[arg-type]
        max_tokens="",  # type: ignore[arg-type]
    )

    errors = validate_console_session_settings(settings, app_config={})

    assert errors == []


def test_validation_accepts_integral_numeric_strings() -> None:
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="m",
        top_k="10",  # type: ignore[arg-type]
        max_tokens="512",  # type: ignore[arg-type]
    )

    errors = validate_console_session_settings(settings, app_config={})

    assert errors == []


def test_validation_rejects_bool_and_non_integral_float_numeric_fields() -> None:
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="m",
        top_k=40.9,  # type: ignore[arg-type]
        max_tokens=True,  # type: ignore[arg-type]
    )

    errors = validate_console_session_settings(settings, app_config={})

    assert "Top K must be 0 or greater." in errors
    assert "Response max tokens must be 1 or greater." in errors


def test_readiness_reports_missing_key_for_supported_openai_instead_of_wip() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        app_config={"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        environ={},
    )

    assert readiness.label == "Missing key"
    assert "OPENAI_API_KEY" in readiness.detail
    assert "not wired" not in readiness.detail


def test_readiness_empty_provider_uses_select_provider_copy_without_empty_quotes() -> (
    None
):
    """FR-07: an unset provider is not an unsupported one; no empty '' copy."""
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="", model=None),
        app_config={},
        environ={},
    )

    assert readiness.label == "Unknown"
    assert readiness.native_send_supported is False
    assert "Select a provider" in readiness.detail
    assert "''" not in readiness.detail
    assert "not available in Console yet" not in readiness.detail


def test_readiness_reports_ready_for_keyless_supported_generic_provider() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="ollama", model="llama3", base_url="http://127.0.0.1:11434"
        ),
        app_config={"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    assert readiness.label == "Ready"
    assert "No API key is required." in readiness.detail
    assert "not wired" not in readiness.detail
    assert readiness.native_send_supported is True


def test_readiness_allows_configured_url_with_trailing_slash() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="ollama", model="llama3", base_url="http://127.0.0.1:11434/"
        ),
        app_config={"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_readiness_allows_llamacpp_configured_v1_endpoint_normalized_to_root() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="llama_cpp", model="llama3", base_url="http://127.0.0.1:9099"
        ),
        app_config={
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099/v1"}}
        },
        environ={},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_readiness_allows_llamacpp_api_base_url_endpoint_normalized_to_root() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="llama_cpp", model="llama3", base_url="http://127.0.0.1:9292"
        ),
        app_config={
            "api_settings": {"llama_cpp": {"api_base_url": "http://127.0.0.1:9292/v1"}}
        },
        environ={},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_readiness_prefers_api_base_url_over_merged_llamacpp_api_url() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="llama_cpp", model="llama3", base_url="http://127.0.0.1:9292"
        ),
        app_config={
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://localhost:8080/completion",
                    "api_base_url": "http://127.0.0.1:9292/v1",
                }
            }
        },
        environ={},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_readiness_allows_llamacpp_default_endpoint_without_saved_config() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="llama3",
            base_url=session_settings.DEFAULT_LLAMACPP_BASE_URL,
        ),
        app_config={"chat_defaults": {"provider": "llama_cpp", "model": "llama3"}},
        environ={},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_readiness_allows_llamacpp_session_endpoint_override() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="llama3",
            base_url="http://127.0.0.1:9099",
        ),
        app_config={
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://localhost:8080/completion",
                    "model": "llama3",
                },
            },
            "chat_defaults": {"provider": "llama_cpp", "model": "llama3"},
        },
        environ={},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_readiness_allows_llamacpp_custom_endpoint_without_saved_config() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="llama_cpp",
            model="llama3",
            base_url="http://127.0.0.1:9191",
        ),
        app_config={"chat_defaults": {"provider": "llama_cpp", "model": "llama3"}},
        environ={},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_readiness_blocks_unsaved_generic_endpoint_with_safe_details() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="ollama",
            model="llama3",
            base_url="http://127.0.0.1:9999/v1",
        ),
        app_config={
            "api_settings": {
                "ollama": {"api_url": "http://127.0.0.1:11434"},
            }
        },
        environ={},
    )

    assert readiness.label == "Endpoint not saved"
    assert readiness.native_send_supported is False
    assert "save the endpoint in Settings" in readiness.detail
    assert "Selected endpoint: http://127.0.0.1:9999/v1" in readiness.detail
    assert "Saved endpoint: http://127.0.0.1:11434" in readiness.detail


def test_settings_summary_includes_runtime_endpoint_credential_and_streaming_rows() -> (
    None
):
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="ollama",
            model="llama3",
            base_url="http://127.0.0.1:11434",
            streaming=False,
        ),
        app_config={
            "api_settings": {
                "ollama": {"api_url": "http://127.0.0.1:11434"},
            }
        },
        environ={},
    )

    state = build_console_settings_summary_state(
        ConsoleSessionSettings(
            provider="ollama",
            model="llama3",
            base_url="http://127.0.0.1:11434",
            streaming=False,
        ),
        ConsoleSettingsContextEstimate(
            used_tokens=None, token_limit=None, label="Context: unavailable"
        ),
        readiness,
    )

    assert state.endpoint_row == "Endpoint: http://127.0.0.1:11434"
    assert state.credential_row == "Credential: not required"
    assert state.transport_row == "Streaming: off"


def test_readiness_explicit_send_capable_injection_allows_supported_generic_provider() -> (
    None
):
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(
            provider="ollama", model="llama3", base_url="http://127.0.0.1:11434"
        ),
        app_config={"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
        native_provider_keys={"ollama"},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_readiness_explicit_send_capable_injection_preserves_direct_providers() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="llama_cpp", model="local-model"),
        app_config={"api_settings": {"llama_cpp": {"model": "local-model"}}},
        environ={},
        native_provider_keys={"ollama"},
    )

    assert readiness.label == "Ready"
    assert readiness.native_send_supported is True


def test_invalid_url_precedes_wip_for_url_provider() -> None:
    settings = ConsoleSessionSettings(
        provider="vllm", model="m", base_url="file:///tmp/x"
    )

    readiness = build_console_settings_readiness(settings, app_config={})

    assert readiness.label == "Invalid URL"


def test_malformed_ipv6_url_returns_validation_and_readiness_errors() -> None:
    settings = ConsoleSessionSettings(
        provider="vllm", model="m", base_url="http://[::1"
    )

    readiness = build_console_settings_readiness(settings, app_config={})
    errors = validate_console_session_settings(settings, app_config={})

    assert readiness.label == "Invalid URL"
    assert "Base URL must be a valid http(s) URL." in errors


def test_whitespace_host_url_returns_validation_and_readiness_errors() -> None:
    settings = ConsoleSessionSettings(
        provider="vllm", model="m", base_url="http://exa mple.com"
    )

    readiness = build_console_settings_readiness(settings, app_config={})
    errors = validate_console_session_settings(settings, app_config={})

    assert readiness.label == "Invalid URL"
    assert "Base URL must be a valid http(s) URL." in errors


def test_invalid_port_urls_return_validation_and_readiness_errors() -> None:
    for invalid_url in ("http://example.com:99999", "http://example.com:nope"):
        settings = ConsoleSessionSettings(
            provider="vllm", model="m", base_url=invalid_url
        )

        readiness = build_console_settings_readiness(settings, app_config={})
        errors = validate_console_session_settings(settings, app_config={})

        assert readiness.label == "Invalid URL"
        assert "Base URL must be a valid http(s) URL." in errors


def test_configured_url_provider_validates_invalid_base_url() -> None:
    settings = ConsoleSessionSettings(
        provider="future_provider",
        model="future-model",
        base_url="file:///tmp/not-http",
    )
    app_config = {
        "api_settings": {"future_provider": {"api_url": "http://127.0.0.1:9000"}}
    }

    readiness = build_console_settings_readiness(
        settings, app_config=app_config, environ={}
    )
    errors = validate_console_session_settings(settings, app_config=app_config)

    assert readiness.label == "Invalid URL"
    assert "Base URL must be a valid http(s) URL." in errors


def test_readiness_labels_cover_missing_key_ready_and_unknown() -> None:
    missing = build_console_settings_readiness(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet"),
        app_config={"api_settings": {"anthropic": {"api_key_env_var": "MISSING_KEY"}}},
        environ={},
        native_provider_keys={"llama_cpp", "local_llamacpp", "anthropic"},
    )
    ready = build_console_settings_readiness(
        ConsoleSessionSettings(provider="llama_cpp", model="m"),
        app_config={},
    )
    unknown = build_console_settings_readiness(
        ConsoleSessionSettings(provider="made_up_provider", model="m"),
        app_config={},
    )

    assert missing.label == "Missing key"
    assert ready.label == "Ready"
    assert ready.native_send_supported is True
    assert unknown.label == "Unknown"


def test_readiness_supported_provider_missing_key_is_not_wip() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet"),
        app_config={"api_settings": {"anthropic": {"api_key_env_var": "MISSING_KEY"}}},
        environ={},
    )

    assert readiness.label == "Missing key"
    assert "not wired" not in readiness.detail


def test_readiness_configured_unknown_non_native_provider_is_unknown() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="future_provider", model="future-model"),
        app_config={
            "api_settings": {"future_provider": {"api_url": "http://127.0.0.1:9000"}}
        },
        environ={},
    )

    assert readiness.label == "Unknown"
    assert "Choose a supported provider" in readiness.detail
    assert "not wired" not in readiness.detail


def test_context_estimate_counts_messages_and_staged_sources() -> None:
    """AUTHORIZED RE-BASELINE (task-6): staged evidence used to affect only
    the label's "; N sources staged" suffix -- `used_tokens` silently
    reported zero for content the send will actually carry. `staged_text`
    now folds into `used_tokens` too; the label-suffix contract this test
    originally pinned is unchanged and still asserted below."""
    without_staged = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello world"}],
        provider="openai",
        model="gpt-3.5-turbo",
        max_tokens_response=512,
        system_prompt="You are concise.",
    )

    estimate = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello world"}],
        provider="openai",
        model="gpt-3.5-turbo",
        staged_source_count=2,
        staged_context_summary="2 staged sources",
        staged_text="Evidence body text carried by the two staged sources.",
        max_tokens_response=512,
        system_prompt="You are concise.",
    )

    assert estimate.used_tokens is not None
    assert estimate.used_tokens > 0
    assert estimate.used_tokens > without_staged.used_tokens
    assert estimate.token_limit == 4096
    assert estimate.token_limit_verified is True
    assert estimate.token_limit_source == "model catalog"
    assert "tokens" in estimate.label
    assert "2 sources staged" in estimate.label
    assert estimate.staged_source_count == 2
    assert estimate.staged_context_summary == "2 staged sources"


def test_unknown_model_uses_8001_unverified_console_fallback() -> None:
    """Keep an unknown model usable without presenting fallback as verified."""
    estimate = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello"}],
        provider="openai",
        model="unlisted-model",
    )

    assert estimate.token_limit == 8001
    assert estimate.token_limit_verified is False
    assert estimate.token_limit_source == "provider fallback"
    assert "estimated; model unverified" in estimate.label


def test_context_estimate_staged_text_delta_tracks_its_size() -> None:
    """task-6: more staged evidence text must mean a bigger `used_tokens`
    delta, not a fixed/ignored bump -- this is what distinguishes "counts
    the evidence" from a hardcoded placeholder."""
    short = build_console_context_estimate(
        messages=[],
        provider="openai",
        model="gpt-3.5-turbo",
        staged_text="short evidence snippet",
    )
    long = build_console_context_estimate(
        messages=[],
        provider="openai",
        model="gpt-3.5-turbo",
        staged_text="short evidence snippet " * 200,
    )

    assert short.used_tokens is not None
    assert long.used_tokens is not None
    assert long.used_tokens > short.used_tokens


def test_context_estimate_large_staged_source_is_not_zero() -> None:
    """task-6: reproduces the critique's exact observation -- '0 tok' with
    five sources staged including a 942 KB corpus. A source of that class
    must move `used_tokens` well off zero."""
    large_source_text = "corpus text " * 80_000  # ~960 KB

    estimate = build_console_context_estimate(
        messages=[],
        provider="openai",
        model="gpt-3.5-turbo",
        staged_source_count=1,
        staged_text=large_source_text,
    )

    assert estimate.used_tokens is not None
    assert estimate.used_tokens > 1000


def test_context_estimate_blank_staged_text_does_not_change_tokens() -> None:
    """Whitespace-only staged text is treated the same as none -- purity
    guard: the builder must not fold in an empty/blank contribution."""
    baseline = build_console_context_estimate(
        messages=[{"role": "user", "content": "hi"}],
        provider="openai",
        model="gpt-3.5-turbo",
    )
    blank = build_console_context_estimate(
        messages=[{"role": "user", "content": "hi"}],
        provider="openai",
        model="gpt-3.5-turbo",
        staged_text="   ",
    )

    assert blank.used_tokens == baseline.used_tokens


def test_context_estimate_uses_longest_matching_token_limit_prefix() -> None:
    def token_counter(
        _messages: list[dict[str, str]], _model: str, _provider: str
    ) -> int:
        return 1

    gpt4_32k = build_console_context_estimate(
        messages=[],
        provider="openai",
        model="gpt-4-32k-0613",
        token_counter=token_counter,
    )
    gpt35_16k = build_console_context_estimate(
        messages=[],
        provider="openai",
        model="gpt-3.5-turbo-16k-0613",
        token_counter=token_counter,
    )

    assert gpt4_32k.token_limit == 32768
    assert gpt35_16k.token_limit == 16384


def test_context_estimate_uses_injected_counter_and_limit_resolver() -> None:
    seen = {}

    def token_counter(messages: list[dict[str, str]], model: str, provider: str) -> int:
        seen["messages"] = messages
        seen["model"] = model
        seen["provider"] = provider
        return 123

    def token_limit_resolver(model: str, provider: str) -> int:
        seen["limit_model"] = model
        seen["limit_provider"] = provider
        return 456

    estimate = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello world"}],
        provider="openai",
        model="gpt-3.5-turbo",
        token_counter=token_counter,
        token_limit_resolver=token_limit_resolver,
    )

    assert estimate.used_tokens == 123
    assert estimate.token_limit == 456
    assert estimate.label == "123 / 456 tokens"
    assert seen == {
        "messages": [{"role": "user", "content": "hello world"}],
        "model": "gpt-3.5-turbo",
        "provider": "openai",
        "limit_model": "gpt-3.5-turbo",
        "limit_provider": "openai",
    }


def test_context_estimate_token_counter_failure_uses_unavailable_copy() -> None:
    def fail_count(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("tokenizer unavailable")

    estimate = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello world"}],
        provider="openai",
        model="gpt-3.5-turbo",
        token_counter=fail_count,
    )

    assert estimate.used_tokens is None
    assert estimate.token_limit is None
    assert estimate.label == "Context: unavailable"


def test_default_settings_rejects_bool_and_fractional_optional_ints() -> None:
    settings = build_default_console_session_settings(
        {
            "chat_defaults": {"provider": "llama_cpp"},
            "api_settings": {"llama_cpp": {"top_k": True, "max_tokens": 1.5}},
        },
    )

    assert settings.top_k is None
    assert settings.max_tokens is None


def test_model_section_lines_compact_summary():
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_model_section_lines,
    )

    summary = ConsoleSettingsSummaryState(
        model_row="Model: gpt-4o (Missing key)",
        context_row="Context: 0 / 8,192 tokens; 4,096 response tokens",
        sampling_row="Sampling: T 0.60, P 0.95, min_p 0.05",
        identity_row="Assistant: General",
        provider_row="Provider: openai",
        transport_row="Streaming: off",
    )
    line1, line2 = build_console_model_section_lines(summary)
    assert line1 == "openai / gpt-4o (Missing key)"
    assert line2 == "T 0.60 · 0 / 8,192 tokens · Streaming: off"


def test_model_section_lines_tolerate_missing_rows():
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_model_section_lines,
    )

    summary = ConsoleSettingsSummaryState(
        model_row="",
        context_row="",
        sampling_row="",
        identity_row="",
    )
    line1, line2 = build_console_model_section_lines(summary)
    assert line1 == "not selected / no model"
    assert line2 == ""


def test_model_section_line_truncates_long_local_model_names():
    """Long gguf names must stay visible (truncated), not word-wrap away.

    Live UAT 2026-07: the one-row rail line rendered ``"llama_cpp / "``
    because the full model token wrapped onto a clipped second row.
    """
    from tldw_chatbook.Chat.console_session_settings import (
        CONSOLE_MODEL_SECTION_MODEL_MAX_CHARS,
        build_console_model_section_lines,
    )

    summary = ConsoleSettingsSummaryState(
        model_row="Model: Qwen3.6-27B-Uncensored-HauhauCS-Aggressive-Q8_K_P.gguf",
        context_row="Context: 0 / 4,096 tokens",
        sampling_row="Sampling: T 0.60",
        identity_row="Assistant: General",
        provider_row="Provider: llama_cpp",
        transport_row="Streaming: off",
    )
    line1, _line2 = build_console_model_section_lines(summary)
    provider_part, _, model_part = line1.partition(" / ")
    assert provider_part == "llama_cpp"
    assert model_part.startswith("Qwen3.6-27B-")
    assert model_part.endswith("…")
    assert len(model_part) <= CONSOLE_MODEL_SECTION_MODEL_MAX_CHARS
    # Short names remain untouched.
    short = ConsoleSettingsSummaryState(
        model_row="Model: gpt-4o",
        context_row="",
        sampling_row="",
        identity_row="",
        provider_row="Provider: openai",
    )
    assert build_console_model_section_lines(short)[0] == "openai / gpt-4o"


def test_context_estimate_counts_system_prompt_tokens():
    """Task 14: the estimate must count a system prompt's own tokens too."""
    without_system = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello"}],
        provider="openai",
        model="gpt-3.5-turbo",
    )
    with_system = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello"}],
        provider="openai",
        model="gpt-3.5-turbo",
        system_prompt="Answer using only formal English, citing sources.",
    )
    assert with_system.used_tokens is not None
    assert without_system.used_tokens is not None
    assert with_system.used_tokens > without_system.used_tokens


# --- Task 5: _estimate_tokens_locally delegates to the real counter --------
#
# Before this task the estimator was a char-ratio placeholder (`del model`,
# `CONSOLE_TOKEN_CHAR_RATIOS`, a fake `len(messages) * 10` overhead). These
# tests pin it to `count_tokens_messages` (Utils/token_counter.py), which is
# built on `estimate_tokens` -- custom tokenizer -> tiktoken -> conservative
# chars floor, never a whitespace word count -- so they hold regardless of
# which tier is active in the environment running them.


def test_estimate_tokens_locally_matches_real_counter_for_short_text() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert _estimate_tokens_locally(
        messages, "gpt-3.5-turbo", "openai"
    ) == count_tokens_messages(messages, "gpt-3.5-turbo", "openai")


def test_estimate_tokens_locally_matches_real_counter_for_long_text() -> None:
    long_text = "The quick brown fox jumps over the lazy dog. " * 200
    messages = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": long_text},
    ]
    assert _estimate_tokens_locally(
        messages, "claude-sonnet-4-6", "anthropic"
    ) == count_tokens_messages(messages, "claude-sonnet-4-6", "anthropic")


def test_estimate_tokens_locally_matches_real_counter_for_code() -> None:
    code = (
        "def fibonacci(n: int) -> int:\n"
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
        "results = [fibonacci(i) for i in range(10)]\n"
        "print({'results': results, 'ok': True})\n"
    )
    messages = [{"role": "assistant", "content": code}]
    assert _estimate_tokens_locally(
        messages, "gpt-4-turbo", "openai"
    ) == count_tokens_messages(messages, "gpt-4-turbo", "openai")


def test_estimate_tokens_locally_matches_real_counter_for_unicode() -> None:
    text = "これはユニコードのテストです。日本語のテキストを推定します。" * 20
    messages = [{"role": "user", "content": text}]
    assert _estimate_tokens_locally(
        messages, "gemini-1.5-pro", "google"
    ) == count_tokens_messages(messages, "gemini-1.5-pro", "google")


def test_estimate_tokens_locally_honors_model_argument() -> None:
    """The placeholder began with `del model`, discarding it entirely."""
    messages = [{"role": "user", "content": "hello there, this is a test message"}]
    gpt_tokens = _estimate_tokens_locally(messages, "gpt-4", "openai")
    claude_tokens = _estimate_tokens_locally(
        messages, "claude-3-opus-20240229", "anthropic"
    )
    assert gpt_tokens != claude_tokens
    assert gpt_tokens == count_tokens_messages(messages, "gpt-4", "openai")
    assert claude_tokens == count_tokens_messages(
        messages, "claude-3-opus-20240229", "anthropic"
    )


def test_estimate_tokens_locally_honors_provider_argument() -> None:
    content = "some context content for token estimation " * 5
    messages = [{"role": "user", "content": content}]
    openai_tokens = _estimate_tokens_locally(messages, "gpt-3.5-turbo", "openai")
    google_tokens = _estimate_tokens_locally(messages, "gpt-3.5-turbo", "google")
    assert openai_tokens == count_tokens_messages(messages, "gpt-3.5-turbo", "openai")
    assert google_tokens == count_tokens_messages(messages, "gpt-3.5-turbo", "google")


def test_estimate_tokens_locally_source_no_longer_discards_model() -> None:
    source = inspect.getsource(_estimate_tokens_locally)
    assert "del model" not in source


def test_estimate_tokens_locally_empty_messages_returns_zero() -> None:
    assert _estimate_tokens_locally([], "gpt-3.5-turbo", "openai") == 0


def test_rail_system_line_none_state_for_blank_or_missing_prompt():
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_rail_system_line,
    )

    assert build_console_rail_system_line(None) == "System: none"
    assert build_console_rail_system_line("   ") == "System: none"


def test_rail_system_line_shows_preview_for_set_prompt():
    from tldw_chatbook.Chat.console_session_settings import (
        build_console_rail_system_line,
    )

    assert build_console_rail_system_line("Be terse.") == "System: Be terse."


def test_rail_system_line_collapses_multiline_and_truncates_long_prompts():
    """Mirrors the task-186 model-line fix: a long/multi-line system prompt
    must collapse to one line AND truncate in the text itself, not rely on
    CSS ellipsis alone, or it silently word-wraps onto a hidden second row."""
    from tldw_chatbook.Chat.console_session_settings import (
        CONSOLE_RAIL_SYSTEM_PREVIEW_MAX_CHARS,
        build_console_rail_system_line,
    )

    multiline_prompt = "Line one.\nLine two continues on and on and on and on."
    line = build_console_rail_system_line(multiline_prompt)
    assert "\n" not in line
    assert line.startswith("System: Line one. Line two")
    assert line.endswith("…")
    preview = line.removeprefix("System: ")
    assert len(preview) <= CONSOLE_RAIL_SYSTEM_PREVIEW_MAX_CHARS


def test_pinned_prefill_defaults_none_and_replaces():
    from dataclasses import replace

    settings = ConsoleSessionSettings(provider="llama_cpp")
    assert settings.pinned_prefill is None
    pinned = replace(settings, pinned_prefill="*She pauses*")
    assert pinned.pinned_prefill == "*She pauses*"
    assert settings.pinned_prefill is None


def test_provider_scoped_defaults_beat_chat_defaults_for_sampling_fields():
    """TASK-342: Save-as-default persists sampling values under
    [console.provider_defaults.<provider>] — a section that only ever holds
    Console-saved defaults, so the boot builder ranks it above chat_defaults
    without letting factory api_settings scalars shadow user-tuned globals
    (that protection is pinned by f14d22dc3's tests)."""
    config = {
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "chat-model",
            "temperature": 0.6,
            "top_p": 0.95,
            "streaming": True,
        },
        "api_settings": {"llama_cpp": {"model": "saved-model"}},
        "console": {
            "provider_defaults": {
                "llama_cpp": {
                    "temperature": 0.88,
                    "top_p": 0.5,
                    "top_k": 17,
                    "max_tokens": 1234,
                    "seed": 42,
                    "presence_penalty": 0.25,
                    "frequency_penalty": 0.75,
                    "reasoning_effort": "high",
                    "reasoning_summary": "detailed",
                    "verbosity": "low",
                    "thinking_effort": "medium",
                    "thinking_budget_tokens": 2048,
                    "min_p": 0.07,
                },
            },
        },
    }

    settings = build_default_console_session_settings(
        app_config=config,
        provider="llama_cpp",
        model=None,
    )

    assert settings.temperature == 0.88
    assert settings.top_p == 0.5
    assert settings.min_p == 0.07
    assert settings.top_k == 17
    assert settings.max_tokens == 1234
    assert settings.seed == 42
    assert settings.presence_penalty == 0.25
    assert settings.frequency_penalty == 0.75
    assert settings.reasoning_effort == "high"
    assert settings.reasoning_summary == "detailed"
    assert settings.verbosity == "low"
    assert settings.thinking_effort == "medium"
    assert settings.thinking_budget_tokens == 2048


def test_chat_defaults_still_apply_when_no_console_saved_defaults_exist():
    # Factory api_settings sampling scalars must STILL lose to chat_defaults
    # (f14d22dc3) — only Console-saved defaults outrank them.
    config = {
        "chat_defaults": {
            "provider": "llama_cpp",
            "model": "chat-model",
            "temperature": 0.6,
            "top_p": 0.9,
        },
        "api_settings": {
            "llama_cpp": {"model": "saved-model", "temperature": 0.7, "top_p": 0.95}
        },
    }

    settings = build_default_console_session_settings(
        app_config=config,
        provider="llama_cpp",
        model=None,
    )

    assert settings.temperature == 0.6
    assert settings.top_p == 0.9


class _SettingsCloseHarness(App[None]):
    CSS = """
    ConsoleSettingsModal { align: center middle; }
    ConsoleSettingsModal #console-settings-modal { width: 100; height: 36; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []
        self.notices: list[str] = []

    def capture(self, result: object) -> None:
        self.results.append(result)

    def notify(self, message: str, **_kwargs: object) -> None:
        self.notices.append(message)


def _settings_close_memory(
    summary: str = "Original generated memory",
) -> ConsoleMemoryRecord:
    return ConsoleMemoryRecord(
        memory_id="memory-1",
        conversation_id="conversation-1",
        boundary_message_id="message-4",
        captured_leaf_message_id="message-8",
        lineage_json='["message-1", "message-4", "message-8"]',
        summary_text=summary,
        provider="llama_cpp",
        model="model-a",
        prompt_id="console.rewind_summarize",
        prompt_revision=2,
        prompt_digest="prompt-digest",
        selected_units_json='["message-1", "message-4"]',
        summarized_prefix_digest="prefix-digest",
        input_tokens=12_000,
        output_tokens=700,
        before_tokens=52_000,
        after_tokens=24_000,
        created_at="2026-08-14T12:00:00+00:00",
    )


def _settings_close_modal(
    *,
    reset_current_memory=None,
    undo_current_memory_reset=None,
    compact_now=None,
    summary: str = "Original generated memory",
) -> ConsoleSettingsModal:
    settings = ConsoleSessionSettings(
        provider="llama_cpp",
        model="model-a",
        max_tokens=4_000,
    )
    estimate = ConsoleSettingsContextEstimate(
        used_tokens=42_000,
        token_limit=100_000,
        label="42,000 / 100,000 tokens",
    )
    return ConsoleSettingsModal(
        settings=settings,
        app_config={"api_settings": {"llama_cpp": {}}},
        providers_models={"llama_cpp": ["model-a"]},
        context_estimate=estimate,
        context_state=build_console_context_control_state(
            settings=settings,
            estimate=estimate,
            active_memory=_settings_close_memory(summary),
        ),
        can_save=True,
        focus_context=True,
        reset_current_memory=reset_current_memory,
        undo_current_memory_reset=undo_current_memory_reset,
        compact_now=compact_now,
    )


async def _request_settings_close(pilot, source: str) -> None:
    if source == "visible-cancel":
        await pilot.click("#console-settings-cancel")
    elif source == "escape":
        await pilot.press("escape")
    else:
        await pilot.click(offset=(0, 0))
    await pilot.pause()


@pytest.mark.parametrize("source", ["visible-cancel", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_settings_memory_reset_close_sources_show_one_three_choice_guard(
    source: str,
) -> None:
    """Show one three-choice reset guard for every close source.

    Args:
        source: Visible control, Escape key, or backdrop dismissal source.
    """
    app = _SettingsCloseHarness()
    reset_calls = 0

    def reset_current() -> tuple[str, int]:
        nonlocal reset_calls
        reset_calls += 1
        return "memory-1", 2

    modal = _settings_close_modal(reset_current_memory=reset_current)
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        modal.query_one("#console-context-reset-current", Button).press()
        await pilot.pause()
        await _request_settings_close(pilot, source)

        guard = modal.query_one("#console-settings-close-guard", Vertical)
        assert guard.display
        assert (
            str(modal.query_one("#console-settings-close-undo", Button).label)
            == "Undo and close"
        )
        assert (
            str(modal.query_one("#console-settings-close-keep", Button).label)
            == "Keep reset and close"
        )
        assert (
            str(modal.query_one("#console-settings-close-return", Button).label)
            == "Return"
        )
        assert modal.query_one("#console-settings-close-undo", Button).display
        assert modal.query_one("#console-settings-close-keep", Button).display
        assert not modal.query_one("#console-settings-close-anyway", Button).display
        assert app.screen is modal
        assert app.results == []

        await pilot.press("escape")
        await pilot.click(offset=(0, 0))
        await pilot.pause()
        assert len(modal.query("#console-settings-close-guard")) == 1
        assert guard.display
        assert modal.focused is modal.query_one("#console-settings-close-undo", Button)
        assert reset_calls == 1


@pytest.mark.asyncio
async def test_settings_memory_reset_undo_and_close_uses_optimistic_undo() -> None:
    app = _SettingsCloseHarness()
    undo_calls: list[tuple[str, int]] = []

    def undo_current(memory_id: str, revision: int) -> bool:
        undo_calls.append((memory_id, revision))
        return True

    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=undo_current,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        modal.query_one("#console-context-reset-current", Button).press()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.click("#console-settings-close-undo")
        await pilot.pause()

        assert undo_calls == [("memory-1", 2)]
        assert app.results == [None]


@pytest.mark.asyncio
async def test_settings_memory_reset_expired_undo_keeps_guard_and_recovery_copy() -> (
    None
):
    app = _SettingsCloseHarness()
    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: False,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        modal.query_one("#console-context-reset-current", Button).press()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.click("#console-settings-close-undo")
        await pilot.pause()

        recovery = "Undo expired because conversation memory changed."
        assert modal.query_one("#console-settings-close-guard", Vertical).display
        assert recovery in str(
            modal.query_one("#console-settings-close-message", Static).renderable
        )
        assert recovery in str(
            modal.query_one("#console-context-action-status", Static).renderable
        )
        assert modal._memory_reset_token == ("memory-1", 2)
        assert app.results == []


@pytest.mark.asyncio
async def test_settings_memory_reset_keep_close_clears_local_undo_opportunity() -> None:
    app = _SettingsCloseHarness()
    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: True,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        modal.query_one("#console-context-reset-current", Button).press()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.click("#console-settings-close-keep")
        await pilot.pause()

        assert modal._memory_reset_token is None
        assert app.results == [None]


@pytest.mark.asyncio
async def test_settings_memory_reset_return_restores_focus_to_undo() -> None:
    app = _SettingsCloseHarness()
    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: True,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        modal.query_one("#console-context-reset-current", Button).press()
        await pilot.pause()
        undo = modal.query_one("#console-context-undo-reset", Button)
        undo.focus()
        await pilot.press("escape")
        await pilot.click("#console-settings-close-return")
        await pilot.pause()
        await pilot.pause()

        assert not modal.query_one("#console-settings-close-guard", Vertical).display
        assert modal.focused is undo
        assert modal._memory_reset_token == ("memory-1", 2)
        assert app.results == []


@pytest.mark.asyncio
async def test_settings_reset_guard_traps_real_tab_navigation_and_enter() -> None:
    app = _SettingsCloseHarness()
    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: False,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        normal_focus = modal.query_one("#console-settings-view-model", Button)
        normal_focus.focus()
        await pilot.press("tab")
        await pilot.pause()
        assert modal.focused is not normal_focus
        assert modal.focused not in modal.query("#console-settings-close-guard Button")

        modal.query_one("#console-context-reset-current", Button).press()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()

        expected_focus = [
            ("tab", "console-settings-close-keep"),
            ("tab", "console-settings-close-return"),
            ("tab", "console-settings-close-undo"),
            ("shift+tab", "console-settings-close-return"),
            ("shift+tab", "console-settings-close-keep"),
            ("shift+tab", "console-settings-close-undo"),
        ]
        assert modal.focused is modal.query_one("#console-settings-close-undo", Button)
        for key, expected_id in expected_focus:
            await pilot.press(key)
            await pilot.pause()
            assert modal.focused is modal.query_one(f"#{expected_id}", Button)

        await pilot.press("enter")
        await pilot.pause()
        assert modal.query_one("#console-settings-close-guard", Vertical).display
        assert modal.focused is modal.query_one("#console-settings-close-undo", Button)
        assert app.results == []


@pytest.mark.parametrize(
    ("choice", "undo_succeeds"),
    [
        ("#console-settings-close-undo", True),
        ("#console-settings-close-keep", False),
    ],
)
@pytest.mark.asyncio
async def test_settings_reset_choice_transitions_to_active_compaction_guard(
    choice: str,
    undo_succeeds: bool,
) -> None:
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def compact_now() -> tuple[bool, str]:
        entered.set()
        await release.wait()
        return True, "Compaction complete."

    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: undo_succeeds,
        compact_now=compact_now,
    )
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            modal.query_one("#console-context-reset-current", Button).press()
            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await asyncio.wait_for(entered.wait(), timeout=1)
            focus = modal.query_one("#console-context-budget-mode", Select)
            focus.focus()

            await pilot.press("escape")
            await pilot.click(choice)
            await pilot.pause()
            await pilot.pause()

            assert len(modal.query("#console-settings-close-guard")) == 1
            assert modal.query_one("#console-settings-close-guard", Vertical).display
            assert not modal.query_one("#console-settings-close-undo", Button).display
            assert not modal.query_one("#console-settings-close-keep", Button).display
            assert modal.query_one("#console-settings-close-anyway", Button).display
            assert modal.focused is modal.query_one(
                "#console-settings-close-anyway", Button
            )
            assert "Provider work may continue and may still be billed." in str(
                modal.query_one("#console-settings-close-message", Static).renderable
            )
            assert modal._memory_reset_token is None
            assert app.results == []

            await pilot.click("#console-settings-close-return")
            await pilot.pause()
            await pilot.pause()
            assert not modal.query_one(
                "#console-settings-close-guard", Vertical
            ).display
            assert modal.focused is focus
            assert "Compacting" in str(
                modal.query_one("#console-context-action-status", Static).renderable
            )
            assert app.screen is modal
            assert app.results == []
    finally:
        release.set()


@pytest.mark.asyncio
async def test_settings_failed_reset_undo_stays_a_reset_guard_during_compaction() -> (
    None
):
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def compact_now() -> tuple[bool, str]:
        entered.set()
        await release.wait()
        return True, "Compaction complete."

    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: False,
        compact_now=compact_now,
    )
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            modal.query_one("#console-context-reset-current", Button).press()
            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await asyncio.wait_for(entered.wait(), timeout=1)

            await pilot.press("escape")
            await pilot.click("#console-settings-close-undo")
            await pilot.pause()

            assert modal.query_one("#console-settings-close-guard", Vertical).display
            assert modal.query_one("#console-settings-close-undo", Button).display
            assert modal.query_one("#console-settings-close-keep", Button).display
            assert not modal.query_one("#console-settings-close-anyway", Button).display
            assert "Undo expired because conversation memory changed." in str(
                modal.query_one("#console-settings-close-message", Static).renderable
            )
            assert modal._memory_reset_token == ("memory-1", 2)
            assert app.results == []
    finally:
        release.set()


@pytest.mark.parametrize("source", ["visible-cancel", "escape", "backdrop"])
@pytest.mark.asyncio
async def test_settings_active_compaction_close_sources_show_acknowledgement(
    source: str,
) -> None:
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def compact_now() -> tuple[bool, str]:
        entered.set()
        await release.wait()
        return True, "Compaction complete."

    modal = _settings_close_modal(compact_now=compact_now)
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await asyncio.wait_for(entered.wait(), timeout=1)
            await _request_settings_close(pilot, source)

            assert modal.query_one("#console-settings-close-guard", Vertical).display
            assert not modal.query_one("#console-settings-close-undo", Button).display
            assert not modal.query_one("#console-settings-close-keep", Button).display
            assert (
                str(modal.query_one("#console-settings-close-anyway", Button).label)
                == "Close anyway"
            )
            assert modal.query_one("#console-settings-close-anyway", Button).display
            assert (
                str(modal.query_one("#console-settings-close-return", Button).label)
                == "Return"
            )
            await pilot.press("escape")
            await pilot.click(offset=(0, 0))
            await pilot.pause()
            assert len(modal.query("#console-settings-close-guard")) == 1
            assert modal.focused is modal.query_one(
                "#console-settings-close-anyway", Button
            )
            assert app.results == []
    finally:
        release.set()


@pytest.mark.asyncio
async def test_settings_active_compaction_return_preserves_progress_and_focus() -> None:
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def compact_now() -> tuple[bool, str]:
        entered.set()
        await release.wait()
        return True, "Compaction complete."

    modal = _settings_close_modal(compact_now=compact_now)
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await asyncio.wait_for(entered.wait(), timeout=1)
            focus = modal.query_one("#console-context-budget-mode", Select)
            focus.focus()
            await pilot.press("escape")
            await pilot.click("#console-settings-close-return")
            await pilot.pause()
            await pilot.pause()

            status = str(
                modal.query_one("#console-context-action-status", Static).renderable
            )
            assert "Compacting" in status
            assert modal.query_one("#console-context-compact-now", Button).disabled
            assert modal.focused is focus
            assert app.screen is modal
            assert app.results == []
    finally:
        release.set()


@pytest.mark.asyncio
async def test_settings_completed_compaction_retires_guard_and_stale_close_warning() -> (
    None
):
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def compact_now() -> tuple[bool, str]:
        entered.set()
        await release.wait()
        finished.set()
        return True, "Compaction complete."

    modal = _settings_close_modal(compact_now=compact_now)
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        modal.query_one("#console-context-compact-now", Button).press()
        await pilot.pause()
        await asyncio.wait_for(entered.wait(), timeout=1)
        focus = modal.query_one("#console-context-budget-mode", Select)
        focus.focus()
        await pilot.press("escape")
        await pilot.pause()
        assert modal.query_one("#console-settings-close-guard", Vertical).display

        release.set()
        await asyncio.wait_for(finished.wait(), timeout=1)
        await pilot.pause()
        await pilot.pause()

        assert not modal.query_one("#console-settings-close-guard", Vertical).display
        assert modal.focused is focus
        assert "Compaction complete." in str(
            modal.query_one("#console-context-action-status", Static).renderable
        )
        assert "may still be billed" not in str(
            modal.query_one("#console-settings-close-message", Static).renderable
        )
        assert app.results == []
        assert app.notices == []

        modal.query_one("#console-settings-close-anyway", Button).press()
        await pilot.pause()
        assert app.notices == []
        assert app.results == [None]


@pytest.mark.asyncio
async def test_settings_compaction_guard_traps_real_tab_navigation_and_enter() -> None:
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def compact_now() -> tuple[bool, str]:
        entered.set()
        await release.wait()
        return True, "Compaction complete."

    modal = _settings_close_modal(compact_now=compact_now)
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await asyncio.wait_for(entered.wait(), timeout=1)
            await pilot.press("escape")
            await pilot.pause()

            expected_focus = [
                ("tab", "console-settings-close-return"),
                ("tab", "console-settings-close-anyway"),
                ("shift+tab", "console-settings-close-return"),
                ("shift+tab", "console-settings-close-anyway"),
            ]
            assert modal.focused is modal.query_one(
                "#console-settings-close-anyway", Button
            )
            for key, expected_id in expected_focus:
                await pilot.press(key)
                await pilot.pause()
                assert modal.focused is modal.query_one(f"#{expected_id}", Button)

            modal.query_one("#console-settings-close-return", Button).focus()
            await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            assert not modal.query_one(
                "#console-settings-close-guard", Vertical
            ).display
            assert app.screen is modal
            assert app.results == []
    finally:
        release.set()


@pytest.mark.asyncio
async def test_settings_close_anyway_queued_presses_commit_one_notice_and_cancel(
    monkeypatch,
) -> None:
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    provider_cancelled = False

    async def compact_now() -> tuple[bool, str]:
        nonlocal provider_cancelled
        entered.set()
        try:
            await release.wait()
            return True, "Compaction complete."
        except asyncio.CancelledError:
            provider_cancelled = True
            raise
        finally:
            finished.set()

    modal = _settings_close_modal(compact_now=compact_now)
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await asyncio.wait_for(entered.wait(), timeout=1)
            await pilot.press("escape")
            await pilot.pause()

            worker = modal._compaction_wait_worker
            assert worker is not None
            cancel_calls = 0
            original_cancel = worker.cancel

            def counted_cancel() -> None:
                nonlocal cancel_calls
                cancel_calls += 1
                original_cancel()

            monkeypatch.setattr(worker, "cancel", counted_cancel)
            close_anyway = modal.query_one("#console-settings-close-anyway", Button)
            close_anyway.press()
            close_anyway.press()
            await pilot.pause()

            assert app.results == [None]
            assert cancel_calls == 1
            assert app.notices == [
                "Provider work may continue and may still be billed."
            ]
            assert not provider_cancelled
            assert not finished.is_set()

            release.set()
            await asyncio.wait_for(finished.wait(), timeout=1)
            assert not provider_cancelled
    finally:
        release.set()


@pytest.mark.asyncio
async def test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh() -> (
    None
):
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    provider_cancelled = False

    async def compact_now() -> tuple[bool, str]:
        nonlocal provider_cancelled
        entered.set()
        try:
            await release.wait()
            return True, "Compaction complete."
        except asyncio.CancelledError:
            provider_cancelled = True
            raise
        finally:
            finished.set()

    modal = _settings_close_modal(compact_now=compact_now)
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await asyncio.wait_for(entered.wait(), timeout=1)
            await pilot.press("escape")
            await pilot.click("#console-settings-close-anyway")
            await pilot.pause()

            assert app.results == [None]
            assert modal._compaction_wait_worker is None
            assert not provider_cancelled
            assert not finished.is_set()
            assert app.notices == [
                "Provider work may continue and may still be billed."
            ]
            assert all("cancel" not in notice.lower() for notice in app.notices)

            release.set()
            await asyncio.wait_for(finished.wait(), timeout=1)
            assert not provider_cancelled

            from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

            fresh_memory = _settings_close_memory(
                "Fresh durable memory after compaction"
            )
            settings = ConsoleSessionSettings(
                provider="llama_cpp",
                model="model-a",
                max_tokens=4_000,
            )
            estimate = ConsoleSettingsContextEstimate(
                used_tokens=24_000,
                token_limit=100_000,
                label="24,000 / 100,000 tokens",
            )
            session = ConsoleChatSession(
                id="session-1",
                settings=settings,
            )
            store = SimpleNamespace(
                active_session_id=session.id,
                switch_session=lambda _session_id: session,
            )

            async def providers_models(
                _provider: str,
                *,
                current_model: str,
            ) -> dict[str, list[str]]:
                return {"llama_cpp": [current_model]}

            async def compact_fresh(_session_id: str) -> tuple[bool, str]:
                return True, "Compaction complete."

            controller = SimpleNamespace(
                run_state=SimpleNamespace(is_send_allowed=True),
                reset_active_context_memory=lambda _session_id: ("memory-1", 3),
                undo_context_memory_reset=lambda _memory_id, _revision: True,
                reset_all_context_memories=lambda _session_id: 1,
                compact_context_now=compact_fresh,
            )
            production_opener = SimpleNamespace(
                app=app,
                _session=SimpleNamespace(
                    _ensure_active_console_session_settings=lambda: settings
                ),
                _ensure_console_chat_controller=lambda: controller,
                _ensure_console_chat_store=lambda: store,
                _active_console_settings_context_estimate=lambda: estimate,
                _active_console_context_control_state=lambda *, estimate: (
                    build_console_context_control_state(
                        settings=settings,
                        estimate=estimate,
                        active_memory=fresh_memory,
                    )
                ),
                _global_chat_display_name=lambda: "User",
                _provider_readiness_app_config=lambda: {
                    "api_settings": {"llama_cpp": {}}
                },
                _providers_models_for_console_settings=providers_models,
                _apply_console_settings_result=lambda *_args, **_kwargs: None,
            )
            await ChatScreen._open_console_settings(  # type: ignore[arg-type]
                production_opener,
                focus_context=True,
            )
            await pilot.pause()
            reopened = app.screen

            assert isinstance(reopened, ConsoleSettingsModal)
            assert reopened is not modal
            assert reopened._memory_reset_token is None
            assert (
                reopened.query_one("#console-context-compact-now", Button).disabled
                is False
            )
            assert "Fresh durable memory after compaction" in str(
                reopened.query_one("#console-settings-memory-review", Static).renderable
            )
            assert "Compacting" not in str(
                reopened.query_one("#console-context-action-status", Static).renderable
            )
    finally:
        release.set()


class TestReasoningEffortHints:
    def test_dotted_qwen_generations_are_effort_capable(self):
        # "none" is included: it is consumed via our enable_thinking=false
        # mapping on dotted Qwens (live-verified). "high" is included: the
        # template aliases it to "xhigh" (live-verified), so warning on it
        # would be a false positive against the actual wire behavior.
        for model in ("Qwen3.8-27B", "qwen3.5-397b-gguf:q4"):
            assert reasoning_effort_hint_for_model(model) == frozenset(
                {"low", "medium", "high", "xhigh", "none"}
            )

    def test_original_qwen3_is_toggle_only(self):
        assert reasoning_effort_hint_for_model("Qwen3-32B") == frozenset({"none"})

    def test_gpt_oss(self):
        assert reasoning_effort_hint_for_model("gpt-oss-120b") == frozenset(
            {"low", "medium", "high"}
        )

    def test_unknown_model_has_no_hint(self):
        assert reasoning_effort_hint_for_model("llama-3-8b") is None
        assert reasoning_effort_hint_for_model(None) is None
        assert reasoning_effort_hint_for_model("") is None


class TestConsoleSettingsWarnings:
    def _settings(self, **overrides):
        base = dict(provider="llama_cpp", model="Qwen3.8-27B")
        base.update(overrides)
        return ConsoleSessionSettings(**base)

    def test_value_outside_hint_warns(self):
        # Non-llama.cpp provider isolates the hint logic (the llama.cpp base
        # fixture would also add the --jinja requirements note). "minimal" is
        # genuinely unconsumed on dotted Qwens: the wire composer's
        # template-safe guard drops it rather than sending it.
        settings = self._settings(provider="openai", reasoning_effort="minimal")
        warnings = console_settings_warnings(settings)
        assert len(warnings) == 1
        assert "minimal" in warnings[0]
        assert "xhigh" in warnings[0]

    def test_value_inside_hint_does_not_warn(self):
        settings = self._settings(provider="openai", reasoning_effort="xhigh")
        assert console_settings_warnings(settings) == []

    def test_unknown_model_does_not_warn(self):
        settings = self._settings(
            provider="openai", model="llama-3-8b", reasoning_effort="high"
        )
        assert console_settings_warnings(settings) == []

    def test_llama_family_thinking_note_included(self):
        settings = self._settings(reasoning_effort="low")
        warnings = console_settings_warnings(settings)
        assert any("--jinja" in w for w in warnings)

    def test_local_llm_thinking_note_included(self):
        # local-llm sends compose llama.cpp-family wire fields, so its users
        # need the --jinja/b9982 requirements note too.
        settings = self._settings(provider="local_llm", reasoning_effort="low")
        warnings = console_settings_warnings(settings)
        assert any("--jinja" in w for w in warnings)

    def test_llama_family_note_requires_a_thinking_value(self):
        settings = self._settings()
        assert console_settings_warnings(settings) == []

    def test_none_effort_on_dotted_qwen_does_not_warn(self):
        # "none" is consumed by dotted Qwens via our enable_thinking=false
        # mapping, so it must not warn as unconsumed.
        settings = self._settings(provider="openai", reasoning_effort="none")
        assert console_settings_warnings(settings) == []


class TestReadinessKeySetCaching:
    """TASK-18909: the readiness key-sets are pure functions of a constant.

    A warm Console switch called `build_console_settings_readiness` ~400
    times; each call re-resolved provider identity for all 29 handler keys
    twice (supported + send-capable) -- 24k identity resolutions, the
    largest app-side cost of the switch. The no-injection path must cache.
    """

    def test_default_key_sets_are_cached_objects(self):
        # Same object returned when no keys are injected: the frozensets are
        # rebuilt from a module constant, so repeated calls must not
        # recompute (and callers may rely on the identity for cheap checks).
        assert (
            session_settings._supported_readiness_keys()
            is session_settings._supported_readiness_keys()
        )
        assert (
            session_settings._send_capable_readiness_keys()
            is session_settings._send_capable_readiness_keys()
        )

    def test_injected_keys_are_not_served_from_cache(self):
        # The test-injection seam (native_provider_keys) changes the result;
        # it must never be served from the no-injection cache.
        with_injection = session_settings._supported_readiness_keys({"openai"})
        assert "openai" in with_injection
        assert with_injection is not session_settings._supported_readiness_keys()

    def test_cached_set_matches_uncached_derivation(self):
        # The cached set equals a fresh derivation over the same constant.
        fresh = session_settings.supported_console_provider_readiness_keys(
            session_settings.CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
        )
        assert session_settings._supported_readiness_keys() == fresh
        assert session_settings._send_capable_readiness_keys() == fresh
