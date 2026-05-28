import builtins
import inspect

import tldw_chatbook.Chat.console_session_settings as session_settings
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_context_estimate,
    build_console_settings_readiness,
    build_default_console_session_settings,
    build_console_model_options,
    build_console_provider_options,
    validate_console_session_settings,
)


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


def test_readiness_does_not_import_gateway_or_config_runtime_modules(monkeypatch) -> None:
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
    assert settings.model == "configured-model"
    assert settings.base_url == "http://127.0.0.1:9099"
    assert settings.temperature == 0.2
    assert settings.top_p == 0.8
    assert settings.min_p == 0.05
    assert settings.top_k == 40
    assert settings.max_tokens == 2048


def test_default_settings_uses_api_base_for_llamacpp_base_url() -> None:
    settings = build_default_console_session_settings(
        {
            "chat_defaults": {"provider": "llama_cpp"},
            "api_settings": {"llama_cpp": {"api_base": "127.0.0.1:9191/v1"}},
        },
        provider="llama_cpp",
    )

    assert settings.base_url == "http://127.0.0.1:9191"


def test_public_helpers_accept_planned_positional_call_forms() -> None:
    config = {
        "chat_defaults": {"provider": "llama_cpp", "model": "chat-default"},
        "api_settings": {"llama_cpp": {"api_url": "127.0.0.1:9099/v1"}},
    }

    settings = build_default_console_session_settings(config, "llama_cpp", None)
    provider_options = build_console_provider_options({"llama_cpp": ["m"]})
    model_options = build_console_model_options("llama_cpp", {"llama_cpp": ["m"]}, "current")
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
    assert [option.value for option in provider_options] == ["llama_cpp"]
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


def test_provider_options_include_all_configured_providers() -> None:
    options = build_console_provider_options(
        providers_models={
            "llama_cpp": ["local-model"],
            "openai": ["gpt-4.1"],
            "anthropic": ["claude-sonnet"],
        }
    )

    assert [option.value for option in options] == ["anthropic", "llama_cpp", "openai"]


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
    assert "Max tokens must be 1 or greater." in errors


def test_readiness_reports_missing_key_for_supported_openai_instead_of_wip() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1"),
        app_config={"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}},
        environ={},
    )

    assert readiness.label == "Missing key"
    assert "OPENAI_API_KEY" in readiness.detail
    assert "not wired" not in readiness.detail


def test_readiness_reports_pending_for_keyless_supported_generic_provider() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="ollama", model="llama3", base_url="http://127.0.0.1:11434"),
        app_config={"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    assert readiness.label == "Pending"
    assert readiness.detail == "Provider ready; Console send support is pending for 'ollama'."
    assert "not wired" not in readiness.detail
    assert readiness.native_send_supported is False


def test_readiness_allows_configured_url_with_trailing_slash() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="ollama", model="llama3", base_url="http://127.0.0.1:11434/"),
        app_config={"api_settings": {"ollama": {"api_url": "http://127.0.0.1:11434"}}},
        environ={},
    )

    assert readiness.label == "Pending"
    assert readiness.native_send_supported is False


def test_readiness_explicit_send_capable_injection_allows_supported_generic_provider() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="ollama", model="llama3", base_url="http://127.0.0.1:11434"),
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
    settings = ConsoleSessionSettings(provider="vllm", model="m", base_url="file:///tmp/x")

    readiness = build_console_settings_readiness(settings, app_config={})

    assert readiness.label == "Invalid URL"


def test_malformed_ipv6_url_returns_validation_and_readiness_errors() -> None:
    settings = ConsoleSessionSettings(provider="vllm", model="m", base_url="http://[::1")

    readiness = build_console_settings_readiness(settings, app_config={})
    errors = validate_console_session_settings(settings, app_config={})

    assert readiness.label == "Invalid URL"
    assert "Base URL must be a valid http(s) URL." in errors


def test_whitespace_host_url_returns_validation_and_readiness_errors() -> None:
    settings = ConsoleSessionSettings(provider="vllm", model="m", base_url="http://exa mple.com")

    readiness = build_console_settings_readiness(settings, app_config={})
    errors = validate_console_session_settings(settings, app_config={})

    assert readiness.label == "Invalid URL"
    assert "Base URL must be a valid http(s) URL." in errors


def test_invalid_port_urls_return_validation_and_readiness_errors() -> None:
    for invalid_url in ("http://example.com:99999", "http://example.com:nope"):
        settings = ConsoleSessionSettings(provider="vllm", model="m", base_url=invalid_url)

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
    app_config = {"api_settings": {"future_provider": {"api_url": "http://127.0.0.1:9000"}}}

    readiness = build_console_settings_readiness(settings, app_config=app_config, environ={})
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
        app_config={"api_settings": {"future_provider": {"api_url": "http://127.0.0.1:9000"}}},
        environ={},
    )

    assert readiness.label == "Unknown"
    assert "Choose a supported provider" in readiness.detail
    assert "not wired" not in readiness.detail


def test_context_estimate_counts_messages_and_staged_sources() -> None:
    estimate = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello world"}],
        provider="openai",
        model="gpt-3.5-turbo",
        staged_source_count=2,
        staged_context_summary="2 staged sources",
        max_tokens_response=512,
        system_prompt="You are concise.",
    )

    assert estimate.used_tokens is not None
    assert estimate.used_tokens > 0
    assert estimate.token_limit == 4096
    assert "tokens" in estimate.label
    assert "2 sources staged" in estimate.label
    assert estimate.staged_source_count == 2
    assert estimate.staged_context_summary == "2 staged sources"


def test_context_estimate_uses_longest_matching_token_limit_prefix() -> None:
    def token_counter(_messages: list[dict[str, str]], _model: str, _provider: str) -> int:
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
