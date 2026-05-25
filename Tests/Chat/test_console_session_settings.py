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


def test_readiness_wip_precedes_missing_key_for_openai() -> None:
    settings = ConsoleSessionSettings(provider="openai", model="gpt-4.1")

    readiness = build_console_settings_readiness(settings, app_config={"api_settings": {}})

    assert readiness.label == "WIP"
    assert "not wired" in readiness.detail


def test_invalid_url_precedes_wip_for_url_provider() -> None:
    settings = ConsoleSessionSettings(provider="vllm", model="m", base_url="file:///tmp/x")

    readiness = build_console_settings_readiness(settings, app_config={})

    assert readiness.label == "Invalid URL"


def test_invalid_url_validation_does_not_call_impure_validate_url(monkeypatch) -> None:
    def impure_validate_url(_url: str) -> bool:
        raise AssertionError("validate_url should not be called by pure settings helpers")

    monkeypatch.setattr(session_settings, "validate_url", impure_validate_url, raising=False)
    settings = ConsoleSessionSettings(provider="vllm", model="m", base_url="file:///tmp/x")

    readiness = build_console_settings_readiness(settings, app_config={})
    errors = validate_console_session_settings(settings, app_config={})

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
    assert unknown.label == "Unknown"


def test_readiness_unsupported_provider_missing_key_is_still_primary_wip() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="anthropic", model="claude-sonnet"),
        app_config={"api_settings": {"anthropic": {"api_key_env_var": "MISSING_KEY"}}},
        environ={},
    )

    assert readiness.label == "WIP"
    assert "missing API key" in readiness.detail


def test_readiness_configured_unknown_non_native_provider_is_wip() -> None:
    readiness = build_console_settings_readiness(
        ConsoleSessionSettings(provider="future_provider", model="future-model"),
        app_config={"api_settings": {"future_provider": {}}},
        environ={},
    )

    assert readiness.label == "WIP"
    assert "not wired" in readiness.detail


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


def test_context_estimate_token_counter_failure_uses_unknown_copy(monkeypatch) -> None:
    def fail_count(*_args: object, **_kwargs: object) -> int:
        raise RuntimeError("tokenizer unavailable")

    monkeypatch.setattr(session_settings, "count_tokens_chat_history", fail_count)

    estimate = build_console_context_estimate(
        messages=[{"role": "user", "content": "hello world"}],
        provider="openai",
        model="gpt-3.5-turbo",
    )

    assert estimate.used_tokens is None
    assert estimate.token_limit is None
    assert estimate.label == "Context: unknown"
