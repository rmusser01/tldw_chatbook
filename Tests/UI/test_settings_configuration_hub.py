from types import SimpleNamespace

from tldw_chatbook.UI.Screens.provider_model_resolution import (
    resolve_effective_provider_model,
)


def _app(
    *,
    provider=None,
    api_model=None,
    model=None,
    defaults=None,
):
    return SimpleNamespace(
        app_config={"chat_defaults": defaults or {}},
        chat_api_provider_value=provider,
        chat_api_model_value=api_model,
        chat_model_value=model,
    )


def test_effective_provider_model_prefers_console_overrides():
    app = _app(
        provider="OpenAI",
        api_model="gpt-4.1",
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(
        app,
        console_provider="Anthropic",
        console_model="claude",
    )

    assert result.provider == "Anthropic"
    assert result.model == "claude"
    assert result.provider_source == "console_control"
    assert result.model_source == "console_control"


def test_effective_provider_model_preserves_configured_provider_when_reactive_is_default_openai():
    app = _app(
        provider="OpenAI",
        api_model=None,
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(app)

    assert result.provider == "llama_cpp"
    assert result.provider_source == "chat_defaults"
    assert result.model == "qwen"


def test_effective_provider_model_prefers_settings_draft_values():
    app = _app(
        provider="OpenAI",
        api_model="gpt-4.1",
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(
        app,
        settings_provider="Ollama",
        settings_model="llama3.1",
    )

    assert result.provider == "Ollama"
    assert result.model == "llama3.1"
    assert result.provider_source == "settings_draft"
    assert result.model_source == "settings_draft"


def test_effective_provider_model_ignores_blank_provider_overrides_for_default_fallback():
    app = _app(
        provider="OpenAI",
        api_model=None,
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(
        app,
        settings_provider=" ",
        console_provider="None",
    )

    assert result.provider == "llama_cpp"
    assert result.provider_source == "chat_defaults"


def test_effective_provider_model_ignores_blank_reactive_provider_for_default_fallback():
    for reactive_provider in ("", " ", "None"):
        app = _app(
            provider=reactive_provider,
            api_model=None,
            model=None,
            defaults={"provider": "llama_cpp", "model": "qwen"},
        )

        result = resolve_effective_provider_model(app)

        assert result.provider == "llama_cpp"
        assert result.provider_source == "chat_defaults"
