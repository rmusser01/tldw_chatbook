from types import SimpleNamespace

import pytest
from textual.widgets import Select

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    resolve_effective_provider_model,
)
from tldw_chatbook.UI.Screens.settings_config_adapter import (
    SettingsConfigAdapter,
    redact_secret_text,
)
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
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


def test_effective_provider_model_ignores_textual_blank_select_provider_for_default_fallback():
    app = _app(
        provider="OpenAI",
        api_model=None,
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    result = resolve_effective_provider_model(app, settings_provider=Select.BLANK)

    assert result.provider == "llama_cpp"
    assert result.provider_source == "chat_defaults"


def test_settings_draft_tracks_dirty_values():
    draft = SettingsDraft(category=SettingsCategoryId.CONSOLE_BEHAVIOR)
    draft.set_value("collapse_large_pastes", True, False)

    assert draft.is_dirty
    assert draft.dirty_keys == {"collapse_large_pastes"}


def test_redact_secret_text_removes_api_key_like_values():
    text = "failed with OPENAI_API_KEY=sk-secret-token and token abc"

    redacted = redact_secret_text(text)

    assert "sk-secret-token" not in redacted
    assert "OPENAI_API_KEY=<redacted>" in redacted


def test_adapter_rejects_non_mapping_toml():
    adapter = SettingsConfigAdapter()

    result = adapter.validate_raw_toml('"not a mapping"')

    assert not result.valid
    assert "top-level TOML value must be a table" in result.message


def test_adapter_rejects_scalar_like_toml_with_table_message():
    adapter = SettingsConfigAdapter()

    for value in (
        "42",
        "true",
        "[1, 2]",
        "nan",
        "inf",
        "0xDEADBEEF",
        "1979-05-27",
        "1979-05-27T07:32:00Z",
    ):
        result = adapter.validate_raw_toml(value)

        assert not result.valid
        assert "top-level TOML value must be a table" in result.message


def test_adapter_accepts_table_headers_before_scalar_fallback():
    adapter = SettingsConfigAdapter()

    for value in ("[section]", "[[items]]"):
        result = adapter.validate_raw_toml(value)

        assert result.valid


def test_adapter_save_values_attempts_all_keys_when_one_save_fails(monkeypatch):
    calls = []

    def fake_save(section, key, value):
        calls.append((section, key, value))
        return key != "provider"

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        fake_save,
    )

    result = SettingsConfigAdapter().save_values(
        "chat_defaults",
        {"provider": "bad", "model": "still-attempted"},
    )

    assert not result
    assert calls == [
        ("chat_defaults", "provider", "bad"),
        ("chat_defaults", "model", "still-attempted"),
    ]


@pytest.mark.asyncio
async def test_settings_defaults_to_overview_category():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Overview" in text
        assert "Provider readiness" in text
        assert "Storage" in text
        assert "Privacy" in text
        assert "Console paste collapse" in text


@pytest.mark.asyncio
async def test_settings_category_selection_updates_detail_and_inspector():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Console Behavior" in text
        assert "Collapse large pasted chunks" in text
        assert "Affects Console" in text


@pytest.mark.asyncio
async def test_settings_tab_focus_and_enter_select_categories():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("tab")
        await pilot.press("down")
        await pilot.press("enter")
        screen = _active_destination_screen(host)

        assert "Providers & Models" in _visible_text(screen)
