from types import SimpleNamespace

import pytest
from textual.widgets import Input, Select, TextArea

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


@pytest.mark.asyncio
async def test_settings_keyboard_category_focus_survives_selection_recompose():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("tab")
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.press("down")
        await pilot.press("enter")
        screen = _active_destination_screen(host)

        assert "Appearance" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_overview_paste_summary_updates_after_toggle(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        await pilot.click("#settings-category-overview")
        screen = _active_destination_screen(host)

        assert "Console paste collapse: Disabled: collapse large pastes" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_paste_toggle_keeps_keyboard_focus_after_refresh(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda *_args, **_kwargs: True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        toggle = screen.query_one("#settings-console-collapse-large-pastes-toggle")
        toggle.focus()

        await pilot.press("enter")
        await pilot.pause()
        assert host.focused is toggle
        assert "Disabled: collapse large pastes" in str(toggle.label)
        assert "Unsaved" in _visible_text(screen)

        await pilot.press("enter")
        await pilot.pause()

        assert host.focused is toggle
        assert "Enabled: collapse large pastes" in str(toggle.label)
        assert "No unsaved changes" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_console_behavior_stages_save_and_revert(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        screen = _active_destination_screen(host)

        assert "Unsaved" in _visible_text(screen)
        assert app.app_config["console"]["collapse_large_pastes"] is True

        await pilot.click("#settings-save-category")

    assert saved == [("console", "collapse_large_pastes", False)]
    assert app.app_config["console"]["collapse_large_pastes"] is False


@pytest.mark.asyncio
async def test_settings_console_behavior_revert_discards_draft(monkeypatch):
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_screen.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        await pilot.click("#settings-console-collapse-large-pastes-toggle")
        screen = _active_destination_screen(host)

        assert "Unsaved" in _visible_text(screen)

        await pilot.click("#settings-revert-category")
        assert "No unsaved changes" in _visible_text(screen)

    assert saved == []
    assert app.app_config["console"]["collapse_large_pastes"] is True


@pytest.mark.asyncio
async def test_settings_provider_category_uses_effective_console_source():
    app = _build_test_app()
    app.chat_api_provider_value = "OpenAI"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "qwen"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "llama_cpp" in text
        assert "qwen" in text
        assert "Source: chat_defaults" in text
        assert screen.query_one("#settings-provider-value", Input).value == "llama_cpp"
        assert screen.query_one("#settings-model-value", Input).value == "qwen"


@pytest.mark.asyncio
async def test_settings_provider_test_redacts_secrets(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAI", "model": "gpt-4.1"}
    app.app_config["api_settings"] = {"openai": {"api_key_env_var": "OPENAI_API_KEY"}}
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-token")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-test-provider")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Provider test" in text
        assert "OPENAI_API_KEY=<redacted>" in text
        assert "sk-" not in text


@pytest.mark.asyncio
async def test_settings_provider_category_saves_chat_defaults(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-provider-value", Input).value = "llama_cpp"
        screen.query_one("#settings-model-value", Input).value = "qwen"
        screen.query_one("#settings-streaming-default", Input).value = "false"
        screen.query_one("#settings-temperature-default", Input).value = "0.2"

        await pilot.click("#settings-save-category")

    assert saved == [
        ("chat_defaults", "provider", "llama_cpp"),
        ("chat_defaults", "model", "qwen"),
        ("chat_defaults", "streaming", False),
        ("chat_defaults", "temperature", 0.2),
    ]
    assert app.app_config["chat_defaults"] == {
        "provider": "llama_cpp",
        "model": "qwen",
        "streaming": False,
        "temperature": 0.2,
    }


@pytest.mark.asyncio
async def test_settings_provider_category_does_not_save_unedited_effective_defaults(monkeypatch):
    app = _build_test_app()
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1"
    app.app_config["chat_defaults"] = {}
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-save-category")

    assert saved == []
    assert app.app_config["chat_defaults"] == {}


@pytest.mark.asyncio
async def test_settings_provider_category_saves_only_dirty_provider_fields(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-model-value", Input).value = "gpt-4.1-mini"

        await pilot.click("#settings-save-category")

    assert saved == [("chat_defaults", "model", "gpt-4.1-mini")]
    assert app.app_config["chat_defaults"] == {
        "provider": "OpenAI",
        "model": "gpt-4.1-mini",
        "streaming": True,
        "temperature": 0.7,
    }


@pytest.mark.asyncio
async def test_settings_provider_test_blocks_unknown_provider():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "OpenAi Typo", "model": "fake-model"}
    app.app_config["api_settings"] = {}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-test-provider")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Unknown provider" in text
        assert "status=blocked" in text


@pytest.mark.asyncio
async def test_settings_provider_test_uses_api_settings_env_var_without_secret_leak(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "Groq", "model": "llama-3.3-70b-versatile"}
    app.app_config["api_settings"] = {
        "groq": {
            "api_key_env_var": "GROQ_API_KEY",
        }
    }
    monkeypatch.setenv("GROQ_API_KEY", "gsk-secret-token")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        await pilot.click("#settings-test-provider")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "env:GROQ_API_KEY" in text
        assert "GROQ_API_KEY=<redacted>" in text
        assert "gsk-secret-token" not in text


@pytest.mark.asyncio
async def test_settings_provider_test_tolerates_invalid_temperature_text():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "Ollama", "model": "llama3"}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        screen.query_one("#settings-temperature-default", Input).value = "not-a-number"

        await pilot.click("#settings-test-provider")
        text = _visible_text(screen)

        assert "Provider test" in text
        assert "status=ready" in text


@pytest.mark.parametrize(
    ("button_id", "expected"),
    [
        ("#settings-category-appearance", "Open Appearance"),
        ("#settings-category-storage", "Config path"),
        ("#settings-category-privacy-security", "Encryption"),
        ("#settings-category-diagnostics", "Validate config"),
        ("#settings-category-advanced-config", "Raw TOML"),
    ],
)
@pytest.mark.asyncio
async def test_settings_first_slice_categories_have_real_content(button_id, expected):
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click(button_id)
        screen = _active_destination_screen(host)

        assert expected in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_diagnostics_validate_and_reload_config_actions():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-diagnostics")
        screen = _active_destination_screen(host)
        await pilot.click("#settings-validate-config")
        await pilot.click("#settings-reload-config")
        text = _visible_text(screen)

        assert "Config validation: valid" in text
        assert "Config reload: loaded" in text


@pytest.mark.asyncio
async def test_settings_advanced_config_shows_raw_editor_and_safety_actions():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Raw TOML bypasses guided validation" in text
        assert screen.query_one("#settings-advanced-config-editor", TextArea)
        assert screen.query_one("#settings-advanced-validate-config")
        assert screen.query_one("#settings-advanced-save-config")


@pytest.mark.asyncio
async def test_settings_advanced_config_blocks_invalid_toml_and_redacts_secret():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)
        editor.text = "OPENAI_API_KEY=sk-secret-token\n[broken"

        await pilot.click("#settings-advanced-validate-config")
        text = _visible_text(screen)

        assert "Advanced config validation: invalid" in text
        assert "sk-secret-token" not in text


@pytest.mark.asyncio
async def test_settings_advanced_config_blocks_non_mapping_toml_on_save():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)
        editor.text = "42"

        await pilot.click("#settings-advanced-save-config")
        text = _visible_text(screen)

        assert "top-level TOML value must be a table" in text


@pytest.mark.asyncio
async def test_settings_advanced_config_saves_atomically_with_backup(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults]\nprovider = \"OpenAI\"\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)
        editor.text = "[chat_defaults]\nprovider = \"Ollama\"\nmodel = \"llama3\"\n"

        await pilot.click("#settings-advanced-save-config")
        text = _visible_text(screen)

        assert "Advanced config save: saved" in text

    assert config_path.read_text(encoding="utf-8") == (
        "[chat_defaults]\nprovider = \"Ollama\"\nmodel = \"llama3\"\n"
    )
    assert config_path.with_suffix(".toml.bak").exists()
