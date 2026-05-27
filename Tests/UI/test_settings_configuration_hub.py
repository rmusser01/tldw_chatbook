import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Input, Select, Static, TextArea

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from tldw_chatbook.UI.Screens.provider_model_resolution import (
    resolve_effective_provider_model,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.UI.Screens.settings_config_adapter import (
    SettingsConfigAdapter,
    redact_secret_text,
)
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
    SettingsValidationResult,
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


async def _wait_for_settings_text(screen, pilot, expected_text: str, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if expected_text in _visible_text(screen):
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for {expected_text!r}. Visible text: {_visible_text(screen)}")


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


def test_effective_provider_model_ignores_blank_model_overrides_for_default_fallback():
    app = _app(
        provider="OpenAI",
        api_model=None,
        model=None,
        defaults={"provider": "llama_cpp", "model": "qwen"},
    )

    for blank_model in ("", " ", "None", Select.BLANK):
        result = resolve_effective_provider_model(
            app,
            settings_model=blank_model,
            console_model=" ",
        )

        assert result.model == "qwen"
        assert result.model_source == "chat_defaults"


def test_effective_provider_model_handles_non_mapping_app_config():
    app = SimpleNamespace(
        app_config=[],
        chat_api_provider_value=None,
        chat_api_model_value=None,
        chat_model_value=None,
    )

    result = resolve_effective_provider_model(app)

    assert result.provider is None
    assert result.model is None


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


def test_adapter_validate_config_file_rejects_corrupt_toml(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults\nprovider = \"OpenAI\"\n", encoding="utf-8")

    result = SettingsConfigAdapter().validate_config_file(config_path)

    assert not result.valid
    assert "Expected" in result.message or "Invalid" in result.message


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
async def test_settings_category_navigation_is_grouped_for_scan():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        for group_title in (
            "Core",
            "Interface",
            "Data & Privacy",
            "Troubleshooting",
            "Expert",
        ):
            assert group_title in text


@pytest.mark.asyncio
async def test_settings_active_category_uses_explicit_nav_marker():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        active = screen.query_one("#settings-category-advanced-config")
        inactive = screen.query_one("#settings-category-diagnostics")

        assert str(active.label) == "> Advanced Config"
        assert str(inactive.label) == "  Diagnostics"
        assert active.has_class("settings-active-section")


def test_settings_active_category_focus_style_keeps_label_readable():
    css_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/components/_agentic_terminal.tcss"
    )
    css = css_path.read_text()
    match = re.search(
        r"Button\.settings-category-button\.settings-active-section:focus\s*\{(?P<body>[^}]*)\}",
        css,
        flags=re.DOTALL,
    )

    assert match
    body = match.group("body")
    assert "reverse" not in body
    assert "text-style: bold underline;" in body


def test_settings_action_button_focus_style_keeps_label_readable():
    css_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/components/_agentic_terminal.tcss"
    )
    css = css_path.read_text()
    match = re.search(
        r"\.settings-action-row Button:focus,\s*#settings-impact-pane Button:focus\s*\{(?P<body>[^}]*)\}",
        css,
        flags=re.DOTALL,
    )

    assert match
    body = match.group("body")
    assert "reverse" not in body
    assert "text-style: bold underline;" in body
    assert "outline: none;" in body


def test_settings_shell_button_focus_does_not_use_heavy_outline():
    css_path = (
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook/css/components/_agentic_terminal.tcss"
    )
    css = css_path.read_text()
    match = re.search(
        r"#settings-shell Button:focus,\s*#settings-shell Button:hover:focus\s*\{(?P<body>[^}]*)\}",
        css,
        flags=re.DOTALL,
    )

    assert match
    body = match.group("body")
    assert "outline: none;" in body
    assert "text-style: bold underline;" in body


@pytest.mark.asyncio
async def test_settings_detail_shows_state_banner_and_structured_rows():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)):
        screen = _active_destination_screen(host)
        banner = screen.query_one("#settings-category-state-banner")
        detail_rows = list(screen.query(".settings-detail-row"))

        assert "State:" in str(banner.renderable)
        assert banner.has_class("settings-state-banner")
        assert len(detail_rows) >= 5


@pytest.mark.asyncio
async def test_settings_inspector_uses_category_specific_guidance():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-storage")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        assert "Affected config: config file path, local database paths, media storage roots" in text
        assert "Recovery: verify paths, reload config, then restart only if storage roots changed" in text
        assert "MCP and tool-control settings live under MCP" not in text


@pytest.mark.asyncio
async def test_settings_inspector_boundary_is_structured_without_duplicate_copy():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-advanced-config")
        screen = _active_destination_screen(host)
        text = _visible_text(screen)

        boundary = "Boundary: save is blocked until the exact current text validates"
        assert str(screen.query_one("#settings-boundary-note").renderable) == boundary
        assert text.count(boundary) == 1


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
async def test_settings_category_search_filters_and_enter_opens_first_match():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("/")
        screen = _active_destination_screen(host)
        search = screen.query_one("#settings-category-search", Input)

        assert search.has_focus

        await pilot.press(*"priv")
        await pilot.pause()

        assert screen.query_one("#settings-category-privacy-security").display
        assert not screen.query_one("#settings-category-providers-models").display

        await pilot.press("enter")
        await pilot.pause()

        assert screen.active_category == SettingsCategoryId.PRIVACY_SECURITY.value
        assert "Privacy & Security" in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_category_search_reports_ranked_matches_and_enter_target():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("/")
        screen = _active_destination_screen(host)

        await pilot.press(*"priv")
        await pilot.pause()

        visible_text = _visible_text(screen)
        assert "Filter: priv | 2 matches | Enter opens Privacy & Security" in visible_text
        assert screen.query_one("#settings-category-privacy-security").has_class(
            "settings-primary-search-match"
        )
        assert screen.query_one("#settings-category-overview").has_class(
            "settings-secondary-search-match"
        )


@pytest.mark.asyncio
async def test_settings_category_search_uses_plain_standard_input_widgets():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.pause()
        screen = _active_destination_screen(host)

        search = screen.query_one("#settings-category-search", Input)
        assert type(search) is Input
        assert not screen.query_one("#settings-category-search-status", Static)._render_markup
        assert not screen.query_one("#settings-category-search-empty", Static)._render_markup


def test_settings_category_search_normalizes_oversized_control_input():
    screen = SettingsScreen(_build_test_app())

    normalized = screen._sanitize_category_search_query("[" + ("x" * 120) + "\x00")

    assert len(normalized) == 80
    assert normalized == "[" + ("x" * 79)
    assert "\x00" not in normalized


@pytest.mark.asyncio
async def test_settings_category_search_escape_clears_filter():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.press("/")
        screen = _active_destination_screen(host)

        await pilot.press(*"zzz")
        await pilot.pause()

        assert "No Settings categories match" in _visible_text(screen)
        assert not any(button.display for button in screen.query(".settings-category-button"))

        await pilot.press("escape")
        await pilot.pause()

        search = screen.query_one("#settings-category-search", Input)
        assert search.value == ""
        assert sum(1 for button in screen.query(".settings-category-button") if button.display) == len(
            screen._category_summaries()
        )


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
async def test_settings_non_editable_categories_disable_guided_save_revert():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: choose Providers or Console." in _visible_text(screen)

        await pilot.click("#settings-category-storage")
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: Storage is read-only." in _visible_text(screen)


@pytest.mark.asyncio
async def test_settings_console_guided_save_revert_enable_only_when_dirty():
    app = _build_test_app()
    app.app_config["console"] = {"collapse_large_pastes": True}
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-console-behavior")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: change a field first." in _visible_text(screen)

        await pilot.click("#settings-console-collapse-large-pastes-toggle")

        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert screen.query_one("#settings-revert-category", Button).disabled is False
        assert "Guided edits: Save or Revert changes." in _visible_text(screen)


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
async def test_settings_provider_guided_save_revert_enable_only_when_dirty():
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True
        assert "Guided edits: change a field first." in _visible_text(screen)

        model = screen.query_one("#settings-model-value", Input)
        model.value = "gpt-4.1-mini"
        screen.handle_model_value_changed(Input.Changed(model, model.value))

        assert screen.query_one("#settings-save-category", Button).disabled is False
        assert screen.query_one("#settings-revert-category", Button).disabled is False
        assert "Guided edits: Save or Revert changes." in _visible_text(screen)


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
async def test_settings_provider_category_saves_llamacpp_endpoint(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1"}
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

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        assert endpoint.value == "http://127.0.0.1:8080/v1"
        endpoint.value = "http://127.0.0.1:9099/v1"

        await pilot.click("#settings-save-category")

    assert saved == [
        ("api_settings.llama_cpp", "api_url", "http://127.0.0.1:9099/v1"),
    ]
    assert app.app_config["api_settings"]["llama_cpp"]["api_url"] == "http://127.0.0.1:9099/v1"


@pytest.mark.asyncio
async def test_settings_provider_category_preserves_existing_endpoint_key(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1"}
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

        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        assert endpoint.value == "https://api.openai.com/v1"
        endpoint.value = "https://proxy.example.com/v1"

        await pilot.click("#settings-save-category")

    assert saved == [
        ("api_settings.openai", "api_base_url", "https://proxy.example.com/v1"),
    ]
    assert app.app_config["api_settings"]["openai"]["api_base_url"] == "https://proxy.example.com/v1"


@pytest.mark.asyncio
async def test_settings_provider_endpoint_validation_blocks_bad_url(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:8080/v1"}
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
        screen.query_one("#settings-provider-endpoint-value", Input).value = "javascript:alert(1)"

        await pilot.click("#settings-save-category")
        text = _visible_text(screen)

        assert "Endpoint must start with http:// or https://" in text

    assert saved == []
    assert app.app_config["api_settings"]["llama_cpp"]["api_url"] == "http://127.0.0.1:8080/v1"


@pytest.mark.asyncio
async def test_settings_provider_endpoint_save_blocks_blank_provider(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "",
        "model": "model-a",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {}
    saved = []

    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-providers-models")
        screen = _active_destination_screen(host)
        provider = screen.query_one("#settings-provider-value", Input)
        provider.value = ""
        screen.handle_provider_value_changed(Input.Changed(provider, ""))
        await pilot.pause()
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://127.0.0.1:9099/v1"
        screen.handle_provider_endpoint_changed(Input.Changed(endpoint, endpoint.value))

        await pilot.click("#settings-save-category")
        text = _visible_text(screen)

        assert "Provider is required before saving an endpoint." in text

    assert saved == []
    assert app.app_config["api_settings"] == {}


@pytest.mark.asyncio
async def test_settings_provider_switch_does_not_save_stale_endpoint(monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "OpenAI",
        "model": "gpt-4.1",
        "streaming": True,
        "temperature": 0.7,
    }
    app.app_config["api_settings"] = {
        "openai": {"api_base_url": "https://api.openai.com/v1"},
        "llama_cpp": {},
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
        provider = screen.query_one("#settings-provider-value", Input)
        provider.value = "llama_cpp"
        screen.handle_provider_value_changed(Input.Changed(provider, "llama_cpp"))

        assert screen.query_one("#settings-provider-endpoint-value", Input).value == ""
        await pilot.click("#settings-save-category")
        await pilot.click("#settings-save-category")

    assert ("api_settings.llama_cpp", "api_url", "https://api.openai.com/v1") not in saved
    assert saved == [("chat_defaults", "provider", "llama_cpp")]
    assert app.app_config["api_settings"]["llama_cpp"] == {}


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
async def test_settings_diagnostics_test_shortcut_runs_validate_and_reload():
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-diagnostics")
        await pilot.press("t")
        screen = _active_destination_screen(host)
        await _wait_for_settings_text(screen, pilot, "Config reload: loaded")
        text = _visible_text(screen)

        assert "Config validation: valid" in text
        assert "Config reload: loaded" in text
        assert "No test action is available" not in text


def test_settings_diagnostics_combined_helper_validates_once(monkeypatch, tmp_path):
    class FakeAdapter:
        validate_calls = 0
        load_calls = 0

        def validate_config_file(self, path):
            FakeAdapter.validate_calls += 1
            return SettingsValidationResult(True, "valid once")

        def load(self, *, force_reload: bool = False):
            FakeAdapter.load_calls += 1
            return {"chat_defaults": {"provider": "OpenAI"}}

    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults]\nprovider = \"OpenAI\"\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)

    screen = SettingsScreen(_build_test_app())
    validation_result, reload_result, loaded_config = screen._diagnostics_validation_and_reload_results()

    assert FakeAdapter.validate_calls == 1
    assert FakeAdapter.load_calls == 1
    assert validation_result == "Config validation: valid - valid once"
    assert reload_result == "Config reload: loaded"
    assert loaded_config == {"chat_defaults": {"provider": "OpenAI"}}


def test_settings_diagnostics_combined_helper_skips_reload_when_invalid(monkeypatch, tmp_path):
    class FakeAdapter:
        validate_calls = 0
        load_calls = 0

        def validate_config_file(self, path):
            FakeAdapter.validate_calls += 1
            return SettingsValidationResult(False, "broken TOML")

        def load(self, *, force_reload: bool = False):
            FakeAdapter.load_calls += 1
            return {"chat_defaults": {"provider": "OpenAI"}}

    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults\nprovider = \"OpenAI\"\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)

    screen = SettingsScreen(_build_test_app())
    validation_result, reload_result, loaded_config = screen._diagnostics_validation_and_reload_results()

    assert FakeAdapter.validate_calls == 1
    assert FakeAdapter.load_calls == 0
    assert validation_result == "Config validation: invalid - broken TOML"
    assert reload_result == "Config reload: failed - broken TOML"
    assert loaded_config is None


def test_settings_diagnostics_strictly_reports_corrupt_toml(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text("[chat_defaults\nprovider = \"OpenAI\"\n", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = _build_test_app()
    screen = SettingsScreen(app)

    assert "Config validation: invalid" in screen._validate_current_config()
    assert "Config reload: failed" in screen._reload_current_config()


def test_settings_storage_check_reports_path_readiness(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "config.toml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = SimpleNamespace(
        app_config={},
        user_data_dir=data_dir,
        notifications_db_path=data_dir / "notifications.db",
        subscriptions_db_path=data_dir / "watchlists.db",
        workspaces_db_path=data_dir / "workspaces.db",
    )
    screen = SettingsScreen(app)

    result = screen._storage_check_results()

    assert result[0] == "Storage check: complete"
    assert "Config path parent: writable" in result
    assert "User data directory: writable" in result
    assert "Notifications DB parent: writable" in result
    assert "Watchlists DB parent: writable" in result
    assert "Workspaces DB parent: writable" in result


def test_settings_storage_check_reports_invalid_config_path(monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", "unsafe$(touch bad).toml")
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    result = screen._storage_check_results()

    assert result[0] == "Storage check: complete"
    assert any(row.startswith("Config path parent: invalid") for row in result)


def test_settings_storage_check_includes_configured_fallback_paths(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    screen = SettingsScreen(SimpleNamespace(app_config={}))

    result = screen._storage_check_results()

    assert any(row.startswith("User data directory:") for row in result)
    assert any(row.startswith("Notifications DB parent:") for row in result)
    assert any(row.startswith("Watchlists DB parent:") for row in result)
    assert any(row.startswith("Workspaces DB parent:") for row in result)


@pytest.mark.asyncio
async def test_settings_storage_test_shortcut_runs_safety_check(monkeypatch, tmp_path):
    config_path = tmp_path / "config" / "config.toml"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = _build_test_app()
    app.user_data_dir = data_dir
    app.notifications_db_path = data_dir / "notifications.db"
    app.subscriptions_db_path = data_dir / "watchlists.db"
    app.workspaces_db_path = data_dir / "workspaces.db"
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await pilot.click("#settings-category-storage")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-check-storage")
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True

        await pilot.press("t")
        await _wait_for_settings_text(screen, pilot, "Storage check: complete")
        text = _visible_text(screen)

        assert "Config path parent: writable" in text
        assert "User data directory: writable" in text
        assert "No test action is available" not in text
        assert screen.query_one("#settings-save-category", Button).disabled is True
        assert screen.query_one("#settings-revert-category", Button).disabled is True


def test_settings_config_path_validates_env_override(monkeypatch):
    app = _build_test_app()
    screen = SettingsScreen(app)
    monkeypatch.setenv("TLDW_CONFIG_PATH", "unsafe$(touch bad).toml")

    with pytest.raises(ValueError):
        screen._config_path()


def test_settings_advanced_config_save_reports_invalid_env_override(monkeypatch):
    app = SimpleNamespace(app_config={})
    screen = SettingsScreen(app)
    text = "[chat_defaults]\nprovider = \"Ollama\"\n"
    screen._advanced_config_validated_text = text
    monkeypatch.setenv("TLDW_CONFIG_PATH", "unsafe$(touch bad).toml")

    result = screen._save_advanced_config_text(text)

    assert "Advanced config save: failed" in result
    assert "dangerous pattern" in result


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
        save_button = screen.query_one("#settings-advanced-save-config")
        assert save_button.disabled
        assert "Last validated: not validated" in text
        assert "Save blocked until the current text validates" in text


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
        await _wait_for_settings_text(screen, pilot, "Advanced config validation: invalid")
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

        save_button = screen.query_one("#settings-advanced-save-config")

        assert save_button.disabled
        assert "top-level TOML value must be a table" in screen._save_advanced_config_text("42")


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

        assert screen.query_one("#settings-advanced-save-config").disabled

        await pilot.click("#settings-advanced-validate-config")
        await _wait_for_settings_text(screen, pilot, "Advanced config validation: valid")
        assert not screen.query_one("#settings-advanced-save-config").disabled
        await pilot.click("#settings-advanced-save-config")
        await _wait_for_settings_text(screen, pilot, "Advanced config save: saved")
        text = _visible_text(screen)

        assert "Advanced config save: saved" in text
        assert "Last validated: current text" in text

    assert config_path.read_text(encoding="utf-8") == (
        "[chat_defaults]\nprovider = \"Ollama\"\nmodel = \"llama3\"\n"
    )
    assert config_path.with_suffix(".toml.bak").exists()
    assert app.app_config["chat_defaults"]["provider"] == "Ollama"
    assert app.app_config["chat_defaults"]["model"] == "llama3"


def test_settings_advanced_config_new_file_save_reports_no_backup(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    app = SimpleNamespace(app_config={})
    screen = SettingsScreen(app)
    text = "[chat_defaults]\nprovider = \"Ollama\"\n"
    screen._advanced_config_validated_text = text

    result = screen._save_advanced_config_text(text)

    assert "Advanced config save: saved" in result
    assert "backup: none (new file)" in result
    assert config_path.exists()
    assert not config_path.with_suffix(".toml.bak").exists()
