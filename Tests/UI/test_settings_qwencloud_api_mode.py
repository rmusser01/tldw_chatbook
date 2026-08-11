"""Canonical Settings coverage for QwenCloud's provider-scoped API mode."""

from __future__ import annotations

import pytest
from textual.widgets import Select, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
)
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_configuration_hub import _open_settings_category
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId


def _qwencloud_app(*, api_mode: object = "__absent__"):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "QwenCloud",
        "model": "qwen3.8-max",
    }
    qwencloud = {
        "api_base_url": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
        "api_key_env_var": "DASHSCOPE_API_KEY",
        "model": "qwen3.8-max",
    }
    if api_mode != "__absent__":
        qwencloud["api_mode"] = api_mode
    app.app_config["api_settings"] = {"qwencloud": qwencloud}
    return app


def _switch_to_qwencloud(screen) -> None:
    """Select QwenCloud through the first-class Settings provider inventory."""
    provider = screen.query_one("#settings-provider-value", Select)
    provider.value = "qwencloud"
    screen.handle_provider_value_changed(Select.Changed(provider, "qwencloud"))


@pytest.mark.asyncio
async def test_qwencloud_api_mode_selector_visibility_options_and_default():
    app = _qwencloud_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        selector = screen.query_one("#settings-provider-api-mode", Select)
        row = screen.query_one("#settings-provider-api-mode-row")
        provider = screen.query_one("#settings-provider-value", Select)

        assert selector.value == "responses"
        assert selector.disabled is False
        assert selector.can_focus is True
        assert row.has_class("settings-gated-profile-hidden") is False
        assert {
            (str(label), str(value))
            for label, value in selector._options
            if value is not Select.NULL
        } == {
            ("Responses", "responses"),
            ("Chat Completions", "chat_completions"),
        }
        assert "API mode" in str(row.query_one(".settings-input-label", Static).content)
        assert provider.value == "qwencloud"

        provider.value = "openai"
        screen.handle_provider_value_changed(Select.Changed(provider, "openai"))
        await pilot.pause()

        assert selector.disabled is True
        assert row.has_class("settings-gated-profile-hidden") is True


@pytest.mark.asyncio
async def test_qwencloud_api_mode_loads_saved_chat_completions():
    app = _qwencloud_app(api_mode="chat_completions")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        selector = screen.query_one("#settings-provider-api-mode", Select)
        assert selector.value == "chat_completions"
        assert selector.disabled is False


@pytest.mark.asyncio
async def test_normalized_saved_qwencloud_mode_does_not_create_mount_draft():
    app = _qwencloud_app(api_mode="  Chat_Completions  ")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert (
            screen.query_one("#settings-provider-api-mode", Select).value
            == "chat_completions"
        )
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts


@pytest.mark.asyncio
async def test_returning_qwencloud_mode_to_original_removes_only_its_namespace():
    app = _qwencloud_app(api_mode="responses")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        screen._stage_provider_value("model", "qwen3.8-max-draft")

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        mode.value = "responses"
        screen.handle_provider_api_mode_changed(Select.Changed(mode, "responses"))

        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["model"] == "qwen3.8-max-draft"
        assert "provider_api_mode:qwencloud" not in draft.values
        assert "provider_api_mode:qwencloud" not in draft.originals


@pytest.mark.asyncio
async def test_immediate_qwencloud_save_snapshots_visible_mode(monkeypatch):
    app = _qwencloud_app(api_mode="responses")
    saved: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = "chat_completions"
        screen.action_settings_save_category(allow_text_entry_focus=True)

        assert saved == [("api_settings.qwencloud", "api_mode", "chat_completions")]
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == (
            "chat_completions"
        )


@pytest.mark.asyncio
async def test_qwencloud_api_mode_accepts_normalized_provider_identity():
    app = _qwencloud_app(api_mode="responses")
    app.app_config["chat_defaults"]["provider"] = "  QWENCLOUD  "
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)

        assert screen.query_one("#settings-provider-value", Select).value == "qwencloud"
        mode = screen.query_one("#settings-provider-api-mode", Select)
        assert mode.disabled is False
        assert mode.value == "responses"


@pytest.mark.asyncio
async def test_qwencloud_readiness_overlays_draft_mode_without_persisting():
    app = _qwencloud_app(api_mode="responses")
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        staged = screen._provider_test_staged_config("QWENCLOUD")

        assert staged["api_settings"]["qwencloud"]["api_mode"] == ("chat_completions")
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"


@pytest.mark.asyncio
async def test_qwencloud_save_failure_preserves_mode_input_and_draft(monkeypatch):
    app = _qwencloud_app(api_mode="responses")
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda _section, _key, _value: False,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        screen.action_settings_save_category(allow_text_entry_focus=True)

        assert mode.value == "chat_completions"
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"
        assert "Failed to save" in str(
            screen.query_one("#settings-provider-save-result", Static).content
        )


@pytest.mark.asyncio
async def test_qwencloud_api_mode_draft_survives_provider_switch():
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"]["openai"] = {
        "api_base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1",
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        provider = screen.query_one("#settings-provider-value", Select)

        # Switch immediately after changing the widget. The provider transition
        # must snapshot QwenCloud even if its queued Changed event has not run.
        mode.value = "chat_completions"
        provider.value = "openai"
        screen.handle_provider_value_changed(Select.Changed(provider, "openai"))
        await pilot.pause()

        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"
        assert mode.disabled is True

        _switch_to_qwencloud(screen)
        await pilot.pause()

        assert mode.value == "chat_completions"
        assert mode.disabled is False


@pytest.mark.asyncio
async def test_saving_other_provider_never_mutates_qwencloud_mode(monkeypatch):
    app = _qwencloud_app(api_mode="responses")
    app.app_config["api_settings"]["openai"] = {
        "api_base_url": "https://api.openai.com/v1",
        "model": "gpt-4.1",
    }
    saved: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        provider = screen.query_one("#settings-provider-value", Select)

        mode.value = "chat_completions"
        screen.handle_provider_api_mode_changed(
            Select.Changed(mode, "chat_completions")
        )
        provider.value = "openai"
        screen.handle_provider_value_changed(Select.Changed(provider, "openai"))
        await pilot.pause()
        await pilot.click("#settings-save-category")

        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"
        assert not any(section == "api_settings.qwencloud" for section, _, _ in saved)
        draft = screen._settings_drafts[SettingsCategoryId.PROVIDERS_MODELS]
        assert draft.values["provider_api_mode:qwencloud"] == "chat_completions"

        _switch_to_qwencloud(screen)
        await pilot.pause()
        assert mode.value == "chat_completions"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("saved_mode", "selected_mode"),
    [
        ("responses", "chat_completions"),
        ("chat_completions", "responses"),
    ],
)
async def test_qwencloud_api_mode_save_and_revert_exact_values(
    monkeypatch,
    saved_mode: str,
    selected_mode: str,
):
    app = _qwencloud_app(api_mode=saved_mode)
    saved: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        mode.value = selected_mode
        screen.handle_provider_api_mode_changed(Select.Changed(mode, selected_mode))
        await pilot.click("#settings-save-category")

        assert saved == [
            ("api_settings.qwencloud", "api_mode", selected_mode),
        ]
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == selected_mode
        draft = screen._settings_drafts.get(SettingsCategoryId.PROVIDERS_MODELS)
        assert draft is None or "provider_api_mode:qwencloud" not in draft.values

        mode.value = saved_mode
        screen.handle_provider_api_mode_changed(Select.Changed(mode, saved_mode))
        await pilot.click("#settings-revert-category")
        await pilot.pause()
        await pilot.click("#confirm-button")
        await pilot.pause()

        assert mode.value == selected_mode
        assert SettingsCategoryId.PROVIDERS_MODELS not in screen._settings_drafts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_mode",
    ["response", "", "   ", 42, False, [], {}],
    ids=("unknown", "empty", "whitespace", "integer", "boolean", "list", "mapping"),
)
async def test_invalid_persisted_qwencloud_mode_blocks_save_and_send(
    monkeypatch,
    invalid_mode: object,
):
    app = _qwencloud_app(api_mode=invalid_mode)
    app.app_config["api_settings"]["qwencloud"]["api_key"] = "KEY-CANARY"
    saved: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)

        assert mode.has_class("settings-invalid-input")
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause()

        assert saved == []
        assert "QwenCloud API mode is invalid" in str(
            screen.query_one("#settings-provider-save-result", Static).content
        )
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == invalid_mode

    gateway = ConsoleProviderGateway(
        config_provider=lambda: app.app_config,
        environ={},
        chat_api_call_fn=lambda **_kwargs: pytest.fail(
            "invalid QwenCloud mode must block before dispatch"
        ),
    )
    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="QwenCloud")
    )

    assert resolution.ready is False
    assert "invalid API mode setting" in resolution.visible_copy
    assert "KEY-CANARY" not in resolution.visible_copy


@pytest.mark.asyncio
async def test_invalid_qwencloud_mode_can_be_explicitly_corrected_to_responses(
    monkeypatch,
):
    app = _qwencloud_app(api_mode="invalid")
    saved: list[tuple[str, str, object]] = []
    monkeypatch.setattr(
        "tldw_chatbook.UI.Screens.settings_config_adapter.save_setting_to_cli_config",
        lambda section, key, value: saved.append((section, key, value)) or True,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        mode.focus()
        await pilot.pause()

        screen.handle_provider_api_mode_changed(Select.Changed(mode, "responses"))
        screen.action_settings_save_category(allow_text_entry_focus=True)

        assert saved == [("api_settings.qwencloud", "api_mode", "responses")]
        assert app.app_config["api_settings"]["qwencloud"]["api_mode"] == "responses"
        assert mode.has_class("settings-invalid-input") is False


@pytest.mark.asyncio
async def test_qwencloud_api_mode_field_guide_describes_mode_contract():
    app = _qwencloud_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(180, 50)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        screen = _active_destination_screen(host)
        mode = screen.query_one("#settings-provider-api-mode", Select)
        mode.focus()
        await pilot.pause()

        assert screen._active_settings_field_id == "settings-provider-api-mode"
        guide = " | ".join(
            f"{label}: {value}"
            for label, value in screen._provider_field_guidance_rows()
        )
        assert "Focused setting: API mode" in guide
        assert "api_settings.qwencloud.api_mode" in guide
        assert "Responses is stateless with store=false" in guide
        assert "Chat Completions disables thinking replay" in guide
        assert "existing function tools work in both" in guide
        assert "QwenCloud built-in tools are excluded" in guide

        ownership = screen._ownership_record(SettingsCategoryId.PROVIDERS_MODELS)
        assert "api_settings.<provider>.api_mode" in ownership.owns_config_sections
