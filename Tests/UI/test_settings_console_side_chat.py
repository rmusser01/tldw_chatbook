"""Settings contracts for the ephemeral selection side-chat preferences."""

from __future__ import annotations

import pytest
from textual.widgets import Input, Static

import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
from Tests.UI.test_destination_shells import (
    DestinationHarness,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
)
from Tests.UI.test_settings_configuration_hub import (
    _open_settings_category,
    _settle_settings_mount_storm,
    _wait_for_settings_search_focus,
    _wait_for_settings_text,
)
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

SIDECAT_MODEL_INPUT = "#settings-console-sidechat-model"
SIDECAT_TEMPLATE_INPUT = "#settings-console-sidechat-prompt-template"


@pytest.mark.asyncio
async def test_console_side_chat_settings_render_loaded_values_and_stage_edits():
    """Both inputs show the saved values; typing stages without mutating config."""
    app = _build_test_app()
    app.app_config["console"] = {
        "sidechat_model": "saved-sidechat-model",
        "sidechat_prompt_template": "Summarize this: {selection}",
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        model_input = screen.query_one(SIDECAT_MODEL_INPUT, Input)
        template_input = screen.query_one(SIDECAT_TEMPLATE_INPUT, Input)

        assert model_input.value == "saved-sidechat-model"
        assert model_input.placeholder == "empty = current session model"
        assert template_input.value == "Summarize this: {selection}"
        help_text = str(
            screen.query_one(
                "#settings-console-sidechat-prompt-template-help", Static
            ).renderable
        )
        assert "{selection}" in help_text

        model_input.value = "draft-sidechat-model"
        await pilot.pause()

        draft = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
        assert draft.dirty_keys == {"sidechat_model"}
        assert draft.values["sidechat_model"] == "draft-sidechat-model"
        assert app.app_config["console"]["sidechat_model"] == "saved-sidechat-model"

        template_input.value = "Explain this: {selection}"
        await pilot.pause()

        draft = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
        assert draft.dirty_keys == {"sidechat_model", "sidechat_prompt_template"}
        assert (
            app.app_config["console"]["sidechat_prompt_template"]
            == "Summarize this: {selection}"
        )


@pytest.mark.asyncio
async def test_console_side_chat_settings_are_searchable_and_have_focused_guidance():
    """The side-chat model field is findable via "/" and explains its contract."""
    app = _build_test_app()
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _settle_settings_mount_storm(pilot)
        screen = _active_destination_screen(host)
        await pilot.press("/")
        await _wait_for_settings_search_focus(screen, pilot)
        search = screen.query_one("#settings-category-search", Input)
        assert search.has_focus
        await pilot.press(*"side chat model")
        await _wait_for_settings_text(screen, pilot, "Console Behavior › Side chat model")
        await pilot.press("enter")
        for _ in range(8):
            await pilot.pause()

        assert screen.active_category == SettingsCategoryId.CONSOLE_BEHAVIOR.value
        assert host.focused is not None and host.focused.id == (
            "settings-console-sidechat-model"
        )
        visible = _visible_text(screen)
        assert "Purpose: Model used by the ephemeral selection side chat." in visible
        assert "Saved as: console.sidechat_model" in visible
        assert "Save: staged - press s to save, r to revert" in visible


@pytest.mark.asyncio
async def test_console_side_chat_save_payload_is_exact_and_updates_runtime(
    monkeypatch,
):
    """Successful category Save persists exactly the two staged string keys."""
    app = _build_test_app()
    app.app_config["console"] = {}
    saved: list[dict[str, dict[str, str]]] = []

    class FakeAdapter:
        def save_sections(self, section_values):
            saved.append(section_values)
            return True

    monkeypatch.setattr(settings_screen_module, "SettingsConfigAdapter", FakeAdapter)
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        model_input = screen.query_one(SIDECAT_MODEL_INPUT, Input)
        template_input = screen.query_one(SIDECAT_TEMPLATE_INPUT, Input)
        assert model_input.value == ""
        assert template_input.value == "Give me more details about: {selection}"

        model_input.value = "gpt-sidechat-test"
        template_input.value = "Explain this: {selection}"
        await pilot.pause()

        assert app.app_config["console"].get("sidechat_model") in (None, "")
        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert saved == [
            {
                "console": {
                    "sidechat_model": "gpt-sidechat-test",
                    "sidechat_prompt_template": "Explain this: {selection}",
                }
            }
        ]
        assert app.app_config["console"]["sidechat_model"] == "gpt-sidechat-test"
        assert (
            app.app_config["console"]["sidechat_prompt_template"]
            == "Explain this: {selection}"
        )
        assert screen.query_one(SIDECAT_MODEL_INPUT, Input).value == "gpt-sidechat-test"
        assert (
            screen.query_one(SIDECAT_TEMPLATE_INPUT, Input).value
            == "Explain this: {selection}"
        )
        assert SettingsCategoryId.CONSOLE_BEHAVIOR not in screen._settings_drafts
        assert "Console behavior settings saved." in _visible_text(screen)


@pytest.mark.asyncio
async def test_console_side_chat_failed_save_keeps_draft(monkeypatch):
    """Persistence failure keeps the staged strings without touching runtime."""
    app = _build_test_app()
    app.app_config["console"] = {}

    class FailingAdapter:
        def save_sections(self, section_values):
            return False

    monkeypatch.setattr(
        settings_screen_module,
        "SettingsConfigAdapter",
        FailingAdapter,
    )
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        screen.query_one(SIDECAT_MODEL_INPUT, Input).value = "draft-model"
        screen.query_one(SIDECAT_TEMPLATE_INPUT, Input).value = "Draft: {selection}"
        await pilot.pause()

        await pilot.click("#settings-save-category")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert (
            screen.query_one(SIDECAT_MODEL_INPUT, Input).value == "draft-model"
        )
        assert (
            screen.query_one(SIDECAT_TEMPLATE_INPUT, Input).value
            == "Draft: {selection}"
        )
        draft = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
        assert draft.dirty_keys == {"sidechat_model", "sidechat_prompt_template"}
        assert app.app_config["console"].get("sidechat_model") in (None, "")
        assert "Your draft is still here" in _visible_text(screen)


@pytest.mark.asyncio
async def test_console_side_chat_revert_restores_loaded_values():
    """Category Revert discards staged side-chat strings and reloads inputs."""
    app = _build_test_app()
    app.app_config["console"] = {
        "sidechat_model": "saved-model",
        "sidechat_prompt_template": "Saved: {selection}",
    }
    host = DestinationHarness(app, "settings")

    async with host.run_test(size=(190, 55)) as pilot:
        await _open_settings_category(pilot, "#settings-category-console-behavior")
        screen = _active_destination_screen(host)
        screen.query_one(SIDECAT_MODEL_INPUT, Input).value = "unsaved-model"
        screen.query_one(SIDECAT_TEMPLATE_INPUT, Input).value = "Unsaved: {selection}"
        await pilot.pause()

        draft = screen._settings_drafts[SettingsCategoryId.CONSOLE_BEHAVIOR]
        assert draft.dirty_keys == {"sidechat_model", "sidechat_prompt_template"}

        screen._revert_category(SettingsCategoryId.CONSOLE_BEHAVIOR)
        await pilot.pause()

        assert SettingsCategoryId.CONSOLE_BEHAVIOR not in screen._settings_drafts
        assert screen.query_one(SIDECAT_MODEL_INPUT, Input).value == "saved-model"
        assert (
            screen.query_one(SIDECAT_TEMPLATE_INPUT, Input).value
            == "Saved: {selection}"
        )
        assert app.app_config["console"]["sidechat_model"] == "saved-model"
