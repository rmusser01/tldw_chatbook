"""Keyboard editing, Revert and Save keep provider draft and saved state honest."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from textual.screen import Screen
from textual.widgets import Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_configuration_hub import (
    _capture_provider_settings_mutations,
    _open_settings_category,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Screens import settings_endpoint_probe
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId

CATEGORY = SettingsCategoryId.PROVIDERS_MODELS
MODEL = "review-model"
ENDPOINT = "http://127.0.0.1:9098"


class ProviderSettingsHarness(DestinationHarness):
    CSS_PATH = TldwCli.CSS_PATH


async def _settle(host, pilot):
    await asyncio.wait_for(host.workers.wait_for_complete(), timeout=10)
    await pilot.wait_for_scheduled_animations()
    await pilot.pause()


async def _tab_to(host, pilot, selector):
    target = host.screen.query_one(selector)
    # TASK-33007.4, raised on purpose: this is a loop guard, not a keystroke
    # budget. Reaching Save selected from the discovered list wraps the whole
    # screen, and the Inspector's "config key" disclosure title is one more
    # stop on the way (101 at 170x48).
    for _ in range(150):
        if host.screen.focused is target:
            _assert_painted(host.screen, target)
            return target
        await pilot.press("tab")
        await _settle(host, pilot)
    pytest.fail(f"Keyboard focus never reached {selector}")


async def _edit(host, pilot, selector, value):
    field = await _tab_to(host, pilot, selector)
    await pilot.press("home", "shift+end", "backspace", *value)
    await _settle(host, pilot)
    assert field.value == value
    _assert_painted(host.screen, field)


async def _edit_custom_model(host, pilot, value):
    """TASK-33007.3: an id no list holds goes in through Custom ID."""
    field = await _tab_to(host, pilot, "#model-search-picker-input")
    picker = host.screen.query_one("#settings-model-picker")
    if not picker.custom_mode:
        await pilot.press("tab")
        await _settle(host, pilot)
        assert host.focused.id == "model-search-picker-custom"
        await pilot.press("enter")
        await _settle(host, pilot)
    assert host.focused is field and picker.custom_mode
    await pilot.press("home", "shift+end", "backspace", *value)
    await _settle(host, pilot)
    assert field.value == value
    assert host.screen.query_one("#settings-model-value", Input).value == value
    _assert_painted(host.screen, field)


async def _revert(host, pilot, *, discard):
    await pilot.press("escape", "r")
    await _settle(host, pilot)
    await _tab_to(
        host,
        pilot,
        "#confirm-button" if discard else "#cancel-button",
    )
    await pilot.press("enter")
    await _settle(host, pilot)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_provider_keyboard_edit_revert_save_and_return(theme, size, monkeypatch):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": "llama_cpp",
        "model": "model-a",
        "temperature": 0.7,
        "streaming": True,
    }
    app.app_config["api_settings"] = {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}
    mutations = _capture_provider_settings_mutations(monkeypatch)
    probe = AsyncMock(side_effect=AssertionError("Editing must not test the endpoint"))
    monkeypatch.setattr(settings_endpoint_probe, "probe_settings_endpoint", probe)
    host = ProviderSettingsHarness(app, "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        screen = host.screen
        assert not screen._category_has_unsaved_changes(CATEGORY)
        await _edit(host, pilot, "#settings-provider-endpoint-value", ENDPOINT)
        # TASK-33007.2, rewritten on purpose: Connect ends in the Key check
        # row and Model moved under "Default model for new chats". Test (t)
        # is not a Tab stop ('t' runs it; parent AC#2), so the actual
        # keyboard traversal runs Endpoint -> Model. TASK-33007.3, rewritten
        # on purpose: Model is the Default model picker, and an id no list
        # holds is typed after Custom ID.
        await pilot.press("tab")
        await _settle(host, pilot)
        assert screen.focused is screen.query_one("#model-search-picker-input")
        _assert_painted(screen, screen.focused)
        await _edit_custom_model(host, pilot, MODEL)
        assert screen._category_has_unsaved_changes(CATEGORY)
        assert mutations == []
        assert app.app_config["chat_defaults"]["model"] == "model-a"

        await _revert(host, pilot, discard=False)
        assert host.screen is screen
        assert screen.query_one("#settings-model-value", Input).value == MODEL
        assert (
            screen.query_one("#settings-provider-endpoint-value", Input).value
            == ENDPOINT
        )
        assert screen._category_has_unsaved_changes(CATEGORY)
        assert mutations == []

        await _revert(host, pilot, discard=True)
        assert screen.query_one("#settings-model-value", Input).value == "model-a"
        assert screen.query_one("#settings-provider-endpoint-value", Input).value == (
            "http://127.0.0.1:9099"
        )
        assert not screen._category_has_unsaved_changes(CATEGORY)
        assert mutations == []

        await _edit_custom_model(host, pilot, MODEL)
        await _edit(host, pilot, "#settings-provider-endpoint-value", ENDPOINT)
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert len(mutations) == 1
        sections, delete_keys = mutations[0]
        assert sections == {
            "chat_defaults": {"provider": "llama_cpp", "model": MODEL},
            "api_settings.llama_cpp": {
                "model": MODEL,
                "api_url": ENDPOINT,
                "credential_source": "none",
            },
            "provider_setup.confirmed": {"llama_cpp": True},
        }
        assert delete_keys == {"api_settings.llama_cpp": ("api_key", "api_key_env_var")}
        assert app.app_config["chat_defaults"] == {
            "provider": "llama_cpp",
            "model": MODEL,
            "temperature": 0.7,
            "streaming": True,
        }
        assert not screen._category_has_unsaved_changes(CATEGORY)
        assert (
            "saved"
            in str(
                screen.query_one("#settings-provider-save-result", Static).renderable
            ).lower()
        )
        await host.push_screen(Screen())
        await host.pop_screen()
        await _settle(host, pilot)
        assert screen.query_one("#settings-model-value", Input).value == MODEL
        assert (
            screen.query_one("#settings-provider-endpoint-value", Input).value
            == ENDPOINT
        )
        assert not screen._category_has_unsaved_changes(CATEGORY)
        assert len(mutations) == 1
        probe.assert_not_called()
