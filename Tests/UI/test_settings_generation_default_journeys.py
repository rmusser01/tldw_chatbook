"""Generation-default profiles remain usable through the real Settings form."""

import copy

import pytest
from textual.widgets import Collapsible, Input, Select, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_configuration_hub import (
    _capture_provider_settings_mutations,
    _open_settings_category,
)
from Tests.UI.test_settings_provider_keyboard_journeys import (
    CATEGORY,
    ProviderSettingsHarness,
    _edit,
    _revert,
    _settle,
    _tab_to,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

DISCLOSURE = "#settings-generation-defaults"
TEMPERATURE = "#settings-model-profile-temperature"


def _app(provider="openai"):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {
        "provider": provider,
        "model": "model-a",
        "temperature": 0.7,
        "streaming": True,
    }
    app.app_config["api_settings"] = {
        provider: {
            "model_defaults": {
                "model-a": {"temperature": 0.2, "max_tokens": 8192, "streaming": True},
                "model-b": {"temperature": 0.8, "top_p": 0.9},
            }
        },
        "other-provider": {"model_defaults": {"model-a": {"temperature": 0.4}}},
    }
    return app


def _painted_text(screen, widget):
    region = widget.region
    strips = list(screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(max(0, region.y), min(region.bottom, len(strips)))
    )


async def _open_generation(host, pilot):
    disclosure = host.screen.query_one(DISCLOSURE, Collapsible)
    if disclosure.collapsed:
        await _tab_to(host, pilot, f"{DISCLOSURE} CollapsibleTitle")
        await pilot.press("enter")
        await _settle(host, pilot)
    assert not disclosure.collapsed
    return disclosure


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_generation_controls_paint_labels_values_and_visible_keyboard_focus(
    provider, theme, size, monkeypatch
):
    mutations = _capture_provider_settings_mutations(monkeypatch)
    host = ProviderSettingsHarness(_app(provider), "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        disclosure = await _open_generation(host, pilot)
        controls = list(disclosure.query("Input, Select"))
        supported = [widget for widget in controls if not widget.disabled]
        seen = []
        for widget in supported:
            # Walk sequentially: hidden/unsupported fields cannot be focus stops.
            await pilot.press("tab")
            await _settle(host, pilot)
            assert host.screen.focused is widget
            seen.append(widget.id)
            _assert_painted(host.screen, widget)
            label = widget.parent.query_one(".settings-input-label", Static)
            _assert_painted(host.screen, label)
            assert str(label.renderable) in _painted_text(host.screen, label)
            if isinstance(widget, Input) and widget.value:
                assert widget.value in _painted_text(host.screen, widget)
            if isinstance(widget, Select):
                await pilot.press("enter")
                await _settle(host, pilot)
                assert widget.expanded
                await pilot.press("escape")
                await _settle(host, pilot)
                assert host.screen.focused is widget
        assert "settings-model-profile-streaming" in seen
        assert not mutations
        assert not host.screen._category_has_unsaved_changes(CATEGORY)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
async def test_generation_keyboard_validation_revert_save_clear_and_return(
    theme, size, monkeypatch
):
    app = _app()
    original = copy.deepcopy(app.app_config)
    mutations = _capture_provider_settings_mutations(monkeypatch)
    host = ProviderSettingsHarness(app, "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        await _open_generation(host, pilot)
        await _edit(host, pilot, TEMPERATURE, "2.1")
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert not mutations
        assert host.screen.query_one(TEMPERATURE, Input).value == "2.1"
        assert host.screen._category_has_unsaved_changes(CATEGORY)
        await _edit(host, pilot, TEMPERATURE, "0.3")
        await _revert(host, pilot, discard=False)
        assert host.screen.query_one(TEMPERATURE, Input).value == "0.3"
        await _revert(host, pilot, discard=True)
        assert host.screen.query_one(TEMPERATURE, Input).value == "0.2"
        await _open_generation(host, pilot)
        await _edit(host, pilot, TEMPERATURE, "0.35")
        # Rebuild while dirty: both disclosure and value remain useful on return.
        screen = host.screen
        screen.mutate_reactive(SettingsScreen.active_category)
        await _settle(host, pilot)
        assert not screen.query_one(DISCLOSURE, Collapsible).collapsed
        assert screen.query_one(TEMPERATURE, Input).value == "0.35"
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert len(mutations) == 1
        sections, deletes = mutations[0]
        assert sections == {
            "api_settings.openai": {
                "model_defaults": {
                    "model-a": {
                        "temperature": 0.35,
                        "max_tokens": 8192,
                        "streaming": True,
                    },
                    "model-b": {"temperature": 0.8, "top_p": 0.9},
                }
            }
        }
        assert deletes == {}
        assert not screen._category_has_unsaved_changes(CATEGORY)
        await _open_generation(host, pilot)
        await _edit(host, pilot, TEMPERATURE, "")
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert len(mutations) == 2
        assert (
            "temperature"
            not in app.app_config["api_settings"]["openai"]["model_defaults"]["model-a"]
        )
        assert app.app_config["chat_defaults"] == original["chat_defaults"]
        assert (
            app.app_config["api_settings"]["other-provider"]
            == original["api_settings"]["other-provider"]
        )


@pytest.mark.asyncio
async def test_generation_nonfinite_value_is_rejected_before_persistence(monkeypatch):
    mutations = _capture_provider_settings_mutations(monkeypatch)
    host = ProviderSettingsHarness(_app(), "settings")
    async with host.run_test(size=(170, 48)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        field = host.screen.query_one(TEMPERATURE, Input)
        field.value = "nan"
        await _settle(host, pilot)
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert not mutations
        assert field.value == "nan"
        assert "Temperature must be between" in host.screen._provider_save_result
        assert host.screen._category_has_unsaved_changes(CATEGORY)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_generation_resize_and_explicit_collapse_keep_state(theme, monkeypatch):
    mutations = _capture_provider_settings_mutations(monkeypatch)
    host = ProviderSettingsHarness(_app(), "settings")
    host.theme = theme
    async with host.run_test(size=(170, 48)) as pilot:
        await _open_settings_category(pilot, "#settings-category-providers-models")
        await _settle(host, pilot)
        await _open_generation(host, pilot)
        await _edit(host, pilot, TEMPERATURE, "0.35")
        for size in ((80, 24), (120, 35), (170, 48)):
            await pilot.resize_terminal(*size)
            await _settle(host, pilot)
            field = host.screen.query_one(TEMPERATURE, Input)
            assert host.screen.focused is field
            assert field.value == "0.35"
            _assert_painted(host.screen, field)
            assert "0.35" in _painted_text(host.screen, field)
        await _tab_to(host, pilot, f"{DISCLOSURE} CollapsibleTitle")
        await pilot.press("enter")
        await _settle(host, pilot)
        assert host.screen.query_one(DISCLOSURE, Collapsible).collapsed
        host.screen.mutate_reactive(SettingsScreen.active_category)
        await _settle(host, pilot)
        assert host.screen.query_one(DISCLOSURE, Collapsible).collapsed
        await _open_generation(host, pilot)
        assert host.screen.query_one(TEMPERATURE, Input).value == "0.35"
        assert not mutations
