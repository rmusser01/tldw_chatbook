"""Settings ▸ Providers & Models ▸ Default model is a searchable picker
(TASK-33007.3).

Mounted with the real application stylesheet at 211x44, the size the spec's
mockup (c) is drawn at, and driven by real keypresses. The discovery service
is a fake in all but the last test, which runs the real service on a private
profile with only the discovery transport stubbed (AC#10).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx
import pytest
import toml
from textual.widgets import Button, Input, OptionList, Select, Static

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from Tests.UI.test_destination_shells import _active_destination_screen
from Tests.UI.test_screen_navigation import _build_test_app
from Tests.UI.test_settings_category_sweep import (
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_configuration_hub import (
    FakeSettingsModelDiscoveryScope,
    _capture_provider_settings_mutations,
    _discovered_model,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness, _region_rows
from Tests.UI.test_settings_provider_key_check import _Provider, _route_discovery_to
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    MergedModelEntry,
    ModelDiscoveryResult,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId, SettingsScreen
from tldw_chatbook.Widgets.model_search_picker import CURRENT_MARK, ModelSearchPicker

_SIZE = (211, 44)
_FAKE_KEY = "sk-proj-abcdefghijklmnop1234"
PROVIDERS_MODELS = SettingsCategoryId.PROVIDERS_MODELS


class _CatalogScope(FakeSettingsModelDiscoveryScope):
    """A catalog listing ``catalog``; Discover reports ``served``."""

    def __init__(self, provider: str, *, catalog=(), served=()) -> None:
        super().__init__(
            result=ModelDiscoveryResult(
                provider=provider,
                provider_list_key=provider,
                endpoint_fingerprint="fixture",
                status="success",
                models=tuple(
                    _discovered_model(model_id, provider=provider)
                    for model_id in served
                ),
            )
        )
        self.provider = provider
        self.catalog = tuple(catalog)

    async def merge_saved_and_discovered_models(self, *, provider, **_kwargs):
        if provider_config_key(provider) != provider_config_key(self.provider):
            return ()
        return tuple(
            MergedModelEntry(
                provider=self.provider,
                provider_list_key=self.provider,
                model_id=model_id,
                display_name=model_id,
                source="persisted_discovered",
                capability_status="known",
                persisted=True,
            )
            for model_id in self.catalog
        )


def _app(provider, model, *, saved=(), catalog=(), served=()):
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": provider, "model": model}
    app.app_config["api_settings"] = {
        "openai": {"api_key": _FAKE_KEY},
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
    }
    app.providers_models = {"OpenAI": [], "llama_cpp": list(saved)}
    app.llm_provider_catalog_scope_service = _CatalogScope(
        provider, catalog=catalog, served=served
    )
    return app


async def _open_providers(host, pilot):
    await _settle_settings(pilot)
    await _click_settings_category(pilot, "providers-models")
    await pilot.pause()
    return _active_destination_screen(host)


async def _settle(host, pilot) -> None:
    await host.workers.wait_for_complete()
    await pilot.pause()


def _widgets(screen):
    picker = screen.query_one("#settings-model-picker", ModelSearchPicker)
    return (
        picker,
        picker.query_one("#model-search-picker-input", Input),
        picker.query_one("#model-search-picker-results", OptionList),
        screen.query_one("#settings-model-value", Input),
    )


def _rows(results: OptionList) -> list[str]:
    return [
        str(results.get_option_at_index(index).prompt)
        for index in range(results.option_count)
    ]


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize(
    ("provider", "model", "saved", "catalog", "served", "expected"),
    [
        pytest.param(
            "openai",
            "gpt-4o",
            (),
            ("gpt-4o", "gpt-4.1"),
            ("gpt-4.1", "o3-mini-new"),
            [
                "Served now",
                "o3-mini-new",
                "Current catalog",
                f"gpt-4o  {CURRENT_MARK}",
                "gpt-4.1",
            ],
            id="cloud-catalog",
        ),
        pytest.param(
            "llama_cpp",
            "current-b",
            ("saved-a", "current-b"),
            (),
            ("served-c",),
            [
                "Served now",
                "served-c",
                "Saved fallback",
                "saved-a",
                f"current-b  {CURRENT_MARK}",
            ],
            id="local-saved",
        ),
    ],
)
async def test_default_model_is_one_row_and_lists_ids_by_where_they_came_from(
    request, provider, model, saved, catalog, served, expected
):
    """AC#1, AC#2, AC#7: one row at rest over a hidden adapter; the open list
    groups saved, catalog and discovered ids, marks the saved default in
    words and highlights it; no ghost-text completion anywhere."""
    app = _app(provider, model, saved=saved, catalog=catalog, served=served)
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)
        row = screen.query_one("#settings-model-row")

        assert field.value == adapter.value == model
        assert row.region.height == 1
        assert field.region.height == 1
        assert not adapter.display
        assert adapter not in screen.focus_chain
        assert adapter.suggester is None and field.suggester is None

        await screen._discover_provider_models()
        await _settle(host, pilot)
        field.focus()
        await pilot.pause()

        assert results.display
        assert _rows(results) == expected, _rows(results)
        # While choosing, the list opens across the row: the Source word and
        # help step aside, so no id or mark is cut to the 36-cell column.
        assert not screen.query_one("#settings-model-source", Static).display
        assert not screen.query_one("#settings-model-help", Static).display
        assert results.region.width > 36 + 24
        current = expected.index(f"{model}  {CURRENT_MARK}")
        assert results.highlighted == current
        painted = "\n".join(_region_rows(screen, results))
        assert f"{model}  {CURRENT_MARK}" in painted, painted


@pytest.mark.asyncio
@private_profile_test
async def test_choosing_a_discovered_model_replaces_a_set_default_and_saves_with_s(
    request, monkeypatch
):
    """AC#3, AC#4, AC#7 (closes C4): one choice stages a discovered model over
    a set default, marks the category dirty, and s saves it; the saved model
    list is not touched."""
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",), served=("o3-mini-new",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)
        await screen._discover_provider_models()
        await _settle(host, pilot)
        before = {key: list(value) for key, value in app.providers_models.items()}

        field.focus()
        await pilot.pause()
        await pilot.press(*"o3-mini", "down", "enter")
        await pilot.pause()

        assert adapter.value == field.value == picker.value == "o3-mini-new"
        assert screen._provider_setting_values_mapping()["model"] == "o3-mini-new"
        assert screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        assert str(screen.query_one("#settings-model-source", Static).renderable) == (
            "edited *"
        )
        assert app.providers_models == before

        await pilot.press("escape")
        await pilot.pause()
        assert host.focused is None
        await pilot.press("s")
        await _settle(host, pilot)

        assert any(
            values.get("chat_defaults", {}).get("model") == "o3-mini-new"
            for values, _deleted in mutations
        ), mutations
        assert all("providers" not in values for values, _deleted in mutations)
        assert app.providers_models == before
        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)


@pytest.mark.asyncio
@private_profile_test
async def test_changing_provider_rescopes_the_picker_to_that_providers_default(
    request,
):
    """AC#6: the new provider's own default is staged and shown; nothing from
    the previous provider stays in the field or the list."""
    app = _app("openai", "gpt-4o", saved=("llama-local-1",), catalog=("gpt-4o",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)

        screen.query_one("#settings-provider-value", Select).value = "llama_cpp"
        await _settle(host, pilot)

        assert adapter.value == field.value == picker.value == "llama-local-1"
        assert screen._provider_setting_values_mapping()["model"] == "llama-local-1"
        field.focus()
        await pilot.pause()
        assert results.display
        listed = _rows(results)
        assert f"llama-local-1  {CURRENT_MARK}" in listed
        assert not any("gpt-4o" in row for row in listed), listed


@pytest.mark.asyncio
@private_profile_test
async def test_revert_puts_the_saved_default_back_in_the_picker(request):
    """AC#7: revert acts on the one value the adapter and the picker share."""
    app = _app("openai", "gpt-4o", catalog=("gpt-4o", "gpt-4.1"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press(*"gpt-4.1", "enter")
        await pilot.pause()
        assert adapter.value == field.value == picker.value == "gpt-4.1"
        assert screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        await pilot.press("escape", "r")
        await _settle(host, pilot)
        await pilot.click("#confirm-button")
        await _settle(host, pilot)

        assert adapter.value == field.value == picker.value == "gpt-4o"
        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)


@pytest.mark.asyncio
@private_profile_test
async def test_an_unlisted_model_id_is_entered_through_custom_id_and_validated(
    request,
):
    """AC#5: Custom ID takes an id no list holds; an id that is not bounded
    single-line text is refused and never staged."""
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, adapter = _widgets(screen)
        custom = picker.query_one("#model-search-picker-custom", Button)
        status = picker.query_one("#model-search-picker-status", Static)
        assert not custom.display or custom.region.width == 0

        field.focus()
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        assert host.focused is custom
        await pilot.press("enter")
        await pilot.pause()
        assert picker.custom_mode and host.focused is field
        await pilot.press(*"my-private-model")
        await pilot.pause()

        assert adapter.value == picker.value == "my-private-model"
        assert screen._provider_setting_values_mapping()["model"] == (
            "my-private-model"
        )
        assert screen._category_has_unsaved_changes(PROVIDERS_MODELS)

        field.value = "m" * 257
        await pilot.pause()
        assert picker.value is None
        assert adapter.value == ""
        assert "Invalid model ID" in str(status.renderable)
        assert not screen._provider_setting_values_mapping().get("model")


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("leave", ["escape", "focus-moves-away"])
async def test_an_invalid_custom_id_is_rolled_back_when_the_field_is_left(
    request, monkeypatch, leave
):
    """AC#5 (review round 1, I2): an invalid Custom ID never outlives the
    field. Leaving it puts back the model held before Custom ID, so the
    footer's "Esc, s" saves that model; it neither stages an empty model nor
    drops the earlier choice behind the invalid text."""
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _app("openai", "gpt-4o", catalog=("gpt-4o", "gpt-4.1"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press(*"gpt-4.1", "enter", "tab", "enter")
        await pilot.pause()
        assert picker.custom_mode and host.focused is field
        field.value = "m" * 257
        await pilot.pause()
        assert adapter.value == ""

        if leave == "escape":
            await pilot.press("escape")
        else:
            screen.set_focus(None)
        await pilot.pause(0.2)

        assert host.focused is None
        assert not picker.custom_mode
        assert adapter.value == field.value == picker.value == "gpt-4.1"
        assert screen._provider_setting_values_mapping()["model"] == "gpt-4.1"
        await pilot.press("s")
        await _settle(host, pilot)

        saved_models = [
            values["chat_defaults"]["model"]
            for values, _deleted in mutations
            if "model" in values.get("chat_defaults", {})
        ]
        assert saved_models == ["gpt-4.1"], mutations


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("rebuild", ["pane", "category-round-trip"])
async def test_discovered_models_stay_in_the_picker_after_the_card_rebuilds(
    request, rebuild
):
    """AC#1 (review round 1, I1): the discovery listing survives a rebuild of
    the card, and the rebuilt picker still offers it as Served now rows."""
    app = _app("llama_cpp", "current-b", saved=("current-b",), served=("served-c",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await screen._discover_provider_models()
        await _settle(host, pilot)

        if rebuild == "pane":
            screen.mutate_reactive(SettingsScreen.active_category)
        else:
            await _click_settings_category(pilot, "appearance")
            await _click_settings_category(pilot, "providers-models")
        await _settle(host, pilot)
        assert [m.model_id for m in screen._model_discovery_models] == ["served-c"]

        _picker, field, results, _adapter = _widgets(screen)
        field.focus()
        await _settle(host, pilot)
        assert {"Served now", "served-c"} <= set(_rows(results)), _rows(results)


@pytest.mark.asyncio
@private_profile_test
async def test_a_discovered_model_becomes_the_saved_default_without_joining_the_list(
    request, monkeypatch
):
    """AC#10: the real discovery service on a private profile, only its
    transport stubbed: discover, pick a model that is not the default, save;
    chat_defaults.model changes on disk and the [providers] list does not."""
    served = "gpt-test-served-only"
    listing = {"data": [{"id": "gpt-4.1-2025-04-14"}, {"id": served}]}
    provider = _Provider(httpx.Response(200, json=listing))
    _route_discovery_to(monkeypatch, provider)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    app = _build_test_app()
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-4o"}
    app.app_config["api_settings"] = {"openai": {"api_key": _FAKE_KEY}}
    providers_before = toml.loads(config_path.read_text()).get("providers", {})
    assert served not in providers_before.get("OpenAI", [])
    # The app reads [providers] at boot; discovery needs its OpenAI list.
    app.providers_models = {
        key: list(value) for key, value in providers_before.items()
    }
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        screen.query_one("#settings-discover-provider-models", Button).press()
        deadline = time.monotonic() + 8
        while not screen._model_discovery_models and time.monotonic() < deadline:
            await pilot.pause(0.05)
        await _settle(host, pilot)
        assert len(provider.requests) == 1, screen._model_discovery_status
        assert {m.model_id for m in screen._model_discovery_models} >= {served}

        picker, field, _results, adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press(*served, "enter", "escape", "s")
        deadline = time.monotonic() + 8
        while screen._category_has_unsaved_changes(PROVIDERS_MODELS):
            assert time.monotonic() < deadline, "save did not finish"
            await pilot.pause(0.05)
        await _settle(host, pilot)

    saved = toml.loads(config_path.read_text())
    assert saved["chat_defaults"]["model"] == served
    assert saved.get("providers", {}) == providers_before
    assert app.providers_models == providers_before
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@private_profile_test
async def test_slash_search_for_model_lands_on_the_picker_field(request):
    """R9: '/' search's Model field match lands on the picker's field, never
    on the hidden adapter. ("Model" alone opens the category by its title, so
    the field tier's own match is landed, as Enter does for a field query.)"""
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        field_id, label = screen._top_field_match("Model", PROVIDERS_MODELS)
        assert (field_id, label) == ("model-search-picker-input", "Model")

        screen._land_search_focus_on_field(field_id, label)
        await pilot.pause()
        assert host.focused is screen.query_one("#model-search-picker-input", Input)
