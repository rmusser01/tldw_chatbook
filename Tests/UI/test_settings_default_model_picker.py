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
from textual import events
from textual.pilot import _get_mouse_message_arguments
from textual.widgets import Button, Collapsible, Input, OptionList, Select, Static
from textual.widgets._collapsible import CollapsibleTitle
from textual.widgets.input import Selection

from Tests.private_profile import private_profile_test
from Tests.UI.test_destination_shells import _active_destination_screen, _static_text
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
from Tests.UI.test_settings_connect_rows import _until
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness
from Tests.UI.test_settings_provider_key_check import _Provider, _route_discovery_to
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.Chat.provider_setup_persistence import (
    MAX_MODEL_ID_CHARS,
    persist_provider_settings_atomic,
)
from tldw_chatbook.config import get_cli_providers_and_models
from tldw_chatbook.LLM_Provider_Catalog.model_discovery_contracts import (
    MergedModelEntry,
    ModelDiscoveryResult,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId, SettingsScreen
from tldw_chatbook.UI.Wizards.first_run_setup_state import validate_first_run_model_id
from tldw_chatbook.Widgets.model_search_picker import (
    CURRENT_MARK,
    ModelSearchPicker,
    normalize_custom_model_id,
)

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


async def _press_discover(host, pilot, screen) -> None:
    """Run Discover models as a keyboard user does: Enter on Advanced ▸ Saved
    model list's title opens it, Enter on Discover models starts the listing;
    then wait for what the listing shows."""
    disclosure = screen.query_one("#settings-advanced-saved-models", Collapsible)
    discover = screen.query_one("#settings-discover-provider-models", Button)
    if disclosure.collapsed:
        disclosure.query_one(CollapsibleTitle).focus()
        await pilot.press("enter")
        await _until(pilot, lambda: discover.region.height > 0)
    discover.focus()
    await pilot.press("enter")
    await _until(pilot, lambda: bool(screen._model_discovery_models))
    assert screen._model_discovery_models, screen._model_discovery_status
    await _settle(host, pilot)


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
            # Capture finding 9 (TASK-33007), rewritten on purpose: gpt-4.1
            # is served now, so it is listed there, not under the catalog.
            [
                "Served now",
                "gpt-4.1",
                "o3-mini-new",
                "Current catalog",
                f"gpt-4o  {CURRENT_MARK}",
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

        await _press_discover(host, pilot, screen)
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
    a set default, marks the category dirty, and s saves it.

    TASK-35011, rewritten on purpose (the owner's 2026-10-14 "Pick = use as
    default" ruling supersedes TASK-33007.3 AC#4): the pick writes nothing,
    and the s that saves it also appends it to the provider's list as the
    file holds it, in the same mutation."""
    mutations = _capture_provider_settings_mutations(monkeypatch)
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",), served=("o3-mini-new",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)
        await _press_discover(host, pilot, screen)
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
        file_list = get_cli_providers_and_models()["OpenAI"]
        await pilot.press("s")
        await _settle(host, pilot)

        (saved,) = [
            values
            for values, _deleted in mutations
            if values.get("chat_defaults", {}).get("model") == "o3-mini-new"
        ]
        assert saved["providers"] == {"OpenAI": [*file_list, "o3-mini-new"]}
        # The lists re-read after the save are the config's (this fake
        # writes no file; the save's in-memory apply is what they read).
        assert app.providers_models == get_cli_providers_and_models()
        assert not screen._category_has_unsaved_changes(PROVIDERS_MODELS)


@pytest.mark.asyncio
@private_profile_test
async def test_changing_provider_rescopes_the_picker_to_that_providers_default(
    request,
):
    """AC#6: the new provider's own default is staged and shown; nothing from
    the previous provider stays in the field or the list.

    TASK-33007.9 (review round 2): the id here is wider than the field even
    while focus widens it across the row (92 cells at 211x44). The field
    selects from the head as Provider does on the row above, so the id keeps
    its head when focused and comes to rest on it wherever the caret was left.
    """
    local = (
        "hf.co/bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF/"
        "Qwen2.5-Coder-32B-Instruct-abliterated-Q4_K_M.gguf"
    )
    app = _app("openai", "gpt-4o", saved=(local,), catalog=("gpt-4o",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)

        def paints_the_head() -> bool:
            cells = field.content_region.width
            assert len(local) > cells, cells
            return local[:cells] in _region_rows(screen, field)[0]

        screen.query_one("#settings-provider-value", Select).value = "llama_cpp"
        await _settle(host, pilot)

        assert adapter.value == field.value == picker.value == local
        assert screen._provider_setting_values_mapping()["model"] == local
        assert paints_the_head(), _region_rows(screen, field)[0]
        field.focus()
        await pilot.pause()
        assert results.display
        listed = _rows(results)
        assert f"{local}  {CURRENT_MARK}" in listed
        assert not any("gpt-4o" in row for row in listed), listed

        assert field.selection == Selection(len(local), 0)
        assert paints_the_head(), _region_rows(screen, field)[0]
        await pilot.press("end")  # the caret is the user's to move
        await pilot.pause()
        assert not paints_the_head()
        await pilot.press("escape")  # leaves the field (task-1560)
        for _ in range(100):
            if not field.has_focus:
                break
            await pilot.pause(0.05)
        await pilot.pause(0.2)  # the picker's blur timer
        assert not field.has_focus and not results.display
        assert paints_the_head(), _region_rows(screen, field)[0]


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
        # The roll-back runs on the picker's blur timer.
        await _until(
            pilot, lambda: not picker.custom_mode and adapter.value == "gpt-4.1"
        )

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
async def test_rolling_back_to_a_bracketed_model_id_names_it_literally(request):
    """A model id may hold "[" and "]". The roll-back toast names the model
    that was put back, and a toast parses Rich markup unless told not to: a
    stray "[/]" in the id raised MarkupError when the toast rendered."""
    model = "local[/]q4"
    app = _app("openai", model, catalog=(model, "gpt-4.1"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, adapter = _widgets(screen)
        field.focus()
        await pilot.pause()
        await pilot.press("tab", "enter")
        await pilot.pause()
        assert picker.custom_mode and host.focused is field
        field.value = "m" * 257
        await pilot.pause()

        await pilot.press("escape")
        # The roll-back runs on the picker's blur timer.
        await _until(pilot, lambda: not picker.custom_mode and adapter.value == model)

        assert not picker.custom_mode
        assert adapter.value == picker.value == model
        kept = f"Invalid model ID not kept; the model is still {model}."
        toast = next(n for n in host._notifications if str(n.message) == kept)
        assert not toast.markup  # the message is the same with markup on


#: TASK-35712: one more than the save writes, and a zero-width space -- a
#: character the old 256-character check let through and the save refuses.
_UNWRITABLE_IDS = ("m" * (MAX_MODEL_ID_CHARS + 1), "gpt\u200b4")
_INVALID_ID_STATUS = (
    "Invalid model ID. Use a single-line value of at most "
    f"{MAX_MODEL_ID_CHARS} characters, with no invisible or control characters."
)


@pytest.mark.parametrize(
    "model_id",
    [
        "vendor/model",
        "m" * MAX_MODEL_ID_CHARS,
        *_UNWRITABLE_IDS,
        "gpt\u00a04",  # no-break space
        "local\u202eq4",  # right-to-left override
        "a\x7fb",
        "a\u2028b",  # line separator
    ],
)
def test_custom_id_and_the_combined_save_refuse_the_same_ids(monkeypatch, model_id):
    """TASK-35712 AC#3: Custom ID held ids of up to 256 characters that the
    combined save refuses past 120 (and some characters it refuses at any
    length). One limit and one check now serve both: an id Custom ID holds
    is one the save writes, and an id it refuses is one the save refuses.
    The first-run wizard, which feeds the same save, had its own copy of the
    check (review of TASK-35712); it agrees through the same predicate."""
    writes = _capture_provider_settings_mutations(monkeypatch)
    held = normalize_custom_model_id(model_id) == model_id
    try:
        wizard_held = validate_first_run_model_id(model_id) == model_id
    except ValueError:
        wizard_held = False
    assert wizard_held == held, model_id
    try:
        persist_provider_settings_atomic(
            None,
            provider="openai",
            model=model_id,
            # The one section a save may write with no connection change.
            section_values={"api_settings.openai": {"model_defaults": {}}},
            delete_keys={},
        )
    except ValueError as exc:
        assert str(exc) == "Provider settings model is invalid."
        assert not held, model_id
    else:
        assert held and writes, model_id


@pytest.mark.asyncio
@private_profile_test
async def test_a_custom_id_the_save_cannot_write_is_refused_as_it_is_typed(request):
    """TASK-35712 AC#1: an id the save cannot write is refused in the Custom
    ID field, with the save's limit named, before s is pressed; the longest
    id it writes is still taken."""
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, _results, adapter = _widgets(screen)
        status = picker.query_one("#model-search-picker-status", Static)
        field.focus()
        await pilot.pause()
        await pilot.press("tab", "enter")
        await pilot.pause()
        assert picker.custom_mode and host.focused is field

        longest = "m" * MAX_MODEL_ID_CHARS
        field.value = longest
        await pilot.pause()
        assert adapter.value == picker.value == longest
        assert screen._provider_setting_values_mapping()["model"] == longest

        for model_id in _UNWRITABLE_IDS:
            field.value = model_id
            await pilot.pause()
            assert picker.value is None, model_id
            assert adapter.value == "", model_id
            assert str(status.renderable) == _INVALID_ID_STATUS, model_id
            assert not screen._provider_setting_values_mapping().get("model")


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("route", ["staged-model", "saved-model"])
async def test_an_id_the_save_refuses_is_named_as_the_cause_not_config_access(
    request, monkeypatch, route
):
    """TASK-35712 AC#2: an id the save cannot write can still reach s: one
    written straight to the form's model field, as a hand-edited draft
    would be, or one already in config.toml when another field is saved.
    s blamed the endpoint ("review the endpoint and model") or config access
    ("the file was not written ... check config access and retry"); the
    save refused the id before touching the file, so the message says so."""
    unwritable = _UNWRITABLE_IDS[0]
    mutations = _capture_provider_settings_mutations(monkeypatch)
    saved = unwritable if route == "saved-model" else "gpt-4o"
    app = _app("openai", saved, catalog=("gpt-4o",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        _picker, _field, _results, adapter = _widgets(screen)
        if route == "staged-model":
            adapter.value = unwritable
        else:
            temperature = "#settings-model-profile-temperature"
            screen.query_one(temperature, Input).value = "0.42"
        await pilot.pause()
        assert adapter.value == unwritable
        assert screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        screen.set_focus(None)
        await pilot.press("s")
        await _settle(host, pilot)

        result = screen._provider_save_result
        assert result == (
            "Provider settings not saved: the model ID is invalid. Use a "
            f"single-line value of at most {MAX_MODEL_ID_CHARS} characters, "
            "with no invisible or control characters. Draft retained."
        )
        assert result in [str(n.message) for n in host._notifications]
        assert mutations == []
        assert PROVIDERS_MODELS in screen._settings_drafts


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
        await _press_discover(host, pilot, screen)

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
async def test_a_discovered_model_becomes_the_saved_default_and_joins_the_list(
    request, monkeypatch
):
    """AC#10: the real discovery service on a private profile, only its
    transport stubbed: discover, pick a model that is not the default, save;
    chat_defaults.model changes on disk.

    TASK-35011, rewritten on purpose (the 2026-10-14 ruling supersedes AC#10's
    "[providers] list did not change"): the same save appends the pick to
    the OpenAI list, and the app's lists are the file's."""
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
    app.providers_models = {key: list(value) for key, value in providers_before.items()}
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
    assert saved["providers"] == {
        **providers_before,
        "OpenAI": [*providers_before["OpenAI"], served],
    }
    assert app.providers_models == saved["providers"]
    assert len(provider.requests) == 1


@pytest.mark.asyncio
@private_profile_test
async def test_the_search_index_model_entry_lands_on_the_picker_field(request):
    """R9: the search index's Model entry is the picker's field, never the
    hidden adapter, and landing on it focuses that field. The landing is
    called directly because no '/' query reaches it: every text that matches
    the label "Model" also matches the category's own title, "Providers &
    Models", and a title match opens the category without landing on a field
    (TASK-23109)."""
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        field_id, label = screen._top_field_match("Model", PROVIDERS_MODELS)
        assert (field_id, label) == ("model-search-picker-input", "Model")
        summary = screen._category_summary_by_id(PROVIDERS_MODELS)
        assert not screen._search_match_is_field_landing("Model", summary)

        screen._land_search_focus_on_field(field_id, label)
        await pilot.pause()
        assert host.focused is screen.query_one("#model-search-picker-input", Input)


@pytest.mark.asyncio
@private_profile_test
async def test_a_typed_filter_counts_matches_and_enter_picks_the_first(request):
    """Capture 02 (TASK-33007 fix 8): with a filter typed the status still
    read "Showing 14 configured models" and no row was highlighted. It says
    how many match, the first is highlighted under its group heading, and
    Enter picks it."""
    app = _app(
        "openai",
        "gpt-4o",
        catalog=("gpt-4o", "o4-mini-2025-04-16", "o3-mini-2025-01-31", "gpt-4.1"),
    )
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)
        status = picker.query_one("#model-search-picker-status", Static)

        field.focus()
        await pilot.pause()
        await pilot.press(*"mini")
        await pilot.pause()

        assert _rows(results) == [
            "Current catalog",
            "o4-mini-2025-04-16",
            "o3-mini-2025-01-31",
        ]
        assert str(status.renderable) == "2 found · Enter picks · Esc cancels"
        assert results.highlighted == 1
        painted = "\n".join(_region_rows(screen, picker))
        assert "2 found · Enter picks · Esc cancels" in painted, painted

        await pilot.press("enter")
        await pilot.pause()
        assert adapter.value == picker.value == "o4-mini-2025-04-16"
        assert str(screen.query_one("#settings-model-source", Static).renderable) == (
            "edited *"
        )


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("held", [False, True])
async def test_a_click_on_a_listed_model_picks_it(request, held):
    """Capture fix 2, finding 3: with the results list unfocusable (finding
    24's fix), a press on a row focused the detail pane instead, the field's
    blur collapsed the picker and hid the list, and nothing was picked (a
    fast click opened an unrelated Select under the shifted rows). A press
    puts focus in the list and keeps it open, and the click picks the row."""
    app = _app("openai", "gpt-4o", catalog=("gpt-4o",), served=("o3-mini-new",))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker, field, results, adapter = _widgets(screen)
        await _press_discover(host, pilot, screen)
        field.focus()
        await pilot.pause()
        row = (3, results.gutter.top + _rows(results).index("o3-mini-new"))
        if held:  # an ordinary press lasts 80-150 ms
            before = results.region
            await pilot.mouse_down(results, offset=row)
            await pilot.pause(0.12)
            assert host.focused is results
            assert results.display and results.region == before
            return
        await pilot.click(results, offset=row)
        await pilot.pause(0.2)
        assert adapter.value == field.value == picker.value == "o3-mini-new"


async def _open_list(pilot, screen, which: str) -> OptionList:
    """Open the Default model or the Provider list as a keyboard user does."""
    if which == "model":
        _picker, field, results, _adapter = _widgets(screen)
        field.focus()
    else:
        screen.query_one("#settings-provider-search", Input).focus()
        await pilot.pause()
        await pilot.press("down")
        results = screen.query_one("#settings-provider-picker", OptionList)
    await _until(pilot, lambda: results.display and results.region.height > 1)
    return results


@pytest.mark.asyncio
@private_profile_test
@pytest.mark.parametrize("which", ["model", "provider"])
@pytest.mark.parametrize("held", [False, True], ids=["released-at-once", "held"])
async def test_a_click_below_an_open_list_lands_on_the_pressed_control_or_none(
    request, which, held
):
    """TASK-33003.12: a press on a control below an open picker list moves
    focus there, the picker's blur collapses the list, and the rows below it
    move up. The click must activate the control that was pressed, or none,
    never the one the collapse moved under the pointer; no Escape first.

    The press and the release are posted to the app as a terminal driver
    posts them: the app makes a Click only when both land on one widget.
    ``Pilot.click`` instead forwards its Click to whatever the offset holds
    after its pauses, so after the collapse it opened the Collapsible that
    moved there: that is the shape this test refuses (checked by swapping
    the two posts for ``pilot.click``). A press held past the collapse
    (an ordinary press lasts 80-150 ms) activates nothing."""
    app = _app("openai", "gpt-4o", catalog=tuple(f"gpt-{n}" for n in range(12)))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        results = await _open_list(pilot, screen, which)
        pressed = screen.query_one("#settings-generation-defaults", Collapsible)
        title = pressed.query_one(CollapsibleTitle)
        point = (title.region.x + 30, title.region.y)
        before = {item.id: item.collapsed for item in screen.query(Collapsible)}
        arguments = _get_mouse_message_arguments(screen, point, button=1)

        host.post_message(events.MouseDown(**arguments))
        if held:
            await _until(pilot, lambda: not results.display)
            await pilot.pause()
            # The collapse reached the screen: another control is under the
            # pointer now, so this release is the case the rider is about.
            assert screen.get_widget_at(*point)[0] is not title
        host.post_message(events.MouseUp(**arguments))
        await _until(pilot, lambda: not results.display)
        await pilot.pause(0.1)

        toggled = {
            item.id
            for item in screen.query(Collapsible)
            if before.get(item.id) != item.collapsed
        }
        assert host.focused is title
        assert toggled <= {pressed.id}, toggled
        if held:
            assert toggled == set()


@pytest.mark.asyncio
@private_profile_test
async def test_koboldcpp_root_and_prefix_root_list_where_switch_model_probes(
    request, monkeypatch
):
    """TASK-36380 AC#1: the real discovery service on a private profile,
    only its transport stubbed. Discover models for a saved koboldcpp root,
    then for a prefix root typed into Endpoint, lists from the /v1/models
    Switch model probes. AC#2: for an endpoint the contract refuses, the line
    beside Discover names the contract's reason, and points at the native
    /api/v1/generate route koboldcpp's sends need."""
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        ConnectionProbeAvailability,
        connection_probe_availability,
        resolve_provider_endpoint,
    )
    from tldw_chatbook.config import save_settings_to_cli_config

    root, prefix = "http://localhost:5001", "http://localhost:5001/kobold"
    ambiguous = "http://localhost:5001/v1/api/v1/generate"
    for endpoint in (root, prefix):  # Switch model probes both
        assert connection_probe_availability("koboldcpp", endpoint) is (
            ConnectionProbeAvailability.MODELS_ROUTE
        )
    assert save_settings_to_cli_config(
        {
            "chat_defaults": {"provider": "koboldcpp", "model": ""},
            "api_settings.koboldcpp": {"api_url": root},
        }
    )
    app = _build_test_app()
    app._init_providers_models()  # [providers] read from the file, as boot does
    served = _Provider(httpx.Response(200, json={"data": [{"id": "kobold-served"}]}))
    _route_discovery_to(monkeypatch, served)
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        assert screen._provider_widget_value() == "koboldcpp"
        await _press_discover(host, pilot, screen)
        field = screen.query_one("#settings-provider-endpoint-value", Input)
        field.value = prefix
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        field.value = ambiguous
        await _settle(host, pilot)
        screen.query_one("#settings-discover-provider-models", Button).focus()
        await pilot.press("enter")
        await _until(
            pilot, lambda: screen._model_discovery_status.startswith("Endpoint path")
        )
        await _settle(host, pilot)
        status = screen.query_one("#settings-model-discovery-status", Static)
        shown = _static_text(status) if status.display else ""

    probed = [
        resolve_provider_endpoint("koboldcpp", e).models_url for e in (root, prefix)
    ]
    assert probed == [
        "http://localhost:5001/v1/models",
        "http://localhost:5001/kobold/v1/models",
    ]
    assert [str(sent.url) for sent in served.requests] == probed
    assert shown == (
        "Endpoint path has ambiguous API suffixes. Use the server's "
        "/api/v1/generate route, such as http://localhost:5001/api/v1/generate."
    ), shown
