"""A discovered model chosen as the default joins the saved list on s (TASK-35011).

The owner's 2026-10-14 "Pick = use as default" ruling: a model Discover
models listed, chosen in Default model, is staged as the default, and the s
that saves it also adds it to the provider's saved model list. Save selected
becomes Add to model list, marked as applying immediately (A2), after the list
in Tab order (B6), with Inspector rows (B19) and receipts that say what
happened (C4); the capability line no longer ties capabilities to saving
(C-M2). Every case runs on a private profile, at 211x44 with the real
stylesheet, with the real discovery service (only its transport stubbed)
unless it says otherwise, and the real writer. Configuration is written to
the file through the production writer, never by seeding
``app.providers_models``.
"""

from __future__ import annotations

import asyncio
import tomllib

import pytest
from textual.widgets import Button, Collapsible, Input, SelectionList, Static
from textual.widgets._collapsible import CollapsibleTitle
from textual.widgets._toast import Toast

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_settings_default_model_picker import (
    _CatalogScope,
    _open_providers,
    _press_discover,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _region_rows, _SettingsCssHarness
from Tests.UI.test_settings_provider_key_check import _route_discovery_to
from Tests.UI.test_settings_saved_lists_follow_file import (
    _booted_app,
    _config_path,
    _listing,
    _saved,
    _until,
    _write,
)
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

pytestmark = pytest.mark.asyncio

_SIZE = (211, 44)
_FAKE_KEY = "sk-proj-abcdefghijklmnop1234"
PROVIDERS_MODELS = SettingsCategoryId.PROVIDERS_MODELS
_ADD = "#settings-save-discovered-provider-models"
_DISCOVER = "#settings-discover-provider-models"
_LIST = "#settings-discovered-models-list"
_CLEAR = "#settings-clear-discovered-provider-models"
_INSTANT = "applies immediately · no Save needed"
_LIST_CHANGED = (
    "The saved model list changed while saving; nothing was written. Press s again."
)
_CAPABILITY = "Capabilities unknown for models no catalog lists; text chat is assumed."
_SAVES_NOTHING = "saves nothing · only Add to model list writes the list"

#: Case 1's providers: a local server, a cloud provider, and Custom 2, whose
#: list key (Custom_2) is not an alias the setup owner knows.
_LLAMA = {
    "chat_defaults": {"provider": "llama_cpp", "model": "m-one"},
    "providers": {"Llama_cpp": ["None", "m-one"]},
}
_OPENAI = {
    "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
    "api_settings.openai": {"api_key": _FAKE_KEY},
}
_CUSTOM_2 = {
    "chat_defaults": {"provider": "custom_2", "model": "c-one"},
    "providers": {"Custom_2": ["c-one"]},
}


def _file() -> dict:
    return tomllib.loads(_config_path().read_text())


def _title(screen) -> str:
    return str(screen.query_one("#settings-advanced-saved-models", Collapsible).title)


def _status(screen) -> str:
    return str(screen.query_one("#settings-model-discovery-status", Static).renderable)


def _saved_toast(host):
    """The s save's toast (other toasts may follow it)."""
    return next(
        toast
        for toast in reversed(list(host._notifications))
        if toast.message.startswith("Provider and model settings saved: ")
    )


async def _pick(host, pilot, screen, model_id: str) -> None:
    """Choose ``model_id`` in Default model: its exact id, then Enter."""
    picker, field, _results, adapter = _widgets(screen)
    field.focus()
    await pilot.pause()
    field.value = model_id
    await pilot.pause()
    await pilot.press("enter")
    await _until(pilot, lambda: picker.value == model_id == adapter.value)
    screen.set_focus(None)
    await _settle(host, pilot)


async def _custom_id(host, pilot, screen, model_id: str) -> None:
    """Type ``model_id`` as a Custom ID, then leave the field."""
    picker, field, _results, adapter = _widgets(screen)
    field.focus()
    await pilot.pause()
    await pilot.press("tab", "enter")
    await _until(pilot, lambda: picker.custom_mode and screen.focused is field)
    field.value = model_id
    await _until(pilot, lambda: adapter.value == model_id)
    screen.set_focus(None)
    await _settle(host, pilot)


async def _save_with_s(host, pilot, screen) -> None:
    await pilot.press("s")
    await _until(pilot, lambda: screen._provider_save_result)
    await _settle(host, pilot)


async def _add(host, pilot, screen, *model_ids: str) -> None:
    """Check ``model_ids`` and press Add to model list; wait for its line."""
    listing = screen.query_one(_LIST, SelectionList)
    listing.deselect_all()
    for model_id in model_ids:
        listing.select(model_id)
    await pilot.pause()
    screen.query_one(_ADD, Button).press()
    await _until(pilot, lambda: not _status(screen).startswith("Saving"))
    await _settle(host, pilot)


# --- Case 1: a pick plus s adds the model ----------------------------------


@private_profile_test
@pytest.mark.parametrize(
    ("provider", "list_key", "profile"),
    (
        pytest.param("llama_cpp", "Llama_cpp", _LLAMA, id="llama-cpp"),
        pytest.param("openai", "OpenAI", _OPENAI, id="openai"),
        pytest.param("custom_2", "Custom_2", _CUSTOM_2, id="custom-2"),
    ),
)
async def test_a_discovered_pick_saved_with_s_joins_the_saved_list(
    request, monkeypatch, provider, list_key, profile
):
    """AC#1, AC#6: a model the current listing holds and the saved list does
    not, chosen in Default model, is the default for new chats after s, and
    the same write appends it to the provider's list as the file holds it
    then (a model another program added since start is kept). The toast
    names it; the row reads "saved" and the title no longer counts it."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write(profile)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-alpha", "fake-beta"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        await _pick(host, pilot, screen, "fake-beta")
        _write({"providers": {list_key: [*_saved(list_key), "hand-added"]}})
        before = _saved(list_key)
        await _save_with_s(host, pilot, screen)
        label = screen._provider_display_label(provider)
        toast = _saved_toast(host)
        listing = screen.query_one(_LIST, SelectionList)
        rows = [
            str(listing.get_option_at_index(i).prompt).split(" · ")[1:3]
            for i in range(listing.option_count)
        ]
        title = _title(screen)
        unsaved = screen._category_has_unsaved_changes(PROVIDERS_MODELS)

    data = _file()
    assert data["providers"][list_key] == [*before, "fake-beta"]
    assert data["chat_defaults"]["model"] == "fake-beta"
    assert app.providers_models[list_key] == data["providers"][list_key]
    assert toast.message.endswith(f" Added fake-beta to {label}'s model list.")
    assert toast.markup is False
    assert ["fake-beta", "saved"] in rows, rows
    assert "joins on save" not in title, title
    assert not unsaved


# --- Case 2: nothing is written before s -------------------------------------


@private_profile_test
@pytest.mark.parametrize("revert_with", ("r", "button"))
async def test_a_pick_reverted_before_s_leaves_the_file_byte_identical(
    request, monkeypatch, revert_with
):
    """AC#2: a pick writes nothing; Revert's Discard changes drops the pick
    with the draft, and the s after it has nothing to save or add."""
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-beta"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        before = _config_path().read_bytes()
        await _pick(host, pilot, screen, "fake-beta")
        picked = _config_path().read_bytes()
        if revert_with == "r":
            await pilot.press("r")
        else:
            await pilot.click("#settings-revert-category")
        await _until(pilot, lambda: isinstance(host.screen, ConfirmationDialog))
        await pilot.click("#confirm-button")
        await _until(
            pilot, lambda: not screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        )
        title = _title(screen)
        await pilot.press("s")
        await _settle(host, pilot)

    assert picked == before
    assert "joins on save" not in title, title
    assert _config_path().read_bytes() == before


# --- Cases 2 and 3 on the whole app: quit-Discard, and Keep editing --------

_SETTINGS_QUIT_TITLE = "Quit and discard unsaved Settings changes?"


async def _whole_app_pick(app, pilot, before: list | None = None) -> object:
    """Settings ▸ Providers & Models, Discover, and choose fake-beta.

    ``before`` receives the config file's bytes just before the pick.
    """
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    await _until(pilot, lambda: isinstance(app.screen, SettingsScreen), timeout=15)
    screen = app.screen
    screen.active_category = PROVIDERS_MODELS.value
    await _until(pilot, lambda: bool(screen.query(_DISCOVER)), timeout=15)
    screen.query_one("#settings-advanced-saved-models", Collapsible).collapsed = False
    discover = screen.query_one(_DISCOVER, Button)
    await _until(pilot, lambda: not discover.disabled and discover.region.height > 0)
    discover.press()
    await _until(pilot, lambda: bool(screen._model_discovery_models), timeout=15)
    _picker, field, _results, adapter = _widgets(screen)
    if before is not None:
        before.append(_config_path().read_bytes())
    field.focus()
    await pilot.pause()
    field.value = "fake-beta"
    await pilot.pause()
    await pilot.press("enter")
    await _until(pilot, lambda: adapter.value == "fake-beta")
    screen.set_focus(None)
    await pilot.pause()
    return screen


def _whole_app(monkeypatch):
    _write(_LLAMA)
    _route_discovery_to(monkeypatch, _listing("fake-beta"))
    app = _build_test_app(configured_default="settings")
    app._init_providers_models()  # [providers] read from the file, as boot does
    cleanups: list[bool] = []

    async def cleanup() -> None:  # stands in for the irreversible shutdown
        cleanups.append(True)
        app.exit()

    monkeypatch.setattr(app, "_run_approved_quit_cleanup", cleanup)
    return app, cleanups


def _quit_prompts(app) -> list:
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    return [
        screen
        for screen in app.screen_stack
        if isinstance(screen, ConfirmationDialog)
        and screen.title == _SETTINGS_QUIT_TITLE
    ]


@private_profile_test
async def test_a_pick_left_by_quit_and_discard_leaves_the_file_byte_identical(
    request, monkeypatch
):
    """AC#2: Ctrl+Q, Discard and quit, after a pick: the file is unchanged."""
    app, cleanups = _whole_app(monkeypatch)
    before: list[bytes] = []
    async with app.run_test(size=_SIZE) as pilot:
        await _whole_app_pick(app, pilot, before)
        picked = _config_path().read_bytes()
        await pilot.press("ctrl+q")
        await _until(pilot, lambda: bool(_quit_prompts(app)), timeout=10)
        await pilot.click("#confirm-button")
        async with asyncio.timeout(10):
            while not cleanups:
                await asyncio.sleep(0.02)
    assert picked == before[0]
    assert _config_path().read_bytes() == before[0]


@private_profile_test
async def test_a_pick_kept_through_leaving_settings_is_added_by_the_next_s(
    request, monkeypatch
):
    """AC#3: pick, leave for Console, Ctrl+Q, Keep editing: back in a fresh
    Settings (first paint focuses the nav bar, so s, not Enter), s saves the
    default and still adds the model. The pending addition rides in
    save_state, as view state the retained-editor check knows."""
    from tldw_chatbook.Backup_Recovery.unsaved_editors import _settings_snapshot_dirty
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    app, cleanups = _whole_app(monkeypatch)
    async with app.run_test(size=_SIZE) as pilot:
        first = await _whole_app_pick(app, pilot)
        await pilot.pause()
        assert _title(first).endswith(" · 1 joins on save"), _title(first)
        state = first.save_state()
        app.post_message(NavigateToScreen("chat"))
        await _until(pilot, lambda: isinstance(app.screen, ChatScreen), timeout=15)
        await pilot.press("ctrl+q")
        await _until(pilot, lambda: bool(_quit_prompts(app)), timeout=10)
        await pilot.press("escape")  # Keep editing
        await _until(
            pilot,
            lambda: isinstance(app.screen, SettingsScreen) and app.screen is not first,
            timeout=15,
        )
        await _until(pilot, lambda: app._quit_in_progress is False)
        back = app.screen
        await _until(pilot, lambda: bool(back.query("#settings-model-value")))
        await pilot.pause()
        await pilot.press("s")
        await _until(pilot, lambda: _saved("Llama_cpp")[-1:] == ["fake-beta"], 10)

    assert state.get("discovered_pick") == ("llama_cpp", "fake-beta")
    assert _settings_snapshot_dirty(state) is True
    fresh = SettingsScreen(app)
    fresh.restore_state(state)
    assert getattr(fresh, "_discovered_pick", None) == ("llama_cpp", "fake-beta")
    assert cleanups == []
    assert _saved("Llama_cpp") == ["None", "m-one", "fake-beta"]
    assert _file()["chat_defaults"]["model"] == "fake-beta"


# --- Case 4: only a discovered pick joins ------------------------------------


@private_profile_test
@pytest.mark.parametrize(
    ("steps", "default"),
    (
        pytest.param((("pick", "gpt-saved"),), "gpt-saved", id="already-saved"),
        pytest.param((("pick", "gpt-catalog"),), "gpt-catalog", id="catalog"),
        pytest.param((("custom", "my-own-id"),), "my-own-id", id="custom-id"),
        pytest.param(
            (("pick", "gpt-new"), ("custom", "my-own-id")),
            "my-own-id",
            id="listed-then-custom-id",
        ),
        pytest.param(
            (("pick", "gpt-new"), ("pick", "gpt-catalog")),
            "gpt-catalog",
            id="listed-then-catalog",
        ),
    ),
)
async def test_a_model_staged_any_other_way_leaves_the_saved_list_unchanged(
    request, monkeypatch, steps, default
):
    """AC#5: an id the saved list already holds, a catalog id the server did
    not list, or a typed Custom ID (also one that replaces a discovered
    pick, as does a later catalog pick) is saved as the default only. The
    catalog comes from a fake scope; the saved list is the file's."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _write({**_OPENAI, "providers": {"OpenAI": ["gpt-4o", "gpt-saved"]}})
    app = _booted_app()
    app.llm_provider_catalog_scope_service = _CatalogScope(
        "openai", catalog=("gpt-4o", "gpt-catalog"), served=("gpt-saved", "gpt-new")
    )
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        for how, model_id in steps:
            if how == "pick":
                await _pick(host, pilot, screen, model_id)
            else:
                await _custom_id(host, pilot, screen, model_id)
        title = _title(screen)
        await _save_with_s(host, pilot, screen)
        toast = _saved_toast(host).message

    data = _file()
    assert data["chat_defaults"]["model"] == default
    assert data["providers"]["OpenAI"] == ["gpt-4o", "gpt-saved"]
    assert "joins on save" not in title, title
    assert "model list" not in toast, toast


@private_profile_test
async def test_a_pick_from_a_listing_the_endpoint_edit_dropped_joins_nothing(
    request, monkeypatch
):
    """AC#5: an endpoint edit drops the listing the pick came from, and the
    pick with it, while the draft still stages that model: s saves the
    endpoint and the default and leaves the saved list unchanged. A Discover
    since the pick keeps it (the reset's other callers do)."""
    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-beta"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        await _pick(host, pilot, screen, "fake-beta")
        await _press_discover(host, pilot, screen)  # a fresh listing keeps it
        kept = _title(screen)
        endpoint = screen.query_one("#settings-provider-endpoint-value", Input)
        endpoint.value = "http://127.0.0.1:9199"
        await _settle(host, pilot)
        staged = screen._provider_setting_values_mapping()["model"]
        dropped = _title(screen)
        await _save_with_s(host, pilot, screen)

    assert kept.endswith(" · 1 joins on save"), kept
    assert staged == "fake-beta"
    assert "joins on save" not in dropped, dropped
    assert _saved("Llama_cpp") == ["None", "m-one"]
    assert _file()["chat_defaults"]["model"] == "fake-beta"


# --- Case 5: the list key no longer resolves ---------------------------------


@private_profile_test
async def test_a_list_key_that_no_longer_resolves_still_saves_the_default(
    request, monkeypatch
):
    """AC#8: after the pick another program adds a second llama.cpp list
    (``llama_cpp`` beside the template's ``Llama_cpp``), so no single list
    is the provider's. s saves the default anyway, writes neither list, and
    the toast says why the model was not added."""
    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-beta"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        await _pick(host, pilot, screen, "fake-beta")
        _write({"providers": {"llama_cpp": ["other"]}})
        await _save_with_s(host, pilot, screen)
        toast = _saved_toast(host)

    data = _file()
    assert data["chat_defaults"]["model"] == "fake-beta"
    assert data["providers"]["Llama_cpp"] == ["None", "m-one"]
    assert data["providers"]["llama_cpp"] == ["other"]
    assert toast.message.endswith(
        " fake-beta was not added to its model list: more than one [providers] "
        "list matches llama.cpp."
    ), toast.message
    assert toast.markup is False


# --- Case 6: the list moved between the read and the locked write ------------


@private_profile_test
async def test_a_list_changed_while_s_saves_writes_nothing_and_keeps_the_draft(
    request, monkeypatch
):
    """AC#7: s reads the list, then writes under the lock only if the locked
    list is still that one. Only the read before the lock is made stale; the
    locked check refuses the write, nothing is written, the draft and the
    pick stay, and the line says to press s again. The check reads nothing
    itself: a config read inside the lock is refused (TASK-33641.6), which
    would read as a failed write, not this conflict."""
    from tldw_chatbook import config

    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-beta"))
    host = _SettingsCssHarness(app, "settings")
    real = config.get_atomic_config_snapshot
    reads: list[int] = []

    def stale():
        snapshot = real()
        reads.append(1)
        snapshot.values["providers"]["Llama_cpp"] = ["None"]
        return snapshot

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        await _pick(host, pilot, screen, "fake-beta")
        before = _config_path().read_bytes()
        monkeypatch.setattr(config, "get_atomic_config_snapshot", stale)
        await _save_with_s(host, pilot, screen)
        result = screen._provider_save_result
        unsaved = screen._category_has_unsaved_changes(PROVIDERS_MODELS)
        title = _title(screen)

    assert result == _LIST_CHANGED, result
    assert _config_path().read_bytes() == before
    assert unsaved
    assert title.endswith(" · 1 joins on save"), title
    assert reads == [1]


# --- Case 7: the title says one joins on save --------------------------------


@private_profile_test
@pytest.mark.parametrize("size", ((211, 44), (235, 52)), ids=("211x44", "235x52"))
async def test_the_saved_model_list_title_says_a_pick_joins_on_save(
    request, monkeypatch, size
):
    """AC#4: while a discovered pick waits for s the title ends "· 1 joins on
    save", on its one row; a saved id picked instead clears it, and s does."""
    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-alpha", "fake-beta", "m-one"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        listing = screen.query_one(_LIST, SelectionList)
        listing.select("fake-alpha")
        listing.select("fake-beta")
        await _settle(host, pilot)
        await _pick(host, pilot, screen, "fake-beta")
        picked = _title(screen)
        disclosure = screen.query_one("#settings-advanced-saved-models", Collapsible)
        heading = disclosure.query_one(CollapsibleTitle)
        heading.scroll_visible(animate=False)
        await pilot.pause()
        height = heading.region.height
        painted = _region_rows(screen, heading)
        await _pick(host, pilot, screen, "m-one")
        saved_id = _title(screen)
        await _pick(host, pilot, screen, "fake-beta")
        await _save_with_s(host, pilot, screen)
        after = _title(screen)

    assert picked == (
        "Saved model list · 1 saved in config · 3 discovered, 2 not saved · "
        "2 selected · 1 joins on save"
    ), picked
    assert height == 1
    assert painted[0].rstrip().endswith("1 joins on save"), painted
    assert "joins on save" not in saved_id, saved_id
    assert after.startswith("Saved model list · 2 saved in config"), after
    assert "joins on save" not in after, after


# --- Case 8: Add to model list ----------------------------------------------


@private_profile_test
@pytest.mark.parametrize("size", ((211, 44), (235, 52)), ids=("211x44", "235x52"))
async def test_add_to_model_list_is_labelled_and_marked_as_applying_now(
    request, monkeypatch, size
):
    """AC#9: Save selected reads Add to model list, beside Clear and the
    instant-apply hint on one row, under the list it acts on."""
    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-beta"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=size) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        add = screen.query_one(_ADD, Button)
        clear = screen.query_one(_CLEAR, Button)
        listing = screen.query_one(_LIST, SelectionList)
        discover = screen.query_one(_DISCOVER, Button)
        row = add.parent
        row.scroll_visible(animate=False)
        await pilot.pause()
        hints = [
            str(hint.renderable)
            for hint in row.query(".settings-instant-apply-hint").results(Static)
        ]
        painted = _region_rows(screen, row)
        label, tooltip = str(add.label), str(add.tooltip)
        same_row = clear.parent is row
        height = row.region.height
        ys = (discover.virtual_region.y, listing.virtual_region.y, row.virtual_region.y)

    assert label == "Add to model list"
    assert "no s needed" in tooltip, tooltip
    assert same_row and hints == [_INSTANT]
    assert height == 1
    assert "Add to model list" in painted[0] and _INSTANT in painted[0], painted
    assert ys[0] < ys[1] < ys[2], ys


@private_profile_test
@pytest.mark.parametrize(
    ("profile", "listed", "checked", "receipt"),
    (
        pytest.param(
            _LLAMA,
            ("fake-alpha", "fake-beta"),
            ("fake-alpha", "fake-beta"),
            "Added fake-alpha and fake-beta to llama.cpp's model list.",
            id="added",
        ),
        pytest.param(
            _LLAMA,
            ("m-one",),
            ("m-one",),
            "The selected models are already in llama.cpp's model list. To make "
            "one the default, choose it in Default model.",
            id="already-listed",
        ),
        pytest.param(
            {
                "chat_defaults": {"provider": "llama_cpp", "model": ""},
                "providers": {"Llama_cpp": ["None"]},
            },
            ("fake-beta",),
            ("fake-beta",),
            "Added fake-beta to llama.cpp's model list. Default model is now "
            "fake-beta; press s to save it.",
            id="filled-empty-default",
        ),
    ),
)
async def test_add_to_model_list_names_what_it_added_or_how_to_choose_a_default(
    request, monkeypatch, profile, listed, checked, receipt
):
    """AC#9, AC#10 (C4 residual): the receipt names the models added, or says
    they were already listed and that choosing one in Default model makes it
    the default; when it fills an empty Default model, it says s still has
    to save that default (and the file's default is untouched)."""
    _write(profile)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing(*listed))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        await _add(host, pilot, screen, *checked)
        line = _status(screen)

    assert line == receipt
    assert _file()["chat_defaults"]["model"] == profile["chat_defaults"]["model"]


@private_profile_test
async def test_the_s_receipt_prints_a_bracketed_model_id_literally(
    request, monkeypatch
):
    """AC#1, TASK-1513: the s toast names the model it added; an id holding
    "[/]" renders literally (markup=False), never as markup."""
    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake[/]beta"))
    host = _SettingsCssHarness(app, "settings")
    expected = " Added fake[/]beta to llama.cpp's model list."

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        await _pick(host, pilot, screen, "fake[/]beta")
        await _save_with_s(host, pilot, screen)
        toast = _saved_toast(host)
        assert toast.message.endswith(expected), toast.message
        # Textual's own toast render: with markup on, "[/]" raises MarkupError.
        painted = Toast(toast).render().plain

    assert toast.markup is False
    assert painted == toast.message
    assert _saved("Llama_cpp") == ["None", "m-one", "fake[/]beta"]


# --- Cases 9 and 10: Tab order and the Inspector -----------------------------


@private_profile_test
async def test_tab_from_discover_reaches_the_list_before_add_and_clear(
    request, monkeypatch
):
    """AC#11 (B6): Discover models, then the discovered list, then Add to
    model list, then Clear."""
    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-beta"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        screen.query_one(_DISCOVER, Button).focus()
        await pilot.pause()
        stops = [screen.focused.id]
        for _ in range(3):
            await pilot.press("tab")
            await pilot.pause()
            stops.append(screen.focused.id)

    assert stops == [
        "settings-discover-provider-models",
        "settings-discovered-models-list",
        "settings-save-discovered-provider-models",
        "settings-clear-discovered-provider-models",
    ]


@private_profile_test
@pytest.mark.parametrize(
    ("selector", "name", "save"),
    (
        (_DISCOVER, "Discover models", _SAVES_NOTHING),
        (_LIST, "Discovered models", _SAVES_NOTHING),
        (_ADD, "Add to model list", _INSTANT),
        (_CLEAR, "Clear", _SAVES_NOTHING),
    ),
    ids=("discover", "list", "add", "clear"),
)
async def test_the_inspector_names_each_saved_model_list_control(
    request, monkeypatch, selector, name, save
):
    """AC#12 (B19): focused, each control is named in the Inspector, with
    what it does, where it saves and whether it writes anything."""
    _write(_LLAMA)
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-beta"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await _press_discover(host, pilot, screen)
        screen.query_one(selector).focus()
        await _settle(host, pilot)
        guide = [
            str(screen.query_one(f"#settings-provider-field-guide-{i}").renderable)
            for i in range(4)
        ]
        saved_as = str(
            screen.query_one("#settings-provider-config-key-saved-as").renderable
        )

    assert guide[0] == f"Focused setting: {name}", guide
    assert guide[1].startswith("Purpose: ") and len(guide[1]) > 20, guide
    assert guide[2].startswith(f"Save: {save}"), guide
    assert guide[3].startswith("Validation: "), guide
    assert saved_as.split(": ", 1)[1].strip(), saved_as


# --- Case 11: the capability line -------------------------------------------


@private_profile_test
async def test_the_capability_line_no_longer_ties_capabilities_to_saving(
    request, monkeypatch
):
    """AC#13 (C-M2): saving never resolves capabilities, so neither Settings'
    discovery card nor the warning Console's Chat settings readiness detail
    shows says it does; both read one shared line."""
    from Tests.UI.test_console_session_settings import (
        FakeConsoleModelDiscoveryScope,
        _merged_model,
    )
    from tldw_chatbook.UI.Screens import provider_model_resolution

    _write(_LLAMA)
    app = _booted_app()
    host = _SettingsCssHarness(app, "settings")
    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        line = str(
            screen.query_one(
                "#settings-model-discovery-capability-warning", Static
            ).renderable
        )
    scope = FakeConsoleModelDiscoveryScope(
        (
            _merged_model(
                "gpt-5",
                source="runtime_discovered",
                capability_status="unknown",
                persisted=False,
            ),
        )
    )
    options = await provider_model_resolution.resolve_provider_model_options(
        {"openai": []}, scope, provider="OpenAI"
    )

    assert line == _CAPABILITY
    assert options[0].warning == _CAPABILITY
    assert "saved" not in _CAPABILITY


# --- Case 12: the combined validator's providers branch ----------------------


def _admits(provider: str, model: str, providers: object) -> bool:
    from tldw_chatbook.Chat.provider_setup_persistence import (
        _ownership_for,
        _validate_combined_provider_settings_mutation,
    )

    try:
        _validate_combined_provider_settings_mutation(
            None, _ownership_for(provider), model, {"providers": providers}, {}
        )
    except ValueError:
        return False
    return True


@private_profile_test
async def test_the_combined_save_admits_only_its_own_list_holding_the_model(request):
    """The validator's providers branch: the mutation's own provider's list
    (Custom_2 for Custom 2, through provider_config_key, which _ownership_for
    cannot spell), holding the model saved; never another provider's key, a
    list without the model, a non-list, a second key, or no model."""
    assert _admits("llama_cpp", "m", {"Llama_cpp": ["None", "m"]})
    assert _admits("custom_2", "m", {"Custom_2": ["m"]})
    assert _admits("Custom 2", "m", {"Custom_2": ["a", "m"]})
    assert not _admits("llama_cpp", "m", {"OpenAI": ["m"]})
    assert not _admits("llama_cpp", "m", {"Llama_cpp": ["a"]})
    assert not _admits("llama_cpp", "m", {"Llama_cpp": "m"})
    assert not _admits("llama_cpp", "m", {"Llama_cpp": ["m"], "llama_cpp": ["m"]})
    assert not _admits("llama_cpp", "m", {"Llama_cpp": ["m", "x\x00"]})
    assert not _admits("llama_cpp", "", {"Llama_cpp": ["m"]})


@private_profile_test
async def test_the_inspector_keeps_one_row_shape_for_the_four_controls(request):
    """B19: the four controls' guides keep the 4 content rows + Save shape
    every guidance branch has, and each says it writes something or not."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    screen = SettingsScreen(_build_test_app())
    for field_id in (
        "settings-discover-provider-models",
        "settings-discovered-models-list",
        "settings-save-discovered-provider-models",
        "settings-clear-discovered-provider-models",
    ):
        screen._active_settings_field_id = field_id
        rows = screen._provider_field_guidance_rows()
        assert len(rows) == 5, (field_id, rows)
        assert dict(rows)["Save"] in {_INSTANT, _SAVES_NOTHING}, (field_id, rows)
