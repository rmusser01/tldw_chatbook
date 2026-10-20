"""Saved model lists keep what the config file holds (TASK-35010).

C3: Save selected, the startup catalog refresh and Advanced Config wrote a
provider's whole list from the startup snapshot the app kept in memory, so a
model added to the file later was dropped and a deleted one came back. M2: the
shipped template seeded vLLM, TabbyAPI, Aphrodite, Custom and Custom 2 with
sample ids no server serves. NO-1: an untouched local server with no model is
still probed. Every case runs on a private profile. Configuration is written
to the file through the production writer, never by seeding
``app.providers_models`` by hand; the app reads it as boot does
(``_init_providers_models``).
"""

from __future__ import annotations

import hashlib
import os
import time
import tomllib
from copy import deepcopy
from pathlib import Path

import httpx
import pytest
import toml
from textual.widgets import Button, Collapsible, SelectionList

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_model_switcher import _settle as _settle_switcher
from Tests.UI.test_console_switcher_local_probe import (
    PROVIDERS_MODELS,
    _group,
    _harness,
    _with_shipped,
    build,
    servers,  # noqa: F401 - the fixture the probe cases take
)
from Tests.UI.test_settings_default_model_picker import (
    _open_providers,
    _press_discover,
    _rows,
    _settle,
    _widgets,
)
from Tests.UI.test_settings_narrow_layout import _SettingsCssHarness
from Tests.UI.test_settings_provider_key_check import _Provider, _route_discovery_to
from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

pytestmark = pytest.mark.asyncio

_SIZE = (211, 44)
_FAKE_KEY = "sk-proj-abcdefghijklmnop1234"
PROVIDERS_MODELS_CATEGORY = SettingsCategoryId.PROVIDERS_MODELS

#: The lists older templates wrote (OD-1 (a)'s frozen set, by list).
_OLD_SAMPLE_LISTS = {
    "vLLM": ["vllm-model-z", "vllm-model-x", "vllm-model-y", "vllm-model-a"],
    "TabbyAPI": ["tabby-model", "tabby-model-2", "tabby-model-3"],
    "Aphrodite": ["aphrodite-engine", "aphrodite-engine-2"],
    "Custom": ["custom-model-alpha", "custom-model-beta"],
    "Custom_2": ["custom-model-gamma", "custom-model-delta"],
}
_OLD_SAMPLES = frozenset(m for models in _OLD_SAMPLE_LISTS.values() for m in models)


def _config_path() -> Path:
    return Path(os.environ["TLDW_CONFIG_PATH"])


def _write(sections: dict) -> None:
    """Persist through the production writer, as Settings does."""
    from tldw_chatbook.config import save_settings_to_cli_config

    assert save_settings_to_cli_config(sections)


def _saved(key: str) -> list[str]:
    return tomllib.loads(_config_path().read_text())["providers"][key]


def _edit_in_place(mutate) -> None:
    """Rewrite the config file in place, as another program's editor might:
    same inode, so the config cache's path identity check still matches."""
    path = _config_path()
    inode = path.stat().st_ino
    data = tomllib.loads(path.read_text())
    mutate(data)
    with path.open("r+", encoding="utf-8") as handle:
        handle.seek(0)
        handle.write(toml.dumps(data))
        handle.truncate()
    assert path.stat().st_ino == inode


def _booted_app():
    """The app as boot builds it: [providers] read from the file."""
    app = _build_test_app()
    app._init_providers_models()
    return app


def _listing(*ids: str) -> _Provider:
    return _Provider(httpx.Response(200, json={"data": [{"id": m} for m in ids]}))


async def _until(pilot, condition, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await pilot.pause(0.02)


async def _save_selected(host, pilot, screen, *model_ids: str) -> str:
    """Discover, tick ``model_ids``, press Add to model list (once Save
    selected); return its line."""
    await _press_discover(host, pilot, screen)
    listing = screen.query_one("#settings-discovered-models-list", SelectionList)
    listing.deselect_all()
    for model_id in model_ids:
        listing.select(model_id)
    await pilot.pause()
    screen.query_one("#settings-save-discovered-provider-models", Button).press()
    await _until(
        pilot,
        lambda: not screen._model_discovery_status.startswith("Saving"),
    )
    await _settle(host, pilot)
    return screen._model_discovery_status


async def _picker_rows(host, pilot, screen) -> list[str]:
    """The Default model picker's open list, then leave it."""
    _picker, field, results, _adapter = _widgets(screen)
    field.focus()
    await _settle(host, pilot)
    rows = _rows(results)
    screen.set_focus(None)
    await _settle(host, pilot)
    return rows


def _saved_count_title(screen) -> str:
    return str(screen.query_one("#settings-advanced-saved-models", Collapsible).title)


def _llama_cpp_profile(*models: str) -> None:
    _write(
        {
            "chat_defaults": {"provider": "llama_cpp", "model": models[0]},
            "providers": {"Llama_cpp": ["None", *models]},
        }
    )


# --- Case 1: Save selected starts from the file --------------------------


@private_profile_test
@pytest.mark.parametrize("edit", ("replaced", "in_place"))
async def test_save_selected_keeps_a_model_the_file_gained_after_start(
    request, monkeypatch, edit
):
    """C3, AC#1, AC#2, AC#4: a model another program added to the saved list
    after the app started is kept by Save selected, whether that program
    replaced the file (the app's own writer) or edited it in place (same
    inode, which the config cache cannot see). The receipt counts the one
    model added, and the in-memory lists are the file's afterwards."""
    _llama_cpp_profile("m-one")
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-beta", "m-one"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        if edit == "replaced":
            _write({"providers": {"Llama_cpp": ["None", "m-one", "hand-added"]}})
        else:
            _edit_in_place(
                lambda data: data["providers"]["Llama_cpp"].append("hand-added")
            )
        line = await _save_selected(host, pilot, screen, "fake-beta", "m-one")
        label = screen._provider_display_label("llama_cpp")

    assert _saved("Llama_cpp") == ["None", "m-one", "hand-added", "fake-beta"]
    # TASK-35011 (C4 residual), rewritten on purpose: it names the model.
    assert line == f"Added fake-beta to {label}'s model list."
    assert app.providers_models["Llama_cpp"] == _saved("Llama_cpp")


@private_profile_test
async def test_save_selected_says_already_saved_when_the_file_lists_them(
    request, monkeypatch
):
    """AC#2: a model the file gained after start counts as already saved,
    so the receipt never claims to add it."""
    _llama_cpp_profile("m-one")
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("hand-added"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        _edit_in_place(lambda data: data["providers"]["Llama_cpp"].append("hand-added"))
        before = _config_path().read_bytes()
        line = await _save_selected(host, pilot, screen, "hand-added")
        label = screen._provider_display_label("llama_cpp")

    # TASK-35011 (C4 residual), rewritten on purpose.
    assert line == (
        f"The selected models are already in {label}'s model list. To make one "
        "the default, choose it in Default model."
    )
    assert _config_path().read_bytes() == before
    assert app.providers_models["Llama_cpp"] == ["None", "m-one", "hand-added"]


# --- Case 2: Advanced Config deletions stay deleted ------------------------


@private_profile_test
async def test_a_model_deleted_in_advanced_config_leaves_every_list_and_stays_gone(
    request, monkeypatch
):
    """C3, AC#1, AC#4: after an Advanced Config save that deletes m-two, the
    Default model picker's Saved fallback, the Saved model list count and a
    Switch model built from the app's lists (no listing evidence) no longer
    show it, without a recompose; the next Save selected does not bring it
    back."""
    _llama_cpp_profile("m-one", "m-two", "m-three")
    app = _booted_app()
    _route_discovery_to(monkeypatch, _listing("fake-alpha"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        picker = screen.query_one("#settings-model-picker")
        assert "m-two" in await _picker_rows(host, pilot, screen)
        assert "3 saved in config" in _saved_count_title(screen)

        raw = screen._raw_config_model()
        await raw.inspect_current()
        data = tomllib.loads(raw.state.text)
        data["providers"]["Llama_cpp"].remove("m-two")
        raw.edit(toml.dumps(data))
        await raw.validate()
        await raw.save()
        await _settle(host, pilot)
        assert raw.result.startswith("Saved"), raw.result

        assert screen.query_one("#settings-model-picker") is picker  # no recompose
        assert "m-two" not in await _picker_rows(host, pilot, screen)
        assert "2 saved in config" in _saved_count_title(screen)
        switcher = build(app, providers_models=app.providers_models, prober=False)
        assert switcher._models_for("llama_cpp") == ("m-one", "m-three")

        await _save_selected(host, pilot, screen, "fake-alpha")

    assert _saved("Llama_cpp") == ["None", "m-one", "m-three", "fake-alpha"]


@private_profile_test
async def test_a_late_advanced_config_save_after_settings_closed_updates_only_the_lists(
    request, monkeypatch
):
    """Review of TASK-35010: Advanced Config's `applied` outlives the screen
    (its session is retained), so a save that finishes after Settings closed
    still adopts the config. The app's config and lists follow the file, and
    the closed screen's widgets are left alone: a popped screen stays
    `is_mounted` in Textual 8, so the guard is `is_attached`."""
    from tldw_chatbook.config import load_settings

    _llama_cpp_profile("m-one")
    app = _booted_app()
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        await host.pop_screen()
        await pilot.pause()
        assert not screen.is_attached
        calls: list[str] = []
        monkeypatch.setattr(
            screen, "_sync_model_picker", lambda *_a, **_k: calls.append("picker")
        )
        monkeypatch.setattr(
            screen, "_refresh_advanced", lambda *_a, **_k: calls.append("advanced")
        )
        _write({"providers": {"Llama_cpp": ["None", "m-one", "m-late"]}})

        screen._adopt_config(load_settings())

        assert app.providers_models["Llama_cpp"] == ["None", "m-one", "m-late"]
        assert calls == []


# --- Case 3: Config reload refreshes the lists -----------------------------


@private_profile_test
@pytest.mark.parametrize("site", ("reload", "validate_and_reload"))
async def test_config_reload_reads_an_in_place_edit_into_the_saved_lists(
    request, monkeypatch, site
):
    """AC#4: both Config reload sites (Diagnostics' Reload, and its
    validate-and-reload worker) re-read the file, even after an in-place
    edit the config cache did not see; the picker and the Saved model list
    count follow. Whole-S1 review: when that re-read cannot republish the
    caches, the lists are still the old ones, so the reload line says they
    were not refreshed instead of "loaded" (only the republish is stubbed)."""
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen

    _llama_cpp_profile("m-one", "m-two")
    app = _booted_app()
    host = _SettingsCssHarness(app, "settings")

    async def reload(screen) -> str:
        if site == "reload":
            screen._run_diagnostics_reload()
        else:
            screen._diagnostics_validation_and_reload_worker()
            await host.workers.wait_for_complete()
        await _settle(host, pilot)
        return screen._diagnostics_reload_result

    def refused():
        return config.LiteralConfigMutationResult(False, False, None, "cache_reload")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        _edit_in_place(
            lambda data: data["providers"].update(Llama_cpp=["None", "m-one", "m-four"])
        )
        loaded = await reload(screen)
        rows = await _picker_rows(host, pilot, screen)
        title = _saved_count_title(screen)
        lists = list(app.providers_models["Llama_cpp"])

        _edit_in_place(
            lambda data: data["providers"].update(Llama_cpp=["None", "m-five"])
        )
        for module in (config, settings_screen):
            monkeypatch.setattr(
                module, "refresh_runtime_config_from_cli_config", refused
            )
        stale = await reload(screen)

    assert lists == ["None", "m-one", "m-four"]
    assert "m-four" in rows and "m-two" not in rows, rows
    assert "2 saved in config" in title, title
    assert loaded.startswith("Config reload: loaded"), loaded
    assert stale.startswith(
        "Config reload: loaded, but the saved model lists were not refreshed; "
        "reload again. Details are in Logs (F3)."
    ), stale
    assert app.providers_models["Llama_cpp"] == ["None", "m-one", "m-four"]


# --- Case 4: the startup refresh's write-through ---------------------------


@private_profile_test
async def test_the_startup_refresh_keeps_an_external_addition_and_appends_new_ids(
    request, monkeypatch, tmp_path
):
    """AC#3: the catalog refresh with save to config, through the production
    writer (no injected callback), keeps a model another program added in
    place and still appends what the provider newly lists."""
    from tldw_chatbook.config import load_settings
    from tldw_chatbook.LLM_Provider_Catalog.local_llm_provider_catalog_service import (
        LocalLLMProviderCatalogService,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
        ModelCatalogSettings,
    )
    from tldw_chatbook.LLM_Provider_Catalog.model_discovery_disk_cache import (
        ModelCatalogDiskStore,
    )

    _write(
        {
            "providers": {"OpenAI": ["kept-a"]},
            "api_settings.openai": {"api_key": _FAKE_KEY},
        }
    )
    load_settings()  # warm, as a running app's cache is
    _edit_in_place(lambda data: data["providers"]["OpenAI"].append("external-x"))
    _route_discovery_to(monkeypatch, _listing("kept-a", "new-b"))
    service = LocalLLMProviderCatalogService(environ={})

    report = await service.refresh_stale_configured_providers(
        catalog_settings=ModelCatalogSettings(write_to_config=frozenset({"openai"})),
        disk_store=ModelCatalogDiskStore(tmp_path / "catalog.json"),
        provider_list_keys=("OpenAI",),
        force=True,
    )

    assert report.outcomes[0].saved_model_ids == ("new-b",), report.outcomes
    assert _saved("OpenAI") == ["kept-a", "external-x", "new-b"]


# --- Cases 5 and 6: the shipped sample ids ---------------------------------


@private_profile_test
async def test_the_template_ships_empty_lists_and_old_sample_ids_are_ignored(request):
    """AC#5, AC#6, AC#10: the template's five lists are empty. On a file an
    older version wrote, the loader (the source of the app's lists, the
    wizard's list and Switch model) drops the 13 sample ids and keeps every
    key and every real id beside them. Settings' default choice ignores a
    sample, and the send path's own resolver still returns it."""
    from tldw_chatbook.Chat.provider_setup_persistence import (
        resolve_remembered_provider_model,
    )
    from tldw_chatbook.config import (
        CONFIG_TOML_CONTENT,
        DEFAULT_CONFIG_FROM_TOML,
        SHIPPED_SAMPLE_MODEL_IDS,
        get_cli_providers_and_models,
        load_settings,
    )
    from tldw_chatbook.UI.Wizards.first_run_setup_state import (
        curated_models_for_provider,
    )

    shipped = tomllib.loads(CONFIG_TOML_CONTENT)["providers"]
    assert {key: shipped[key] for key in _OLD_SAMPLE_LISTS} == {
        key: [] for key in _OLD_SAMPLE_LISTS
    }
    assert SHIPPED_SAMPLE_MODEL_IDS == _OLD_SAMPLES

    _write(
        {
            "providers": {
                **_OLD_SAMPLE_LISTS,
                "vLLM": [*_OLD_SAMPLE_LISTS["vLLM"], "my-vllm"],
            }
        }
    )
    lists = get_cli_providers_and_models()
    assert {key: lists[key] for key in _OLD_SAMPLE_LISTS} == {
        "vLLM": ["my-vllm"],
        "TabbyAPI": [],
        "Aphrodite": [],
        "Custom": [],
        "Custom_2": [],
    }
    assert curated_models_for_provider(lists, "vllm") == ["my-vllm"]
    assert curated_models_for_provider(lists, "aphrodite") == []

    config = load_settings()
    assert DEFAULT_CONFIG_FROM_TOML["api_settings"]["aphrodite"]["model"] == (
        "aphrodite-engine"
    )
    assert resolve_remembered_provider_model(config, "aphrodite") == "aphrodite-engine"
    assert (
        resolve_remembered_provider_model(config, "aphrodite", skip_samples=True)
        is None
    )
    assert resolve_remembered_provider_model(
        config, "llama_cpp", skip_samples=True
    ) == resolve_remembered_provider_model(config, "llama_cpp")


@private_profile_test
@pytest.mark.parametrize(
    ("provider", "list_key", "real"),
    (("vllm", "vLLM", ()), ("aphrodite", "Aphrodite", ()), ("vllm", "vLLM", ("mine",))),
    ids=("vllm", "aphrodite", "vllm-with-a-real-id"),
)
async def test_an_old_sample_list_leaves_default_model_unset_until_save_selected(
    request, monkeypatch, provider, list_key, real
):
    """AC#6, AC#7, AC#8: on a file holding the old samples, switching
    Settings to vLLM (whose table has no model) or Aphrodite (whose table
    still holds the shipped aphrodite-engine) leaves Default model not set,
    and the picker and the Saved model list count show no sample; a real id
    saved beside them shows, and is the default. Save selected then fills an
    unset Default model as an unsaved draft, and that write drops the
    samples from the file and keeps the real id. Booting never rewrites the
    file for this."""
    _write(
        {
            "chat_defaults": {"provider": "openai", "model": "gpt-4o"},
            "providers": {
                **_OLD_SAMPLE_LISTS,
                list_key: [*_OLD_SAMPLE_LISTS[list_key], *real],
            },
        }
    )
    before_boot = hashlib.sha256(_config_path().read_bytes()).hexdigest()
    app = _booted_app()
    assert hashlib.sha256(_config_path().read_bytes()).hexdigest() == before_boot
    _route_discovery_to(monkeypatch, _listing("fake-alpha"))
    host = _SettingsCssHarness(app, "settings")

    async with host.run_test(size=_SIZE) as pilot:
        screen = await _open_providers(host, pilot)
        await _settle(host, pilot)
        screen._apply_provider_value_change(provider)
        await _settle(host, pilot)
        _picker, _field, _results, adapter = _widgets(screen)
        unset = adapter.value
        rows = await _picker_rows(host, pilot, screen)
        title = _saved_count_title(screen)

        await _save_selected(host, pilot, screen, "fake-alpha")
        filled = adapter.value
        unsaved = screen._category_has_unsaved_changes(PROVIDERS_MODELS_CATEGORY)

    assert unset == "".join(real), unset
    assert not _OLD_SAMPLES & set(rows), rows
    assert all(any(row.startswith(m) for row in rows) for m in real), rows
    assert f"{len(real)} saved in config" in title, title
    # Filled as a draft (TASK-369): nothing about the default is saved yet.
    assert (filled, unsaved) == ((*real, "fake-alpha")[0], True)
    assert tomllib.loads(_config_path().read_text())["chat_defaults"]["model"] == (
        "gpt-4o"
    )
    assert _saved(list_key) == [*real, "fake-alpha"]


# --- Cases 7 and 9: Switch model on old samples and untouched servers ------


@private_profile_test
async def test_switch_model_lists_no_sample_id_and_untouched_refusals_stay_quiet(
    servers,  # noqa: F811 - the probe fixture, imported above
    request,
):
    """AC#6, AC#9: Switch model built from the loader's lists of an
    old-sample file lists no sample id but lists a real id saved beside
    them, and untouched TabbyAPI and Aphrodite tables (they still name a
    sample model) that refuse stay under NOT RUNNING."""
    from tldw_chatbook.config import get_cli_providers_and_models

    _write({"providers": {**_OLD_SAMPLE_LISTS, "vLLM": ["vllm-model-z", "my-vllm"]}})
    lists = get_cli_providers_and_models()
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        offline = build(app, providers_models=lists, prober=False)
        models = {key: offline._models_for(key) for key in ("vllm", "tabbyapi")}
        switcher = build(
            app, config=_with_shipped("tabbyapi", "aphrodite"), providers_models=lists
        )
        await app.push_screen(switcher)
        await _settle_switcher(app, pilot)
        rows = [row for row in switcher._rows if row.kind in {"pair", "setup"}]
        quiet = " ".join(row.text for row in _group(switcher, "NOT RUNNING"))

    assert models == {"vllm": ("my-vllm",), "tabbyapi": ()}, models
    assert not _OLD_SAMPLES & {row.model for row in rows}, rows
    assert "TabbyAPI" in quiet and "Aphrodite Engine" in quiet, quiet


#: The five URL-based servers a fresh profile leaves with no model, at their
#: template ports (config.py's [api_settings.*] tables).
_MODEL_LESS = {
    "vllm": 8000,
    "llama_cpp": 8080,
    "koboldcpp": 5001,
    "oobabooga": 5000,
    "local_llamafile": 8001,
}


@private_profile_test
@pytest.mark.parametrize(
    "running",
    (("vllm", "koboldcpp"), ("llama_cpp", "oobabooga", "local_llamafile")),
    ids=("vllm-koboldcpp-run", "llamacpp-oobabooga-llamafile-run"),
)
async def test_an_untouched_server_with_no_model_is_probed_listed_or_not_running(
    servers,  # noqa: F811 - the probe fixture, imported above
    request,
    running,
):
    """NO-1 (a), its scope widened 2026-10-19 and ruled 2026-10-20: vLLM's
    emptied list, and the "None" lists of llama.cpp, koboldcpp, Oobabooga and
    llamafile, leave each template table with no model, so readiness says
    model_missing before any endpoint result. Each is still probed: one that
    runs is listed with the models it served, one that refuses is under NOT
    RUNNING, and none shows under NEEDS SETUP. Each of the five runs in one
    case and refuses in the other."""
    for key in running:
        servers.up.add(_MODEL_LESS[key])
        servers.listing[_MODEL_LESS[key]] = [f"{key}-served-a", f"{key}-served-b"]
    providers_models = {
        **PROVIDERS_MODELS,
        "vllm": [],
        **{key: ["None"] for key in _MODEL_LESS if key != "vllm"},
    }
    app = _harness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = build(
            app, config=_with_shipped(*_MODEL_LESS), providers_models=providers_models
        )
        await app.push_screen(switcher)
        await _settle_switcher(app, pilot)
        await _settle_switcher(app, pilot)
        setup = {row.provider for row in _group(switcher, "NEEDS SETUP")}
        ready = {
            (row.provider, row.model)
            for row in _group(switcher, "READY PROVIDERS")
            if row.kind == "pair"
        }
        quiet = " ".join(row.text for row in _group(switcher, "NOT RUNNING"))
        names = {key: switcher._display(key) for key in _MODEL_LESS}

    unprobed = set(_MODEL_LESS.values()) - set(servers.ports)
    assert not unprobed, unprobed
    assert not set(_MODEL_LESS) & setup, setup
    for key in _MODEL_LESS:
        if key in running:
            assert (key, f"{key}-served-a") in ready, (key, ready)
            assert names[key] not in quiet, quiet
        else:
            assert names[key] in quiet, (key, quiet)
            assert not any(provider == key for provider, _model in ready), ready


@private_profile_test
async def test_koboldcpps_native_endpoint_lists_at_v1_models_and_sends_unchanged(
    request,
):
    """NO-1, ruled 2026-10-20: koboldcpp's template endpoint is KoboldAI's
    native /api/v1/generate. It resolves with koboldcpp's OpenAI-compatible
    /v1/models as its listing route, so the switcher may probe it, while the
    URL a send uses and the URL a Settings save writes stay the native one.
    Other providers given that path are still refused, and a koboldcpp root
    resolves as before. AC#10: an untouched Aphrodite still sends with its
    table's model."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.provider_endpoint_contract import (
        ConnectionProbeAvailability,
        connection_probe_availability,
        resolve_provider_endpoint,
    )
    from tldw_chatbook.Chat.provider_setup_persistence import (
        ProviderSetupDraft,
        build_provider_setup_mutation,
    )
    from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML

    template = DEFAULT_CONFIG_FROM_TOML["api_settings"]
    native = template["koboldcpp"]["api_url"]
    assert native == "http://localhost:5001/api/v1/generate"

    resolution = resolve_provider_endpoint("koboldcpp", native)
    assert resolution.errors == ()
    assert resolution.models_url == "http://localhost:5001/v1/models"
    assert resolution.chat_url == resolution.persisted_endpoint == native
    assert connection_probe_availability("koboldcpp", native) is (
        ConnectionProbeAvailability.MODELS_ROUTE
    )
    assert resolve_provider_endpoint("vllm", native).errors
    assert resolve_provider_endpoint("koboldcpp", f"{native}/x").errors
    root = resolve_provider_endpoint("koboldcpp", "http://localhost:5001")
    assert (root.chat_url, root.models_url) == (
        "http://localhost:5001/v1/chat/completions",
        "http://localhost:5001/v1/models",
    )

    config = deepcopy(DEFAULT_CONFIG_FROM_TOML)
    mutation = build_provider_setup_mutation(
        ProviderSetupDraft("koboldcpp", "kobold-model", native, "none", 0, 0), config
    )
    assert mutation.section_values["api_settings.koboldcpp"]["api_url"] == native

    tables = {
        "koboldcpp": {**template["koboldcpp"], "model": "kobold-model"},
        "aphrodite": deepcopy(template["aphrodite"]),
    }
    gateway = ConsoleProviderGateway(
        config_provider=lambda: {"api_settings": tables}, environ={}
    )
    try:
        kobold = await gateway.resolve_for_send(
            ConsoleProviderSelection(provider="koboldcpp")
        )
        aphrodite = await gateway.resolve_for_send(
            ConsoleProviderSelection(provider="aphrodite")
        )
        kwargs = ConsoleProviderGateway._chat_api_kwargs(
            kobold, [{"role": "user", "content": "hi"}]
        )
    finally:
        await gateway.aclose()
    assert (kobold.ready, kobold.base_url) == (True, native)
    assert kwargs.get("api_base_url", native) == native
    assert aphrodite.model == "aphrodite-engine"


# --- Case 8 and D-a: what the locked write reports --------------------------


@private_profile_test
async def test_the_locked_write_drops_samples_and_maps_a_refusal_to_an_error(
    request, monkeypatch
):
    """D-a, AC#8: a Save selected that adds nothing to a list still holding
    sample ids writes the list without them and says already saved.
    TASK-33641.6: a config read from inside the write transaction raises
    RecoveryRequired at once; the transaction maps that to status="error"
    with today's message, and the file is untouched."""
    from tldw_chatbook import config
    from tldw_chatbook.LLM_Provider_Catalog import model_discovery_persistence

    _write({"providers": {"vLLM": ["vllm-model-z", "mine", "vllm-model-a"]}})
    cleaned = model_discovery_persistence.persist_discovered_models_to_settings(
        providers_config=config.get_cli_providers_and_models(),
        requested_provider="vllm",
        model_ids=["mine"],
    )
    assert (cleaned.status, cleaned.saved_model_ids) == ("saved", ())
    assert _saved("vLLM") == ["mine"]

    _llama_cpp_profile("m-one")
    before = _config_path().read_bytes()

    def reentrant(*_args, **_kwargs):
        config.get_atomic_config_snapshot()  # inside the lock: refused
        raise AssertionError("the write lock let a second read in")

    monkeypatch.setattr(
        model_discovery_persistence, "append_models_to_provider_list", reentrant
    )
    result = model_discovery_persistence.persist_discovered_models_to_settings(
        providers_config=config.get_cli_providers_and_models(),
        requested_provider="llama_cpp",
        model_ids=["fake-beta"],
    )

    assert (result.status, result.saved_model_ids) == ("error", ())
    assert result.message == "Could not save discovered models to the providers list."
    assert _config_path().read_bytes() == before
