"""Guided search setup uses real drafts and atomic configuration persistence."""

import asyncio
import os
import threading
import tomllib
from pathlib import Path

import pytest

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook import config
from tldw_chatbook.UI.Screens.settings_config_models import (
    SettingsCategoryId,
    SettingsDraft,
)
from tldw_chatbook.UI.Screens.settings_web_search import WebSearchSettings
from tldw_chatbook.Web_Scraping import search_backend_settings as catalog


class SearchSettingsHarness(DestinationHarness):
    CSS_PATH = str(BUNDLED_STYLESHEET)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text(
        '[SearchSettings]\nsearch_provider_default = "serper"\n[SearchEngines]\nserper_search_api_key = "saved-secret"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
    for spec in catalog.BACKENDS.values():
        for field in spec.fields:
            monkeypatch.delenv(field.env_var, raising=False)
    for name in (
        "_CONFIG_CACHE",
        "_CONFIG_CACHE_SOURCE",
        "_SETTINGS_CACHE",
        "_SETTINGS_CACHE_SOURCE",
        "_CONFIG_GENERATION",
        "settings",
    ):
        monkeypatch.setattr(config, name, getattr(config, name))
    config.load_cli_config_and_ensure_existence(force_reload=True)
    draft = SettingsDraft(SettingsCategoryId.WEB_SEARCH)
    model = WebSearchSettings(lambda: draft, lambda: None)
    return model, path


@pytest.mark.asyncio
async def test_other_backend_setup_preserves_default_and_saves_atomic_delta(setup):
    model, path = setup
    model.select_backend("brave")
    model.edit("brave_search_api_key", "replacement")
    model.select_backend("serper")
    model.clear("serper_search_api_key")
    assert model.default_backend == "serper"
    assert model.draft.is_dirty
    await model.save()
    raw = tomllib.loads(path.read_text())
    assert raw["SearchSettings"]["search_provider_default"] == "serper"
    assert raw["SearchEngines"] == {"brave_search_api_key": "replacement"}
    assert not model.draft.is_dirty


def test_blank_secret_keeps_saved_and_revert_preserves_config(setup):
    model, path = setup
    original = path.read_text()
    assert model.input_value(catalog.BACKENDS["serper"].fields[0]) == ""
    model.edit("serper_search_api_key", "new")
    model.edit("serper_search_api_key", "")
    assert not model.draft.is_dirty
    model.clear("serper_search_api_key")
    assert model.draft.is_dirty
    model.revert()
    assert not model.draft.is_dirty
    assert path.read_text() == original


@pytest.mark.asyncio
async def test_probe_is_explicit_saved_only_and_stale_completion_is_discarded(
    setup, monkeypatch
):
    model, _ = setup
    started, release = threading.Event(), threading.Event()
    calls = []

    def probe(backend):
        calls.append(backend)
        started.set()
        release.wait(3)
        return catalog.ProbeResult(True, "Search completed successfully.", 1)

    monkeypatch.setattr(catalog, "probe_saved_backend", probe)
    model.edit("serper_search_api_key", "draft")
    await model.test_saved()
    assert calls == []
    model.revert()
    task = asyncio.create_task(model.test_saved())
    await asyncio.to_thread(started.wait, 2)
    model.select_backend("brave")
    release.set()
    await task
    assert calls == ["serper"]
    assert "successfully" not in model.test_status


@pytest.mark.asyncio
async def test_save_failure_retains_draft_and_has_safe_message(setup, monkeypatch):
    model, _ = setup
    model.edit("serper_search_api_key", "draft-secret")

    def fail(*args, **kwargs):
        raise OSError("secret in exception")

    monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", fail)
    await model.save()
    assert model.draft.is_dirty
    assert "could not" in model.save_status.lower()
    assert "secret" not in model.save_status


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (80, 24)])
async def test_settings_navigation_preserves_masked_drafts_and_save_revert_work(
    setup, size
):
    from textual.widgets import Input, Select

    import Tests.UI.test_settings_category_sweep as sweep

    _, path = setup
    host = SearchSettingsHarness(sweep._build_test_app(), "settings")
    async with host.run_test(size=size) as pilot:
        await sweep._settle_settings(pilot)
        await sweep._click_settings_category(pilot, "web-search")
        screen = host.screen
        from Tests.UI.test_destination_shells import _static_text

        assert "No unsaved changes" in _static_text(
            screen.query_one("#web-search-save-hint")
        )
        if capture_dir := os.environ.get("SEARCH_UX_CAPTURE_DIR"):
            folder = Path(capture_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"settings-{size[0]}x{size[1]}.svg").write_text(
                host.export_screenshot(), encoding="utf-8"
            )
            screen.query_one("#settings-detail-pane-body").scroll_end(animate=False)
            await pilot.pause()
            assert screen.query_one("#web-search-test").region.y < size[1] - 1
            (folder / f"settings-{size[0]}x{size[1]}-test.svg").write_text(
                host.export_screenshot(), encoding="utf-8"
            )
        assert "web-search" in [
            category.value
            for group, categories in screen._category_groups()
            if group == "Core"
            for category in categories
        ]
        field = screen.query_one("#web-search-serper_search_api_key", Input)
        assert field.password and not field.value
        field.focus()
        await pilot.press("n", "e", "w")
        await pilot.pause()
        assert screen._category_has_unsaved_changes(SettingsCategoryId.WEB_SEARCH)
        screen.query_one("#web-search-backend", Select).value = "brave"
        await pilot.pause()
        assert screen.query_one("#web-search-default", Select).value == "serper"
        await sweep._click_settings_category(pilot, "overview")
        await sweep._click_settings_category(pilot, "web-search")
        screen.query_one("#web-search-backend", Select).value = "serper"
        await pilot.pause()
        assert (
            screen.query_one("#web-search-serper_search_api_key", Input).value == "new"
        )
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert (
            tomllib.loads(path.read_text())["SearchEngines"]["serper_search_api_key"]
            == "new"
        )
        assert not screen._category_has_unsaved_changes(SettingsCategoryId.WEB_SEARCH)
        # Saved credentials disappear from the replacement field immediately.
        assert screen.query_one("#web-search-serper_search_api_key", Input).value == ""
        screen.query_one("#web-search-default", Select).value = "brave"
        await pilot.pause()
        screen.action_settings_revert_category(allow_text_entry_focus=True)
        await pilot.pause()
        assert screen._category_has_unsaved_changes(SettingsCategoryId.WEB_SEARCH)
        await pilot.click("#confirm-button")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert screen.query_one("#web-search-default", Select).value == "serper"
        assert not screen._category_has_unsaved_changes(SettingsCategoryId.WEB_SEARCH)


@pytest.mark.asyncio
async def test_all_backend_fields_render_and_switching_is_clean(setup):
    from textual.widgets import Input, Select

    import Tests.UI.test_settings_category_sweep as sweep

    host = SearchSettingsHarness(sweep._build_test_app(), "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await sweep._settle_settings(pilot)
        await sweep._click_settings_category(pilot, "web-search")
        screen = host.screen
        for backend, spec in catalog.BACKENDS.items():
            screen.query_one("#web-search-backend", Select).value = backend
            await pilot.pause()
            assert len(screen.query("#settings-web-search-fields Input")) == len(
                spec.fields
            )
            for field in spec.fields:
                widget = screen.query_one(f"#web-search-{field.key}", Input)
                assert widget.region.width > 10 and widget.content_region.height >= 1
            assert not screen._category_has_unsaved_changes(
                SettingsCategoryId.WEB_SEARCH
            )


@pytest.mark.asyncio
async def test_external_change_is_not_overwritten(setup):
    model, path = setup
    model.edit("serper_search_api_key", "draft")
    config.save_setting_to_cli_config(
        "SearchEngines", "serper_search_api_key", "external"
    )
    await model.save()
    assert model.draft.is_dirty
    assert "changed elsewhere" in model.save_status
    assert (
        tomllib.loads(path.read_text())["SearchEngines"]["serper_search_api_key"]
        == "external"
    )


@pytest.mark.asyncio
async def test_clear_legacy_alias_and_env_precedence_are_explicit(setup, monkeypatch):
    model, path = setup
    config.save_setting_to_cli_config(
        "SearchEngines", "search_engine_api_key_bing", "legacy"
    )
    model.revert()
    monkeypatch.setenv("BING_SEARCH_API_KEY", "environment-secret")
    field = catalog.BACKENDS["bing"].fields[0]
    model.clear(field.key)
    assert "Environment: BING_SEARCH_API_KEY" in model.field_status(field)
    assert "environment-secret" not in model.field_status(field)
    await model.save()
    raw = tomllib.loads(path.read_text())
    assert "search_engine_api_key_bing" not in raw["SearchEngines"]
    assert (
        catalog.resolve_backend_fields("bing", model.raw)[field.key]
        == "environment-secret"
    )


@pytest.mark.asyncio
async def test_category_search_finds_backend_terms(setup):
    from textual.widgets import Input

    import Tests.UI.test_settings_category_sweep as sweep

    host = SearchSettingsHarness(sweep._build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await sweep._settle_settings(pilot)
        for query in ("web search", "brave", "searxng", "api key"):
            host.screen.query_one("#settings-category-search", Input).value = query
            await pilot.pause()
            assert host.screen.query_one("#settings-category-web-search").display


@pytest.mark.asyncio
async def test_probe_completion_does_not_repaint_another_category(setup, monkeypatch):
    import Tests.UI.test_settings_category_sweep as sweep
    from Tests.UI.test_destination_shells import _static_text

    started, release = threading.Event(), threading.Event()

    def probe(backend):
        started.set()
        release.wait(3)
        return catalog.ProbeResult(True, "Search completed successfully.", 1)

    monkeypatch.setattr(catalog, "probe_saved_backend", probe)
    host = SearchSettingsHarness(sweep._build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await sweep._settle_settings(pilot)
        await sweep._click_settings_category(pilot, "web-search")
        screen = host.screen
        screen.run_worker(screen._web_search_model().test_saved())
        assert await asyncio.to_thread(started.wait, 2)
        screen._select_category("overview", restore_focus=True)
        await pilot.pause()
        banner = _static_text(screen.query_one(".settings-state-banner"))
        release.set()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert _static_text(screen.query_one(".settings-state-banner")) == banner


def test_existing_searx_alias_is_visible_in_editable_url(setup):
    model, _ = setup
    config.save_setting_to_cli_config(
        "SearchEngines", "search_engine_searx_api", "http://localhost:9090/search"
    )
    model.revert()
    model.select_backend("searx")
    field = catalog.BACKENDS["searx"].fields[0]
    assert model.input_value(field) == "http://localhost:9090/search"
