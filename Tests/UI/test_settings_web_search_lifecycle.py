"""Guided backend configuration survives input and destination lifecycles."""

import asyncio
import threading
import tomllib

import pytest
from textual.widgets import Button, Input, Select

from Tests.UI.test_settings_category_sweep import (
    _build_test_app,
    _click_settings_category,
    _settle_settings,
)
from Tests.UI.test_settings_web_search import SearchSettingsHarness
from Tests.UI.test_settings_web_search import (
    setup as setup,  # noqa: PLC0414 - explicit pytest fixture re-export
)
from tldw_chatbook import config
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.settings_web_search_panel import WebSearchSettingsPanel


@pytest.mark.asyncio
async def test_status_refresh_keeps_queued_masked_input(setup):
    host = SearchSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "web-search")
        model = host.screen._web_search_model()
        field = host.screen.query_one("#web-search-serper_search_api_key", Input)
        field.focus()
        await pilot.pause()
        original = field.post_message
        armed = True

        def post(message):
            nonlocal armed
            if armed and isinstance(message, Input.Changed):
                armed = False
                asyncio.get_running_loop().call_soon(model._emit)
            return original(message)

        field.post_message = post
        await pilot.press("x")
        await pilot.pause()
        assert field.password and field.value == "x"
        assert model.draft.values["SearchEngines.serper_search_api_key"] == "x"


@pytest.mark.asyncio
async def test_backend_and_masked_draft_survive_destination_recreation(setup):
    host = SearchSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "web-search")
        screen = host.screen
        screen.query_one("#web-search-backend", Select).value = "brave"
        await pilot.pause()
        screen.query_one("#web-search-brave_search_api_key", Input).value = "new-key"
        await pilot.pause()
        state = screen.save_state()
        restored = SettingsScreen(host.app_instance)
        restored.restore_state(state)
        await host.switch_screen(restored)
        await _settle_settings(pilot)
        assert restored.query_one("#web-search-backend", Select).value == "brave"
        field = restored.query_one("#web-search-brave_search_api_key", Input)
        assert field.password and field.value == "new-key"
        assert restored.query_one("#web-search-default", Select).value == "serper"
        assert "Not tested" in restored._web_search_model().test_status


@pytest.mark.asyncio
@pytest.mark.parametrize("write_fails", [False, True])
async def test_save_result_reaches_recreated_settings(setup, monkeypatch, write_fails):
    _, path = setup
    host = SearchSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "web-search")
        screen = host.screen
        model = screen._web_search_model()
        model.edit("serper_search_api_key", "replacement-key")
        started, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        original = config.apply_settings_mutation_to_cli_config

        def save(*args, **kwargs):
            started.set()
            release.wait(4)
            try:
                if write_fails:
                    raise OSError("synthetic-sensitive-error")
                return original(*args, **kwargs)
            finally:
                finished.set()

        monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", save)
        screen.query_one("#settings-category-web-search").focus()
        await pilot.press("s")
        assert await asyncio.to_thread(started.wait, 2)
        state = screen.save_state()
        restored = SettingsScreen(host.app_instance)
        restored.restore_state(state)
        try:
            await host.switch_screen(restored)
            await pilot.pause()
        finally:
            release.set()
        assert await asyncio.to_thread(finished.wait, 3)
        await host.workers.wait_for_complete()
        await pilot.pause()
        current = restored._web_search_model()
        raw = tomllib.loads(path.read_text())
        assert "synthetic-sensitive-error" not in current.save_status
        if write_fails:
            assert raw["SearchEngines"]["serper_search_api_key"] == "saved-secret"
            assert current.draft.is_dirty
            assert "Could not save" in current.save_status
            assert (
                restored.query_one("#web-search-serper_search_api_key", Input).value
                == "replacement-key"
            )
        else:
            assert raw["SearchEngines"]["serper_search_api_key"] == "replacement-key"
            assert not restored._category_has_unsaved_changes(
                SettingsCategoryId.WEB_SEARCH
            )
            assert "Saved." in current.save_status
            assert (
                restored.query_one("#web-search-serper_search_api_key", Input).value
                == ""
            )


@pytest.mark.asyncio
async def test_pending_test_stays_in_progress_after_navigation_and_discards_result(
    setup, monkeypatch
):
    from tldw_chatbook.Web_Scraping import search_backend_settings as catalog

    started, release = threading.Event(), threading.Event()
    calls = []

    def probe(backend):
        calls.append(backend)
        started.set()
        release.wait(4)
        return catalog.ProbeResult(True, "Old successful result", 1)

    monkeypatch.setattr(catalog, "probe_saved_backend", probe)
    host = SearchSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(80, 24)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "web-search")
        screen = host.screen
        test = screen.query_one("#web-search-test")
        test.focus()
        await pilot.press("enter")
        assert await asyncio.to_thread(started.wait, 2)
        restored = SettingsScreen(host.app_instance)
        restored.restore_state(screen.save_state())
        try:
            await host.switch_screen(restored)
            await pilot.pause()
            model = restored._web_search_model()
            assert model.testing
            assert restored.query_one("#web-search-test").disabled
            assert "finishing" in model.test_status
            assert calls == ["serper"]
        finally:
            release.set()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert not model.testing
        assert "Not tested" in model.test_status
        assert "Old successful" not in model.test_status
        assert not restored.query_one("#web-search-test").disabled


@pytest.mark.asyncio
async def test_recreated_clear_detects_intervening_credential_edit(setup):
    _, path = setup
    host = SearchSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "web-search")
        screen = host.screen
        screen._web_search_model().clear("serper_search_api_key")
        restored = SettingsScreen(host.app_instance)
        restored.restore_state(screen.save_state())
        await host.switch_screen(restored)
        await _settle_settings(pilot)
        current = restored._web_search_model()
        key = "SearchEngines.serper_search_api_key"
        assert current.draft.values[key] is None
        assert current.draft.originals[key] == "saved-secret"
        config.save_setting_to_cli_config(
            "SearchEngines", "serper_search_api_key", "changed-elsewhere"
        )
        await current.save()
        assert (
            tomllib.loads(path.read_text())["SearchEngines"]["serper_search_api_key"]
            == "changed-elsewhere"
        )
        assert current.draft.values[key] is None and current.draft.is_dirty
        assert "changed elsewhere" in current.save_status


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["clear-serper_search_api_key", "revert"])
async def test_explicit_clear_or_revert_wins_over_pending_input(setup, action):
    _, path = setup
    original = path.read_text()
    host = SearchSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "web-search")
        screen = host.screen
        panel = screen.query_one(WebSearchSettingsPanel)
        field = panel.query_one("#web-search-serper_search_api_key", Input)
        with panel.prevent(Input.Changed):
            field.value = "pending-replacement"
        await panel.button_pressed(
            Button.Pressed(panel.query_one(f"#web-search-{action}", Button))
        )
        if action == "revert":
            await pilot.pause()
            assert host.screen.query("#confirm-button")
            await pilot.click("#confirm-button")
            await host.workers.wait_for_complete()
        await pilot.pause()
        model = screen._web_search_model()
        if action == "revert":
            assert not model.draft.is_dirty
        else:
            assert model.draft.values["SearchEngines.serper_search_api_key"] is None
        assert panel.query_one("#web-search-serper_search_api_key", Input).value == ""
        assert path.read_text() == original


@pytest.mark.asyncio
async def test_committed_write_with_reload_failure_reports_saved_to_disk(
    setup, monkeypatch
):
    model, path = setup
    original = config.apply_settings_mutation_to_cli_config

    def save(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result.file_replaced

        def fail_reload(*args, **kwargs):
            raise OSError("synthetic-sensitive-error")

        monkeypatch.setattr(config, "load_cli_config_and_ensure_existence", fail_reload)
        return config.ConfigMutationResult(True, False, "cache_reload")

    monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", save)
    model.edit("serper_search_api_key", "replacement-key")
    await model.save()
    assert (
        tomllib.loads(path.read_text())["SearchEngines"]["serper_search_api_key"]
        == "replacement-key"
    )
    assert not model.draft.is_dirty
    assert "Saved to disk" in model.save_status
    assert "Restart" in model.save_status
    assert "synthetic-sensitive-error" not in model.save_status
    model.edit("serper_search_api_key", "next-replacement")
    assert (
        model.draft.originals["SearchEngines.serper_search_api_key"]
        == "replacement-key"
    )
