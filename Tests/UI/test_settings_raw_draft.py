"""Raw TOML recovery through the mounted canonical Settings screen."""

import asyncio
import threading

import pytest
from textual.widgets import TextArea

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from Tests.UI.test_destination_shells import DestinationHarness, _static_text
from Tests.UI.test_settings_category_sweep import (
    _build_test_app,
    _click_settings_category,
    _settle_settings,
)
from tldw_chatbook import config
from tldw_chatbook.UI.Screens.settings_advanced_config import AdvancedConfigSettings
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog


class RawSettingsHarness(DestinationHarness):
    CSS_PATH = str(BUNDLED_STYLESHEET)


@pytest.fixture
def raw_profile(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    path.write_text('[SearchSettings]\nsearch_provider_default = "serper"\n')
    path.with_suffix(".toml.bak").write_text(
        '[SearchSettings]\nsearch_provider_default = "brave"\n'
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(path))
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
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ['[broken\napi_key = "synthetic-key"', ""])
async def test_invalid_and_empty_raw_drafts_survive_navigation(raw_profile, text):
    host = RawSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        screen = host.screen
        screen.query_one("#settings-advanced-config-editor", TextArea).text = text
        await pilot.pause()
        await _click_settings_category(pilot, "overview")
        await _click_settings_category(pilot, "advanced-config")
        assert (
            screen.query_one("#settings-advanced-config-editor", TextArea).text == text
        )
        assert screen._category_has_unsaved_changes(SettingsCategoryId.ADVANCED_CONFIG)
        assert "Unsaved" in _static_text(screen.query_one(".settings-state-banner"))
        assert "Save (s)" not in _static_text(
            screen.query_one(".settings-state-banner")
        )
        assert screen.query_one("#settings-advanced-save-config").disabled


@pytest.mark.asyncio
async def test_raw_draft_survives_settings_destination_recreation(raw_profile):
    host = RawSettingsHarness(_build_test_app(), "settings")
    text = '# unfinished edit\n[SearchSettings]\nsearch_provider_default = "exa"\n'
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        host.screen.query_one("#settings-advanced-config-editor", TextArea).text = text
        await pilot.pause()
        state = host.screen.save_state()
    restored = RawSettingsHarness(_build_test_app(), "settings", restored_state=state)
    async with restored.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        assert (
            restored.screen.query_one("#settings-advanced-config-editor", TextArea).text
            == text
        )
        assert restored.screen._category_has_unsaved_changes(
            SettingsCategoryId.ADVANCED_CONFIG
        )


@pytest.mark.asyncio
async def test_load_backup_requires_confirmation_before_replacing_draft(raw_profile):
    original = raw_profile.read_text()
    host = RawSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        screen = host.screen
        editor = screen.query_one("#settings-advanced-config-editor", TextArea)
        editor.text = "# valuable draft\n[SearchSettings]\n"
        await pilot.pause()
        await pilot.click("#settings-advanced-load-backup")
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert editor.text.startswith("# valuable draft")
        assert raw_profile.read_text() == original


@pytest.mark.asyncio
async def test_validation_result_does_not_validate_newer_text(raw_profile):
    model = AdvancedConfigSettings(lambda: None, lambda loaded: None)
    await model.inspect_current()
    original = model.adapter.validate_raw_toml
    started, release = threading.Event(), threading.Event()

    def validate(text):
        started.set()
        release.wait(3)
        return original(text)

    model.adapter.validate_raw_toml = validate
    task = asyncio.create_task(model.validate())
    assert await asyncio.to_thread(started.wait, 2)
    model.edit("[broken")
    release.set()
    await task
    assert not model.can_save
    assert "Ready to save" not in model.result


@pytest.mark.asyncio
async def test_save_keeps_edits_made_while_write_is_running(raw_profile):
    model = AdvancedConfigSettings(lambda: None, lambda loaded: None)
    await model.inspect_current()
    submitted = '[SearchSettings]\nsearch_provider_default = "brave"\n'
    newer = submitted + "# later edit\n"
    model.edit(submitted)
    await model.validate()
    original = model.adapter.replace_snapshot
    started, release = threading.Event(), threading.Event()

    def save(text, snapshot):
        started.set()
        release.wait(3)
        return original(text, snapshot)

    model.adapter.replace_snapshot = save
    task = asyncio.create_task(model.save())
    assert await asyncio.to_thread(started.wait, 2)
    model.edit(newer)
    release.set()
    await task
    assert raw_profile.read_text() == submitted
    assert model.state.text == newer and model.state.is_dirty
    assert not model.can_save
    assert "Newer edits remain unsaved" in model.result


@pytest.mark.asyncio
async def test_delayed_backup_never_replaces_newer_draft(raw_profile):
    model = AdvancedConfigSettings(lambda: None, lambda loaded: None)
    await model.inspect_current()
    original = model.adapter.read_backup_serialized
    started, release = threading.Event(), threading.Event()

    def read():
        started.set()
        release.wait(3)
        return original()

    model.adapter.read_backup_serialized = read
    task = asyncio.create_task(model.replace_draft("backup", model.state.revision))
    assert await asyncio.to_thread(started.wait, 2)
    model.edit("# newest draft")
    release.set()
    await task
    assert model.state.text == "# newest draft"
    assert "Newer edits kept" in model.result


@pytest.mark.asyncio
async def test_external_edit_blocks_raw_save_and_preserves_backup(raw_profile):
    model = AdvancedConfigSettings(lambda: None, lambda loaded: None)
    await model.inspect_current()
    model.edit('[SearchSettings]\nsearch_provider_default = "brave"\n')
    await model.validate()
    backup = raw_profile.with_suffix(".toml.bak").read_text()
    config.save_setting_to_cli_config(
        "SearchSettings", "search_provider_default", "exa"
    )
    external = raw_profile.read_text()
    await model.inspect_current()
    assert not model.can_save and model.state.file_changed
    await model.save()
    assert raw_profile.read_text() == external
    assert raw_profile.with_suffix(".toml.bak").read_text() == backup
    assert model.state.is_dirty and "brave" in model.state.text


@pytest.mark.asyncio
async def test_save_failure_keeps_raw_work_and_safe_diagnostic(raw_profile):
    model = AdvancedConfigSettings(lambda: None, lambda loaded: None)
    await model.inspect_current()
    model.edit("# credential-draft\n[SearchSettings]\n")
    await model.validate()

    def fail(*args):
        raise OSError("synthetic-sensitive-content")

    model.adapter.replace_snapshot = fail
    await model.save()
    assert model.state.is_dirty and "credential-draft" in model.state.text
    assert "synthetic-sensitive-content" not in model.status


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["snapshot", "runtime", "view"])
@pytest.mark.parametrize("newer_edit", [False, True])
async def test_committed_raw_save_survives_refresh_failure(
    raw_profile, monkeypatch, failure_phase, newer_edit
):
    original_text = raw_profile.read_text()
    submitted = '[SearchSettings]\nsearch_provider_default = "brave"\n'
    newer = submitted + "# later edit\n"

    def fail_refresh(*args, **kwargs):
        raise RuntimeError("synthetic-sensitive-content")

    model = AdvancedConfigSettings(
        lambda: None, fail_refresh if failure_phase == "view" else lambda loaded: None
    )
    await model.inspect_current()
    model.edit(submitted)
    await model.validate()
    if failure_phase == "runtime":
        monkeypatch.setattr(config, "_publish_runtime_config_unlocked", fail_refresh)
    elif failure_phase == "snapshot":
        read = config._try_read_cli_config_serialized_unlocked

        def fail_post_write_read(path):
            serialized = read(path)
            if serialized == submitted:
                fail_refresh()
            return serialized

        monkeypatch.setattr(
            config, "_try_read_cli_config_serialized_unlocked", fail_post_write_read
        )

    replace = model.adapter.replace_snapshot
    started, release = threading.Event(), threading.Event()

    def replace_then_wait(text, snapshot):
        try:
            return replace(text, snapshot)
        finally:
            started.set()
            release.wait(3)

    model.adapter.replace_snapshot = replace_then_wait
    task = asyncio.create_task(model.save())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert raw_profile.read_text() == submitted
        if newer_edit:
            model.edit(newer)
    finally:
        release.set()
        await task

    assert raw_profile.with_suffix(".toml.bak").read_text() == original_text
    assert model.state.baseline_text == submitted
    assert model.state.text == (newer if newer_edit else submitted)
    assert model.state.is_dirty is newer_edit
    assert model.status.startswith("Saved to disk")
    assert "restart" in model.status.lower()
    assert "synthetic-sensitive-content" not in model.status
    if newer_edit:
        assert not model.can_save
        assert "Newer edits remain unsaved" in model.status
    if failure_phase == "snapshot":
        assert model.state.snapshot is None
        assert not model.can_save
        assert "Revert" in model.status
    else:
        assert model.state.snapshot.serialized == submitted


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_edit", [False, True])
async def test_missing_committed_snapshot_requires_explicit_reload_after_navigation(
    raw_profile, monkeypatch, newer_edit
):
    submitted = '[SearchSettings]\nsearch_provider_default = "brave"\n'
    external = '[SearchSettings]\nsearch_provider_default = "exa"\n'
    newer = submitted + "# valuable newer edit\n"
    host = RawSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        screen = host.screen
        model = screen._raw_config_model()
        screen.query_one(TextArea).text = submitted
        await model.validate()
        read = config._try_read_cli_config_serialized_unlocked

        def fail_post_write_read(path):
            serialized = read(path)
            if serialized == submitted:
                raise OSError("synthetic post-write read failure")
            return serialized

        with monkeypatch.context() as fault:
            fault.setattr(
                config, "_try_read_cli_config_serialized_unlocked", fail_post_write_read
            )
            await model.save()
        assert raw_profile.read_text() == submitted
        assert model.state.snapshot is None
        if newer_edit:
            screen.query_one(TextArea).text = newer
        await _click_settings_category(pilot, "overview")
        raw_profile.write_text(external)
        await _click_settings_category(pilot, "advanced-config")
        await host.workers.wait_for_complete()
        await model.validate()
        assert model.state.snapshot is None
        assert not model.can_save
        assert screen.query_one("#settings-advanced-save-config").disabled
        assert "Revert" in model.status
        await model.save()
        assert raw_profile.read_text() == external
        assert screen.query_one(TextArea).text == (newer if newer_edit else submitted)

        await pilot.click("#settings-advanced-revert-config")
        if isinstance(host.screen, ConfirmationDialog):
            await pilot.click("#confirm-button")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert model.state.snapshot.serialized == external
        assert screen.query_one(TextArea).text == external
        await model.validate()
        assert model.can_save


@pytest.mark.asyncio
async def test_revert_cancel_keeps_draft_then_confirm_reloads_latest(raw_profile):
    host = RawSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(180, 50)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        screen = host.screen
        editor = screen.query_one(TextArea)
        editor.text = "# keep until confirmed"
        await pilot.pause()
        await pilot.click("#settings-advanced-revert-config")
        await pilot.click("#cancel-button")
        await pilot.pause()
        assert editor.text == "# keep until confirmed"
        config.save_setting_to_cli_config(
            "SearchSettings", "search_provider_default", "tavily"
        )
        await pilot.click("#settings-advanced-revert-config")
        await pilot.click("#confirm-button")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert "tavily" in editor.text
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.ADVANCED_CONFIG
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 35), (80, 24)])
async def test_raw_controls_fit_and_keyboard_save_revert_work(raw_profile, size):
    import os
    from pathlib import Path

    from textual.widgets import Button

    host = RawSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=size) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        screen = host.screen
        editor = screen.query_one(TextArea)
        for button in screen.query("#settings-advanced-config-actions Button"):
            assert button.region.width >= 16
            assert 0 <= button.region.y < size[1] - 1
        assert editor.content_region.height >= 3
        if capture_dir := os.environ.get("RAW_UX_CAPTURE_DIR"):
            folder = Path(capture_dir)
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"raw-{size[0]}x{size[1]}.svg").write_text(
                host.export_screenshot()
            )
        editor.text = "#"
        editor.focus()
        editor.move_cursor((0, 1))
        await pilot.press("r")
        await pilot.pause()
        assert host.screen is screen and editor.text == "#r"
        screen.query_one("#settings-advanced-validate-config", Button).focus()
        await pilot.press("enter")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert not screen.query_one("#settings-advanced-save-config", Button).disabled
        await pilot.press("tab")
        assert host.focused.id == "settings-advanced-save-config"
        await pilot.press("enter")
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.ADVANCED_CONFIG
        )
        assert (
            raw_profile.read_text() == ""
        )  # TOML owner normalizes a comment-only table.
        editor.text = "# newer work"
        await pilot.pause()
        screen.query_one("#settings-advanced-validate-config", Button).focus()
        await pilot.press("r")
        assert isinstance(host.screen, ConfirmationDialog)
        await pilot.press("escape")
        await pilot.pause()
        assert editor.text == "# newer work"
        if capture_dir:
            (folder / f"raw-{size[0]}x{size[1]}-draft.svg").write_text(
                host.export_screenshot()
            )


@pytest.mark.asyncio
async def test_initial_config_read_does_not_block_ui_thread(raw_profile, monkeypatch):
    from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter

    main_thread = threading.get_ident()
    original = SettingsConfigAdapter.read_snapshot
    started, release = threading.Event(), threading.Event()
    calls = []

    def read(adapter):
        calls.append(threading.get_ident())
        started.set()
        release.wait(3)
        return original(adapter)

    monkeypatch.setattr(SettingsConfigAdapter, "read_snapshot", read)
    model = AdvancedConfigSettings(lambda: None, lambda loaded: None)
    assert not calls, "Constructing the panel must not read or lock the config file"
    task = asyncio.create_task(model.inspect_current())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        assert calls == [calls[0]] and calls[0] != main_thread
        model.edit("# work arriving during initial load")
    finally:
        release.set()
    await task
    assert model.state.snapshot is not None
    assert model.state.text == "# work arriving during initial load"
    assert model.state.is_dirty and not model.state.file_changed


@pytest.mark.asyncio
async def test_save_completion_on_another_category_keeps_its_banner(raw_profile):
    host = RawSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        screen = host.screen
        model = screen._raw_config_model()
        submitted = '[SearchSettings]\nsearch_provider_default = "brave"\n'
        model.edit(submitted)
        await model.validate()
        original = model.adapter.replace_snapshot
        started, release = threading.Event(), threading.Event()

        def save(text, snapshot):
            started.set()
            release.wait(4)
            return original(text, snapshot)

        model.adapter.replace_snapshot = save
        task = asyncio.create_task(model.save())
        try:
            assert await asyncio.to_thread(started.wait, 2)
            screen.query_one("#settings-category-overview").focus()
            await pilot.press("enter")
            await pilot.pause()
            assert screen.active_category == "overview"
            banner = _static_text(screen.query_one(".settings-state-banner"))
        finally:
            release.set()
        await task
        await pilot.pause()
        assert _static_text(screen.query_one(".settings-state-banner")) == banner
        await _click_settings_category(pilot, "advanced-config")
        assert "brave" in screen.query_one(TextArea).text
        assert not screen._category_has_unsaved_changes(
            SettingsCategoryId.ADVANCED_CONFIG
        )


@pytest.mark.asyncio
async def test_save_finishes_after_settings_destination_is_recreated(raw_profile):
    from textual.screen import Screen

    from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen

    app = _build_test_app()
    host = RawSettingsHarness(app, "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        screen = host.screen
        model = screen._raw_config_model()
        model.edit('[SearchSettings]\nsearch_provider_default = "brave"\n')
        await model.validate()
        original = model.adapter.replace_snapshot
        started, release = threading.Event(), threading.Event()

        def save(text, snapshot):
            started.set()
            release.wait(4)
            return original(text, snapshot)

        model.adapter.replace_snapshot = save
        await pilot.click("#settings-advanced-save-config")
        assert await asyncio.to_thread(started.wait, 2)
        state = screen.save_state()
        restored = SettingsScreen(app)
        restored.restore_state(state)
        try:
            await host.switch_screen(Screen())
            await host.switch_screen(restored)
            await pilot.pause()
        finally:
            release.set()
        await host.workers.wait_for_complete()
        await pilot.pause()
        assert app.app_config["SearchSettings"]["search_provider_default"] == "brave"
        assert not restored._category_has_unsaved_changes(
            SettingsCategoryId.ADVANCED_CONFIG
        )
        assert not restored._raw_config_model().state.file_changed


@pytest.mark.asyncio
async def test_status_refresh_cannot_erase_a_queued_keystroke(raw_profile):
    host = RawSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        model = host.screen._raw_config_model()
        editor = host.screen.query_one(TextArea)
        editor.focus()
        await pilot.pause()
        original = editor.post_message
        armed = True

        def post(message):
            nonlocal armed
            if armed and isinstance(message, TextArea.Changed):
                armed = False
                asyncio.get_running_loop().call_soon(model._emit)
            return original(message)

        editor.post_message = post
        before = editor.text
        await pilot.press("x")
        await pilot.pause()
        assert editor.text == "x" + before
        assert model.state.text == editor.text and model.state.is_dirty


@pytest.mark.asyncio
async def test_old_inspect_failure_cannot_invalidate_successful_revert(raw_profile):
    model = AdvancedConfigSettings(lambda: None, lambda loaded: None)
    await model.inspect_current()
    original = model.adapter.read_snapshot
    started, release = threading.Event(), threading.Event()

    def read():
        if threading.current_thread().name and not started.is_set():
            started.set()
            release.wait(3)
            raise OSError("old read failure")
        return original()

    model.adapter.read_snapshot = read
    task = asyncio.create_task(model.inspect_current())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        await model.replace_draft("revert", model.state.revision)
    finally:
        release.set()
    await task
    assert not model.state.file_changed
    assert "Reloaded current config" in model.status


@pytest.mark.asyncio
async def test_backup_actions_serialize_and_repeated_loads_use_current_backup(
    raw_profile, monkeypatch
):
    started, release = threading.Event(), threading.Event()
    host = RawSettingsHarness(_build_test_app(), "settings")
    async with host.run_test(size=(120, 35)) as pilot:
        await _settle_settings(pilot)
        await _click_settings_category(pilot, "advanced-config")
        model = host.screen._raw_config_model()
        original = model.adapter.read_backup_serialized
        calls = []

        def read():
            calls.append(True)
            started.set()
            assert release.wait(4)
            return original()

        monkeypatch.setattr(model.adapter, "read_backup_serialized", read)
        button = host.screen.query_one("#settings-advanced-load-backup")
        button.focus()
        await pilot.press("enter")
        try:
            assert await asyncio.to_thread(started.wait, 2)
            assert button.disabled
            await model.replace_draft("backup", model.state.revision)
            assert len(calls) == 1
        finally:
            release.set()
        await host.workers.wait_for_complete()
        assert "brave" in model.state.text
        assert "Backup loaded" in model.result
        raw_profile.with_suffix(".toml.bak").write_text(
            '[SearchSettings]\nsearch_provider_default = "exa"\n'
        )
        await pilot.pause()
        from tldw_chatbook.Widgets.settings_advanced_config_panel import (
            AdvancedConfigPanel,
        )

        assert model.state.is_dirty and not model.busy
        host.screen.query_one(AdvancedConfigPanel).request_replacement("backup")
        await pilot.pause()
        await pilot.click("#confirm-button")
        await host.workers.wait_for_complete()
        assert len(calls) == 2
        assert "exa" in model.state.text
        assert "Backup loaded" in model.result
        assert model.state.validated_revision is None
