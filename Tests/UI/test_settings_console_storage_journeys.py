"""Console and Storage fields remain usable under real Settings styles."""

import pytest
from textual.widgets import Button, Checkbox, Collapsible, Input, Select

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_settings_overview_search_journeys import _category, _painted
from Tests.UI.test_settings_speech_tts_panel import _StyledDestinationHarness


@pytest.mark.asyncio
@pytest.mark.timeout(180)
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(190, 55), (80, 24)])
@pytest.mark.parametrize("category", ["Console Behavior", "Storage"])
@private_profile_test
async def test_console_and_storage_controls_paint_readable_values(
    request, theme, size, category
):
    """Clipped controls or narrow value slivers prevent reliable keyboard edits."""
    from Tests.UI.test_settings_console_reasoning_history import (
        _app_with_local_console_target,
    )

    host = _StyledDestinationHarness(_app_with_local_console_target(), "settings")
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, category)
        card = host.screen.query_one(
            "#settings-console-behavior-card"
            if category == "Console Behavior"
            else "#settings-storage-card"
        )
        for group in card.query(Collapsible):
            group.collapsed = False
        await pilot.pause()
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        controls = [
            control
            for control in card.query("Button, Checkbox, Input, Select")
            if control.id and control.display and not control.disabled
        ]
        assert controls
        failures = []
        for control in controls:
            control.focus()
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            try:
                assert host.screen.focused is control
                _assert_painted(host.screen, control)
                if isinstance(control, Input):
                    assert control.content_region.width >= 12, (
                        control.id,
                        control.content_region,
                    )
                    if control.value:
                        await pilot.press("end")
                        assert control.value[-8:] in _painted(host, control), (
                            control.id,
                            control.value,
                            _painted(host, control),
                        )
                elif isinstance(control, (Button, Checkbox)):
                    painted = " ".join(_painted(host, control).split())
                    assert " ".join(str(control.label).split()) in painted, (
                        control.id,
                        painted,
                    )
                    if isinstance(control, Checkbox):
                        assert "X" in painted, (control.id, painted)
                elif isinstance(control, Select):
                    label = next(
                        str(label)
                        for label, value in control._options
                        if value == control.value
                    )
                    assert " ".join(label.split()) in " ".join(
                        _painted(host, control).split()
                    ), (control.id, label, _painted(host, control))
            except AssertionError as exc:
                failures.append(f"{control.id}: {str(exc).splitlines()[0]}")
        assert not failures, "\n".join(failures)


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["remote-images", "status-row-position"])
@private_profile_test
async def test_console_instant_toggle_failed_save_rolls_back_and_retries(
    request, monkeypatch, setting
):
    """An unsuccessful write must not leave runtime and restart values different."""
    import tomllib
    from pathlib import Path

    from Tests.UI.test_settings_provider_keyboard_journeys import _settle
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    app = _build_test_app()
    app.app_config["console"] = {"status_chips_position": "above"}
    app.app_config["COMPREHENSIVE_CONFIG_RAW"] = {
        "chat": {"images": {"render_remote_images": False}}
    }
    host = _StyledDestinationHarness(app, "settings")
    real_writer = config.apply_settings_mutation_to_cli_config

    def fail(*args, **kwargs):
        return config.ConfigMutationResult(False, False, "before_replace")

    # Both imports address the same public mutation boundary; the legacy bool
    # wrapper looks it up in config, while structured consumers import it.
    monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", fail)
    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", fail)
    path = Path(config.get_cli_config_path())
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        screen = host.screen
        button = screen.query_one(f"#settings-console-{setting}-toggle", Button)
        before = path.read_bytes()
        original_label = str(button.label)
        button.focus()
        await pilot.press("enter")
        await _settle(host, pilot)
        assert str(button.label) == original_label
        assert path.read_bytes() == before
        value = (
            screen._remote_images_enabled()
            if setting == "remote-images"
            else screen._status_row_position_value()
        )
        assert value == (False if setting == "remote-images" else "above")
        assert "Could not save" in screen._console_behavior_result
        monkeypatch.setattr(
            config, "apply_settings_mutation_to_cli_config", real_writer
        )
        monkeypatch.setattr(
            module, "apply_settings_mutation_to_cli_config", real_writer
        )
        assert host.screen.focused is button
        await pilot.press("enter")
        await _settle(host, pilot)
        saved = tomllib.loads(path.read_text())
        if setting == "remote-images":
            assert saved["chat"]["images"]["render_remote_images"] is True
            assert screen._remote_images_enabled() is True
        else:
            assert saved["console"]["status_chips_position"] == "below"
            assert screen._status_row_position_value() == "below"


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["remote-images", "status-row-position"])
@pytest.mark.parametrize("first_fails", [False, True])
@private_profile_test
async def test_console_instant_toggle_latest_choice_survives_pending_write(
    request, monkeypatch, setting, first_fails
):
    """Returning to the saved choice while a write runs cannot save stale state."""
    import asyncio
    import threading
    import tomllib
    from pathlib import Path

    from Tests.UI.test_settings_provider_keyboard_journeys import _settle
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    initial = False if setting == "remote-images" else "above"
    section, key = (
        ("chat.images", "render_remote_images")
        if setting == "remote-images"
        else ("console", "status_chips_position")
    )
    config.apply_settings_mutation_to_cli_config({section: {key: initial}})
    app = _build_test_app()
    app.app_config["console"] = {"status_chips_position": "above"}
    app.app_config["COMPREHENSIVE_CONFIG_RAW"] = {
        "chat": {"images": {"render_remote_images": False}}
    }
    host = _StyledDestinationHarness(app, "settings")
    entered, release = threading.Event(), threading.Event()
    real_writer = config.apply_settings_mutation_to_cli_config
    calls = []

    def gated(payload, *args, **kwargs):
        calls.append(payload)
        if len(calls) == 1:
            entered.set()
            assert release.wait(8)
            if first_fails:
                return config.ConfigMutationResult(False, False, "before_replace")
        return real_writer(payload, *args, **kwargs)

    monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", gated)
    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", gated)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        button = host.screen.query_one(f"#settings-console-{setting}-toggle", Button)
        button.focus()
        try:
            await pilot.press("enter")
            assert await asyncio.to_thread(entered.wait, 3)
            await pilot.press("enter")
            await pilot.press("escape", "/", *"Storage", "enter")
            await pilot.pause()
        finally:
            release.set()
        await _settle(host, pilot)
        await _category(host, pilot, "Console Behavior")
        saved = tomllib.loads(Path(config.get_cli_config_path()).read_text())
        if setting == "remote-images":
            assert saved["chat"]["images"][key] is False
            assert host.screen._remote_images_enabled() is False
        else:
            assert saved[section][key] == "above"
            assert host.screen._status_row_position_value() == "above"
        assert not host.screen._category_has_unsaved_changes(
            host.screen.active_category
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["remote-images", "status-row-position"])
@private_profile_test
async def test_console_instant_toggle_keeps_replaced_file_value_on_cache_failure(
    request, monkeypatch, setting
):
    """A post-replace refresh failure must not roll runtime back behind disk."""
    import tomllib
    from pathlib import Path

    from Tests.UI.test_settings_provider_keyboard_journeys import _settle
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    app = _build_test_app()
    app.app_config["console"] = {"status_chips_position": "above"}
    app.app_config["COMPREHENSIVE_CONFIG_RAW"] = {
        "chat": {"images": {"render_remote_images": False}}
    }
    host = _StyledDestinationHarness(app, "settings")
    real_writer = config.apply_settings_mutation_to_cli_config

    def partial(payload, *args, **kwargs):
        assert real_writer(payload, *args, **kwargs).file_replaced
        return config.ConfigMutationResult(True, False, "cache_reload")

    monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", partial)
    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", partial)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        host.screen.query_one(f"#settings-console-{setting}-toggle", Button).focus()
        await pilot.press("enter")
        await _settle(host, pilot)
        saved = tomllib.loads(Path(config.get_cli_config_path()).read_text())
        if setting == "remote-images":
            assert saved["chat"]["images"]["render_remote_images"] is True
            assert host.screen._remote_images_enabled() is True
        else:
            assert saved["console"]["status_chips_position"] == "below"
            assert host.screen._status_row_position_value() == "below"
        assert "saved" in host.screen._console_behavior_result
        assert "refresh" in host.screen._console_behavior_result


async def _replace_field(host, pilot, selector, value):
    field = host.screen.query_one(selector, Input)
    field.focus()
    await pilot.wait_for_scheduled_animations()
    _assert_painted(host.screen, field)
    await pilot.press("home", "shift+end", "backspace", *value)
    await pilot.pause()
    assert field.value == value
    return field


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["Console Behavior", "Storage"])
@pytest.mark.parametrize("size", [(190, 55), (80, 24)])
@private_profile_test
async def test_console_and_storage_staged_keyboard_recovery(
    request, tmp_path, monkeypatch, category, size
):
    """Drafts survive navigation/failure; Storage saves cannot reconnect live DBs."""
    import tomllib
    from pathlib import Path

    from Tests.UI.test_settings_provider_keyboard_journeys import _revert, _settle
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    app = _build_test_app()
    app.app_config.setdefault("console", {})["paste_collapse_threshold"] = 50
    handles = {
        key: value
        for key, value in vars(app).items()
        if key.endswith("_db") and value is not None
    }
    assert handles
    host = _StyledDestinationHarness(app, "settings")
    path = Path(config.get_cli_config_path())
    storage = category == "Storage"
    selector = (
        "#settings-storage-workspaces-db-path"
        if storage
        else "#settings-console-paste-collapse-threshold"
    )
    target = tmp_path / "next-launch" / "workspaces.db"
    value = str(target) if storage else "120"
    section, key = (
        ("database", "workspaces_db_path")
        if storage
        else ("console", "paste_collapse_threshold")
    )
    async with host.run_test(size=size) as pilot:
        await _category(host, pilot, category)
        screen = host.screen
        original = screen.query_one(selector, Input).value
        active = app.app_config[section].get(key)
        before = path.read_bytes()
        await _replace_field(host, pilot, selector, "../outside.db" if storage else "")
        if storage:
            assert screen.query_one("#settings-save-category", Button).disabled
        else:
            await pilot.press("escape", "s")
            await _settle(host, pilot)
            assert "must be a whole number" in screen._console_behavior_result
        assert path.read_bytes() == before
        await _replace_field(host, pilot, selector, value)
        assert not screen.query_one("#settings-save-category", Button).disabled
        assert app.app_config[section].get(key) == active
        assert path.read_bytes() == before
        if storage:
            screen.query_one("#settings-check-storage", Button).focus()
            await pilot.press("enter")
            await _settle(host, pilot)
            assert not target.parent.exists()
            assert path.read_bytes() == before
        await _category(host, pilot, "Overview")
        await _category(host, pilot, category)
        assert screen.query_one(selector, Input).value == value
        await _revert(host, pilot, discard=False)
        assert screen.query_one(selector, Input).value == value
        await _revert(host, pilot, discard=True)
        assert screen.query_one(selector, Input).value == original
        assert path.read_bytes() == before
        await _replace_field(host, pilot, selector, value)
        real_writer = config.apply_settings_mutation_to_cli_config

        def fail(*args, **kwargs):
            return config.ConfigMutationResult(False, False, "before_replace")

        monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", fail)
        monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", fail)
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert screen._category_has_unsaved_changes(screen.active_category)
        assert "Failed" in (
            screen._storage_result if storage else screen._console_behavior_result
        )
        assert screen.query_one(selector, Input).value == value
        assert path.read_bytes() == before
        assert app.app_config[section].get(key) == active
        monkeypatch.setattr(
            config, "apply_settings_mutation_to_cli_config", real_writer
        )
        monkeypatch.setattr(
            module, "apply_settings_mutation_to_cli_config", real_writer
        )
        await pilot.press("escape", "s")
        await _settle(host, pilot)
        assert not screen._category_has_unsaved_changes(screen.active_category)
        expected = value if storage else int(value)
        assert tomllib.loads(path.read_text())[section][key] == expected
        assert app.app_config[section][key] == expected
        assert all(getattr(app, key) is handle for key, handle in handles.items())
        assert not target.parent.exists()
        if storage:
            assert "Restart Chatbook" in screen._storage_result
        await _category(host, pilot, "Overview")
        await _category(host, pilot, category)
        assert screen.query_one(selector, Input).value == value


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["remote-images", "status-row-position"])
@private_profile_test
async def test_console_instant_toggle_rebases_after_configuration_reload(
    request, setting
):
    import tomllib
    from pathlib import Path

    from Tests.UI.test_settings_provider_keyboard_journeys import _settle
    from tldw_chatbook import config

    section, key, original, enabled = (
        ("chat.images", "render_remote_images", False, True)
        if setting == "remote-images"
        else ("console", "status_chips_position", "above", "below")
    )
    config.apply_settings_mutation_to_cli_config({section: {key: original}})
    app = _build_test_app()
    host = _StyledDestinationHarness(app, "settings")
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        host.screen.query_one(f"#settings-console-{setting}-toggle", Button).focus()
        await pilot.press("enter")
        await _settle(host, pilot)
        config.apply_settings_mutation_to_cli_config({section: {key: original}})
        assert host.screen._reload_current_config() == "Config reload: loaded"
        await _category(host, pilot, "Overview")
        await _category(host, pilot, "Console Behavior")
        host.screen.query_one(f"#settings-console-{setting}-toggle", Button).focus()
        await pilot.press("enter")
        await _settle(host, pilot)
        saved = tomllib.loads(Path(config.get_cli_config_path()).read_text())
        table = saved["chat"]["images"] if section == "chat.images" else saved[section]
        assert table[key] == enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", ["remote-images", "status-row-position"])
@pytest.mark.parametrize("edit_after_return", [False, True])
@pytest.mark.parametrize("cache_failure", [False, True])
@private_profile_test
async def test_console_instant_toggle_drains_across_settings_recreation(
    request, monkeypatch, setting, edit_after_return, cache_failure
):
    import asyncio
    import threading
    import tomllib
    from pathlib import Path

    from textual.screen import Screen

    from Tests.UI.test_settings_provider_keyboard_journeys import _settle
    from tldw_chatbook import config
    from tldw_chatbook.UI.Screens import settings_screen as module

    section, key, original, enabled = (
        ("chat.images", "render_remote_images", False, True)
        if setting == "remote-images"
        else ("console", "status_chips_position", "above", "below")
    )
    config.apply_settings_mutation_to_cli_config({section: {key: original}})
    app = _build_test_app()
    host = _StyledDestinationHarness(app, "settings")
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    real_writer = config.apply_settings_mutation_to_cli_config
    calls = []

    def gated(payload, *args, **kwargs):
        calls.append(payload)
        if len(calls) == 1:
            entered.set()
            assert release.wait(15)
        result = real_writer(payload, *args, **kwargs)
        finished.set()
        if cache_failure and len(calls) == 1:
            return config.ConfigMutationResult(True, False, "cache_reload")
        return result

    monkeypatch.setattr(config, "apply_settings_mutation_to_cli_config", gated)
    monkeypatch.setattr(module, "apply_settings_mutation_to_cli_config", gated)
    async with host.run_test(size=(190, 55)) as pilot:
        await _category(host, pilot, "Console Behavior")
        host.screen.query_one(f"#settings-console-{setting}-toggle", Button).focus()
        try:
            await pilot.press("enter")
            assert await asyncio.to_thread(entered.wait, 3)
            await pilot.press("enter")
            old_screen = host.screen
            await host.switch_screen(Screen())
            await pilot.pause()
            assert not old_screen.is_attached
            await host.switch_screen(module.SettingsScreen(app))
            await pilot.pause()
            # Do not await all workers until the admitted write is released.
            await pilot.press("escape", "/", *"Console Behavior", "enter")
            await pilot.pause()
            if edit_after_return:
                host.screen._console_behavior_result = "Could not save earlier choice."
                host.screen.query_one(
                    f"#settings-console-{setting}-toggle", Button
                ).focus()
                await pilot.press("enter")
        finally:
            release.set()
        assert await asyncio.to_thread(finished.wait, 5)
        await _settle(host, pilot)
        saved = tomllib.loads(Path(config.get_cli_config_path()).read_text())
        expected = enabled if edit_after_return else original
        table = saved["chat"]["images"] if section == "chat.images" else saved[section]
        assert table[key] == expected
        runtime = (
            host.screen._remote_images_enabled()
            if setting == "remote-images"
            else host.screen._status_row_position_value()
        )
        assert runtime == expected
        assert "saved" in host.screen._console_behavior_result
        assert "Could not save" not in host.screen._console_behavior_result
        if cache_failure and edit_after_return:
            assert "refresh" in host.screen._console_behavior_result
