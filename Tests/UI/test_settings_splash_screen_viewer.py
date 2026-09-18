from unittest.mock import patch

import pytest
from textual.app import App
from textual.css.query import NoMatches
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static

from Tests.private_profile import private_profile_test
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.Widgets import settings_splash_screen_viewer as splash_module
from tldw_chatbook.Widgets.settings_splash_screen_viewer import (
    DEFAULT_SPLASH_CONFIG,
    SettingsSplashScreenViewer,
)


def _mutation_writer(write):
    """Adapt existing single-preference spies to the structured owner API."""

    def mutation(sections):
        [(section, values)] = sections.items()
        [(key, value)] = values.items()
        succeeded = write(section, key, value) is not False
        return ConfigMutationResult(
            succeeded, succeeded, None if succeeded else "before_replace"
        )

    return mutation


class _SplashTestApp(App[None]):
    """Minimal app for testing the splash settings viewer in isolation."""

    CSS = """
    Screen { align: center middle; }
    """

    def compose(self):
        yield SettingsSplashScreenViewer()


@pytest.fixture
def splash_app():
    return _SplashTestApp()


@pytest.mark.asyncio
@private_profile_test
async def test_settings_splash_viewer_can_compose(request, splash_app):
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        assert viewer.is_mounted


@pytest.mark.asyncio
@private_profile_test
async def test_settings_splash_viewer_loads_defaults(request, splash_app):
    """Test that viewer loads defaults when config has no configured values."""

    def fake_get_cli_setting(section, key=None, default=None):
        # Return the default parameter to simulate no configured values
        # This tests the fallback behavior
        return default

    with patch(
        "tldw_chatbook.Widgets.settings_splash_screen_viewer.get_cli_setting",
        side_effect=fake_get_cli_setting,
    ):
        # Recreate the app with patched get_cli_setting
        splash_app = _SplashTestApp()
        async with splash_app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            viewer = splash_app.query_one(SettingsSplashScreenViewer)

            enabled = viewer.query_one("#settings-splash-enabled", Checkbox)
            assert enabled.value == DEFAULT_SPLASH_CONFIG["enabled"]

            duration = viewer.query_one("#settings-splash-duration", Input)
            assert float(duration.value) == DEFAULT_SPLASH_CONFIG["duration"]


@pytest.mark.asyncio
@private_profile_test
async def test_settings_splash_viewer_card_list_populated(request, splash_app):
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        card_list = viewer.query_one("#settings-splash-card-list", OptionList)
        assert len(card_list.options) > 0


@pytest.mark.asyncio
@private_profile_test
async def test_settings_splash_viewer_default_select_contains_random(
    request, splash_app
):
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        select = viewer.query_one("#settings-splash-default-select", Select)
        values = [str(option[1]) for option in select._options]
        assert "random" in values


@pytest.mark.asyncio
@private_profile_test
async def test_settings_splash_viewer_selection_triggers_preview(request, splash_app):
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        card_list = viewer.query_one("#settings-splash-card-list", OptionList)
        # Highlight the first real card
        card_list.highlighted = 0
        await pilot.pause()

        container = viewer.query_one("#settings-splash-preview-scroll")
        assert len(container.children) > 0


# ---- task-1561: text-labeled toggle states ----


@pytest.mark.asyncio
@private_profile_test
async def test_splash_checkboxes_carry_text_state_labels(request, splash_app):
    """Each toggle row shows an On/Off Static that flips with the toggle.

    The checkbox alone carries state visually, which is
    unreadable in reduced-color terminals (task-1561).
    """
    from textual.widgets import Static

    from tldw_chatbook.Widgets.settings_splash_screen_viewer import (
        switch_state_label,
    )

    assert switch_state_label(True) == "On"
    assert switch_state_label(False) == "Off"

    with patch(
        "tldw_chatbook.Widgets.settings_splash_screen_viewer.apply_settings_mutation_to_cli_config",
        return_value=ConfigMutationResult(True, True, None),
    ):
        async with splash_app.run_test(size=(120, 50)) as pilot:
            checkbox = pilot.app.query_one("#settings-splash-enabled", Checkbox)
            state = pilot.app.query_one("#settings-splash-enabled-state", Static)
            initial = bool(checkbox.value)
            assert str(state.renderable) == switch_state_label(initial)

            checkbox.toggle()
            await pilot.pause()

            assert str(state.renderable) == switch_state_label(not initial)


@pytest.mark.asyncio
@private_profile_test
async def test_settings_splash_viewer_has_single_default_card_control(
    request, splash_app
):
    """task-1376: the 'Default card' Select is the one control that sets the
    default splash card; the duplicate 'Set as default' button is gone."""
    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        select = viewer.query_one("#settings-splash-default-select", Select)
        assert select is not None

        with pytest.raises(NoMatches):
            viewer.query_one("#settings-splash-set-default", Button)

        button_labels = [str(button.label) for button in viewer.query(Button)]
        assert button_labels == ["Play selected"]


@pytest.mark.asyncio
@private_profile_test
async def test_settings_splash_viewer_default_select_persists_on_change(
    request, splash_app, monkeypatch
):
    """task-1376: the remaining control stays instant-apply: changing the
    Select saves card_selection immediately (no separate commit step)."""
    saved: list[tuple[str, str, object]] = []

    def fake_save(section: str, key: str, value: object) -> None:
        saved.append((section, key, value))

    monkeypatch.setattr(
        splash_module,
        "apply_settings_mutation_to_cli_config",
        _mutation_writer(fake_save),
    )

    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)

        select = viewer.query_one("#settings-splash-default-select", Select)
        card_values = [option[1] for option in select._options if option[1] != "random"]
        assert card_values, "expected at least one splash card option"

        target = card_values[0]
        select.value = target
        await pilot.pause()

        assert ("splash_screen", "card_selection", target) in saved
        assert viewer._config["card_selection"] == target


@pytest.mark.asyncio
@private_profile_test
async def test_failing_persist_worker_surfaces_error_without_crashing_the_app(
    request, splash_app, monkeypatch
):
    """task-15470 review round: `_persist_splash_config_value` used to call
    `self.call_from_thread(...)` on its failure path -- but `call_from_
    thread` exists only on `App`, not on this `Vertical` widget. When the
    config write actually raised, the `except` handler's own
    `self.call_from_thread` call raised a SECOND, uncaught `AttributeError`
    inside a `@work(thread=True)` worker -- fatal to the whole app by
    default (`exit_on_error=True`). Textual re-raises that fatal exception
    when `run_test()`'s context manager exits, so with the bug present this
    test fails with an `AttributeError` traceback instead of ever reaching
    the assertions below (confirmed by temporarily reverting the fix and
    re-running this exact test).

    Also pins the adjacent fix: a failed write must not leave `_config`
    diverged from what is actually on disk -- the optimistic in-memory
    update must revert.
    """

    def failing_save(section: str, key: str, value: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(
        splash_module,
        "apply_settings_mutation_to_cli_config",
        _mutation_writer(failing_save),
    )

    async with splash_app.run_test(size=(120, 50)) as pilot:
        await pilot.pause()
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        previous = viewer._config["skip_on_keypress"]

        checkbox = viewer.query_one("#settings-splash-skip-on-keypress", Checkbox)
        checkbox.value = not previous
        await pilot.pause()

        status = viewer.query_one("#settings-splash-status", Static)
        for _ in range(50):
            if "Error saving skip_on_keypress" in str(status.renderable):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"failure status never appeared; last seen: {status.renderable!r}"
            )

        # The optimistic in-memory value must not diverge from what is
        # actually on disk once the write is known to have failed.
        assert viewer._config["skip_on_keypress"] == previous

        # The app must still be alive and responsive -- not merely "the
        # exception hasn't been re-raised yet" (that only happens when the
        # `async with` block exits, below). Driving a second, independent
        # control through a full failure-and-revert cycle proves the
        # message loop, workers, and `call_from_thread` callbacks are all
        # still functioning normally after the first failure.
        other_checkbox = viewer.query_one("#settings-splash-enabled", Checkbox)
        other_previous = viewer._config["enabled"]
        other_checkbox.value = not other_previous

        for _ in range(50):
            if "Error saving enabled" in str(status.renderable):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "second control's failure status never appeared -- app "
                f"stopped responding; last seen: {status.renderable!r}"
            )
        assert viewer._config["enabled"] == other_previous


@pytest.mark.asyncio
@private_profile_test
async def test_splash_animation_speed_persists_to_the_section_it_loads(
    request, splash_app
):
    """A speed saved in Settings must survive recreating the viewer."""
    from tldw_chatbook import config

    async with splash_app.run_test(size=(120, 50)) as pilot:
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        field = viewer.query_one("#settings-splash-animation-speed", Input)
        field.value = "1.75"
        field.focus()
        await pilot.press("enter")
        await splash_app.workers.wait_for_complete()
        await pilot.pause()
        assert (
            config.get_cli_setting("splash_screen.effects", "animation_speed", None)
            == 1.75
        )
        assert SettingsSplashScreenViewer()._load_config()["animation_speed"] == 1.75


@pytest.mark.asyncio
@private_profile_test
async def test_failed_splash_toggle_restores_visible_control_and_state(
    request, splash_app, monkeypatch
):
    def fail(*args):
        raise OSError("test write failure")

    monkeypatch.setattr(
        splash_module, "apply_settings_mutation_to_cli_config", _mutation_writer(fail)
    )
    async with splash_app.run_test(size=(120, 50)) as pilot:
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        for key in ("enabled", "show_progress", "skip_on_keypress"):
            selector = f"#settings-splash-{key.replace('_', '-')}"
            control = viewer.query_one(selector, Checkbox)
            previous = control.value
            control.toggle()
            await pilot.pause()
            await splash_app.workers.wait_for_complete()
            await pilot.pause()
            assert control.value is previous
            label = viewer.query_one(f"{selector}-state", Static)
            assert str(label.renderable) == splash_module.switch_state_label(previous)
            assert f"Error saving {key}" in str(
                viewer.query_one("#settings-splash-status", Static).renderable
            )


@pytest.mark.asyncio
@private_profile_test
async def test_pending_splash_write_keeps_confirmed_state_and_gates_same_control(
    request, splash_app, monkeypatch
):
    import asyncio
    import threading

    started, release = threading.Event(), threading.Event()
    writes = []

    def delayed_save(section, key, value):
        writes.append((section, key, value))
        started.set()
        assert release.wait(5)
        return True

    monkeypatch.setattr(
        splash_module,
        "apply_settings_mutation_to_cli_config",
        _mutation_writer(delayed_save),
    )
    async with splash_app.run_test(size=(120, 50)) as pilot:
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        control = viewer.query_one("#settings-splash-enabled", Checkbox)
        previous = control.value
        control.focus()
        await pilot.pause()
        await pilot.press("space")
        try:
            assert await asyncio.to_thread(started.wait, 3)
            assert splash_app.screen.focused is control
            assert viewer._config["enabled"] == previous
            assert "Saving" in str(
                viewer.query_one("#settings-splash-status", Static).renderable
            )
            assert not viewer._save_config_value("enabled", previous)
            assert len(writes) == 1
        finally:
            release.set()
            await splash_app.workers.wait_for_complete()
        await pilot.pause()
        assert splash_app.screen.focused is control
        assert viewer._config["enabled"] is not previous


@pytest.mark.asyncio
@pytest.mark.parametrize("file_replaced", [False, True])
@private_profile_test
async def test_splash_write_reports_file_and_refresh_outcomes_separately(
    request, splash_app, monkeypatch, file_replaced
):
    import tomllib
    from pathlib import Path

    from tldw_chatbook import config

    real_write = splash_module.apply_settings_mutation_to_cli_config

    def write(sections):
        if file_replaced:
            assert real_write(sections).file_replaced
        return ConfigMutationResult(
            file_replaced, False, "cache_reload" if file_replaced else "before_replace"
        )

    monkeypatch.setattr(splash_module, "apply_settings_mutation_to_cli_config", write)
    async with splash_app.run_test(size=(120, 50)) as pilot:
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        control = viewer.query_one("#settings-splash-enabled", Checkbox)
        previous = control.value
        control.toggle()
        await pilot.pause()
        await splash_app.workers.wait_for_complete()
        await pilot.pause()
        expected = not previous if file_replaced else previous
        assert control.value is expected
        assert viewer._config["enabled"] is expected
        saved = tomllib.loads(Path(config.get_cli_config_path()).read_text())
        assert saved["splash_screen"]["enabled"] is expected
        text = str(viewer.query_one("#settings-splash-status", Static).renderable)
        assert ("refresh failed" if file_replaced else "Error saving") in text


@pytest.mark.asyncio
@private_profile_test
async def test_splash_late_completion_is_safe_after_removing_viewer(
    request, splash_app, monkeypatch
):
    import asyncio
    import threading

    started, release = threading.Event(), threading.Event()
    completed = asyncio.Event()

    def write(sections):
        started.set()
        assert release.wait(5)
        return ConfigMutationResult(True, True, None)

    monkeypatch.setattr(splash_module, "apply_settings_mutation_to_cli_config", write)
    async with splash_app.run_test(size=(120, 50)) as pilot:
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        finish = viewer._finish_persist

        def finished(*args):
            finish(*args)
            completed.set()

        monkeypatch.setattr(viewer, "_finish_persist", finished)
        viewer.query_one("#settings-splash-enabled", Checkbox).toggle()
        try:
            assert await asyncio.to_thread(started.wait, 3)
            await viewer.remove()
            assert not viewer.is_attached
        finally:
            release.set()
        await asyncio.wait_for(completed.wait(), 3)
        await pilot.pause()
        assert not splash_app.query(SettingsSplashScreenViewer)


@pytest.mark.asyncio
@private_profile_test
async def test_splash_older_write_does_not_replace_newer_status(
    request, splash_app, monkeypatch
):
    import asyncio
    import threading

    started = {key: threading.Event() for key in ("enabled", "show_progress")}
    release = {key: threading.Event() for key in started}
    completed = {key: asyncio.Event() for key in started}

    def write(sections):
        [values] = sections.values()
        [key] = values
        started[key].set()
        assert release[key].wait(5)
        success = key == "enabled"
        return ConfigMutationResult(
            success, success, None if success else "before_replace"
        )

    monkeypatch.setattr(splash_module, "apply_settings_mutation_to_cli_config", write)
    async with splash_app.run_test(size=(120, 50)) as _pilot:
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        finish = viewer._finish_persist

        def finished(key, *args):
            finish(key, *args)
            completed[key].set()

        monkeypatch.setattr(viewer, "_finish_persist", finished)
        try:
            for key, signal in started.items():
                viewer.query_one(
                    f"#settings-splash-{key.replace('_', '-')}", Checkbox
                ).toggle()
                assert await asyncio.to_thread(signal.wait, 3)
            release["enabled"].set()
            await asyncio.wait_for(completed["enabled"].wait(), 3)
            status = viewer.query_one("#settings-splash-status", Static)
            assert str(status.renderable) == "Saving show progress…"
            release["show_progress"].set()
            await asyncio.wait_for(completed["show_progress"].wait(), 3)
            assert "Error saving show_progress" in str(status.renderable)
        finally:
            for gate in release.values():
                gate.set()
            await splash_app.workers.wait_for_complete()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_save_succeeds", [True, False])
@private_profile_test
async def test_splash_pending_input_preserves_newer_text_for_next_enter(
    request, splash_app, monkeypatch, first_save_succeeds
):
    import asyncio
    import threading

    started, release = threading.Event(), threading.Event()
    writes = []

    def write(sections):
        writes.append(sections)
        if len(writes) == 1:
            started.set()
            assert release.wait(5)
        saved = first_save_succeeds or len(writes) > 1
        return ConfigMutationResult(saved, saved, None if saved else "before_replace")

    monkeypatch.setattr(splash_module, "apply_settings_mutation_to_cli_config", write)
    async with splash_app.run_test(size=(120, 50)) as pilot:
        viewer = splash_app.query_one(SettingsSplashScreenViewer)
        field = viewer.query_one("#settings-splash-duration", Input)
        previous = viewer._config["duration"]
        field.focus()
        await pilot.pause()
        await pilot.press("home", "shift+end", "backspace", *"1.5", "enter")
        try:
            assert await asyncio.to_thread(started.wait, 3)
            await pilot.press("home", "shift+end", "backspace", *"2.0", "enter")
            assert field.value == "2.0"
            assert len(writes) == 1
        finally:
            release.set()
            await splash_app.workers.wait_for_complete()
        await pilot.pause()
        assert field.value == "2.0"
        assert splash_app.screen.focused is field
        assert viewer._config["duration"] == (1.5 if first_save_succeeds else previous)
        assert "Press Enter" in str(
            viewer.query_one("#settings-splash-status", Static).renderable
        )
        await pilot.press("enter")
        await splash_app.workers.wait_for_complete()
        await pilot.pause()
        assert viewer._config["duration"] == 2.0
        assert len(writes) == 2
