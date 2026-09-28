"""TASK-33078 / TASK-33121: theme file actions and Use keep the UI thread free.

With 50 saved themes each editor file action resolved names through a
backup-scoped folder scan on the UI thread (measured 0.75-2.5 s of event-loop
stall per action here), and Use wrote the launch default (~140 ms) there too.
"""

import asyncio
import gc
import threading
import time
from types import SimpleNamespace

import pytest

from Tests.private_profile import private_profile_test
from Tests.UI.test_settings_theme_picker_screen import _highlight, _host, _saved_theme
from tldw_chatbook.css.Themes import theme_catalog as tc
from tldw_chatbook.Widgets.settings_theme_editor import SettingsThemeEditor

#: The editor's I/O steps, each recorded with the thread it ran on.
_IO_STEPS = ("_scan_theme_files", "_write_toml", "_unlink", "_read_toml", "_write_export_file", "_parse_import")


class _StallMeter:
    """The longest gap between event-loop turns while it runs -- how long
    the UI thread was blocked at most.

    The cyclic GC is paused while it runs: a full collection of the
    Settings screen's object graph stalls the loop for 50-600 ms at random
    points (sampled: ``_weakrefset._remove`` inside the test's own
    ``pilot.pause``), which is not the action's cost.
    """

    def __init__(self) -> None:
        self.worst_ms = 0.0

    async def _beat(self) -> None:
        last = self._start
        while True:
            await asyncio.sleep(0.002)
            now = time.perf_counter()
            self.worst_ms = max(self.worst_ms, (now - last) * 1000)
            last = now

    def __enter__(self) -> "_StallMeter":
        gc.collect()
        gc.disable()
        self._start = time.perf_counter()
        self._task = asyncio.ensure_future(self._beat())
        return self

    def __exit__(self, *exc) -> None:
        self._task.cancel()
        gc.enable()


def _record_io_threads(monkeypatch, delay: float = 0.0) -> list[tuple[str, threading.Thread]]:
    """Record (step, thread) for each editor I/O step; ``delay`` slows each."""
    calls: list[tuple[str, threading.Thread]] = []
    for step in _IO_STEPS:
        real = getattr(SettingsThemeEditor, step)

        def recorded(self, *args, _real=real, _step=step):
            calls.append((_step, threading.current_thread()))
            if delay:
                time.sleep(delay)
            return _real(self, *args)

        monkeypatch.setattr(SettingsThemeEditor, step, recorded)
    return calls


async def _settle(host, pilot) -> None:
    for _ in range(3):
        await host.workers.wait_for_complete()
        await pilot.pause(0.05)


async def _quiet(host, pilot) -> None:
    """Let the previous action's tail (the picker's rescan and rebuild)
    finish, so a meter times only its own action."""
    await _settle(host, pilot)
    await pilot.pause(0.3)
    await _settle(host, pilot)


async def _wait_for_screen(host, pilot, screen_type) -> None:
    for _ in range(60):
        if isinstance(host.screen, screen_type):
            await pilot.pause(0.1)  # composed
            return
        await pilot.pause(0.05)
    raise AssertionError(f"{type(host.screen).__name__} is up, not {screen_type.__name__}")


def _themes_dir():
    from tldw_chatbook import config

    return config._get_effective_config_path().parent / "themes"


@pytest.mark.asyncio
@private_profile_test
async def test_file_actions_keep_the_ui_thread_free_with_fifty_themes(request, monkeypatch):
    """AC#1/#2 (TASK-33078): every editor file action -- Save, Save as,
    Rename, Delete, Import, Export -- scans, reads and writes on a worker
    thread through the editor's own backup scope, and the UI thread is not
    blocked for 100 ms by the action. AC#1 (TASK-33121): Use writes its
    launch default off the UI thread too.

    Save is not stall-timed: it returns the pane to the picker, and that
    view switch (restyle + list render, the same as Back) costs ~170 ms at
    211x44 with or without files. Export's two I/O halves (read, then write)
    are timed apart from its path prompt: pushing or dismissing that modal
    repaints the whole Settings screen (~250 ms), like every Settings prompt.
    """
    from textual.widgets import Input

    from tldw_chatbook.UI.Screens.settings_screen import RagProfileNameModal
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    config_threads = []
    real_write = tc._write_launch_default

    def recorded_config_write(name):
        config_threads.append(threading.current_thread())
        return real_write(name)

    monkeypatch.setattr(tc, "_write_launch_default", recorded_config_write)
    host = _host()
    for i in range(50):
        _saved_theme(host, f"mine{i:02d}")
    source = _themes_dir().parent / "incoming.toml"
    source.write_text('[theme]\nname = "incoming"\n[colors]\nprimary = "#224466"\n', encoding="utf-8")
    export_to = _themes_dir().parent / "exported.toml"
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "mine10")
        await _settle(host, pilot)
        settings = host.screen
        pane = settings.query_one("#settings-theme-pane")
        editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        # The editor opens on a saved theme with a real edit (not measured:
        # Edit's load is not one of the file actions in scope).
        pane.open_editor("mine10", "edit")
        editor.query_one("#settings-theme-color-primary", Input).value = "#123456"
        await _settle(host, pilot)
        calls = _record_io_threads(monkeypatch)
        stalls = {}

        editor.on_save_theme()
        await _settle(host, pilot)
        assert "#123456" in (_themes_dir() / "mine10.toml").read_text(encoding="utf-8")

        await _quiet(host, pilot)
        with _StallMeter() as meter:
            settings._handle_theme_save_as_result("mine10_as")
            await _settle(host, pilot)
        stalls["save as"] = meter.worst_ms
        assert (_themes_dir() / "mine10_as.toml").exists()

        await _quiet(host, pilot)
        with _StallMeter() as meter:
            label = editor.dialog_label("mine11")
            refusal = editor.rename_refusal("mine11", "mine11_renamed")
            settings._handle_theme_rename_result("mine11", "mine11_renamed")
            await _settle(host, pilot)
        stalls["rename"] = meter.worst_ms
        assert label == "'Mine11' (mine11.toml)" and refusal is None
        assert (_themes_dir() / "mine11_renamed.toml").exists()
        assert not (_themes_dir() / "mine11.toml").exists()

        await _quiet(host, pilot)
        with _StallMeter() as meter:
            editor.run_file_action(editor.request_delete("mine12"))
            await _wait_for_screen(host, pilot, ConfirmationDialog)
            await pilot.click("#confirm-button")
            await _settle(host, pilot)
        stalls["delete"] = meter.worst_ms
        assert not (_themes_dir() / "mine12.toml").exists()

        await _quiet(host, pilot)
        with _StallMeter() as meter:
            assert editor.import_refusal(str(source)) is None
            settings._handle_theme_import_result(str(source))
            await _settle(host, pilot)
        stalls["import"] = meter.worst_ms
        assert (_themes_dir() / "incoming.toml").exists()

        prompts = []
        real_prompt = SettingsThemeEditor._prompt_export
        monkeypatch.setattr(SettingsThemeEditor, "_prompt_export", lambda self, *a, **k: prompts.append(a))
        await _quiet(host, pilot)
        with _StallMeter() as meter:
            editor.run_file_action(editor.export_theme("mine13"))
            await _settle(host, pilot)
        stalls["export read"] = meter.worst_ms
        (name, file_name, data), = prompts
        await _quiet(host, pilot)
        with _StallMeter() as meter:
            editor.run_file_action(editor._export(export_to.with_name("direct.toml"), data))
            await _settle(host, pilot)
        stalls["export write"] = meter.worst_ms
        # And the whole path through the prompt, unmeasured.
        real_prompt(editor, name, file_name, data)
        await _wait_for_screen(host, pilot, RagProfileNameModal)
        host.screen.query_one("#settings-rag-profile-name-input", Input).value = str(export_to)
        await pilot.click("#settings-rag-profile-name-confirm")
        await _settle(host, pilot)
        assert export_to.exists()

        assert {step for step, _ in calls} == set(_IO_STEPS)
        # The one UI-thread read: Import's inline dialog check parses the
        # typed file (at most 64 KB, non-blocking open); the import itself
        # parses it again off the UI thread.
        on_ui = [step for step, thread in calls if thread is threading.main_thread()]
        assert on_ui == ["_parse_import"], on_ui
        assert [t for step, t in calls if step == "_parse_import"][1] is not threading.main_thread()
        assert all(ms < 100 for ms in stalls.values()), stalls

        # TASK-33121: Use's config write runs off the UI thread.
        await _highlight(host, pilot, "mine20")
        settings.query_one("#settings-theme-picker").use_highlighted()
        await _settle(host, pilot)
        assert tc.current_launch_default() == "mine20"
        assert config_threads and all(t is not threading.main_thread() for t in config_threads)


@pytest.mark.asyncio
@private_profile_test
async def test_file_actions_run_one_at_a_time(request, monkeypatch):
    """Two actions started together never interleave their I/O: the second
    waits for the first (the editor's file-action lock), and both land."""
    host = _host()
    for name in ("one", "two", "three"):
        _saved_theme(host, name)
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "one")
        await _settle(host, pilot)
        settings = host.screen
        editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        busy = {"now": 0, "most": 0}
        lock = threading.Lock()
        real_write = SettingsThemeEditor._write_toml

        def slow_write(self, path, data):
            with lock:
                busy["now"] += 1
                busy["most"] = max(busy["most"], busy["now"])
            time.sleep(0.15)
            try:
                return real_write(self, path, data)
            finally:
                with lock:
                    busy["now"] -= 1

        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", slow_write)
        first = editor.run_file_action(editor.rename_user_theme("one", "one_b"))
        second = editor.run_file_action(editor.rename_user_theme("two", "two_b"))
        await first.wait()
        await second.wait()
        await _settle(host, pilot)
        assert first.result is True and second.result is True
        assert busy["most"] == 1
        names = {p.stem for p in _themes_dir().glob("*.toml")}
        assert {"one_b", "two_b", "three"} <= names and not {"one", "two"} & names
        # The picker's own rescan (TASK-32957 generation guard) lands the
        # final listing, not one from before the renames.
        ids = {e.id for e in settings.query_one("#settings-theme-picker").entries}
        assert {"one_b", "two_b"} <= ids and not {"one", "two"} & ids


@pytest.mark.asyncio
@private_profile_test
async def test_action_racing_the_picker_scan_lands_the_newest_listing(request, monkeypatch):
    """A picker rescan started before a Delete, and slower than it, must not
    put the deleted theme back in the list (the scan generation guard)."""
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    host = _host()
    for name in ("keep", "gone"):
        _saved_theme(host, name)
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "keep")
        await _settle(host, pilot)
        settings = host.screen
        editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        picker = settings.query_one("#settings-theme-picker")
        real_listing = SettingsThemeEditor.user_theme_listing
        slow_once = {"left": 1}

        def listing(self):
            result = real_listing(self)  # scanned before the delete
            if slow_once["left"]:
                slow_once["left"] -= 1
                time.sleep(0.4)  # ...and lands after it
            return result

        monkeypatch.setattr(SettingsThemeEditor, "user_theme_listing", listing)
        picker.refresh_catalog()
        editor.run_file_action(editor.request_delete("gone"))
        await _wait_for_screen(host, pilot, ConfirmationDialog)
        await pilot.click("#confirm-button")
        await _settle(host, pilot)
        await asyncio.sleep(0.5)
        await _settle(host, pilot)
        assert not (_themes_dir() / "gone.toml").exists()
        assert "gone" not in {e.id for e in picker.entries}


def _gate(monkeypatch, step: str) -> tuple[threading.Event, threading.Event]:
    """Hold the editor's first ``step`` call on its worker thread.

    Returns ``(entered, release)``: set when the step is reached; set it to
    let the step run.
    """
    entered, release = threading.Event(), threading.Event()
    real = getattr(SettingsThemeEditor, step)
    held = {"left": 1}

    def gated(self, *args):
        if held["left"]:
            held["left"] -= 1
            entered.set()
            assert release.wait(10), f"{step} gate never released"
        return real(self, *args)

    monkeypatch.setattr(SettingsThemeEditor, step, gated)
    return entered, release


async def _until(pilot, condition, what: str) -> None:
    for _ in range(200):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.mark.asyncio
@private_profile_test
async def test_leaving_settings_mid_action_still_finishes_it(request, monkeypatch):
    """The user leaves Theme while a Rename is between its steps (new file
    written, old file not yet removed): tearing the editor down does not
    cut it there -- the rename still completes (new file registered, old
    file removed). An editor-owned worker would be cancelled at unmount."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

    host = _host()
    _saved_theme(host, "leaving")
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "leaving")
        await _settle(host, pilot)
        settings = host.screen
        editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        entered, release = _gate(monkeypatch, "_unlink")
        worker = editor.run_file_action(editor.rename_user_theme("leaving", "left"))
        await _until(pilot, entered.is_set, "the rename to reach its unlink")
        # Mid-sequence: both files exist, the action is still running.
        assert (_themes_dir() / "left.toml").exists() and (_themes_dir() / "leaving.toml").exists()
        settings._select_category(SettingsCategoryId.APPEARANCE.value)
        await pilot.pause(0.2)
        assert not editor.is_attached
        release.set()
        await worker.wait()
        await _settle(host, pilot)
        assert worker.result is True
        assert (_themes_dir() / "left.toml").exists() and not (_themes_dir() / "leaving.toml").exists()
        assert "left" in host.available_themes and "leaving" not in host.available_themes


@pytest.mark.asyncio
@private_profile_test
async def test_a_save_landing_in_a_later_session_changes_only_its_own_file(request, monkeypatch):
    """Review I-1: Save alpha (slow scan) -> Back -> Edit beta -> edit ->
    alpha's save lands. It writes alpha's palette to alpha.toml and nothing
    else: no overwrite dialog over beta, no rename of the editor to alpha,
    no jump back to the picker. The leave prompt's Save then writes beta's
    edit to beta.toml -- not over alpha."""
    from textual.widgets import Input

    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    real_scan = SettingsThemeEditor._scan_theme_files
    slow = {"on": False}

    def scan(self):
        if slow["on"]:
            time.sleep(0.8)
        return real_scan(self)

    monkeypatch.setattr(SettingsThemeEditor, "_scan_theme_files", scan)
    host = _host()
    _saved_theme(host, "alpha")
    _saved_theme(host, "beta")
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "alpha")
        await _settle(host, pilot)
        settings = host.screen
        pane = settings.query_one("#settings-theme-pane")
        editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        pane.open_editor("alpha", "edit")
        await _settle(host, pilot)
        slow["on"] = True
        save = editor.run_file_action(editor.save_theme())  # alpha, unmodified, in flight
        await pilot.pause(0.05)
        pane.show_picker()  # Back: nothing modified, so no prompt
        await pilot.pause(0.05)
        slow["on"] = False
        pane.open_editor("beta", "edit")
        await pilot.pause(0.05)
        editor.query_one("#settings-theme-color-primary", Input).value = "#BADBAD"
        await pilot.pause(0.1)
        await save.wait()  # alpha's save lands in beta's session
        await pilot.pause(0.3)
        assert not isinstance(host.screen, ConfirmationDialog)
        assert pane.current == "settings-theme-editor-view"
        assert (editor.current_theme_name, editor._loaded_user_theme) == ("beta", "beta")
        assert editor.query_one("#settings-theme-name", Input).value == "beta"
        assert editor.is_modified
        await editor.save_theme()  # the leave prompt's Save
        await _settle(host, pilot)
        assert 'primary = "#0099FF"' in (_themes_dir() / "alpha.toml").read_text()
        assert 'primary = "#BADBAD"' in (_themes_dir() / "beta.toml").read_text()


@pytest.mark.asyncio
@private_profile_test
async def test_a_stale_file_action_shows_no_dialog_in_the_new_session(request, monkeypatch):
    """Review I-1: a Delete whose scan lands after the editor opened on
    another theme does not push its confirmation there; it says why."""
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    host = _host()
    for name in ("doomed", "other"):
        _saved_theme(host, name)
    notes = []
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "doomed")
        await _settle(host, pilot)
        settings = host.screen
        pane = settings.query_one("#settings-theme-pane")
        editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        entered, release = _gate(monkeypatch, "_scan_theme_files")
        worker = editor.run_file_action(editor.request_delete("doomed"))
        await _until(pilot, entered.is_set, "the delete's scan")
        pane.open_editor("other", "edit")
        host.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
        release.set()
        await worker.wait()
        await pilot.pause(0.2)
        assert not isinstance(host.screen, ConfirmationDialog)
        assert (_themes_dir() / "doomed.toml").exists()
        assert any("Did not delete 'doomed'" in message for message, _ in notes)


@pytest.mark.asyncio
@private_profile_test
async def test_file_actions_of_two_editor_instances_run_one_at_a_time(request, monkeypatch):
    """Review M-3: leave Theme and come back mid-Rename -- the new editor's
    action still waits for the old editor's (one lock per app)."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

    host = _host()
    for name in ("first", "second"):
        _saved_theme(host, name)
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "first")
        await _settle(host, pilot)
        settings = host.screen
        old_editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        entered, release = _gate(monkeypatch, "_write_toml")
        writes = []
        gated_write = SettingsThemeEditor._write_toml

        def counted(self, path, data):
            writes.append(path.stem)
            return gated_write(self, path, data)

        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", counted)
        first = old_editor.run_file_action(old_editor.rename_user_theme("first", "first_b"))
        await _until(pilot, entered.is_set, "the first rename's write")
        # Not _settle: that waits for every worker, the held rename too.
        settings._select_category(SettingsCategoryId.APPEARANCE.value)
        await pilot.pause(0.5)
        settings._select_category(SettingsCategoryId.THEME.value)
        await pilot.pause(0.5)
        new_editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        assert new_editor is not old_editor and not old_editor.is_attached
        second = new_editor.run_file_action(new_editor.rename_user_theme("second", "second_b"))
        await pilot.pause(0.4)
        assert writes == ["first_b"]  # the second rename is still queued
        release.set()
        await first.wait()
        await second.wait()
        await _settle(host, pilot)
        assert writes == ["first_b", "second_b"]
        assert first.result is True and second.result is True


@pytest.mark.asyncio
@private_profile_test
async def test_quit_waits_for_a_file_action_between_its_steps(request, monkeypatch):
    """Review M-2: quitting while a Rename is between writing the new file
    and removing the old one lets it finish before the exit (which cancels
    every worker) -- no two files for one theme."""
    import tldw_chatbook.app as app_module

    host = _host()
    _saved_theme(host, "quitter")
    order = []
    monkeypatch.setattr(
        app_module,
        "persist_cli_config_for_shutdown",
        lambda: order.append(("config", (_themes_dir() / "quitter.toml").exists())) or True,
    )
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "quitter")
        await _settle(host, pilot)
        editor = host.screen.query_one("#settings-theme-editor", SettingsThemeEditor)
        entered, release = _gate(monkeypatch, "_unlink")
        worker = editor.run_file_action(editor.rename_user_theme("quitter", "quitted"))
        await _until(pilot, entered.is_set, "the rename's unlink")
        threading.Timer(0.3, release.set).start()
        host._save_shutdown_caches_with_timeout = lambda: None
        await asyncio.to_thread(app_module.TldwCli._run_blocking_quit_persistence, host)
        assert order == [("config", False)]  # the old file was gone before the quit went on
        assert worker.is_finished and worker.result is True


@pytest.mark.asyncio
@private_profile_test
async def test_leaving_during_a_save_waits_for_it_instead_of_prompting(request, monkeypatch):
    """Review M-4: Back or leaving Settings while the edits are being saved
    does not ask about "unsaved changes"; it waits for the Save. A Save that
    fails keeps the edits dirty, says so, and the leave stays put."""
    from textual.widgets import Input

    from tldw_chatbook.Widgets.settings_theme_editor import ThemeLeaveModal

    host = _host()
    _saved_theme(host, "busy")
    notes = []
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "busy")
        await _settle(host, pilot)
        settings = host.screen
        pane = settings.query_one("#settings-theme-pane")
        editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        pane.open_editor("busy", "edit")
        await _settle(host, pilot)
        editor.query_one("#settings-theme-color-primary", Input).value = "#123456"
        await pilot.pause(0.1)
        assert editor.is_modified

        # Back mid-save: no prompt; the Save lands and returns to the picker.
        entered, release = _gate(monkeypatch, "_scan_theme_files")
        editor.on_save_theme()
        await _until(pilot, entered.is_set, "the save's scan")
        settings.query_one("#settings-theme-back").press()
        await pilot.pause(0.3)
        assert not isinstance(host.screen, ThemeLeaveModal)
        release.set()
        await _settle(host, pilot)
        assert 'primary = "#123456"' in (_themes_dir() / "busy.toml").read_text()
        assert pane.current == "settings-theme-picker" and not editor.is_modified

        # Leaving Settings mid-save that then fails: no prompt, stay, dirty.
        pane.open_editor("busy", "edit")
        await _settle(host, pilot)
        editor.query_one("#settings-theme-color-primary", Input).value = "#654321"
        await pilot.pause(0.1)
        entered, release = _gate(monkeypatch, "_scan_theme_files")

        def failing_write(self, path, data):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(SettingsThemeEditor, "_write_toml", failing_write)
        host.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
        editor.on_save_theme()
        await _until(pilot, entered.is_set, "the save's scan")
        leaving = asyncio.ensure_future(settings.confirm_navigation())
        await pilot.pause(0.3)
        assert not isinstance(host.screen, ThemeLeaveModal) and not leaving.done()
        release.set()
        assert await leaving is False
        assert editor.is_modified
        assert any("Failed to save theme" in message for message, _ in notes)


# -- TASK-33121: the launch-default writer ------------------------------------


def _slow_config(monkeypatch, delay: float = 0.0, fail: bool = False):
    written: list[tuple[str, threading.Thread]] = []

    def apply(mutation):
        time.sleep(delay)
        if fail:
            raise OSError("disk full")
        written.append((mutation["general"]["default_theme"], threading.current_thread()))
        return SimpleNamespace(file_replaced=True, caches_reloaded=True)

    monkeypatch.setattr(tc, "_apply_config_mutation", apply)
    return written


@pytest.mark.asyncio
async def test_launch_default_writes_land_in_order_off_the_ui_thread(monkeypatch):
    written = _slow_config(monkeypatch, delay=0.05)
    app = SimpleNamespace(app_config={"general": {}})
    results = await asyncio.gather(
        tc.persist_launch_default_async(app, "first"),
        tc.persist_launch_default_async(app, "second"),
    )
    # Review M-1: only the latest write records its outcome.
    assert results == [(True, True, False), (True, True, True)]
    assert [name for name, _ in written] == ["first", "second"]  # the last Use wins
    assert all(thread is not threading.main_thread() for _, thread in written)
    assert app.app_config["general"]["default_theme"] == "second"
    # A blocking write (Revert, the palette) queues behind pending ones.
    pending = asyncio.ensure_future(tc.persist_launch_default_async(app, "third"))
    await asyncio.sleep(0)
    assert await asyncio.to_thread(tc.persist_launch_default, app, "fourth") == (True, True)
    await pending
    assert [name for name, _ in written][-2:] == ["third", "fourth"]


@pytest.mark.asyncio
async def test_quit_waits_for_a_pending_launch_default_write(monkeypatch):
    """AC#3: the quit path's off-loop persistence waits for queued writes
    before the app exits."""
    import tldw_chatbook.app as app_module

    written = _slow_config(monkeypatch, delay=0.3)
    order: list[str] = []
    monkeypatch.setattr(app_module, "persist_cli_config_for_shutdown", lambda: order.append("config") or True)
    pending = asyncio.ensure_future(tc.persist_launch_default_async(SimpleNamespace(), "chosen"))
    await asyncio.sleep(0)
    quitting = SimpleNamespace(_save_shutdown_caches_with_timeout=lambda: None)
    await asyncio.to_thread(app_module.TldwCli._run_blocking_quit_persistence, quitting)
    assert [name for name, _ in written] == ["chosen"]  # written before the quit went on
    assert order == ["config"]
    await pending


@pytest.mark.asyncio
async def test_a_failed_launch_default_write_raises_to_the_caller(monkeypatch):
    _slow_config(monkeypatch, fail=True)
    with pytest.raises(OSError):
        await tc.persist_launch_default_async(SimpleNamespace(), "x")
    assert tc.wait_for_launch_default_writes(timeout=1) is True


@pytest.mark.asyncio
@private_profile_test
async def test_use_reports_a_write_that_raised(request, monkeypatch):
    """AC#2: a write that raises off the UI thread is still reported."""
    host = _host()
    notes = []
    _slow_config(monkeypatch, fail=True)
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "nord")
        host.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
        host.screen.query_one("#settings-theme-picker").use_highlighted()
        await _settle(host, pilot)
        assert host.theme == "nord"
        assert ("Nord applied; the launch default was not saved", "warning") in notes


@pytest.mark.asyncio
async def test_a_superseded_launch_default_write_leaves_app_config_to_the_later_one(monkeypatch):
    """Review M-1: Use's async write, then a blocking write (Revert, the
    palette) in its ~140 ms window: the later write owns app_config, and
    the earlier reports itself as not the latest (so Use says nothing)."""
    _slow_config(monkeypatch, delay=0.1)
    app = SimpleNamespace(app_config={"general": {}})
    use = asyncio.ensure_future(tc.persist_launch_default_async(app, "A"))
    await asyncio.sleep(0)
    assert tc._persist_launch_default(app, "prev") == (True, True)
    assert await use == (True, True, False)
    assert app.app_config["general"]["default_theme"] == "prev"


@pytest.mark.asyncio
@private_profile_test
async def test_use_then_quick_revert_shows_no_stale_use_toast(request, monkeypatch):
    """Review M-1 through the picker: Revert within Use's write window --
    the Use toast does not arrive after the Revert."""
    from textual.widgets import Button

    # The write outlasts the pause below (~0.5 s while the picker repaints),
    # so the Revert really lands inside it; Use now queues at once (#2).
    _slow_config(monkeypatch, delay=1.5)
    host = _host()
    notes = []
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "nord")
        picker = host.screen.query_one("#settings-theme-picker")
        before = str(host.theme)
        host.notify = lambda message, **kw: notes.append((message, kw.get("severity")))
        picker.use_highlighted()
        await pilot.pause(0.02)
        picker.query_one("#settings-theme-revert", Button).press()
        await _settle(host, pilot)
        assert str(host.theme) == before
        assert not any("is now your theme" in message for message, _ in notes)


# -- PR #2877 Qodo round -------------------------------------------------------


@pytest.mark.asyncio
@private_profile_test
async def test_quit_waits_for_a_confirmed_delete_and_starts_no_new_action(request, monkeypatch):
    """Review #1: a Delete confirmed in its dialog runs as a file action,
    so the quit waits for its unlink; once the quit has begun, no new file
    action starts (the exit would cut it between its steps)."""
    from textual.widgets import Button

    import tldw_chatbook.app as app_module
    from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog

    host = _host()
    _saved_theme(host, "confirmed")
    _saved_theme(host, "spared")
    order = []
    monkeypatch.setattr(
        app_module,
        "persist_cli_config_for_shutdown",
        lambda: order.append((_themes_dir() / "confirmed.toml").exists()) or True,
    )
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "confirmed")
        await _settle(host, pilot)
        editor = host.screen.query_one("#settings-theme-editor", SettingsThemeEditor)
        editor.run_file_action(editor.request_delete("confirmed"))
        await _until(
            pilot,
            lambda: isinstance(host.screen, ConfirmationDialog) and host.screen.query("#confirm-button"),
            "the delete confirmation",
        )
        entered, release = _gate(monkeypatch, "_unlink")
        host.screen.query_one("#confirm-button", Button).press()
        await _until(pilot, entered.is_set, "the confirmed unlink")
        threading.Timer(0.3, release.set).start()
        host._save_shutdown_caches_with_timeout = lambda: None
        await asyncio.to_thread(app_module.TldwCli._run_blocking_quit_persistence, host)
        assert order == [False]  # the confirmed delete landed before the quit went on
        assert editor.run_file_action(editor.request_delete("spared")) is None
        await _settle(host, pilot)
        assert (_themes_dir() / "spared.toml").exists()


@pytest.mark.asyncio
async def test_quit_waits_for_theme_work_within_one_deadline():
    """Review #8: file actions and launch-default writes that both stall
    share the quit's one timeout -- not one each."""
    loop = asyncio.get_running_loop()
    stalled = SimpleNamespace(group=tc.THEME_FILE_ACTION_GROUP, wait=lambda: asyncio.sleep(10))
    app = SimpleNamespace(
        workers=[stalled],
        call_from_thread=lambda fn, *a: asyncio.run_coroutine_threadsafe(fn(*a), loop).result(),
    )
    writer_stall = tc._LAUNCH_DEFAULT_WRITER.submit(time.sleep, 1.2)
    started = time.monotonic()
    await asyncio.to_thread(tc.wait_for_theme_quit_work, app, 0.4)
    elapsed = time.monotonic() - started
    await asyncio.wrap_future(writer_stall)
    assert 0.35 < elapsed < 0.65, elapsed  # two separate 0.4 s waits took ~0.8 s
    assert tc.theme_quit_started(app)


@pytest.mark.asyncio
@private_profile_test
async def test_revert_before_the_use_worker_runs_restores_the_old_default_without_blocking(request, monkeypatch):
    """Review #2: Revert pressed before Use's worker has run still writes
    after Use's write, so the old launch default is what stays on disk.
    Review #4: Revert does not wait for the writes on the UI thread."""
    written = _slow_config(monkeypatch, delay=0.1)
    host = _host()
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "nord")
        picker = host.screen.query_one("#settings-theme-picker")
        launch = tc.current_launch_default()
        assert launch != "nord" and launch in host.available_themes
        picker.use_highlighted()
        started = time.perf_counter()
        picker._revert_pressed(SimpleNamespace(stop=lambda: None))  # no yield: the Use worker has not run
        blocked = time.perf_counter() - started
        await _settle(host, pilot)
        assert [name for name, _ in written] == ["nord", launch]
        assert blocked < 0.05, blocked


@pytest.mark.asyncio
@private_profile_test
async def test_a_superseded_use_failure_is_not_reported(request, monkeypatch):
    """Review #9: Use's write fails after a Revert asked for a later write;
    the later write owns the outcome, so no stale "not saved" toast."""

    def apply(mutation):
        time.sleep(0.1)
        if mutation["general"]["default_theme"] == "nord":
            raise OSError("disk full")
        return SimpleNamespace(file_replaced=True, caches_reloaded=True)

    monkeypatch.setattr(tc, "_apply_config_mutation", apply)
    host = _host()
    notes = []
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "nord")
        picker = host.screen.query_one("#settings-theme-picker")
        host.notify = lambda message, **kw: notes.append(message)
        picker.use_highlighted()
        picker._revert_pressed(SimpleNamespace(stop=lambda: None))
        await _settle(host, pilot)
        assert not any("not saved" in message or "not restored" in message for message in notes), notes
    # The catalog API: a superseded failure settles as not the latest.
    first = asyncio.ensure_future(tc.persist_launch_default_async(SimpleNamespace(), "nord"))
    await asyncio.sleep(0)
    assert await tc.persist_launch_default_async(SimpleNamespace(), "later") == (True, True, True)
    assert await first == (False, False, False)


@pytest.mark.asyncio
@private_profile_test
async def test_an_older_slower_scan_does_not_replace_a_newer_one(request, monkeypatch):
    """Review #6: scans run on worker threads; the inline checks keep the
    one started last, even when an earlier scan finishes after it."""
    editor = SettingsThemeEditor()
    first_in, first_go = threading.Event(), threading.Event()
    scans = iter([({"old": None}, {}), ({"new": None}, {})])

    def read(self):
        scan = next(scans)
        if "old" in scan[0]:
            first_in.set()
            assert first_go.wait(5)
        return scan

    monkeypatch.setattr(SettingsThemeEditor, "_read_theme_folder", read)
    slow = threading.Thread(target=editor._scan_theme_files)
    slow.start()
    assert first_in.wait(5)
    editor._scan_theme_files()
    first_go.set()
    slow.join()
    assert "new" in editor._last_scan[0]
