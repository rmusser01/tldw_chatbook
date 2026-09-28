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
        monkeypatch.setattr(SettingsThemeEditor, "_prompt_export", lambda self, *a: prompts.append(a))
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


@pytest.mark.asyncio
@private_profile_test
async def test_leaving_settings_mid_action_still_finishes_it(request, monkeypatch):
    """The user leaves Theme while a Rename is writing: the rename still
    completes (new file registered, old file removed), without errors."""
    from tldw_chatbook.UI.Screens.settings_screen import SettingsCategoryId

    host = _host()
    _saved_theme(host, "leaving")
    async with host.run_test(size=(211, 44)) as pilot:
        await _highlight(host, pilot, "leaving")
        await _settle(host, pilot)
        settings = host.screen
        editor = settings.query_one("#settings-theme-editor", SettingsThemeEditor)
        _record_io_threads(monkeypatch, delay=0.2)
        worker = editor.run_file_action(editor.rename_user_theme("leaving", "left"))
        await pilot.pause(0.05)
        settings._select_category(SettingsCategoryId.APPEARANCE.value)
        await pilot.pause(0.1)
        assert not editor.is_attached
        await worker.wait()
        await _settle(host, pilot)
        assert worker.result is True
        assert (_themes_dir() / "left.toml").exists() and not (_themes_dir() / "leaving.toml").exists()
        assert "left" in host.available_themes and "leaving" not in host.available_themes


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
    assert results == [(True, True), (True, True)]
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
