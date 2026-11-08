"""Snapshot preference ownership, stale drafts, and canonical category Save."""

import asyncio
import contextlib
import re
import sys
import threading
import time
import traceback

import pytest
from textual.widgets import Button, Checkbox, Collapsible, Input

from Tests.app_module_patches import set_app_global
from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from tldw_chatbook.LLM_Management import snapshot_settings as preferences
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


def _frame_of(thread):
    """The innermost tldw_chatbook frame (else the last one) a thread runs."""
    frame = sys._current_frames().get(thread.ident)
    stack = traceback.extract_stack(frame) if frame else []
    entry = ([e for e in stack if "tldw_chatbook" in e.filename] or stack or [None])[-1]
    if entry is None:
        return f"{thread.name} (no frame)"
    path = entry.filename.rsplit("tldw_chatbook/", 1)[-1].rsplit("/lib/", 1)[-1]
    return f"{thread.name} at {path}:{entry.lineno} {entry.name}"


def _cpu_seconds(thread):
    """CPU seconds a thread has used, where the platform can say."""
    with contextlib.suppress(AttributeError, OSError, OverflowError, TypeError):
        return time.clock_gettime(time.pthread_getcpuclockid(thread.ident))
    return None


async def _until(predicate, timeout=5):
    """Wait for a save thread's outcome without walking the DOM (TASK-36530).

    ``pilot.pause()`` walks every Settings node on the event loop, and a busy
    loop starves the save thread of the GIL; a bare sleep keeps the loop idle.
    """
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class _SaveTimeline:
    """Where the keyboard Save's time went, for its timeout note (TASK-36530).

    Test-only wrappers stamp the save worker, the save thread and its config
    write lock, so production code carries no instrumentation. The note also
    names what the save could be waiting behind: a busy default executor, a
    config lock owner, or a CPU-bound event loop (the GIL hand-off the save's
    storage-admission syscalls pay, which stretched it to seconds in CI).
    """

    PHASES = (
        ("dispatch", "enter", "worker_start"),
        ("executor queue", "worker_start", "thread_start"),
        ("lock wait", "thread_start", "lock_held"),
        ("write", "lock_held", "thread_end"),
        ("resume", "thread_end", "worker_end"),
    )

    def __init__(self, monkeypatch, screen):
        from tldw_chatbook import config

        self.marks, self.thread, self.done = {}, None, asyncio.Event()
        save, lock = preferences.save_snapshot_preferences, config._config_write_lock
        draft = screen._save_snapshot_preferences_draft

        def timed_save(value, **kwargs):
            self.thread = threading.current_thread()
            self.marks["thread_start"] = time.perf_counter()
            try:
                return save(value, **kwargs)
            finally:
                self.marks["thread_end"] = time.perf_counter()

        @contextlib.contextmanager
        def timed_lock(path):
            with lock(path):
                if threading.current_thread() is self.thread:
                    self.marks["lock_held"] = time.perf_counter()
                yield

        async def timed_draft(*args):
            self.marks["worker_start"] = time.perf_counter()
            try:
                await draft(*args)
            finally:
                self.marks["worker_end"] = time.perf_counter()
                self.done.set()

        monkeypatch.setattr(preferences, "save_snapshot_preferences", timed_save)
        monkeypatch.setattr(config, "_config_write_lock", timed_lock)
        monkeypatch.setattr(screen, "_save_snapshot_preferences_draft", timed_draft)

    def start(self):
        """Stamp Enter, and the event loop's CPU clock beside it."""
        self.loop_cpu = time.thread_time()
        self.marks["enter"] = time.perf_counter()

    def note(self):
        """Each finished phase, the one still running, and what it may wait on."""
        from tldw_chatbook import config

        now, phases = time.perf_counter(), []
        for name, start, end in self.PHASES:
            if start not in self.marks:
                break
            if end not in self.marks:
                phases.append(f"{name} still running at {now - self.marks[start]:.2f} s")
                break
            phases.append(f"{name} {self.marks[end] - self.marks[start]:.2f} s")
        running = self.thread is not None and "thread_end" not in self.marks
        save_cpu = _cpu_seconds(self.thread) if running else None
        executor = getattr(asyncio.get_running_loop(), "_default_executor", None)
        busy = [
            _frame_of(thread)
            for thread in list(getattr(executor, "_threads", ()))
            if any(
                e.name == "run" and e.filename.endswith("futures/thread.py")
                for e in traceback.extract_stack(sys._current_frames().get(thread.ident))
            )
        ]
        threads = {thread.ident: thread for thread in threading.enumerate()}
        owners = [
            f"{name} held by {_frame_of(threads[int(owner[1])])}"
            for name, held in (
                ("rebuild lock", config._SETTINGS_REBUILD_LOCK),
                ("file lock", config._CONFIG_FILE_LOCK),
            )
            if (owner := re.search(r"owner=(\d+)", repr(held)))
            and int(owner[1]) in threads
        ]
        return (
            f"Save phases: {', '.join(phases)}; "
            f"save thread: {_frame_of(self.thread) if running else 'not running'}"
            f"{'' if save_cpu is None else f' ({save_cpu:.2f} s CPU so far)'}; "
            f"event loop CPU since Enter: {time.thread_time() - self.loop_cpu:.2f} s "
            f"of {now - self.marks['enter']:.2f} s; default executor "
            f"{len(busy)}/{getattr(executor, '_max_workers', '?')} busy, "
            f"{executor._work_queue.qsize() if executor else '?'} queued {busy}; "
            f"config locks: {owners or 'free'}"
        )


@pytest.fixture
def no_snapshot_splash(monkeypatch):
    from tldw_chatbook.config import get_cli_setting

    set_app_global(monkeypatch, "get_cli_setting", lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad", [{"keep_count": 0}, {"keep_count": 1.5}, {"enabled": "yes"}]
)
@private_profile_test
async def test_malformed_f9_preferences_leave_provider_surface_and_revert_recovery(
    request, no_snapshot_splash, bad
):
    from tldw_chatbook import config

    assert config.apply_settings_mutation_to_cli_config(
        {"llamacpp_snapshots": bad}
    ).fully_applied
    app = _build_test_app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        field = screen.query_one("#settings-snapshot-keep", Input)
        assert field.disabled and field.value == ""
        assert not screen.query_one("#settings-provider-search", Input).disabled
        assert "Advanced Config" in str(
            screen.query_one("#settings-snapshot-result").render()
        )
        # The guided Save/Revert pair can mount after the fields (TASK-38993:
        # absent after one pause, 3 of 3 serially at load 50-155).
        await _until(lambda: screen.query("#settings-revert-category"), timeout=10)
        assert not screen.query_one("#settings-revert-category", Button).disabled
        screen.query_one("#settings-revert-category", Button).press()
        await pilot.pause()
        assert field.disabled
        assert preferences.save_snapshot_preferences(
            preferences.SnapshotPreferences(enabled=True, keep_count=23)
        )
        screen.query_one("#settings-revert-category", Button).press()
        await pilot.pause()
        assert not field.disabled and field.value == "23"
        field.value = "24"
        await pilot.pause()
        assert config.apply_settings_mutation_to_cli_config(
            {"llamacpp_snapshots": {"keep_count": "broken-private-value"}}
        ).fully_applied
        screen.action_settings_revert_category(allow_text_entry_focus=True)
        await pilot.pause()
        app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause()
        assert field.disabled
        assert "broken-private-value" not in str(
            screen.query_one("#settings-snapshot-result").render()
        )


@pytest.mark.asyncio
@private_profile_test
async def test_f9_snapshot_save_rejects_noninteger_values_without_coercion(
    request, no_snapshot_splash,
):
    app = _build_test_app()
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        original = preferences.load_snapshot_preferences()
        for raw in (True, 27.0, 27.5, float("nan"), "27.0", "NaN", "0", "1001"):
            await screen._save_snapshot_preferences_draft(
                (original.enabled, raw), original, False
            )
            assert preferences.load_snapshot_preferences() == original
            assert "Not saved" in str(
                screen.query_one("#settings-snapshot-result").render()
            )


@pytest.mark.asyncio
# Named ids: the UI census takes one node per line, without whitespace.
@pytest.mark.parametrize(
    "raw", ["027", "+27", " 27 "], ids=["leading-zero", "plus-sign", "padded"]
)
@pytest.mark.parametrize("combined", [False, True])
@private_profile_test
async def test_success_canonicalizes_unchanged_f9_draft_and_continues_once(
    request, no_snapshot_splash, monkeypatch, raw, combined
):
    from tldw_chatbook import config

    app = _build_test_app()
    writes = []
    original_save = preferences.save_snapshot_preferences

    def save(value, **kwargs):
        writes.append(value)
        assert len(writes) == 1, "duplicate snapshot persistence"
        return original_save(value, **kwargs)

    monkeypatch.setattr(preferences, "save_snapshot_preferences", save)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        if combined:
            screen.query_one(
                "#settings-model-value", Input
            ).value = "snapshot-review-model"
        field = screen.query_one("#settings-snapshot-keep", Input)
        field.value = raw
        await pilot.pause()
        continuations = []
        original_action = screen.action_settings_save_category

        def action(**kwargs):
            continuations.append(True)
            original_action(**kwargs)

        monkeypatch.setattr(screen, "action_settings_save_category", action)
        original_action(allow_text_entry_focus=True)
        # Until the save worker settles: its thread's write lands before the
        # worker canonicalizes the field (TASK-38993; a 0.2 s pause raced it).
        await _until(lambda: not screen._snapshot_preferences_saving)
        assert len(writes) == 1
        assert preferences.load_snapshot_preferences().keep_count == 27
        assert field.value == "27"
        assert not screen._snapshot_preferences_dirty()
        assert len(continuations) == int(combined)
        if combined:
            assert (
                config.get_cli_setting("chat_defaults", "model")
                == "snapshot-review-model"
            ), screen._provider_save_result


@pytest.mark.asyncio
@private_profile_test
async def test_f9_save_preserves_new_edits_while_persistence_waits(
    request, no_snapshot_splash, monkeypatch
):
    app = _build_test_app()
    entered, release = threading.Event(), threading.Event()
    original_save = preferences.save_snapshot_preferences
    writes = []

    def save(value, **kwargs):
        writes.append(value)
        entered.set()
        assert release.wait(5)
        return original_save(value, **kwargs)

    monkeypatch.setattr(preferences, "save_snapshot_preferences", save)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        # Stage the model through its Input, as the app does. The card's mount
        # echo (an Input.Changed for the saved model, queued when the Input is
        # built) then always lands first. A direct _stage_provider_value raced
        # a late echo, which restaged the saved model and dropped the draft.
        await _until(lambda: len(screen.query("#settings-model-value")))
        screen._set_model_field_value("snapshot-review-model", quiet=False)
        await _until(
            lambda: screen._provider_draft_value("model") == "snapshot-review-model"
        )
        field = screen.query_one("#settings-snapshot-keep", Input)
        field.value = "027"
        await pilot.pause()
        continuations = []
        original_action = screen.action_settings_save_category

        def action(**kwargs):
            continuations.append(True)
            original_action(**kwargs)

        monkeypatch.setattr(screen, "action_settings_save_category", action)
        original_action(allow_text_entry_focus=True)
        try:
            await _until(entered.is_set)
            # Model the interval before delivery of the newer Input.Changed.
            with screen.prevent(Input.Changed):
                field.value = "32"
                assert screen._snapshot_preferences_raw[1] == "027"
                release.set()
                await _until(lambda: not screen._snapshot_preferences_saving)
            assert field.value == "32"
            assert continuations == []
            field.post_message(Input.Changed(field, "32"))
            await pilot.pause()
        finally:
            release.set()
        await pilot.pause(0.2)
        assert len(writes) == 1
        assert preferences.load_snapshot_preferences().keep_count == 27
        assert field.value == "32"
        assert screen._snapshot_preferences_dirty()
        assert screen._provider_draft().is_dirty


@pytest.mark.asyncio
@private_profile_test
async def test_f9_snapshot_draft_uses_category_save_offthread_and_reverts(
    request, monkeypatch
):
    from tldw_chatbook.config import get_cli_setting

    set_app_global(monkeypatch, "get_cli_setting", lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ))
    app = _build_test_app()
    threads = []
    original = preferences.save_snapshot_preferences

    def save(value, **kwargs):
        threads.append(threading.get_ident())
        return original(value, **kwargs)

    monkeypatch.setattr(preferences, "save_snapshot_preferences", save)
    async with app.run_test(size=(140, 45)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        field = screen.query_one("#settings-snapshot-keep", Input)
        field.value = "27"
        screen.query_one("#settings-snapshot-enabled", Checkbox).value = True
        await pilot.pause()
        assert (
            preferences.load_snapshot_preferences() == preferences.SnapshotPreferences()
        )
        screen.action_settings_save_category(allow_text_entry_focus=True)
        # The worker too, or the next Save returns early while it still runs.
        await _until(
            lambda: preferences.load_snapshot_preferences().keep_count == 27
            and not screen._snapshot_preferences_saving
        )
        assert threads and all(value != threading.get_ident() for value in threads)
        field.value = "0"
        await pilot.pause()
        screen.action_settings_save_category(allow_text_entry_focus=True)
        await pilot.pause()
        assert preferences.load_snapshot_preferences().keep_count == 27
        screen.action_settings_revert_category(allow_text_entry_focus=True)
        await pilot.pause()
        app.screen.query_one("#confirm-button", Button).press()
        await pilot.pause()
        assert field.value == "27"


@pytest.mark.asyncio
@private_profile_test
async def test_f9_stale_snapshot_draft_requires_revert_before_save(request, monkeypatch):
    from tldw_chatbook.config import get_cli_setting

    set_app_global(monkeypatch, "get_cli_setting", lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ))
    app = _build_test_app()
    async with app.run_test(size=(80, 24)) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        screen.query_one("#settings-snapshot-keep", Input).value = "31"
        await pilot.pause()
        newer = preferences.SnapshotPreferences(enabled=True, keep_count=14)
        await asyncio.to_thread(preferences.save_snapshot_preferences, newer)
        screen.action_settings_save_category(allow_text_entry_focus=True)
        # Until the save worker settles; "Revert" alone is in the draft's own
        # copy before any Save, so the refusal's copy is what this pins.
        await _until(lambda: not screen._snapshot_preferences_saving)
        assert preferences.load_snapshot_preferences() == newer
        assert "changed elsewhere" in str(
            screen.query_one("#settings-snapshot-result").render()
        )
        assert screen.query_one("#settings-snapshot-keep", Input).value == "31"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (140, 45)])
@private_profile_test
async def test_f9_snapshot_controls_paint_and_keyboard_save(request, size, monkeypatch):
    from Tests.UI.test_llamacpp_snapshot_manager import frame, painted_text
    from tldw_chatbook.config import get_cli_setting

    set_app_global(monkeypatch, "get_cli_setting", lambda section, key=None, default=None: (
            False
            if (section, key) == ("splash_screen", "enabled")
            else get_cli_setting(section, key, default)
        ))
    app = _build_test_app()
    async with app.run_test(size=size) as pilot:
        screen = SettingsScreen(app)
        await app.push_screen(screen)
        screen._select_category("providers-models")
        await pilot.pause()
        screen.query_one("#settings-snapshot-controls", Collapsible).collapsed = False
        checkbox = screen.query_one("#settings-snapshot-enabled", Checkbox)
        # The opened disclosure is laid out on a later refresh, and focusing a
        # widget that has no region yet scrolls nowhere: wait for the row,
        # then for the focus to bring it into view (failed 2 of 3 under load).
        async with asyncio.timeout(10):
            while checkbox.region.height == 0:
                await pilot.pause(0.01)
        checkbox.focus()
        await pilot.wait_for_scheduled_animations()
        async with asyncio.timeout(10):
            while checkbox not in screen._compositor.visible_widgets:
                await pilot.pause(0.01)
        await pilot.press("space")
        await pilot.pause()
        assert checkbox.value
        assert checkbox in screen._compositor.visible_widgets
        frame(app, f"settings-{size[0]}-snapshot-draft")
        assert "Enable snapshots" in painted_text(app, checkbox)
        assert (
            dict(screen._provider_field_guidance_rows())["Saved as"]
            == "llamacpp_snapshots.enabled"
        )
        assert checkbox.label.plain == "Enable snapshots"
        assert (
            "next launch"
            in screen.query_one("#settings-snapshot-launch-scope").render().plain
        )
        save = screen.query_one("#settings-save-category", Button)
        save.focus()
        await pilot.wait_for_scheduled_animations()
        before_enter = (
            save.has_focus,
            save.disabled,
            screen._snapshot_preferences_dirty(),
            screen._snapshot_preferences_saving,
        )
        timeline = _SaveTimeline(monkeypatch, screen)
        timeline.start()
        await pilot.press("enter")
        try:
            # Await the worker; never poll with pilot.pause() here. Each pause
            # walks the whole Settings DOM on the event loop, and a CPU-bound
            # loop starves the save thread of the GIL (TASK-36530). The bound
            # rests on the measured save times in that task.
            await asyncio.wait_for(timeline.done.wait(), 5)
        except TimeoutError as exc:
            after_timeout = (
                save.has_focus,
                save.disabled,
                screen._snapshot_preferences_dirty(),
                screen._snapshot_preferences_saving,
            )
            save_workers = [
                worker.state.name for worker in app.workers
                if worker.group == "settings-snapshot-save"
            ]
            loaded_enabled = getattr(screen._snapshot_preferences_loaded, "enabled", None)
            exc.add_note(
                f"Snapshot Save before Enter (focused, disabled, dirty, saving): "
                f"{before_enter}; after timeout: "
                f"{after_timeout}; checkbox enabled: {checkbox.value}; "
                f"loaded enabled: {loaded_enabled}; save workers: {save_workers}; "
                f"{timeline.note()}"
            )
            raise
        assert preferences.load_snapshot_preferences().enabled, timeline.note()
