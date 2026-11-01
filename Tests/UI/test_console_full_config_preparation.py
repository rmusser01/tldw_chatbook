"""FULL waits for its existing checked config owner before rail presentation."""

import asyncio
import sys
import threading
from contextlib import contextmanager, nullcontext
from types import MethodType, SimpleNamespace

import pytest

from Tests.UI.test_console_checked_display_scope import _actual_calls, _screen, _warm
from tldw_chatbook import config
from tldw_chatbook.Backup_Recovery import raw_participants as raw
from tldw_chatbook.Backup_Recovery import storage_admission as storage
from tldw_chatbook.Backup_Recovery.config_participants import checked_config_identity
from tldw_chatbook.UI.Console_Modules.attach_visit import ConsoleAttachVisit
from tldw_chatbook.UI.Screens import chat_screen

pytestmark = pytest.mark.bootstrap_profile


def _full_screen(screen, monkeypatch):
    """Keep original FULL and rail/config entry; omit unrelated UI projections."""
    effects = []
    # This native-read host is unmounted; no archive or Terminal return exists.
    screen.app_instance._ui_ready = False
    screen._console_terminal_return_reading = None

    def no_op(*_args, **_kwargs):
        return None

    async def async_no_op(*_args, **_kwargs):
        return None

    runtime = SimpleNamespace(view=screen, _attached_generation=1)
    screen._console_runtime = lambda: runtime
    screen._console_attach_visit_generation = 1
    screen._current_console_attach_visit = lambda: ConsoleAttachVisit(
        runtime, screen, 1, screen._console_attach_visit_generation, True
    )
    screen._console_sync_in_progress = False
    screen._console_sync_requested = False
    screen._console_attach_sync_complete = False
    screen._console_attach_reconciled = False
    screen._console_attach_reconcile_running = False
    screen._console_control_bar_replay_whole_sync = False
    screen._message = SimpleNamespace(reconcile_console_speech_context=no_op)
    screen._image = SimpleNamespace(
        _h3_image_edit_registry=lambda: SimpleNamespace(drop_session=no_op),
        _reconcile_h3_image_edit_completions=no_op,
    )
    screen._session = SimpleNamespace(
        _sync_console_session_draft=no_op,
        _maybe_refresh_stale_default_console_settings=no_op,
        schedule_manual_read_acknowledgement=no_op,
    )
    screen._retrieval = SimpleNamespace(
        _warm_console_effective_scope_cache_if_stale=async_no_op,
        _refresh_active_dictionaries_summary_if_scope_changed=async_no_op,
        _refresh_active_world_books_summary_if_scope_changed=async_no_op,
    )
    screen._character = SimpleNamespace(
        _refresh_active_character_avatar_if_scope_changed=async_no_op,
    )
    screen._character_context = SimpleNamespace(refresh_if_scope_changed=async_no_op)
    screen._workspace = SimpleNamespace(tick_workspace_build_scope=nullcontext)
    for name in (
        "_record_ui_worker_started",
        "_record_ui_worker_finished",
        "_sync_console_chat_core_state",
        "_current_console_rail_state",
        "_sync_console_settings_summary",
        "_sync_console_settings_recovery_surfaces",
        "_sync_console_live_work_readiness_rows",
        "_sync_console_mode_bar",
        "_dispatch_active_console_roleplay_refresh",
        "_sync_console_workspace_context",
        "_sync_console_rail_visibility_if_changed",
        "_dispatch_console_rail_preference_prune",
        "_start_console_attach_reconciliation",
    ):
        setattr(screen, name, no_op)
    screen._sync_console_native_session_tabs = async_no_op
    screen._sync_native_console_transcript = async_no_op
    screen._sync_console_control_bar_under_config = lambda _rail: effects.append("rail")
    screen._request_console_control_bar_sync = lambda **_: effects.append("retry")
    screen.call_after_refresh = lambda _: effects.append("attach")
    screen._sync_console_rail_and_controls = MethodType(
        chat_screen.ChatScreen._sync_console_rail_and_controls, screen
    )
    monkeypatch.setattr(
        chat_screen.project_instruction_ui,
        "sync_project_instruction_status_for_screen",
        no_op,
    )
    return effects


@pytest.mark.asyncio
async def test_expired_original_full_awaits_config_without_main_native_entry(
    tmp_path, monkeypatch, record_property
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    try:
        _full_screen(screen, monkeypatch)
        projection = await _warm(screen, tasks)
        # Let the real checked display expire; changing its proof clock would
        # select a different refusal route from ordinary resume.
        await asyncio.sleep(projection.max_age + 0.02)
        with _actual_calls() as calls:
            completed = await chat_screen.ChatScreen._sync_native_console_chat_ui(
                screen
            )
        record_property("main_config_entries", calls["main_scopes"])
        record_property("main_native_opens", calls["main_opens"])
        assert completed is True
        assert screen._console_attach_sync_complete
        assert calls["main_scopes"] == 0, calls
        assert calls["main_opens"] == 0, calls
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@contextmanager
def _held_original_config_read():
    """Hold the real read after its second source check, retaining actual leases."""
    entered, release = threading.Event(), threading.Event()
    observed = []
    returns = 0
    main = threading.get_ident()
    old_main, old_thread = sys.getprofile(), threading.getprofile()

    def observe(frame, event, _argument):
        nonlocal returns
        if (
            threading.get_ident() != main
            and event == "return"
            and frame.f_code is checked_config_identity.__code__
        ):
            returns += 1
            if returns == 2:
                observed.append(raw._states[frame.f_locals["active"]])
                entered.set()
                assert release.wait(10), "test did not release original config read"

    threading.setprofile_all_threads(observe)
    try:
        yield entered, release, observed
    finally:
        release.set()
        threading.setprofile_all_threads(old_thread)
        sys.setprofile(old_main)


async def _wait_for_read(entered, full):
    """Observe producer errors instead of waiting only for a never-set event."""
    waiter = asyncio.create_task(asyncio.to_thread(entered.wait, 5))
    try:
        done, _pending = await asyncio.wait(
            (waiter, full), timeout=6, return_when=asyncio.FIRST_COMPLETED
        )
        if full in done:
            await full
            pytest.fail("FULL completed before its checked config read retired")
        assert waiter in done and await waiter, "original config read did not start"
    finally:
        await waiter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["none", "session", "source", "visit", "maintenance"]
)
async def test_full_waits_responsively_and_rechecks_held_original_config(
    tmp_path, monkeypatch, change
):
    database, store, _controller, screen, tasks = _screen(tmp_path)
    full = None
    try:
        effects = _full_screen(screen, monkeypatch)
        store.create_session(session_id="other")
        store.active_session_id = "session-1"
        projection = await _warm(screen, tasks)
        await asyncio.sleep(projection.max_age + 0.02)
        with _held_original_config_read() as (entered, release, observed):
            full = asyncio.create_task(
                chat_screen.ChatScreen._sync_native_console_chat_ui(screen)
            )
            try:
                await _wait_for_read(entered, full)
                state = observed[0]
                assert state.active and projection.pending and projection._reads
                # These turns happen before release: the original native call
                # is blocked, while UI callbacks still get scheduled.
                heartbeat = []
                asyncio.get_running_loop().call_soon(heartbeat.append, True)
                await asyncio.sleep(0)
                assert heartbeat == [True]
                assert not full.done() and "rail" not in effects
                assert not screen._console_attach_sync_complete
                if change == "session":
                    store.active_session_id = "other"
                elif change == "source":
                    identity = config.current_config_identity()
                    monkeypatch.setattr(
                        config,
                        "current_config_identity",
                        lambda: (identity[0] + 1, identity[1]),
                    )
                elif change == "visit":
                    screen._console_attach_visit_generation += 1
                elif change == "maintenance":
                    screen._console_sync_maintenance_paused = True
                release.set()
                completed = await full
                if change == "none":
                    assert completed is True and "rail" in effects
                    assert screen._console_attach_sync_complete
                else:
                    assert completed is False and "rail" not in effects
                    assert not screen._console_attach_sync_complete
                assert not state.active
                with storage._lock:
                    assert all(
                        lease not in storage._live_leases for lease in state.leases
                    )
            finally:
                release.set()
                await asyncio.gather(full, *tasks, return_exceptions=True)
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()


@pytest.mark.asyncio
async def test_cancelled_full_keeps_original_config_read_owned_until_retirement(
    tmp_path, monkeypatch
):
    database, _store, _controller, screen, tasks = _screen(tmp_path)
    try:
        effects = _full_screen(screen, monkeypatch)
        projection = await _warm(screen, tasks)
        await asyncio.sleep(projection.max_age + 0.02)
        with _held_original_config_read() as (entered, release, observed):
            full = asyncio.create_task(
                chat_screen.ChatScreen._sync_native_console_chat_ui(screen)
            )
            try:
                await _wait_for_read(entered, full)
                state = observed[0]
                full.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await full
                assert projection.pending and projection._reads and state.active
                assert not screen._console_attach_sync_complete
                assert "rail" not in effects
                with storage._lock:
                    assert all(lease in storage._live_leases for lease in state.leases)
                release.set()
                await asyncio.gather(*tasks)
                assert not projection.pending and not projection._reads
                assert not state.active
                with storage._lock:
                    assert all(
                        lease not in storage._live_leases for lease in state.leases
                    )
            finally:
                release.set()
                await asyncio.gather(full, *tasks, return_exceptions=True)
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
