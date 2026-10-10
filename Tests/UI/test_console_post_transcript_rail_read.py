"""The original post-transcript rail refresh keeps checked display ownership."""

import asyncio
import sys
import threading
from contextlib import nullcontext
from types import MethodType, SimpleNamespace

import pytest

from Tests.UI.test_console_checked_display_scope import _actual_calls, _screen, _warm
from tldw_chatbook.Widgets.Console import (
    console_project_instructions as project_instruction_ui,
)
from tldw_chatbook.Backup_Recovery import (
    raw_participants as raw,
    storage_admission as storage,
)
from tldw_chatbook.UI.Console_Modules import console_spend_projection as spend
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Console_Modules.attach_visit import ConsoleAttachVisit

pytestmark = pytest.mark.bootstrap_profile


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["none", "before-controls", "after-transcript", "cold-read"]
)
async def test_final_rail_refresh_after_transcript_rechecks_display_owner(
    tmp_path, monkeypatch, mutation
):
    database, store, controller, screen, tasks = _screen(tmp_path)
    stage, mappings, published, timers = ["before"], [], [], []
    replaced = False
    transcript_called = asyncio.Event()
    roleplay_calls = []
    rail = object()
    calls = None
    display_native_counts = []

    def noop(*_args, **_kwargs):
        pass

    async def settled(*_args, **_kwargs):
        await asyncio.sleep(0)

    def current_rail():
        before = (calls["main_scopes"], calls["main_opens"]) if calls else None
        mappings.append((stage[0], screen._provider_readiness_app_config()))
        if before is not None:
            display_native_counts.append(
                (calls["main_scopes"] - before[0], calls["main_opens"] - before[1])
            )
        return rail

    def replace_once(where):
        nonlocal replaced
        if mutation == where and not replaced:
            screen.app_instance.app_config = dict(screen.app_instance.app_config)
            replaced = True

    async def transcript():
        await asyncio.sleep(0)
        replace_once("after-transcript")
        assert roleplay_calls, "transcript preceded the live roleplay refresh"
        stage[0] = "after"
        transcript_called.set()

    screen._console_sync_in_progress = screen._console_sync_requested = False
    # This display-only fixture owns no mounted runtime attachment.
    screen._current_console_attach_visit = lambda: ConsoleAttachVisit(
        None, screen, None, 0, False
    )
    screen._console_attach_sync_complete = False
    screen._console_chat_controller = controller
    screen._message = SimpleNamespace(reconcile_console_speech_context=noop)
    screen._image = SimpleNamespace(
        _h3_image_edit_registry=lambda: SimpleNamespace(drop_session=noop),
        _reconcile_h3_image_edit_completions=noop,
    )
    screen._session = SimpleNamespace(
        _sync_console_session_draft=noop,
        schedule_manual_read_acknowledgement=noop,
        _maybe_refresh_stale_default_console_settings=noop,
    )
    screen._retrieval = SimpleNamespace(
        _warm_console_effective_scope_cache_if_stale=settled,
        _refresh_active_dictionaries_summary_if_scope_changed=settled,
        _refresh_active_world_books_summary_if_scope_changed=settled,
    )
    screen._character = SimpleNamespace(
        _refresh_active_character_avatar_if_scope_changed=settled
    )
    screen._character_context = SimpleNamespace(
        refresh_presentation_if_scope_changed=settled
    )
    screen._workspace = SimpleNamespace(tick_workspace_build_scope=nullcontext)
    for name in (
        "_record_ui_worker_started",
        "_record_ui_worker_finished",
        "_sync_console_chat_core_state",
        "_sync_console_settings_summary",
        "_sync_console_control_bar_under_config",
        "_sync_console_settings_recovery_surfaces",
        "_sync_console_live_work_readiness_rows",
        "_sync_console_mode_bar",
        "_dispatch_active_console_roleplay_refresh",
        "_sync_console_workspace_context",
        "_dispatch_console_rail_preference_prune",
    ):
        setattr(screen, name, noop)
    screen._dispatch_active_console_roleplay_refresh = lambda: roleplay_calls.append(
        True
    )
    screen._sync_console_native_session_tabs = settled
    screen._sync_native_console_transcript = transcript
    screen._current_console_rail_state = current_rail
    screen._sync_console_rail_visibility_if_changed = published.append
    screen._run_console_config_sync = MethodType(
        ChatScreen._run_console_config_sync, screen
    )

    def rail_and_controls():
        replace_once("before-controls")
        return ChatScreen._sync_console_rail_and_controls(screen)

    screen._sync_console_rail_and_controls = rail_and_controls
    monkeypatch.setattr(
        project_instruction_ui, "sync_project_instruction_status_for_screen", noop
    )
    try:
        projection = await _warm(screen, tasks)
        for name in (
            "_request_console_control_bar_sync",
            "_run_coalesced_control_bar_sync",
            "_sync_native_console_chat_ui",
        ):
            setattr(screen, name, MethodType(getattr(ChatScreen, name), screen))
        screen.set_timer = lambda delay, callback: timers.append((delay, callback))
        screen.call_after_refresh = lambda callback: timers.append((None, callback))
        if mutation == "cold-read":
            # Hold the return of the original checked reader after its native
            # scope retires. This isolates display latency from live authority
            # checks: no reader, lock, permission or data result is replaced.
            entered, release = threading.Event(), threading.Event()
            reader, reader_code = (
                projection.read_current,
                projection.read_current.__code__,
            )
            previous_main, previous_thread = sys.getprofile(), threading.getprofile()
            assert previous_main is None and previous_thread is None
            observer_failures = []

            def observe(frame, event, _argument):
                if (
                    event == "return"
                    and frame.f_code is reader_code
                    and frame.f_locals.get("projection") is projection
                    and not entered.is_set()
                ):
                    # A profile "return" also reports an exceptional exit
                    # with None. Only a successful original checked result can
                    # establish this intervention point.
                    result = _argument
                    if type(result) is not spend.ConsoleReadinessConfigRead:
                        observer_failures.append(
                            "checked reader did not return a result"
                        )
                        return
                    proof = result._display_proof
                    request = projection._read_request
                    if not (
                        type(proof) is spend._CheckedDisplayProof
                        and request is not None
                        and proof.projection is projection
                        and proof.screen is screen
                        and proof.reader is reader
                        and proof.owner is request[2]
                        and proof.value is result.value
                        and result.source_before
                        == result.source_after
                        == proof.source
                        == request[2][0]
                    ):
                        observer_failures.append("checked reader proof did not match")
                        return
                    active = frame.f_locals.get("active")
                    with storage._lock:
                        if active is None or active in raw._states:
                            observer_failures.append(
                                "checked operation had not retired"
                            )
                            return
                    entered.set()
                    if not release.wait(10):
                        observer_failures.append("checked reader hold expired")

            sync = None
            threading.setprofile_all_threads(observe)
            try:
                # A real owner replacement makes the prior display ineligible.
                screen.app_instance.app_config = dict(screen.app_instance.app_config)
                assert projection.run(lambda: None) is False
                assert await asyncio.to_thread(entered.wait, 10)
                assert projection.pending and not projection._settled.is_set()
                sync = asyncio.create_task(
                    ChatScreen._sync_native_console_chat_ui(screen)
                )
                await asyncio.wait_for(transcript_called.wait(), 2)
                assert not release.is_set() and projection.pending
                assert roleplay_calls == [True]
                assert not published, "cold settings published a rail"
            finally:
                release.set()
                try:
                    if sync is not None:
                        await sync
                    await asyncio.gather(*tasks, return_exceptions=True)
                finally:
                    threading.setprofile_all_threads(previous_thread)
                    sys.setprofile(previous_main)
            assert not observer_failures
            assert projection.read_current is reader and reader.__code__ is reader_code
            return

        # Use the full original asynchronous refresh and its actual checked
        # source/native readers. Only unrelated UI regions are inert here.
        with _actual_calls() as calls:
            await ChatScreen._sync_native_console_chat_ui(screen)
        after = [mapping for phase, mapping in mappings if phase == "after"]
        assert stage == ["after"], "cold rail prevented transcript publication"
        assert roleplay_calls == [True]
        # Core and roleplay deliberately retain their two fresh live operations.
        # The display callbacks must add no native scope or filesystem work.
        assert calls["main_scopes"] == 2, calls
        assert calls["main_opens"] > 0, calls
        assert all(counts == (0, 0) for counts in display_native_counts)
        if mutation != "none":
            assert not after and not published
            assert screen._console_control_bar_replay_whole_sync
            # A cold projection can decline before the wrapped native-entry
            # retry runs. The full refresh must schedule its own trailing run.
            assert len(timers) == 1
            delay, callback = timers.pop()
            assert delay == 0.2
            assert callback.__func__ is ChatScreen._run_coalesced_control_bar_sync
            await asyncio.gather(*tasks)
            assert not published, "background read bypassed the pending replay"
            callback()
            await asyncio.gather(*tasks)
            assert published == [rail]
            assert mappings[-1][1] is projection.value
            assert projection.key == projection._key()
            assert not screen._console_control_bar_replay_whole_sync
            assert not screen._console_control_bar_sync_scheduled
            assert not screen._console_sync_requested
            assert not timers
        else:
            assert after == [projection.value, projection.value]
            assert after[0] is projection.value
            assert published == [rail]
            assert not timers
        assert not screen._console_sync_in_progress
    finally:
        await asyncio.gather(*tasks, return_exceptions=True)
        database.close()
