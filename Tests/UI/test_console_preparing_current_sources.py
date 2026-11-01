"""Actual received-owner positives and source/custom polling boundaries."""

import sys
from types import MethodType

import pytest

from Tests.UI.test_console_hook_review_send_freeze import _until
from Tests.UI.test_console_poll_reconciliation import (
    _OriginalPreparingPolls,
    _qualify_warm_capture_sources,
)
from Tests.UI.test_console_received_intent_feedback import (
    _attribute_original_received_read,
    _held_received_record,
    _received_console_case,
    _send,
)
from tldw_chatbook.UI.Console_Modules import composer_run_controls as controls

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_preparing_defining_sources_decline_changed_callbacks(
    monkeypatch, record_property
):
    """A naturally qualified receipt cannot skip newly changed producers."""
    async with _received_console_case(
        monkeypatch, "preparing-defining-sources", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        _send(case, "enter")
        record = await _held_received_record(case)
        assert await _until(
            lambda: controls.preparing_poll_record(case.console) is record, 5
        )
        screen = case.console
        callbacks = (
            (screen, "_sync_console_chat_core_state"),
            (screen, "_dispatch_active_console_roleplay_refresh"),
            (screen._session, "_build_console_turn_execution_context"),
            (case.controller, "_provider_selection"),
        )
        qualified = []
        # These synchronous interventions cannot allow the held callback to
        # advance or fabricate a completed FULL witness between controls.
        for owner, name in callbacks:
            original = getattr(owner, name)
            function = original.__func__
            for field, replacement in (
                ("__code__", function.__code__.replace()),
                ("__defaults__", (None,)),
                ("__kwdefaults__", {"changed_source": None}),
            ):
                with monkeypatch.context() as changed:
                    changed.setattr(function, field, replacement)
                    assert controls.preparing_poll_record(screen) is None, (name, field)
                assert controls.preparing_poll_record(screen) is record
                qualified.append((name, field))

            def custom(self, callback=original):
                return callback()

            with monkeypatch.context() as changed:
                changed.setattr(owner, name, MethodType(custom, owner))
                assert controls.preparing_poll_record(screen) is None, name
            assert controls.preparing_poll_record(screen) is record
            qualified.append((name, "instance_callback"))
        for name in ("poll_source", "poll_source_is_current"):
            function = getattr(controls, name)
            with monkeypatch.context() as changed:
                changed.setattr(function, "__code__", function.__code__.replace())
                assert controls.preparing_poll_record(screen) is None, name
            assert controls.preparing_poll_record(screen) is record
            qualified.append((name, "code"))
        record_property("qualified_current_source_mutations", qualified)
        assert case.provider_calls == []
        assert case.composer.draft_text() == case.draft
        assert case.probe.live_at_entry and not case.probe.release_timed_out


async def test_preparing_custom_core_keeps_natural_full_no_argument_route(
    monkeypatch, record_property
):
    """An actual custom callback remains live on the next natural timer pass."""
    async with _received_console_case(
        monkeypatch, "preparing-custom-core", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        _send(case, "send-button")
        record = await _held_received_record(case)
        assert await _until(
            lambda: controls.preparing_poll_record(case.console) is record, 5
        )
        original = case.console._sync_console_chat_core_state
        calls = []

        def custom(self):
            assert self is case.console
            calls.append(self)
            return original()

        observed = _OriginalPreparingPolls(case, record)
        with monkeypatch.context() as changed, observed.installed():
            changed.setattr(
                case.console,
                "_sync_console_chat_core_state",
                MethodType(custom, case.console),
            )
            assert controls.preparing_poll_record(case.console) is None
            assert await _until(lambda: bool(observed.qualified_indexes()), 5)
        rows = observed.completed
        record_property("custom_core_original_poll_rows", rows)
        assert calls
        assert any(row["core_returns"] > 0 for row in rows)
        assert case.provider_calls == []
        assert case.composer.draft_text() == case.draft
        assert case.probe.live_at_entry and not case.probe.release_timed_out


@pytest.mark.parametrize(
    "draft", ["A newer authored draft", ""], ids=["newer-draft", "authored-empty"]
)
async def test_preparing_authored_input_returns_to_natural_full(
    monkeypatch, record_property, draft
):
    """A real composer edit invalidates narrowing without consuming that edit."""
    async with _received_console_case(
        monkeypatch, f"preparing-input-{bool(draft)}", durable=True
    ) as case:
        await _qualify_warm_capture_sources(case)
        _send(case, "enter")
        record = await _held_received_record(case)
        assert await _until(
            lambda: controls.preparing_poll_record(case.console) is record, 5
        )
        observed = _OriginalPreparingPolls(case, record)
        with observed.installed():
            case.composer.load_draft(draft)
            assert not case.store.session_inputs_are_current(
                record.received_intent.inputs
            )
            assert controls.preparing_poll_record(case.console) is None
            assert await _until(lambda: bool(observed.qualified_indexes()), 5)
        record_property("authored_input_original_poll_rows", observed.completed)
        assert any(row["core_returns"] > 0 for row in observed.completed)
        assert case.composer.draft_text() == draft
        assert case.store.session_draft(case.session.id) == draft
        assert case.provider_calls == []
        assert case.probe.live_at_entry and not case.probe.release_timed_out


@pytest.mark.parametrize("queued_full", [False, True], ids=["natural", "queued-full"])
async def test_preparing_changed_source_after_original_tab_await_replays_full(
    monkeypatch, record_property, queued_full
):
    """A source change while the real surface lock is held cannot publish success."""
    import asyncio

    from Tests.UI.test_console_poll_reconciliation import (
        _assert_original_configuration_custody,
        _original_transition_events,
        _transition_parent,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        ConsoleSessionSurface,
    )

    async with _received_console_case(
        monkeypatch, f"preparing-held-source-{queued_full}", durable=True
    ) as case:
        _attribute_original_received_read(case, record_property)
        await _qualify_warm_capture_sources(case)
        _send(case, "enter")
        record = await _held_received_record(case)
        console = case.console
        setup_deadline = asyncio.get_running_loop().time() + 5
        assert await _until(
            lambda: (
                controls.preparing_poll_record(console) is record
                and not console._console_sync_in_progress
                and console._console_session_tabs_sync_calls == 0
            ),
            max(0, setup_deadline - asyncio.get_running_loop().time()),
        )
        surface = console.query_one("#console-session-surface", ConsoleSessionSurface)
        lock = surface._session_sync_lock
        timer = console._console_transcript_sync_timer
        assert timer is not None
        full_code = ChatScreen._sync_native_console_chat_ui.__code__
        held, results, calls = {}, [], []
        active_full = {}
        original = console._sync_console_chat_core_state
        hold_started = None
        timeline, timeline_dropped = [], 0
        observed_timer_task = timer._task

        def observe_hold(kind, label, frame, value):
            nonlocal timeline_dropped
            owner = frame.f_locals.get("self")
            if owner is not console and owner is not surface:
                return
            if (
                label in {"record", "source"}
                and kind != sys.monitoring.events.PY_RETURN
            ):
                return
            if len(timeline) == 512:
                timeline_dropped += 1
                return
            parent = (
                _transition_parent(frame, full_code, console)
                if label == "surface"
                else None
            )
            is_waiter = type(value) is asyncio.Future and any(
                waiter is value for waiter in (lock._waiters or ())
            )
            timeline.append(
                (
                    round((asyncio.get_running_loop().time() - hold_started) * 1000),
                    label,
                    {
                        sys.monitoring.events.PY_START: "start",
                        sys.monitoring.events.PY_YIELD: "yield",
                        sys.monitoring.events.PY_RETURN: "return",
                    }[kind],
                    frame.f_lineno,
                    asyncio.current_task() is observed_timer_task,
                    value is record if label == "record" else value is True,
                    frame.f_locals.get("poll_record") is record,
                    parent is not None and parent.f_locals.get("poll_record") is record,
                    is_waiter,
                    is_waiter and not value.done(),
                )
            )

        def hold_failure():
            def task_state(task):
                if task is None:
                    return {"present": False}
                error = getattr(task, "_exception", None)
                waiter = getattr(task, "_fut_waiter", None)
                return {
                    "present": True,
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                    "error_type": None if error is None else type(error).__name__,
                    "surface_waiter": any(
                        item is waiter for item in (lock._waiters or ())
                    ),
                }

            now = asyncio.get_running_loop().time()
            details = {
                "hold_elapsed_ms": round((now - hold_started) * 1000),
                "setup_budget_remaining_ms": round((setup_deadline - now) * 1000),
                "setup_remaining_at_hold_ms": round(
                    (setup_deadline - hold_started) * 1000
                ),
                "probe_entered": case.probe.entered.is_set(),
                "probe_release_requested": case.probe.release.is_set(),
                "probe_release_timed_out": case.probe.release_timed_out,
                "probe_thread_alive": (
                    case.probe.thread is not None and case.probe.thread.is_alive()
                ),
                "probe_physically_retired": case.probe.retired(),
                "timer_same": console._console_transcript_sync_timer is timer,
                "timer_task_same": timer._task is observed_timer_task,
                "timer_active": timer._active.is_set(),
                "timer_task": task_state(observed_timer_task),
                "current_timer_task": task_state(timer._task),
                "record_task": task_state(record.task),
                "record_still_custodied": any(
                    item is record for item in case.runtime._turn_custody.values()
                ),
                "screen_closing": console._closing,
                "screen_closed": console._closed,
                "runtime_disposed": case.runtime._disposed,
                "maintenance_paused": vars(console).get(
                    "_console_sync_maintenance_paused", False
                ),
                "attach_reconciled": console._console_attach_reconciled,
                "attach_running": console._console_attach_reconcile_running,
                "sync_in_progress": console._console_sync_in_progress,
                "sync_requested": console._console_sync_requested,
                "whole_replay": vars(console).get(
                    "_console_control_bar_replay_whole_sync", False
                ),
                "tab_calls": console._console_session_tabs_sync_calls,
                "completed_source_present": console._console_completed_full_source
                is not None,
                "lock_owned_by_test": owned_lock,
                "lock_locked": lock.locked(),
                "lock_waiters": len(lock._waiters or ()),
                "lock_pending_waiters": sum(
                    not item.done() for item in (lock._waiters or ())
                ),
                "timeline": timeline,
                "timeline_dropped": timeline_dropped,
            }
            record_property("preparing_tab_hold_failure", details)
            # CI retains captured stderr even without a JUnit artifact; this
            # bounded state contains flags and source locations, never bodies.
            import json

            print(
                "P0_PREPARING_TAB_HOLD_FAILURE "
                + json.dumps(details, sort_keys=True, separators=(",", ":")),
                file=sys.stderr,
                flush=True,
            )
            return details

        def custom(self):
            assert self is console
            calls.append(asyncio.current_task())
            return original()

        def observe(kind, label, frame, value):
            observe_hold(kind, label, frame, value)
            if label == "surface" and frame.f_locals.get("self") is surface:
                if kind == sys.monitoring.events.PY_YIELD and not held:
                    full = _transition_parent(frame, full_code, console)
                    if full is None or full.f_locals.get("poll_record") is not record:
                        return
                    assert asyncio.current_task() is timer._task
                    assert type(value) is asyncio.Future and not value.done()
                    assert any(waiter is value for waiter in lock._waiters)
                    held.update(
                        full=active_full[id(full)],
                        task=asyncio.current_task(),
                        future=value,
                    )
            elif label == "full" and frame.f_locals.get("self") is console:
                if kind == sys.monitoring.events.PY_START:
                    # Frame addresses may be reused after return. Retain only a
                    # per-invocation token, never the original frame itself.
                    active_full[id(frame)] = object()
                elif kind == sys.monitoring.events.PY_RETURN:
                    identity = active_full.pop(id(frame), None)
                    if held:
                        assert identity is not None
                        results.append(
                            (
                                identity,
                                frame.f_locals.get("poll_record") is record,
                                value,
                            )
                        )

        completed_display = []

        def completed_original_display(kind, label, frame, value):
            if (
                kind == sys.monitoring.events.PY_RETURN
                and frame.f_locals.get("self") is console
                and frame.f_locals.get("poll_record") is record
                and value is True
                and asyncio.current_task() is timer._task
            ):
                completed_display.append(True)

        with _original_transition_events(
            [("completed_display", ChatScreen, "_sync_native_console_chat_ui")],
            completed_original_display,
        ):
            if queued_full:
                # Sampled idle fields do not retire an already queued FULL.
                # Keep its original worker and publication path intact.
                asyncio.get_running_loop().call_soon(
                    lambda: console.run_worker(
                        console._sync_native_console_chat_ui(),
                        exclusive=True,
                        group="console-sync",
                    )
                )
            assert await _until(
                lambda: (
                    bool(completed_display)
                    and controls.preparing_poll_record(console) is record
                    and not console._console_sync_in_progress
                    and console._console_session_tabs_sync_calls == 0
                    and not getattr(
                        console, "_console_control_bar_replay_whole_sync", False
                    )
                ),
                max(0, setup_deadline - asyncio.get_running_loop().time()),
            ), "No original natural Preparing display completed before the tab hold"
        assert type(lock) is asyncio.Lock and not lock.locked()
        await lock.acquire()
        owned_lock = True
        hold_started = asyncio.get_running_loop().time()
        try:
            with (
                monkeypatch.context() as changed,
                _original_transition_events(
                    [
                        ("surface", ConsoleSessionSurface, "sync_sessions"),
                        ("full", ChatScreen, "_sync_native_console_chat_ui"),
                        ("record", controls, "preparing_poll_record"),
                        ("source", controls, "poll_source_is_current"),
                    ],
                    observe,
                ),
            ):
                found_hold = await _until(lambda: bool(held), 5)
                assert found_hold, hold_failure()
                _assert_original_configuration_custody(case)
                assert not console._console_sync_requested
                changed.setattr(
                    console,
                    "_sync_console_chat_core_state",
                    MethodType(custom, console),
                )
                assert controls.preparing_poll_record(console) is None
                assert not held["future"].done() and not calls
                lock.release()
                owned_lock = False
                assert await _until(
                    lambda: (
                        any(
                            identity is held["full"] and narrowed and value is False
                            for identity, narrowed, value in results
                        )
                        and any(
                            identity is not held["full"]
                            and not narrowed
                            and value is True
                            for identity, narrowed, value in results
                        )
                        and not console._console_sync_in_progress
                        and not console._console_sync_requested
                    ),
                    5,
                ), results
                assert calls and all(task is not held["task"] for task in calls)
                assert case.composer.draft_text() == case.draft
                _assert_original_configuration_custody(case)
                record_property(
                    "changed_source_full_results",
                    [(narrowed, value) for _, narrowed, value in results],
                )
                record_property("changed_source_custom_full_calls", len(calls))
        finally:
            if owned_lock:
                lock.release()
