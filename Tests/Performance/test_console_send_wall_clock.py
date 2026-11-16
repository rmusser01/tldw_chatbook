"""Measure real Enter and Send events through the full application.

This diagnostic preserves the original profile, typing, provider adapter modes,
turn custody, persistence, trace settlement, heartbeat and teardown. It does not
establish physical terminal feedback or provider response latency. Each route
records its first send and two later sends in a fresh private process, using
Textual's production task factory. The default installs no phase observer.
"""

import asyncio
import contextlib
import json
import os
import time
from dataclasses import replace

import pytest

from Tests.Performance.test_console_native_pause_probe import (
    MAX_CAPTURED_SEND_SECONDS,
    Observation,
    _await_probe_trace_settlement,
)
from Tests.private_profile import private_profile_test


@pytest.mark.asyncio
@pytest.mark.timeout(900)
@pytest.mark.parametrize("route", ["enter", "send-button"])
@private_profile_test
async def test_clean_console_send_wall_clock(monkeypatch, tmp_path, request, route):
    from Tests.Performance.test_console_keystroke_work_census import _scratch_env
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat import console_send_diagnostics
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        ConsoleRunStatus,
    )
    from textual.pilot import Pilot
    from textual import events
    from textual.widgets import Button

    _scratch_env(monkeypatch, tmp_path)
    original_wait = Pilot._wait_for_screen

    async def wait(self, timeout=180):
        return await original_wait(self, timeout=max(timeout, 180))

    monkeypatch.setattr(Pilot, "_wait_for_screen", wait)
    observed = Observation()
    result = {"complete": False, "provider_calls": 0, "route": route}
    heartbeat = None
    phase_probe = None
    loop = asyncio.get_running_loop()
    previous_factory = loop.get_task_factory()
    loop.set_task_factory(asyncio.eager_task_factory)
    try:
        # Deliberately omit native-operation monitoring, audit hooks and stack sampler.
        # This separate diagnostic measures their combined observation cost.
        original_stage = console_send_diagnostics.record_send_stage

        def stage(name, status="entered", **kwargs):
            observed.stages.append(
                dict(
                    phase=observed.phase,
                    stage=name,
                    outcome=status,
                    time=time.perf_counter(),
                )
            )
            return original_stage(name, status, **kwargs)

        monkeypatch.setattr(console_send_diagnostics, "record_send_stage", stage)
        heartbeat = asyncio.create_task(observed.heartbeat())
        app = TldwCli()
        async with app.run_test(size=(140, 42)) as pilot:
            # The existing observer and heartbeat include the original initial
            # task's receipt preparation. Keep this prerequisite inside the
            # original 900s deadline before asserting the completed screen.
            while not getattr(app, "_initial_screen_pushed", False):
                await asyncio.sleep(0.01)
            screen = app.screen
            assert type(screen).__name__ == "ChatScreen"
            composer = screen._console_composer_or_none()
            assert composer is not None
            controller = screen._ensure_console_chat_controller()
            gateway = controller.provider_gateway
            from tldw_chatbook.Chat.console_provider_gateway import (
                _CONSOLE_PROVIDER_RESOLUTION_NATIVE_METHODS,
            )
            from tldw_chatbook.Chat.console_received_dispatch import stock_native_methods

            def adapter(**_kwargs):
                result.setdefault("send_to_adapter_seconds", []).append(
                    time.perf_counter() - send_started
                )
                result["provider_calls"] += 1
                streaming = bool(_kwargs.get("streaming"))
                result.setdefault("provider_streaming", []).append(streaming)
                if streaming:
                    common = {
                        "id": "native-pause",
                        "object": "chat.completion.chunk",
                        "created": 1,
                        "model": "gpt-4o",
                    }
                    events = [
                        {
                            **common,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": "Immediate native probe reply",
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        },
                        {
                            **common,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "stop"}
                            ],
                        },
                    ]
                    return iter(
                        [
                            *(
                                "data: " + json.dumps(event) + "\n\n"
                                for event in events
                            ),
                            "data: [DONE]\n\n",
                        ]
                    )
                return {
                    "choices": [
                        {"message": {"content": "Immediate native probe reply"}}
                    ]
                }

            monkeypatch.setattr(gateway, "_chat_api_call_fn", adapter)
            runtime = screen._console_runtime()
            tasks = []
            custody = []
            received = asyncio.Event()

            def observe_custody(submission, turn_id):
                record = runtime._turn_custody[turn_id]
                assert record.session_id == submission.session_id
                assert record.turn_id == turn_id == submission.turn_id
                assert isinstance(record.task, asyncio.Task)
                hooks = screen._hooks
                custody.append(
                    (
                        submission.session_id,
                        turn_id,
                        record.task,
                        hooks,
                        hooks.pending_send_identity,
                        time.perf_counter(),
                    )
                )
                tasks.append(record.task)
                received.set()

            original_accept = runtime.accept_turn

            def accept(request, **kwargs):
                turn_id = original_accept(request, **kwargs)
                observe_custody(request, turn_id)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", accept)
            original_receive = runtime.accept_received_intent

            def receive(intent, **kwargs):
                turn_id = original_receive(intent, **kwargs)
                observe_custody(intent, turn_id)
                return turn_id

            monkeypatch.setattr(runtime, "accept_received_intent", receive)
            with observed.phase_scope("idle"):
                await asyncio.sleep(3)
            # Direct ticks time only the actual callback, avoiding Pilot cost.
            with observed.phase_scope("credential_poll"):
                for _ in range(8):
                    screen._poll_console_credential_readiness()
            with observed.phase_scope("typing"):
                for _ in range(8):
                    await pilot.press("a")
                await asyncio.sleep(0.5)
            trace_store = controller.store
            if os.environ.get("TLDW_SEND_PHASE_PROBE") == "1":
                from Tests.Performance.console_send_phase_probe import SendPhaseProbe

                phase_probe = SendPhaseProbe(
                    controller, gateway, lambda: observed.phase
                )
                phase_probe.start()
            for index in range(1, 4):
                # A completed first reply can still own its one-key onboarding
                # save. The original capture gate refuses new sends during that
                # publication. Report its settlement separately, then measure an
                # eligible warm send without changing that production gate.
                publication = runtime._first_send_task
                publication_started = time.perf_counter()
                publication_pending = publication is not None and not publication.done()
                if publication_pending:
                    await asyncio.shield(publication)
                result.setdefault("between_send_publication_seconds", []).append(
                    {
                        "send": index,
                        "pending": publication_pending,
                        "seconds": time.perf_counter() - publication_started,
                    }
                )
                assert not runtime._first_send_publishing
                session_id = controller.store.active_session_id
                settings = controller.store.session_settings(session_id)
                assert settings is not None
                streaming = index == 3
                if settings.streaming != streaming:
                    # Use real session state. Wrapping resolve_for_send opts out
                    # of the production native-read ownership/handoff path.
                    controller.store.replace_session_settings(
                        session_id, replace(settings, streaming=streaming)
                    )
                assert stock_native_methods(
                    gateway, _CONSOLE_PROVIDER_RESOLUTION_NATIVE_METHODS
                )
                composer = screen._console_composer_or_none()
                assert composer is not None
                composer.load_draft(f"Native pause probe message {index}")
                composer.focus()
                await pilot.pause()
                send_button = screen.query_one("#console-send-message", Button)
                assert app.focused is composer
                if composer._send_blocked or send_button.disabled:
                    # Failure-only resident facts: no readiness derivation, new
                    # imports, forced sync, waits, or configuration reads.
                    projection = getattr(screen, "_console_readiness_config_projection", None)
                    ack = getattr(screen, "_console_send_ack", None)
                    pending_ack = getattr(ack, "_pending", {}).get(session_id)
                    flight = getattr(screen, "_console_send_flight", None)
                    run = controller._run_states.get(session_id)
                    queue = controller.prompt_queue_registry._states.get(session_id)
                    cadence = getattr(screen, "_console_config_cadence", None)
                    result["pre_action_readiness_failure"] = {
                        "send": index,
                        "perf_counter": time.perf_counter(),
                        "composer_blocked": composer._send_blocked,
                        "button_disabled": send_button.disabled,
                        "send_label": str(composer._send_label)[:80],
                        "disabled_reason": str(composer._send_disabled_reason)[:300],
                        "setup_code": str(composer._setup_blocked_reason.code)[:80],
                        "setup_reason": str(composer._setup_blocked_reason.text)[:300],
                        "queue_reason": str(composer._queue_blocked_reason)[:300],
                        "attachment_block": composer._attachment_block is not None,
                        "dispatch_recovery_block": composer._dispatch_recovery_blocked,
                        "wake_turn_active": composer._wake_turn_active,
                        "cached_run_active": composer._run_active,
                        "run_status": getattr(getattr(run, "status", None), "value", None),
                        "queue_mode": getattr(getattr(queue, "mode", None), "value", None),
                        "queue_waiting": len(getattr(queue, "waiting", ())),
                        "queue_claimed": getattr(queue, "claimed", None) is not None,
                        "queue_reservation": getattr(getattr(queue, "reservation", None), "value", None),
                        "first_send_publishing": runtime._first_send_publishing,
                        "first_send_task_done": publication.done() if publication is not None else None,
                        "same_view": runtime.view is screen,
                        "same_controller": runtime.chat_controller is controller,
                        "same_store": runtime.chat_store is controller.store,
                        "same_visible_draft_session": screen._console_visible_draft_session_id == session_id,
                        "custody_count": len(runtime._turn_custody),
                        "pending_ack": pending_ack is not None,
                        "ack_admitted": getattr(pending_ack, "admitted", None),
                        "ack_held": getattr(pending_ack, "held", None),
                        "ack_standing_in": session_id in getattr(ack, "_standing_in", {}),
                        "unfinished_send_flights": sum(not task.done() for task in getattr(flight, "tasks", ())),
                        "pending_visible_send": screen._console_pending_send is not None,
                        "sync_in_progress": screen._console_sync_in_progress,
                        "sync_requested": screen._console_sync_requested,
                        "tab_sync_calls": screen._console_session_tabs_sync_calls,
                        "replay_whole_sync": getattr(screen, "_console_control_bar_replay_whole_sync", False),
                        "cadence_owed": getattr(cadence, "owed", None),
                        "run_poll_present": screen._console_transcript_sync_timer is not None,
                        "readiness_pending": getattr(projection, "pending", None),
                        "readiness_has_value": getattr(projection, "value", None) is not None,
                        "readiness_retry_delay": getattr(projection, "_retry_delay", None),
                        "unfinished_workers": [
                            (worker.group, worker.state.name)
                            for worker in [
                                worker for worker in app.workers
                                if worker.node is screen and not worker.is_finished
                            ][:8]
                        ],
                    }
                assert not composer._send_blocked
                assert not send_button.disabled
                send_deadline = time.perf_counter() + MAX_CAPTURED_SEND_SECONDS
                with observed.phase_scope(f"send_{index}"):
                    screen._session._sync_console_session_draft()
                    before = len(tasks)
                    received.clear()
                    session_id = controller.store.active_session_id
                    hooks = screen._hooks
                    send_started = time.perf_counter()
                    if route == "enter":
                        key = events.Key("enter", "\r")
                        key.set_sender(app)
                        app._driver.send_message(key)
                    else:
                        send_button.press()
                    await asyncio.wait_for(
                        received.wait(), max(0, send_deadline - time.perf_counter())
                    )
                    assert screen.is_mounted and runtime.view is screen
                    assert screen._hooks is hooks
                    assert controller.store.active_session_id == session_id
                    assert len(tasks) == before + 1, "send never reached turn custody"
                    (
                        accepted_session,
                        turn_id,
                        task,
                        owner,
                        identity,
                        accepted_at,
                    ) = custody[-1]
                    assert accepted_session == session_id and task is tasks[-1]
                    assert all(prior[1] != turn_id for prior in custody[:before])
                    assert owner is hooks
                    assert identity is None or identity[0] == session_id
                    assert accepted_at <= send_deadline
                    await asyncio.wait_for(asyncio.shield(task), 180)
                    assert controller.run_state.status is ConsoleRunStatus.COMPLETED
                    await screen._sync_native_console_chat_ui()
                    await asyncio.sleep(0.25)
                    assert controller.store is trace_store
                    await _await_probe_trace_settlement(
                        trace_store, deadline=send_deadline
                    )
                    assert controller.store is trace_store
            if phase_probe is not None:
                phase_probe.stop()
            messages = controller.store.messages_for_session(
                controller.store.active_session_id
            )
            result["user_messages"] = sum(
                m.role is ConsoleMessageRole.USER for m in messages
            )
            result["assistant_messages"] = sum(
                m.role is ConsoleMessageRole.ASSISTANT for m in messages
            )
            result["file_backed_database"] = (
                str(app.chachanotes_db.db_path) != ":memory:"
            )
            assert result["file_backed_database"]
            assert (
                result["user_messages"]
                == result["assistant_messages"]
                == result["provider_calls"]
                == 3
            )
            with app.chachanotes_db.transaction() as cursor:
                result["trace_states"] = [
                    row[0]
                    for row in cursor.execute("SELECT state FROM console_trace_calls")
                ]
                result["response_links"] = cursor.execute(
                    "SELECT COUNT(*) FROM console_trace_response_links"
                ).fetchone()[0]
                result["dispatch_checkpoints"] = cursor.execute(
                    "SELECT COUNT(*) FROM console_dispatch_checkpoints"
                ).fetchone()[0]
            assert result["trace_states"] == ["complete"] * 3
            assert result["response_links"] == 3
            assert result["dispatch_checkpoints"] == 0
            assert result["provider_streaming"] == [False, False, True]
            result["stock_provider_resolution"] = stock_native_methods(
                gateway, _CONSOLE_PROVIDER_RESOLUTION_NATIVE_METHODS
            )
            assert result["stock_provider_resolution"]
            # The unchanged real bar must now avoid the previously sampled
            # stylesheet cascade. Retain the same eight-call diagnostic control
            # for before/after comparison; dedicated mounted tests verify paint.
            from tldw_chatbook.Widgets.Console.console_control_bar import (
                ConsoleControlBar,
            )

            bar = screen.query_one(ConsoleControlBar)
            bar._set_recovery_height(False)

            def layout_state():
                return (
                    bar.classes,
                    bar.styles.height,
                    bar.styles.min_height,
                    bar.styles.max_height,
                )

            expected_layout = layout_state()
            with observed.phase_scope("unchanged_layout"):
                for _ in range(8):
                    bar._set_recovery_height(False)
            assert layout_state() == expected_layout
            with observed.phase_scope("unchanged_layout_control"):
                for _ in range(8):
                    if layout_state() != expected_layout:
                        bar._set_recovery_height(False)
            assert layout_state() == expected_layout
            result["diagnostic_only"] = True
            result["native_work_census"] = False
            # Original acceptance test and its budgets remain unchanged.
            result["complete"] = True
            observed.phase = "shutdown"
    finally:
        try:
            observed.stop.set()
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            sampler = getattr(observed, "sampler", None)
            if sampler is not None:
                sampler.join(timeout=2)
        finally:
            try:
                observed.stop_seams()
            finally:
                try:
                    if phase_probe is not None:
                        phase_probe.stop()
                        result["send_phase_probe"] = phase_probe.report()
                finally:
                    loop.set_task_factory(previous_factory)
                    observed.write(result)
