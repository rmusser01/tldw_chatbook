"""Matched full-app diagnostic with only the heavy work observer omitted.

This diagnostic preserves the original profile, typing, provider adapter modes,
turn custody, persistence, trace settlement, heartbeat and teardown. It does not
establish rendered feedback or speed acceptance. The original budget test stays
unchanged and authoritative. Set TLDW_SEND_PHASE_PROBE=1 to add bounded scalar
phase spans to the existing probe result; the default installs no phase observer.
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
@private_profile_test
async def test_clean_console_send_wall_clock(monkeypatch, tmp_path, request):
    from Tests.Performance.test_console_keystroke_work_census import _scratch_env
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.Chat import console_send_diagnostics
    from tldw_chatbook.Chat.console_chat_models import (
        ConsoleMessageRole,
        ConsoleRunStatus,
    )
    from textual.pilot import Pilot

    _scratch_env(monkeypatch, tmp_path)
    original_wait = Pilot._wait_for_screen

    async def wait(self, timeout=180):
        return await original_wait(self, timeout=max(timeout, 180))

    monkeypatch.setattr(Pilot, "_wait_for_screen", wait)
    observed = Observation()
    result = {"complete": False, "provider_calls": 0}
    heartbeat = None
    phase_probe = None
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
            original_resolve = gateway.resolve_for_send

            async def resolve(selection):
                return replace(
                    await original_resolve(selection), streaming=len(tasks) >= 3
                )

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

            monkeypatch.setattr(gateway, "resolve_for_send", resolve)
            monkeypatch.setattr(gateway, "_chat_api_call_fn", adapter)
            runtime = screen._console_runtime()
            tasks = []
            custody = []

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
                send_deadline = time.perf_counter() + MAX_CAPTURED_SEND_SECONDS
                with observed.phase_scope(f"send_{index}"):
                    screen._session._sync_console_session_draft()
                    composer.load_draft(f"Native pause probe message {index}")
                    before = len(tasks)
                    session_id = controller.store.active_session_id
                    hooks = screen._hooks
                    send_started = time.perf_counter()
                    await screen._send_console_message_from_visible_action(
                        session_id=session_id
                    )
                    pending = None
                    if len(tasks) == before:
                        pending = hooks.pending_send_identity
                        assert (
                            pending is not None and pending[0] == session_id
                        ), "send never reached turn custody or a pending hook worker"
                        # Hook dispatch may return before its worker admits the
                        # turn. Observe only that exact Send within its old budget.
                        while len(tasks) == before:
                            assert screen.is_mounted and runtime.view is screen
                            assert screen._hooks is hooks
                            assert controller.store.active_session_id == session_id
                            assert (
                                hooks.pending_send_identity == pending
                            ), "pending Send cleared or changed without turn custody"
                            assert (
                                time.perf_counter() < send_deadline
                            ), "pending Send never reached turn custody within budget"
                            await asyncio.sleep(0.01)
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
                    if pending is not None:
                        assert owner is hooks and identity == pending
                        assert (
                            accepted_at <= send_deadline
                        ), "pending Send reached turn custody after its budget"
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
                    observed.write(result)
