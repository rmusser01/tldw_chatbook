"""Stop in the pre-provider "Connecting tools…" window (TASK-33621.22).

Found by the TASK-33625.1 verifier and reproduced live on dev ed7481a30c
(Anthropic haiku, Ctrl+G 0.9 s after Enter, while the row read "Connecting
tools…"). Between "console agent reply start" and the provider call, the agent
reply composes the run's tools, waits for trace maintenance, moves the durable
checkpoint to DISPATCH_STARTED and builds the profile tools. Only the bridge
call after that window and the checkpoint move itself handled a Stop. A Stop
anywhere else raised a bare ``CancelledError`` out of the reply, and the stream
wrapper then retired the run's cancel event, so the accepted turn's handler no
longer saw an explicit Stop and marked the reply failed:

- after the checkpoint moved, Stop had already settled the row as Stopped, so
  that ``mark_message_failed`` raised ``ValueError``;
- before it moved, Stop could not settle the ACCEPTED checkpoint at all
  ("Dispatch terminal settlement previously failed.") and left "Response
  recovery failed".

Either way the turn's preparation never settled and kept the tab's send slot:
every later send was refused with "Draft or chat changed; Send again."

Two more ways to the same wedge showed up live while pressing Stop repeatedly:

- a repaint's preparation read of the session in flight at the press sent
  Stop down its "cancel the reads" branch, which never marked the run as
  stopped by the user;
- a press after the reply had settled the Stop, while the turn was finishing,
  cancelled the turn's intake custody, so the turn ended cancelled.

This drives the stock received-intent path (real ChatScreen, runtime,
controller, store and file-backed database). Only the provider call is a
double; one setup step and the turn's last await are held so that each press
lands where the live ones did.
"""

from __future__ import annotations

import asyncio
import sys
import threading

import pytest
from loguru import logger
from textual.widgets import Button

from Tests.UI.test_console_hook_review_send_freeze import _key
from Tests.UI.test_console_received_intent_feedback import (
    _received_console_case,
    _send,
    _until,
)
from Tests.UI.test_console_send_admission_off_pump import _allow_second_turns
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
from tldw_chatbook.Chat.console_send_diagnostics import SendDiagnostic

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

FOLLOW_UP = "A follow-up after Stop"
REPLY = "received intent reply"
# Real Windows admission took 48.296s before the first provider failure in
# qualification. This bounds whole-turn readiness, not Stop responsiveness;
# cancellation, settlement, paint and product timeouts retain their bounds.
TURN_READY_SECONDS = 60 if sys.platform == "win32" else 20
#: Failure signatures of the wedge: the runtime turn and the accepted-turn
#: settlement that ended in an exception instead of a Stop.
FAILURE_LOGS = (
    "turn ended with exception_type",
    "settlement failed",
    "Durable turn commit failed",
)


def _painted(host) -> str:
    return "\n".join(strip.text for strip in host.screen._compositor.render_strips())


@pytest.mark.parametrize(
    "window",
    [
        # The durable checkpoint is still ACCEPTED: MCP and local tool
        # composition, the window a slow MCP server stretches.
        "composing tools",
        # The checkpoint is DISPATCH_STARTED; the profile tools are built.
        "after the dispatch checkpoint",
    ],
    # No spaces: the UI PR gate census lists these node ids one per line.
    ids=["composing-tools", "after-dispatch-checkpoint"],
)
async def test_stop_while_connecting_tools_lets_the_next_send_reach_the_provider(
    monkeypatch, window
):
    logged = []
    sink = logger.add(lambda message: logged.append(message.record["message"]))
    stages = []
    record = SendDiagnostic.record

    def spy(self, phase, status="entered", **fields):
        stages.append((phase, status, fields.get("exception_type")))
        return record(self, phase, status, **fields)

    monkeypatch.setattr(SendDiagnostic, "record", spy)
    # Exchange capture off, with the real setting: in this harness a second
    # turn's trace reservation fails with capture on even when nothing was
    # stopped (unsupported_surface_change on a plain two-send control), which
    # would hide what this test is about. The live check ran with capture on.
    assert config_module.save_setting_to_cli_config(
        "console", "exchange_capture", False
    )
    try:
        async with _received_console_case(
            monkeypatch, f"stop-setup-{len(window)}", durable=True
        ) as case:
            _allow_second_turns(case.host)
            case.probe.release.set()  # do not hold the configuration read
            controller = case.controller
            session_id = case.session.id
            held = asyncio.Event()
            never = asyncio.Event()
            compose = controller._compose_agent_request_providers
            build_profile = controller._personal_context_builder
            composed = []

            async def hold_composition(**kwargs):
                composed.append(True)
                if window == "composing tools" and not held.is_set():
                    held.set()
                    await never.wait()  # only Stop ends this wait
                return await compose(**kwargs)

            async def hold_profile_tools(*args, **kwargs):
                if (
                    window == "after the dispatch checkpoint"
                    and composed
                    and not held.is_set()
                ):
                    held.set()
                    await never.wait()  # only Stop ends this wait
                return await build_profile(*args, **kwargs)

            monkeypatch.setattr(
                controller, "_compose_agent_request_providers", hold_composition
            )
            monkeypatch.setattr(
                controller, "_personal_context_builder", hold_profile_tools
            )
            # How each turn's runtime task ended: a stopped turn still returns.
            outcomes = []
            finish = case.runtime._finish_custodied_turn

            def record_outcome(task, **kwargs):
                outcomes.append(
                    "cancelled"
                    if task.cancelled()
                    else type(task.exception()).__name__
                    if task.exception() is not None
                    else "returned"
                )
                return finish(task, **kwargs)

            monkeypatch.setattr(case.runtime, "_finish_custodied_turn", record_outcome)

            _send(case, "enter")
            assert await _until(held.is_set, TURN_READY_SECONDS)
            # The live precondition: the row names the window, Stop is offered,
            # and the provider has not been called.
            assert await _until(lambda: "Connecting tools" in _painted(case.host), 10)
            stop = case.console.query_one("#console-stop-generation", Button)
            assert await _until(
                lambda: stop.styles.display == "block" and not stop.disabled, 10
            )
            recovery = case.store.dispatch_recovery_for_session(session_id)
            assert recovery is not None
            assert recovery.in_flight is (window != "composing tools")
            preparation = case.store.preparation_for_session(session_id)
            assert preparation is not None
            assert recovery.preparation_id == preparation.preparation_id
            assert case.provider_calls == []

            # While the row's elapsed time ticks, a repaint's preparation read
            # of this session can be in flight when Stop lands. Stop used to
            # cancel that read and return early, leaving the run unmarked as a
            # user Stop (no transcript record), so hold one in flight here.
            read_started, read_released = threading.Event(), threading.Event()

            def hold_read():
                read_started.set()
                return read_released.wait(30)

            read = asyncio.ensure_future(
                run_preparation_read(
                    hold_read,
                    creator=controller,
                    session_id=session_id,
                    reads=controller._preparation_reads,
                    require_current=lambda: None,
                )
            )
            assert await _until(read_started.is_set, 5)
            (unrelated_read,) = [
                handle
                for handle in controller._preparation_reads
                if handle.task is read and handle.session_id == session_id
            ]

            # The UI supplies an app-owned monitor, so monitor.close is no
            # longer a turn await. Hold the actual queue tail after its real
            # settlement, while the accepted turn still has runtime custody.
            tail_armed, in_tail = asyncio.Event(), asyncio.Event()
            tail_released = asyncio.Event()
            tail_tasks = []
            queue = controller.prompt_queue_coordinator
            after_turn = queue._after_turn

            async def hold_tail(owner_id, result):
                outcome = await after_turn(owner_id, result)
                if (
                    owner_id == session_id
                    and tail_armed.is_set()
                    and not in_tail.is_set()
                ):
                    tail_tasks.append(asyncio.current_task())
                    in_tail.set()
                    await tail_released.wait()
                return outcome

            monkeypatch.setattr(queue, "_after_turn", hold_tail)
            stop_requested = controller._active_cancel_events[session_id]
            tail_armed.set()
            try:
                _key(case.host, "ctrl+g", "\x07")  # Stop, as the user pressed it
                assert await _until(stop_requested.is_set, 5)
                read_cancel_requests = read.cancelling()  # asserted at the end
            finally:
                read_released.set()
            (read_outcome,) = await asyncio.wait_for(
                asyncio.gather(read, return_exceptions=True), 10
            )
            try:
                assert await _until(in_tail.is_set, 10)
                (custody,) = [
                    row
                    for row in case.runtime._turn_custody.values()
                    if row.session_id == session_id
                ]
                assert custody.inputs.durable_accepted
                assert custody.task is tail_tasks[0]
                assert not custody.task.done()
                assert outcomes == []
                assert (
                    controller.run_state_for(session_id).status
                    is ConsoleRunStatus.STOPPED
                )
                assert case.store.dispatch_recovery_for_session(session_id) is None
                assert case.store.preparation_for_session(session_id) is None
                for _ in range(3):  # and again, as the live repro did
                    _key(case.host, "ctrl+g", "\x07")
                    await asyncio.sleep(0.05)
            finally:
                tail_released.set()

            def turn_ended() -> bool:
                # The runtime releases a turn's custody once its task ends.
                return not any(
                    row.session_id == session_id
                    for row in case.runtime._turn_custody.values()
                )

            assert await _until(turn_ended, 15)
            # AC1: the turn ends as Stopped, recorded in the transcript.
            assert outcomes == ["returned"]
            messages = case.store.all_messages_for_session(session_id)
            assert [
                message.status
                for message in messages
                if message.role is ConsoleMessageRole.ASSISTANT
            ] == ["stopped"]
            assert any(
                message.role is ConsoleMessageRole.SYSTEM
                and message.content == "Response stopped by user."
                for message in messages
            ), [(message.role.value, message.content) for message in messages]
            assert controller.run_state_for(session_id).status is (
                ConsoleRunStatus.STOPPED
            )
            assert case.store.dispatch_recovery_for_session(session_id) is None
            assert case.store.preparation_for_session(session_id) is None
            assert case.store.preparation_by_id(preparation.preparation_id) is None
            assert case.provider_calls == []

            # AC2: the next send in the same tab reaches the provider.
            case.composer.load_draft(FOLLOW_UP)
            case.composer.focus()
            assert await _until(lambda: case.host.focused is case.composer, 5)
            # Enter before the store holds the draft is refused as a changed
            # input (RecoveryRequired), with or without a Stop before it.
            assert await _until(
                lambda: case.store.session_draft(session_id) == FOLLOW_UP, 5
            )
            _send(case, "enter")
            assert await _until(
                lambda: len(case.provider_calls) == 1, TURN_READY_SECONDS
            )
            assert await _until(lambda: REPLY in _painted(case.host), 20)
            assert await _until(lambda: len(outcomes) == 2, 20)
            assert outcomes == ["returned", "returned"]
            painted = _painted(case.host)
            assert "Unsent turn" not in painted
            assert case.runtime.recoveries_for_session(session_id) == ()
            assert not [
                note.message
                for note in case.host._notifications
                if "Send again" in note.message
            ]
            # The repaint's read was not Stop's to cancel: it finished itself.
            assert read_cancel_requests == 0
            assert read_outcome is True
            assert unrelated_read not in controller._preparation_reads
    finally:
        logger.remove(sink)
    # AC1: nothing logged the Stop as a failure.
    assert not [line for line in logged if any(sign in line for sign in FAILURE_LOGS)]
    assert not [stage for stage in stages if stage[1] == "failed"]


@pytest.mark.parametrize(
    "window",
    [
        # Tools are composing; the failed row is not yet prepared for the
        # retry, so it is still "failed".
        "composing tools",
        # The row was prepared for the retry ("pending"); the profile tools
        # are built.
        "after the retry was prepared",
    ],
    ids=["composing-tools", "after-retry-prepared"],
)
async def test_stop_while_a_retry_connects_tools_frees_the_tab(monkeypatch, window):
    """A Retry of a failed reply enters the same window with a "failed" row.

    Stop there used to settle the row as if a fresh reply were streaming:
    ``mark_message_stopped`` refused the still-"failed" row, the Stop returned
    early without stopping the run, and the Retry reported success. The run
    state stayed "Agent running." and every later send in the tab was refused
    with "A run is already running in this tab."
    """
    from tldw_chatbook.Chat import Chat_Functions as chat_functions
    from tldw_chatbook.Widgets.Console.console_transcript import ConsoleTranscript

    assert config_module.save_setting_to_cli_config(
        "console", "exchange_capture", False
    )
    async with _received_console_case(
        monkeypatch, f"stop-retry-{len(window)}", durable=True
    ) as case:
        _allow_second_turns(case.host)
        case.probe.release.set()  # do not hold the configuration read
        controller = case.controller
        session_id = case.session.id
        answer = chat_functions.chat_api_call
        failing = threading.Event()
        failing.set()

        def fail_first_turn(**kwargs):
            # Every attempt of the first turn fails (the gateway retries once).
            if failing.is_set():
                raise RuntimeError("first attempt exploded")
            return answer(**kwargs)

        monkeypatch.setattr(chat_functions, "chat_api_call", fail_first_turn)

        def assistant_rows():
            return [
                message
                for message in case.store.all_messages_for_session(session_id)
                if message.role is ConsoleMessageRole.ASSISTANT
            ]

        def turn_ended() -> bool:
            return not any(
                row.session_id == session_id
                for row in case.runtime._turn_custody.values()
            )

        _send(case, "enter")
        assert await _until(
            lambda: [row.status for row in assistant_rows()] == ["failed"],
            TURN_READY_SECONDS,
        ), [(row.status, row.content[:80]) for row in assistant_rows()]
        assert await _until(turn_ended, 15)
        (failed,) = assistant_rows()
        assert case.provider_calls == []
        failing.clear()

        held = asyncio.Event()
        never = asyncio.Event()
        compose = controller._compose_agent_request_providers
        build_profile = controller._personal_context_builder
        composed = []

        async def hold_composition(**kwargs):
            composed.append(True)
            if window == "composing tools" and not held.is_set():
                held.set()
                await never.wait()  # only Stop ends this wait
            return await compose(**kwargs)

        async def hold_profile_tools(*args, **kwargs):
            if window != "composing tools" and composed and not held.is_set():
                held.set()
                await never.wait()  # only Stop ends this wait
            return await build_profile(*args, **kwargs)

        monkeypatch.setattr(
            controller, "_compose_agent_request_providers", hold_composition
        )
        monkeypatch.setattr(controller, "_personal_context_builder", hold_profile_tools)
        retried = []
        retry = controller.retry_message

        async def record_retry(*args, **kwargs):
            result = await retry(*args, **kwargs)
            retried.append(result)
            return result

        monkeypatch.setattr(controller, "retry_message", record_retry)

        # Retry, as the user presses it on the failed row.
        transcript = case.console.query_one(
            "#console-native-transcript", ConsoleTranscript
        )
        transcript.select_message(failed.id)
        await case.console._sync_native_console_chat_ui()
        retry_button = f"#console-message-action-retry-{failed.id}"
        assert await _until(lambda: bool(case.console.query(retry_button)), 10)
        case.console.query_one(retry_button, Button).press()
        assert await _until(held.is_set, TURN_READY_SECONDS)
        if window != "composing tools":
            # The prepared row names the window; an unprepared one still
            # reads as the failure being retried.
            assert await _until(lambda: "Connecting tools" in _painted(case.host), 10)
        stop = case.console.query_one("#console-stop-generation", Button)
        assert await _until(
            lambda: stop.styles.display == "block" and not stop.disabled, 10
        )
        expected_row = "failed" if window == "composing tools" else "pending"
        assert [row.status for row in assistant_rows()] == [expected_row]

        _key(case.host, "ctrl+g", "\x07")  # Stop, as the user pressed it
        assert await _until(lambda: retried, 15)

        # AC1: the retry ends Stopped, recorded in the transcript. A row
        # stopped before the retry prepared it stays "failed", so Retry is
        # offered again; one already prepared settles "stopped".
        assert controller.run_state_for(session_id).status is (
            ConsoleRunStatus.STOPPED
        ), controller.run_state_for(session_id)
        assert [row.status for row in assistant_rows()] == [
            "failed" if window == "composing tools" else "stopped"
        ]
        messages = case.store.all_messages_for_session(session_id)
        assert [
            message.content
            for message in messages
            if message.role is ConsoleMessageRole.SYSTEM
            and message.content == "Response stopped by user."
        ] == ["Response stopped by user."], [
            (message.role.value, message.content) for message in messages
        ]
        assert not controller._replies_in_pre_provider_setup
        assert case.store.dispatch_recovery_for_session(session_id) is None
        assert case.store.preparation_for_session(session_id) is None
        assert case.provider_calls == []

        # AC2: the next send in the same tab reaches the provider.
        case.composer.load_draft(FOLLOW_UP)
        case.composer.focus()
        assert await _until(lambda: case.host.focused is case.composer, 5)
        assert await _until(
            lambda: case.store.session_draft(session_id) == FOLLOW_UP, 5
        )
        _send(case, "enter")
        assert await _until(
            lambda: len(case.provider_calls) == 1, TURN_READY_SECONDS
        ), [note.message for note in case.host._notifications]
        assert await _until(lambda: REPLY in _painted(case.host), 20)
        assert await _until(turn_ended, 15)
        assert assistant_rows()[-1].content == REPLY
        assert assistant_rows()[-1].status == "complete"
        assert controller.run_state_for(session_id).status is ConsoleRunStatus.COMPLETED
        assert case.runtime.recoveries_for_session(session_id) == ()
        assert not [
            note.message
            for note in case.host._notifications
            if "already running" in note.message or "Send again" in note.message
        ]
