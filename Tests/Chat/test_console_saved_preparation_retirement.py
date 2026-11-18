"""Real saved-session preparation on a reused worker, without a provider call."""

from __future__ import annotations

import asyncio
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from types import CodeType

import pytest

from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case

domain_case = _domain_case
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@asynccontextmanager
async def _routing(monkeypatch, loop, callback, monitor):
    with monkeypatch.context() as routing, monitor:
        routing.setattr(loop, "run_in_executor", callback)
        yield


@pytest.mark.parametrize("domain_case", ["control_saved"], indirect=True)
@pytest.mark.parametrize(
    "borrowed, stop",
    [
        (False, None),
        (True, None),
        (True, "cancel"),
        (True, "source-change"),
        (True, "submit-failure"),
        (True, "enqueue-failure"),
    ],
)
async def test_saved_preparations_retire_on_reused_original_worker(
    domain_case, monkeypatch, record_property, borrowed, stop
):
    from dataclasses import replace

    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        ControlIdentity,
        ReceiveTurn,
        UpdateInputs,
        _command_source,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
    from tldw_chatbook.Chat.console_configuration_preparation import (
        capture_console_turn_configuration,
        capture_console_turn_configuration_owned,
    )
    from tldw_chatbook.Chat.console_execution_retirement import _native
    from tldw_chatbook.Chat.console_library_policy_coordinator import (
        ConsoleLibraryPolicyCoordinator,
    )
    from tldw_chatbook.Chat.prompt_history import PromptHistory

    case = domain_case
    case.controller.prompt_history = PromptHistory(
        case.database.db_path.parent / "history.jsonl"
    )
    case.sources.configuration["api_settings"] = {
        "openai": {"api_key": "task62-local-readiness-only"}
    }
    case.store.replace_session_settings(
        case.session.id,
        ConsoleSessionSettings(
            provider="openai", model="gpt-4o-mini", base_url="http://127.0.0.1:1/v1"
        ),
    )
    case.store.replace_session_trace_privacy_override(
        case.session.id,
        capture_enabled=False,
        pii_redaction_enabled=None,
        expected_policy_revision=case.store.capture_policy_state(
            case.session.id
        ).policy_revision,
    )
    conversation_id = case.store.persistence.create_conversation(
        conversation_title="Reused preparation worker"
    )
    case.store.publish_first_persisted_conversation(case.session.id, conversation_id)
    # Match an actually opened saved session's durable Library-policy binding;
    # first-persistence publication alone does not load that coordinator.
    await case.store.library_policy_coordinator.load(case.session.id, conversation_id)
    await (
        case.runtime._execution_service.ensure_permission_operations().set_kill_switch(
            False
        )
    )
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="retirement-profile", service_epoch="retirement-service"
    )

    def identity(sequence):
        return ControlIdentity(
            "retirement-profile",
            "retirement-service",
            sequence,
            f"reused-command-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    loop = asyncio.get_running_loop()
    original_submit = loop.run_in_executor
    retained = []
    capture_code = next(
        code
        for code in capture_console_turn_configuration_owned.__code__.co_consts
        if type(code) is CodeType and code.co_name == "capture"
    )
    library_code = next(
        code
        for code in ConsoleLibraryPolicyCoordinator._run_repository_call.__code__.co_consts
        if type(code) is CodeType and code.co_name == "owned_call"
    )
    entered, release = threading.Event(), threading.Event()
    held = []
    submissions = []
    submit_error = RuntimeError("Original worker submission failed")

    def hold_capture(frame):
        observation = getattr(_native, "read", None)
        if stop not in {"cancel", "source-change"} or observation is None or held:
            return
        if observation.read.callback.__code__ is not capture_code:
            return
        held.append(observation)
        entered.set()
        assert release.wait(10), "Original final capture was not released"

    @asynccontextmanager
    async def independent_connection(executor):
        connection = None
        if borrowed:

            def acquire_original():
                connection = case.database.get_connection()
                connection.execute("BEGIN")
                participant = case.database._maintenance_participant
                return connection, participant, participant.connections[connection]

            connection, participant, lease = await original_submit(
                executor, acquire_original
            )
        try:
            yield connection
        finally:
            if connection is not None:
                # Its real original caller retains ownership; preparation must
                # not close this thread's already-open handle on its behalf.
                def release_original():
                    try:
                        assert case.database._local.conn is connection
                        assert case.database._maintenance_participant is participant
                        assert participant.connections[connection] is lease
                        assert connection.in_transaction
                        assert connection.execute("SELECT 1").fetchone()[0] == 1
                    finally:
                        connection.rollback()
                        case.database.close_connection()

                await original_submit(executor, release_original)

    with ThreadPoolExecutor(max_workers=1) as executor:

        def route_original_read(selected, callback, *args):
            source = _command_source.get()
            if (
                stop in {"submit-failure", "enqueue-failure"}
                and selected is not None
                and source is not None
                and source.controls is controls
                and args
            ):
                invoke = args[0]
                captured = dict(
                    zip(
                        invoke.__code__.co_freevars,
                        (cell.cell_contents for cell in invoke.__closure__),
                        strict=True,
                    )
                )
                read = captured.get("read")
                if read is not None and read.callback.__code__ is capture_code:
                    future = (
                        original_submit(selected, callback, *args)
                        if stop == "enqueue-failure"
                        else None
                    )
                    submissions.append((selected, read, future))
                    raise submit_error
            if selected is None and source is not None and source.controls is controls:
                selected = executor
            return original_submit(selected, callback, *args)

        async with (
            independent_connection(executor) as borrowed_connection,
            _routing(
                monkeypatch,
                loop,
                route_original_read,
                _observe(
                    capture_console_turn_configuration.__code__,
                    sys.monitoring.events.PY_RETURN,
                    hold_capture,
                ),
            ),
        ):
            for attempt in range(1 if stop else 2):
                sequence = attempt * 3 + 1
                turn_id = f"reused-turn-{attempt}"
                controls.dispatch(
                    UpdateInputs(identity(sequence), "original draft", (sequence, 1))
                )
                receipt = controls.dispatch(
                    ReceiveTurn(
                        identity(sequence + 1),
                        turn_id,
                        case.session.draft_revision,
                        _selection(case),
                    )
                )
                invocation = receipt.turn_retirement
                try:
                    if stop in {"submit-failure", "enqueue-failure"}:
                        outcomes = await asyncio.gather(
                            receipt.work, return_exceptions=True
                        )
                        assert outcomes == [submit_error]
                        assert await asyncio.shield(receipt.retired) is True
                        assert len(submissions) == 1
                        executor_used, read, future = submissions[0]
                        if future is not None:
                            assert await asyncio.shield(future) is None
                        assert read.retired.done() and not read.callback_returned
                        assert all(
                            not thread.is_alive() for thread in executor_used._threads
                        )
                        assert not any(
                            plan.turn_id == turn_id for plan in controls.commit_plans
                        )
                        continue
                    if stop:
                        assert await _until(
                            lambda receipt=receipt: entered.is_set()
                            or receipt.work.done()
                        )
                        if receipt.work.done():
                            receipt.work.result()
                        assert entered.is_set() and not receipt.retired.done()
                        observation = held[0]
                        assert (
                            not observation.closed
                            and not observation.read.retired.done()
                        )
                        if stop == "cancel":
                            controls.dispatch(
                                CancelTurn(identity(sequence + 2), turn_id)
                            )
                        else:
                            case.store.replace_session_settings(
                                case.session.id,
                                replace(case.session.settings, model="changed-model"),
                            )
                        await asyncio.sleep(0)
                        assert not receipt.work.done() and not receipt.retired.done()
                        release.set()
                        outcome = (
                            await asyncio.gather(receipt.work, return_exceptions=True)
                        )[0]
                        if stop == "cancel":
                            assert isinstance(outcome, asyncio.CancelledError)
                        else:
                            from tldw_chatbook.Backup_Recovery.bootstrap import (
                                RecoveryRequired,
                            )

                            assert isinstance(outcome, RecoveryRequired)
                        assert await asyncio.shield(receipt.retired) is True
                        assert observation.confirmed()
                        assert not observation.creator[1].is_alive()
                        assert not any(
                            plan.turn_id == turn_id for plan in controls.commit_plans
                        )
                        continue
                    assert await _until(
                        lambda turn_id=turn_id, receipt=receipt: any(
                            plan.turn_id == turn_id for plan in controls.commit_plans
                        )
                        or receipt.work.done()
                    )
                    if receipt.work.done():
                        receipt.work.result()
                    reads = tuple(invocation.reads)
                    retained.append(reads)
                    record_property(f"attempt_{attempt}_read_count", len(reads))
                    assert reads
                    assert all(read.read.task is receipt.work for read in reads)
                    capture_read = next(
                        read
                        for read in reads
                        if read.read.callback.__code__ is capture_code
                    )
                    assert capture_read.connections
                    assert not capture_read.creator[1].is_alive()
                    library_read = next(
                        read
                        for read in reads
                        if read.read.callback.__code__ is library_code
                    )
                    assert library_read.connections
                    assert not library_read.creator[1].is_alive()
                    assert capture_read.creator[1] is not library_read.creator[1]
                    assert all(
                        item[1] is not borrowed_connection
                        for observed in (capture_read, library_read)
                        for item in observed.connections
                    )
                    # Re-evaluate the original proof after all subsequent reads,
                    # including original commit reservation, have completed.
                    assert all(
                        read.confirmed() for cohort in retained for read in cohort
                    )
                    cancelled = controls.dispatch(
                        CancelTurn(identity(sequence + 2), turn_id)
                    )
                    assert cancelled.retired is receipt.retired
                    outcomes = await asyncio.gather(
                        receipt.work, return_exceptions=True
                    )
                    assert isinstance(outcomes[0], asyncio.CancelledError)
                    assert await asyncio.shield(receipt.retired) is True
                    assert invocation.logical_closed and invocation.source_closed
                    assert turn_id not in controls._turn_retirements
                finally:
                    release.set()
                    if not receipt.work.done():
                        receipt.work.cancel()
                    async with asyncio.timeout(10):
                        await asyncio.gather(receipt.work, return_exceptions=True)
                        if invocation.settlement is not None:
                            await asyncio.gather(
                                invocation.settlement, return_exceptions=True
                            )
