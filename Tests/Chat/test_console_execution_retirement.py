"""Actual local graph retirement; these are not process/UI integration proofs."""

from __future__ import annotations

import asyncio

import pytest

from Tests.Chat.test_console_domain_boundary import domain_case as _domain_case

domain_case = _domain_case
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize("domain_case", ["control"], indirect=True)
@pytest.mark.parametrize("stage", ["before-entry", "before-ack"])
async def test_original_received_turn_retains_its_own_retirement(
    domain_case, stage, record_property
):
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook.Chat.console_control_dispatch import (
        CancelTurn,
        CommitPlanAck,
        ControlIdentity,
        ReceiveTurn,
        UpdateInputs,
        _ControlReservation,
        _PendingPlan,
    )
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
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
            f"retirement-command-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    controls.dispatch(UpdateInputs(identity(1), "retain original input", (1, 1)))
    received = controls.dispatch(
        ReceiveTurn(
            identity(2),
            "retirement-turn",
            case.session.draft_revision,
            _selection(case),
        )
    )
    invocation = received.turn_retirement
    task = received.work
    assert invocation is controls._turn_retirements["retirement-turn"]
    assert invocation.task is task
    assert received.retired is invocation.retired
    assert not received.retired.done()
    first_error = None
    try:
        if stage == "before-ack":
            assert await _until(lambda: controls.commit_plans or task.done())
            if task.done():
                task.result()  # Preserve the actual preparation exception.
            assert len(controls.commit_plans) == 1
            planned = controls.commit_plans[0]
            assert planned.received_request_id == received.identity.request_id
            assert planned.turn_id == "retirement-turn"
            assert invocation.source is not None and not invocation.source_closed
            reads = tuple(invocation.reads)
            assert reads
            assert all(read.read.task is task for read in reads)
            assert all(read.read.retired.done() and read.confirmed() for read in reads)
            assert any(
                read.observations["raw"]
                or read.observations["lease"]
                or read.connections
                for read in reads
            )
            record_property("task62_actual_retired_preparation_reads", len(reads))
            pending = controls._plans[planned.turn_id, planned.ordinal]
            assert type(pending) is _PendingPlan and pending.task is task
            assert pending.state == "waiting" and not pending.acknowledged.done()
            owner = case.controller._ordinary_native_commits[
                planned.fingerprint.preparation_id
            ]
            assert owner._issuer is case.controller and owner.store is case.store
            assert owner.task is task
            assert owner.acceptance is pending.plan.received_acceptance
            assert case.store._native_commit_owners_by_preparation[
                planned.fingerprint.preparation_id
            ] is owner
            reservation = owner.control_reservation
            assert type(reservation) is _ControlReservation
            assert reservation.controls is controls and reservation.owner is owner
            assert reservation.pending is pending
            assert len(reservation.result) == 1
            assert reservation.result[0] is pending.plan
            assert not reservation.writer_started and not reservation.native_retired
            assert not reservation.cancel_requested
            assert controls._effects == {reservation.effect}
            assert reservation.effect.kind == "commit_reservation"
            assert reservation.effect.owner is reservation
            assert not reservation.effect.completed.done()
            assert not reservation.effect.physically_retired
        else:
            assert invocation.source is None
            assert invocation.reads == []
        cancelled = controls.dispatch(CancelTurn(identity(3), "retirement-turn"))
        assert cancelled.applied.result() is True
        assert cancelled.retired is received.retired
        assert cancelled.work is task
        outcomes = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(outcomes[0], asyncio.CancelledError)
        record_property("task62_original_task_outcome", "CancelledError")
        assert await _until(received.retired.done)
        assert received.retired.result() is True
        assert invocation.logical_closed
        assert "retirement-turn" not in controls._turn_retirements
        assert controls._turns["retirement-turn"] == (
            received.identity, None, None, None
        )
        assert invocation.plan_keys == []
        if stage == "before-ack":
            pending = controls._plans[planned.turn_id, planned.ordinal]
            assert pending.state == "abandoned"
            assert pending.plan is pending.task is pending.acknowledged is None
            assert pending.projection.recovery is None
            with pytest.raises(RuntimeError, match="no longer pending"):
                controls.dispatch(
                    CommitPlanAck(
                        identity(4),
                        planned.received_request_id,
                        planned.turn_id,
                        planned.ordinal,
                        planned.fingerprint,
                    )
                )
        assert invocation.reads == [] and invocation.trees == []
        assert invocation.counts is not None
        record_property("task62_retirement_stage", stage)
        record_property("task62_retirement_counts", str(invocation.counts))
    except BaseException as error:
        first_error = error
        record_property("task62_first_body_error", f"{type(error).__name__}: {error}")
        raise
    finally:
        if not task.done():
            task.cancel()
        try:
            async with asyncio.timeout(10):
                await asyncio.gather(task, return_exceptions=True)
                if invocation.settlement is not None:
                    await asyncio.gather(invocation.settlement, return_exceptions=True)
        except BaseException as cleanup_error:
            record_property("task62_cleanup_error", type(cleanup_error).__name__)
            if first_error is None:
                raise


@pytest.mark.parametrize("domain_case", ["control_saved"], indirect=True)
async def test_actual_archive_close_uncertainty_never_becomes_turn_retired(
    domain_case, monkeypatch, request, record_property
):
    from concurrent.futures import ThreadPoolExecutor
    import sqlite3
    import sys
    import threading

    import pytest_timeout
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_domain_boundary import _selection
    from Tests.Chat.test_console_first_send_atomicity import _until
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.Chat.console_control_dispatch import (
        ControlIdentity,
        Drain,
        ReceiveTurn,
        UpdateInputs,
        _command_source,
    )
    from tldw_chatbook.Chat.console_control_sessions import _saved_read
    from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings
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
            provider="openai",
            model="gpt-4o-mini",
            base_url="http://127.0.0.1:1/v1",
        ),
    )
    conversation_id = case.store.persistence.create_conversation(
        conversation_title="Retained archive uncertainty"
    )
    case.store.publish_first_persisted_conversation(case.session.id, conversation_id)
    await (
        case.runtime._execution_service.ensure_permission_operations().set_kill_switch(
            False
        )
    )
    controls = case.runtime._execution_service.open_local_controls(
        profile_epoch="retirement-profile", service_epoch="retirement-service"
    )
    local = case.sources.conversation_scope_service.local_service
    database = case.database
    loop = asyncio.get_running_loop()
    original_submit = loop.run_in_executor
    entered, release = threading.Event(), threading.Event()
    captured, attempts = [], []
    close_error = sqlite3.OperationalError("task62 exact acquired archive close")
    receipt = None
    first_error = None

    def identity(sequence):
        return ControlIdentity(
            "retirement-profile",
            "retirement-service",
            sequence,
            f"uncertain-{sequence}",
            case.session.id,
            case.session.incarnation_id,
        )

    def observe_return(frame):
        if frame.f_locals.get("self") is not local or captured:
            return
        binding = _saved_read.get()
        assert binding is not None and binding.database is database
        connection = database._local.conn
        assert connection is not None and database._connection_quiescence.is_registered(
            connection
        )
        participant = database._maintenance_participant
        with storage._lock:
            lease = participant.connections[connection]
        original_close = type(connection).close

        def fail_exact_close(candidate):
            if candidate is connection:
                attempts.append(candidate)
                raise close_error
            return original_close(candidate)

        injector.setattr(type(connection), "close", fail_exact_close)
        captured.append(
            (binding, connection, participant, lease, threading.current_thread())
        )
        entered.set()
        assert release.wait(10), "Original archive read was not released"

    def close_on_creator(connection, creator):
        assert (
            threading.current_thread() is creator and database._local.conn is connection
        )
        database.close_connection()

    with ThreadPoolExecutor(max_workers=1) as executor:

        def route_original_read(selected, callback, *args):
            source = _command_source.get()
            if selected is None and source is not None and source.controls is controls:
                selected = executor
            return original_submit(selected, callback, *args)

        try:
            with (
                monkeypatch.context() as injector,
                _observe(
                    ChatConversationService.get_conversation_archive_states.__code__,
                    sys.monitoring.events.PY_RETURN,
                    observe_return,
                ),
            ):
                assert float(request.config.getini("timeout")) == 300
                assert not pytest_timeout.is_debugging()
                injector.setattr(loop, "run_in_executor", route_original_read)
                controls.dispatch(
                    UpdateInputs(identity(1), "preserve uncertain draft", (1, 1))
                )
                receipt = controls.dispatch(
                    ReceiveTurn(
                        identity(2),
                        "uncertain-turn",
                        case.session.draft_revision,
                        _selection(case),
                    )
                )
                invocation = receipt.turn_retirement
                assert await _until(lambda: entered.is_set() or receipt.work.done())
                assert entered.is_set(), (
                    repr(receipt.work.result()) if receipt.work.done() else "pending"
                )
                binding, connection, participant, lease, creator = captured[0]
                assert not binding.turn_owner.retired and not receipt.retired.done()
                release.set()
                async with asyncio.timeout(10):
                    outcomes = await asyncio.gather(
                        receipt.work, return_exceptions=True
                    )
                    await asyncio.shield(receipt.retired)
                assert attempts == [connection]
                assert isinstance(outcomes[0], BaseException)
                assert receipt.retired.result() is False
                assert controls._turn_retirements["uncertain-turn"] is invocation
                retained = controls._turns["uncertain-turn"]
                assert retained[0] is receipt.identity
                assert retained[1] is invocation.record and invocation.record is not None
                assert retained[2] is receipt.work and invocation.task is receipt.work
                assert retained[3] is controls._domain.sources
                assert invocation.logical_closed and invocation.source_closed
                effect = next(
                    e for e in invocation.source.effects if e.kind == "archive_read"
                )
                assert (
                    effect.completed.result() is False and not effect.physically_retired
                )
                assert effect in controls._effects and not binding.turn_owner.retired
                with storage._lock:
                    assert (
                        participant.connections[connection] is lease
                        and lease in storage._live_leases
                    )
                drain = controls.dispatch(Drain(identity(3)))
                async with asyncio.timeout(10):
                    with pytest.raises(
                        RuntimeError, match="native effect retirement is unproven"
                    ):
                        await asyncio.shield(drain.applied)
                await asyncio.gather(drain.work, return_exceptions=True)
                assert receipt.retired.result() is False
                record_property(
                    "task62_uncertain_original_outcome", type(outcomes[0]).__name__
                )
                record_property("task62_actual_archive_close_attempts", len(attempts))
        except BaseException as error:
            first_error = error
            record_property(
                "task62_first_body_error", f"{type(error).__name__}: {error}"
            )
            raise
        finally:
            release.set()
            try:
                async with asyncio.timeout(10):
                    if receipt is not None:
                        if not receipt.work.done():
                            receipt.work.cancel()
                        await asyncio.gather(receipt.work, return_exceptions=True)
                    if captured:
                        binding, connection, participant, lease, creator = captured[0]
                        await asyncio.shield(
                            original_submit(
                                executor, close_on_creator, connection, creator
                            )
                        )
                        assert not database._connection_quiescence.is_registered(
                            connection
                        )
                        with storage._lock:
                            assert (
                                connection not in participant.connections
                                and lease not in storage._live_leases
                            )
                        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
                            sqlite3.Connection.in_transaction.__get__(connection)
                        assert receipt.retired.result() is False
                        assert effect.completed.result() is False
                        record_property(
                            "task62_uncertainty_after_creator_cleanup",
                            "original false retained",
                        )
            except BaseException as error:
                record_property(
                    "task62_cleanup_error", f"{type(error).__name__}: {error}"
                )
                if first_error is None:
                    raise
