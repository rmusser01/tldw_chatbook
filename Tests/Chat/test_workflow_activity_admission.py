"""Runtime admission protects native work independently of viewed chat."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

pytestmark = pytest.mark.bootstrap_profile


async def journey(tmp_path, monkeypatch):
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("task-1",))
    )
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        await rig.coordinator.approve(decision, command_id="approve-1")
        await rig.wait_for_provider_entry("task-1")
        assert not rig.controller.fleet_wake.try_claim_automatic_primary(
            "workflow-test-chat", object()
        )
        assert [key for key, _ in rig.provider.calls].count("planning") == 1
    finally:
        await rig.close()


def test_phase_holds_original_primary_slot_during_child(tmp_path):
    _run(
        tmp_path,
        "activity",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def native_capacity_notification_journey(tmp_path, monkeypatch):
    import asyncio
    import os
    import sys
    import threading

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Chat import console_preparation_reads as native_reads
    from tldw_chatbook.Chat.console_workflow_runs import ConsoleWorkflowCoordinator

    accounting = sys.argv[1] == "capacity-accounting"
    scenario = sys.argv[2] if accounting else "native"
    accounting_failure = RuntimeError("original quiesce producer failed after commit")
    accounting_return = {}
    expiry = {}
    if scenario == "cleanup-expired":
        os.environ["TLDW_AGENTS_MAX_WORKFLOW_WALL_SECONDS"] = "10"
    rig = await build_workflow_rig(
        tmp_path,
        monkeypatch,
        WorkflowProvider(hold=("task-1",) if accounting else ()),
        change_review=scenario == "capture" or scenario.startswith("capture-"),
    )
    retained = False
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await asyncio.shield(rig.coordinator._runs[rig.workflow.id].driver)
        await rig.wait_for_status("awaiting_phase_review")
        os.environ["TLDW_AGENTS_MAX_RUNTIME_SUBAGENTS"] = "1"
        os.environ["TLDW_AGENTS_RESERVED_MANUAL_SUBAGENTS"] = "1"
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        if scenario in {"quiesce-unknown", "restart-failed"}:
            code = type(rig.service.db.automatic_work).quiesce_workflow_phase.__code__

            def committed_quiesce(frame, event, result):
                if (
                    event != "return"
                    or frame.f_code is not code
                    or type(result) is not tuple
                ):
                    return
                cell = frame.f_locals.get("_quiescence")
                if cell is None or cell.coordinator is not rig.coordinator:
                    return
                threading.setprofile_all_threads(None)
                accounting_return.update(cell=cell, interval=result)
                raise accounting_failure

            threading.setprofile_all_threads(committed_quiesce)
        if scenario == "cleanup-expired":
            from tldw_chatbook.Chat.console_workflow_preparation import (
                _PreparedWorkflowReply,
            )
            from tldw_chatbook.Chat.console_workflow_runs import _policy_identity

            policy = _policy_identity()

            def exhaust_original_capture(frame, event, result):
                if (
                    event != "return"
                    or frame.f_code is not _PreparedWorkflowReply._capture.__code__
                    or result is not None
                ):
                    return
                original = frame.f_locals["self"]
                if original.phase.attempt.purpose == "planning":
                    return
                threading.setprofile_all_threads(None)
                actual_run = rig.coordinator._runs[rig.workflow.id]
                record = original._finite_reads[-1]
                assert original.phase is actual_run.phase
                assert (
                    record.prepared is original and record.issuing is actual_run.driver
                )
                assert original._finite == (record.owner, record.claim)
                assert (
                    record.owner.is_admitted and record.operation.owner is record.owner
                )
                assert record.operation in record.owner._operations
                assert (
                    record.claim is not None
                    and record.claim.id in rig.runtime.worktree_activity._claims
                )
                assert record.proof[0].source is original
                assert record.proof[1] is record.proof[0]._producer
                assert not record.proof[1].done()
                with rig.service.db.connection() as conn:
                    assert not conn.in_transaction
                ledger = rig.service.db.automatic_work
                started = ledger.workflow_active_seconds(actual_run.workflow_id)
                hold = max(0, 10 - started + 0.02)
                assert hold <= 10
                if hold:
                    assert not threading.Event().wait(hold)
                ended = ledger.workflow_active_seconds(actual_run.workflow_id)
                assert ended >= 10 and _policy_identity() == policy
                expiry.update(
                    prepared=original,
                    read=record,
                    owner=record.owner,
                    claim=record.claim,
                    operation=record.operation,
                    phase=original.phase,
                    source=original.source,
                    policy=policy,
                    started=started,
                    ended=ended,
                    hold=hold,
                )
                print(
                    "G_ORIGINAL_CAPTURE_NATURAL_EXPIRY",
                    started,
                    ended,
                    hold,
                    flush=True,
                )

            threading.setprofile_all_threads(exhaust_original_capture)
        await rig.coordinator.approve(decision, command_id="capacity-event")
        run = rig.coordinator._runs[rig.workflow.id]
        if scenario in {"quiesce-unknown", "restart-failed"}:
            from tldw_chatbook.Agents.activation import execution
            from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

            await asyncio.shield(run.driver)
            cell = accounting_return["cell"]
            interval = accounting_return["interval"]
            assert run.phase._quiescence is cell and cell.mode == "quiesce"
            assert cell.proof[1].exception() is accounting_failure
            assert cell.error is run.accounting_error is accounting_failure
            assert cell.producer_error is accounting_failure and cell.idle_proof is None
            assert run.status == "recovery_required"
            prepared = run.preparations["task-1"]
            assert prepared._closed and prepared._native_closed
            prepared._finite_read_return(prepared._finite_reads[-1])
            assert not prepared.authorization._native_launch
            with (
                owned_goal_storage(rig.service),
                execution(rig.controller._agent_bridge),
                run.phase.context.scope(),
            ):
                ledger = rig.service.db.automatic_work
                with pytest.raises(AutomaticWorkRefused, match="clock_unknown"):
                    ledger.workflow_active_seconds(run.workflow_id)
                with pytest.raises((PermissionError, AutomaticWorkRefused)):
                    ledger.resume_workflow_phase(
                        run.phase.attempt.id,
                        owner_id=run.phase.attempt.owner_id,
                        _quiescence=cell,
                        _cleanup=True,
                    )
                with rig.service.db.connection() as conn:
                    row = conn.execute(
                        "SELECT * FROM workflow_active_intervals WHERE id=?",
                        (interval[0],),
                    ).fetchone()
                    assert tuple(row) == interval
                    assert (
                        conn.execute(
                            "SELECT retirement FROM workflow_phase_attempts WHERE native_attempt_id=?",
                            (run.phase.attempt.id,),
                        ).fetchone()[0]
                        == "unknown"
                    )
                    assert (
                        conn.execute(
                            "SELECT count(*) FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                            (run.workflow_id,),
                        ).fetchone()[0]
                        == 0
                    )
            snapshot = await rig.coordinator.get(run.workflow_id)
            assert (
                snapshot.status == "recovery_required" and snapshot.accounting.uncertain
            )
            assert snapshot.accounting.used["generation"] == 2
            assert snapshot.accounting.used["child_launch"] == 0
            intake, token = run.intake, run.token
            assert intake is not None and token is not None
            for _ in range(2):
                with pytest.raises(
                    RuntimeError, match="workflow_original_shutdown_unresolved"
                ):
                    await rig.runtime.dispose()
                assert rig.coordinator._runs[run.workflow_id] is run
                assert run.intake is intake and run.token is token
            assert [key for key, _ in rig.provider.calls] == ["planning"]
            if scenario == "restart-failed":
                _save_accounting_restart_boundary(rig, run, cell, tmp_path)
            print("G_ACCOUNTING_ORIGINAL_QUIESCE_UNKNOWN; no creator drain", flush=True)
            retained = True
            os._exit(17)
        while run.wake is None or not run.wake._waiters:
            assert not run.driver.done(), "original readiness wait never armed"
            await asyncio.sleep(0.01)
        waiter = run.wake._waiters[0]
        prepared = run.preparations["task-1"]
        authorization = prepared.authorization
        native = authorization._native_preparation[0]
        before = await rig.coordinator.get(rig.workflow.id)
        for _ in range(5):
            await asyncio.sleep(0)
            assert run.wake._waiters[0] is waiter and not waiter.done()
        assert not authorization._native_launch and not authorization._native_dispatch
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        assert (
            before.accounting.used["generation"] == 2
            and before.accounting.used["child_launch"] == 0
        )
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        if accounting:
            import copy

            from tldw_chatbook.Agents.activation import execution
            from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
            from tldw_chatbook.Chat.console_workflow_runs import (
                _prove_workflow_read,
                _quiescence_source,
            )

            clock = run.phase._quiescence
            assert clock.mode == "idle" and clock.phase is run.phase
            _prove_workflow_read(clock.idle_proof, rig.controller, clock, run.driver)
            with pytest.raises(AutomaticWorkRefused, match="quiescence_source_changed"):
                _quiescence_source(copy.copy(clock))
            original_read = prepared._finite_reads[-1]
            with pytest.raises(RuntimeError, match="finite result changed"):
                prepared._finite_read_return(copy.copy(original_read))

            def native_seconds():
                with (
                    owned_goal_storage(rig.service),
                    execution(rig.controller._agent_bridge),
                    run.phase.context.scope(),
                ):
                    return rig.service.db.automatic_work.workflow_active_seconds(
                        run.workflow_id
                    )

            def passive_seconds():
                with owned_goal_storage(rig.service):
                    return rig.service.db.automatic_work.workflow_active_seconds(
                        run.workflow_id
                    )

            def direct_receivers_refuse():
                with (
                    owned_goal_storage(rig.service),
                    execution(rig.controller._agent_bridge),
                    run.phase.context.scope(),
                    rig.service.db.connection() as conn,
                ):
                    ledger = rig.service.db.automatic_work

                    def rows():
                        return tuple(
                            tuple(tuple(row) for row in conn.execute(query))
                            for query in (
                                "SELECT * FROM workflow_active_intervals ORDER BY rowid",
                                "SELECT * FROM workflow_phase_attempts ORDER BY rowid",
                                "SELECT * FROM automatic_wake_attempts ORDER BY rowid",
                                "SELECT * FROM automatic_work_reservations ORDER BY rowid",
                            )
                        )

                    original_rows = rows()
                    for proof in (None, copy.copy(clock), clock):
                        with pytest.raises((PermissionError, AutomaticWorkRefused)):
                            ledger.resume_workflow_phase(
                                run.phase.attempt.id,
                                owner_id=run.phase.attempt.owner_id,
                                _quiescence=proof,
                                _cleanup=True,
                            )
                        assert rows() == original_rows
                    with pytest.raises(PermissionError):
                        ledger.resume_workflow_phase(
                            run.phase.attempt.id,
                            owner_id=run.phase.attempt.owner_id,
                            _quiescence=clock,
                        )
                    assert rows() == original_rows

            await asyncio.to_thread(direct_receivers_refuse)
            measured = await asyncio.to_thread(native_seconds)
            assert measured > 0
            if scenario == "cleanup-expired":
                assert expiry["prepared"] is prepared and expiry["phase"] is run.phase
                assert expiry["source"] is prepared.source
                assert expiry["read"] is original_read
                prepared._finite_read_return(original_read)
                assert original_read.positive and original_read.drains == [True]
                assert (
                    original_read.owner is expiry["owner"]
                    and not expiry["owner"].is_admitted
                )
                assert original_read.claim is expiry["claim"]
                assert expiry["claim"].id not in rig.runtime.worktree_activity._claims
                assert original_read.operation is expiry["operation"]
                assert _policy_identity() == expiry["policy"] and measured >= 10
            await asyncio.sleep(0.05)
            assert await asyncio.to_thread(native_seconds) == measured
            with pytest.raises(AutomaticWorkRefused, match="clock_unknown"):
                await asyncio.to_thread(passive_seconds)
            passive = await rig.service.call(rig.service.get, run.workflow_id)
            assert passive.accounting.uncertain
            assert not before.accounting.uncertain
            with rig.service.db.connection() as conn:
                phase_row = conn.execute(
                    "SELECT retirement FROM workflow_phase_attempts WHERE native_attempt_id=?",
                    (run.phase.attempt.id,),
                ).fetchone()
                assert phase_row[0] == "unknown"
                assert (
                    conn.execute(
                        "SELECT count(*) FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                        (run.workflow_id,),
                    ).fetchone()[0]
                    == 0
                )
            if scenario == "restart-live":
                _save_accounting_restart_boundary(rig, run, clock, tmp_path)
            if scenario in {"finite-owner", "finite-claim", "finite-operation"}:
                carried = original_read.proof[1].result()
                assert carried[0] is original_read and carried[1] is prepared
                assert carried[5] is original_read.owner
                assert carried[6] is original_read.claim
                assert carried[6] is not None
                intake, token = run.intake, run.token
                if scenario == "finite-owner":
                    original_read.owner = run.acceptance.owner
                    assert original_read.owner is not carried[5]
                elif scenario == "finite-claim":
                    original_read.claim = run.acceptance.claim
                    assert original_read.claim is not carried[6]
                else:
                    shadow_calls = []

                    def replacement_finish():
                        shadow_calls.append(True)
                        raise AssertionError("replacement operation must not execute")

                    assert original_read.operation is carried[7]
                    original_read.operation.finish = replacement_finish
                failure = (
                    "finite operation changed"
                    if scenario == "finite-operation"
                    else "finite result changed"
                )
                with pytest.raises(RuntimeError, match=failure):
                    prepared._finite_read_return(original_read)
                with pytest.raises(AutomaticWorkRefused, match="clock_unknown"):
                    await asyncio.to_thread(native_seconds)
                await rig.coordinator.stop(
                    run.workflow_id,
                    expected_control_revision=before.control_revision,
                )
                await asyncio.shield(run.driver)
                assert run.status == "recovery_required" and run.accounting_error
                assert rig.coordinator._runs[run.workflow_id] is run
                assert run.intake is intake and run.token is token
                assert original_read.proof[1].result() is carried
                if scenario == "finite-operation":
                    assert shadow_calls == []
                assert not authorization._native_launch
                assert [key for key, _ in rig.provider.calls] == ["planning"]
                for _ in range(2):
                    with pytest.raises(
                        RuntimeError, match="workflow_original_shutdown_unresolved"
                    ):
                        await rig.runtime.dispose()
                    assert rig.coordinator._runs[run.workflow_id] is run
                    assert run.intake is intake and run.token is token
                print(
                    "G_ACCOUNTING_ORIGINAL_FINITE_RETAINED; no creator drain",
                    flush=True,
                )
                retained = True
                os._exit(17)
            if scenario == "cleanup-late":
                from tldw_chatbook.DB.automatic_work import _CLOCK_OWNER_ID

                original_interval = clock.interval
                failure = RuntimeError(
                    "original cleanup accounting failed after commit"
                )
                entered = []
                code = type(
                    rig.service.db.automatic_work
                ).resume_workflow_phase.__code__

                def failed_cleanup_return(frame, event, result):
                    if (
                        event == "return"
                        and frame.f_code is code
                        and frame.f_locals.get("_quiescence") is clock
                        and frame.f_locals.get("_cleanup") is True
                        and result is True
                    ):
                        threading.setprofile_all_threads(None)
                        entered.append(clock)
                        raise failure

                intake, token = run.intake, run.token
                threading.setprofile_all_threads(failed_cleanup_return)
                await rig.coordinator.stop(
                    run.workflow_id,
                    expected_control_revision=before.control_revision,
                )
                await asyncio.shield(run.driver)
                assert entered == [clock]
                assert clock.mode == "cleanup" and run.phase._quiescence is clock
                assert (
                    clock.error
                    is clock.producer_error
                    is run.accounting_error
                    is failure
                )
                read, producer, retired, callback = clock.proof
                assert producer is read._producer and producer.exception() is failure
                assert (
                    retired is read.retired
                    and retired.done()
                    and retired.result() is None
                )
                assert callback is clock.callback
                assert prepared._closed and prepared._native_closed
                close_read = prepared._finite_reads[-1]
                assert close_read is not original_read
                prepared._finite_read_return(close_read)
                assert close_read.positive and close_read.proof[1].exception() is None
                assert close_read.proof[2].result() is None
                with (
                    owned_goal_storage(rig.service),
                    rig.service.db.connection() as conn,
                ):
                    previous = conn.execute(
                        "SELECT * FROM workflow_active_intervals WHERE id=?",
                        (original_interval[0],),
                    ).fetchone()
                    assert tuple(previous) == original_interval
                    opened = conn.execute(
                        "SELECT * FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                        (run.workflow_id,),
                    ).fetchall()
                    assert len(opened) == 1
                    assert opened[0]["id"] != original_interval[0]
                    assert opened[0]["clock_owner_id"] == _CLOCK_OWNER_ID
                    phase = conn.execute(
                        "SELECT status,retirement FROM workflow_phase_attempts WHERE native_attempt_id=?",
                        (run.phase.attempt.id,),
                    ).fetchone()
                    assert tuple(phase) == ("accepted", "pending")
                active = await asyncio.to_thread(passive_seconds)
                await asyncio.sleep(0.05)
                assert await asyncio.to_thread(passive_seconds) > active >= measured
                recovery = await rig.coordinator.get(run.workflow_id)
                assert recovery.status == run.status == "recovery_required"
                print(
                    "G_OPEN_CLEANUP_ACCOUNTING_FLAG",
                    recovery.accounting.uncertain,
                    flush=True,
                )
                assert recovery.accounting.used == before.accounting.used
                assert [key for key, _ in rig.provider.calls] == ["planning"]
                assert not authorization._native_launch
                assert rig.coordinator._runs[run.workflow_id] is run
                assert run.intake is intake and run.token is token
                with pytest.raises(
                    ValueError, match="remaining_requires_clean_checkpoint"
                ):
                    await rig.service.call(
                        rig.service.prepare_remaining_decision,
                        run.workflow_id,
                        phase_id="phase-1",
                    )
                quality = await rig.service.call(
                    rig.service.prepare_quality_review,
                    run.workflow_id,
                    phase_id="phase-1",
                )
                assert (
                    not quality.evidence.ready and quality.reason == "phase_not_settled"
                )
                with pytest.raises(
                    ValueError, match="resume_requires_clean_checkpoint"
                ):
                    await rig.service.call(
                        rig.service.resume,
                        run.workflow_id,
                        expected_control_revision=recovery.control_revision,
                        command_id="late-cleanup-resume",
                    )
                with pytest.raises(
                    PermissionError, match="original producer still active"
                ):
                    await rig.coordinator.approve(
                        decision, command_id="late-cleanup-admission"
                    )
                for _ in range(2):
                    with pytest.raises(
                        RuntimeError, match="workflow_original_shutdown_unresolved"
                    ):
                        await rig.runtime.dispose()
                    assert rig.coordinator._runs[run.workflow_id] is run
                    assert run.intake is intake and run.token is token
                print(
                    "G_ACCOUNTING_ORIGINAL_CLEANUP_UNKNOWN; no creator drain",
                    flush=True,
                )
                retained = True
                os._exit(17)
            if scenario in {"cleanup-held", "cleanup-budget", "cleanup-expired"}:
                if scenario in {"cleanup-budget", "cleanup-expired"}:
                    from tldw_chatbook.Agents.automatic_work_budget import (
                        AutomaticWorkLimits,
                    )

                    if scenario == "cleanup-budget":
                        os.environ["TLDW_AGENTS_MAX_WORKFLOW_WALL_SECONDS"] = "0"
                    assert AutomaticWorkLimits.from_settings(
                        "workflow_phase"
                    ).wall_seconds == (0 if scenario == "cleanup-budget" else 10)

                    def no_new_work():
                        with (
                            owned_goal_storage(rig.service),
                            execution(rig.controller._agent_bridge),
                            run.phase.context.scope(),
                            rig.service.db.connection() as conn,
                            pytest.raises(AutomaticWorkRefused, match="wall_budget"),
                        ):
                            rig.service.db.automatic_work._check_admission(
                                conn,
                                run.phase.attempt.chain_id,
                                kind="generation",
                                limits=AutomaticWorkLimits.from_settings(
                                    "workflow_phase"
                                ),
                            )

                    await asyncio.to_thread(no_new_work)
                from tldw_chatbook.Chat.console_workflow_preparation import (
                    _PreparedWorkflowReply,
                )

                close_code = next(
                    code
                    for code in _PreparedWorkflowReply.close.__code__.co_consts
                    if getattr(code, "co_name", None) == "finish"
                )
                entered, release = threading.Event(), threading.Event()
                cleanup = {}

                def original_cleanup(frame, event, _result):
                    if (
                        event == "call"
                        and frame.f_code is close_code
                        and frame.f_locals.get("self") is prepared
                    ):
                        threading.setprofile_all_threads(None)
                        cleanup["read"] = prepared._finite_reads[-1]
                        original = frame.f_back
                        assert original.f_locals["record"] is cleanup["read"]
                        assert cleanup["read"].callback.__code__ is original.f_code
                        assert original.f_locals["callback"].__code__ is frame.f_code
                        assert cleanup["read"].issuing is run.driver
                        entered.set()
                        assert release.wait(10), "original cleanup hold not released"

                intake, token = run.intake, run.token
                original_interval = clock.interval
                threading.setprofile_all_threads(original_cleanup)
                stopping = asyncio.create_task(
                    rig.coordinator.stop(
                        run.workflow_id,
                        expected_control_revision=before.control_revision,
                    )
                )
                try:
                    assert await asyncio.to_thread(entered.wait, 10)
                    assert not cleanup["read"].proof[1].done()
                    if scenario == "cleanup-budget":
                        assert (
                            run.phase._quiescence is clock and clock.mode == "cleanup"
                        )
                        assert clock.error is run.accounting_error
                        assert isinstance(clock.error, AutomaticWorkRefused)
                        assert clock.error.reason == "workflow_native_source_changed"
                        with pytest.raises(AutomaticWorkRefused, match="clock_unknown"):
                            await asyncio.to_thread(passive_seconds)
                    else:
                        assert run.phase._quiescence is None and clock.mode == "active"
                        active = await asyncio.to_thread(passive_seconds)
                        await asyncio.sleep(0.05)
                        assert (
                            await asyncio.to_thread(passive_seconds)
                            > active
                            >= measured
                        )
                    assert run.intake is intake and run.token is token
                    assert [key for key, _ in rig.provider.calls] == ["planning"]
                finally:
                    release.set()
                await stopping
                await asyncio.shield(run.driver)
                prepared._finite_read_return(cleanup["read"])
                stopped = await rig.coordinator.get(run.workflow_id)
                if scenario == "cleanup-budget":
                    close_read = cleanup["read"]
                    assert close_read.positive and close_read.scopes_returned
                    assert close_read.proof[1].exception() is None
                    assert close_read.proof[2].result() is None
                    assert prepared._closed and prepared._native_closed
                    assert run.status == stopped.status == "recovery_required"
                    assert stopped.accounting.uncertain
                    assert stopped.accounting.used == before.accounting.used
                    assert (
                        run.phase._quiescence is clock
                        and clock.error is run.accounting_error
                    )
                    assert run.intake is intake and run.token is token
                    with (
                        owned_goal_storage(rig.service),
                        rig.service.db.connection() as conn,
                    ):
                        row = conn.execute(
                            "SELECT * FROM workflow_active_intervals WHERE id=?",
                            (original_interval[0],),
                        ).fetchone()
                        assert tuple(row) == original_interval
                        assert (
                            conn.execute(
                                "SELECT count(*) FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                                (run.workflow_id,),
                            ).fetchone()[0]
                            == 0
                        )
                        assert (
                            conn.execute(
                                "SELECT retirement FROM workflow_phase_attempts WHERE native_attempt_id=?",
                                (run.phase.attempt.id,),
                            ).fetchone()[0]
                            == "unknown"
                        )
                    with pytest.raises(AutomaticWorkRefused, match="clock_unknown"):
                        await asyncio.to_thread(passive_seconds)
                    assert not authorization._native_launch
                    assert [key for key, _ in rig.provider.calls] == ["planning"]
                    for _ in range(2):
                        with pytest.raises(
                            RuntimeError, match="workflow_original_shutdown_unresolved"
                        ):
                            await rig.runtime.dispose()
                        assert rig.coordinator._runs[run.workflow_id] is run
                        assert run.intake is intake and run.token is token
                        assert clock.error is run.accounting_error
                    print(
                        "G_ACCOUNTING_ORIGINAL_POLICY_CHANGED; no creator drain",
                        flush=True,
                    )
                    retained = True
                    os._exit(17)
                assert stopped.status == "stopped"
                assert stopped.accounting.used["generation"] == 2
                assert stopped.accounting.used["child_launch"] == 0
                assert not stopped.accounting.uncertain
                return
        observed = []
        capacity = rig.runtime.execution_capacity

        def increase_child_capacity():
            os.environ["TLDW_AGENTS_RESERVED_MANUAL_SUBAGENTS"] = "0"
            capacity.set_child_limits(1, 0)

        if sys.argv[1] in {"capacity-stop", "capacity-fenced"}:
            intake, token = run.intake, run.token
            saved_lifetime, saved_scope = prepared._lifetime, prepared._source_scope
            closed_reads, scope_returns = [], []
            if sys.argv[1] == "capacity-stop":
                invoke_code = next(
                    code
                    for code in native_reads.run_preparation_read.__code__.co_consts
                    if getattr(code, "co_name", None) == "invoke"
                )

                def original_close_return(frame, event, result):
                    if (
                        event == "return"
                        and frame.f_code is invoke_code
                        and frame.f_locals["read"].source is prepared
                    ):
                        closed_reads.append(frame.f_locals["read"])
                    elif (
                        event == "return"
                        and frame.f_code is saved_scope.gen.gi_code
                        and frame.f_locals.get("self") is rig.runtime
                    ):
                        parent = frame.f_back
                        while (
                            parent is not None
                            and parent.f_code
                            is not type(prepared)._release_configuration.__code__
                        ):
                            parent = parent.f_back
                        if (
                            parent is not None
                            and parent.f_locals.get("self") is prepared
                            and parent.f_locals.get("scope") is saved_scope
                        ):
                            scope_returns.append(saved_scope)
                            threading.setprofile_all_threads(None)

                threading.setprofile_all_threads(original_close_return)
                stopping = await rig.coordinator.stop(
                    rig.workflow.id, expected_control_revision=before.control_revision
                )
                assert stopping.status in {"stopping", "stopped"}
            else:
                native.fleet.fence()
                with native.fleet._lock:
                    assert native.fleet._fenced and not native.fleet._live_ids
                increase_child_capacity()
            await asyncio.wait_for(asyncio.shield(run.driver), 20)
            assert (
                not authorization._native_launch and not authorization._native_dispatch
            )
            assert (
                prepared.authorization is authorization
                and authorization._native_preparation[0] is native
            )
            assert run.phase._task_scopes["task-1"] is authorization
            assert prepared._lifetime is saved_lifetime
            snapshot = await rig.coordinator.get(rig.workflow.id)
            assert snapshot.results == () and not snapshot.accounting.uncertain
            assert snapshot.accounting.used["generation"] == 2
            assert (
                snapshot.accounting.used["child_launch"] == 0
                and snapshot.accounting.used["model_call"] == 1
            )
            with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
                phase_row = conn.execute(
                    "SELECT status,retirement FROM workflow_phase_attempts WHERE native_attempt_id=?",
                    (run.phase.attempt.id,),
                ).fetchone()
                task_row = conn.execute(
                    "SELECT status,retirement,run_id FROM workflow_task_attempts WHERE id=?",
                    (authorization.claim.id,),
                ).fetchone()
            if sys.argv[1] == "capacity-stop":
                assert (
                    snapshot.status == "stopped"
                    and prepared._closed
                    and prepared._native_closed
                )
                assert prepared._source_scope is None and scope_returns == [saved_scope]
                assert (
                    saved_scope.gen.gi_frame is None and not saved_scope.gen.gi_running
                )
                assert prepared not in rig.runtime._workflow_preparations
                assert tuple(phase_row) == ("closed", "drained") and tuple(
                    task_row
                ) == ("closed", "drained", None)
                (read,) = closed_reads
                assert (
                    read._producer.exception() is None and read.retired.result() is None
                )
                control_read, producer, retired, _callback = run.control_read
                assert (
                    producer is control_read._producer
                    and producer.exception() is None
                    and retired.result() is None
                )
                assert rig.workflow.id not in rig.coordinator._runs
                assert rig.store.received_turn_for_session(run.session_id) is None
                assert (
                    run.session_id
                    not in rig.controller._fleet_wake._automatic_primary_claims
                )
                assert rig.runtime.execution_capacity.snapshot().executions == ()
                increase_child_capacity()
                await asyncio.sleep(0)
                assert (
                    not authorization._native_launch
                    and not authorization._native_dispatch
                )
                assert [key for key, _args in rig.provider.calls] == ["planning"]
                return
            assert snapshot.status == run.status == "recovery_required"
            assert (
                tuple(phase_row) == ("accepted", "pending")
                and task_row["run_id"] is None
            )
            assert not prepared._closed and not prepared._native_closed
            assert (
                prepared._source_scope is saved_scope
                and saved_scope.gen.gi_frame is not None
            )
            assert rig.coordinator._runs[rig.workflow.id] is run
            assert rig.store.received_turn_for_session(run.session_id) is intake
            assert (
                rig.controller._fleet_wake._automatic_primary_claims[run.session_id]
                is token
            )
            for _ in range(2):
                with pytest.raises(
                    RuntimeError, match="workflow_original_shutdown_unresolved"
                ):
                    await rig.runtime.dispose()
                assert not rig.runtime._disposed and not capacity.snapshot().closed
                assert (
                    rig.coordinator._runs[rig.workflow.id] is run
                    and not prepared._closed
                )
                assert [key for key, _args in rig.provider.calls] == ["planning"]
            retained = True
            raise RuntimeError("G_REQUIRED_CUSTODY_RETAINED_FLEET_FENCE")
        if sys.argv[1] == "capacity-lost-wake":
            original_ready = ConsoleWorkflowCoordinator._wave_ready.__code__

            def release_after_original_recheck(frame, event, result):
                if frame.f_code is original_ready and event == "return":
                    assert frame.f_locals["run"] is run and result == []
                    assert frame.f_locals["candidates"] == [prepared]
                    observed.append((run, prepared))
                    sys.setprofile(None)
                    increase_child_capacity()

            sys.setprofile(release_after_original_recheck)
            # A genuine independent availability increase triggers one recheck;
            # the actual child release happens after that recheck, before wait.
            os.environ["TLDW_AGENTS_MAX_RUNTIME_TOOL_WORKERS"] = str(
                capacity.max_tool_workers + 1
            )
            capacity.set_tool_limits(
                capacity.max_tool_workers + 1, capacity.reserved_manual_tool_workers
            )
        else:
            increase_child_capacity()
        if accounting:
            await rig.wait_for_provider_entry("task-1")
            assert run.phase._quiescence is None and clock.mode == "active"
            active = await asyncio.to_thread(native_seconds)
            await asyncio.sleep(0.05)
            assert await asyncio.to_thread(native_seconds) > active >= measured
            with rig.service.db.connection() as conn:
                assert (
                    conn.execute(
                        "SELECT count(*) FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                        (run.workflow_id,),
                    ).fetchone()[0]
                    == 1
                )
            rig.provider_release()
        complete = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert (
            prepared.authorization is authorization
            and authorization._native_preparation[0] is native
        )
        assert run.phase._task_scopes["task-1"] is authorization
        receipt = authorization._native_launch[0]
        assert (
            complete.results[0].attempt_id
            == authorization.claim.id
            == receipt.attempt_id
        )
        assert (
            complete.results[0].run_id
            == receipt.run_id
            == authorization.claim.launch_run_id
        )
        assert (
            complete.accounting.used["generation"] == 2
            and complete.accounting.used["child_launch"] == 1
        )
        assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
        if scenario == "capture" or scenario.startswith("capture-"):
            from tldw_chatbook.Chat.console_workflow_wave import _workflow_wave_return
            from tldw_chatbook.Workspaces.change_review_finalization import (
                ChangeReviewCompletion,
            )

            assert len(run.waves) == 1
            saved = run.waves[0]
            assert _workflow_wave_return(saved[0], saved) is saved
            assert saved[2].result() is ChangeReviewCompletion.OBSERVED
            assert saved[3].result() is ChangeReviewCompletion.PUBLISHED
            assert {item[0] for item in saved[7]} == {"baseline", "finalize"}
            assert saved[6][0][0] is prepared and saved[6][0][1] is receipt
            if scenario.startswith("capture-"):
                await asyncio.shield(run.driver)
                assert run.workflow_id not in rig.coordinator._runs
                wave = saved[0]
                assert wave.closed and _workflow_wave_return(wave, saved) is saved
                with pytest.raises(RuntimeError, match="wave return changed"):
                    _workflow_wave_return(copy.copy(wave), saved)
                copied_source = (*saved[:4], copy.copy(saved[4]), *saved[5:])
                with pytest.raises(RuntimeError, match="wave return changed"):
                    _workflow_wave_return(wave, copied_source)
                first = saved[7][0]
                copied_ticket = (first[0], copy.copy(first[1]), *first[2:])
                changed = (*saved[:7], (copied_ticket, *saved[7][1:]))
                with pytest.raises(RuntimeError, match="wave return changed"):
                    _workflow_wave_return(wave, changed)
                assert _workflow_wave_return(wave, saved) is saved
                replacement_calls = []

                def replacement(*_args, **_kwargs):
                    replacement_calls.append(True)
                    raise AssertionError("changed returned source must not execute")

                if scenario == "capture-ticket":
                    first[1].returned = replacement
                elif scenario == "capture-source":
                    object.__setattr__(saved[4], "check", replacement)
                else:
                    assert scenario == "capture-wave"
                    wave.check_snapshot_source = replacement
                with pytest.raises(
                    (PermissionError, RuntimeError),
                    match="original workflow (wave source|snapshot return) changed",
                ):
                    _workflow_wave_return(wave, saved)
                assert replacement_calls == []
                assert not wave.runtime._workflow_waves
                assert rig.runtime.execution_capacity.snapshot().executions == ()
                assert complete.accounting.used["model_call"] == 2
                assert not complete.accounting.uncertain
        if sys.argv[1] == "capacity-lost-wake":
            assert observed == [(run, prepared)]
    finally:
        sys.setprofile(None)
        threading.setprofile_all_threads(None)
        if not retained:
            await rig.close()


@pytest.mark.parametrize(
    "mode", ["capacity-notification", "capacity-lost-wake", "capacity-accounting"]
)
def test_original_native_capacity_notification_wakes_armed_phase(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize(
    "scenario",
    [
        "capture",
        "capture-ticket",
        "capture-source",
        "capture-wave",
        "finite-owner",
        "finite-claim",
        "finite-operation",
        "quiesce-unknown",
        "cleanup-held",
        "cleanup-expired",
        "cleanup-budget",
        "cleanup-late",
    ],
)
def test_original_accounting_requires_same_finite_and_wave_returns(tmp_path, scenario):
    def run():
        _run(
            tmp_path,
            "capacity-accounting",
            scenario,
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )

    if scenario.startswith("finite-") or scenario in {
        "quiesce-unknown",
        "cleanup-late",
        "cleanup-budget",
    }:
        marker = (
            "G_ACCOUNTING_ORIGINAL_QUIESCE_UNKNOWN"
            if scenario == "quiesce-unknown"
            else "G_ACCOUNTING_ORIGINAL_CLEANUP_UNKNOWN"
            if scenario == "cleanup-late"
            else "G_ACCOUNTING_ORIGINAL_POLICY_CHANGED"
            if scenario == "cleanup-budget"
            else "G_ACCOUNTING_ORIGINAL_FINITE_RETAINED"
        )
        with pytest.raises(AssertionError, match=marker):
            run()
    else:
        run()


@pytest.mark.parametrize("mode", ["capacity-stop", "capacity-fenced"])
def test_armed_capacity_stop_drains_but_native_fence_retains_original(tmp_path, mode):
    if mode == "capacity-fenced":
        with pytest.raises(
            AssertionError, match="G_REQUIRED_CUSTODY_RETAINED_FLEET_FENCE"
        ):
            _run(
                tmp_path,
                mode,
                "native",
                script=Path(__file__)
                .with_name("workflow_coordinator_probe.py")
                .read_text(),
                timeout=60,
            )
    else:
        _run(
            tmp_path,
            mode,
            "native",
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


def _save_accounting_restart_boundary(rig, run, cell, root):
    """Save passive original rows before deliberately losing this private process."""
    import json
    import os

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

    assert run.phase._quiescence is cell
    assert [key for key, _ in rig.provider.calls] == ["planning"]
    with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
        conn.execute("BEGIN")
        try:
            intervals = [
                list(row)
                for row in conn.execute(
                    "SELECT * FROM workflow_active_intervals WHERE workflow_id=? ORDER BY id",
                    (run.workflow_id,),
                )
            ]
            assert (
                conn.execute(
                    "SELECT count(*) FROM workflow_active_intervals WHERE workflow_id=? AND ended_at IS NULL",
                    (run.workflow_id,),
                ).fetchone()[0]
                == 0
            )
            assert (
                conn.execute(
                    "SELECT state FROM automatic_wake_attempts WHERE id=?",
                    (run.phase.attempt.id,),
                ).fetchone()[0]
                == "accepted"
            )
            assert (
                conn.execute(
                    "SELECT retirement FROM workflow_phase_attempts WHERE native_attempt_id=?",
                    (run.phase.attempt.id,),
                ).fetchone()[0]
                == "unknown"
            )
            reservations = [
                list(row)
                for row in conn.execute(
                    "SELECT id,kind,amount,actual_amount,state FROM automatic_work_reservations WHERE chain_id=? ORDER BY id",
                    (run.phase.attempt.chain_id,),
                )
            ]
            runs = [
                list(row)
                for row in conn.execute(
                    "SELECT id,status FROM agent_runs WHERE work_chain_id=? ORDER BY id",
                    (run.phase.attempt.chain_id,),
                )
            ]
        finally:
            conn.rollback()
    receipt = {
        "pid": os.getpid(),
        "pgid": os.getpgid(0),
        "db": str(rig.service.db.db_path),
        "workflow": run.workflow_id,
        "chain": run.phase.attempt.chain_id,
        "attempt": run.phase.attempt.id,
        "owner": run.phase.attempt.owner_id,
        "intervals": intervals,
        "reservations": reservations,
        "runs": runs,
        "positive_idle": cell.mode == "idle",
    }
    (root / "accounting-boundary.json").write_text(json.dumps(receipt))
    print("G_ORIGINAL_ACCOUNTING_RESTART_BOUNDARY; no creator drain", flush=True)
    os._exit(17)


def _read_accounting_after_restart(root):
    """A fresh native reader receives rows, never a restored workflow authority."""
    import json
    from uuid import uuid4

    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
    from tldw_chatbook.Backup_Recovery import participants, storage_admission
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    receipt = json.loads((root / "accounting-boundary.json").read_text())
    assert current_automatic_work() is None
    database = AgentRunsDB(receipt["db"])
    participant = participants._repository_participant(database)
    with storage_admission._lock:
        original_connections = tuple(participant.connections.items())
    assert original_connections
    ledger = database.automatic_work
    owner = "actual-restart-" + uuid4().hex
    try:
        with database.connection() as conn:
            assert (
                conn.execute(
                    "SELECT state FROM automatic_wake_attempts WHERE id=?",
                    (receipt["attempt"],),
                ).fetchone()[0]
                == "accepted"
            )
        with pytest.raises(AutomaticWorkRefused, match="clock_unknown"):
            ledger.workflow_active_seconds(receipt["workflow"])
        ledger.recover(current_owner_id=owner)
        for _ in range(2):
            with database.connection() as conn:
                assert (
                    conn.execute(
                        "SELECT state FROM automatic_wake_attempts WHERE id=?",
                        (receipt["attempt"],),
                    ).fetchone()[0]
                    == "review_required"
                )
                assert tuple(
                    conn.execute(
                        "SELECT status,retirement FROM workflow_phase_attempts WHERE native_attempt_id=?",
                        (receipt["attempt"],),
                    ).fetchone()
                ) == ("uncertain", "unknown")
                assert (
                    conn.execute(
                        "SELECT status FROM workflow_runs WHERE id=?",
                        (receipt["workflow"],),
                    ).fetchone()[0]
                    == "recovery_required"
                )
                assert [
                    list(row)
                    for row in conn.execute(
                        "SELECT * FROM workflow_active_intervals WHERE workflow_id=? ORDER BY id",
                        (receipt["workflow"],),
                    )
                ] == receipt["intervals"]
                reservations = [
                    list(row)
                    for row in conn.execute(
                        "SELECT id,kind,amount,actual_amount,state FROM automatic_work_reservations WHERE chain_id=? ORDER BY id",
                        (receipt["chain"],),
                    )
                ]
                assert len(reservations) == len(receipt["reservations"])
                for before, after in zip(
                    receipt["reservations"], reservations, strict=True
                ):
                    assert after[:4] == before[:4]
                    assert after[4] == (
                        "uncertain" if before[4] == "reserved" else before[4]
                    )
                assert [
                    list(row)
                    for row in conn.execute(
                        "SELECT id,status FROM agent_runs WHERE work_chain_id=? ORDER BY id",
                        (receipt["chain"],),
                    )
                ] == receipt["runs"]
            with pytest.raises(AutomaticWorkRefused, match="clock_unknown"):
                ledger.workflow_active_seconds(receipt["workflow"])
            snapshot = ledger.snapshot(receipt["chain"])
            assert snapshot.status == "review_required" and snapshot.uncertain
            assert snapshot.used["generation"] == 2
            assert snapshot.used["model_call"] == 1
            with pytest.raises(AutomaticWorkRefused):
                ledger.check_active(receipt["chain"], owner_id=owner)
            ledger.recover(current_owner_id=owner)
    finally:
        database.close()
    with storage_admission._lock:
        assert not participant.connections
        assert all(
            lease not in storage_admission._live_leases
            for _, lease in original_connections
        )
    print("G_FRESH_ACCOUNTING_READER_RETURNED; producer was not drained", flush=True)


@pytest.mark.parametrize("scenario", ["restart-live", "restart-failed"])
def test_actual_fresh_process_keeps_unknown_workflow_clock(tmp_path, scenario):
    import json
    import os
    import subprocess
    import sys
    import time

    producer = tmp_path / "producer"
    producer.mkdir()
    with pytest.raises(
        AssertionError, match="G_ORIGINAL_ACCOUNTING_RESTART_BOUNDARY"
    ) as producer_exit:
        _run(
            producer,
            "capacity-accounting",
            scenario,
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )
    boundary = json.loads((producer / "accounting-boundary.json").read_text())
    original = producer_exit.value.__traceback__
    while original.tb_frame.f_code is not _run.__code__:
        original = original.tb_next
    returned = original.tb_frame.f_locals["result"]
    assert returned.returncode == 17
    boundary["returned_command"] = returned.args
    boundary["actual_returncode"] = returned.returncode
    assert boundary["positive_idle"] is (scenario == "restart-live")
    script = """
import os
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
with user_fixture_default_owner():
    from Tests.Chat.test_workflow_activity_admission import _read_accounting_after_restart
    _read_accounting_after_restart(Path(os.environ["XDG_DATA_HOME"]).parent)
"""
    environment = dict(os.environ)
    environment.update(
        HOME=str(producer / "home"),
        USERPROFILE=str(producer / "home"),
        XDG_CONFIG_HOME=str(producer / "config"),
        XDG_DATA_HOME=str(producer / "data"),
        TLDW_CONFIG_PATH=str(producer / "config" / "config.toml"),
        TLDW_TEST_MODE="1",
        TLDW_DISABLE_CONFIG_WATCH="1",
    )
    command = [sys.executable, "-X", "utf8", "-c", script]
    lifecycle = {"producer": boundary, "command": command, "created_epoch": time.time()}
    with (tmp_path / "accounting-reader.log").open("w") as output:
        reader = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        lifecycle.update(pid=reader.pid, pgid=os.getpgid(reader.pid))
        (tmp_path / "accounting-reader-lifecycle.json").write_text(
            json.dumps(lifecycle)
        )
        try:
            code = reader.wait(timeout=20)
        except subprocess.TimeoutExpired:
            reader.kill()  # Only this SAME fresh reader Popen, never a foreign PID.
            reader.wait(timeout=5)
            lifecycle.update(
                returncode=reader.returncode, forced=True, returned_epoch=time.time()
            )
            (tmp_path / "accounting-reader-lifecycle.json").write_text(
                json.dumps(lifecycle)
            )
            raise
    lifecycle.update(returncode=code, forced=False, returned_epoch=time.time())
    (tmp_path / "accounting-reader-lifecycle.json").write_text(json.dumps(lifecycle))
    output = (tmp_path / "accounting-reader.log").read_text()
    assert code == 0, output[-8000:]
    assert "G_FRESH_ACCOUNTING_READER_RETURNED" in output
