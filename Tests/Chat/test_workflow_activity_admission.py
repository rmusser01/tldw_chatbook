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

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
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
        await rig.coordinator.approve(decision, command_id="capacity-event")
        run = rig.coordinator._runs[rig.workflow.id]
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
        if sys.argv[1] == "capacity-lost-wake":
            assert observed == [(run, prepared)]
    finally:
        sys.setprofile(None)
        threading.setprofile_all_threads(None)
        if not retained:
            await rig.close()


@pytest.mark.parametrize("mode", ["capacity-notification", "capacity-lost-wake"])
def test_original_native_capacity_notification_wakes_armed_phase(tmp_path, mode):
    _run(
        tmp_path,
        mode,
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


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
