"""The approved phase holds native admission through actual task settlement."""

from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

pytestmark = pytest.mark.bootstrap_profile


async def journey(tmp_path, monkeypatch):
    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        snapshot = await rig.wait_for_status("awaiting_phase_review")
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        assert snapshot.accounting.used["generation"] == 1
        assert snapshot.accounting.used["child_launch"] == 0
        assert rig.runtime.execution_capacity.snapshot().executions == ()
    finally:
        await rig.close()


def test_planning_never_launches_execution_before_approval(tmp_path):
    _run(
        tmp_path,
        "scheduling",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def routing_journey(tmp_path, monkeypatch):
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowRuntimeSnapshot

    rig = await build_workflow_rig(
        tmp_path,
        monkeypatch,
        WorkflowProvider(hold=("planning",)),
        target_models=("worker-a", "worker-b"),
    )
    original = rig.workflow.request
    seen = []
    unsubscribe = rig.coordinator.subscribe(seen.append)
    try:
        rig.workflow = await rig.coordinator.start(
            original, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_provider_entry()
        assert rig.workflow.request == original
        assert tuple(target.model for target in original.targets) == (
            "worker-a",
            "worker-b",
        )
        assert rig.provider.calls[0][1]["model"] == "test-model"
        assert all(
            type(view) is WorkflowRuntimeSnapshot and not hasattr(view, "plans")
            for view in seen
        )
        assert all(
            len(view.tasks) <= 12 and len(view.model_dump_json().encode()) < 128 * 1024
            for view in seen
        )
        unsubscribe()
        count = len(seen)
        paused = await rig.coordinator.pause(
            rig.workflow.id, expected_control_revision=rig.workflow.control_revision
        )
        assert (
            paused.status == "pausing"
            and rig.runtime.execution_capacity.snapshot().executions
        )
        with pytest.raises(
            PermissionError, match="workflow original producer still active"
        ):
            await rig.coordinator.resume(
                rig.workflow.id,
                expected_control_revision=paused.control_revision,
                command_id="active-planning-resume",
            )
        rig.provider_release()
        paused = await rig.wait_for_status("paused")
        assert (
            paused.plans[-1].request == original
            and paused.accounting.used["generation"] == 1
        )
        assert (
            rig.runtime.execution_capacity.snapshot().executions == ()
            and len(seen) == count
        )
        with pytest.raises(ValueError, match="revision_conflict"):
            await rig.coordinator.resume(
                rig.workflow.id,
                expected_control_revision=paused.control_revision - 1,
                command_id="stale-resume",
            )
        resumed = await rig.coordinator.resume(
            rig.workflow.id,
            expected_control_revision=paused.control_revision,
            command_id="planning-resume",
        )
        assert (
            resumed.status == "awaiting_phase_review" and resumed.plans == paused.plans
        )
        assert (
            await rig.coordinator.resume(
                rig.workflow.id,
                expected_control_revision=paused.control_revision,
                command_id="planning-resume",
            )
            == resumed
        )
        with pytest.raises(ValueError, match="workflow_command_conflict"):
            await rig.coordinator.start(
                original.model_copy(update={"objective": "different"}),
                launch_id=rig.workflow.launch_id,
            )
        invalid = original.model_copy(
            update={
                "targets": (
                    original.targets[0].model_copy(
                        update={"endpoint_ref": "unqualified-endpoint"}
                    ),
                )
            }
        )
        with pytest.raises(PermissionError, match="requested target unavailable"):
            await rig.coordinator.start(invalid, launch_id="unqualified-target")
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            assert conn.execute("SELECT COUNT(*) FROM workflow_runs").fetchone()[0] == 1
        assert [key for key, _ in rig.provider.calls] == ["planning"]
    finally:
        unsubscribe()
        await rig.close()


def test_all_targets_and_separate_planner_pause_and_unsubscribe(tmp_path):
    _run(
        tmp_path,
        "routing",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def endpoint_journey(tmp_path, monkeypatch):
    import copy

    endpoint = "http://127.0.0.1:19431/v1"
    rig = await build_workflow_rig(
        tmp_path,
        monkeypatch,
        WorkflowProvider(),
        session_base_url=endpoint,
        target_models=("test-model", "worker-b"),
    )
    first = rig.provider.plan["phases"][0]["tasks"][0]
    second = copy.deepcopy(first)
    second.update(
        id="task-2",
        instructions="Execute task-2",
        target=rig.workflow.request.targets[1].model_dump(mode="json"),
    )
    rig.provider.plan["phases"][0]["tasks"].append(second)
    request = rig.workflow.request
    try:
        rig.workflow = await rig.coordinator.start(
            request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        await rig.coordinator.approve(decision, command_id="endpoint-approve")
        complete = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert complete.request == request and len(complete.results) == 2
        calls = dict(rig.provider.calls)
        assert set(calls) == {"planning", "task-1", "task-2"}
        assert calls["planning"].get("api_base_url") == endpoint
        assert calls["task-1"].get("api_base_url") == endpoint
        assert calls["task-2"].get("api_base_url") != endpoint
        assert calls["task-2"]["model"] == "worker-b"
        paused = await rig.coordinator.pause(
            rig.workflow.id, expected_control_revision=complete.control_revision
        )
        with pytest.raises(ValueError, match="resume_requires_completed_planning"):
            await rig.coordinator.resume(
                rig.workflow.id,
                expected_control_revision=paused.control_revision,
                command_id="latest-execution-is-not-planning",
            )
        unchanged = await rig.coordinator.get(rig.workflow.id)
        assert (
            unchanged.plans == complete.plans and unchanged.results == complete.results
        )
        assert (
            unchanged.accounting == complete.accounting and len(rig.provider.calls) == 3
        )
    finally:
        await rig.close()


def test_original_session_endpoint_and_unrelated_configured_target(tmp_path):
    _run(
        tmp_path,
        "endpoint",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def capacity_race_journey(tmp_path, monkeypatch):
    import asyncio
    import copy
    import inspect
    import os
    import sys
    import threading
    import time
    import types

    from tldw_chatbook.Agents.agent_service import AgentService, WorkflowChildReceipt
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply
    from tldw_chatbook.Chat.console_workflow_wave import _WorkflowWave
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewCompletion,
    )

    mode = sys.argv[1]
    fault = mode in {"inner-wait-exit", "outer-wait-exit"}
    retained_exit = False
    os.environ["TLDW_AGENTS_RESERVED_MANUAL_SUBAGENTS"] = "0"
    os.environ["TLDW_AGENTS_MAX_LIVE_SUBAGENTS"] = "2"
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("task-1",)), change_review=True
    )
    first_task = rig.provider.plan["phases"][0]["tasks"][0]
    second = copy.deepcopy(first_task)
    second.update(id="task-2", instructions="Execute task-2")
    rig.provider.plan["phases"][0]["tasks"].append(second)
    first_launched, second_waited = threading.Event(), threading.Event()
    original_launch = AgentService.launch_workflow_task.__code__
    originals = {}
    waits = []
    first_receipts = []
    saved = []
    retry_waves = []
    invoke_code = next(
        code
        for code in _PreparedWorkflowReply._read.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    capture_code = next(
        code
        for code in _WorkflowWave._open.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "capture"
    )
    installed = []
    from tldw_chatbook.Backup_Recovery.participants import _core_operation

    inner_code = next(
        code
        for code in original_launch.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "launch"
    )
    wave_launch_code = next(
        code
        for code in _WorkflowWave._launch.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "launch"
    )
    core_exit_code = _core_operation.__wrapped__.__code__
    outer_exit_code = _PreparedWorkflowReply.activate_source.__wrapped__.__code__

    def save_original(authorization, native, candidate):
        prepared = authorization._bridge_preparation[0]
        assert (
            prepared.authorization is authorization and prepared._wave is installed[0]
        )
        original = types.SimpleNamespace(
            prepared=prepared,
            authorization=authorization,
            native=native,
            frame=prepared.native,
            configuration=prepared.configuration,
            source=prepared.source,
            claim=authorization.claim,
            phase=prepared.phase,
            lifetime=prepared._lifetime,
            source_scope=prepared._source_scope,
            first_wave=installed[0],
            candidate=candidate,
            operation=prepared._launch_operation,
            fence=installed[0].fence,
        )
        saved.append(original)
        return original

    def actual_capacity_race(frame, event, result):
        if fault and event == "return":
            if (
                mode == "inner-wait-exit"
                and frame.f_code is inner_code
                and type(result) is tuple
            ):
                parent = frame.f_back
                while parent is not None and parent.f_code is not original_launch:
                    parent = parent.f_back
                assert parent is not None
                authorization = parent.f_locals["authorization"]
                if authorization.task.id == "task-2":
                    native = authorization._native_preparation[0]
                    assert native.wait_return == [None]
                    save_original(authorization, native, result)
            if saved:
                original = saved[0]
                expected = (
                    core_exit_code if mode == "inner-wait-exit" else outer_exit_code
                )
                if frame.f_code is expected:
                    parent = frame.f_back
                    target = (
                        original_launch
                        if mode == "inner-wait-exit"
                        else wave_launch_code
                    )
                    while parent is not None and parent.f_code is not target:
                        parent = parent.f_back
                    matched = parent is not None and (
                        parent.f_locals.get("authorization") is original.authorization
                        if mode == "inner-wait-exit"
                        else parent.f_locals.get("item") is original.prepared
                        and parent.f_locals.get("wait") is original.candidate
                        and frame.f_locals.get("self") is original.prepared
                    )
                    if matched:
                        threading.setprofile_all_threads(None)
                        second_waited.set()
                        print(
                            "G_SAME_ORIGINAL_WAIT_EXIT_ERROR",
                            mode,
                            time.monotonic(),
                            flush=True,
                        )
                        raise original.candidate[0]
        if frame.f_code is not original_launch:
            return
        authorization = frame.f_locals["authorization"]
        native = authorization._native_preparation[0]
        if event == "call":
            print(
                "G_CAPACITY_LAUNCH_ENTRY",
                authorization.task.id,
                time.monotonic(),
                flush=True,
            )
            originals.setdefault(authorization.task.id, native)
            assert originals[authorization.task.id] is native
            if authorization.task.id == "task-2" and not first_launched.is_set():
                assert first_launched.wait(10)
        elif (
            event == "return"
            and authorization.task.id == "task-1"
            and type(result) is WorkflowChildReceipt
        ):
            first_receipts.append(result)
            if mode == "child-race":
                os.environ["TLDW_AGENTS_MAX_RUNTIME_SUBAGENTS"] = "1"
            else:
                native.fleet.set_max_live(1)
            first_launched.set()
        elif (
            event == "return"
            and authorization.task.id == "task-2"
            and native.wait_return
        ):
            candidate = native.wait_return[0]
            assert (
                type(candidate) is tuple
                and not authorization._native_launch
                and not authorization._native_dispatch
            )
            save_original(authorization, native, candidate)
            waits.append((native, candidate))
            if not fault:
                threading.setprofile_all_threads(None)
                print(
                    "G_CAPACITY_ORIGINAL_WAIT_PROFILE_REMOVED",
                    time.monotonic(),
                    flush=True,
                )
                second_waited.set()

    def original_capture_returned():
        run = rig.coordinator._runs.get(rig.workflow.id)
        if run is None or retry_waves or (not installed and len(run.preparations) != 2):
            return
        frame = inspect.currentframe().f_back
        while frame is not None:
            if frame.f_code is invoke_code:
                callback = frame.f_locals.get("callback")
                if (
                    type(callback) is types.FunctionType
                    and callback.__code__ is capture_code
                ):
                    wave = dict(
                        zip(
                            callback.__code__.co_freevars,
                            (cell.cell_contents for cell in callback.__closure__),
                        )
                    )["self"]
                    assert type(wave) is _WorkflowWave and wave.phase is run.phase
                    assert all(
                        run.preparations[item.task_id] is item for item in wave.prepared
                    )
                    assert frame.f_locals["self"] is wave.prepared[0]
                    assert wave.source is not None and wave.fence is None
                    if not installed:
                        installed.append(wave)
                        threading.setprofile_all_threads(actual_capacity_race)
                        print(
                            "G_CAPACITY_ORIGINAL_CAPTURE_RETURN",
                            time.monotonic(),
                            flush=True,
                        )
                    else:
                        assert len(saved) == 1
                        original = saved[0]
                        assert wave is not original.first_wave and wave.prepared == (
                            original.prepared,
                        )
                        assert original.prepared.authorization is original.authorization
                        assert (
                            original.authorization._native_preparation[0]
                            is original.native
                        )
                        assert original.prepared.native is original.frame
                        assert original.prepared.configuration is original.configuration
                        assert original.prepared.source is original.source
                        assert original.prepared._lifetime is original.lifetime
                        assert original.prepared._source_scope is original.source_scope
                        assert (
                            original.phase._task_scopes[original.prepared.task_id]
                            is original.authorization
                        )
                        assert (
                            original.authorization._bridge_preparation[0]
                            is original.prepared
                        )
                        assert original.authorization.claim is original.claim
                        assert original.prepared.phase is original.phase is wave.phase
                        assert (
                            original.first_wave.unlaunched[original.prepared]
                            is original.candidate[0]
                        )
                        assert (
                            original.first_wave.closed
                            and original.first_wave.driver.done()
                        )
                        assert original.first_wave.driver.exception() is None
                        retry_waves.append(wave)
                        print(
                            "G_CAPACITY_SAME_PREPARATION_SECOND_CAPTURE",
                            time.monotonic(),
                            flush=True,
                        )
                        unsubscribe()
                    return
            frame = frame.f_back

    unsubscribe = lambda: None
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        planning = rig.coordinator._runs[rig.workflow.id].driver
        await asyncio.shield(planning)
        print("G_CAPACITY_ORIGINAL_PLANNING_RETURN", time.monotonic(), flush=True)
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        unsubscribe = rig.runtime.worktree_activity.subscribe(original_capture_returned)
        await rig.coordinator.approve(decision, command_id="capacity-race")
        while not first_launched.is_set():
            run = rig.coordinator._runs.get(rig.workflow.id)
            if run is None or run.driver.done():
                raise AssertionError("native first capacity launch did not occur")
            await asyncio.sleep(0.01)
        assert installed
        print("G_CAPACITY_WAIT_START", time.monotonic(), flush=True)
        arrived = await asyncio.to_thread(second_waited.wait, 15)
        assert arrived
        if fault:
            original = saved[0]
            run = rig.coordinator._runs[rig.workflow.id]
            await rig.wait_for_provider_entry("task-1")
            rig.provider_release()
            await asyncio.shield(run.driver)
            assert (
                await rig.coordinator.get(rig.workflow.id)
            ).status == "recovery_required"
            wave, prepared, authorization = (
                original.first_wave,
                original.prepared,
                original.authorization,
            )
            assert (
                wave.driver.done() and wave.driver.exception() is original.candidate[0]
            )
            assert wave.unproven_launches == {prepared} and not wave.unlaunched
            assert not wave.closed and not prepared._closed and not retry_waves
            assert wave.baseline_return.result() is ChangeReviewCompletion.OBSERVED
            assert not wave.ending_return.done()
            assert prepared._launch_operation is original.operation is not None
            assert (
                prepared.source is original.source
                and prepared._lifetime is original.lifetime
            )
            assert prepared._source_scope is original.source_scope
            assert (
                prepared.native is original.frame
                and prepared.configuration is original.configuration
            )
            assert (
                prepared.authorization is authorization
                and authorization._bridge_preparation[0] is prepared
            )
            assert authorization._native_preparation[0] is original.native
            assert original.phase._task_scopes[prepared.task_id] is authorization
            assert (
                wave.fence is original.fence
                and rig.runtime.worktree_activity._waves[wave] is original.fence
            )
            assert original.candidate[1].is_admitted
            assert (
                not authorization._native_launch and not authorization._native_dispatch
            )
            if mode == "inner-wait-exit":
                assert original.native.wait_return == [None]
            else:
                assert original.native.wait_return[0] is original.candidate
            snapshot = await rig.coordinator.get(rig.workflow.id)
            assert (
                len(snapshot.results) == 1 and snapshot.results[0].status == "completed"
            )
            assert not snapshot.accounting.uncertain
            assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
            for _ in range(2):
                with pytest.raises(
                    RuntimeError, match="workflow_original_shutdown_unresolved"
                ):
                    await rig.runtime.dispose()
                assert rig.coordinator._runs[rig.workflow.id] is run
                assert (
                    prepared._launch_operation is original.operation and not wave.closed
                )
                assert not rig.runtime.execution_capacity.snapshot().closed
            retained_exit = True
            raise RuntimeError("G_REQUIRED_CUSTODY_RETAINED_SAME_WAIT_EXIT")
        assert len(waits) == 1 and waits[0][1] is not None
        original = saved[0]
        original_second = original.native
        run = rig.coordinator._runs[rig.workflow.id]
        prepared = original.prepared
        assert run.preparations["task-2"] is prepared
        assert prepared.authorization._native_preparation[0] is original_second
        assert not prepared._closed and not prepared.authorization._native_dispatch
        await rig.wait_for_provider_entry("task-1")
        assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
        assert any(
            owner.child
            for owner in rig.runtime.execution_capacity.snapshot().executions
        )
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            windows = conn.execute(
                "SELECT state,baseline FROM workflow_change_windows"
            ).fetchall()
        assert [tuple(row) for row in windows] == [("active", "completed")]
        rig.provider_release()
        complete = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert len(complete.results) == 2 and all(
            result.status == "completed" for result in complete.results
        )
        assert originals["task-2"] is original_second and len(waits) == 1
        assert [key for key, _ in rig.provider.calls].count("task-2") == 1
        assert complete.accounting.used["generation"] == 2
        assert complete.accounting.used["child_launch"] == 2
        assert complete.accounting.used["model_call"] == 3
        assert len(retry_waves) == 1 and len(first_receipts) == 1
        assert prepared.authorization is original.authorization
        assert original.authorization._native_preparation[0] is original_second
        assert original.authorization.claim is original.claim
        receipt = original.authorization._native_launch[0]
        assert type(receipt) is WorkflowChildReceipt
        assert retry_waves[0].receipts == [(prepared, receipt)]
        assert (
            retry_waves[0].receipts[0][0] is prepared
            and retry_waves[0].receipts[0][1] is receipt
        )
        assert original.first_wave.unlaunched[prepared] is original.candidate[0]
        assert not original.first_wave.unproven_launches
        assert installed[0].receipts[0][1] is first_receipts[0]
        assert receipt.attempt_id == original.claim.id
        assert receipt.run_id == original.claim.launch_run_id
        assert receipt.run_id == next(
            result.run_id
            for result in complete.results
            if result.attempt_id == original.claim.id
        )
        for wave in (*installed, *retry_waves):
            assert wave.baseline_return.result() is ChangeReviewCompletion.OBSERVED
            assert wave.ending_return.result() is ChangeReviewCompletion.PUBLISHED
            assert (
                wave.closed and wave.driver.done() and wave.driver.exception() is None
            )
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            windows = conn.execute(
                "SELECT id,ordinal,state,baseline,ending,snapshot_count FROM workflow_change_windows ORDER BY ordinal"
            ).fetchall()
            counts = [
                conn.execute(
                    "SELECT COUNT(*) FROM workflow_change_window_refs WHERE window_id=?",
                    (row[0],),
                ).fetchone()[0]
                for row in windows
            ]
            tasks = conn.execute(
                "SELECT id,run_id,status,retirement FROM workflow_task_attempts ORDER BY ordinal"
            ).fetchall()
        assert len(tasks) == 2 and all(
            tuple(row)[2:] == ("completed", "drained") for row in tasks
        )
        assert any(
            row["id"] == original.claim.id and row["run_id"] == receipt.run_id
            for row in tasks
        )
        assert [tuple(row)[1:] for row in windows] == [
            (ordinal, "completed", "completed", "completed", 0) for ordinal in (1, 2)
        ]
        assert counts == [0, 0]
    finally:
        unsubscribe()
        first_launched.set()
        rig.provider_release()
        threading.setprofile_all_threads(None)
        if not retained_exit:
            await rig.close()


@pytest.mark.parametrize("mode", ["inner-wait-exit", "outer-wait-exit"])
def test_same_wait_object_from_original_scope_exit_never_becomes_pending(
    tmp_path, mode
):
    with pytest.raises(
        AssertionError, match="G_REQUIRED_CUSTODY_RETAINED_SAME_WAIT_EXIT"
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


@pytest.mark.parametrize("mode", ["child-race", "fleet-race"])
def test_actual_capacity_race_retries_same_preparation_after_native_window_return(
    tmp_path, mode
):
    _run(
        tmp_path,
        mode,
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def direct_planner_journey(tmp_path, monkeypatch):
    from tldw_chatbook.Chat.console_chat_controller import (
        capture_run_admitted_workspace_roots,
    )
    from tldw_chatbook.Chat.console_workflow_preparation import (
        prepare_workflow_planning,
    )

    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(), public_coordinator=False
    )
    try:
        assert not hasattr(rig.runtime, "_workflow_coordinator")
        session = rig.store.sessions()[0]
        binding = capture_run_admitted_workspace_roots(
            session=session, registry=rig.service.persistence.workspace_registry
        )[0]
        request = rig.workflow.request
        prepared = await prepare_workflow_planning(
            rig.controller,
            session_id=session.id,
            binding=binding,
            command_id="direct-planning",
            objective=request.objective,
            policy=request.policy,
            _request=request,
        )
        snapshot = await prepared.run_planning()
        assert (
            snapshot.status == "awaiting_approval"
            and snapshot.plans[-1].request == request
        )
        assert prepared._planning_settled and not prepared.primary_owner.is_admitted
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        assert not hasattr(rig.runtime, "_workflow_coordinator")
    finally:
        await rig.close()


def test_direct_native_planner_keeps_lazy_coordinator_absent(tmp_path):
    _run(
        tmp_path,
        "direct-planner",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )
