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
    import dis
    import inspect
    import os
    import sys
    import threading
    import time
    import types

    from tldw_chatbook.Agents.agent_service import (
        AgentService,
        WorkflowChildReceipt,
        _WorkflowLaunchReturn,
    )
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
    original_launch = AgentService._workflow_launch_result.__code__
    original_finish = AgentService._finish_workflow_launch.__code__
    observer_errors = []
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

    def save_original(authorization, native, candidate, returned):
        entry = returned.entry
        prepared = entry.preparation
        assert (
            prepared.authorization is authorization
            and authorization._bridge_preparation[0] is prepared
            and authorization._native_preparation[0] is native is entry.native
            and entry.authorization is authorization
            and entry.wave is prepared._wave is installed[0]
            and entry.mode == "W"
            and not returned.duplicate
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
            returned=returned,
            operation=returned.operation,
            owner=returned.owner,
            fence=installed[0].fence,
        )
        saved.append(original)
        return original

    def record_capacity_race(frame, event, result):
        if event not in {"call", "return"}:
            return None
        normal = event == "return" and dis.opname[
            frame.f_code.co_code[frame.f_lasti]
        ] in {"RETURN_VALUE", "RETURN_CONST"}
        if fault and normal:
            expected_result = (
                inner_code if mode == "inner-wait-exit" else original_launch
            )
            if (
                frame.f_code is expected_result
                and type(result) is _WorkflowLaunchReturn
            ):
                entry = result.entry
                if entry.authorization.task.id == "task-2":
                    assert result.known and result.wait and result.error is not None
                    assert entry.native.wait_return == [None]
                    assert entry.authorization._native_launch == [None]
                    assert result.operation is entry.preparation._launch_operation
                    assert not saved
                    save_original(entry.authorization, entry.native, result, result)
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
                        parent.f_locals.get("entry") is original.returned.entry
                        if mode == "inner-wait-exit"
                        else parent.f_locals.get("item") is original.prepared
                        and parent.f_locals.get("result") is original.returned
                        and frame.f_locals.get("self") is original.prepared
                    )
                    if matched:
                        assert original.native.wait_return == [None]
                        assert original.authorization._native_launch == [None]
                        return original.returned.error
        if frame.f_code is original_launch and event == "call":
            authorization = frame.f_locals["authorization"]
            native = authorization._native_preparation[0]
            assert frame.f_locals["self"] is native.service
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
        elif frame.f_code is original_finish and normal:
            returned = frame.f_locals["result"]
            assert type(returned) is _WorkflowLaunchReturn
            entry = returned.entry
            authorization, native = entry.authorization, entry.native
            if entry.wave is not installed[0]:
                return None
            assert frame.f_locals["self"] is entry.service is native.service
            assert returned.known and not returned.duplicate and result is None
            if authorization.task.id == "task-1":
                assert (
                    returned.error is None
                    and type(returned.receipt) is WorkflowChildReceipt
                )
                assert entry.preparation._launch_operation is None
                first_receipts.append(returned.receipt)
                if mode == "child-race":
                    os.environ["TLDW_AGENTS_MAX_RUNTIME_SUBAGENTS"] = "1"
                else:
                    native.fleet.set_max_live(1)
                first_launched.set()
            elif authorization.task.id == "task-2" and returned.wait:
                candidate = native.wait_return[0]
                assert (
                    type(candidate) is tuple
                    and len(candidate) == 3
                    and candidate[0] is returned.error
                    and candidate[1] is returned.owner
                    and candidate[2] is returned.claim
                    and not authorization._native_launch
                    and not authorization._native_dispatch
                )
                save_original(authorization, native, candidate, returned)
                waits.append((native, candidate))
                threading.setprofile_all_threads(None)
                print(
                    "G_CAPACITY_ORIGINAL_WAIT_PROFILE_REMOVED",
                    time.monotonic(),
                    flush=True,
                )
                second_waited.set()
        return None

    observed_codes = {original_launch, original_finish}
    if fault:
        observed_codes.update((inner_code, core_exit_code, outer_exit_code))

    def actual_capacity_race(frame, event, result):
        if frame.f_code not in observed_codes:
            return
        try:
            injected = record_capacity_race(frame, event, result)
        except BaseException as error:  # noqa: BLE001 - observer failures cannot alter native return.
            observer_errors.append(error)
            threading.setprofile_all_threads(None)
            first_launched.set()
            second_waited.set()
            return
        if injected is not None:
            # Only this deliberate exact original scope-return fault is injected.
            threading.setprofile_all_threads(None)
            second_waited.set()
            print("G_SAME_ORIGINAL_WAIT_EXIT_ERROR", mode, time.monotonic(), flush=True)
            raise injected

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
        if observer_errors:
            raise observer_errors[0]
        assert installed
        print("G_CAPACITY_WAIT_START", time.monotonic(), flush=True)
        arrived = await asyncio.to_thread(second_waited.wait, 15)
        assert arrived
        if observer_errors:
            raise observer_errors[0]
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
                wave.driver.done()
                and wave.driver.exception() is original.returned.error
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
            assert (
                original.owner is original.returned.owner and original.owner.is_admitted
            )
            assert (
                authorization._native_launch == [None]
                and not authorization._native_dispatch
            )
            assert original.native.wait_return == [None]
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


async def foreign_survivor_setup_journey(tmp_path, monkeypatch):
    """One bounded native setup diagnostic, not fleet-event causal evidence."""
    import asyncio
    import os
    import sys
    import threading

    from Tests.Chat.test_console_agent_bridge import _run as original_parent
    from Tests.Chat.workflow_rig import WorkflowFleetProvider
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.execution_capacity import WorkOrigin
    from tldw_chatbook.Agents.fleet_coordinator import FleetCoordinator
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Backup_Recovery import participants, storage_admission
    from tldw_chatbook.Chat.console_chat_store import ConsoleMessageRole
    from tldw_chatbook.Chat.console_workflow_runs import ConsoleWorkflowCoordinator
    from tldw_chatbook.Chat.console_workflow_wave import _WorkflowWave
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB

    os.environ["TLDW_AGENTS_MAX_LIVE_SUBAGENTS"] = "2"
    os.environ["TLDW_AGENTS_RESERVED_MANUAL_SUBAGENTS"] = "0"
    os.environ["TLDW_AGENTS_SUBAGENTS_OUTLIVE_TURN"] = "true"
    causal = sys.argv[2] == "event"
    provider = WorkflowFleetProvider()
    rig = await build_workflow_rig(tmp_path, monkeypatch, provider, change_review=True)
    parent = None
    saved = {}
    parent_registrations = []
    parent_close_returns = []
    finish_entered = (threading.Event(), threading.Event())
    finish_release = (threading.Event(), threading.Event())
    child_body_returns, finish_returns, notifications = [], [], []
    event_transitions, wait_returns, selections, waves = [], [], [], []

    def finish_observation():
        if (
            "open_return" not in saved
            or len(thread_returns) != 2
            or len(target_returns) != 2
            or len(settled_returns) != 2
            or not registrations
            or not all(
                any(
                    close[0] is record and close[1] and close[2]
                    for close in close_returns
                )
                for record in registrations
            )
        ):
            return
        assert len(event_transitions) == 1 and wait_returns == [saved["wake"]]
        saved["profile_disposed"] = True
        threading.setprofile_all_threads(None)

    def unchanged_preparation():
        prepared, auth, native, frame, configuration, source, lifetime, scope = saved[
            "workflow"
        ]
        assert prepared.authorization is auth
        assert auth._native_preparation[0] is native
        assert prepared.native is frame and prepared.configuration is configuration
        assert prepared.source is source and prepared._lifetime is lifetime
        assert prepared._source_scope is scope
        assert prepared.phase is saved["phase"]
        assert auth.claim is saved["claim"]
        assert auth._bridge_preparation[0] is prepared
        assert saved["phase"]._task_scopes[prepared.task_id] is auth
        return prepared, auth, native

    def event_observe(frame, event, result):
        wake = saved.get("wake")
        if wake is None:
            return
        code = frame.f_code
        if (
            event == "return"
            and code is asyncio.BaseEventLoop.call_soon_threadsafe.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is saved["released"].__code__
            and frame.f_back.f_locals.get("run") is saved["run"]
        ):
            assert result._callback.__self__ is wake
            assert result._callback.__func__ is asyncio.Event.set
            caller = frame.f_back
            kind = "other"
            while caller is not None:
                if (
                    caller.f_code
                    is type(
                        rig.controller._agent_bridge
                    )._notify_fleet_consumers.__code__
                ):
                    kind = "fleet"
                    break
                caller = caller.f_back
            notifications.append((result, kind, threading.current_thread()))
        elif code is asyncio.Event.set.__code__ and frame.f_locals.get("self") is wake:
            waiter = saved.get("waiter")
            if waiter is None:
                return
            if event == "call":
                caller = frame.f_back
                handle = (
                    caller.f_locals.get("self")
                    if caller is not None
                    and caller.f_code is asyncio.Handle._run.__code__
                    else None
                )
                saved["setting"] = (handle, not waiter.done())
            elif event == "return":
                handle, pending = saved.pop("setting")
                if pending and waiter.done():
                    event_transitions.append(handle)
        elif (
            event == "return"
            and code is asyncio.Event.wait.__code__
            and frame.f_locals.get("self") is wake
            and result is True
            and saved.get("waiter") is not None
        ):
            assert saved["run"].wake is wake
            wait_returns.append(wake)
        elif (
            event == "return"
            and code is ConsoleWorkflowCoordinator._wave_ready.__code__
            and frame.f_locals.get("run") is saved.get("run")
            and result
        ):
            prepared, _auth, native = unchanged_preparation()
            assert result == [prepared]
            with native.fleet._lock:
                assert child_threads[1][0] in native.fleet._live_ids
            selections.append(prepared)
        elif event == "return" and code is _WorkflowWave.__init__.__code__:
            wave = frame.f_locals["self"]
            if wave.phase is saved.get("phase"):
                prepared, _auth, _native = unchanged_preparation()
                assert wave.prepared == (prepared,)
                waves.append(wave)
        elif (
            event == "return"
            and code is _WorkflowWave._open.__code__
            and result is None
        ):
            from tldw_chatbook.Workspaces.change_review_finalization import (
                ChangeReviewCompletion,
            )

            wave = frame.f_locals["self"]
            if wave.phase is saved.get("phase"):
                assert waves == [wave]
                assert wave.baseline_return.done()
                assert wave.baseline_return.result() is ChangeReviewCompletion.OBSERVED
                assert wave.source is not None and wave.reservation is not None
                saved["open_return"] = (
                    wave,
                    wave.driver,
                    wave.source,
                    wave.reservation,
                    wave.baseline_return,
                    wave.ending_return,
                )
                finish_observation()

    observed = threading.Event()
    finished = threading.Event()
    child_threads = ()
    registrations, close_returns, settled_returns, target_returns, thread_returns = (
        [],
        [],
        [],
        [],
        [],
    )
    database = rig.controller.app.chachanotes_db

    def child_observe(frame, event, result):
        if causal:
            event_observe(frame, event, result)
        thread = threading.current_thread()
        original = next((row for row in child_threads if row[1] is thread), None)
        if original is None:
            return
        handle_id, _thread, handle, target, owner = original
        if causal and frame.f_code is FleetCoordinator.finish.__code__:
            if (
                frame.f_locals.get("self") is saved["workflow"][2].fleet
                and frame.f_locals.get("handle_id") == handle_id
            ):
                if event == "call":
                    assert any(
                        row[0] is thread and row[1][0] == handle.run_id
                        for row in child_body_returns
                    )
                    index = next(
                        i for i, row in enumerate(child_threads) if row is original
                    )
                    finish_entered[index].set()
                    assert finish_release[index].wait(10), (
                        "original finish hold expired"
                    )
                elif event == "return":
                    finish_returns.append(original)
            return
        if event != "return":
            return
        if causal and frame.f_code is AgentService._run_owned.__code__:
            assert frame.f_locals.get("execution_owner") is owner
            assert result[0] == handle.run_id and result[1].status == "done"
            child_body_returns.append((thread, result))
        elif (
            frame.f_code is participants._register_core_connection.__code__
            and frame.f_locals.get("repository") is database
        ):
            connection = frame.f_locals["connection"]
            participant, lease = frame.f_locals["participant"], frame.f_locals["lease"]
            assert result is connection and lease.resource_thread is thread
            with storage_admission._lock:
                assert participant.connections.get(connection) is lease
                assert lease in storage_admission._live_leases
            stack = []
            caller = frame.f_back
            while caller is not None:
                stack.append(
                    (caller.f_code.co_filename, caller.f_code.co_name, caller.f_lineno)
                )
                caller = caller.f_back
            registrations.append((thread, connection, participant, lease, stack))
            print("G_FOREIGN_ORIGINAL_CONNECTION", handle_id, stack, flush=True)
        elif (
            frame.f_code is CharactersRAGDB._close_connection_handle.__code__
            and frame.f_locals.get("self") is database
        ):
            connection = frame.f_locals["conn"]
            for record in registrations:
                if record[0] is thread and record[1] is connection:
                    with storage_admission._lock:
                        close_returns.append(
                            (
                                record,
                                connection not in record[2].connections,
                                record[3] not in storage_admission._live_leases,
                            )
                        )
        elif (
            frame.f_code
            is type(rig.controller._agent_bridge)._on_fleet_child_settled.__code__
            and frame.f_locals.get("run_id") == handle.run_id
        ):
            settled_returns.append((thread, handle))
        elif frame.f_code is target.__code__:
            assert frame.f_locals.get("child_owner") is owner
            target_returns.append(thread)
        elif frame.f_code is threading.Thread.run.__code__:
            thread_returns.append(thread)
            sys.setprofile(None)
            if causal:
                finish_observation()
            elif len(thread_returns) == len(child_threads):
                threading.setprofile_all_threads(None)

    try:
        session = rig.store._sessions["workflow-test-chat"]
        resolution = await rig.gateway.resolve_for_send(
            rig.controller._provider_selection_for_session(session.id)
        )
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await asyncio.shield(rig.coordinator._runs[rig.workflow.id].driver)
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )

        def worker_observe(frame, event, result):
            if event != "return":
                return
            if (
                frame.f_code is participants._register_core_connection.__code__
                and type(frame.f_locals.get("repository")) is WorkspaceDB
                and threading.current_thread() is parent
            ):
                repository = frame.f_locals["repository"]
                connection = frame.f_locals["connection"]
                participant, lease = (
                    frame.f_locals["participant"],
                    frame.f_locals["lease"],
                )
                stack = []
                caller = frame.f_back
                while caller is not None:
                    stack.append(
                        (
                            caller.f_code.co_filename,
                            caller.f_code.co_name,
                            caller.f_lineno,
                        )
                    )
                    caller = caller.f_back
                parent_registrations.append(
                    (
                        repository,
                        connection,
                        participant,
                        lease,
                        parent,
                        result is connection,
                        stack,
                    )
                )
            elif (
                frame.f_code is WorkspaceDB.close.__code__
                and threading.current_thread() is parent
            ):
                repository = frame.f_locals.get("self")
                with storage_admission._lock:
                    for record in parent_registrations:
                        if record[0] is repository and record[4] is parent:
                            parent_close_returns.append(
                                (
                                    record,
                                    record[1] not in record[2].connections,
                                    record[3] not in storage_admission._live_leases,
                                )
                            )
            elif frame.f_code is AgentService.__init__.__code__:
                service = frame.f_locals["self"]
                saved["service"] = service
                saved["context"] = service._automatic_work
                saved["chain"] = service._work_chain_id
            elif frame.f_code is AgentService._run_owned.__code__:
                if frame.f_locals["self"] is saved.get("service"):
                    saved["owner"] = frame.f_locals.get("owner")
                    saved["original_result"] = result

        def ready(frame, event, _result):
            nonlocal parent, child_threads
            if (
                event != "call"
                or frame.f_code is not ConsoleWorkflowCoordinator._wave_ready.__code__
                or not frame.f_locals["candidates"]
            ):
                return
            sys.setprofile(None)
            run = frame.f_locals["run"]
            prepared = frame.f_locals["candidates"][0]
            auth = prepared.authorization
            native = auth._native_preparation[0]
            snapshot = (
                prepared,
                auth,
                native,
                prepared.native,
                prepared.configuration,
                prepared.source,
                prepared._lifetime,
                prepared._source_scope,
            )
            saved["workflow"] = snapshot
            saved.update(run=run, phase=run.phase, claim=auth.claim, wake=run.wake)
            if causal:
                with rig.controller._agent_bridge._fleet_activity_lock:
                    callbacks = tuple(
                        rig.controller._agent_bridge._fleet_activity_consumers.values()
                    )
                original_callbacks = []
                for callback in callbacks:
                    captured = dict(
                        zip(
                            callback.__code__.co_freevars,
                            (cell.cell_contents for cell in callback.__closure__ or ()),
                        )
                    )
                    if (
                        captured.get("run") is run
                        and captured.get("self") is rig.coordinator
                    ):
                        original_callbacks.append(callback)
                assert len(original_callbacks) == 1
                saved["released"] = original_callbacks[0]
            inbox = native.fleet.message_inbox
            assert not run.waves and prepared._finite is None
            for read in prepared._finite_reads:
                prepared._finite_read_return(read)
            assistant = rig.store.append_message(
                session.id, role=ConsoleMessageRole.ASSISTANT, content=""
            )

            def drive_parent():
                sys.setprofile(worker_observe)
                try:
                    saved["outcome"] = participants.run_finite_local_worker(
                        original_parent,
                        rig.controller._agent_bridge,
                        rig.store,
                        session,
                        assistant.id,
                        conversation_id=session.persisted_conversation_id,
                        resolution=resolution,
                        expected_progress_owner_id=prepared.native.progress_owner_id,
                        agent_messages=[
                            {"role": "user", "content": provider.parent_instruction}
                        ],
                    )
                except BaseException as error:  # noqa: BLE001 - retain the original diagnostic worker failure unchanged.
                    saved["error"] = error
                finally:
                    sys.setprofile(None)
                    finished.set()

            parent = threading.Thread(
                target=drive_parent, name="original-foreign-parent"
            )
            parent.start()
            try:
                if not finished.wait(10):
                    stack = []
                    current = sys._current_frames().get(parent.ident)
                    while current is not None:
                        stack.append(
                            (
                                current.f_code.co_filename,
                                current.f_code.co_name,
                                current.f_lineno,
                            )
                        )
                        current = current.f_back
                    print("G_FOREIGN_ORIGINAL_PARENT_STACK", stack, flush=True)
                    raise AssertionError(
                        "original parent did not return while host observed setup"
                    )
                parent.join()
                if "error" in saved:
                    print(
                        "G_FOREIGN_ORIGINAL_PARENT_ERROR",
                        type(saved["error"]).__name__,
                        str(saved["error"]),
                        flush=True,
                    )
                    raise saved["error"]
                assert parent_registrations
                with storage_admission._lock:
                    for record in parent_registrations:
                        assert record[4] is parent and record[5]
                        assert record[2].repository() is record[0]
                        assert record[1] not in record[2].connections
                        assert record[3] not in storage_admission._live_leases
                        assert any(
                            close[0] is record and close[1] and close[2]
                            for close in parent_close_returns
                        )
                assert all(event.wait(1) for event in provider.child_entered.values())
                assert saved["outcome"].status == "done"
                service = saved["service"]
                assert saved["context"] is None and service._automatic_work is None
                assert service.work_origin is WorkOrigin.MANUAL
                assert (
                    service._work_chain_id
                    == saved["chain"]
                    != run.phase.attempt.chain_id
                )
                assert service._fleet is native.fleet
                assert service._fleet.message_inbox is inbox
                assert (
                    rig.controller._agent_bridge._fleet_coordinators[
                        session.persisted_conversation_id
                    ]
                    is native.fleet
                )
                with native.fleet._lock:
                    assert (
                        len(native.fleet._live_ids) == 2 and native.fleet.max_live == 2
                    )
                owner = saved["owner"]
                drained = []
                owner.on_drained(drained.append)
                assert drained == [True] and not owner.is_admitted
                assert all(
                    thread.is_alive() for thread in service._fleet_threads.values()
                )
                assert len(service._fleet_threads) == 2
                retained_threads = []
                for handle_id, thread in tuple(service._fleet_threads.items()):
                    handle = native.fleet.get(handle_id)
                    target = thread._target
                    closed = dict(
                        zip(
                            target.__code__.co_freevars,
                            (cell.cell_contents for cell in target.__closure__),
                        )
                    )
                    child_owner = closed["child_owner"]
                    assert (
                        child_owner.is_admitted and child_owner.run_id == handle.run_id
                    )
                    retained_threads.append(
                        (handle_id, thread, handle, target, child_owner)
                    )
                child_threads = tuple(retained_threads)
                with (
                    owned_goal_storage(rig.service),
                    rig.service.db.connection() as conn,
                ):
                    rows = conn.execute(
                        "SELECT work_chain_id,agent_kind,status FROM agent_runs WHERE parent_run_id=?",
                        (saved["original_result"][0],),
                    ).fetchall()
                assert len(rows) == 2 and all(row[0] == saved["chain"] for row in rows)
                assert all(row[2] == "running" for row in rows)
                assert all(
                    first is second
                    for first, second in zip(
                        snapshot,
                        (
                            prepared,
                            prepared.authorization,
                            auth._native_preparation[0],
                            prepared.native,
                            prepared.configuration,
                            prepared.source,
                            prepared._lifetime,
                            prepared._source_scope,
                        ),
                    )
                )
                assert auth._bridge_preparation[0] is prepared
                assert run.phase._task_scopes["task-1"] is auth
                assert not auth._native_launch and not run.waves
                print("G_FOREIGN_ORIGINAL_SETUP_PROVEN", flush=True)
                observed.set()
            finally:
                if observed.is_set():
                    threading.setprofile_all_threads(child_observe)
                for gate in provider.child_release.values():
                    gate.set()

        sys.setprofile(ready)
        await rig.coordinator.approve(decision, command_id="foreign-setup")
        run = rig.coordinator._runs[rig.workflow.id]
        if causal:
            async with asyncio.timeout(10):
                while not (
                    all(gate.is_set() for gate in finish_entered)
                    and run.wake._waiters
                    and not run.wake._waiters[0].done()
                ):
                    assert not run.driver.done(), "original G wait never armed"
                    await asyncio.sleep(0)
            waiter = run.wake._waiters[0]
            saved["waiter"] = waiter
            for _ in range(3):
                await asyncio.sleep(0)
                assert run.wake is saved["wake"] and not waiter.done()
                assert tuple(run.wake._waiters) == (waiter,)
            prepared, authorization, native = unchanged_preparation()
            assert not run.waves and not waves and not authorization._native_launch
            quiescence = run.phase._quiescence
            assert quiescence is not None and quiescence.mode == "idle"
            assert (
                quiescence.proof[1].done() and quiescence.proof[1].exception() is None
            )
            assert quiescence.proof[2].done() and quiescence.proof[2].result() is None
            assert [key for key, _ in rig.provider.calls] == ["planning"]
            with native.fleet._lock:
                assert native.fleet._live_ids == {row[0] for row in child_threads}
            # Earlier capacity notifications have completed before this exact arm.
            assert all(
                row[0] not in rig.coordinator.loop._ready for row in notifications
            )
            finish_release[0].set()
            async with asyncio.timeout(10):
                while not selections:
                    assert not run.driver.done(), (
                        "original event did not select saved preparation"
                    )
                    await asyncio.sleep(0)
            assert wait_returns == [saved["wake"]]
            assert len(event_transitions) == 1
            cause = next(row for row in notifications if row[0] is event_transitions[0])
            assert cause[1] == "fleet" and cause[2] is child_threads[0][1]
            assert child_threads[0] in finish_returns
            assert child_threads[1] not in finish_returns
            assert selections[0] is prepared
            finish_release[1].set()
            print("G_ORIGINAL_FLEET_EVENT_CAUSE_PROVEN", flush=True)
        await asyncio.shield(run.driver)
        assert observed.is_set(), "original foreign survivor setup not established"
        assert (
            await rig.coordinator.get(rig.workflow.id)
        ).status == "awaiting_result_review"
        if causal:
            from tldw_chatbook.Workspaces.change_review_finalization import (
                ChangeReviewCompletion,
            )

            assert len(waves) == 1 and len(run.waves) == 1
            wave = waves[0]
            assert saved["profile_disposed"]
            assert all(
                before is after
                for before, after in zip(
                    saved["open_return"],
                    (
                        wave,
                        wave.driver,
                        wave.source,
                        wave.reservation,
                        wave.baseline_return,
                        wave.ending_return,
                    ),
                )
            )
            assert run.waves[0][0] is wave
            assert (
                wave.closed and wave.driver.done() and wave.driver.exception() is None
            )
            assert wave.baseline_return.result() is ChangeReviewCompletion.OBSERVED
            assert wave.ending_return.result() is ChangeReviewCompletion.PUBLISHED
            prepared, authorization, *_ = saved["workflow"]
            receipt = authorization._native_launch[0]
            assert wave.receipts == [(prepared, receipt)]
            assert receipt.attempt_id == saved["claim"].id
            assert receipt.run_id == saved["claim"].launch_run_id
            complete = await rig.coordinator.get(rig.workflow.id)
            assert len(complete.results) == 1
            assert complete.results[0].attempt_id == receipt.attempt_id
            assert complete.results[0].run_id == receipt.run_id
            assert complete.results[0].status == "completed"
            assert complete.accounting.used["generation"] == 2
            assert complete.accounting.used["child_launch"] == 1
            assert complete.accounting.used["model_call"] == 2
            assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
            with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
                windows = conn.execute(
                    "SELECT id,state,baseline,ending,snapshot_count FROM workflow_change_windows WHERE workflow_id=?",
                    (rig.workflow.id,),
                ).fetchall()
                assert len(windows) == 1
                assert tuple(windows[0])[1:] == (
                    "completed",
                    "completed",
                    "completed",
                    0,
                )
                assert (
                    conn.execute(
                        "SELECT count(*) FROM workflow_change_window_refs WHERE window_id=?",
                        (windows[0][0],),
                    ).fetchone()[0]
                    == 0
                )
        for _handle_id, thread, _handle, _target, owner in child_threads:
            await asyncio.to_thread(thread.join, 5)
            assert not thread.is_alive()
            assert thread in target_returns and thread in thread_returns
            drained = []
            owner.on_drained(drained.append)
            assert drained == [True] and not owner.is_admitted
        assert len(child_threads) == 2
        assert len(settled_returns) == 2
        with storage_admission._lock:
            current = [
                {
                    "same_child": any(thread is row[1] for row in child_threads),
                    "same_connection_retained": participant.connections.get(connection)
                    is lease,
                    "same_lease_live": lease in storage_admission._live_leases,
                    "same_thread_dead": not thread.is_alive(),
                    "same_close_returned": any(
                        row[0][1] is connection and row[1] and row[2]
                        for row in close_returns
                    ),
                }
                for thread, connection, participant, lease, _stack in registrations
            ]
        print("G_FOREIGN_ORIGINAL_CONNECTION_TAIL", current, flush=True)
        assert current and all(
            row["same_child"]
            and row["same_thread_dead"]
            and row["same_close_returned"]
            and not row["same_connection_retained"]
            and not row["same_lease_live"]
            for row in current
        )
    finally:
        sys.setprofile(None)
        threading.setprofile_all_threads(None)
        for gate in finish_release:
            gate.set()
        for gate in provider.child_release.values():
            gate.set()
        if parent is not None:
            await asyncio.to_thread(parent.join, 5)
            assert not parent.is_alive(), "original parent retained"
        try:
            await rig.close()
        except AssertionError as original_failure:
            trace = original_failure.__traceback__
            while trace is not None and not (
                trace.tb_frame.f_code is type(rig.creators).retire.__code__
                and trace.tb_frame.f_locals.get("self") is rig.creators
            ):
                trace = trace.tb_next
            if trace is not None:
                path = trace.tb_frame.f_locals.get("path")
                original_participant = trace.tb_frame.f_locals.get("participant")
                with storage_admission._lock:
                    leases = tuple(
                        lease
                        for lease in storage_admission._live_leases
                        if lease.resource_path == path
                    )
                    facts = []
                    for lease in leases:
                        matches = [
                            row for row in parent_registrations if row[3] is lease
                        ]
                        facts.append(
                            {
                                "same_parent_thread": lease.resource_thread is parent,
                                "same_saved_child": any(
                                    lease.resource_thread is row[1]
                                    for row in child_threads
                                ),
                                "same_creator_participant": getattr(
                                    lease, "resource_participant", None
                                )
                                is original_participant,
                                "parent_returned": parent is not None
                                and not parent.is_alive(),
                                "original_registrations": [
                                    {
                                        "actual_return": row[5],
                                        "same_repository": row[2].repository()
                                        is row[0],
                                        "same_connection_lease": row[2].connections.get(
                                            row[1]
                                        )
                                        is row[3],
                                        "same_parent": row[4] is parent,
                                        "native_owner": row[2].owner_id,
                                        "caller": row[6],
                                    }
                                    for row in matches
                                ],
                            }
                        )
                print("G_ORIGINAL_CREATOR_REMAINING_PATH", str(path), facts, flush=True)
            raise


def test_real_foreign_survivor_setup_uses_original_native_sources(tmp_path):
    _run(
        tmp_path,
        "fleet-setup",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_actual_foreign_finish_notifies_same_armed_workflow_event(tmp_path):
    _run(
        tmp_path,
        "fleet-setup",
        "event",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def mixed_wake_journey(tmp_path, monkeypatch):
    """Real enabled exclusion, then unrelated delivery after positive G return."""
    import asyncio
    import contextvars
    import dis
    import inspect
    import json
    import os
    import sys
    import threading
    import time
    import types

    from Tests.Chat.test_console_agent_bridge import _run as original_parent
    from Tests.Chat.workflow_rig import WorkflowFleetProvider
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation, WorkOrigin
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Backup_Recovery import participants, storage_admission
    from tldw_chatbook.Chat.console_chat_models import ConsoleSubmissionOrigin
    from tldw_chatbook.Chat.console_chat_store import ConsoleMessageRole
    from tldw_chatbook.Chat.console_fleet_wake import (
        ConsoleFleetWakeCoordinator,
        autowake_enabled,
    )

    os.environ["TLDW_AGENTS_MAX_LIVE_SUBAGENTS"] = "2"
    os.environ["TLDW_AGENTS_RESERVED_MANUAL_SUBAGENTS"] = "0"
    os.environ["TLDW_AGENTS_SUBAGENTS_OUTLIVE_TURN"] = "true"
    provider = WorkflowFleetProvider()
    notification_failure = sys.argv[2] in {
        "queued-notify-error",
        "queued-notify-cancel",
    }
    notification_cancel = sys.argv[2] == "queued-notify-cancel"
    permanent_close = sys.argv[2] == "queued-close"
    queued = sys.argv[2] == "queued" or notification_failure or permanent_close
    trace_context = queued or sys.argv[2] in {"context", "owner-census"}
    notification_error = ValueError("original Q1 scheduling entry failed")
    notification_entry = []
    loop = asyncio.get_running_loop()
    factory_calls = []
    original_factory = loop.get_task_factory()

    def original_task_factory(actual_loop, coroutine, *, context=None):
        task = asyncio.Task(coroutine, loop=actual_loop, context=context)
        if coroutine.cr_code is ConsoleFleetWakeCoordinator._deliver.__code__:
            factory_calls.append((coroutine, task, context, tuple(context.items())))
        return task

    if trace_context:
        assert original_factory is None
        loop.set_task_factory(original_task_factory)
    try:
        rig = await build_workflow_rig(
            tmp_path,
            monkeypatch,
            provider,
            change_review=True,
            native_fleet_consumers=True,
        )
    except BaseException:
        if trace_context:
            loop.set_task_factory(original_factory)
        raise
    wake, bridge = rig.controller.fleet_wake, rig.controller._agent_bridge
    events, submissions, controller_entries = [], [], []
    custody_sources, custody_returns = {}, []
    parent, children = None, []
    observer_errors, observer_checks = [], []
    observer_labels = []
    tree_records, snapshot_records = {}, []
    child_target_returns = []
    tree_returns, ticket_returns, operation_returns, claim_returns = [], [], [], []
    primary_error = None
    approval = None
    close_task = None
    retry_entered, retry_release, retry_posted = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    shutdown = {"order": [], "attempts": [], "schedules_after_fence": []}
    queued_entered, queued_release = threading.Event(), threading.Event()
    queued_state = {
        "waves": [],
        "refusals": [],
        "blocked": [],
        "timers": [],
        "release": [],
    }
    saved = {}
    provenance = {
        "retry": [],
        "handles": [],
        "timers": [],
        "schedule": [],
        "entry": [],
        "prepare": [],
    }
    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Agents.automatic_work_runtime import (
        _CURRENT as automatic_context,
    )
    from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner
    from tldw_chatbook.Agents.workflow_task_scope import current_task_authorization
    from tldw_chatbook.Backup_Recovery.admission_runtime import execution_identity
    from tldw_chatbook.Chat import console_preparation_reads as native_reads
    from tldw_chatbook.Chat.console_agent_sources import _AgentSources
    from tldw_chatbook.Chat.console_ordinary_tree import _OrdinaryTurn
    from tldw_chatbook.Workspaces.change_review_finalization import (
        ChangeReviewFinalizationCoordinator,
    )
    from tldw_chatbook.Workspaces.change_review_ownership import (
        _SnapshotSource,
        _SnapshotTicket,
    )
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivitySettlement

    def normal_return(frame, event):
        return event == "return" and dis.opname[
            frame.f_code.co_code[frame.f_lasti]
        ] in {"RETURN_VALUE", "RETURN_CONST"}

    from tldw_chatbook.Chat.console_workflow_preparation import (
        _PreparedWorkflowReply,
        current_workflow_preparation,
    )

    observed_codes = frozenset(
        function.__code__
        for function in (
            AgentService.__init__,
            AgentService._run_owned,
            threading.Thread.start,
            type(wake).retry_soon,
            type(wake)._deliver,
            type(loop).call_soon_threadsafe,
            type(loop).call_later,
            type(loop).create_task,
            _AgentSources.prepare,
            type(rig.runtime)._submit_fleet_wake,
            type(rig.runtime)._release_custody,
            type(rig.controller)._submit_draft_lifecycle,
            _OrdinaryTurn.admit,
            _OrdinaryTurn._retire_scratch,
            _SnapshotSource.prepare_ordinary,
            _SnapshotSource.prepare_survivor,
            ChangeReviewFinalizationCoordinator.register,
            _SnapshotTicket.returned,
            OwnedOperation.finish,
            WorktreeActivitySettlement.settle,
        )
    )
    child_target_codes = set()
    from tldw_chatbook.Chat import console_workflow_wave as native_waves
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.store_kernel import SQLiteStoreKernel

    original_close = SQLiteStoreKernel.close
    metadata_closes = []

    execute_code = type(rig.coordinator)._execute.__code__
    notify_release_code = type(rig.coordinator)._notify_execution_release.__code__
    configure_code = _PreparedWorkflowReply._configure.__code__
    execution_read_code = next(
        code
        for code in _PreparedWorkflowReply._read.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    workflow_source_reads = []
    wave_return_code = native_waves._workflow_wave_return.__code__
    metadata_code = next(
        code
        for code in type(rig.coordinator)._metadata.__wrapped__.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "read_metadata"
    )
    retry_code = next(
        code
        for code in type(wake)._schedule_delayed_retry.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "retry"
    )
    if permanent_close:
        observed_codes |= {
            type(wake)._attempt_all.__code__,
            type(wake).dispose.__code__,
            type(rig.runtime).dispose.__code__,
            type(rig.runtime)._dispose_owned.__code__,
            type(rig.coordinator).close.__code__,
            type(rig.controller).shutdown.__code__,
        }
    if queued:
        observed_codes |= {
            execute_code,
            wave_return_code,
            metadata_code,
            original_close.__code__,
            retry_code,
            type(wake)._priority_allows.__code__,
            type(wake).try_claim_automatic_primary.__code__,
            type(wake)._attempt.__code__,
            type(rig.coordinator)._release.__code__,
        }

    def queued_observe(frame, event, result):
        code, local = frame.f_code, frame.f_locals
        if (
            event == "call"
            and code is execute_code
            and local.get("self") is rig.coordinator
            and "run" not in queued_state
        ):
            actual = local["run"]
            record_checks(
                "queued_execute",
                (
                    rig.coordinator._runs.get(actual.workflow_id) is actual
                    and actual.driver is asyncio.current_task(),
                ),
            )
            queued_state.update(
                run=actual,
                driver=asyncio.current_task(),
                phase=actual.phase,
            )
        if not normal_return(frame, event) or "run" not in queued_state:
            return
        actual = queued_state["run"]
        if code is original_close.__code__ and local.get("self") is rig.service.db:
            ancestor = frame.f_back
            while ancestor is not None and ancestor.f_code is not metadata_code:
                ancestor = ancestor.f_back
            if ancestor is not None:
                metadata = ancestor.f_locals
                cell = metadata["cell"]
                database, connection = local["self"], local["conn"]
                binding = native_reads._source_binding.get()
                if (
                    cell is actual.final_metadata
                    and cell.proof is not None
                    and type(database) is AgentRunsDB
                    and not database.is_memory_db
                    and inspect.getattr_static(database, "close") is original_close
                    and metadata["database"] is database
                    and metadata["conn"] is connection
                    and connection is not None
                    and getattr(database._thread_local, "conn", None) is None
                    and binding.active
                    and binding.thread is threading.current_thread()
                    and binding.read is cell.proof[0]
                ):
                    metadata_closes.append(
                        (
                            cell,
                            database,
                            connection,
                            threading.current_thread(),
                            binding.read,
                        )
                    )
        elif code is wave_return_code:
            caller = frame.f_back
            if (
                caller is not None
                and caller.f_code is execute_code
                and caller.f_locals.get("self") is rig.coordinator
                and caller.f_locals.get("run") is actual
                and local["wave"].phase is queued_state["phase"]
                and asyncio.current_task() is queued_state["driver"]
                and actual.driver is queued_state["driver"]
            ):
                queued_state["waves"].append(result)
        elif code is metadata_code and local["cell"] is actual.final_metadata:
            cell, proof = local["cell"], local["cell"].proof
            read, producer, retired, callback = proof
            invocation = inspect.getclosurevars(callback).nonlocals
            guarded = invocation["guarded_read"]
            originals = inspect.getclosurevars(local["original_sources"]).nonlocals
            binding = native_reads._source_binding.get()
            identities = (
                invocation["self"] is rig.coordinator,
                originals["self"] is rig.coordinator,
                invocation["cell"] is cell is actual.final_metadata,
                invocation["controller"] is rig.controller,
                local["service"] is rig.service,
                local["database"] is rig.service.db,
                invocation["persistence"] is rig.service.persistence,
                invocation["registry"] is rig.service.persistence.workspace_registry,
                cell.run is actual and cell.phase is actual.phase,
                cell.issuing is actual.driver is invocation["issuing"],
                read is binding.read and read.task is actual.driver,
                read.source is rig.coordinator and read.creator is rig.controller,
                read.callback is callback
                and read._producer is producer
                and read.retired is retired,
                guarded.__wrapped__.__code__ is code,
                inspect.getclosurevars(guarded.__wrapped__).nonlocals["result"]
                is local["result"],
                result is None
                and any(
                    saved_cell is cell
                    and saved_db is local["database"]
                    and saved_conn is local["conn"]
                    and saved_thread is threading.current_thread()
                    and saved_read is read
                    for saved_cell, saved_db, saved_conn, saved_thread, saved_read in metadata_closes
                ),
                local["result"][0] is None,
                local["result"][1].status == "awaiting_result_review",
                bool(queued_state["waves"]),
                not cell.disposed.done() and not producer.done(),
            )
            queued_state["metadata_checks"] = tuple(
                zip(
                    (
                        "callback_coordinator",
                        "source_coordinator",
                        "same_cell",
                        "controller",
                        "service",
                        "database",
                        "persistence",
                        "registry",
                        "run_phase",
                        "original_issuer",
                        "read_binding",
                        "read_source",
                        "read_callback_producer_retired",
                        "callback_code",
                        "actual_result_closure",
                        "same_original_connection_close_return",
                        "no_history",
                        "actual_review_status",
                        "original_wave_return",
                        "producer_and_disposition_pending",
                    ),
                    identities,
                    strict=True,
                )
            )
            record_checks("metadata", identities)
            if not all(identities):
                return
            queued_state.update(cell=cell, proof=proof, view=local["result"][1])
            queued_state["hold_deadline"] = time.monotonic() + 10
            queued_entered.set()
            if not queued_release.wait(10):
                observer_errors.append(
                    TimeoutError("original queued metadata hold10 expired")
                )
        elif code is type(wake)._priority_allows.__code__ and local.get("self") is wake:
            queued_state.setdefault("priority", []).append((result, frame.f_lineno))
        elif (
            code is type(wake).try_claim_automatic_primary.__code__
            and local.get("self") is wake
        ):
            caller = frame.f_back
            if (
                result is False
                and local.get("existing") is actual.token
                and caller is not None
                and caller.f_code is type(wake)._attempt.__code__
                and caller.f_locals.get("self") is wake
                and caller.f_locals["delivery"].slot_token is local["token"]
            ):
                queued_state["refusals"].append((caller, actual.token, local["token"]))
        elif code is retry_code and local.get("self") is wake:
            ancestor = frame.f_back
            if ancestor is not None and ancestor.f_code is asyncio.Handle._run.__code__:
                handle = ancestor.f_locals["self"]
                if any(row[0] is handle for row in provenance["timers"]):
                    queued_state["timers"].append(handle)
        elif code is type(wake)._attempt.__code__ and local.get("self") is wake:
            delivery, task, coroutine = (
                local.get("delivery"),
                local.get("task"),
                local.get("coroutine"),
            )
            if task is not None:
                original_frame = coroutine.cr_frame
                record_checks(
                    "delivery_registration",
                    (
                        wake._active.get(conversation) is delivery,
                        wake._delivery_tasks.get(task) == conversation,
                        wake._automatic_primary_claims.get(session.id)
                        is delivery.slot_token,
                        delivery.session_id == session.id,
                        task.get_coro() is coroutine,
                        coroutine.cr_code is type(wake)._deliver.__code__,
                        original_frame.f_locals.get("self") is wake,
                        original_frame.f_locals.get("conversation_id") == conversation,
                        original_frame.f_locals.get("session_id") == session.id,
                        original_frame.f_locals.get("slot_token")
                        is delivery.slot_token,
                    ),
                )
                queued_state.setdefault("deliveries", []).append(
                    (delivery, task, coroutine, delivery.slot_token)
                )
            if any(
                row[0] is frame for row in queued_state["refusals"]
            ) and wake._registry_lock.acquire(blocking=False):
                try:
                    queued_state["blocked"].append(
                        (
                            tuple(wake._pending.get(conversation, {})),
                            wake._automatic_primary_claims.get(session.id)
                            is actual.token,
                            conversation not in wake._active,
                            wake._retry_timer,
                        )
                    )
                finally:
                    wake._registry_lock.release()
        elif code is type(wake).retry_soon.__code__ and local.get("self") is wake:
            caller = frame.f_back
            queued_state.setdefault("retry_returns", []).append(
                (
                    queued_release.is_set(),
                    id(threading.current_thread()),
                    (caller.f_code.co_filename, caller.f_code.co_name, caller.f_lineno)
                    if caller is not None
                    else None,
                )
            )
        elif (
            code is type(rig.coordinator)._release.__code__
            and local.get("run") is actual
        ):
            binding = local["binding"]
            record_checks(
                "original_release",
                (
                    result is binding,
                    binding[0] is rig.coordinator,
                    binding[1] is actual,
                    binding[2] is actual.phase,
                    binding[3] is actual.driver is asyncio.current_task(),
                    binding[7] is rig.store,
                    binding[8] is wake,
                    binding[9] is actual.intake,
                    binding[10] is actual.token,
                    rig.store.received_turn_for_session(session.id)
                    is not actual.intake,
                    wake._automatic_primary_claims.get(session.id) is not actual.token,
                    actual.workflow_id not in rig.coordinator._runs,
                ),
            )
            queued_state["release"].append((actual, actual.final_metadata, binding))

    def observe_shutdown(frame, event, result):
        code, local = frame.f_code, frame.f_locals
        thread = threading.current_thread()
        if (
            code is type(wake).retry_soon.__code__
            and event == "call"
            and local.get("self") is wake
            and not retry_entered.is_set()
            and any(thread is row[1] for row in children)
        ):
            caller = frame.f_back
            if caller is None or caller.f_code is not type(wake)._intake.__code__:
                return
            with wake._registry_lock:
                pending = tuple(wake._pending.get(conversation, {}))
            original_id = next(row[2] for row in children if row[1] is thread)
            record_checks(
                "child_retry",
                (
                    caller.f_locals.get("self") is wake,
                    original_id in pending,
                    wake._loop is loop,
                    not wake._disposed,
                ),
            )
            shutdown.update(
                retry=frame,
                child=thread,
                guard=_guard.context.get(),
                identity=execution_identity(),
                pending=pending,
            )
            retry_entered.set()
            if not retry_release.wait(
                max(0, queued_state["hold_deadline"] - time.monotonic())
            ):
                observer_errors.append(
                    TimeoutError("original metadata10 exhausted at child retry entry")
                )
        elif (
            normal_return(frame, event)
            and code is type(loop).call_soon_threadsafe.__code__
            and frame.f_back is shutdown.get("retry")
        ):
            callback = local["callback"]
            record_checks(
                "posted_handle",
                (
                    local["self"] is loop,
                    thread is shutdown["child"],
                    callback.__self__ is wake,
                    callback.__func__ is type(wake)._attempt_all,
                    result._loop is loop,
                    result._callback.__self__ is wake,
                    result._callback.__func__ is type(wake)._attempt_all,
                    result._context.get(_guard.context) is shutdown["guard"],
                    _guard.context.get() is shutdown["guard"],
                    execution_identity() == shutdown["identity"],
                ),
            )
            shutdown["handle"] = result
            shutdown["order"].append("posted")
            retry_posted.set()
        elif (
            code is type(loop).create_task.__code__
            and normal_return(frame, event)
            and getattr(local.get("coro"), "cr_code", None)
            is type(rig.controller).shutdown.__code__
        ):
            caller = frame.f_back
            outer = caller.f_back
            coroutine = local["coro"]
            original_frame = coroutine.cr_frame
            record_checks(
                "native_controller_shutdown_task_creation",
                (
                    local["self"] is loop,
                    caller.f_code is asyncio.create_task.__code__,
                    caller.f_locals.get("coro") is coroutine,
                    outer.f_code is type(rig.runtime)._dispose_owned.__code__,
                    outer.f_locals.get("self") is rig.runtime,
                    outer.f_locals.get("_recovery_owner") is wake,
                    asyncio.current_task() is shutdown.get("task"),
                    result.get_coro() is coroutine,
                    result.get_loop() is loop,
                    original_frame.f_locals.get("self") is rig.controller,
                    "controller_task" not in shutdown,
                ),
            )
            shutdown["controller_task"] = (result, coroutine, original_frame, outer)
        elif (
            code is type(rig.runtime).dispose.__code__
            and event == "call"
            and local.get("self") is rig.runtime
        ):
            if "task" not in shutdown:
                shutdown["task"] = asyncio.current_task()
                shutdown["order"].append("dispose")
        elif (
            code is type(wake).dispose.__code__
            and local.get("self") is wake
            and (event == "call" or normal_return(frame, event))
        ):
            caller = frame.f_back
            if caller.f_code is type(rig.runtime).dispose.__code__:
                if normal_return(frame, event):
                    record_checks(
                        "early_native_fence",
                        (
                            asyncio.current_task() is shutdown.get("task"),
                            wake._disposed,
                            rig.runtime._workflow_closing,
                            caller.f_locals.get("self") is rig.runtime,
                            caller.f_locals.get("recovery_owner") is wake,
                            result is None,
                        ),
                    )
                    shutdown["order"].append("fenced")
            elif caller.f_code is type(rig.controller).begin_shutdown.__code__:
                outer = caller.f_back
                record_checks(
                    "later_idempotent_native_fence",
                    (
                        caller.f_locals.get("self") is rig.controller,
                        rig.controller._fleet_wake is wake,
                        wake._disposed,
                    ),
                )
                if outer.f_code is type(rig.runtime)._dispose_owned.__code__:
                    kind = "direct"
                    record_checks(
                        "direct_idempotent_native_fence",
                        (
                            asyncio.current_task() is shutdown.get("task"),
                            outer.f_locals.get("self") is rig.runtime,
                            outer.f_locals.get("_recovery_owner") is wake,
                        ),
                    )
                elif outer.f_code is type(rig.controller).shutdown.__code__:
                    kind = "controller_task"
                    task, coroutine, original_frame, _owned = shutdown[
                        "controller_task"
                    ]
                    record_checks(
                        "controller_task_idempotent_native_fence",
                        (
                            asyncio.current_task() is task,
                            task.get_coro() is coroutine,
                            outer is original_frame,
                            outer.f_locals.get("self") is rig.controller,
                        ),
                    )
                else:
                    kind = "unknown"
                    record_checks("unknown_later_native_fence_caller", (False,))
                if event == "call":
                    shutdown.setdefault("later_fence_entries", []).append((kind, frame))
                else:
                    record_checks(
                        "later_native_fence_return",
                        (
                            any(
                                entry is frame and entered_kind == kind
                                for entered_kind, entry in shutdown.get(
                                    "later_fence_entries", ()
                                )
                            ),
                            result is None,
                        ),
                    )
                    shutdown.setdefault("later_fence_returns", []).append((kind, frame))
            else:
                record_checks("unknown_native_fence_caller", (False,))
        elif (
            code is type(rig.coordinator).close.__code__
            and event == "call"
            and local.get("self") is rig.coordinator
        ):
            if "close" not in shutdown["order"]:
                record_checks(
                    "g_close",
                    (
                        asyncio.current_task() is shutdown.get("task"),
                        local.get("_runtime_disposal") is rig.runtime,
                        wake._disposed,
                    ),
                )
                shutdown["order"].append("close")
        elif code is type(wake)._attempt_all.__code__ and local.get("self") is wake:
            ancestor = frame.f_back
            if (
                ancestor is not None
                and ancestor.f_code is asyncio.Handle._run.__code__
                and ancestor.f_locals["self"] is shutdown.get("handle")
            ):
                record_checks(
                    "same_handle_execution",
                    (
                        asyncio.current_task() is None,
                        wake._disposed,
                        rig.runtime._workflow_closing,
                        not shutdown["task"].done(),
                        not queued_state["proof"][1].done(),
                    ),
                )
                if event == "call":
                    shutdown["order"].append("handle_entry")
                elif normal_return(frame, event):
                    shutdown["order"].append("handle_return")
        elif code is type(wake)._attempt.__code__ and event == "call":
            if "fenced" in shutdown["order"]:
                shutdown["attempts"].append((local.get("self"), frame.f_back))
        elif (
            code is type(rig.runtime)._dispose_owned.__code__
            and local.get("self") is rig.runtime
        ):
            if event == "call" and "owned_entry" not in shutdown["order"]:
                shutdown["order"].append("owned_entry")
            elif normal_return(frame, event):
                task, coroutine, _original_frame, original_owned = shutdown[
                    "controller_task"
                ]
                record_checks(
                    "original_controller_task_in_returned_drain",
                    (
                        frame is original_owned,
                        asyncio.current_task() is shutdown["task"],
                        task in local["drain_tasks"],
                        task.get_coro() is coroutine,
                        task.done()
                        and not task.cancelled()
                        and task.exception() is None,
                        shutdown.get("controller_return") is task,
                    ),
                )
                shutdown["order"].append("owned_return")
        elif code is type(rig.controller).shutdown.__code__ and normal_return(
            frame, event
        ):
            task, coroutine, original_frame, _owned = shutdown["controller_task"]
            record_checks(
                "original_controller_shutdown_return",
                (
                    frame is original_frame,
                    local["self"] is rig.controller,
                    asyncio.current_task() is task,
                    task.get_coro() is coroutine,
                    result is None,
                ),
            )
            shutdown["controller_return"] = task
        elif (
            code is type(loop).call_soon_threadsafe.__code__
            and event == "call"
            and "fenced" in shutdown["order"]
        ):
            callback = local.get("callback")
            if (
                getattr(callback, "__self__", None) is wake
                and getattr(callback, "__func__", None) is type(wake)._attempt_all
            ):
                shutdown["schedules_after_fence"].append(callback)

    def context_observe(frame, event, result):
        code = frame.f_code
        if (
            code is type(wake).retry_soon.__code__
            and frame.f_locals.get("self") is wake
        ):
            thread = threading.current_thread()
            task = asyncio.current_task() if thread is threading.main_thread() else None
            if (
                thread in {row[1] for row in children}
                or any(task is row[0] for row in provenance["entry"])
                or queued
                and task is queued_state.get("driver")
            ):
                provenance["retry"].append(
                    (
                        event,
                        threading.current_thread(),
                        _guard.context.get(),
                        execution_identity(),
                        task,
                        frame.f_back.f_code if frame.f_back is not None else None,
                    )
                )
        elif event == "return" and code is type(loop).call_soon_threadsafe.__code__:
            callback = frame.f_locals.get("callback")
            if (
                getattr(callback, "__self__", None) is wake
                and getattr(callback, "__func__", None) is type(wake)._attempt_all
            ):
                ancestor = frame.f_back
                while (
                    ancestor is not None
                    and ancestor.f_code is not asyncio.Handle._run.__code__
                ):
                    ancestor = ancestor.f_back
                caller = frame.f_back.f_back if frame.f_back is not None else None
                provenance["handles"].append(
                    (
                        result,
                        result._context.get(_guard.context),
                        threading.current_thread(),
                        execution_identity(),
                        asyncio.current_task()
                        if threading.current_thread() is threading.main_thread()
                        else None,
                        caller.f_code if caller is not None else None,
                        ancestor.f_locals["self"] if ancestor is not None else None,
                    )
                )
        elif (
            normal_return(frame, event)
            and code is type(loop).call_later.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is type(wake)._schedule_delayed_retry.__code__
            and frame.f_back.f_locals.get("self") is wake
        ):
            ancestor = frame.f_back
            while (
                ancestor is not None
                and ancestor.f_code is not asyncio.Handle._run.__code__
            ):
                ancestor = ancestor.f_back
            provenance["timers"].append(
                (
                    result,
                    result._context.get(_guard.context),
                    ancestor.f_locals["self"] if ancestor is not None else None,
                )
            )
        elif event == "return" and code is type(loop).create_task.__code__:
            parent_frame = frame.f_back
            if (
                parent_frame is not None
                and parent_frame.f_code is type(wake)._attempt.__code__
                and parent_frame.f_locals.get("self") is wake
            ):
                while (
                    parent_frame is not None
                    and parent_frame.f_code is not asyncio.Handle._run.__code__
                ):
                    parent_frame = parent_frame.f_back
                handle = (
                    parent_frame.f_locals.get("self")
                    if parent_frame is not None
                    else None
                )
                provenance["schedule"].append(
                    (result, handle, _guard.context.get(), execution_identity())
                )
        elif (
            event == "call"
            and code is type(wake)._deliver.__code__
            and frame.f_locals.get("self") is wake
        ):
            task = asyncio.current_task()
            if not any(row[0] is task for row in provenance["entry"]):
                provenance["entry"].append(
                    (
                        task,
                        _guard.context.get(),
                        execution_identity(),
                        current_execution_owner(),
                        current_task_authorization(),
                        current_workflow_preparation(),
                        current_automatic_work(),
                    )
                )
        elif (
            event == "call"
            and code is _AgentSources.prepare.__code__
            and frame.f_locals["self"].controller is rig.controller
        ):
            source = frame.f_locals["self"]
            binding = native_reads._source_binding.get()
            original = frame.f_back
            if queued and original.f_code is configure_code:
                prepared = original.f_locals["self"]
                read, callback = binding.read, binding.read.callback
                captured = dict(
                    zip(
                        callback.__code__.co_freevars,
                        (cell.cell_contents for cell in callback.__closure__),
                        strict=True,
                    )
                )
                record, issuing = captured["record"], captured["issuing"]
                record_checks(
                    "g_configure",
                    (
                        type(prepared) is _PreparedWorkflowReply,
                        source.workflow_preparation is prepared,
                        prepared.source is source,
                        prepared.phase is queued_state["phase"],
                        prepared.runtime is rig.runtime,
                        prepared.controller is rig.controller,
                        prepared in rig.runtime._workflow_preparations,
                        binding.active,
                        read.source is prepared,
                        read.creator is rig.controller,
                        read.task is issuing is queued_state["driver"],
                        callback.__code__ is execution_read_code,
                        captured["self"] is prepared,
                        captured["callback"].__self__ is prepared,
                        captured["callback"].__func__
                        is _PreparedWorkflowReply._configure,
                        record.prepared is prepared,
                        record.issuing is issuing,
                        record.callback is callback,
                        record.proof[0] is read,
                        record.proof[1] is read._producer,
                        record.proof[2] is read.retired,
                        record.proof[3] is callback,
                    ),
                )
                workflow_source_reads.append(
                    (
                        source,
                        prepared,
                        read,
                        read._producer,
                        read.retired,
                        record,
                        issuing,
                        callback,
                        record.proof,
                        record.owner,
                        record.claim,
                        record.operation,
                    )
                )
                return
            provenance["prepare"].append(
                (
                    source,
                    binding,
                    binding.read,
                    binding.read._producer,
                    _guard.context.get(),
                    execution_identity(),
                    original.f_locals.get("callback"),
                    current_automatic_work(),
                )
            )

    def report_context(error, task):
        if not trace_context:
            return
        workers = [row for row in provenance["prepare"] if row[2].task is task]
        schedules = [
            row
            for row in provenance["schedule"]
            if any(row[0] is item[3] for item in submissions if item[2] is task)
        ]
        entries = [
            row
            for row in provenance["entry"]
            if schedules and row[0] is schedules[0][0]
        ]
        state = workers[0][4] if workers else None
        retries = [row for row in provenance["retry"] if row[2] is state]
        handles = [row for row in provenance["handles"] if row[1] is state]
        timer_chain = [row for row in provenance["timers"] if row[1] is state]
        same = bool(
            state is not None
            and workers
            and schedules
            and entries
            and retries
            and handles
        )
        if same:
            source, binding, read, producer, _state, identity, callback, _context = (
                workers[0]
            )
            same = bool(
                schedules[0][2] is entries[0][1] is state
                and any(row[0] is schedules[0][1] for row in handles + timer_chain)
                and any(
                    row[0] == "call" and row[3] == state.identity for row in retries
                )
                and any(
                    row[0] == "return" and row[3] == state.identity for row in retries
                )
                and identity != state.identity
                and entries[0][2] != state.identity
                and binding.read is read
                and read._producer is producer
                and read.creator is rig.controller
                and read.task is task
                and callback.__self__ is source
                and callback.__func__ is _AgentSources.prepare
                and source.received_claim is not None
                and producer.done()
                and producer.exception() is error
            )
        print(
            "G_MIXED_CONTEXT_PROVENANCE",
            {
                "same_chain": same,
                "state": id(state),
                "origin_identity": (
                    state.identity[0],
                    state.identity[1],
                    id(state.identity[2]),
                )
                if state is not None
                else None,
                "child_threads": [id(row[1]) for row in retries],
                "handles": [id(row[0]) for row in handles],
                "timers": [id(row[0]) for row in timer_chain],
                "delivery_tasks": [id(row[0]) for row in entries],
                "delivery_identity": [
                    (row[2][0], row[2][1], id(row[2][2])) for row in entries
                ],
                "custody_task": id(task),
                "producer": id(workers[0][3]) if workers else None,
                "worker_identity": [
                    (row[5][0], row[5][1], id(row[5][2])) for row in workers
                ],
                "state_live_at_report": state.live if state is not None else None,
            },
            flush=True,
        )
        if not same:
            error.add_note(
                "Original mixed context provenance incomplete; no child-origin claim"
            )

    def settled(event):
        with wake._registry_lock:
            events.append((event, dict(wake._pending.get(event.conversation_id, {}))))

    def parent_observe(frame, event, result):
        caller = frame.f_back
        if (
            event == "call"
            and frame.f_code is threading.Thread.start.__code__
            and caller is not None
            and caller.f_code is AgentService._launch_prepared_child_result.__code__
            and caller.f_locals.get("self") is saved.get("service")
            and caller.f_locals.get("fleet") is original_fleet
            and threading.current_thread() is parent
        ):
            thread = frame.f_locals["self"]
            children.append(
                (
                    caller.f_locals["handle"].handle_id,
                    thread,
                    caller.f_locals["child_run_id"],
                    caller.f_locals["child_owner"],
                )
            )
            saved.setdefault("targets", {})[thread] = thread._target
            child_target_codes.add(thread._target.__code__)
        if not normal_return(frame, event):
            return
        if frame.f_code is AgentService.__init__.__code__:
            saved["service"] = frame.f_locals["self"]
        elif frame.f_code is AgentService._run_owned.__code__ and frame.f_locals[
            "self"
        ] is saved.get("service"):
            saved.update(owner=frame.f_locals["owner"], returned=result)

    def host_observe(frame, event, result):
        if (
            normal_return(frame, event)
            and frame.f_code is type(rig.runtime)._submit_fleet_wake.__code__
            and frame.f_locals.get("self") is rig.runtime
            and isinstance(result, str)
        ):
            authorization = frame.f_locals["wake_authorization"]
            record = rig.runtime._turn_custody[result]
            issuing = asyncio.current_task()
            delivery = wake._active[authorization.conversation_id]
            record_checks(
                "wake_submit",
                (
                    wake._delivery_tasks[issuing] == authorization.conversation_id,
                    delivery.authorization is authorization,
                    current_automatic_work() is authorization.context,
                    record.task.get_context().get(automatic_context)
                    is authorization.context,
                ),
            )
            custody_sources[record.task] = (record, record.store, record.received_claim)
            submissions.append((authorization, record, record.task, issuing, delivery))
        elif (
            normal_return(frame, event)
            and frame.f_code is type(rig.runtime)._release_custody.__code__
            and frame.f_locals.get("self") is rig.runtime
        ):
            original = frame.f_locals.get("record")
            for task, (record, store, claim) in tuple(custody_sources.items()):
                if original is record:
                    record_checks(
                        "custody_release",
                        (
                            frame.f_locals["turn_id"] == record.turn_id,
                            record.turn_id not in rig.runtime._turn_custody,
                            record.task
                            is record.store
                            is record.received_claim
                            is None,
                            claim is not None
                            and store.received_turn_for_session(record.session_id)
                            is not claim,
                        ),
                    )
                    custody_returns.append((record, task, store, claim))
        elif (
            event == "call"
            and frame.f_code is type(rig.controller)._submit_draft_lifecycle.__code__
            and frame.f_locals.get("self") is rig.controller
            and frame.f_locals.get("origin") is ConsoleSubmissionOrigin.AGENT_WAKE
        ):
            authorization = frame.f_locals["wake_authorization"]
            record_checks(
                "wake_context",
                (
                    current_automatic_work() is authorization.context,
                    asyncio.current_task().get_context().get(automatic_context)
                    is authorization.context,
                ),
            )
            if not any(row[0] is authorization for row in controller_entries):
                controller_entries.append((authorization, asyncio.current_task()))

    def lifecycle_observe(frame, event, result):
        if not normal_return(frame, event):
            return
        instance = frame.f_locals.get("self")
        code = frame.f_code
        thread = threading.current_thread()
        target = saved.get("targets", {}).get(thread)
        if target is not None and code is target.__code__:
            child_target_returns.append((thread, target))
        if code is _OrdinaryTurn.admit.__code__ and any(
            instance.source is row[0] for row in provenance["prepare"]
        ):
            tree_records[instance] = (
                instance.source,
                instance.owner,
                instance.claim,
                instance.retired,
            )
        elif (
            code
            in {
                _SnapshotSource.prepare_ordinary.__code__,
                _SnapshotSource.prepare_survivor.__code__,
            }
            and instance.ordinary_tree in tree_records
        ):
            with instance._holds_lock:
                held = tuple(instance._holds.items())
            record = next(
                (row for source, row in snapshot_records if source is instance), None
            )
            if record is None:
                record = {"tickets": {}, "reservation": None}
                snapshot_records.append((instance, record))
            for kind, ticket in held:
                record["tickets"][kind] = (ticket, ticket.operation)
        elif code is ChangeReviewFinalizationCoordinator.register.__code__ and any(
            source is frame.f_locals.get("native_source")
            for source, _record in snapshot_records
        ):
            for source, record in snapshot_records:
                if source is frame.f_locals["native_source"]:
                    record["reservation"] = result
                    record["coordinator"] = instance
                    record["workers"] = tuple(instance._workers)
                    record["publisher"] = instance._publisher
        elif code is _SnapshotTicket.returned.__code__ and any(
            instance is ticket
            for _source, record in snapshot_records
            for ticket, _operation in record["tickets"].values()
        ):
            if instance.retired.is_set():
                ticket_returns.append((instance, instance.operation, instance.source))
        elif code is OwnedOperation.finish.__code__ and any(
            instance.owner is row[1] for row in tree_records.values()
        ):
            operation_returns.append(instance)
        elif code is WorktreeActivitySettlement.settle.__code__ and any(
            instance is row[2] for row in tree_records.values()
        ):
            claim_returns.append((instance, frame.f_locals["durable"]))
        elif (
            code is _OrdinaryTurn._retire_scratch.__code__
            and instance in tree_records
            and frame.f_locals.get("elected") is True
        ):
            tree_returns.append((instance, instance.retired))

    def observe(frame, event, result):
        if event not in {"call", "return"} or (
            frame.f_code not in observed_codes
            and frame.f_code not in child_target_codes
        ):
            return
        if (
            notification_failure
            and not notification_entry
            and event == "call"
            and frame.f_code is type(loop).call_soon_threadsafe.__code__
            and frame.f_locals.get("self") is loop
        ):
            caller = frame.f_back
            notifying = caller.f_back if caller is not None else None
            callback = frame.f_locals.get("callback")
            if (
                caller is not None
                and caller.f_code is type(wake).retry_soon.__code__
                and caller.f_locals.get("self") is wake
                and notifying is not None
                and notifying.f_code is notify_release_code
                and notifying.f_locals.get("self") is rig.coordinator
            ):
                actual = queued_state["run"]
                cell = queued_state["cell"]
                read, producer, retired, _callback = queued_state["proof"]
                binding = notifying.f_locals["binding"]
                # This one intentional injection is outside the fact-only observer.
                assert asyncio.current_task() is actual.driver
                assert getattr(callback, "__self__", None) is wake
                assert getattr(callback, "__func__", None) is type(wake)._attempt_all
                assert any(row[2] is binding for row in queued_state["release"])
                assert cell.run is actual and cell.issuing is actual.driver
                assert read._producer is producer and read.retired is retired
                assert producer.done() and producer.exception() is None
                assert retired.done() and retired.result() is None
                assert cell.disposed.done() and cell.disposed.result() is True
                assert actual.workflow_id not in rig.coordinator._runs
                assert rig.store.received_turn_for_session(session.id) is None
                with wake._registry_lock:
                    assert session.id not in wake._automatic_primary_claims
                    pending = tuple(wake._pending.get(conversation, {}))
                notification_entry.append((actual.driver, cell, binding, pending))
                threading.setprofile_all_threads(None)
                sys.setprofile(None)
                raise notification_error
        # Observer failures are delivered on the test host, never through native code.
        try:
            if permanent_close:
                observe_shutdown(frame, event, result)
            if queued:
                queued_observe(frame, event, result)
            if threading.current_thread() is parent:
                parent_observe(frame, event, result)
            host_observe(frame, event, result)
            context_observe(frame, event, result)
            lifecycle_observe(frame, event, result)
        except BaseException as error:  # noqa: BLE001 - preserve observer failure separately.
            observer_errors.append(error)

    def record_checks(group, values):
        # Label only already-evaluated predicates; never revisit mutable sources.
        observer_checks.extend(values)
        observer_labels.extend(
            (group, index, value) for index, value in enumerate(values)
        )

    def check_observer():
        if not all(observer_checks):
            print(
                "G_MIXED_FALSE_PREDICATES",
                [row for row in observer_labels if not row[2]],
                shutdown["order"],
                flush=True,
            )
        if observer_errors:
            raise observer_errors[0]
        assert all(observer_checks), "original mixed observer identity mismatch"

    try:
        bridge.on_fleet_child_settled("workflow-mixed-observer", settled)
        with bridge._fleet_child_fanout._lock:
            consumers = tuple(bridge._fleet_child_fanout._consumers)
        assert consumers[-1] == ("workflow-mixed-observer", settled)
        native_consumer = next(
            callback for name, callback in consumers if name == wake.NAME
        )
        assert native_consumer.__self__ is wake
        assert native_consumer.__func__ is type(wake).on_child_settled
        assert autowake_enabled()
        session = rig.store._sessions["workflow-test-chat"]
        conversation = session.persisted_conversation_id
        resolution = await rig.gateway.resolve_for_send(
            rig.controller._provider_selection_for_session(session.id)
        )
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await asyncio.shield(rig.coordinator._runs[rig.workflow.id].driver)
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        if queued:
            threading.setprofile_all_threads(observe)
            approval = asyncio.create_task(
                rig.coordinator.approve(decision, command_id="mixed-workflow")
            )
            # G completion retains its original whole-process bound. The worker's
            # separate hold10 starts only at the exact terminal metadata return.
            while not queued_entered.is_set():
                check_observer()
                if approval.done():
                    if approval.cancelled():
                        await asyncio.shield(approval)
                    if (error := approval.exception()) is not None:
                        raise error
                if "run" in queued_state:
                    original = queued_state["run"]
                    driver = queued_state["driver"]
                    assert original.driver is driver
                    if driver.done():
                        if driver.cancelled():
                            await asyncio.shield(driver)
                        if (error := driver.exception()) is not None:
                            raise error
                        raise AssertionError("original terminal metadata hold missing")
                    assert rig.coordinator._runs.get(original.workflow_id) is original
                    assert original.phase is queued_state["phase"]
                await asyncio.sleep(0.01)
            check_observer()
            run = queued_state["run"]
            assert run.final_metadata is queued_state["cell"]
            assert run.phase._retired
            assert len(run.waves) == 1 and any(
                run.waves[0] is row for row in queued_state["waves"]
            )
            original_wave = run.waves[0]
            assert original_wave[0].closed and original_wave[1].done()
            assert original_wave[1].exception() is None
            assert original_wave[2].done() and original_wave[3].done()
            assert rig.coordinator._runs[rig.workflow.id] is run
            assert rig.store.received_turn_for_session(session.id) is run.intake
            assert wake._automatic_primary_claims.get(session.id) is run.token
            assert not wake.has_pending(conversation) and not provider.wake_calls
            assert rig.runtime.execution_capacity.snapshot().executions == ()
            completed = None
        else:
            await rig.coordinator.approve(decision, command_id="mixed-workflow")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.shield(run.driver)
        authorization = run.phase._task_scopes["task-1"]
        prepared = authorization._bridge_preparation[0]
        native = authorization._native_preparation[0]

        async def completed_workflow():
            completed = await rig.coordinator.get(rig.workflow.id)
            assert completed.status == "awaiting_result_review"
            assert (
                len(completed.results) == 1
                and completed.results[0].status == "completed"
            )
            workflow_events = [
                row
                for row in events
                if row[0].child.run_id == completed.results[0].run_id
            ]
            assert len(workflow_events) == 1
            assert workflow_events[0][0].child.workflow_task is authorization.claim
            assert completed.results[0].run_id not in workflow_events[0][1]
            if not queued:
                assert not provider.wake_calls
            assert rig.workflow.id not in rig.coordinator._runs
            intake = rig.store.received_turn_for_session(session.id)
            token = wake._automatic_primary_claims.get(session.id)
            if queued:
                check_observer()
                assert intake is not run.intake and token is not run.token
                if token is not None:
                    delivery = wake._active[conversation]
                    assert delivery.slot_token is token
                    assert any(
                        row[0] is delivery
                        and row[3] is token
                        and row[1].get_coro() is row[2]
                        and wake._delivery_tasks.get(row[1]) == conversation
                        for row in queued_state["deliveries"]
                    )
                if intake is not None:
                    assert any(
                        claim is intake
                        and store is rig.store
                        and record.task is task
                        and record.received_claim is claim
                        for task, (record, store, claim) in custody_sources.items()
                    )
            else:
                assert intake is None and token is None
            return completed

        if not queued:
            completed = await completed_workflow()
            assert not wake.has_pending(conversation)
            assert rig.runtime.execution_capacity.snapshot().executions == ()
        original_fleet, inbox = native.fleet, native.fleet.message_inbox
        assistant = rig.store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )

        def drive_parent():
            sys.setprofile(observe)
            try:
                saved["outcome"] = participants.run_finite_local_worker(
                    original_parent,
                    bridge,
                    rig.store,
                    session,
                    assistant.id,
                    conversation_id=conversation,
                    resolution=resolution,
                    expected_progress_owner_id=prepared.native.progress_owner_id,
                    agent_messages=[
                        {"role": "user", "content": provider.parent_instruction}
                    ],
                )
            except BaseException as error:  # noqa: BLE001 - preserve original native fixture failure.
                saved["error"] = error
            finally:
                sys.setprofile(None)

        parent = threading.Thread(target=drive_parent, name="mixed-original-parent")
        setup_deadline = loop.time() + 10
        parent.start()
        await asyncio.to_thread(parent.join, max(0, setup_deadline - loop.time()))
        assert not parent.is_alive(), "original independent parent did not return"
        if "error" in saved:
            raise saved["error"]
        service = saved["service"]
        assert saved["outcome"].status == "done" and service._automatic_work is None
        assert service.work_origin is WorkOrigin.MANUAL
        assert service._work_chain_id != run.phase.attempt.chain_id
        assert (
            service._fleet is original_fleet and original_fleet.message_inbox is inbox
        )
        assert bridge._fleet_coordinators[conversation] is original_fleet
        drained = []
        saved["owner"].on_drained(drained.append)
        assert drained == [True] and not saved["owner"].is_admitted
        with storage_admission._lock:
            assert not any(
                lease.resource_thread is parent
                for lease in storage_admission._live_leases
            )
        check_observer()
        assert all(
            service._fleet_threads.get(handle_id) is thread
            for handle_id, thread, _run_id, _owner in children
        )
        assert len(children) == 2, (
            "original parent did not retain exactly two child Threads"
        )
        for entered in provider.child_entered.values():
            reached = await asyncio.to_thread(
                entered.wait, max(0, setup_deadline - loop.time())
            )
            if not reached:
                print(
                    "G_MIXED_SETUP_ENTRY_MISSING",
                    [
                        (id(thread), thread.is_alive(), owner.is_admitted)
                        for _handle_id, thread, _handle, owner in children
                    ],
                    [event.is_set() for event in provider.child_entered.values()],
                    flush=True,
                )
            assert reached, "original child provider did not enter within setup10"
        assert all(
            owner.is_admitted and owner.run_id == run_id and thread.is_alive()
            for _handle_id, thread, run_id, owner in children
        )
        child_ids = {row[2] for row in children}
        with owned_goal_storage(rig.service):
            assert all(
                rig.service.db.get_run(run_id)["work_chain_id"]
                == service._work_chain_id
                for run_id in child_ids
            )
        threading.setprofile_all_threads(observe)
        for gate in provider.child_release.values():
            gate.set()
        if permanent_close:
            while not retry_entered.is_set():
                check_observer()
                assert not queued_state["proof"][1].done()
                await asyncio.sleep(0.01)
            check_observer()
            assert (
                approval.done()
                and not approval.cancelled()
                and approval.exception() is None
            )
            assert _guard.context.get() is None and current_automatic_work() is None
            assert asyncio.get_running_loop() is loop
            assert not retry_posted.is_set()
            retry_release.set()
            assert retry_posted.wait(
                max(0, queued_state["hold_deadline"] - time.monotonic())
            ), "original retry Handle did not return within metadata10"
            check_observer()
            handle = shutdown["handle"]
            assert handle in loop._ready and not handle.cancelled()
            assert not queued_state["proof"][1].done()
            # A real native close Task enters before yielding to the queued Handle.
            # Construct it outside every profiler callback; never invoke the Handle.
            close_task = asyncio.Task(
                rig.runtime.dispose(), loop=loop, eager_start=True
            )
            assert shutdown["task"] is close_task and not close_task.done()
            assert shutdown["order"] == ["posted", "dispose", "fenced", "close"]
            assert (
                rig.runtime._workflow_closing
                and rig.coordinator._closing
                and wake._disposed
            )
            assert (
                not rig.runtime._disposed
                and not rig.runtime.execution_capacity.snapshot().closed
            )
            assert rig.store.received_turn_for_session(session.id) is run.intake
            assert wake._automatic_primary_claims[session.id] is run.token
            assert (
                not queued_state["proof"][1].done()
                and not queued_state["proof"][2].done()
            )
            while "handle_return" not in shutdown["order"]:
                check_observer()
                assert not queued_state["proof"][1].done()
                await asyncio.sleep(0.01)
            check_observer()
            assert shutdown["order"] == [
                "posted",
                "dispose",
                "fenced",
                "close",
                "handle_entry",
                "handle_return",
            ]
            assert (
                shutdown["attempts"] == []
                and not submissions
                and not provider.wake_calls
            )
            queued_release.set()
            await asyncio.shield(run.driver)
            await asyncio.shield(approval)
            await asyncio.shield(close_task)
            check_observer()
            cell = queued_state["cell"]
            read, producer, retired, callback = queued_state["proof"]
            assert run.final_metadata is cell and cell.proof is queued_state["proof"]
            assert read.task is run.driver and read.callback is callback
            assert read._producer is producer and read.retired is retired
            assert (
                producer.done()
                and not producer.cancelled()
                and producer.exception() is None
            )
            assert (
                retired.done() and not retired.cancelled() and retired.result() is None
            )
            assert cell.disposed.done() and cell.disposed.result() is True
            assert cell.body_error is cell.producer_error is cell.source_error is None
            assert any(
                row[0] is run and row[1] is cell for row in queued_state["release"]
            )
            assert (
                rig.workflow.id not in rig.coordinator._runs
                and cell not in rig.coordinator._metadata_cells
            )
            assert rig.store.received_turn_for_session(session.id) is None
            assert session.id not in wake._automatic_primary_claims
            assert shutdown["attempts"] == shutdown["schedules_after_fence"] == []
            assert shutdown["order"][-2:] == ["owned_entry", "owned_return"]
            assert [kind for kind, _frame in shutdown["later_fence_entries"]] == [
                "direct",
                "controller_task",
            ]
            assert shutdown["later_fence_returns"] == shutdown["later_fence_entries"]
            assert (
                rig.runtime._disposed
                and rig.runtime.execution_capacity.snapshot().closed
            )
            assert not submissions and not factory_calls and not provider.wake_calls
            assert [key for key, _ in provider.calls] == ["planning", "task-1"]
            # Runtime close has sealed coordinator.get. The original passive DB
            # read verifies persisted results/counts without reopening admission.
            with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
                completed = rig.service.get(rig.workflow.id)
                assert (
                    len(completed.results) == 1
                    and completed.results[0].status == "completed"
                )
                assert (
                    completed.results[0].run_id
                    == authorization._native_launch[0].run_id
                )
                assert completed.accounting == queued_state["view"].accounting
                assert completed.accounting.used["generation"] == 2
                assert completed.accounting.used["model_call"] == 2
                assert completed.accounting.used["child_launch"] == 1
                assert not completed.accounting.uncertain
                assert (
                    conn.execute(
                        "SELECT count(*) FROM automatic_wake_attempts WHERE chain_id=?",
                        (service._work_chain_id,),
                    ).fetchone()[0]
                    == 0
                )
                charges = rig.service.db.automatic_work.snapshot(service._work_chain_id)
                assert charges.used["generation"] == charges.used["model_call"] == 0
                assert not charges.uncertain
            # Same bounded joins as ordinary success/finally, while original
            # target-return observation remains installed. A failed bound still
            # fails even if the mandatory finally cleanup later returns.
            for _id, thread, _run_id, _owner in children:
                await asyncio.to_thread(thread.join, 5)
                assert not thread.is_alive()
            check_observer()
            assert all(
                any(
                    row[0] is thread and row[1] is saved["targets"][thread]
                    for row in child_target_returns
                )
                for _id, thread, _run_id, _owner in children
            )
            assert rig.runtime.execution_capacity.snapshot().executions == ()
            print(
                "G_Q1_POSTED_ORIGINAL_HANDLE_AFTER_PERMANENT_FENCE",
                id(handle),
                shutdown["order"],
                flush=True,
            )
            return
        if queued:
            # The original metadata worker has its own unchanged10s hold.
            while not (
                queued_state["timers"]
                and any(
                    set(row[0]) == child_ids and row[1] and row[2] and row[3] is None
                    for row in queued_state["blocked"]
                )
            ):
                check_observer()
                assert not queued_state["proof"][1].done(), (
                    "original hold returned before blocked wake"
                )
                await asyncio.sleep(0.01)
            assert not submissions and not provider.wake_calls
            assert rig.coordinator._runs[rig.workflow.id] is run
            assert rig.store.received_turn_for_session(session.id) is run.intake
            assert wake._automatic_primary_claims.get(session.id) is run.token
            print("G_QUEUED_ORIGINAL_TOKEN_REFUSAL", sorted(child_ids), flush=True)
            if notification_cancel:
                assert run.driver.cancel()
                await asyncio.sleep(0)
                assert not run.driver.done() and not queued_state["proof"][1].done()
                assert rig.coordinator._runs[rig.workflow.id] is run
                assert rig.store.received_turn_for_session(session.id) is run.intake
                assert wake._automatic_primary_claims.get(session.id) is run.token
            queued_release.set()
            if notification_failure:
                try:
                    await run.driver
                except BaseException as error:  # noqa: BLE001 - require the exact original disposition.
                    original_error = error
                else:
                    raise AssertionError("original scheduling error was not delivered")
            else:
                await asyncio.shield(run.driver)
            await asyncio.shield(approval)
            cell = queued_state["cell"]
            read, producer, retired, callback = queued_state["proof"]
            assert cell is run.final_metadata and cell.proof is queued_state["proof"]
            assert (
                cell.run is run
                and cell.phase is run.phase
                and cell.issuing is run.driver
            )
            assert read.callback is callback and read.task is run.driver
            assert read._producer is producer and read.retired is retired
            assert (
                producer.done()
                and not producer.cancelled()
                and producer.exception() is None
            )
            assert (
                retired.done() and not retired.cancelled() and retired.result() is None
            )
            assert cell.disposed.done() and cell.disposed.result() is True
            assert cell.body_error is cell.producer_error is cell.source_error is None
            assert any(
                row[0] is run and row[1] is cell for row in queued_state["release"]
            )
            completed = await completed_workflow()
            print(
                "G_QUEUED_ORIGINAL_WORKFLOW_RELEASED",
                id(cell),
                id(producer),
                flush=True,
            )
            if notification_failure:
                check_observer()
                assert len(notification_entry) == 1
                assert notification_entry[0][0] is run.driver
                assert notification_entry[0][1] is cell
                assert set(notification_entry[0][3]) == child_ids
                if notification_cancel:
                    assert original_error is cell.caller_error
                    assert isinstance(original_error, asyncio.CancelledError)
                    assert original_error.__cause__ is notification_error
                    assert run.driver.cancelled()
                else:
                    assert original_error is notification_error
                    assert run.driver.exception() is notification_error
                    assert cell.caller_error is None
                assert run.status != "recovery_required"
                assert rig.workflow.id not in rig.coordinator._runs
                assert cell not in rig.coordinator._metadata_cells
                assert not submissions and not provider.wake_calls
                assert not factory_calls and not controller_entries
                with wake._registry_lock:
                    assert set(wake._pending[conversation]) == child_ids
                    assert conversation not in wake._active
                    assert not wake._delivery_tasks
                    assert wake._retry_timer is None
                    assert session.id not in wake._automatic_primary_claims
                assert rig.store.received_turn_for_session(session.id) is None
                assert all(
                    any(
                        row[0] is thread and row[1] is saved["targets"][thread]
                        for row in child_target_returns
                    )
                    for _handle_id, thread, _run_id, _owner in children
                )
                assert (
                    len([row for row in events if row[0].child.run_id in child_ids])
                    == 2
                )
                with (
                    owned_goal_storage(rig.service),
                    rig.service.db.connection() as conn,
                ):
                    assert (
                        conn.execute(
                            "SELECT count(*) FROM automatic_wake_attempts WHERE chain_id=?",
                            (service._work_chain_id,),
                        ).fetchone()[0]
                        == 0
                    )
                    charges = rig.service.db.automatic_work.snapshot(
                        service._work_chain_id
                    )
                    assert charges.used["generation"] == charges.used["model_call"] == 0
                    assert not charges.uncertain
                    assert (
                        conn.execute(
                            "SELECT count(*) FROM automatic_work_chains"
                        ).fetchone()[0]
                        == 2
                    )
                assert (
                    await rig.coordinator.get(rig.workflow.id)
                ).accounting == completed.accounting
                assert [key for key, _ in provider.calls] == ["planning", "task-1"]
                assert rig.runtime.execution_capacity.snapshot().executions == ()
                print(
                    "G_Q1_ORIGINAL_NOTIFICATION_FAILURE",
                    notification_cancel,
                    flush=True,
                )
                return
        async with asyncio.timeout(20):
            while not (
                submissions
                and all(
                    any(
                        event.child.run_id == child_run_id for event, _pending in events
                    )
                    for _handle_id, _thread, child_run_id, _owner in children
                )
                and all(
                    any(
                        row[0] is thread and row[1] is saved["targets"][thread]
                        for row in child_target_returns
                    )
                    for _handle_id, thread, _run_id, _owner in children
                )
                and not wake.has_pending(conversation)
                and all(row[2].done() and row[3].done() for row in submissions)
            ):
                check_observer()
                for (
                    _authorization,
                    _record,
                    original,
                    _issuing,
                    _delivery,
                ) in submissions:
                    if original.done() and not original.cancelled():
                        error = original.exception()
                        if error is not None:
                            import traceback

                            report_context(error, original)
                            traceback.print_exception(error)
                            raise error
                await asyncio.sleep(0.01)
            check_observer()
            source_counts = (
                len(workflow_source_reads),
                len(provenance["prepare"]),
                len(submissions),
                [id(row[2].task) for row in provenance["prepare"]],
                [id(row[2]) for row in submissions],
            )
            assert len(provenance["prepare"]) == len(submissions), source_counts
            if queued:
                assert len(workflow_source_reads) == 1, source_counts
                (
                    source,
                    workflow_prepared,
                    read,
                    producer,
                    retired,
                    record,
                    issuing,
                    callback,
                    proof,
                    owner,
                    claim,
                    operation,
                ) = workflow_source_reads[0]
                assert workflow_prepared is prepared and prepared.source is source
                assert record in prepared._finite_reads and record.proof is proof
                assert read.task is issuing is run.driver
                assert read.callback is callback and read._producer is producer
                assert read.retired is retired
                assert producer.done() and not producer.cancelled()
                assert producer.exception() is None
                assert retired.done() and not retired.cancelled()
                assert retired.result() is None
                actual = producer.result()
                assert len(actual) == 10 and all(
                    actual[index] is expected
                    for index, expected in enumerate(
                        (
                            record,
                            prepared,
                            issuing,
                            callback,
                            proof,
                            owner,
                            claim,
                            operation,
                        )
                    )
                )
                assert record.scopes_returned and record.positive
                assert record.wrapper_error is record.body_error is None
                assert run.phase._task_scopes["task-1"] is prepared.authorization
                assert prepared.authorization._bridge_preparation[0] is prepared
            for authorization, _record, task, _delivery_task, _delivery in submissions:
                (prepared_read,) = [
                    row for row in provenance["prepare"] if row[2].task is task
                ]
                source = prepared_read[0]
                (tree,) = [
                    tree for tree, row in tree_records.items() if row[0] is source
                ]
                original_source, owner, claim, retired = tree_records[tree]
                assert tree.source is original_source and tree.owner is owner
                assert tree.claim is claim and tree.retired is retired
                await asyncio.shield(asyncio.wrap_future(retired))
                while not any(
                    row[0] is tree and row[1] is retired for row in tree_returns
                ):
                    check_observer()
                    await asyncio.sleep(0)
                assert (
                    retired.done()
                    and not retired.cancelled()
                    and retired.exception() is None
                )
                assert any(row[0] is claim and row[1] is True for row in claim_returns)
                drained = []
                owner.on_drained(drained.append)
                assert drained == [True] and not owner.is_admitted
                assert tree not in rig.runtime._ordinary_turns
                for snapshot, record in snapshot_records:
                    if snapshot.ordinary_tree is not tree:
                        continue
                    assert record["reservation"] is not None
                    assert snapshot.owner is owner and snapshot.claim is claim
                    for ticket, operation in record["tickets"].values():
                        while not any(row[0] is ticket for row in ticket_returns):
                            check_observer()
                            await asyncio.sleep(0)
                        assert ticket.source is snapshot and ticket.owner is owner
                        assert (
                            ticket.operation is operation and operation.owner is owner
                        )
                        assert any(
                            original is operation for original in operation_returns
                        )
                        assert ticket.retired.is_set() and not ticket._failed_write
                print(
                    "G_MIXED_ORIGINAL_TREE_RETURN",
                    id(tree),
                    id(owner),
                    [
                        (
                            id(snapshot),
                            [
                                (kind, id(ticket), id(operation))
                                for kind, (ticket, operation) in record[
                                    "tickets"
                                ].items()
                            ],
                        )
                        for snapshot, record in snapshot_records
                        if snapshot.ordinary_tree is tree
                    ],
                    flush=True,
                )
            while not all(
                any(
                    row[0] is thread and row[1] is saved["targets"][thread]
                    for row in child_target_returns
                )
                for _handle_id, thread, _run_id, _owner in children
            ):
                check_observer()
                await asyncio.sleep(0)
            check_observer()
        threading.setprofile_all_threads(None)
        sys.setprofile(None)
        assert 1 <= len(submissions) <= 2
        if trace_context:
            assert len(factory_calls) == len(submissions)
            for coroutine, task, context, initial_items in factory_calls:
                assert type(context) is contextvars.Context and initial_items == ()
                assert task.get_context() is context and task.get_coro() is coroutine
                assert any(row[3] is task for row in submissions)
            assert (
                len(provenance["entry"])
                == len(provenance["prepare"])
                == len(submissions)
            )
            child_triggered = 0
            workflow_triggered = 0
            for entry in provenance["entry"]:
                assert entry[1] is None and all(item is None for item in entry[3:])
                (scheduled,) = [
                    row for row in provenance["schedule"] if row[0] is entry[0]
                ]
                state = scheduled[2]
                trigger = scheduled[1]
                visited = set()
                while True:
                    assert trigger is not None and id(trigger) not in visited
                    visited.add(id(trigger))
                    handles = [
                        row for row in provenance["handles"] if row[0] is trigger
                    ]
                    if handles:
                        (origin,) = handles
                        if origin[6] is None or origin[4] is not None:
                            break
                        trigger = origin[6]
                    else:
                        (timer,) = [
                            row for row in provenance["timers"] if row[0] is trigger
                        ]
                        trigger = timer[2]
                assert origin[1] is state
                if origin[2] in {row[1] for row in children}:
                    child_triggered += 1
                    assert state is not None and origin[3] == state.identity
                    assert any(
                        row[0] == "call"
                        and row[1] is origin[2]
                        and row[2] is state
                        and row[3] == state.identity
                        for row in provenance["retry"]
                    )
                    assert any(
                        row[0] == "return"
                        and row[1] is origin[2]
                        and row[2] is state
                        and row[3] == state.identity
                        for row in provenance["retry"]
                    )
                elif queued and origin[4] is run.driver:
                    workflow_triggered += 1
                    assert state is None and origin[5] is notify_release_code
                    assert any(
                        row[0] is run and row[2][3] is run.driver
                        for row in queued_state["release"]
                    )
                    assert any(
                        row[0] == "call"
                        and row[4] is run.driver
                        and row[5] is notify_release_code
                        for row in provenance["retry"]
                    )
                    assert any(
                        row[0] == "return"
                        and row[4] is run.driver
                        and row[5] is notify_release_code
                        for row in provenance["retry"]
                    )
                else:
                    assert state is None and origin[4] is not None
                    assert origin[5] is type(wake)._deliver.__code__
                    assert any(row[3] is origin[4] for row in submissions)
            if queued:
                assert workflow_triggered == 1
            else:
                assert child_triggered >= 1
            for (
                source,
                binding,
                read,
                producer,
                state,
                _identity,
                callback,
                issued,
            ) in provenance["prepare"]:
                assert (
                    state is None
                    and binding.read is read
                    and read._producer is producer
                )
                assert (
                    callback.__self__ is source
                    and callback.__func__ is _AgentSources.prepare
                )
                assert (
                    producer.done()
                    and not producer.cancelled()
                    and producer.exception() is None
                )
                assert read.retired.done() and read.retired.result() is None
                assert any(
                    row[2] is read.task and row[0].context is issued
                    for row in submissions
                )
            print(
                "G_MIXED_FRESH_DELIVERY_AND_NATIVE_CUSTODY_CONTEXT",
                len(factory_calls),
                flush=True,
            )
        assert len(provider.wake_calls) == len(submissions) == len(controller_entries)
        for authorization, record, task, delivery_task, _delivery in submissions:
            original_record, store, claim = custody_sources[task]
            assert original_record is record
            assert any(
                row[0] is record
                and row[1] is task
                and row[2] is store
                and row[3] is claim
                for row in custody_returns
            )
            assert record.task is record.store is record.received_claim is None
            assert task.done() and not task.cancelled()
            assert task.exception() is None and task.result().accepted
            assert not delivery_task.cancelled() and delivery_task.exception() is None
            assert any(
                row[0] is authorization and row[1] is task for row in controller_entries
            )
            assert authorization._coordinator is wake
            assert authorization.work_chain_id == service._work_chain_id
        ordinary_events = [row for row in events if row[0].child.run_id in child_ids]
        assert len(ordinary_events) == 2
        assert all(
            row[0].child.workflow_task is None and row[0].child.settled_after_turn
            for row in ordinary_events
        )
        # Post-consumer pending views may already have been consumed by the host.
        # Actual inclusion is proved below by original completed wake attempts.
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            attempts = conn.execute(
                "SELECT id,run_ids_json,state FROM automatic_wake_attempts WHERE chain_id=?",
                (service._work_chain_id,),
            ).fetchall()
            assert {row[0] for row in attempts} == {
                row[0].attempt_id for row in submissions
            }
            delivered = [run_id for row in attempts for run_id in json.loads(row[1])]
            assert len(delivered) == 2 and set(delivered) == child_ids
            assert all(row[2] == "completed" for row in attempts)
            charges = rig.service.db.automatic_work.snapshot(service._work_chain_id)
            assert (
                charges.used["generation"]
                == charges.used["model_call"]
                == len(attempts)
            )
            assert not charges.uncertain
            tokens = conn.execute(
                "SELECT actual_amount,state FROM automatic_work_reservations WHERE chain_id=? AND kind='tokens'",
                (service._work_chain_id,),
            ).fetchall()
            assert len(tokens) == len(attempts) and all(
                tuple(row) == (10, "settled") for row in tokens
            )
            assert (
                conn.execute("SELECT count(*) FROM automatic_work_chains").fetchone()[0]
                == 2
            )
        assert (
            await rig.coordinator.get(rig.workflow.id)
        ).accounting == completed.accounting
        assert [key for key, _ in provider.calls] == ["planning", "task-1"]
        for _handle_id, thread, _handle, owner in children:
            await asyncio.to_thread(thread.join, 5)
            assert not thread.is_alive()
            returned = []
            owner.on_drained(returned.append)
            assert returned == [True] and not owner.is_admitted
            with storage_admission._lock:
                assert not any(
                    lease.resource_thread is thread
                    for lease in storage_admission._live_leases
                )
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        assert rig.store.received_turn_for_session(session.id) is None
        assert session.id not in wake._automatic_primary_claims
    except BaseException as error:
        primary_error = error
        try:
            # Sequential, nonblocking observations; unavailable is unknown.
            delivery = {"submissions": len(submissions), "children": [], "tasks": []}
            for _handle_id, thread, run_id, _owner in children:
                delivery["children"].append(
                    (
                        id(thread),
                        run_id,
                        any(event.child.run_id == run_id for event, _ in events),
                        any(
                            row[0] is thread and row[1] is saved["targets"][thread]
                            for row in child_target_returns
                        ),
                    )
                )
            delivery["registry"] = "unavailable"
            if wake._registry_lock.acquire(blocking=False):
                try:
                    delivery["registry"] = (
                        tuple(wake._pending.get(conversation, {})),
                        bool(wake._pending_progress.get(conversation)),
                        id(wake._active.get(conversation)),
                        tuple(
                            id(task)
                            for task, cid in wake._delivery_tasks.items()
                            if cid == conversation
                        ),
                    )
                finally:
                    wake._registry_lock.release()
            for _auth, _record, custody, issuing, _delivery in submissions:
                for role, task in (("custody", custody), ("delivery", issuing)):
                    task_error = (
                        task.exception()
                        if task.done() and not task.cancelled()
                        else None
                    )
                    stack, seen = [], set()
                    coroutine = task.get_coro()
                    while coroutine is not None and id(coroutine) not in seen:
                        seen.add(id(coroutine))
                        frame = getattr(coroutine, "cr_frame", None)
                        if frame is not None:
                            stack.append(
                                (
                                    frame.f_code.co_filename,
                                    frame.f_code.co_name,
                                    frame.f_lineno,
                                )
                            )
                        coroutine = getattr(coroutine, "cr_await", None)
                    delivery["tasks"].append(
                        (
                            role,
                            id(task),
                            task.done(),
                            task.cancelled(),
                            type(task_error).__name__ if task_error else None,
                            stack,
                        )
                    )
            print("G_MIXED_DELIVERY_FAILURE_CONJUNCTS", delivery, flush=True)
            if queued:
                original = queued_state.get("run")
                driver = queued_state.get("driver")
                driver_stack, seen = [], set()
                coroutine = driver.get_coro() if driver is not None else None
                while coroutine is not None and id(coroutine) not in seen:
                    seen.add(id(coroutine))
                    frame = getattr(coroutine, "cr_frame", None)
                    if frame is not None:
                        driver_stack.append(
                            (
                                frame.f_code.co_filename,
                                frame.f_code.co_name,
                                frame.f_lineno,
                            )
                        )
                    coroutine = getattr(coroutine, "cr_await", None)
                observer_details = []
                for original_error in observer_errors:
                    frames = []
                    traceback = original_error.__traceback__
                    while traceback is not None:
                        frame = traceback.tb_frame
                        frames.append(
                            (
                                frame.f_code.co_filename,
                                frame.f_code.co_name,
                                traceback.tb_lineno,
                            )
                        )
                        traceback = traceback.tb_next
                    observer_details.append(
                        (type(original_error).__name__, str(original_error), frames)
                    )
                cell = original.final_metadata if original is not None else None
                proof = cell.proof if cell is not None else None
                print(
                    "G_QUEUED_FAILURE_STATE",
                    {
                        "metadata_close_returns": [
                            tuple(map(id, row)) for row in metadata_closes
                        ],
                        "observer_errors": observer_details,
                        "metadata_checks": queued_state.get("metadata_checks"),
                        "original_driver": (
                            id(driver),
                            driver.done(),
                            driver.cancelled(),
                            type(driver.exception()).__name__
                            if driver.done()
                            and not driver.cancelled()
                            and driver.exception() is not None
                            else None,
                            driver_stack,
                        )
                        if driver is not None
                        else None,
                        "run_association": (
                            original.driver is driver,
                            rig.coordinator._runs.get(original.workflow_id) is original,
                            original.phase is queued_state["phase"],
                            original.phase._retired,
                            original.status,
                        )
                        if original is not None
                        else None,
                        "waves": [
                            (id(row), id(row[0]), id(row[1]), row[1].done())
                            for row in queued_state["waves"]
                        ],
                        "cell": (
                            id(cell),
                            cell.issuing is driver,
                            cell.disposed.done(),
                            type(cell.body_error).__name__ if cell.body_error else None,
                            type(cell.source_error).__name__
                            if cell.source_error
                            else None,
                            type(cell.producer_error).__name__
                            if cell.producer_error
                            else None,
                            (
                                id(proof[0]),
                                id(proof[1]),
                                proof[1].done(),
                                proof[2].done(),
                            )
                            if proof is not None
                            else None,
                        )
                        if cell is not None
                        else None,
                        "original_release": [
                            (id(row[0]), id(row[1])) for row in queued_state["release"]
                        ],
                        "refusals": len(queued_state["refusals"]),
                        "timers_returned": list(map(id, queued_state["timers"])),
                        "retry_returns": queued_state.get("retry_returns", ()),
                        "timer_now": id(wake._retry_timer)
                        if wake._retry_timer is not None
                        else None,
                        "approval_done": approval.done()
                        if approval is not None
                        else None,
                    },
                    flush=True,
                )
            frames = sys._current_frames()
            rows = []
            for tree, (source, owner, claim, retired) in tuple(tree_records.items()):
                operations, owner_state, tree_state = None, "unavailable", "unavailable"
                if owner.capacity._lock.acquire(blocking=False):
                    try:
                        operations = tuple(owner._operations)
                        owner_state = (
                            owner._root_finished,
                            owner._cleanup_unproven,
                            owner._drain_outcome,
                            owner.capacity._owners.get(owner.execution_id) is owner,
                        )
                    finally:
                        owner.capacity._lock.release()
                if tree.lock.acquire(blocking=False):
                    try:
                        tree_state = (
                            tree.sealed,
                            tree.source_scopes,
                            tuple(map(id, tree.primary_hosts)),
                            tuple(map(id, tree.future_events)),
                            tree.retired.done(),
                        )
                    finally:
                        tree.lock.release()
                tickets = []
                for snapshot, record in snapshot_records:
                    if snapshot.ordinary_tree is tree:
                        held = None
                        if snapshot._holds_lock.acquire(blocking=False):
                            try:
                                held = tuple(snapshot._holds.values())
                            finally:
                                snapshot._holds_lock.release()
                        for kind, (ticket, operation) in record["tickets"].items():
                            ticket_state = "unavailable"
                            if ticket._lock.acquire(blocking=False):
                                try:
                                    ticket_state = (
                                        ticket._worker_returned,
                                        ticket._handoff_returned,
                                        ticket._failed_write,
                                        ticket._retired,
                                        ticket.retired.is_set(),
                                    )
                                finally:
                                    ticket._lock.release()
                            tickets.append(
                                (
                                    kind,
                                    id(ticket),
                                    id(operation),
                                    any(operation is op for op in operations)
                                    if operations is not None
                                    else None,
                                    any(ticket is item for item in held)
                                    if held is not None
                                    else None,
                                    ticket_state,
                                )
                            )
                        coordinator = record.get("coordinator")
                        reservation = record["reservation"]
                        coordinator_state = "missing_or_replaced"
                        if (
                            coordinator is not None
                            and snapshot.issuer.coordinator is coordinator
                        ):
                            coordinator_state = "unavailable"
                            if coordinator._lock.acquire(blocking=False):
                                try:
                                    state = coordinator._states.get(reservation.id)
                                    if (
                                        state is not None
                                        and state.public is reservation
                                        and state.native_source is snapshot
                                    ):
                                        pending = state.pending_operation
                                        coordinator_state = (
                                            state.phase,
                                            state.operation_inflight,
                                            pending.kind
                                            if pending is not None
                                            else None,
                                            state.pending_reason,
                                            state.cancelled,
                                            any(
                                                item is snapshot
                                                for item in coordinator._native_retirements
                                            ),
                                        )
                                    else:
                                        coordinator_state = "absent_or_mismatched"
                                finally:
                                    coordinator._lock.release()
                        native_stacks = []
                        for native_thread in (
                            *record.get("workers", ()),
                            record.get("publisher"),
                        ):
                            if native_thread is None:
                                continue
                            ident = native_thread.ident
                            alive = native_thread.is_alive()
                            frame = (
                                frames.get(ident)
                                if alive and native_thread.ident == ident
                                else None
                            )
                            stack, matches = [], []
                            while frame is not None:
                                stack.append(
                                    (
                                        frame.f_code.co_filename,
                                        frame.f_code.co_name,
                                        frame.f_lineno,
                                    )
                                )
                                local_operation = None
                                if (
                                    frame.f_code
                                    is ChangeReviewFinalizationCoordinator._worker_loop.__code__
                                    and frame.f_locals.get("self") is coordinator
                                ):
                                    local_operation = frame.f_locals.get("operation")
                                elif (
                                    frame.f_code
                                    is ChangeReviewFinalizationCoordinator._consume_operation_result.__code__
                                    and frame.f_locals.get("self") is coordinator
                                ):
                                    local_operation = frame.f_locals.get("result")
                                for kind, (ticket, _operation) in record[
                                    "tickets"
                                ].items():
                                    if (
                                        getattr(local_operation, "ticket", None)
                                        is ticket
                                    ):
                                        matches.append(
                                            (kind, id(ticket), local_operation.kind)
                                        )
                                frame = frame.f_back
                            native_stacks.append(
                                (id(native_thread), ident, alive, stack, matches)
                            )
                        print(
                            "G_MIXED_SAME_SNAPSHOT_FAILURE",
                            id(snapshot),
                            id(reservation),
                            id(coordinator),
                            coordinator_state,
                            native_stacks,
                            flush=True,
                        )
                rows.append(
                    (
                        id(tree),
                        id(source),
                        id(owner),
                        id(claim),
                        owner_state,
                        tree_state,
                        [(id(op), op.kind, op._stopping) for op in operations]
                        if operations is not None
                        else None,
                        tickets,
                    )
                )
            print("G_MIXED_EXACT_FAILURE_STATE", rows, flush=True)
        except BaseException as observer_error:  # noqa: BLE001 - never replace the original failure.
            observer_errors.append(observer_error)
            print(
                "G_MIXED_FAILURE_DIAGNOSTIC_ERROR",
                type(observer_error).__name__,
                flush=True,
            )
        raise
    finally:
        retry_release.set()
        queued_release.set()
        threading.setprofile_all_threads(None)
        sys.setprofile(None)
        cleanup_errors = []
        try:
            for gate in provider.child_release.values():
                gate.set()
            for thread in ([parent] if parent is not None else []) + [
                row[1] for row in children
            ]:
                try:
                    if thread.ident is not None:
                        await asyncio.to_thread(thread.join, 5)
                    assert not thread.is_alive()
                except BaseException as error:  # noqa: BLE001 - preserve every original cleanup failure.
                    cleanup_errors.append(error)
            for _handle_id, thread, _run_id, owner in children:
                try:
                    returned = []
                    owner.on_drained(returned.append)
                    assert returned == [True] and not owner.is_admitted
                    with storage_admission._lock:
                        assert not any(
                            lease.resource_thread is thread
                            for lease in storage_admission._live_leases
                        )
                except BaseException as error:  # noqa: BLE001 - still attempt native runtime disposal.
                    cleanup_errors.append(error)
        finally:
            if close_task is not None:
                try:
                    await asyncio.shield(close_task)
                except BaseException as error:  # noqa: BLE001 - preserve the actual original disposal outcome.
                    cleanup_errors.append(error)
            try:
                await rig.close()
            except BaseException as error:  # noqa: BLE001 - native refusal remains a failure.
                cleanup_errors.append(error)
            finally:
                if approval is not None:
                    try:
                        await asyncio.shield(approval)
                    except BaseException as error:  # noqa: BLE001 - retain the original command outcome too.
                        cleanup_errors.append(error)
                threading.setprofile_all_threads(None)
                if trace_context:
                    loop.set_task_factory(original_factory)
        if cleanup_errors:
            import traceback

            for error in cleanup_errors:
                traceback.print_exception(error)
            if primary_error is not None:
                primary_error.add_note(
                    "Original mixed cleanup also failed; exact exceptions printed above"
                )
            else:
                raise cleanup_errors[0]


def test_enabled_native_workflow_exclusion_and_unrelated_wake_delivery(tmp_path):
    _run(
        tmp_path,
        "mixed-wake",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_original_native_mixed_wake_context_provenance(tmp_path):
    _run(
        tmp_path,
        "mixed-wake",
        "context",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_original_ordinary_wake_queued_before_workflow_release(tmp_path):
    _run(
        tmp_path,
        "mixed-wake",
        "queued",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_original_posted_wake_handle_executes_after_native_permanent_fence(tmp_path):
    _run(
        tmp_path,
        "mixed-wake",
        "queued-close",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize("cancelled", [False, True])
def test_original_postrelease_notification_error_preserves_disposition(
    tmp_path, cancelled
):
    _run(
        tmp_path,
        "mixed-wake",
        "queued-notify-cancel" if cancelled else "queued-notify-error",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_original_mixed_failure_boundary_identifies_native_owner(tmp_path):
    _run(
        tmp_path,
        "mixed-wake",
        "owner-census",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )
