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


async def planning_wake_journey(tmp_path, monkeypatch):
    """An actual pending survivor wake crosses the original F/G release."""
    import asyncio
    import concurrent.futures
    import contextvars
    import dis
    import json
    import os
    import sys
    import threading
    import time
    import traceback
    import types

    from Tests.Chat.test_console_agent_bridge import _run as original_parent
    from Tests.Chat.workflow_rig import WorkflowFleetProvider
    from tldw_chatbook.Agents.activation import _guard
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.automatic_work_runtime import (
        _CURRENT as automatic_context,
    )
    from tldw_chatbook.Agents.automatic_work_runtime import (
        current_automatic_work,
    )
    from tldw_chatbook.Agents.execution_capacity import (
        ExecutionOwner,
        OwnedOperation,
        WorkOrigin,
        current_execution_owner,
    )
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_task_scope import current_task_authorization
    from tldw_chatbook.Backup_Recovery import participants, storage_admission
    from tldw_chatbook.Chat import console_preparation_reads as planning_reads
    from tldw_chatbook.Chat import console_workflow_runs as workflow_runs
    from tldw_chatbook.Chat.console_agent_sources import _AgentSources
    from tldw_chatbook.Chat.console_chat_controller import (
        capture_run_admitted_workspace_roots,
    )
    from tldw_chatbook.Chat.console_chat_store import ConsoleMessageRole
    from tldw_chatbook.Chat.console_ordinary_tree import _OrdinaryTurn
    from tldw_chatbook.Chat.console_workflow_preparation import (
        _PreparedWorkflowReply,
        current_workflow_preparation,
        prepare_workflow_planning,
    )
    from tldw_chatbook.Workspaces.change_review_ownership import (
        _SnapshotSource,
        _SnapshotTicket,
    )
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivitySettlement

    os.environ["TLDW_AGENTS_MAX_LIVE_SUBAGENTS"] = "2"
    os.environ["TLDW_AGENTS_RESERVED_MANUAL_SUBAGENTS"] = "0"
    os.environ["TLDW_AGENTS_SUBAGENTS_OUTLIVE_TURN"] = "true"
    mode = sys.argv[2]
    direct = mode == "direct"
    notify_failure = mode in {"notify-error", "notify-cancel"}
    cancel_failure = mode == "notify-cancel"
    metadata_failure = mode == "metadata-unknown"
    metadata_error = ValueError("original final planning metadata outer reader failed")
    schedule_error = ValueError("original planning notification scheduling failed")
    metadata_gate = threading.Event()
    shutdown = mode in {"shutdown-fence", "shutdown-before-fence"}
    before_fence = mode == "shutdown-before-fence"
    fence_error = ValueError("original planning wake disposal failed before the fence")
    close_task = None
    direct_prepared = direct_driver = None
    g_code = workflow_runs.ConsoleWorkflowCoordinator._plan.__code__
    provider = WorkflowFleetProvider()
    rig = await build_workflow_rig(
        tmp_path,
        monkeypatch,
        provider,
        public_coordinator=not direct,
        native_fleet_consumers=True,
    )
    session = rig.store.sessions()[0]
    conversation = session.persisted_conversation_id
    wake, bridge = rig.controller.fleet_wake, rig.controller._agent_bridge
    loop = asyncio.get_running_loop()
    resolution = await rig.gateway.resolve_for_send(
        rig.controller._provider_selection_for_session(session.id)
    )
    state, original, children = {}, {}, []
    events, errors, retries, handles, attempts = [], [], [], [], []
    physical, child_returns, deliveries = [], [], []
    priorities, send_refusals, state_results = [], [], []
    task_calls, wake_tasks, submissions, custody_returns = {}, [], [], []
    source_reads, trees, tree_returns, claim_returns = [], {}, [], []
    tickets, ticket_returns, operation_returns = {}, [], []
    held, release = threading.Event(), threading.Event()
    changed = asyncio.Event()
    parent = None
    primary_error = None
    run_code = _PreparedWorkflowReply.run_planning.__code__
    invoke_code = next(
        code
        for code in run_code.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    finalize_code = next(
        code
        for code in invoke_code.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "finalize"
    )
    reader_code = next(
        code
        for code in planning_reads.run_preparation_read.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    codes = {
        run_code,
        invoke_code,
        concurrent.futures.Future.result.__code__,
        AgentService.__init__.__code__,
        AgentService._run_owned.__code__,
        threading.Thread.start.__code__,
        type(wake).retry_soon.__code__,
        type(wake)._attempt_all.__code__,
        type(wake)._attempt.__code__,
        type(wake)._priority_allows.__code__,
        type(rig.controller).send_refusal_copy.__code__,
        type(rig.controller).run_state_for.__code__,
        type(wake).try_claim_automatic_primary.__code__,
        type(wake).release_automatic_primary.__code__,
        type(wake)._deliver.__code__,
        type(loop).call_soon_threadsafe.__code__,
        type(loop).create_task.__code__,
        type(rig.runtime)._submit_fleet_wake.__code__,
        type(rig.runtime)._release_custody.__code__,
        _AgentSources.prepare.__code__,
        _OrdinaryTurn.admit.__code__,
        _OrdinaryTurn._retire_scratch.__code__,
        _SnapshotSource.prepare_ordinary.__code__,
        _SnapshotSource.prepare_survivor.__code__,
        _SnapshotTicket.returned.__code__,
        OwnedOperation.finish.__code__,
        type(rig.store).release_received_turn.__code__,
        ExecutionOwner.finish_root.__code__,
        WorktreeActivitySettlement.settle.__code__,
        _PreparedWorkflowReply._release_configuration.__code__,
        _PreparedWorkflowReply._release_planning_slots.__code__,
        workflow_runs._metadata_publication_return.__code__,
        g_code,
    }

    if notify_failure or metadata_failure:
        from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

        snapshot_body = WorkflowRunService.runtime_snapshot.__wrapped__.__code__
        received_code = _PreparedWorkflowReply._planning_release_received.__code__
        codes.update((snapshot_body, received_code))
        if metadata_failure:
            codes.update(
                (reader_code, _PreparedWorkflowReply._notify_planning_release.__code__)
            )
    if shutdown:
        codes.update(
            (
                type(wake).dispose.__code__,
                _PreparedWorkflowReply._notify_planning_release.__code__,
            )
        )
    monitor = sys.monitoring
    event_types = monitor.events
    post_codes = frozenset(codes)
    post_code_ids = frozenset(map(id, post_codes))
    tool_id = 3
    monitor_owned = local_mode = profile_attempted = global_attempted = False
    saved_callbacks, armed_codes = {}, []

    def returned(frame, event):
        return event == "return" and dis.opname[
            frame.f_code.co_code[frame.f_lasti]
        ] in {"RETURN_VALUE", "RETURN_CONST"}

    def check():
        if errors:
            raise errors[0]

    def notify_change():
        if threading.current_thread() is threading.main_thread():
            changed.set()
        else:
            loop.call_soon_threadsafe(changed.set)

    def driver_returned(task):
        assert task is state["driver"]
        changed.set()

    def original_custody():
        captured = state.get("invocation")
        if captured is None:
            return {"captured": False}
        prepared = state["prepared"]
        (
            frame,
            caller,
            worker,
            binding,
            read,
            callback,
            producer,
            retired,
            owner,
            claim,
        ) = captured
        returned_read = prepared._planning_read_return
        return {
            "captured": True,
            "same_frames": frame.f_back is caller
            and frame.f_locals.get("self") is prepared
            and caller.f_code is reader_code,
            "same_binding": caller.f_locals.get("binding") is binding
            and binding.read is read,
            "same_worker": binding.thread is worker
            and binding.process == os.getpid()
            and binding.task is None,
            "active_binding": binding.active,
            "same_read": caller.f_locals.get("read") is read
            and caller.f_locals.get("callback") is callback
            and read.callback is callback
            and read.source is prepared
            and read.creator is rig.controller
            and read.task is state["driver"]
            and read._producer is producer
            and read.retired is retired,
            "same_return_tuple": type(returned_read) is tuple
            and len(returned_read) == 4
            and all(
                left is right
                for left, right in zip(
                    returned_read, (read, producer, retired, callback)
                )
            ),
            "producer_pending": not producer.done() and not producer.cancelled(),
            "retired_pending": not retired.done() and not retired.cancelled(),
            "same_owner": prepared.primary_owner is owner,
            "same_claim": prepared.primary_claim is claim,
            "physical_not_returned": not prepared._planning_physical_returned,
            "invoke_not_returned": not state.get("invocation_returned", False),
        }

    def refusal_kind(attempt):
        if attempt in state.get("same_token_refusals", []):
            return "same_f_token"
        finalizer_state = state.get("finalizer_run_state")
        for priority, caller, result, _position in priorities:
            if caller is not attempt or result is not False:
                continue
            for refusal, priority_caller, copy, _return_position in send_refusals:
                if (
                    priority_caller is priority
                    and copy == "A run is already running in this tab."
                    and finalizer_state is not None
                    and any(
                        source is refusal and native_state is finalizer_state
                        for source, native_state in state_results
                    )
                ):
                    return "same_f_finalizer_run_state"
        return None

    def ordinary_failure():
        release_handle = state.get("release_handle")
        # Independent, non-atomic observations of existing native references.
        created, issued = tuple(wake_tasks), tuple(submissions)
        observed_reads, known_trees = tuple(source_reads), tuple(trees.items())
        known_tickets = tuple(tickets.items())
        active = wake._active.get(conversation)
        result = {
            "local_mode": local_mode,
            "release_handle_seen": release_handle is not None,
            "release_handle_entered": release_handle is not None
            and any(item is release_handle for item in state.get("handle_entries", [])),
            "release_attempts": len(state.get("release_attempts", ())),
            "release_retry_normal": state.get("release_retry_return") is not None,
            "pending": sorted(wake._pending.get(conversation, {})),
            "provider_wake_calls": len(provider.wake_calls),
            "observed_code_unwinds": list(state.get("post_unwinds", ())),
            "deliveries": [],
            "submissions": [],
        }

        def status(future):
            if type(future) is concurrent.futures.Future:
                return {"concurrent_state": future._state}
            if type(future) in (asyncio.Future, asyncio.Task):
                return {"done": future.done(), "cancelled": future.cancelled()}
            return None

        def task_state(task, authorization, custody_task, follow_provider=True):
            issuer_sources = tuple(
                source
                for source, read, producer, retired, original_task in observed_reads
                if original_task is custody_task
                and read.task is custody_task
                and read.creator is rig.controller
                and read.callback.__self__ is source
                and read.callback.__func__ is _AgentSources.prepare
                and read._producer is producer
                and read.retired is retired
            )
            rows, coro = [], task.get_coro()
            for _ in range(32):
                if type(coro) is not types.CoroutineType:
                    break
                frame, code = coro.cr_frame, coro.cr_code
                row = {
                    "code": code.co_name,
                    "file": code.co_filename,
                    "line": frame.f_lineno if frame is not None else None,
                }
                if frame is not None:
                    local = frame.f_locals
                    if (
                        code is type(wake)._deliver.__code__
                        and local.get("self") is wake
                    ):
                        row.update(
                            same_session=local.get("session_id") == session.id,
                            same_conversation=local.get("conversation_id")
                            == conversation,
                            active_slot=active is not None
                            and active.slot_token is local.get("slot_token"),
                            same_authorization=authorization is not None
                            and local.get("authorization") is authorization,
                            terminal=status(local.get("terminal")),
                        )
                    if (
                        follow_provider
                        and code
                        is type(rig.controller._chat_start).run_provider_worker.__code__
                        and local.get("self") is rig.controller._chat_start
                    ):
                        item, op = local.get("item"), local.get("operation")
                        worker = getattr(item, "provider_worker", None)
                        if (
                            local.get("session_id") == session.id
                            and item is not None
                            and item.accepted
                            and rig.controller._chat_start._active.get(session.id)
                            is item
                            and type(worker) is asyncio.Task
                            and worker.get_coro() is op
                            and worker is not task
                        ):
                            row["original_provider_task"] = task_state(
                                worker, authorization, custody_task, False
                            )
                    if (
                        code
                        is type(rig.controller)._run_maintenance_agent_call.__code__
                        and local.get("self") is rig.controller
                    ):
                        row.update(
                            original_bridge=getattr(
                                local.get("function"), "__self__", None
                            )
                            is bridge
                            and getattr(local.get("function"), "__func__", None)
                            is type(bridge).run_reply,
                            known_source=any(
                                local.get("sources") is source
                                for source in issuer_sources
                            ),
                            known_tree=any(
                                local.get("tree") is tree
                                and local.get("sources") is tree_data[0]
                                and tree.source is tree_data[0]
                                and tree_data[0].ordinary_tree is tree
                                and any(
                                    tree_data[0] is source for source in issuer_sources
                                )
                                for tree, tree_data in known_trees
                            ),
                            has_native_host=local.get("native_host") is not None,
                            registered_native_host=local.get("native_host") is not None
                            and any(
                                local.get("tree") is tree
                                and local["native_host"] in tree.primary_hosts
                                and any(
                                    tree_data[0] is source for source in issuer_sources
                                )
                                for tree, tree_data in known_trees
                            ),
                            retained_worker=status(local.get("task")),
                            same_session=local.get("kwargs", {}).get("session_id")
                            == session.id,
                            same_chain=authorization is not None
                            and local.get("kwargs", {}).get("work_chain_id")
                            == authorization.work_chain_id,
                        )
                        worker, owned = local.get("worker"), local.get("owned_call")
                        retained = local.get("task")
                        if (
                            row["original_bridge"]
                            and row["known_tree"]
                            and row["known_source"]
                            and row["same_session"]
                            and row["same_chain"]
                            and row["has_native_host"]
                            and type(retained) is asyncio.Task
                            and retained in rig.controller._maintenance_background
                            and type(worker) is types.FunctionType
                            and getattr(worker, "__wrapped__", None) is owned
                        ):
                            row["bound_worker_stacks"] = []
                            for tid, worker_frame in sys._current_frames().items():
                                labels, matched = [], False
                                for _ in range(64):
                                    if worker_frame is None:
                                        break
                                    labels.append(
                                        (
                                            worker_frame.f_code.co_name,
                                            worker_frame.f_lineno,
                                        )
                                    )
                                    if (
                                        worker_frame.f_code is worker.__code__
                                        and worker_frame.f_locals.get("function")
                                        is owned
                                        and worker_frame.f_locals.get("service")
                                        is bridge
                                    ):
                                        matched = True
                                    worker_frame = worker_frame.f_back
                                if matched:
                                    row["bound_worker_stacks"].append((tid, labels))
                rows.append(row)
                coro = coro.cr_await
            return {"state": status(task), "chain": rows}

        for task, context, handle in created:
            matches = [row for row in issued if row[1] is task]
            auth = matches[0][0] if len(matches) == 1 else None
            result["deliveries"].append(
                {
                    "original_release_handle": release_handle is not None
                    and handle is release_handle,
                    "entered": any(item is task for item in deliveries),
                    "registered": wake._delivery_tasks.get(task) == conversation,
                    "created_context": task.get_context() is context,
                    "task": task_state(
                        task, auth, matches[0][3] if len(matches) == 1 else None
                    ),
                }
            )
        for auth, delivery, custody, task, store, claim in issued:
            ordinary = {
                "attempt_id": auth.attempt_id,
                "accepted": auth.accepted,
                "release_delivery": any(
                    t is delivery and h is release_handle and h is not None
                    for t, _, h in created
                ),
                "active_authorization": active is not None
                and active.authorization is auth,
                "registered_custody": rig.runtime._turn_custody.get(custody.turn_id)
                is custody,
                "same_task": custody.task is task,
                "same_context": task.get_context().get(automatic_context)
                is auth.context,
                "same_store": custody.store is store,
                "same_claim": claim is not None and custody.received_claim is claim,
                "custody_return": any(
                    c is custody and t is task for c, t in tuple(custody_returns)
                ),
                "task": task_state(task, auth, task),
                "reads": [],
            }
            for source, read, producer, retired, owner_task in observed_reads:
                if owner_task is not task:
                    continue
                source_state = {
                    "same_read_task": read.task is task,
                    "same_callback": read.creator is rig.controller
                    and read.callback.__self__ is source
                    and read.callback.__func__ is _AgentSources.prepare,
                    "same_producer_retired": read._producer is producer
                    and read.retired is retired,
                    "producer": status(producer),
                    "retired": status(retired),
                    "trees": [],
                    "bound_workers": [],
                }
                for tid, frame in sys._current_frames().items():
                    labels, matched = [], False
                    for _ in range(64):
                        if frame is None:
                            break
                        labels.append((frame.f_code.co_name, frame.f_lineno))
                        if frame.f_code is reader_code:
                            loc = frame.f_locals
                            binding = loc.get("binding")
                            if (
                                loc.get("read") is read
                                and loc.get("callback") is read.callback
                                and type(binding) is planning_reads._HookSourceBinding
                                and binding.read is read
                                and binding.thread.ident == tid
                                and binding.task is None
                                and binding.active
                            ):
                                matched = True
                        frame = frame.f_back
                    if matched:
                        source_state["bound_workers"].append((tid, labels))
                for tree, (
                    original_source,
                    owner,
                    settlement,
                    tree_retired,
                ) in known_trees:
                    if original_source is source:
                        source_state["trees"].append(
                            {
                                "same_owner_claim": tree.owner is owner
                                and tree.claim is settlement,
                                "same_activity": tree.activity
                                is rig.runtime._worktree_activity,
                                "owner_registered": owner.capacity._owners.get(
                                    owner.execution_id
                                )
                                is owner,
                                "claim_installed": getattr(
                                    tree.activity._claims.get(settlement.id),
                                    "custody",
                                    None,
                                )
                                is settlement,
                                "claim_return": any(
                                    c is settlement and durable is True
                                    for c, durable in tuple(claim_returns)
                                ),
                                "tree_registered": tree in rig.runtime._ordinary_turns,
                                "tree_retired": status(tree_retired),
                                "tree_return": any(
                                    t is tree and fut is tree_retired
                                    for t, fut in tuple(tree_returns)
                                ),
                                "tickets": [
                                    {
                                        "kind": operation.kind,
                                        "same_owner": ticket.owner is owner,
                                        "same_operation": ticket.operation is operation,
                                        "retired": ticket.retired.is_set(),
                                        "failed_write": ticket._failed_write,
                                        "ticket_return": any(
                                            t is ticket for t in tuple(ticket_returns)
                                        ),
                                        "operation_return": any(
                                            op is operation
                                            for op in tuple(operation_returns)
                                        ),
                                    }
                                    for ticket, (snapshot, operation) in known_tickets
                                    if snapshot.ordinary_tree is tree
                                ],
                            }
                        )
                ordinary["reads"].append(source_state)
            result["submissions"].append(ordinary)
        return result

    def failure_snapshot():
        prepared, run, driver = (
            state.get("prepared"),
            state.get("run"),
            state.get("driver"),
        )
        report = {
            "same_run_prepared": run is not None and run.prepared is prepared,
            "direct": direct,
            "same_driver": (driver is not None and driver is direct_driver)
            if direct
            else (run is not None and run.driver is driver),
            "same_registry": run is not None
            and rig.coordinator._runs.get(run.workflow_id) is run,
            "task_done": driver.done() if driver is not None else None,
            "task_cancelled": driver.cancelled() if driver is not None else None,
            "coroutines": [],
            "workers": [],
        }
        report["finalizer_retry_seen"] = "finalizer_retry" in state
        finalizer_handle = state.get("finalizer_handle")
        report["finalizer_handle_seen"] = finalizer_handle is not None
        report["finalizer_handle_entered"] = finalizer_handle is not None and any(
            item is finalizer_handle for item in state.get("handle_entries", [])
        )
        report["later_hold_reached"] = held.is_set()
        report["retries"] = [
            {"caller": caller.co_name, "held": was_held, "released": was_released}
            for _frame, caller, was_held, was_released in retries
        ]
        report["handles"] = [
            {
                "finalizer": handle is finalizer_handle,
                "entered": any(
                    item is handle for item in state.get("handle_entries", [])
                ),
                "released": was_released,
            }
            for handle, _caller, was_released in handles
        ]
        report["attempts"] = [
            {
                "same_children": row[1] == {child[3] for child in children},
                "same_token": row[2] is state.get("token"),
                "same_intake": row[3] is state.get("intake"),
                "timer": row[4] is not None,
                "active": row[5],
                "held": row[6],
                "released": row[7],
                "remaining": row[8],
                "custody": row[9],
                "position": row[10],
                "finalizer_attempt": row[0] in state.get("finalizer_attempts", []),
                "same_token_refusal": row[0] in state.get("same_token_refusals", []),
                "qualified_refusal": refusal_kind(row[0]),
                "priorities": [
                    {
                        "result": result,
                        "position": position,
                        "send_refusals": [
                            {"copy": copy, "position": return_position}
                            for _refusal, priority_caller, copy, return_position in send_refusals
                            if priority_caller is priority
                        ],
                    }
                    for priority, caller, result, position in priorities
                    if caller is row[0]
                ],
            }
            for row in attempts
        ]
        read = callback = None
        coroutine = driver.get_coro() if driver is not None else None
        for _ in range(32):
            if type(coroutine) is not types.CoroutineType:
                break
            frame = coroutine.cr_frame
            row = {
                "code": coroutine.cr_code.co_name,
                "file": coroutine.cr_code.co_filename,
                "line": frame.f_lineno if frame is not None else None,
            }
            if frame is not None and coroutine.cr_code is run_code:
                row["original_planning"] = (
                    frame.f_locals.get("self") is prepared
                    and frame.f_locals.get("_coordinator_run") is run
                    and frame.f_locals.get("driver") is driver
                )
            if (
                frame is not None
                and coroutine.cr_code is planning_reads.run_preparation_read.__code__
            ):
                local = frame.f_locals
                candidate, candidate_callback = local.get("read"), local.get("callback")
                if (
                    type(candidate) is planning_reads.ConsolePreparationRead
                    and type(candidate_callback) is types.FunctionType
                    and candidate_callback.__code__ is invoke_code
                    and local.get("creator") is rig.controller
                    and local.get("source") is prepared
                    and candidate.source is prepared
                    and candidate.creator is rig.controller
                    and candidate.task is driver
                    and candidate.callback is candidate_callback
                ):
                    read, callback = candidate, candidate_callback
                    row["original_read"] = True
            report["coroutines"].append(row)
            coroutine = coroutine.cr_await
        returned_read = prepared._planning_read_return if prepared is not None else None
        report["planning_read_captured"] = (
            type(returned_read) is tuple and len(returned_read) == 4
        )
        report["original_read_found"] = read is not None
        if read is not None:
            report["captured_read_matches"] = (
                type(returned_read) is tuple
                and len(returned_read) == 4
                and returned_read[0] is read
                and returned_read[1] is read._producer
                and returned_read[2] is read.retired
                and returned_read[3] is callback
            )
            for label, future in (
                ("producer", read._producer),
                ("retired", read.retired),
            ):
                report[label] = (
                    {"done": future.done(), "cancelled": future.cancelled()}
                    if type(future) is asyncio.Future
                    else "unknown"
                )
            reader_code = next(
                code
                for code in planning_reads.run_preparation_read.__code__.co_consts
                if isinstance(code, types.CodeType) and code.co_name == "invoke"
            )
            for thread_id, top in sys._current_frames().items():
                frame, labels, matches = top, [], []
                for _ in range(32):
                    if frame is None:
                        break
                    labels.append(
                        (frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno)
                    )
                    if (
                        frame.f_code is reader_code
                        and frame.f_locals.get("read") is read
                        and frame.f_locals.get("callback") is callback
                    ):
                        binding = frame.f_locals.get("binding")
                        matches.append(
                            {
                                "same_read": True,
                                "same_binding": type(binding)
                                is planning_reads._HookSourceBinding
                                and binding.read is read,
                                "same_thread": type(binding)
                                is planning_reads._HookSourceBinding
                                and binding.thread.ident == thread_id,
                                "task_none": type(binding)
                                is planning_reads._HookSourceBinding
                                and binding.task is None,
                                "active": binding.active
                                if type(binding) is planning_reads._HookSourceBinding
                                else None,
                            }
                        )
                    frame = frame.f_back
                if matches:
                    report["workers"].append(
                        {"thread": thread_id, "matches": matches, "stack": labels}
                    )
        print("PLANNING_WAKE_FAILURE_BOUNDARY", json.dumps(report), flush=True)
        print(
            "PLANNING_WAKE_ORDINARY_FAILURE_BOUNDARY",
            json.dumps(ordinary_failure()),
            flush=True,
        )

    def settled(event):
        events.append(event)

    def parent_body():
        try:
            original["outcome"] = participants.run_finite_local_worker(
                original_parent,
                bridge,
                rig.store,
                session,
                original["assistant"].id,
                conversation_id=conversation,
                resolution=resolution,
                expected_progress_owner_id=state["prepared"].native.progress_owner_id,
                agent_messages=[
                    {"role": "user", "content": provider.parent_instruction}
                ],
            )
        except BaseException as error:  # noqa: BLE001 - preserve genuine setup refusal.
            original["error"] = error

    def pre_root(frame):
        nonlocal parent
        prepared, run = frame.f_locals["self"], frame.f_locals["_coordinator_run"]
        driver = asyncio.current_task()
        if direct:
            assert run is None and prepared is direct_prepared
            assert driver is direct_driver and driver.get_coro().cr_frame is frame
            assert (
                not hasattr(rig.runtime, "_workflow_coordinator")
                and rig.coordinator is None
            )
        else:
            assert run.prepared is prepared and run.driver is driver
            assert rig.coordinator._runs.get(run.workflow_id) is run
        assert prepared.primary_owner is None
        token, intake = prepared._planning_token, prepared._planning_intake
        assert token is wake._automatic_primary_claims[session.id]
        assert intake is rig.store.received_turn_for_session(session.id)
        native_service = prepared.native.service
        fleet = native_service._injected_fleet_coordinator
        assert fleet is not None and bridge._fleet_coordinators[conversation] is fleet
        inbox = fleet.message_inbox
        assert native_service._message_inbox is inbox
        state.update(
            prepared=prepared,
            run=run,
            token=token,
            intake=intake,
            driver=driver,
            native_service=native_service,
            fleet=fleet,
            inbox=inbox,
        )
        driver.add_done_callback(driver_returned)
        original["assistant"] = rig.store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        deadline = time.monotonic() + 10
        parent = threading.Thread(target=parent_body, name="planning-wake-parent")
        parent.start()
        parent.join(max(0, deadline - time.monotonic()))
        assert not parent.is_alive(), "original parent did not return in setup10"
        if "error" in original:
            raise original["error"]
        assert original["outcome"].status == "done"
        service, owner = original["service"], original["owner"]
        assert (
            service._automatic_work is None and service.work_origin is WorkOrigin.MANUAL
        )
        assert service._work_chain_id != prepared.phase.attempt.chain_id
        assert service._fleet is service._injected_fleet_coordinator is fleet
        assert service._message_inbox is fleet.message_inbox is inbox
        assert bridge._fleet_coordinators[conversation] is fleet
        assert prepared.native.service is native_service
        assert native_service._injected_fleet_coordinator is fleet
        assert native_service._message_inbox is inbox
        drained = []
        owner.on_drained(drained.append)
        assert drained == [True] and not owner.is_admitted
        assert len(children) == 2
        for event in provider.child_entered.values():
            assert event.wait(max(0, deadline - time.monotonic())), (
                "original child provider did not enter within shared setup10"
            )
        for gate in provider.child_release.values():
            gate.set()
        for thread, target, child_owner, run_id in children:
            thread.join(max(0, deadline - time.monotonic()))
            assert not thread.is_alive(), "original child did not return in setup10"
            assert any(row[0] is thread and row[1] is target for row in child_returns)
            drained = []
            child_owner.on_drained(drained.append)
            assert drained == [True] and not child_owner.is_admitted
            assert child_owner.run_id == run_id
        with storage_admission._lock:
            assert not any(
                lease.resource_thread is parent
                or any(lease.resource_thread is row[0] for row in children)
                for lease in storage_admission._live_leases
            )
        child_ids = {row[3] for row in children}
        assert {event.child.run_id for event in events} == child_ids
        assert all(
            event.child.settled_after_turn and event.child.workflow_task is None
            for event in events
        )
        assert set(wake._pending[conversation]) == child_ids
        assert not provider.wake_calls and conversation not in wake._active
        state["setup_returned"] = True
        print("PLANNING_WAKE_ORIGINAL_PRE_ROOT_SETUP", sorted(child_ids), flush=True)

    def observe_delivery(frame, event, result):
        if "release_retry" not in state:
            return
        code, local = frame.f_code, frame.f_locals
        instance = local.get("self")
        if code is type(loop).create_task.__code__ and instance is loop:
            caller = frame.f_back
            if (
                caller.f_code is type(wake)._attempt.__code__
                and caller.f_locals.get("self") is wake
                and caller.f_locals.get("conversation_id") == conversation
                and getattr(local.get("coro"), "cr_code", None)
                is type(wake)._deliver.__code__
            ):
                if event == "call":
                    context = local["context"]
                    assert type(context) is contextvars.Context and not tuple(
                        context.items()
                    )
                    ancestor = caller
                    while (
                        ancestor is not None
                        and ancestor.f_code is not asyncio.Handle._run.__code__
                    ):
                        ancestor = ancestor.f_back
                    assert ancestor is not None
                    task_calls[frame] = (
                        local["coro"],
                        context,
                        ancestor.f_locals["self"],
                    )
                else:
                    coroutine, context, handle = task_calls.pop(frame)
                    assert returned(frame, event)
                    assert (
                        result.get_coro() is coroutine
                        and result.get_context() is context
                    )
                    wake_tasks.append((result, context, handle))
        elif code is type(rig.runtime)._submit_fleet_wake.__code__ and returned(
            frame, event
        ):
            if instance is rig.runtime and isinstance(result, str):
                authorization = local["wake_authorization"]
                delivery = asyncio.current_task()
                custody = rig.runtime._turn_custody[result]
                assert wake._delivery_tasks[delivery] == conversation
                assert wake._active[conversation].authorization is authorization
                assert current_automatic_work() is authorization.context
                assert (
                    custody.task.get_context().get(automatic_context)
                    is authorization.context
                )
                submissions.append(
                    (
                        authorization,
                        delivery,
                        custody,
                        custody.task,
                        custody.store,
                        custody.received_claim,
                    )
                )
        elif code is type(rig.runtime)._release_custody.__code__ and returned(
            frame, event
        ):
            if instance is rig.runtime:
                for _auth, _delivery, custody, task, store, claim in submissions:
                    if local.get("record") is custody:
                        assert local["turn_id"] == custody.turn_id
                        assert custody.turn_id not in rig.runtime._turn_custody
                        assert (
                            custody.task
                            is custody.store
                            is custody.received_claim
                            is None
                        )
                        assert (
                            claim is not None
                            and store.received_turn_for_session(custody.session_id)
                            is not claim
                        )
                        custody_returns.append((custody, task))
        elif (
            code is _AgentSources.prepare.__code__
            and event == "call"
            and instance.controller is rig.controller
        ):
            binding = planning_reads._source_binding.get()
            if binding is not None:
                read = binding.read
                for (
                    authorization,
                    _delivery,
                    _custody,
                    task,
                    _store,
                    _claim,
                ) in submissions:
                    if read.task is task:
                        assert (
                            binding.active
                            and binding.thread is threading.current_thread()
                        )
                        assert (
                            read.creator is rig.controller
                            and read.callback.__self__ is instance
                        )
                        assert read.callback.__func__ is _AgentSources.prepare
                        assert frame.f_back.f_locals.get("callback") is read.callback
                        assert _guard.context.get() is None
                        assert current_automatic_work() is authorization.context
                        source_reads.append(
                            (instance, read, read._producer, read.retired, task)
                        )
        elif returned(frame, event):
            if code is _OrdinaryTurn.admit.__code__ and any(
                row[0] is instance.source for row in source_reads
            ):
                trees[instance] = (
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
                and instance.ordinary_tree in trees
            ):
                with instance._holds_lock:
                    for ticket in instance._holds.values():
                        tickets[ticket] = (instance, ticket.operation)
            elif (
                code is _SnapshotTicket.returned.__code__
                and instance in tickets
                and instance.retired.is_set()
            ):
                ticket_returns.append(instance)
            elif code is OwnedOperation.finish.__code__ and any(
                instance.owner is item[1] for item in trees.values()
            ):
                operation_returns.append(instance)
            elif code is WorktreeActivitySettlement.settle.__code__ and any(
                instance is item[2] for item in trees.values()
            ):
                claim_returns.append((instance, local["durable"]))
            elif (
                code is _OrdinaryTurn._retire_scratch.__code__
                and instance in trees
                and local.get("elected") is True
            ):
                tree_returns.append((instance, instance.retired))

    def adverse_receipts(frame, event, result):
        if (
            not (notify_failure or metadata_failure)
            or not returned(frame, event)
            or "prepared" not in state
        ):
            return
        code, run, prepared = frame.f_code, state["run"], state["prepared"]
        if code is received_code and frame.f_locals.get("self") is prepared:
            caller = frame.f_back
            assert caller.f_code is g_code and caller.f_locals["run"] is run
            assert (
                asyncio.current_task() is run.driver
                and result is prepared._planning_release_return
            )
            caught = frame.f_locals["_cancellation"]
            assert result[5] is caught and "g_received" not in state
            if cancel_failure:
                assert (
                    type(caught) is asyncio.CancelledError
                    and caller.f_locals["error"] is caught
                )
                assert caller.f_locals["planning_returned"] is False
            else:
                assert caught is None and caller.f_locals["planning_returned"] is True
            state["g_received"] = result, caught
        elif (
            code is snapshot_body
            and (mode == "notify-error" or metadata_failure)
            and frame.f_locals.get("self") is rig.service
        ):
            cell = run.final_metadata
            if cell is None or frame.f_locals.get("_accounting") is not cell:
                return
            assert (
                cell.run is run
                and cell.issuing is run.driver
                and not cell.disposed.done()
            )
            read, producer, retired, callback = cell.proof
            ancestor = frame.f_back
            while ancestor is not None and not (
                ancestor.f_code is reader_code
                and ancestor.f_locals.get("read") is read
                and ancestor.f_locals.get("callback") is callback
            ):
                ancestor = ancestor.f_back
            assert (
                ancestor is not None
                and read.creator is rig.controller
                and read.source is rig.coordinator
            )
            assert (
                read.task is run.driver
                and read.callback is callback
                and read._producer is producer
            )
            assert (
                not producer.done() and not retired.done() and read.retired is retired
            )
            assert prepared._planning_release_return is state["g_received"][0]
            assert rig.coordinator._runs.get(run.workflow_id) is run and not state.get(
                "release_retry"
            )
            assert (
                not provider.wake_calls
                and not deliveries
                and "metadata_hold" not in state
            )
            state["metadata_hold"] = cell, read, producer, retired
            if metadata_failure:
                state["metadata_reader"] = ancestor
            notify_change()
            if not metadata_gate.wait(max(0, state["driver_until"] - loop.time())):
                raise TimeoutError(
                    "original planning metadata hold exceeded shared driver10"
                )

    def shutdown_host_ledger():
        assert threading.current_thread() is threading.main_thread()
        assert (
            asyncio.current_task() is not state["run"].driver
            and asyncio.current_task() is not close_task
        )
        assert (
            not rig.runtime._disposed
            and not rig.runtime.execution_capacity.snapshot().closed
        )
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            chain = original["service"]._work_chain_id
            return (
                tuple(
                    conn.execute(
                        "SELECT id FROM automatic_wake_attempts WHERE chain_id=?",
                        (chain,),
                    )
                ),
                tuple(
                    conn.execute(
                        "SELECT id FROM automatic_work_reservations WHERE chain_id=? AND kind IN ('generation','model_call')",
                        (chain,),
                    )
                ),
            )

    def shutdown_receipts(frame, event, result):
        if not shutdown or close_task is None:
            return
        prepared, run = state["prepared"], state["run"]
        code, loc, caller = frame.f_code, frame.f_locals, frame.f_back
        if (
            code is type(wake).dispose.__code__
            and returned(frame, event)
            and loc.get("self") is wake
            and caller is close_task.get_coro().cr_frame
        ):
            assert (
                asyncio.current_task() is close_task
                and caller.f_code is type(rig.runtime).dispose.__code__
            )
            assert (
                caller.f_locals["recovery_owner"] is wake
                and wake._disposed
                and rig.runtime._workflow_closing
            )
            assert (
                not wake._pending
                and prepared._planning_token
                is wake._automatic_primary_claims[session.id]
                is state["token"]
            )
            assert (
                not prepared.primary_owner._root_finished
                and prepared._planning_release_return is None
            )
            state["fence_return"] = frame
        if (
            code is _PreparedWorkflowReply._notify_planning_release.__code__
            and event == "call"
            and loc.get("self") is prepared
        ):
            assert (
                caller.f_code is g_code
                and caller is run.driver.get_coro().cr_frame
                and asyncio.current_task() is run.driver
            )
            assert (
                loc["returned"] is prepared._planning_release_return
                and len(loc["returned"]) == 6
            )
            assert (
                loc["returned"][0] is prepared._planning_release_binding
                and loc["returned"][1] is prepared._planning_read_return
            )
            assert (
                loc["_publication"] is run.final_metadata
                and loc["_popped_run"] is caller.f_locals["popped_run"] is run
            )
            assert (
                run.final_metadata.disposed.result() is True
                and run.final_metadata not in rig.coordinator._metadata_cells
            )
            assert (
                run.workflow_id not in rig.coordinator._runs
                and prepared not in rig.runtime._workflow_preparations
            )
            assert (
                prepared._planning_token
                is prepared._planning_intake
                is prepared._source_scope
                is None
            )
            assert (
                prepared.primary_claim.id not in rig.runtime._worktree_activity._claims
                and prepared.primary_owner._root_finished
            )
            assert rig.runtime._workflow_closing and wake._disposed is (
                not before_fence
            )
            assert set(wake._pending.get(conversation, {})) == (
                {row[3] for row in children} if before_fence else set()
            )
            assert (
                not provider.wake_calls
                and not deliveries
                and not submissions
                and not wake._delivery_tasks
            )
            assert "shutdown_notify" not in state
            state["shutdown_notify"] = frame
        elif code is type(wake).retry_soon.__code__ and caller is state.get(
            "shutdown_notify"
        ):
            assert not before_fence and wake._disposed and loc["self"] is wake
            if event == "call":
                assert "inert_call" not in state
                state["inert_call"] = frame
            elif returned(frame, event):
                assert frame is state["inert_call"] and result is None
                state["inert_return"] = frame

    def record(frame, event, result):
        nonlocal local_mode, global_attempted
        code, local = frame.f_code, frame.f_locals
        shutdown_receipts(frame, event, result)
        if (
            metadata_failure
            and code is _PreparedWorkflowReply._notify_planning_release.__code__
            and event == "call"
        ):
            state.setdefault("notifier_entries", []).append(frame)
        adverse_receipts(frame, event, result)
        observe_delivery(frame, event, result)
        current = threading.current_thread()
        if event == "call" and code is run_code and not state:
            pre_root(frame)
            return
        if current is parent:
            if event == "call" and code is threading.Thread.start.__code__:
                caller = frame.f_back
                if (
                    caller.f_code is AgentService._launch_prepared_child_result.__code__
                    and caller.f_locals.get("self") is original.get("service")
                    and caller.f_locals.get("fleet") is state["fleet"]
                    and "child_owner" in caller.f_locals
                ):
                    thread = local["self"]
                    target = thread._target
                    children.append(
                        (
                            thread,
                            target,
                            caller.f_locals["child_owner"],
                            caller.f_locals["child_run_id"],
                        )
                    )
                    codes.add(target.__code__)
            if returned(frame, event):
                if code is AgentService.__init__.__code__:
                    original["service"] = local["self"]
                elif code is AgentService._run_owned.__code__ and (
                    local["self"] is original.get("service")
                ):
                    original["owner"] = local["owner"]
        if returned(frame, event):
            for thread, target, _owner, _run_id in children:
                if current is thread and code is target.__code__:
                    child_returns.append((thread, target))
        if "prepared" not in state:
            return
        prepared = state["prepared"]
        if code is invoke_code and local.get("self") is prepared:
            if event == "call":
                assert "invocation" not in state
                caller = frame.f_back
                assert caller is not None and caller.f_code is reader_code
                binding, read, callback = (
                    caller.f_locals["binding"],
                    caller.f_locals["read"],
                    caller.f_locals["callback"],
                )
                assert type(binding) is planning_reads._HookSourceBinding
                assert type(read) is planning_reads.ConsolePreparationRead
                assert binding.read is read and binding.thread is current
                assert binding.process == os.getpid() and binding.task is None
                assert binding.active and read.callback is callback
                assert callback.__code__ is invoke_code
                assert read.source is prepared and read.creator is rig.controller
                assert read.task is state["driver"]
                assert type(read._producer) is asyncio.Future
                assert type(read.retired) is asyncio.Future
                state["invocation"] = (
                    frame,
                    caller,
                    current,
                    binding,
                    read,
                    callback,
                    read._producer,
                    read.retired,
                    prepared.primary_owner,
                    prepared.primary_claim,
                )
            else:
                assert state["invocation"][0] is frame
                state["invocation_returned"] = True
            return
        if (
            code is concurrent.futures.Future.result.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is invoke_code
            and frame.f_back.f_locals.get("self") is prepared
        ):
            if event == "call":
                assert "finalizer_result_call" not in state
                state["finalizer_result_call"] = frame, frame.f_back, local["self"]
                return
            result_frame, invoke, future = state.pop("finalizer_result_call")
            assert result_frame is frame and invoke is frame.f_back
            if not returned(frame, event):
                return
            read, producer, retired, callback = prepared._planning_read_return
            assert callback.__code__ is invoke_code and read.callback is callback
            assert read.task is state["driver"] and read.source is prepared
            assert read.creator is rig.controller
            assert read._producer is producer and read.retired is retired
            assert not producer.done() and not retired.done()
            assert invoke.f_locals["owner"] is prepared.primary_owner
            assert future.done()
            native_service = state["native_service"]
            assert prepared.native.service is native_service
            assert (
                native_service._fleet
                is native_service._injected_fleet_coordinator
                is state["fleet"]
            )
            assert (
                native_service._message_inbox
                is state["fleet"].message_inbox
                is state["inbox"]
            )
            assert bridge._fleet_coordinators[conversation] is state["fleet"]
            finalizer = state["finalizer_retry"]
            assert finalizer[1] is invoke.f_locals["outcome"]
            assert finalizer[2] == invoke.f_locals["run_id"]
            assert invoke is state["invocation"][0]
            assert all(original_custody().values())
            state.update(finalizer_future=future, held_invoke=invoke)
            held.set()
            hold_deadline = time.monotonic() + 5
            notify_change()
            if not release.wait(max(0, hold_deadline - time.monotonic())):
                errors.append(TimeoutError("original planning finalizer hold5 expired"))
                notify_change()
            return
        if (
            code is type(wake).retry_soon.__code__
            and event == "call"
            and (local.get("self") is wake)
        ):
            caller = frame.f_back
            retries.append((frame, caller.f_code, held.is_set(), release.is_set()))
            if (
                caller.f_code
                is _PreparedWorkflowReply._notify_planning_release.__code__
            ):
                if shutdown:
                    assert (
                        caller is state["shutdown_notify"]
                        and frame is state["inert_call"]
                    )
                    return
                run = state["run"]
                notification_caller = caller.f_back
                returned_release = caller.f_locals["returned"]
                assert (
                    "release_retry" not in state and held.is_set() and release.is_set()
                )
                assert caller.f_locals["self"] is prepared
                assert returned_release is prepared._planning_release_return
                assert (
                    caller.f_locals["binding"]
                    is returned_release[0]
                    is prepared._planning_release_binding
                )
                assert returned_release[1] is prepared._planning_read_return
                if direct:
                    assert run is None and caller.f_locals["_publication"] is None
                    assert caller.f_locals["_popped_run"] is None
                    assert notification_caller is returned_release[0][5]
                    assert notification_caller.f_code is run_code
                    assert (
                        notification_caller.f_locals["release_return"]
                        is returned_release
                    )
                    assert notification_caller.f_locals["_coordinator_run"] is None
                    assert returned_release[0][6] is returned_release[0][7] is None
                    assert (
                        asyncio.current_task()
                        is direct_driver
                        is returned_release[0][3]
                    )
                    assert (
                        not hasattr(rig.runtime, "_workflow_coordinator")
                        and rig.coordinator is None
                    )
                else:
                    assert caller.f_locals["_publication"] is run.final_metadata
                    assert caller.f_locals["_popped_run"] is run
                    assert notification_caller.f_code is g_code
                    assert notification_caller.f_locals["popped_run"] is run
                    assert asyncio.current_task() is run.driver
                    assert run.workflow_id not in rig.coordinator._runs
                    assert (
                        run.final_metadata.disposed.done()
                        and run.final_metadata.disposed.result() is True
                    )
                assert (
                    prepared._source_scope
                    is prepared._planning_intake
                    is prepared._planning_token
                    is None
                )
                assert not prepared.primary_owner.is_admitted
                assert (
                    prepared.primary_claim.id
                    not in rig.runtime.worktree_activity._claims
                )
                pending = set(wake._pending.get(conversation, {}))
                assert pending == {row[3] for row in children}
                assert conversation not in wake._active and wake._retry_timer is None
                assert not deliveries and not provider.wake_calls
                assert monitor_owned and not local_mode
                assert not (
                    task_calls
                    or wake_tasks
                    or submissions
                    or source_reads
                    or trees
                    or tickets
                )
                state["release_retry"] = frame, returned_release, pending
                global_attempted = True
                monitor.set_events(
                    tool_id, event_types.PY_THROW | event_types.PY_UNWIND
                )
                local_mode = True
                assert (
                    profile_attempted
                    and sys.getprofile() is observe
                    and threading.getprofile() is observe
                )
                threading.setprofile_all_threads(None)
                assert sys.getprofile() is None and threading.getprofile() is None
                print(
                    "PLANNING_WAKE_ORIGINAL_RELEASE_NOTIFICATION",
                    sorted(pending),
                    flush=True,
                )
            if (
                caller.f_code is type(rig.controller)._set_run_state.__code__
                and caller.f_locals.get("self") is rig.controller
                and caller.f_locals.get("target") == session.id
            ):
                ancestor = caller.f_back
                native_finalizer = None
                while ancestor is not None:
                    if (
                        ancestor.f_code
                        is type(rig.controller)._finalize_agent_reply.__code__
                        and ancestor.f_locals.get("self") is rig.controller
                        and ancestor.f_locals.get("session_id") == session.id
                    ):
                        native_finalizer = ancestor
                    if (
                        ancestor.f_code is finalize_code
                        and ancestor.f_locals.get("self") is prepared
                    ):
                        assert native_finalizer is not None
                        assert "finalizer_retry" not in state
                        state["finalizer_run_state"] = caller.f_locals["run_state"]
                        assert (
                            rig.controller._run_states.get(session.id)
                            is state["finalizer_run_state"]
                        )
                        state["finalizer_retry"] = (
                            frame,
                            native_finalizer.f_locals["outcome"],
                            native_finalizer.f_locals["run_id"],
                        )
                        break
                    ancestor = ancestor.f_back
        elif code is type(loop).call_soon_threadsafe.__code__ and returned(
            frame, event
        ):
            callback = local.get("callback")
            if getattr(callback, "__self__", None) is wake and (
                getattr(callback, "__func__", None) is type(wake)._attempt_all
            ):
                handles.append((result, frame.f_back, release.is_set()))
                if frame.f_back is state.get("finalizer_retry", (None,))[0]:
                    state["finalizer_handle"] = result
                if frame.f_back is state.get("release_retry", (None,))[0]:
                    assert "release_handle" not in state
                    state["release_handle"] = result
        elif code is type(wake)._attempt_all.__code__ and event == "call":
            caller = frame.f_back
            if (
                local.get("self") is wake
                and caller.f_code is asyncio.Handle._run.__code__
            ):
                state.setdefault("handle_entries", []).append(caller.f_locals["self"])
        elif code is type(wake)._attempt.__code__ and returned(frame, event):
            if (
                local.get("self") is wake
                and local.get("conversation_id") == conversation
            ):
                attempts.append(
                    (
                        frame,
                        set(wake._pending.get(conversation, {})),
                        wake._automatic_primary_claims.get(session.id),
                        rig.store.received_turn_for_session(session.id),
                        wake._retry_timer,
                        conversation in wake._active,
                        held.is_set(),
                        release.is_set(),
                        local.get("remaining"),
                        original_custody(),
                        (frame.f_lasti, frame.f_lineno),
                    )
                )
                ancestor = frame.f_back
                while (
                    ancestor is not None
                    and ancestor.f_code is not asyncio.Handle._run.__code__
                ):
                    ancestor = ancestor.f_back
                if ancestor is not None and ancestor.f_locals["self"] is state.get(
                    "finalizer_handle"
                ):
                    state.setdefault("finalizer_attempts", []).append(frame)
                if ancestor is not None and ancestor.f_locals["self"] is state.get(
                    "release_handle"
                ):
                    state.setdefault("release_attempts", []).append(frame)
                notify_change()
        elif code is type(wake)._priority_allows.__code__ and returned(frame, event):
            caller = frame.f_back
            if (
                local.get("self") is wake
                and local.get("session_id") == session.id
                and caller.f_code is type(wake)._attempt.__code__
                and caller.f_locals.get("self") is wake
                and caller.f_locals.get("conversation_id") == conversation
            ):
                priorities.append(
                    (frame, caller, result, (frame.f_lasti, frame.f_lineno))
                )
        elif code is type(rig.controller).send_refusal_copy.__code__ and returned(
            frame, event
        ):
            caller = frame.f_back
            if (
                local.get("self") is rig.controller
                and local.get("session_id") == session.id
                and caller.f_code is type(wake)._priority_allows.__code__
                and caller.f_locals.get("self") is wake
                and caller.f_locals.get("session_id") == session.id
            ):
                send_refusals.append(
                    (frame, caller, result, (frame.f_lasti, frame.f_lineno))
                )
        elif code is type(rig.controller).run_state_for.__code__ and returned(
            frame, event
        ):
            caller = frame.f_back
            if (
                local.get("self") is rig.controller
                and local.get("session_id") == session.id
                and caller.f_code is type(rig.controller).send_refusal_copy.__code__
                and caller.f_locals.get("self") is rig.controller
                and caller.f_locals.get("session_id") == session.id
            ):
                state_results.append((caller, result))
        elif code is type(wake).try_claim_automatic_primary.__code__ and returned(
            frame, event
        ):
            if (
                local.get("self") is wake
                and result is False
                and (local.get("existing") is state["token"])
            ):
                state.setdefault("same_token_refusals", []).append(frame.f_back)
        elif code is type(wake)._deliver.__code__ and event == "call":
            task = asyncio.current_task()
            if local.get("self") is wake and task not in deliveries:
                assert _guard.context.get() is None and current_automatic_work() is None
                assert (
                    current_execution_owner() is None
                    and current_task_authorization() is None
                )
                assert current_workflow_preparation() is None
                deliveries.append(task)
        elif returned(frame, event):
            if (
                code is ExecutionOwner.finish_root.__code__
                and local.get("self") is prepared.primary_owner
                or code is WorktreeActivitySettlement.settle.__code__
                and local.get("self") is prepared.primary_claim
                or code
                in {
                    _PreparedWorkflowReply._release_configuration.__code__,
                    _PreparedWorkflowReply._release_planning_slots.__code__,
                    run_code,
                }
                and local.get("self") is prepared
                or code is type(wake).release_automatic_primary.__code__
                and local.get("self") is wake
                and local.get("token") is state["token"]
                or code is type(rig.store).release_received_turn.__code__
                and local.get("self") is rig.store
                and local.get("claim") is state["intake"]
                or code is workflow_runs._metadata_publication_return.__code__
                and local.get("run") is state["run"]
                or code is g_code
                and local.get("run") is state["run"]
            ):
                physical.append((code, result, dict(local)))

    def record_checked(frame, event, result):
        if frame.f_code not in codes or event not in {"call", "return", "yield"}:
            return
        try:
            record(frame, event, result)
        except BaseException as error:  # noqa: BLE001 - observers cannot change native return.
            errors.append(error)
            try:
                notify_change()
            except BaseException as notification_error:  # noqa: BLE001 - observer notifications cannot alter native return.
                errors.append(notification_error)

    def observe(frame, event, result):
        if not local_mode:
            record_checked(frame, event, result)
            if (
                metadata_failure
                and frame is state.get("metadata_reader")
                and returned(frame, event)
            ):
                try:
                    cell, read, producer, retired = state["metadata_hold"]
                    callback = cell.proof[3]
                    assert (
                        frame.f_code is reader_code
                        and frame.f_locals["read"] is read
                        and frame.f_locals["callback"] is callback
                    )
                    assert (
                        read.source is rig.coordinator
                        and read.task is state["run"].driver
                    )
                    assert (
                        read._producer is producer
                        and not producer.done()
                        and read.retired is retired
                        and not retired.done()
                    )
                    assert (
                        not frame.f_locals["binding"].active and metadata_gate.is_set()
                    )
                    assert (
                        state["run"].final_metadata is cell and not cell.disposed.done()
                    )
                    assert "metadata_fault" not in state and not errors
                    state["metadata_fault"] = frame
                except BaseException as observer_error:  # noqa: BLE001 - keep diagnostic failure distinct.
                    errors.append(observer_error)
                    notify_change()
                else:
                    # Deliberate only for the exact original reader normal return.
                    raise metadata_error

    def adapter(callback):
        def observed(code, offset, arg=None):
            if (
                before_fence
                and not local_mode
                and callback is post_call
                and code is type(wake).dispose.__code__
                and close_task is not None
                and "fence_fault" not in state
            ):
                try:
                    frame = sys._getframe(1)
                    caller = frame.f_back
                    if (
                        caller is not close_task.get_coro().cr_frame
                        or caller.f_code is not type(rig.runtime).dispose.__code__
                    ):
                        return
                    assert (
                        frame.f_code is code
                        and frame.f_locals["self"] is wake
                        and asyncio.current_task() is close_task
                    )
                    assert (
                        caller.f_locals["self"] is rig.runtime
                        and caller.f_locals["recovery_owner"] is wake
                    )
                    assert (
                        rig.runtime._workflow_closing
                        and not wake._disposed
                        and not rig.runtime._disposed
                    )
                    assert (
                        held.is_set()
                        and not release.is_set()
                        and state["prepared"]._planning_release_return is None
                    )
                    assert set(wake._pending[conversation]) == {
                        row[3] for row in children
                    }
                    state["fence_fault"] = frame, caller
                except BaseException as observation_error:  # noqa: BLE001 - observation errors cannot become the intended fault.
                    errors.append(observation_error)
                    notify_change()
                    return
                # Deliberate PY_START fault before original wake.dispose can latch.
                raise fence_error
            if not local_mode or id(code) not in post_code_ids:
                return
            try:
                frame = sys._getframe(1)
                assert frame.f_code is code
                deliberate = callback(frame, code, offset, arg) is schedule_error
            except BaseException as observer_error:  # noqa: BLE001 - no monitoring error may affect native return.
                errors.append(observer_error)
                try:
                    notify_change()
                except BaseException as signal_error:  # noqa: BLE001 - retain diagnostic error separately.
                    errors.append(signal_error)
                return
            if deliberate:
                # Sole explicit test fault; original call did not post a Handle.
                raise schedule_error

        return observed

    def post_call(frame, code, offset, arg):
        if (
            notify_failure
            and code is type(loop).call_soon_threadsafe.__code__
            and frame.f_back is state["release_retry"][0]
        ):
            caller = frame.f_back
            notify = caller.f_back
            run, prepared = state["run"], state["prepared"]
            callback = frame.f_locals["callback"]
            assert frame.f_locals["self"] is loop and caller.f_locals["self"] is wake
            assert (
                callback.__self__ is wake
                and callback.__func__ is type(wake)._attempt_all
            )
            assert (
                notify.f_code
                is _PreparedWorkflowReply._notify_planning_release.__code__
            )
            assert (
                notify.f_locals["self"] is prepared
                and notify.f_locals["returned"] is state["g_received"][0]
            )
            assert (
                asyncio.current_task() is run.driver
                and notify.f_locals["_popped_run"] is run
            )
            assert (
                notify.f_locals["_publication"] is run.final_metadata
                and run.final_metadata.disposed.result() is True
            )
            assert (
                run.workflow_id not in rig.coordinator._runs
                and "scheduling_fault" not in state
            )
            assert set(wake._pending[conversation]) == {row[3] for row in children}
            state["scheduling_fault"] = frame, caller, notify
            return schedule_error
        if frame is not state["release_retry"][0]:
            record_checked(frame, "call", arg)

    def post_return(frame, code, offset, arg):
        assert frame.f_lasti == offset
        assert dis.opname[code.co_code[offset]] in {"RETURN_CONST", "RETURN_VALUE"}
        if frame is state["release_retry"][0]:
            assert code is type(wake).retry_soon.__code__ and arg is None
            state["release_retry_return"] = frame
        record_checked(frame, "return", arg)

    def post_yield(frame, code, offset, arg):
        record_checked(frame, "yield", arg)

    def post_unwind(frame, code, offset, exc):
        if (
            notify_failure
            and code is g_code
            and frame.f_locals.get("run") is state["run"]
        ):
            assert asyncio.current_task() is state["run"].driver
            assert state["run"].driver.get_coro().cr_frame is frame
            assert frame.f_locals["release_return"] is state["g_received"][0]
            state["inner_unwind"] = exc, frame.f_locals["cancellation"], exc.__cause__
        observations = state.setdefault("post_unwinds", [])
        if len(observations) < 32:
            observations.append((code.co_name, offset, type(exc).__name__))
        if frame in task_calls:
            task_calls.pop(frame)
            raise AssertionError("original wake create_task unwound") from exc

    callbacks = {
        event_types.PY_START: adapter(post_call),
        event_types.PY_RESUME: adapter(post_call),
        event_types.PY_THROW: adapter(post_call),
        event_types.PY_RETURN: adapter(post_return),
        event_types.PY_YIELD: adapter(post_yield),
        event_types.PY_UNWIND: adapter(post_unwind),
    }
    local_events = (
        event_types.PY_START
        | event_types.PY_RESUME
        | event_types.PY_RETURN
        | event_types.PY_YIELD
    )
    bridge.on_fleet_child_settled("planning-wake-original-observer", settled)
    try:
        assert sys.getprofile() is None and threading.getprofile() is None
        monitor.use_tool_id(tool_id, "planning-wake-native-delivery")
        monitor_owned = True
        assert monitor.get_events(tool_id) == 0
        assert all(monitor.get_local_events(tool_id, code) == 0 for code in post_codes)
        for event, callback in callbacks.items():
            previous = monitor.register_callback(tool_id, event, callback)
            saved_callbacks[event] = previous
            assert previous is None
        for code in post_codes:
            armed_codes.append(code)
            monitor.set_local_events(tool_id, code, local_events)
        profile_attempted = True
        threading.setprofile_all_threads(observe)
        if direct:
            assert rig.coordinator is None and not hasattr(
                rig.runtime, "_workflow_coordinator"
            )
            binding = capture_run_admitted_workspace_roots(
                session=session, registry=rig.service.persistence.workspace_registry
            )[0]
            request = rig.workflow.request
            direct_prepared = await prepare_workflow_planning(
                rig.controller,
                session_id=session.id,
                binding=binding,
                command_id="direct-planning-wake",
                objective=request.objective,
                policy=request.policy,
                _request=request,
            )
            assert loop.get_task_factory() is None
            direct_driver = loop.create_task(direct_prepared.run_planning())
            # The direct issuer first enters its own synchronous setup10. Start
            # arrival10 only after this actual pre-root setup has returned.
            await asyncio.sleep(0)
            check()
            if not state.get("setup_returned"):
                if direct_driver.done():
                    direct_driver.result()
                raise AssertionError(
                    "direct driver did not return its original setup on first step"
                )
            assert (
                state["driver"] is direct_driver
                and state["prepared"] is direct_prepared
            )
            assert state["run"] is None and not hasattr(
                rig.runtime, "_workflow_coordinator"
            )
        else:
            rig.workflow = await rig.coordinator.start(
                rig.workflow.request, launch_id=rig.workflow.launch_id
            )
        async with asyncio.timeout(10):
            while True:
                changed.clear()
                check()
                if held.is_set():
                    break
                if state.get("driver") is not None and state["driver"].done():
                    await asyncio.shield(state["driver"])
                    raise AssertionError("original planning finalizer hold not reached")
                await changed.wait()
        check()
        prepared, run = state["prepared"], state["run"]
        child_ids = {row[3] for row in children}
        assert state["setup_returned"]
        async with asyncio.timeout(4):
            while True:
                changed.clear()
                check()
                if any(
                    row[1] == child_ids
                    and row[2] is state["token"]
                    and row[3] is state["intake"]
                    and row[4] is None
                    and not row[5]
                    and not row[7]
                    and row[8] is not None
                    and row[8] <= 0
                    and all(row[9].values())
                    and row[0] in state.get("finalizer_attempts", [])
                    and refusal_kind(row[0]) is not None
                    for row in attempts
                ):
                    break
                await changed.wait()
        assert any(
            caller is type(rig.controller)._set_run_state.__code__
            for _frame, caller, _held, _released in retries
        )
        assert handles and all(
            handle in state.get("handle_entries", [])
            for handle, _caller, _after in handles
        )
        assert wake._retry_timer is None and not provider.wake_calls
        assert state["finalizer_handle"] in state["handle_entries"]
        assert state["finalizer_attempts"]
        assert prepared.primary_owner.is_admitted
        assert not prepared._planning_read_return[1].done()
        print(
            "PLANNING_WAKE_ORIGINAL_FINALIZER_RETRY_CONSUMED_AND_NATIVE_REFUSED",
            len(retries),
            len(handles),
            flush=True,
        )
        if shutdown:
            assert loop.get_task_factory() is None
            assert (
                not rig.runtime._workflow_closing
                and not wake._disposed
                and set(wake._pending[conversation]) == child_ids
            )
            close_task = loop.create_task(rig.runtime.dispose())
            state["close_task"] = close_task
            await asyncio.sleep(0)
            check()
            assert (
                not close_task.done() and not run.driver.done() and not release.is_set()
            )
            assert (
                rig.runtime._workflow_closing
                and rig.coordinator._closing
                and run.control == "stop"
                and prepared.cancel_event.is_set()
            )
            assert (
                not rig.runtime._disposed
                and not rig.runtime.execution_capacity.snapshot().closed
            )
            assert (
                prepared.primary_owner.is_admitted
                and prepared._planning_release_return is None
            )
            assert (
                prepared._planning_token
                is wake._automatic_primary_claims[session.id]
                is state["token"]
            )
            assert (
                prepared._planning_intake
                is rig.store.received_turn_for_session(session.id)
                is state["intake"]
            )
            if before_fence:
                assert (
                    "fence_fault" in state
                    and "fence_return" not in state
                    and not wake._disposed
                )
                assert set(wake._pending[conversation]) == child_ids
            else:
                assert (
                    "fence_return" in state
                    and "fence_fault" not in state
                    and wake._disposed
                    and not wake._pending
                )
            assert shutdown_host_ledger() == ((), ())
            print("PLANNING_SHUTDOWN_ORIGINAL_F_HELD", mode, flush=True)
            release.set()
            async with asyncio.timeout(10):
                if before_fence:
                    with pytest.raises(ValueError) as disposal:
                        await asyncio.shield(close_task)
                    assert disposal.value is close_task.exception() is fence_error
                    state["first_fence_error_returned"] = True
                else:
                    assert await asyncio.shield(close_task) is None
            check()
            assert (
                run.driver.done()
                and not run.driver.cancelled()
                and "shutdown_notify" in state
            )
            assert "release_handle" not in state and not any(
                caller is state.get("inert_call")
                for _handle, caller, _released in handles
            )
            assert (
                run.workflow_id not in rig.coordinator._runs
                and run.final_metadata.disposed.result() is True
            )
            read, producer, retired, callback = prepared._planning_release_return[1]
            assert (
                read.task is run.driver
                and read.source is prepared
                and read.callback is callback
            )
            assert (
                read._producer is producer
                and producer.done()
                and not producer.cancelled()
                and producer.exception() is None
            )
            assert (
                read.retired is retired and retired.done() and retired.result() is None
            )
            drained = []
            prepared.primary_owner.on_drained(drained.append)
            assert (
                drained == [True]
                and prepared.primary_claim.id
                not in rig.runtime._worktree_activity._claims
            )
            assert (
                prepared._planning_token
                is prepared._planning_intake
                is prepared._source_scope
                is None
            )
            assert (
                prepared not in rig.runtime._workflow_preparations
                and wake._automatic_primary_claims.get(session.id) is not state["token"]
            )
            assert (
                not deliveries
                and not wake_tasks
                and not submissions
                and not provider.wake_calls
                and not wake._delivery_tasks
            )
            if before_fence:
                failure = run.driver.exception()
                assert (
                    type(failure) is PermissionError
                    and str(failure) == "planning wake shutdown fence unproven"
                )
                tb = failure.__traceback__
                while tb.tb_next is not None:
                    tb = tb.tb_next
                assert (
                    tb.tb_frame is state["shutdown_notify"]
                    and "inert_call" not in state
                )
                assert (
                    not wake._disposed and set(wake._pending[conversation]) == child_ids
                )
                assert (
                    not rig.runtime._disposed
                    and not rig.runtime.execution_capacity.snapshot().closed
                )
                assert shutdown_host_ledger() == ((), ())
                print(
                    "PLANNING_SHUTDOWN_ORIGINAL_FENCE_ERROR_AND_POST_POP_REFUSAL",
                    flush=True,
                )
            else:
                assert (
                    run.driver.exception() is None
                    and state["inert_call"] is state["inert_return"]
                )
                assert (
                    wake._disposed
                    and not wake._pending
                    and rig.runtime._disposed
                    and rig.runtime.execution_capacity.snapshot().closed
                )
                print(
                    "PLANNING_SHUTDOWN_ORIGINAL_DISPOSED_RETRY_INERT_NO_ADMISSION",
                    flush=True,
                )
            check()
            return
        if notify_failure or metadata_failure:
            # Calling the live G-associated P as if it were direct cannot acquire another root.
            original_binding = prepared._planning_release_binding
            with pytest.raises(
                PermissionError, match="original G planning association required"
            ):
                await prepared.run_planning()
            assert (
                prepared._planning_release_binding is original_binding
                and prepared._planning_release_return is None
            )
            assert (
                run.prepared is prepared
                and rig.coordinator._runs.get(run.workflow_id) is run
            )
            assert (
                prepared.primary_owner.is_admitted
                and prepared._planning_token is state["token"]
            )
            assert (
                prepared._planning_intake is state["intake"] and not provider.wake_calls
            )
            state["driver_until"] = loop.time() + 10
            if cancel_failure:
                assert run.driver.cancel("original planning caller cancellation")
            release.set()
            async with asyncio.timeout_at(state["driver_until"]):
                if not cancel_failure:
                    while True:
                        changed.clear()
                        check()
                        if "metadata_hold" in state:
                            break
                        if run.driver.done():
                            run.driver.result()
                            raise AssertionError(
                                "original final metadata hold not reached"
                            )
                        await changed.wait()
                    cell, meta_read, meta_producer, meta_retired = state[
                        "metadata_hold"
                    ]
                    assert (
                        run.final_metadata is cell
                        and rig.coordinator._metadata_cells.get(cell) is cell.disposed
                    )
                    assert (
                        not cell.disposed.done()
                        and not meta_producer.done()
                        and not meta_retired.done()
                    )
                    assert meta_read.task is run.driver and not run.driver.done()
                    assert rig.coordinator._runs.get(run.workflow_id) is run
                    assert (
                        not prepared.primary_owner.is_admitted
                        and prepared._source_scope is None
                    )
                    assert prepared._planning_release_return is state["g_received"][0]
                    assert (
                        "release_retry" not in state
                        and not deliveries
                        and set(wake._pending[conversation]) == child_ids
                    )
                    print(
                        "PLANNING_WAKE_ORIGINAL_METADATA_HELD_WITHOUT_NOTIFICATION",
                        flush=True,
                    )
                    metadata_gate.set()
                if metadata_failure:
                    assert await asyncio.shield(run.driver) is None
                    check()
                    assert (
                        run.driver.exception() is None
                        and state["metadata_fault"] is state["metadata_reader"]
                    )
                    assert cell is run.final_metadata and cell.proof[0] is meta_read
                    assert (
                        meta_producer.done()
                        and meta_producer.exception()
                        is cell.producer_error
                        is metadata_error
                    )
                    assert meta_retired.done() and meta_retired.result() is None
                    assert (
                        cell.disposed.result() is False
                        and rig.coordinator._metadata_cells[cell] is cell.disposed
                    )
                    assert (
                        cell.caller_error is None
                        and cell.source_error is not None
                        and cell.body_error is None
                    )
                    assert (
                        run.status == "recovery_required"
                        and rig.coordinator._runs.get(run.workflow_id) is run
                    )
                    assert (
                        prepared._planning_release_return is state["g_received"][0]
                        and prepared._planning_settled
                        and prepared._closed
                    )
                    drained = []
                    prepared.primary_owner.on_drained(drained.append)
                    assert drained == [True] and not prepared.primary_owner.is_admitted
                    assert (
                        prepared.primary_claim.id
                        not in rig.runtime._worktree_activity._claims
                    )
                    assert (
                        prepared._planning_token
                        is prepared._planning_intake
                        is prepared._source_scope
                        is None
                    )
                    assert (
                        prepared not in rig.runtime._workflow_preparations
                        and set(wake._pending[conversation]) == child_ids
                    )
                    assert (
                        not state.get("notifier_entries")
                        and "release_retry" not in state
                        and not deliveries
                        and not provider.wake_calls
                    )
                    assert (
                        not wake_tasks
                        and not submissions
                        and wake._retry_timer is None
                        and not wake._delivery_tasks
                    )
                    with (
                        owned_goal_storage(rig.service),
                        rig.service.db.connection() as conn,
                    ):
                        chain = original["service"]._work_chain_id
                        assert not conn.execute(
                            "SELECT id FROM automatic_wake_attempts WHERE chain_id=?",
                            (chain,),
                        ).fetchall()
                        assert not conn.execute(
                            "SELECT id FROM automatic_work_reservations WHERE chain_id=? AND kind IN ('generation','model_call')",
                            (chain,),
                        ).fetchall()
                    state["metadata_retained"] = cell
                    print(
                        "PLANNING_METADATA_UNKNOWN_F_RETURNED_G_RETAINED_NO_NOTIFY",
                        flush=True,
                    )
                    return
                with pytest.raises(
                    asyncio.CancelledError if cancel_failure else ValueError
                ) as external:
                    await asyncio.shield(run.driver)
            check()
            assert (
                "scheduling_fault" in state
                and "release_retry_return" not in state
                and "release_handle" not in state
            )
            inner, saved, cause = state["inner_unwind"]
            returned_release, caught = state["g_received"]
            assert returned_release is prepared._planning_release_return
            assert (
                returned_release[1] is prepared._planning_read_return
                and returned_release[5] is caught
            )
            assert (
                run.final_metadata.disposed.result() is True
                and run.final_metadata.caller_error is None
            )
            assert run.final_metadata not in rig.coordinator._metadata_cells
            assert (
                run.status != "recovery_required"
                and run.workflow_id not in rig.coordinator._runs
            )
            if cancel_failure:
                assert (
                    inner is caught is saved and type(inner) is asyncio.CancelledError
                )
                assert cause is schedule_error and run.driver.cancelled()
                print(
                    "PLANNING_WAKE_ORIGINAL_CANCEL_WITH_SCHEDULING_CAUSE",
                    "outer_same",
                    external.value is inner,
                    flush=True,
                )
            else:
                assert (
                    caught is saved is None
                    and inner is external.value is schedule_error
                )
                assert run.driver.exception() is schedule_error
            read, producer, retired, callback = prepared._planning_read_return
            assert (
                read.callback is callback
                and read.task is run.driver
                and read.source is prepared
            )
            assert (
                producer.done()
                and not producer.cancelled()
                and producer.exception() is None
            )
            assert (
                retired.done() and not retired.cancelled() and retired.result() is None
            )
            assert producer.result().id == run.workflow_id
            drained = []
            prepared.primary_owner.on_drained(drained.append)
            assert drained == [True] and not prepared.primary_owner.is_admitted
            assert (
                prepared.primary_claim.id not in rig.runtime._worktree_activity._claims
            )
            assert (
                prepared._planning_intake
                is prepared._planning_token
                is prepared._source_scope
                is None
            )
            assert prepared not in rig.runtime._workflow_preparations
            assert (
                not provider.wake_calls
                and not deliveries
                and not wake_tasks
                and not submissions
            )
            assert (
                set(wake._pending[conversation]) == child_ids
                and conversation not in wake._active
            )
            assert wake._retry_timer is None and not wake._delivery_tasks
            with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
                chain = original["service"]._work_chain_id
                assert not conn.execute(
                    "SELECT id FROM automatic_wake_attempts WHERE chain_id=?", (chain,)
                ).fetchall()
                assert not conn.execute(
                    "SELECT id FROM automatic_work_reservations WHERE chain_id=? AND kind IN ('generation','model_call')",
                    (chain,),
                ).fetchall()
            print(
                "PLANNING_WAKE_SCHEDULING_ERROR_AFTER_ORIGINAL_POP",
                mode,
                sorted(child_ids),
                flush=True,
            )
            return
        release.set()
        result = await asyncio.wait_for(asyncio.shield(state["driver"]), 10)
        check()
        assert local_mode and state["release_retry_return"] is state["release_retry"][0]
        if direct:
            assert state["driver"] is direct_driver and direct_driver.result() is result
            assert run is None and rig.coordinator is None
            assert not hasattr(rig.runtime, "_workflow_coordinator")
            assert any(
                code is run_code and value is result and local["self"] is prepared
                for code, value, local in physical
            )
        else:
            assert any(
                code is g_code and local.get("run") is run
                for code, _, local in physical
            )
            assert run.workflow_id not in rig.coordinator._runs
        read, producer, retired, callback = prepared._planning_read_return
        assert read.callback is callback and read._producer is producer
        assert (
            producer.done()
            and not producer.cancelled()
            and producer.exception() is None
        )
        assert retired.done() and not retired.cancelled() and retired.result() is None
        drained = []
        prepared.primary_owner.on_drained(drained.append)
        assert drained == [True] and not prepared.primary_owner.is_admitted
        assert prepared.primary_claim.id not in rig.runtime.worktree_activity._claims
        assert prepared._planning_token is prepared._planning_intake is None
        assert (
            prepared._source_scope is None
            and prepared not in rig.runtime._workflow_preparations
        )
        assert any(
            code is type(wake).release_automatic_primary.__code__ and result is True
            for code, result, _ in physical
        )
        assert any(
            code is type(rig.store).release_received_turn.__code__ and result is True
            for code, result, _ in physical
        )
        assert any(
            code is _PreparedWorkflowReply._release_configuration.__code__
            and result[0] is prepared
            for code, result, _ in physical
        )
        if direct:
            assert result is producer.result() and result.status == "awaiting_approval"
            assert result.plans[-1].request == request
            assert not any(
                code is g_code
                or code is workflow_runs._metadata_publication_return.__code__
                for code, _, _ in physical
            )
        else:
            assert any(
                code is workflow_runs._metadata_publication_return.__code__
                and local["cell"] is run.final_metadata
                for code, _, local in physical
            )
            assert (
                run.final_metadata.disposed.done()
                and run.final_metadata.disposed.result() is True
            )
            snapshot = await rig.coordinator.get(run.workflow_id)
            assert snapshot.status == "awaiting_phase_review"
        assert [key for key, _ in provider.calls] == ["planning"]
        print(
            "PLANNING_WAKE_ORIGINAL_RELEASE_PROVEN",
            {
                "pending": sorted(wake._pending.get(conversation, {})),
                "retries": len(retries),
                "timer": wake._retry_timer is not None,
                "deliveries": len(deliveries),
            },
            flush=True,
        )
        deadline = loop.time() + 5
        while wake.has_pending(conversation) or not provider.wake_calls:
            check()
            if loop.time() >= deadline:
                raise AssertionError(
                    "Original planning physically returned but unrelated native completion remained pending: "
                    + repr(
                        {
                            "pending": sorted(wake._pending.get(conversation, {})),
                            "timer": wake._retry_timer is not None,
                            "retries": len(retries),
                            "deliveries": len(deliveries),
                        }
                    )
                )
            await asyncio.sleep(0.01)
        for task in deliveries:
            await asyncio.wait_for(
                asyncio.shield(task), max(0.001, deadline - loop.time())
            )
        check()
        assert 1 <= len(provider.wake_calls) <= 2
        assert state["release_handle"] in state["handle_entries"]
        assert state["release_attempts"]
        assert any(
            handle is state["release_handle"] for _task, _context, handle in wake_tasks
        )
        assert len(wake_tasks) == len(submissions) == len(provider.wake_calls)
        for authorization, delivery, custody, task, _store, _claim in submissions:
            assert any(created is delivery for created, _context, _handle in wake_tasks)
            assert (
                delivery in deliveries
                and delivery.done()
                and delivery.exception() is None
            )
            assert (
                authorization.accepted
                and authorization.work_chain_id == original["service"]._work_chain_id
            )
            (source_row,) = [row for row in source_reads if row[4] is task]
            source, read, producer, retired, _task = source_row
            assert read._producer is producer and read.retired is retired
            assert (
                producer.done()
                and not producer.cancelled()
                and producer.exception() is None
            )
            assert (
                retired.done() and not retired.cancelled() and retired.result() is None
            )
            (tree,) = [tree for tree, item in trees.items() if item[0] is source]
            _source, owner, claim, tree_retired = trees[tree]
            observed_tickets = [
                ticket
                for ticket, item in tickets.items()
                if item[0].ordinary_tree is tree
            ]
            while (
                (custody, task) not in custody_returns
                or not task.done()
                or (tree, tree_retired) not in tree_returns
                or any(ticket not in ticket_returns for ticket in observed_tickets)
            ):
                check()
                if loop.time() >= deadline:
                    raise AssertionError(
                        "original wake custody/tree/ticket did not return within delivery5"
                    )
                await asyncio.sleep(0.01)
            assert (
                not task.cancelled()
                and task.exception() is None
                and task.result().accepted
            )
            assert (
                tree_retired.done()
                and not tree_retired.cancelled()
                and tree_retired.exception() is None
            )
            assert (
                tree not in rig.runtime._ordinary_turns
                and (claim, True) in claim_returns
            )
            drained = []
            owner.on_drained(drained.append)
            assert drained == [True] and not owner.is_admitted
            for ticket in observed_tickets:
                snapshot, operation = tickets[ticket]
                assert ticket.source is snapshot and ticket.owner is owner
                assert ticket.operation is operation and operation in operation_returns
                assert ticket.retired.is_set() and not ticket._failed_write
            print(
                "PLANNING_WAKE_ORIGINAL_ORDINARY_RETURN",
                authorization.attempt_id,
                len(observed_tickets),
                flush=True,
            )
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            rows = conn.execute(
                "SELECT run_ids_json,state,id FROM automatic_wake_attempts WHERE chain_id=?",
                (original["service"]._work_chain_id,),
            ).fetchall()
            delivered = [run_id for row in rows for run_id in json.loads(row[0])]
            assert len(delivered) == 2 and set(delivered) == child_ids
            assert all(row[1] == "completed" for row in rows)
            assert {row[2] for row in rows} == {
                item[0].attempt_id for item in submissions
            }
            charges = rig.service.db.automatic_work.snapshot(
                original["service"]._work_chain_id
            )
            assert charges.used["generation"] == charges.used["model_call"] == len(rows)
            assert not charges.uncertain
    except BaseException as error:
        primary_error = error
        print(
            "PLANNING_WAKE_PRIMARY_FAILURE",
            type(error).__name__,
            str(error),
            flush=True,
        )
        try:
            failure_snapshot()
        except BaseException as diagnostic_error:  # noqa: BLE001 - diagnostics cannot replace the original failure.
            print(
                "PLANNING_WAKE_DIAGNOSTIC_ERROR",
                type(diagnostic_error).__name__,
                str(diagnostic_error),
                flush=True,
            )
        raise
    finally:
        cleanup_observer_start = len(errors)
        metadata_gate.set()
        release.set()
        for gate in provider.child_release.values():
            gate.set()
        cleanup_errors = []
        for thread in ([parent] if parent is not None else []) + [
            row[0] for row in children
        ]:
            try:
                await asyncio.to_thread(thread.join, 5)
                assert not thread.is_alive(), (
                    "original planning-wake fixture thread retained"
                )
            except BaseException as error:  # noqa: BLE001 - still attempt actual native disposal.
                cleanup_errors.append(error)
        if direct_driver is not None:
            # The direct caller owns this original Task, including failed setup.
            while not direct_driver.done():
                try:
                    await asyncio.shield(direct_driver)
                except asyncio.CancelledError as waiter_error:
                    if waiter_error is not primary_error:
                        cleanup_errors.append(waiter_error)
                except BaseException as waiter_error:  # noqa: BLE001 - consume the actual Task outcome below.
                    if not direct_driver.done() and waiter_error is not primary_error:
                        cleanup_errors.append(waiter_error)
            try:
                outcome = direct_driver.result()
            except BaseException as driver_error:  # noqa: BLE001 - retain separate actual cleanup outcome.
                print(
                    "PLANNING_WAKE_DIRECT_DRIVER_RETURNED",
                    type(driver_error).__name__,
                    flush=True,
                )
                if driver_error is not primary_error:
                    cleanup_errors.append(driver_error)
            else:
                print(
                    "PLANNING_WAKE_DIRECT_DRIVER_RETURNED",
                    type(outcome).__name__,
                    getattr(outcome, "status", None),
                    flush=True,
                )
        if close_task is not None:
            while not close_task.done():
                try:
                    await asyncio.shield(close_task)
                except BaseException as waiter_error:  # noqa: BLE001 - original result is consumed below.
                    if not close_task.done() and waiter_error is not primary_error:
                        cleanup_errors.append(waiter_error)
            try:
                close_task.result()
            except BaseException as original_shutdown_error:  # noqa: BLE001 - preserve the original independent shutdown outcome.
                if original_shutdown_error is not primary_error and not (
                    before_fence
                    and original_shutdown_error is fence_error
                    and state.get("first_fence_error_returned")
                ):
                    cleanup_errors.append(original_shutdown_error)
        if before_fence and "fence_fault" in state:
            print("PLANNING_SHUTDOWN_SEPARATE_NATIVE_RIG_CLOSE_ATTEMPT", flush=True)
        try:
            await rig.close()
            if metadata_failure:
                raise AssertionError(
                    "unresolved original metadata unexpectedly returned creators"
                )
            if direct:
                assert rig.coordinator is None and not hasattr(
                    rig.runtime, "_workflow_coordinator"
                )
            print("PLANNING_WAKE_ORIGINAL_CREATORS_RETURNED", flush=True)
        except BaseException as error:  # noqa: BLE001 - do not erase the original failure.
            if (
                metadata_failure
                and primary_error is None
                and type(error) is RuntimeError
                and str(error) == "workflow_original_metadata_shutdown_unresolved"
            ):
                cell = state["metadata_retained"]
                tb = error.__traceback__
                while tb.tb_next is not None:
                    tb = tb.tb_next
                assert tb.tb_frame.f_code is type(rig.coordinator).close.__code__
                assert tb.tb_frame.f_locals["self"] is rig.coordinator
                assert (
                    rig.coordinator._metadata_cells[cell] is cell.disposed
                    and cell.disposed.result() is False
                )
                assert (
                    rig.coordinator._runs.get(run.workflow_id) is run
                    and not rig.runtime._disposed
                    and not rig.runtime.execution_capacity.snapshot().closed
                )
                traceback.print_exception(error)
                print(
                    "PLANNING_METADATA_REAL_CLOSE_REFUSED_NO_CREATOR_RETURN", flush=True
                )
            else:
                cleanup_errors.append(error)
        finally:

            def remove(call, *args):
                try:
                    call(*args)
                except BaseException as cleanup_error:  # noqa: BLE001 - attempt every owned removal.
                    cleanup_errors.append(cleanup_error)

            if profile_attempted:
                current_profile, future_profile = (
                    sys.getprofile(),
                    threading.getprofile(),
                )
                if current_profile is observe or future_profile is observe:
                    if (current_profile is None or current_profile is observe) and (
                        future_profile is None or future_profile is observe
                    ):
                        remove(threading.setprofile_all_threads, None)
                    else:
                        if current_profile is observe:
                            remove(sys.setprofile, None)
                        if future_profile is observe:
                            remove(threading.setprofile, None)
                        cleanup_errors.append(
                            AssertionError(
                                "original all-thread observer displaced; other threads unknown"
                            )
                        )
            if monitor_owned:
                local_mode = False
                before_monitor_removal = len(cleanup_errors)
                if global_attempted:
                    remove(monitor.set_events, tool_id, 0)
                for code in armed_codes:
                    remove(monitor.set_local_events, tool_id, code, 0)
                for event, previous in saved_callbacks.items():
                    remove(monitor.register_callback, tool_id, event, previous)
                if len(cleanup_errors) == before_monitor_removal:
                    remove(monitor.free_tool_id, tool_id)
                    if monitor.get_tool(tool_id) is not None:
                        cleanup_errors.append(
                            AssertionError("original monitoring tool not freed")
                        )
                    else:
                        print("PLANNING_WAKE_ORIGINAL_OBSERVER_RETURNED", flush=True)
                else:
                    cleanup_errors.append(
                        AssertionError(
                            "monitor removal unresolved; owned tool id retained"
                        )
                    )
        cleanup_errors.extend(errors[cleanup_observer_start:])
        for error in cleanup_errors:
            traceback.print_exception(error)
        if cleanup_errors:
            if primary_error is not None:
                primary_error.add_note(
                    "Original native cleanup also failed; exact errors printed"
                )
            else:
                raise cleanup_errors[0]


def test_original_ordinary_wake_queued_before_planning_release(tmp_path):
    _run(
        tmp_path,
        "planning-wake",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_direct_planning_release_drives_ordinary_wake_without_coordinator(tmp_path):
    _run(
        tmp_path,
        "planning-wake",
        "direct",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def planning_release_integrity_journey(tmp_path, monkeypatch):
    """Refuse a changed original slot source and retain an actual partial return."""
    import asyncio
    import dis
    import sys

    from tldw_chatbook.Chat.console_chat_controller import (
        capture_run_admitted_workspace_roots,
    )
    from tldw_chatbook.Chat.console_workflow_preparation import (
        _planning_release_source,
        _PreparedWorkflowReply,
        prepare_workflow_planning,
    )

    mode = sys.argv[2]
    assert mode in {"source", "partial", "membership"}
    rig = await build_workflow_rig(
        tmp_path,
        monkeypatch,
        WorkflowProvider(),
        public_coordinator=False,
        native_fleet_consumers=True,
    )
    session = rig.store.sessions()[0]
    store, wake = rig.store, rig.controller.fleet_wake
    received_code = type(store).release_received_turn.__code__
    token_code = type(wake).release_automatic_primary.__code__
    slots_code = _PreparedWorkflowReply._release_planning_slots.__code__
    notify_code = _PreparedWorkflowReply._notify_planning_release.__code__
    sentinel = ValueError("original planning token release entry failed")
    replacement_calls, notifications, received, attempted, errors = [], [], [], [], []
    prepared = token = intake = scope = None
    primary = None
    membership = []

    def retained():
        release_binding = prepared._planning_release_binding
        owner, claim = prepared.primary_owner, prepared.primary_claim
        read, producer, retired, callback = prepared._planning_read_return
        assert read.creator is rig.controller and read.source is prepared
        assert (
            read.task
            is release_binding[3]
            is prepared._planning_driver
            is asyncio.current_task()
        )
        assert (
            read.callback is callback
            and read._producer is producer
            and read.retired is retired
        )
        assert (
            producer.done()
            and not producer.cancelled()
            and producer.exception() is None
        )
        assert retired.done() and not retired.cancelled() and retired.result() is None
        assert (
            prepared._planning_terminal_returned
            and prepared._planning_physical_returned
        )
        assert owner.is_admitted and not owner._root_finished
        assert rig.runtime._worktree_activity._claims[claim.id].custody is claim
        assert prepared._source_scope is scope and scope.gen.gi_frame is not None
        assert (
            prepared in rig.runtime._workflow_preparations
            and not prepared._planning_settled
            and not prepared._closed
        )
        assert (
            prepared._planning_intake is intake
            and prepared._planning_token
            is wake._automatic_primary_claims[session.id]
            is token
        )
        assert prepared._planning_release_return is None

    def replacement(*args):
        replacement_calls.append(args)
        raise AssertionError("replaced planning received release was called")

    def observe(frame, event, value):
        code = frame.f_code
        if code not in (
            received_code,
            token_code,
            notify_code,
            _planning_release_source.__code__,
        ):
            return
        inject = False
        try:
            local, caller = frame.f_locals, frame.f_back
            if (
                mode == "membership"
                and code is _planning_release_source.__code__
                and event == "call"
                and local.get("prepared") is prepared
                and prepared._planning_terminal_returned
                and not membership
            ):
                release_binding = prepared._planning_release_binding
                assert local["binding"] is release_binding
                assert local["scope_live"] is local["slots_live"] is True
                assert (
                    caller.f_back is release_binding[5]
                    and release_binding[5].f_code
                    is _PreparedWorkflowReply.run_planning.__code__
                )
                assert (
                    release_binding[13] is intake
                    and release_binding[14] is token
                    and release_binding[17] is scope
                )
                assert release_binding[6] is release_binding[7] is None
                retained()
                assert (
                    store.received_turn_for_session(session.id) is intake
                    and not intake.sealed
                )
                assert store.release_received_turn(intake) is True
                assert (
                    intake.sealed
                    and store.received_turn_for_session(session.id) is None
                )
                membership.append((frame, release_binding))
            elif (
                code is notify_code
                and event == "call"
                and local.get("self") is prepared
            ):
                notifications.append(frame)
            elif (
                caller is not None
                and caller.f_code is slots_code
                and caller.f_locals.get("self") is prepared
            ):
                if (
                    code is received_code
                    and local.get("self") is store
                    and local.get("claim") is intake
                ):
                    if event == "return" and dis.opname[
                        code.co_code[frame.f_lasti]
                    ] in {"RETURN_VALUE", "RETURN_CONST"}:
                        received.append((value, caller.f_locals.get("_completion")))
                elif (
                    mode == "partial"
                    and code is token_code
                    and event == "call"
                    and local.get("self") is wake
                    and local.get("session_id") == session.id
                    and local.get("token") is token
                ):
                    release_binding = prepared._planning_release_binding
                    assert caller.f_locals["_completion"] is release_binding
                    assert frame.f_back.f_back is release_binding[5]
                    assert asyncio.current_task() is release_binding[3]
                    assert (
                        len(received) == 1
                        and received[0][0] is True
                        and received[0][1] is release_binding
                    )
                    assert (
                        prepared._planning_intake is None
                        and prepared._planning_token is token
                    )
                    assert store.received_turn_for_session(session.id) is not intake
                    assert wake._automatic_primary_claims.get(session.id) is token
                    assert (
                        caller.f_locals["_configuration_return"][1] is scope
                        and prepared._source_scope is None
                    )
                    assert (
                        prepared not in rig.runtime._workflow_preparations
                        and scope.gen.gi_frame is None
                    )
                    assert (
                        prepared._closed
                        and prepared._planning_settled
                        and not prepared.primary_owner.is_admitted
                    )
                    assert (
                        prepared.primary_claim.id
                        not in rig.runtime._worktree_activity._claims
                    )
                    assert prepared._planning_release_return is None and not attempted
                    attempted.append((frame, caller, release_binding))
                    inject = True
        except BaseException as error:  # noqa: BLE001 - observation errors do not change native behavior.
            errors.append(error)
        if inject:
            # Single intentional fault at the native release entry; no replacement return.
            raise sentinel

    try:
        assert rig.coordinator is None and not hasattr(
            rig.runtime, "_workflow_coordinator"
        )
        request = rig.workflow.request
        binding = capture_run_admitted_workspace_roots(
            session=session, registry=rig.service.persistence.workspace_registry
        )[0]
        prepared = await prepare_workflow_planning(
            rig.controller,
            session_id=session.id,
            binding=binding,
            command_id="direct-planning-integrity",
            objective=request.objective,
            policy=request.policy,
            _request=request,
        )
        token, intake, scope = (
            prepared._planning_token,
            prepared._planning_intake,
            prepared._source_scope,
        )
        assert (
            token is not None
            and wake._automatic_primary_claims.get(session.id) is token
        )
        assert (
            intake is not None and store.received_turn_for_session(session.id) is intake
        )
        assert scope is not None and prepared in rig.runtime._workflow_preparations
        assert (
            prepared.primary_owner is None and prepared._planning_release_return is None
        )
        assert sys.getprofile() is None
        sys.setprofile(observe)
        if mode == "source":
            assert "release_received_turn" not in vars(store)
            store.release_received_turn = replacement
            with pytest.raises(
                PermissionError, match="original planning slot receiver changed"
            ):
                await prepared.run_planning()
            assert (
                not replacement_calls
                and not notifications
                and not received
                and not attempted
            )
            assert (
                prepared.primary_owner is prepared.primary_claim is None
                and not rig.provider.calls
            )
            assert prepared._planning_release_return is None and not prepared._closed
            assert (
                prepared._source_scope is scope
                and prepared in rig.runtime._workflow_preparations
            )
            assert (
                prepared._planning_token
                is wake._automatic_primary_claims[session.id]
                is token
            )
            assert (
                prepared._planning_intake
                is store.received_turn_for_session(session.id)
                is intake
            )
            assert vars(store)["release_received_turn"] is replacement
            del store.release_received_turn
            # Only the actual unused close is now run; never retry planning on restored source.
            await prepared.close()
            assert (
                len(received) == 1 and received[0][0] is True and received[0][1] is None
            )
            assert (
                prepared._planning_intake
                is prepared._planning_token
                is prepared._source_scope
                is None
            )
            assert prepared not in rig.runtime._workflow_preparations
            assert wake._automatic_primary_claims.get(session.id) is not token
            assert store.received_turn_for_session(session.id) is not intake
            print(
                "PLANNING_RELEASE_SOURCE_REFUSED_AND_UNUSED_CLOSE_RETURNED", flush=True
            )
        elif mode == "membership":
            with pytest.raises(
                PermissionError, match="^original planning received claim changed$"
            ) as raised:
                await prepared.run_planning()
            assert not errors and len(membership) == 1
            tb = raised.value.__traceback__
            while tb.tb_next is not None:
                tb = tb.tb_next
            assert (
                tb.tb_frame is membership[0][0]
                and membership[0][1] is prepared._planning_release_binding
            )
            retained()
            assert intake.sealed and store.received_turn_for_session(session.id) is None
            assert not received and not attempted and not notifications
            print("PLANNING_TERMINAL_ORIGINAL_RECEIVED_MEMBERSHIP_REFUSED", flush=True)
        else:
            with pytest.raises(ValueError) as raised:
                await prepared.run_planning()
            assert raised.value is sentinel and len(attempted) == 1
            assert prepared._source_scope is prepared._planning_intake is None
            assert (
                prepared._planning_token
                is wake._automatic_primary_claims[session.id]
                is token
            )
            read, producer, retired, callback = prepared._planning_read_return
            assert (
                read.callback is callback
                and read._producer is producer
                and read.retired is retired
            )
            assert (
                producer.done()
                and not producer.cancelled()
                and producer.exception() is None
            )
            assert (
                retired.done() and not retired.cancelled() and retired.result() is None
            )
            drained = []
            prepared.primary_owner.on_drained(drained.append)
            assert drained == [True] and not prepared.primary_owner.is_admitted
            assert (
                prepared.primary_claim.id not in rig.runtime._worktree_activity._claims
            )
            print("PLANNING_PARTIAL_NATIVE_RECEIVED_RETURN_TOKEN_RETAINED", flush=True)
        assert not errors and not notifications and not replacement_calls
        assert prepared._planning_release_return is None
        assert rig.coordinator is None and not hasattr(
            rig.runtime, "_workflow_coordinator"
        )
    except BaseException as error:
        primary = error
        raise
    finally:
        if vars(store).get("release_received_turn") is replacement:
            del store.release_received_turn
        if sys.getprofile() is observe:
            sys.setprofile(None)
        try:
            await rig.close()
            if mode == "membership":
                raise AssertionError(
                    "retained planning root unexpectedly passed creator cleanup"
                )
            print(
                "PLANNING_INTEGRITY_CREATORS_RETURNED",
                mode,
                "token_still_original",
                token is not None
                and wake._automatic_primary_claims.get(session.id) is token,
                flush=True,
            )
        except BaseException as cleanup_error:
            import traceback

            if mode == "membership" and primary is None:
                tb = cleanup_error.__traceback__
                while tb.tb_next is not None:
                    tb = tb.tb_next
                if (
                    type(cleanup_error) is not AssertionError
                    or tb.tb_frame.f_code is not type(rig).close.__code__
                ):
                    raise
                assert rig.runtime._disposed and tb.tb_frame.f_locals["self"] is rig
                retained()
                (execution,) = rig.runtime.execution_capacity.snapshot().executions
                assert (
                    execution.execution_id == prepared.primary_owner.execution_id
                    and not execution.root_finished
                )
                assert (
                    intake.sealed
                    and store.received_turn_for_session(session.id) is None
                )
                traceback.print_exception(cleanup_error)
                print(
                    "PLANNING_MEMBERSHIP_RIG_CLOSE_REFUSED_RETAINED_ORIGINAL_ROOT_NO_CREATOR_RETURN",
                    flush=True,
                )
            elif primary is None:
                raise
            else:
                traceback.print_exception(cleanup_error)
                primary.add_note("Original integrity cleanup failed separately")


def _expect_retained_planning_child(args, kwargs, *markers):
    """The exact original process succeeded, but its required creator did not return."""
    with pytest.raises(AssertionError) as missing:
        _run(*args, **kwargs)
    tb = missing.value.__traceback__
    while tb.tb_next is not None:
        tb = tb.tb_next
    assert tb.tb_frame.f_code is _run.__code__
    original = tb.tb_frame.f_locals
    assert original["result"].returncode == 0
    rows = original["stdout"].splitlines()
    assert "retired and reopened" not in original["stdout"]
    assert all(rows.count(marker) == 1 for marker in markers)


@pytest.mark.parametrize("mode", ["source", "partial", "membership"])
def test_planning_release_refuses_changed_source_and_incomplete_slots(tmp_path, mode):
    args = tmp_path, "planning-integrity", mode
    kwargs = {
        "script": Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        "timeout": 60,
    }
    if mode == "membership":
        _expect_retained_planning_child(
            args,
            kwargs,
            "PLANNING_TERMINAL_ORIGINAL_RECEIVED_MEMBERSHIP_REFUSED",
            "PLANNING_MEMBERSHIP_RIG_CLOSE_REFUSED_RETAINED_ORIGINAL_ROOT_NO_CREATOR_RETURN",
        )
    else:
        _run(*args, **kwargs)


@pytest.mark.parametrize("mode", ["notify-error", "notify-cancel"])
def test_original_planning_notification_error_after_return_preserves_disposition(
    tmp_path, mode
):
    _run(
        tmp_path,
        "planning-wake",
        mode,
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_planning_metadata_unknown_keeps_original_run_and_skips_notification(tmp_path):
    _expect_retained_planning_child(
        (tmp_path, "planning-wake", "metadata-unknown"),
        {
            "script": Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            "timeout": 60,
        },
        "PLANNING_METADATA_UNKNOWN_F_RETURNED_G_RETAINED_NO_NOTIFY",
        "PLANNING_METADATA_REAL_CLOSE_REFUSED_NO_CREATOR_RETURN",
        "PLANNING_WAKE_ORIGINAL_OBSERVER_RETURNED",
    )


@pytest.mark.parametrize("mode", ["shutdown-fence", "shutdown-before-fence"])
def test_planning_permanent_shutdown_fences_original_ordinary_notification(
    tmp_path, mode
):
    _run(
        tmp_path,
        "planning-wake",
        mode,
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )
