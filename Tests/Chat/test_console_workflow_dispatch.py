"""Accepted planning and approved tasks use actual native Console owners."""

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
        first = await rig.coordinator.approve(decision, command_id="approve-1")
        await rig.wait_for_provider_entry("task-1")
        again = await rig.coordinator.approve(decision, command_id="approve-1")
        assert (first.id, first.control_revision) == (again.id, again.control_revision)
        assert [key for key, _ in rig.provider.calls].count("task-1") == 1
        with rig.service.db.connection() as conn:
            assert (
                conn.execute(
                    "SELECT count(*) FROM workflow_task_attempts WHERE run_id IS NOT NULL"
                ).fetchone()[0]
                == 1
            )
            assert (
                conn.execute("SELECT count(*) FROM workflow_phase_attempts").fetchone()[
                    0
                ]
                == 2
            )
        owners = rig.runtime.execution_capacity.snapshot().executions
        assert sum(owner.child for owner in owners) == 1
        rig.provider_release()
        await rig.wait_for_status("awaiting_result_review")
    finally:
        await rig.close()


def test_plan_then_duplicate_approval_launches_one_native_task(tmp_path):
    _run(
        tmp_path,
        "dispatch",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def pending_planning_missing_read_journey(tmp_path, monkeypatch):
    import asyncio
    import dis
    import os
    import threading

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    original_error = RuntimeError("original pending planning reader was never issued")
    saved = {}
    command = None

    def counts():
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            return tuple(
                conn.execute(query).fetchone()[0]
                for query in (
                    "SELECT count(*) FROM workflow_phase_attempts",
                    "SELECT count(*) FROM automatic_wake_attempts",
                    "SELECT count(*) FROM automatic_work_reservations",
                )
            )

    def observe(frame, event, _result):
        if (
            event != "return"
            or frame.f_code
            is not _PreparedWorkflowReply._planning_return_source.__code__
            or dis.opname[frame.f_code.co_code[frame.f_lasti]]
            not in {"RETURN_VALUE", "RETURN_CONST"}
        ):
            return
        caller = frame.f_back
        prepared = frame.f_locals["self"]
        if (
            caller is None
            or caller.f_code is not run_preparation_read.__code__
            or "read" in caller.f_locals
            or caller.f_locals.get("source") is not prepared
            or prepared.controller is not rig.controller
        ):
            return
        record = prepared._planning_reads[-1]
        run = prepared._planning_pending_run
        if (
            record.callback is not caller.f_locals["callback"]
            or record.prepared is not prepared
            or run.prepared is not prepared
            or record.issuing is not asyncio.current_task()
            or run.issuing is not record.issuing
        ):
            return
        threading.setprofile_all_threads(None)
        saved.update(prepared=prepared, record=record, run=run, issuer=record.issuing)
        raise original_error

    try:
        before = counts()
        threading.setprofile_all_threads(observe)
        command = asyncio.create_task(
            rig.coordinator.start(
                rig.workflow.request, launch_id=rig.workflow.launch_id
            )
        )
        with pytest.raises(RuntimeError) as public_error:
            await command
        assert saved, "original pre-reader boundary not reached"
        run, prepared, record = saved["run"], saved["prepared"], saved["record"]
        assert record.wrapper_error is original_error
        assert record.proof is record.owner is record.claim is None
        assert record.body_error is None and not record.positive
        assert public_error.value is run.planning_error is saved["issuer"].exception()
        assert run in rig.coordinator._pending_planning and run.prepared is prepared
        assert (
            prepared._planning_intake is not None
            and prepared._planning_token is not None
        )
        assert run.status == "recovery_required" and not prepared._closed
        assert counts() == before and rig.provider.calls == []
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await rig.runtime.dispose()
            assert (
                not rig.runtime._disposed and run in rig.coordinator._pending_planning
            )
            assert (
                prepared._planning_intake is not None
                and prepared._planning_token is not None
            )
        print("G_PENDING_F_ORIGINAL_READ_ABSENT_RETAINED", flush=True)
        os._exit(17)
    finally:
        threading.setprofile_all_threads(None)
        if command is not None and not command.done():
            command.cancel()
            await asyncio.gather(command, return_exceptions=True)
        await rig.close()


async def pending_planning_journey(tmp_path, monkeypatch):
    import asyncio
    import inspect
    import sys
    import threading

    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply
    from tldw_chatbook.Chat.console_workflow_runs import _accept_workflow_phase
    from tldw_chatbook.DB.recovery_workflow_schema import validate_workflow_rows

    boundary = sys.argv[2]
    if boundary == "initial-no-read":
        return await pending_planning_missing_read_journey(tmp_path, monkeypatch)
    competing_revision = boundary == "initial-accepted-revision"
    repair = boundary.startswith("repair")
    accepted = "accepted" in boundary
    wrapper = boundary.endswith("wrapper")
    unknown = boundary.endswith("unknown") or wrapper
    shutdown = boundary.endswith("shutdown")
    pause = boundary.endswith("pause")
    original_error = RuntimeError("original planning receipt did not return")
    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    entered, release = threading.Event(), threading.Event()
    revision_entered, revision_release = threading.Event(), threading.Event()
    saved, revision_saved = {}, {}
    command = stopping = None
    try:
        if repair:
            rig.workflow = await rig.coordinator.start(
                rig.workflow.request, launch_id=rig.workflow.launch_id
            )
            initial = rig.coordinator._runs[rig.workflow.id]
            await asyncio.shield(initial.driver)
            rig.workflow = await rig.wait_for_status("awaiting_phase_review")
        ledger = rig.service.db.automatic_work
        code = (
            next(
                item
                for item in _PreparedWorkflowReply._read_planning.__code__.co_consts
                if inspect.iscode(item) and item.co_name == "invoke"
            )
            if wrapper
            else _accept_workflow_phase.__code__
            if accepted
            else type(ledger).prepare_workflow_planning.__code__
        )

        def observe(frame, event, result):
            if event != "return" or frame.f_code is not code or result is None:
                return
            if wrapper:
                prepared = frame.f_locals.get("self")
                if (
                    not isinstance(result, tuple)
                    or len(result) != 10
                    or result[1] is not prepared
                    or type(result[8]) is not tuple
                    or prepared.phase is not result[8][0]
                ):
                    return
            else:
                caller = frame.f_back
                if caller is None or caller.f_code.co_name != "commit_request":
                    return
                prepared = caller.f_locals.get("prepared")
            if prepared is None or prepared.controller is not rig.controller:
                return
            threading.setprofile_all_threads(None)
            saved.update(
                prepared=prepared,
                run=prepared._planning_pending_run,
                result=result,
                record=prepared._planning_reads[-1],
            )
            entered.set()
            assert release.wait(15), "original planning return hold expired"
            if unknown:
                raise original_error

        threading.setprofile_all_threads(observe)
        if repair:
            command = asyncio.create_task(
                rig.coordinator.revise(
                    rig.workflow.id,
                    expected_control_revision=rig.workflow.control_revision,
                    instruction="Review the same plan again",
                    command_id="pending-repair",
                )
            )
        else:
            command = asyncio.create_task(
                rig.coordinator.start(
                    rig.workflow.request, launch_id=rig.workflow.launch_id
                )
            )
        assert await asyncio.to_thread(entered.wait, 10)
        run, prepared = saved["run"], saved["prepared"]
        assert rig.coordinator._runs[run.workflow_id] is run
        assert run.prepared is prepared and run.phase is None
        if wrapper:
            assert saved["record"].drains == [True]
            assert not saved["record"].owner.is_admitted
            assert saved["record"].claim.id not in rig.runtime.worktree_activity._claims
            assert not saved["record"].proof[1].done()
        else:
            assert saved["record"].owner.is_admitted
            assert saved["record"].claim.id in rig.runtime.worktree_activity._claims
        original_issuer = run.issuing
        revision = run.planning_snapshot.control_revision
        if pause:
            paused = await rig.coordinator.pause(
                run.workflow_id, expected_control_revision=revision
            )
            assert paused.status == "pausing" and not command.done()
            release.set()
            rig.workflow = await command
            await asyncio.shield(run.driver)
            paused = await rig.wait_for_status("paused")
            assert [key for key, _ in rig.provider.calls] == ["planning"]
            assert prepared._planning_settled and prepared._closed
            assert rig.runtime.execution_capacity.snapshot().executions == ()
            return
        if competing_revision:
            import dis

            from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
            from tldw_chatbook.Chat.console_workflow_runs import (
                ConsoleWorkflowCoordinator,
            )
            from tldw_chatbook.DB.base_db import run_owned_db_call

            def revision_observe(frame, event, result):
                if (
                    event == "call"
                    and frame.f_code
                    is ConsoleWorkflowCoordinator._pending_planning_control.__code__
                    and frame.f_locals.get("run") is run
                ):
                    revision_saved.setdefault("issuer", asyncio.current_task())
                if (
                    event == "call"
                    and frame.f_code is run_owned_db_call.__code__
                    and prepared._closed
                    and threading.current_thread() is threading.main_thread()
                    and asyncio.current_task() is revision_saved.get("issuer")
                ):
                    caller = frame.f_back
                    if (
                        caller is not None
                        and caller.f_code is WorkflowRunService.call.__code__
                    ):
                        operation = caller.f_locals["operation"]
                        if (
                            operation.__self__ is rig.service
                            and operation.__func__ is WorkflowRunService.get
                        ):
                            invoke = frame.f_locals["operation"]
                            closed = dict(
                                zip(
                                    invoke.__code__.co_freevars,
                                    (cell.cell_contents for cell in invoke.__closure__),
                                )
                            )
                            assert caller.f_locals["invoke"] is invoke
                            assert frame.f_locals["database"] is rig.service.db
                            assert (
                                closed["operation"] is operation
                                and closed["self"] is rig.service
                            )
                            revision_saved.update(
                                invoke=invoke,
                                operation=operation,
                                args=closed["args"],
                                kwargs=closed["kwargs"],
                            )
                invoke = revision_saved.get("invoke")
                if (
                    invoke is not None
                    and event == "return"
                    and frame.f_code is invoke.__code__
                    and dis.opname[frame.f_code.co_code[frame.f_lasti]]
                    in {"RETURN_VALUE", "RETURN_CONST"}
                    and frame.f_locals.get("operation") is revision_saved["operation"]
                    and frame.f_locals.get("args") is revision_saved["args"]
                    and frame.f_locals.get("kwargs") is revision_saved["kwargs"]
                    and frame.f_locals.get("self") is rig.service
                ):
                    threading.setprofile_all_threads(None)
                    revision_saved.update(
                        snapshot=result,
                        worker=threading.current_thread(),
                        control_return=run.control_return,
                    )
                    revision_entered.set()
                    assert revision_release.wait(10), (
                        "original Stop metadata return hold expired"
                    )

            threading.setprofile_all_threads(revision_observe)
        stopping = asyncio.create_task(
            rig.runtime.dispose()
            if shutdown
            else rig.coordinator.stop(
                run.workflow_id, expected_control_revision=revision
            )
        )
        async with asyncio.timeout(10):
            while not prepared.cancel_event.is_set():
                await asyncio.sleep(0)
        assert original_issuer is run.issuing and not stopping.done()
        if boundary.endswith("repeat"):
            original_issuer.cancel()
            await asyncio.sleep(0)
            original_issuer.cancel()
            assert not original_issuer.done()
        release.set()
        if competing_revision:
            import os

            assert await asyncio.to_thread(revision_entered.wait, 10)
            assert prepared._closed and prepared._finite is None
            prepared._planning_reads_returned()
            assert prepared._planning_intake is prepared._planning_token is None
            assert revision_saved["control_return"] is run.control_return
            assert not run.control_return.is_set()
            assert revision_saved["snapshot"].control_revision == revision
            assert revision_saved["worker"] is not threading.current_thread()
            competing = await rig.service.call(
                rig.service.set_status,
                run.workflow_id,
                status="paused",
                expected_control_revision=revision,
            )
            assert (
                competing.control_revision == revision + 1
                and competing.status == "paused"
            )
            revision_release.set()
            with pytest.raises(ValueError, match="revision_conflict") as refused:
                await stopping
            assert refused.value is revision_saved["issuer"].exception()
            with pytest.raises(asyncio.CancelledError):
                await command
            actual = await rig.service.call(rig.service.get, run.workflow_id)
            assert actual.control_revision == revision + 1 and actual.status == "paused"
            assert (
                actual.accounting.used["generation"] == 1 and rig.provider.calls == []
            )
            assert run.control_unknown and run.status == "recovery_required"
            assert run in rig.coordinator._pending_planning
            assert rig.coordinator._runs[run.workflow_id] is run
            assert run.control_return.is_set()
            assert prepared._planning_intake is prepared._planning_token is None
            assert rig.runtime.execution_capacity.snapshot().executions == ()
            with rig.service.db.connection() as conn:
                native = conn.execute(
                    "SELECT n.state,p.status,p.retirement,r.state FROM automatic_wake_attempts n "
                    "JOIN workflow_phase_attempts p ON p.native_attempt_id=n.id "
                    "JOIN automatic_work_reservations r ON r.id=n.generation_reservation_id WHERE n.id=?",
                    (prepared._planning_attempt.id,),
                ).fetchone()
                assert tuple(native) == ("completed", "closed", "drained", "committed")
                assert validate_workflow_rows(conn) == ()
            for _ in range(2):
                with pytest.raises(
                    RuntimeError, match="workflow_original_shutdown_unresolved"
                ):
                    await rig.runtime.dispose()
                assert (
                    not rig.runtime._disposed
                    and run in rig.coordinator._pending_planning
                )
            print("G_PENDING_F_COMPETING_REVISION_RETAINED", flush=True)
            os._exit(17)
        if unknown:
            import os

            failure = "reader return unproven" if wrapper else "receipt missing"
            with pytest.raises(RuntimeError, match=failure):
                await stopping
            if wrapper:
                with pytest.raises(RuntimeError, match=failure) as public_error:
                    await command
                assert public_error.value is original_issuer.exception()
                assert public_error.value is run.planning_error
                assert isinstance(saved["record"].wrapper_error, asyncio.CancelledError)
            else:
                with pytest.raises(asyncio.CancelledError):
                    await command
            if wrapper:
                assert saved["record"].proof[1].exception() is original_error
                assert not saved["record"].positive
                carrier = saved["result"]
                record = saved["record"]
                assert all(
                    actual is expected
                    for actual, expected in zip(
                        carrier[:8],
                        (
                            record,
                            prepared,
                            original_issuer,
                            record.callback,
                            record.proof,
                            record.owner,
                            record.claim,
                            record.operation,
                        ),
                    )
                )
                assert prepared.phase is carrier[8][0]
            else:
                assert saved["record"].body_error is original_error
                assert saved["record"].positive
                assert prepared._planning_attempt is None
            assert prepared._planning_prepare_entered and not prepared._closed
            assert (
                prepared._planning_intake is not None
                and prepared._planning_token is not None
            )
            assert run.status == "recovery_required"
            assert rig.coordinator._runs[run.workflow_id] is run
            assert run in rig.coordinator._pending_planning
            for _ in range(2):
                with pytest.raises(RuntimeError, match=failure):
                    await rig.runtime.dispose()
                assert not rig.runtime._disposed
            with rig.service.db.connection() as conn:
                native = conn.execute(
                    "SELECT state FROM automatic_wake_attempts WHERE id=?",
                    ((prepared.phase.attempt if wrapper else saved["result"]).id,),
                ).fetchone()
                assert native[0] == ("accepted" if wrapper else "prepared")
                generation = conn.execute(
                    "SELECT r.state FROM automatic_work_reservations r JOIN automatic_wake_attempts n "
                    "ON n.generation_reservation_id=r.id WHERE n.id=?",
                    ((prepared.phase.attempt if wrapper else saved["result"]).id,),
                ).fetchone()
                assert generation[0] == ("committed" if wrapper else "reserved")
            assert rig.provider.calls == []
            print("G_PENDING_PLANNING_ORIGINAL_RECEIPT_RETAINED", flush=True)
            os._exit(17)
        stopped = await stopping
        with pytest.raises(asyncio.CancelledError):
            await command
        if shutdown:
            assert (
                rig.runtime._disposed
                and rig.runtime.execution_capacity.snapshot().closed
            )
        else:
            assert stopped.status == "stopped"
            assert stopped.control_revision == revision + 1
        assert run not in rig.coordinator._pending_planning
        assert run.workflow_id not in rig.coordinator._runs
        assert prepared._closed and prepared._finite is None
        prepared._planning_reads_returned()
        assert prepared._planning_token is prepared._planning_intake is None
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        assert [key for key, _ in rig.provider.calls] == (
            ["planning"] if repair else []
        )
        attempt = prepared._planning_attempt
        with rig.service.db.connection() as conn:
            row = conn.execute(
                "SELECT n.state,p.status,p.retirement,r.state FROM automatic_wake_attempts n "
                "JOIN workflow_phase_attempts p ON p.native_attempt_id=n.id "
                "JOIN automatic_work_reservations r ON r.id=n.generation_reservation_id "
                "WHERE n.id=?",
                (attempt.id,),
            ).fetchone()
            assert tuple(row) == (
                ("completed", "closed", "drained", "committed")
                if accepted
                else ("aborted", "closed", "drained", "released")
            )
            assert validate_workflow_rows(conn) == ()
            assert (
                conn.execute("SELECT count(*) FROM workflow_task_attempts").fetchone()[
                    0
                ]
                == 0
            )
        if not shutdown:
            again = await rig.coordinator.stop(
                run.workflow_id, expected_control_revision=revision
            )
            assert again == stopped
    finally:
        threading.setprofile_all_threads(None)
        release.set()
        revision_release.set()
        for task in (stopping, command):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await rig.close()


@pytest.mark.parametrize(
    "boundary",
    [
        "initial-prepared",
        "initial-accepted",
        "repair-prepared",
        "repair-accepted",
        "initial-prepared-repeat",
        "initial-accepted-shutdown",
        "initial-prepared-pause",
    ],
)
def test_pending_original_planning_stop_returns_same_preparation(tmp_path, boundary):
    _run(
        tmp_path,
        "planning-pending",
        boundary,
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize(
    "boundary", ["initial-prepared-unknown", "initial-accepted-wrapper"]
)
def test_pending_planning_missing_original_receipt_retains_custody(tmp_path, boundary):
    with pytest.raises(
        AssertionError, match="G_PENDING_PLANNING_ORIGINAL_RECEIPT_RETAINED"
    ):
        _run(
            tmp_path,
            "planning-pending",
            boundary,
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


@pytest.mark.parametrize(
    "boundary,marker",
    [
        ("initial-no-read", "G_PENDING_F_ORIGINAL_READ_ABSENT_RETAINED"),
        ("initial-accepted-revision", "G_PENDING_F_COMPETING_REVISION_RETAINED"),
    ],
)
def test_pending_planning_absent_read_and_competing_revision_retain_original(
    tmp_path, boundary, marker
):
    with pytest.raises(AssertionError, match=marker):
        _run(
            tmp_path,
            "planning-pending",
            boundary,
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


async def next_phase_journey(tmp_path, monkeypatch):
    import asyncio
    import copy
    import json
    import sys

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

    mode = sys.argv[2]
    dependent = sys.argv[1] == "phase-dependency"

    class FailedPredecessorProvider(WorkflowProvider):
        def __call__(self, **kwargs):
            from Tests.Chat.test_console_agent_bridge import _fence

            response = super().__call__(**kwargs)
            messages = kwargs.get("messages", kwargs.get("messages_payload", ()))
            if any(row.get("content") == "Execute task-1" for row in messages):
                response["choices"][0]["message"]["content"] = _fence(
                    "fs_read",
                    {
                        "path": "missing.txt",
                        "root_alias": self.plan["request"]["binding"]["binding_id"],
                    },
                )
            return response

    provider = FailedPredecessorProvider() if mode == "failed" else WorkflowProvider()
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, provider, approve_missing_file_read=mode == "failed"
    )
    if dependent:
        (Path(rig.workflow.request.binding.locator) / "source.py").write_text(
            "original artifact\n"
        )
        rig.provider.plan["phases"][0]["tasks"][0].update(
            outputs=["source.py"],
            footprint=[{"path": "source.py", "access": "write", "subtree": False}],
        )
        if mode == "failed":
            first = rig.provider.plan["phases"][0]["tasks"][0]
            first["allowed_tools"] = ["fs_read"]
            first["footprint"].append(
                {"path": "missing.txt", "access": "read", "subtree": False}
            )
    second = copy.deepcopy(rig.provider.plan["phases"][0])
    second["id"] = "phase-2"
    second["tasks"][0].update(id="task-2", instructions="Execute task-2")
    if dependent:
        second["tasks"][0].update(
            dependencies=["task-1"],
            inputs=["task-1:source.py"],
            outputs=[],
            footprint=[{"path": "source.py", "access": "read", "subtree": False}],
        )
    rig.provider.plan["phases"].append(second)

    async def assert_predecessor_refused(decision, before):
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            counts = tuple(
                conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in ("workflow_phase_attempts", "automatic_work_reservations")
            )
        refusal = (
            "remaining_decision_required"
            if mode == "failed"
            else "stale_workflow_decision"
        )
        with pytest.raises(ValueError, match=refusal):
            await rig.coordinator.approve(decision, command_id="invalid-predecessor")
        after = await rig.coordinator.get(rig.workflow.id)
        assert after.results == before.results and after.accounting == before.accounting
        assert rig.workflow.id not in rig.coordinator._runs
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        assert all(key != "task-2" for key, _ in rig.provider.calls)
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            assert counts == tuple(
                conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                for table in ("workflow_phase_attempts", "automatic_work_reservations")
            )

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        await rig.coordinator.approve(decision, command_id="phase-1-approve")
        if mode == "failed":
            original = rig.coordinator._runs[rig.workflow.id]
            await asyncio.shield(original.driver)
            failed = await rig.coordinator.get(rig.workflow.id)
            assert failed.status == "paused" and not failed.accounting.uncertain
            assert len(failed.results) == 1 and failed.results[0].status == "failed"
            assert original.phase._retired
            assert rig.runtime.execution_capacity.snapshot().executions == ()
            assert [key for key, _ in rig.provider.calls] == [
                "planning",
                "task-1",
                "task-1",
                "task-1",
            ]
            assert failed.accounting.used["generation"] == 2
            assert failed.accounting.used["child_launch"] == 1
            assert failed.accounting.used["model_call"] == 4
            with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
                assert (
                    rig.service.db.get_run(failed.results[0].run_id)["status"]
                    == "stuck"
                )
                usage = conn.execute(
                    "SELECT state,actual_amount FROM automatic_work_reservations WHERE chain_id=? AND kind='tokens'",
                    (original.phase.attempt.chain_id,),
                ).fetchall()
                assert len(usage) == 4 and all(
                    tuple(row) == ("settled", 10) for row in usage
                )
            refused = await rig.service.call(
                rig.service.prepare_decision, rig.workflow.id, phase_id="phase-2"
            )
            assert not refused.evidence.ready and refused.reason == "task_not_completed"
            await assert_predecessor_refused(refused, failed)
            return
        reviewed = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert len(reviewed.results) == 1 and reviewed.results[0].status == "completed"
        first_result = reviewed.results[0]
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        quality = await rig.service.call(
            rig.service.prepare_quality_review, rig.workflow.id, phase_id="phase-1"
        )
        next_review = await rig.coordinator.approve(
            quality, command_id="phase-1-quality"
        )
        assert next_review.status == "awaiting_phase_review"
        assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
        # A newly approved phase has its own exact native attempt and one task entry.
        next_decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-2"
        )
        if mode == "stale":
            (Path(rig.workflow.request.binding.locator) / "source.py").write_text(
                "external artifact edit after review\n"
            )
            refused = await rig.service.call(
                rig.service.prepare_decision, rig.workflow.id, phase_id="phase-2"
            )
            assert not refused.evidence.ready and refused.reason == "stale_task_output"
            assert next_review.accounting.used["generation"] == 2
            assert next_review.accounting.used["child_launch"] == 1
            assert next_review.accounting.used["model_call"] == 2
            await assert_predecessor_refused(next_decision, next_review)
            return
        await rig.coordinator.approve(next_decision, command_id="phase-2-approve")
        final = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert len(final.results) == 2 and all(
            result.status == "completed" for result in final.results
        )
        quality = await rig.service.call(
            rig.service.prepare_quality_review, rig.workflow.id, phase_id="phase-2"
        )
        final = await rig.coordinator.approve(quality, command_id="phase-2-quality")
        assert final.status == "completed" and final.accounting.used["generation"] == 3
        assert final.accounting.used["child_launch"] == 2
        assert final.accounting.used["model_call"] == 3
        assert [key for key, _ in rig.provider.calls] == [
            "planning",
            "task-1",
            "task-2",
        ]
        if dependent:
            assert first_result in final.results
            request = next(call for key, call in rig.provider.calls if key == "task-2")
            messages = request.get("messages", request.get("messages_payload", ()))
            assert any(
                row.get("content")
                == json.dumps(
                    [{"source": "task-1:source.py", "handoff": first_result.handoff}],
                    ensure_ascii=False,
                )
                for row in messages
            )
    finally:
        await rig.close()


def test_native_result_review_quality_acceptance_and_next_phase(tmp_path):
    _run(
        tmp_path,
        "next-phase",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_native_cross_phase_dependency_uses_original_result_handoff(tmp_path):
    _run(
        tmp_path,
        "phase-dependency",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


@pytest.mark.parametrize("mode", ["stale", "failed"])
def test_actual_prior_phase_evidence_refuses_invalid_dependent_approval(tmp_path, mode):
    _run(
        tmp_path,
        "phase-dependency",
        mode,
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def refusal_and_remaining_journey(tmp_path, monkeypatch):
    import asyncio
    import copy
    import sys

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

    partial = sys.argv[1] == "partial-resume"
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("task-1",) if partial else ())
    )
    if partial:
        second = copy.deepcopy(rig.provider.plan["phases"][0]["tasks"][0])
        second.update(
            id="task-2", instructions="Execute task-2", dependencies=["task-1"]
        )
        rig.provider.plan["phases"][0]["tasks"].append(second)
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await asyncio.shield(rig.coordinator._runs[rig.workflow.id].driver)
        before = await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        if partial:
            await rig.coordinator.approve(decision, command_id="first-partial")
            from tldw_chatbook.Agents.agent_service import WorkflowChildReceipt

            original = rig.coordinator._runs[rig.workflow.id]
            while True:
                prepared = original.preparations.get("task-1")
                launches = (
                    prepared.authorization._native_launch
                    if prepared is not None
                    else ()
                )
                if launches and type(launches[0]) is WorkflowChildReceipt:
                    receipt = launches[0]
                    authorization = prepared.authorization
                    native = authorization._native_preparation[0]
                    assert prepared.phase is original.phase is native.phase
                    assert original.phase._task_scopes["task-1"] is authorization
                    assert authorization._bridge_preparation[0] is prepared
                    assert receipt.attempt_id == authorization.claim.id
                    assert receipt.run_id == authorization.claim.launch_run_id
                    assert receipt.owner.is_admitted
                    break
                assert not original.driver.done(), (
                    "original task-1 launch did not return"
                )
                await asyncio.sleep(0.01)
            await rig.wait_for_provider_entry("task-1")
            running = await rig.coordinator.get(rig.workflow.id)
            assert rig.coordinator._runs[rig.workflow.id] is original
            paused = await rig.coordinator.pause(
                rig.workflow.id, expected_control_revision=running.control_revision
            )
            assert paused.status == "pausing" and not original.driver.done()
            assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
            rig.provider_release()
            await asyncio.shield(original.driver)
            paused = await rig.wait_for_status("paused")
            assert len(paused.results) == 1 and paused.results[0].status == "completed"
            retained = paused.results[0]
            assert (
                not paused.accounting.uncertain
                and rig.runtime.execution_capacity.snapshot().executions == ()
            )
            before = await rig.coordinator.resume(
                rig.workflow.id,
                expected_control_revision=paused.control_revision,
                command_id="partial-resume",
            )
            assert (
                before.status == "awaiting_phase_review"
                and before.plans == paused.plans
            )
            decision = await rig.service.call(
                rig.service.prepare_remaining_decision,
                rig.workflow.id,
                phase_id="phase-1",
            )
            assert (
                decision.reason == "remaining_ready"
                and retained.attempt_id in decision.result_refs
            )
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            counts = tuple(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("workflow_phase_attempts", "automatic_work_reservations")
            )
        stale = decision.model_copy(
            update={"control_revision": decision.control_revision - 1}
        )
        with pytest.raises(ValueError, match="stale_workflow_decision") as refusal:
            await rig.coordinator.approve(stale, command_id="stale-approval")
        command = rig.coordinator._commands[("approve", "stale-approval")][1]
        assert command.exception() is refusal.value
        from tldw_chatbook.Chat.console_workflow_runs import (
            _accept_remaining_workflow_phase,
        )

        traceback = refusal.value.__traceback__
        while (
            traceback.tb_frame.f_code is not _accept_remaining_workflow_phase.__code__
        ):
            traceback = traceback.tb_next
        cell = traceback.tb_frame.f_locals["cell"]
        assert cell.body_error is refusal.value and cell.preparation_entered is False
        assert cell.owner_return == [True] and not cell.owner.is_admitted
        assert (
            cell.read_return[1].exception() is None
            and cell.read_return[2].result() is None
        )
        assert cell.claim.id not in rig.runtime.worktree_activity._claims
        with pytest.raises(ValueError) as duplicate:
            await rig.coordinator.approve(stale, command_id="stale-approval")
        assert duplicate.value is refusal.value
        assert rig.workflow.id not in rig.coordinator._runs
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            assert (
                tuple(
                    conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    for table in (
                        "workflow_phase_attempts",
                        "automatic_work_reservations",
                    )
                )
                == counts
            )
        current = await rig.coordinator.get(rig.workflow.id)
        assert (
            current.accounting == before.accounting
            and current.results == before.results
        )
        await rig.coordinator.approve(decision, command_id="separate-valid-approval")
        complete = await rig.wait_for_status("awaiting_result_review", timeout=20)
        assert all(result.status == "completed" for result in complete.results)
        assert [key for key, _ in rig.provider.calls] == (
            ["planning", "task-1", "task-2"] if partial else ["planning", "task-1"]
        )
        assert complete.accounting.used["generation"] == (3 if partial else 2)
        assert complete.accounting.used["child_launch"] == (2 if partial else 1)
        if partial:
            assert complete.results[0] == retained
            assert complete.plans == before.plans
        quality = await rig.service.call(
            rig.service.prepare_quality_review, rig.workflow.id, phase_id="phase-1"
        )
        assert (
            await rig.coordinator.approve(quality, command_id="quality-complete")
        ).status == "completed"
    finally:
        rig.provider_release()
        await rig.close()


@pytest.mark.parametrize("mode", ["stale-initial", "partial-resume"])
def test_refusal_returns_admission_and_explicit_remaining_preserves_sibling(
    tmp_path, mode
):
    _run(
        tmp_path,
        mode,
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def repair_journey(tmp_path, monkeypatch):
    import asyncio
    import hashlib
    import json

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.LLM_Calls.provider_outcomes import PreEffectPermanentError

    class LocalRefusalProvider(WorkflowProvider):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            if self.calls[-1][0] == "task-1":
                raise PreEffectPermanentError(
                    "trusted local fixture refused before dispatch"
                )
            return result

    rig = await build_workflow_rig(tmp_path, monkeypatch, LocalRefusalProvider())
    instruction = "Repair the failed task while preserving its original evidence."
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        planning = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(planning.driver)
        initial = await rig.coordinator.get(rig.workflow.id)
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        await rig.coordinator.approve(decision, command_id="failed-phase")
        execution = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(execution.driver)
        failed = await rig.coordinator.get(rig.workflow.id)
        assert failed.status == "paused" and len(failed.results) == 1
        assert failed.results[0].status == "failed" and not failed.accounting.uncertain
        assert (
            failed.accounting.used["generation"] == 2
            and failed.accounting.used["child_launch"] == 1
        )
        assert (
            failed.accounting.used["model_call"] == 2
            and failed.accounting.used["tokens"] == 10
        )
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        with pytest.raises(ValueError, match="remaining_requires_repair"):
            await rig.service.call(
                rig.service.prepare_remaining_decision,
                rig.workflow.id,
                phase_id="phase-1",
            )
        rig.provider.plan["phases"][0]["tasks"][0]["instructions"] = (
            "Repaired task-1 requires new approval"
        )
        await rig.coordinator.revise(
            rig.workflow.id,
            expected_control_revision=failed.control_revision,
            instruction=instruction,
            command_id="repair-explicit",
        )
        repair = rig.coordinator._runs[rig.workflow.id]
        assert repair.prepared._planning_repair_source[3] == instruction
        assert (
            repair.phase.attempt.repair_instruction_sha256
            == hashlib.sha256(instruction.encode()).hexdigest()
        )
        await asyncio.shield(repair.driver)
        reviewed = await rig.coordinator.get(rig.workflow.id)
        assert reviewed.status == "awaiting_phase_review" and len(reviewed.plans) == 2
        assert (
            reviewed.request == initial.request
            and reviewed.plans[0] == initial.plans[0]
        )
        assert reviewed.results == failed.results
        assert (
            reviewed.accounting.used["generation"] == 3
            and reviewed.accounting.used["child_launch"] == 1
        )
        assert (
            reviewed.accounting.used["model_call"] == 3
            and reviewed.accounting.used["tokens"] == 20
        )
        assert not reviewed.accounting.uncertain
        messages = rig.provider.calls[-1][1].get(
            "messages", rig.provider.calls[-1][1].get("messages_payload")
        )
        seed = next(
            row["content"].split("\n", 1)[1]
            for row in messages
            if row.get("content", "").startswith(
                "Produce exactly one strict JSON WorkflowPlan"
            )
        )
        body = json.loads(seed)["workflow_planning"]
        assert body["repair_instruction"] == instruction
        assert body["previous_plans"] == [initial.plans[0].model_dump(mode="json")]
        assert body["repair_evidence_untrusted"][0]["result"] == failed.results[
            0
        ].model_dump(mode="json")
        await rig.coordinator.revise(
            rig.workflow.id,
            expected_control_revision=failed.control_revision,
            instruction=instruction,
            command_id="repair-explicit",
        )
        assert [key for key, _kwargs in rig.provider.calls] == [
            "planning",
            "task-1",
            "planning",
        ]
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            attempts = conn.execute(
                "SELECT status,retirement FROM workflow_task_attempts"
            ).fetchall()
            tokens = conn.execute(
                "SELECT state,actual_amount FROM automatic_work_reservations WHERE kind='tokens' ORDER BY created_at"
            ).fetchall()
            phases = conn.execute(
                "SELECT purpose,retirement FROM workflow_phase_attempts ORDER BY ordinal"
            ).fetchall()
        assert [tuple(row) for row in attempts] == [("failed", "drained")]
        assert [tuple(row) for row in tokens] == [
            ("settled", 10),
            ("settled", 0),
            ("settled", 10),
        ]
        assert [tuple(row) for row in phases] == [
            ("planning", "drained"),
            ("execution", "drained"),
            ("planning", "drained"),
        ]
        assert (
            await rig.coordinator.get(rig.workflow.id)
        ).accounting == reviewed.accounting
        assert rig.runtime.execution_capacity.snapshot().executions == ()
    finally:
        await rig.close()


def test_native_failed_run_requires_explicit_repair_preserving_history_and_spend(
    tmp_path,
):
    _run(
        tmp_path,
        "repair",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )
