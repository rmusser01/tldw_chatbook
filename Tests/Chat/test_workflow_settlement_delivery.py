"""Stop revokes work while physical ownership remains visible until retirement."""

import threading
from pathlib import Path

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

pytestmark = pytest.mark.bootstrap_profile


async def journey(tmp_path, monkeypatch):
    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Chat.console_provider_gateway import _automatic_generation

    original = _automatic_generation.__wrapped__.__code__
    observed = []

    def hold_known_usage(frame, event, _argument):
        if (
            event != "return"
            or frame.f_code is not original
            or not frame.f_locals.get("completed")
        ):
            return
        automatic = frame.f_locals["automatic"]
        if automatic is None or automatic.workflow is None or observed:
            return
        assert frame.f_locals["actual"] == 10
        observed.append((automatic.workflow, frame.f_locals["reservation"]))
        rig.provider.entered.setdefault("known-usage", threading.Event()).set()
        print(
            "G_ORIGINAL_KNOWN_USAGE_RETURN_HELD", frame.f_locals["actual"], flush=True
        )
        assert rig.provider.released.wait(10)

    threading.setprofile_all_threads(hold_known_usage)
    try:
        started = rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_provider_entry("known-usage")
        assert observed[0][0] is rig.coordinator._runs[rig.workflow.id].phase
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            row = conn.execute(
                "SELECT state,actual_amount FROM automatic_work_reservations WHERE id=?",
                (observed[0][1],),
            ).fetchone()
        assert tuple(row) == ("settled", 10)
        stopped = await rig.coordinator.stop(
            rig.workflow.id, expected_control_revision=started.control_revision
        )
        assert stopped.status == "stopping"
        assert rig.runtime.execution_capacity.snapshot().executions
        rig.provider_release()
        settled = await rig.wait_for_status("stopped")
        assert settled.plans == ()
        assert settled.accounting.used["generation"] == 1
        with pytest.raises(ValueError, match="workflow_terminal"):
            await rig.coordinator.pause(
                rig.workflow.id, expected_control_revision=settled.control_revision
            )
        with pytest.raises(ValueError, match="resume_requires_clean_checkpoint"):
            await rig.coordinator.resume(
                rig.workflow.id,
                expected_control_revision=settled.control_revision,
                command_id="no-plan-resume",
            )
        unchanged = await rig.coordinator.get(rig.workflow.id)
        assert unchanged.plans == () and unchanged.accounting == settled.accounting
        assert [key for key, _kwargs in rig.provider.calls] == ["planning"]
    finally:
        rig.provider_release()
        threading.setprofile_all_threads(None)
        await rig.close()


def test_stop_held_planner_never_publishes_a_late_plan(tmp_path):
    _run(
        tmp_path,
        "settlement",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def unknown_usage_journey(tmp_path, monkeypatch):
    from Tests.Chat.test_workflow_recovery_cancellation import cancelled_planner_journey

    await cancelled_planner_journey(tmp_path)


def test_stop_before_usage_preserves_charges_and_releases_returned_resources(tmp_path):
    _run(
        tmp_path,
        "unknown-usage",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def pending_acceptance_journey(tmp_path, monkeypatch):
    import asyncio

    from tldw_chatbook.Chat.console_workflow_runs import _accept_workflow_phase

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    entered, release = threading.Event(), threading.Event()
    observed = []
    original = _accept_workflow_phase.__code__

    def hold_original_acceptance(frame, event, result):
        if (
            event == "return"
            and frame.f_code is original
            and result is not None
            and result.attempt.purpose == "execution"
        ):
            observed.append(result)
            entered.set()
            assert release.wait(10)

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        threading.setprofile_all_threads(hold_original_acceptance)
        approval = asyncio.create_task(
            rig.coordinator.approve(decision, command_id="held-initial")
        )
        assert await asyncio.to_thread(entered.wait, 10)
        accepted = await rig.service.call(rig.service.get, rig.workflow.id)
        assert accepted.control_revision == decision.control_revision + 1
        pending = rig.coordinator._runs[rig.workflow.id]
        stop = asyncio.create_task(
            rig.coordinator.stop(
                rig.workflow.id, expected_control_revision=accepted.control_revision
            )
        )

        async def cancellation_arrives():
            while not pending.issuing.cancelling():
                if stop.done():
                    await stop
                    raise AssertionError(
                        "Stop returned before cancelling original acceptance"
                    )
                await asyncio.sleep(0.01)

        await asyncio.wait_for(cancellation_arrives(), 5)
        assert not stop.done()
        assert rig.runtime.execution_capacity.snapshot().executions
        release.set()
        stopped = await stop
        with pytest.raises(asyncio.CancelledError):
            await approval
        assert stopped.status == "stopped"
        assert rig.runtime.execution_capacity.snapshot().executions == ()
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        assert rig.workflow.id not in rig.coordinator._runs
        duplicate = await rig.coordinator.stop(
            rig.workflow.id, expected_control_revision=accepted.control_revision
        )
        assert duplicate == stopped
        with pytest.raises(asyncio.CancelledError):
            await rig.coordinator.approve(decision, command_id="held-initial")
        with rig.service.db.connection() as conn:
            attempts = conn.execute(
                "SELECT status,retirement FROM workflow_phase_attempts WHERE purpose='execution'"
            ).fetchall()
            saved = conn.execute(
                "SELECT payload_json FROM workflow_decisions WHERE id=?", (decision.id,)
            ).fetchone()
        assert [tuple(row) for row in attempts] == [("closed", "drained")]
        assert saved is not None and observed[0].attempt.workflow_id == rig.workflow.id
    finally:
        release.set()
        threading.setprofile_all_threads(None)
        await rig.close()


def test_stop_pending_initial_acceptance_drains_original_cell(tmp_path):
    _run(
        tmp_path,
        "pending-acceptance",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def pending_wrapper_failure_journey(tmp_path, monkeypatch):
    import asyncio
    import types

    from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
    from tldw_chatbook.Chat.console_workflow_runs import _RemainingAcceptance

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    entered, release = threading.Event(), threading.Event()
    observed = []
    invoke = next(
        code
        for code in run_preparation_read.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    failure = RuntimeError("original E wrapper failed after callback return")

    def fail_original_wrapper(frame, event, _result):
        if event != "return" or frame.f_code is not invoke or observed:
            return
        read = frame.f_locals["read"]
        if type(read.source) is not _RemainingAcceptance or read.source.phase is None:
            return
        observed.append(read)
        entered.set()
        assert release.wait(10)
        raise failure

    rig.workflow = await rig.coordinator.start(
        rig.workflow.request, launch_id=rig.workflow.launch_id
    )
    await rig.wait_for_status("awaiting_phase_review")
    decision = await rig.service.call(
        rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
    )
    threading.setprofile_all_threads(fail_original_wrapper)
    approval = asyncio.create_task(
        rig.coordinator.approve(decision, command_id="unknown-initial")
    )
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        run = rig.coordinator._runs[rig.workflow.id]
        cell = observed[0].source
        assert run.acceptance is cell and cell.scopes_returned and not cell.returned
        approval.cancel()
        await asyncio.sleep(0)
        approval.cancel()
        assert not approval.done() and cell.owner.is_admitted
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await approval
        assert (
            observed[0]._producer.done()
            and observed[0]._producer.exception() is failure
        )
        assert observed[0].retired.done()
        assert cell.scopes_returned and cell.returned and cell.terminal_scopes_returned
        assert cell.phase is None and cell.terminal_result.status == "awaiting_resume"
        assert not cell.owner.is_admitted and cell.owner_return == [True]
        assert cell.claim.id not in rig.runtime.worktree_activity._claims
        assert rig.workflow.id not in rig.coordinator._runs
        assert not rig.runtime.execution_capacity.snapshot().executions
        assert [key for key, _ in rig.provider.calls] == ["planning"]
    finally:
        release.set()
        threading.setprofile_all_threads(None)
        await rig.close()


def test_pending_wrapper_error_and_repeated_cancel_return_unused_native_admission(
    tmp_path,
):
    _run(
        tmp_path,
        "pending-wrapper",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def planning_control_return_journey(tmp_path, monkeypatch):
    import asyncio
    import sys
    import types

    from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
    from tldw_chatbook.Chat.console_provider_gateway import _automatic_generation
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply
    from tldw_chatbook.Chat.console_workflow_runs import WorkflowPhaseAuthorization

    mode = sys.argv[1]
    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    usage_ready, control_held, terminal = (threading.Event() for _ in range(3))
    control_release = threading.Event()
    usage_code = _automatic_generation.__wrapped__.__code__
    invoke_code = next(
        code
        for code in run_preparation_read.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    planning_code = next(
        code
        for code in _PreparedWorkflowReply.run_planning.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    failure = RuntimeError("original control wrapper return failed")
    observed = []

    def hold_original_returns(frame, event, _result):
        if event != "return":
            return
        if (
            frame.f_code is usage_code
            and frame.f_locals.get("completed")
            and not usage_ready.is_set()
        ):
            assert frame.f_locals["actual"] == 10
            usage_ready.set()
            assert rig.provider.released.wait(10)
        elif frame.f_code is invoke_code and not control_held.is_set():
            read = frame.f_locals["read"]
            if type(read.source) is WorkflowPhaseAuthorization:
                observed.append(read)
                control_held.set()
                assert control_release.wait(10)
                if mode == "control-unknown":
                    raise failure
        elif frame.f_code is planning_code:
            terminal.set()

    threading.setprofile_all_threads(hold_original_returns)
    retained = mode == "control-unknown"
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        assert await asyncio.to_thread(usage_ready.wait, 10)
        run = rig.coordinator._runs[rig.workflow.id]
        prepared = run.prepared
        owner, claim = prepared.primary_owner, prepared.primary_claim
        with pytest.raises(PermissionError, match="G planning association required"):
            await prepared.run_planning()
        stop = asyncio.create_task(
            rig.coordinator.stop(
                rig.workflow.id, expected_control_revision=rig.workflow.control_revision
            )
        )
        assert await asyncio.to_thread(control_held.wait, 10)
        original_event = run.control_return
        assert not original_event.is_set()
        current = await rig.service.call(rig.service.get, rig.workflow.id)
        with pytest.raises(PermissionError, match="control pending"):
            await rig.coordinator.pause(
                rig.workflow.id, expected_control_revision=current.control_revision
            )
        assert run.control_return is original_event
        if mode == "control-association":
            rig.coordinator._plan = lambda *_args: None
        rig.provider_release()
        assert await asyncio.to_thread(terminal.wait, 10)
        # Wait for the original F tracked wrapper, not merely its callback flag.
        read, producer, retired = prepared._planning_read_return

        async def original_read_returns():
            while not producer.done() or not retired.done():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(original_read_returns(), 5)
        assert producer.exception() is None and read.retired is retired
        assert owner.is_admitted and claim.id in rig.runtime.worktree_activity._claims
        assert (
            prepared._planning_intake is not None
            and prepared._planning_token is not None
        )
        if mode == "control-driver-cancel":
            run.driver.cancel()
            await asyncio.sleep(0)
            run.driver.cancel()
            await asyncio.sleep(0)
            assert not run.driver.done() and owner.is_admitted
        stop.cancel()
        await asyncio.sleep(0)
        stop.cancel()
        assert not stop.done()
        control_release.set()
        with pytest.raises(asyncio.CancelledError):
            await stop
        if mode == "control-driver-cancel":
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(run.driver), 5)
        else:
            await asyncio.wait_for(asyncio.shield(run.driver), 5)
        if retained:
            assert (
                owner.is_admitted and claim.id in rig.runtime.worktree_activity._claims
            )
            assert (
                prepared._planning_intake is not None
                and prepared._planning_token is not None
            )
            assert run.status == "recovery_required"
            if mode == "control-unknown":
                assert (
                    run.control_unknown and observed[0]._producer.exception() is failure
                )
            print("G_ORIGINAL_F_CONTROL_RETURN_CUSTODY_RETAINED", mode, flush=True)
        else:
            assert (
                not owner.is_admitted
                and claim.id not in rig.runtime.worktree_activity._claims
            )
            assert (
                prepared._planning_intake is None and prepared._planning_token is None
            )
            assert rig.runtime.execution_capacity.snapshot().executions == ()
            if mode != "control-driver-cancel":
                assert (await rig.coordinator.get(rig.workflow.id)).status == "stopped"
    finally:
        control_release.set()
        rig.provider_release()
        threading.setprofile_all_threads(None)
        if not retained:
            await rig.close()
    if retained:
        raise RuntimeError("G_REQUIRED_CUSTODY_RETAINED_F_CONTROL_RETURN")


@pytest.mark.parametrize(
    "mode",
    [
        "control-positive",
        "control-driver-cancel",
        "control-unknown",
        "control-association",
    ],
)
def test_planning_root_waits_for_original_control_return(tmp_path, mode):
    if mode == "control-unknown":
        with pytest.raises(
            AssertionError, match="G_REQUIRED_CUSTODY_RETAINED_F_CONTROL_RETURN"
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


async def pending_shutdown_journey(tmp_path, monkeypatch):
    import asyncio

    from tldw_chatbook.Chat.console_workflow_runs import _accept_workflow_phase

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    entered, release = threading.Event(), threading.Event()
    original = _accept_workflow_phase.__code__

    def hold_accepted(frame, event, result):
        if (
            event == "return"
            and frame.f_code is original
            and result is not None
            and result.attempt.purpose == "execution"
        ):
            entered.set()
            assert release.wait(10)

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        threading.setprofile_all_threads(hold_accepted)
        approval = asyncio.create_task(
            rig.coordinator.approve(decision, command_id="shutdown-initial")
        )
        assert await asyncio.to_thread(entered.wait, 10)
        run = rig.coordinator._runs[rig.workflow.id]
        cell = run.acceptance
        closing = asyncio.create_task(rig.runtime.dispose())

        async def original_cancel():
            while not run.issuing.cancelling():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(original_cancel(), 5)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        assert not closing.done() and cell.owner.is_admitted
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await closing
        with pytest.raises(asyncio.CancelledError):
            await approval
        assert (
            cell.returned
            and cell.terminal_scopes_returned
            and cell.owner_return == [True]
        )
        assert cell.terminal_result.status == "awaiting_resume"
        assert cell.terminal_result.control_revision == decision.control_revision + 2
        assert (
            cell.read_return[1].exception() is None
            and cell.terminal_read_return[1].exception() is None
        )
        assert cell.read_return[2].done() and cell.terminal_read_return[2].done()
        assert (
            not cell.owner.is_admitted
            and cell.claim.id not in rig.runtime.worktree_activity._claims
        )
        assert rig.workflow.id not in rig.coordinator._runs
        assert [key for key, _ in rig.provider.calls] == ["planning"]
    finally:
        release.set()
        threading.setprofile_all_threads(None)
        await rig.close()


def test_shutdown_after_initial_commit_awaits_original_unused_return(tmp_path):
    _run(
        tmp_path,
        "pending-shutdown",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def incomplete_acceptance_journey(tmp_path, monkeypatch):
    import sys

    from tldw_chatbook.Chat.console_preparation_reads import run_preparation_read
    from tldw_chatbook.Chat.console_workflow_runs import _RemainingAcceptance

    mode = sys.argv[1]
    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    prepare_code = rig.service.db.automatic_work._prepare_workflow.__code__
    failure = RuntimeError("original acceptance creation return failed")
    observed = []

    def fail_original_creation(frame, event, result):
        if observed:
            return
        if (
            mode == "pending-prepared"
            and event == "return"
            and frame.f_code is prepare_code
            and result is not None
            and result.purpose == "execution"
        ):
            observed.append(result)
            raise failure
        if (
            mode == "pending-no-reader"
            and event == "call"
            and frame.f_code is run_preparation_read.__code__
            and type(frame.f_locals.get("source")) is _RemainingAcceptance
        ):
            observed.append(frame.f_locals["source"])
            raise failure

    rig.workflow = await rig.coordinator.start(
        rig.workflow.request, launch_id=rig.workflow.launch_id
    )
    await rig.wait_for_status("awaiting_phase_review")
    decision = await rig.service.call(
        rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
    )
    threading.setprofile_all_threads(fail_original_creation)
    try:
        with pytest.raises(RuntimeError):
            await rig.coordinator.approve(decision, command_id=mode)
        assert observed
        run = rig.coordinator._runs[rig.workflow.id]
        cell = run.acceptance
        assert (
            cell is not None and run.status == "recovery_required" and run.phase is None
        )
        if mode == "pending-prepared":
            assert cell.returned and cell.owner_return == [True]
            assert cell.body_error is failure and cell.attempt is None
            with rig.service.db.connection() as conn:
                assert (
                    conn.execute(
                        "SELECT status FROM workflow_phase_attempts WHERE purpose='execution'"
                    ).fetchone()[0]
                    == "prepared"
                )
        else:
            assert cell.read_return is None and cell.owner is None and not cell.returned
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        for _ in range(2):
            with pytest.raises(
                RuntimeError, match="workflow_original_shutdown_unresolved"
            ):
                await rig.runtime.dispose()
            assert (
                not rig.runtime._disposed
                and not rig.runtime.execution_capacity.snapshot().closed
            )
            assert rig.coordinator._runs[rig.workflow.id] is run
            assert not rig.controller.fleet_wake.try_claim_automatic_primary(
                run.session_id, object()
            )
    finally:
        threading.setprofile_all_threads(None)
    raise RuntimeError("G_REQUIRED_CUSTODY_RETAINED_INCOMPLETE_ACCEPTANCE")


@pytest.mark.parametrize("mode", ["pending-prepared", "pending-no-reader"])
def test_incomplete_original_acceptance_refuses_repeated_shutdown(tmp_path, mode):
    with pytest.raises(
        AssertionError, match="G_REQUIRED_CUSTODY_RETAINED_INCOMPLETE_ACCEPTANCE"
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


async def final_metadata_journey(tmp_path, monkeypatch):
    import asyncio
    import sys

    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.Chat import console_preparation_reads as native

    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("planning",))
    )
    held, release = threading.Event(), threading.Event()
    original = None
    close = None
    seen = []
    unsubscribe = rig.coordinator.subscribe(seen.append)
    snapshot_code = WorkflowRunService.runtime_snapshot.__wrapped__.__code__
    direct_close = sys.argv[1] == "metadata-direct-close"
    late_failure = sys.argv[1] == "metadata-late-failure"
    uncertain = late_failure

    failure = RuntimeError("original final metadata wrapper failed")
    invoke_code = next(
        code
        for code in native.run_preparation_read.__code__.co_consts
        if getattr(code, "co_name", None) == "invoke"
    )
    snapshot_returned = False
    retained = False

    def hold_original_final_snapshot(frame, event, result):
        nonlocal snapshot_returned
        if (
            event == "return"
            and frame.f_code is snapshot_code
            and original is not None
            and frame.f_locals.get("self") is rig.service
            and result is not None
            and result.status == "awaiting_phase_review"
            and original.prepared._planning_settled
            and not original.prepared.primary_owner.is_admitted
        ):
            snapshot_returned = True
            if late_failure:
                return
        elif not (
            late_failure
            and snapshot_returned
            and event == "return"
            and frame.f_code is invoke_code
            and frame.f_locals["read"].source is rig.coordinator
            and frame.f_locals["read"].task is original.driver
        ):
            return
        threading.setprofile_all_threads(None)
        held.set()
        assert release.wait(10), "original final metadata worker was not released"
        if late_failure:
            raise failure

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        original = rig.coordinator._runs[rig.workflow.id]
        await rig.wait_for_provider_entry("planning")
        threading.setprofile_all_threads(hold_original_final_snapshot)
        rig.provider_release()
        assert await asyncio.to_thread(held.wait, 15)
        assert rig.coordinator._runs.get(rig.workflow.id) is original
        reads = [
            read
            for read in rig.runtime._preparation_reads
            if read.task is original.driver
        ]
        assert len(reads) == 1
        read = reads[0]
        close = asyncio.create_task(
            rig.coordinator.close() if direct_close else rig.runtime.dispose()
        )
        await asyncio.sleep(0)
        callbacks = len(seen)
        for _ in range(3):
            original.driver.cancel()
            close.cancel()
            await asyncio.sleep(0.01)
            assert not original.driver.done() and not close.done()
            assert rig.coordinator._runs[rig.workflow.id] is original
            assert (
                not rig.runtime._disposed
                and not rig.runtime.execution_capacity.snapshot().closed
            )
            assert not read._producer.done() and not read.retired.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await close
        with pytest.raises(asyncio.CancelledError):
            await original.driver
        if uncertain:
            cell = original.final_metadata
            assert cell.proof[0] is read and cell.proof[1] is read._producer
            assert read._producer.exception() is cell.producer_error is failure
            assert isinstance(cell.caller_error, asyncio.CancelledError)
            assert cell.source_error is None and cell.disposed.result() is True
            assert cell not in rig.coordinator._metadata_cells
            assert original.driver.cancelled()
        assert (
            read._producer.done()
            and not read._producer.cancelled()
            and read._producer.exception() is (failure if uncertain else None)
        )
        assert read.retired.done() and read.retired.result() is None
        assert rig.workflow.id not in rig.coordinator._runs and len(seen) == callbacks
        assert rig.runtime._disposed is (not direct_close)
        assert rig.runtime.execution_capacity.snapshot().closed is (not direct_close)
    finally:
        release.set()
        rig.provider_release()
        threading.setprofile_all_threads(None)
        if original is not None and not original.driver.done():
            await asyncio.shield(original.driver)
        if close is not None and not close.done():
            await asyncio.shield(close)
        unsubscribe()
        if not retained:
            await rig.close()


def test_final_original_metadata_worker_and_host_return_precede_shutdown(tmp_path):
    _run(
        tmp_path,
        "metadata-positive",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_final_metadata_preserves_cancellation_and_post_return_error_without_retaining_resources(
    tmp_path,
):
    _run(
        tmp_path,
        "metadata-late-failure",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def notification_failure_journey(tmp_path, monkeypatch):
    import asyncio

    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(), change_review=True
    )
    notified = []
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_status("awaiting_phase_review")
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )

        def failed_notification(binding):
            notified.append(binding)
            raise RuntimeError("notification delivery failed")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                rig.coordinator, "_notify_execution_release", failed_notification
            )
            await rig.coordinator.approve(decision, command_id="notification-failure")
            run = rig.coordinator._runs[rig.workflow.id]
            with pytest.raises(RuntimeError, match="notification delivery failed"):
                await asyncio.shield(run.driver)
        assert len(notified) == 1
        assert rig.workflow.id not in rig.coordinator._runs
        assert not rig.runtime.execution_capacity.snapshot().executions
        assert not rig.runtime.worktree_activity._claims
        snapshot = await rig.coordinator.get(rig.workflow.id)
        assert snapshot.status == "awaiting_result_review"
        assert (
            snapshot.results[0].status == "completed"
            and not snapshot.accounting.uncertain
        )
    finally:
        await rig.close()


def test_notification_failure_does_not_resurrect_returned_native_execution(tmp_path):
    _run(
        tmp_path,
        "notification-failure",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def external_metadata_journey(tmp_path, monkeypatch):
    import asyncio

    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    held, release = threading.Event(), threading.Event()
    unrelated = asyncio.Event()
    returned = asyncio.Event()
    caller = closing = None
    observed = []
    original_code = WorkflowRunService.runtime_snapshot.__wrapped__.__code__

    def hold_read(frame, event, result):
        if (
            event == "return"
            and frame.f_code is original_code
            and frame.f_locals.get("self") is rig.service
        ):
            threading.setprofile_all_threads(None)
            held.set()
            assert release.wait(10)

    async def external_caller():
        observed.append(await rig.coordinator.get(rig.workflow.id))
        returned.set()
        await unrelated.wait()

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        original = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(original.driver)
        assert (
            await rig.coordinator.get(rig.workflow.id)
        ).status == "awaiting_phase_review"
        assert rig.workflow.id not in rig.coordinator._runs
        threading.setprofile_all_threads(hold_read)
        caller = asyncio.create_task(external_caller())
        assert await asyncio.to_thread(held.wait, 10)
        (cell,) = rig.coordinator._metadata_cells
        read = cell.proof[0]
        assert cell.issuing is caller and cell.run is None and cell.phase is None
        assert read.task is caller and read.source is rig.coordinator
        closing = asyncio.create_task(rig.runtime.dispose())
        await asyncio.sleep(0.01)
        assert not closing.done() and not cell.disposed.done()
        with pytest.raises(RuntimeError, match="workflow coordinator closed"):
            await rig.coordinator.get(rig.workflow.id)
        release.set()
        await asyncio.wait_for(returned.wait(), 10)
        await asyncio.wait_for(asyncio.shield(closing), 10)
        assert not caller.done(), (
            "close joined unrelated work after external get returned"
        )
        assert (
            observed[0].id == rig.workflow.id
            and observed[0].status == "awaiting_phase_review"
        )
        assert read._producer.exception() is None and read.retired.result() is None
        assert cell.disposed.result() is True and not rig.coordinator._metadata_cells
        unrelated.set()
        await caller
    finally:
        release.set()
        unrelated.set()
        threading.setprofile_all_threads(None)
        if caller is not None and not caller.done():
            await asyncio.shield(caller)
        if closing is not None and not closing.done():
            await asyncio.shield(closing)
        await rig.close()


def test_shutdown_drains_external_metadata_host_disposition_without_joining_caller(
    tmp_path,
):
    _run(
        tmp_path,
        "metadata-external",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def coherent_metadata_journey(tmp_path, monkeypatch):
    import asyncio

    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    held, release, committed = threading.Event(), threading.Event(), threading.Event()
    snapshots, committed_snapshots, views = [], [], []
    first = second = pause = None
    get_code = WorkflowRunService.get.__wrapped__.__code__
    view_code = WorkflowRunService.runtime_snapshot.__wrapped__.__code__
    control_code = WorkflowRunService.set_status.__wrapped__.__code__

    def observe_original_reads(frame, event, result):
        if event != "return" or frame.f_locals.get("self") is not rig.service:
            return
        if frame.f_code is get_code and not held.is_set():
            parent = frame.f_back
            while parent is not None and parent.f_code.co_name != "read_metadata":
                parent = parent.f_back
            assert parent is not None and parent.f_locals["conn"].in_transaction
            snapshots.append(result)
            held.set()
            assert release.wait(10)
        elif frame.f_code is control_code:
            committed_snapshots.append(result)
            committed.set()
        elif frame.f_code is view_code and held.is_set():
            views.append(result)
            threading.setprofile_all_threads(None)

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        original = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(original.driver)
        before = await rig.coordinator.get(rig.workflow.id)
        assert before.status == "awaiting_phase_review"
        threading.setprofile_all_threads(observe_original_reads)
        first = asyncio.create_task(rig.coordinator.get(rig.workflow.id))
        assert await asyncio.to_thread(held.wait, 10)
        (cell,) = rig.coordinator._metadata_cells
        pause = asyncio.create_task(
            rig.coordinator.pause(
                rig.workflow.id, expected_control_revision=before.control_revision
            )
        )
        assert await asyncio.to_thread(committed.wait, 10)
        assert committed_snapshots[0].control_revision == before.control_revision + 1
        second = asyncio.create_task(rig.coordinator.get(rig.workflow.id))
        await asyncio.sleep(0.01)
        second_cell = next(
            item for item in rig.coordinator._metadata_cells if item.issuing is second
        )
        assert second_cell.proof is None and not second_cell.entered
        assert not first.done() and not second.done() and not pause.done()
        release.set()
        old = await asyncio.wait_for(asyncio.shield(first), 10)
        new = await asyncio.wait_for(asyncio.shield(second), 10)
        paused = await asyncio.wait_for(asyncio.shield(pause), 10)
        assert (
            old.control_revision
            == snapshots[0].control_revision
            == views[0].control_revision
            == before.control_revision
        )
        assert old.status == views[0].status == "awaiting_phase_review"
        assert (
            new.control_revision
            == paused.control_revision
            == before.control_revision + 1
        )
        assert new.status == paused.status == "paused"
        assert cell.disposed.result() is True and second_cell.disposed.result() is True
        assert (
            cell.proof[1].exception() is None
            and second_cell.proof[1].exception() is None
        )
        assert not rig.coordinator._metadata_cells
    finally:
        release.set()
        threading.setprofile_all_threads(None)
        for task in (first, second, pause):
            if task is not None and not task.done():
                await asyncio.shield(task)
        await rig.close()


def test_concurrent_get_uses_one_snapshot_and_sequences_original_host_disposition(
    tmp_path,
):
    _run(
        tmp_path,
        "metadata-coherent",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def equal_counter_metadata_journey(tmp_path, monkeypatch):
    import asyncio

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Chat.console_provider_gateway import _automatic_generation

    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("planning",))
    )
    held, release = threading.Event(), threading.Event()
    seen, reservation = [], []
    unsubscribe = rig.coordinator.subscribe(seen.append)
    original_code = _automatic_generation.__wrapped__.__code__

    def hold_known_settlement(frame, event, result):
        if (
            event != "return"
            or frame.f_code is not original_code
            or not frame.f_locals.get("completed")
        ):
            return
        automatic = frame.f_locals["automatic"]
        if automatic is None or automatic.workflow is not original.phase:
            return
        assert frame.f_locals["actual"] == 10
        reservation.append(frame.f_locals["reservation"])
        threading.setprofile_all_threads(None)
        held.set()
        assert release.wait(10)

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        original = rig.coordinator._runs[rig.workflow.id]
        await rig.wait_for_provider_entry()
        await rig.coordinator._publish(rig.workflow.id)
        before = seen[-1]
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            (initial,) = conn.execute(
                "SELECT id,state,amount,actual_amount FROM automatic_work_reservations WHERE kind='tokens'"
            ).fetchall()
        assert initial["state"] == "committed" and initial["actual_amount"] is None
        assert before.accounting.used["tokens"] == initial["amount"] > 10
        assert before.accounting.reserved["tokens"] == 0
        threading.setprofile_all_threads(hold_known_settlement)
        rig.provider_release()
        assert await asyncio.to_thread(held.wait, 10)
        with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
            actual = conn.execute(
                "SELECT state,actual_amount FROM automatic_work_reservations WHERE id=?",
                (reservation[0],),
            ).fetchone()
        assert reservation[0] == initial["id"] and tuple(actual) == ("settled", 10)
        delivered = len(seen)
        await rig.coordinator._publish(rig.workflow.id)
        after = seen[-1]
        assert len(seen) == delivered + 1
        assert (before.control_revision, before.progress_cursor) == (
            after.control_revision,
            after.progress_cursor,
        )
        assert (
            after.accounting.reserved["tokens"] == 0
            and after.accounting.used["tokens"] == 10
        )
        assert before.status == after.status == "planning"
        release.set()
        await asyncio.shield(original.driver)
        assert seen[-1].status == "awaiting_phase_review"
        assert (await rig.coordinator.get(rig.workflow.id)).accounting.used[
            "tokens"
        ] == 10
    finally:
        release.set()
        rig.provider_release()
        threading.setprofile_all_threads(None)
        unsubscribe()
        await rig.close()


def test_equal_native_counters_deliver_actual_accounting_settlement(tmp_path):
    _run(
        tmp_path,
        "metadata-equal-counters",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


async def superseded_metadata_journey(tmp_path, monkeypatch):
    import asyncio

    from tldw_chatbook.Agents.workflow_run_service import WorkflowRunService
    from tldw_chatbook.Chat.console_workflow_runs import (
        _accept_remaining_workflow_phase,
    )

    rig = await build_workflow_rig(tmp_path, monkeypatch, WorkflowProvider())
    seen = []
    unsubscribe = rig.coordinator.subscribe(seen.append)
    metadata_release, acceptance_release = threading.Event(), threading.Event()
    readers, approvals = [], []
    native_read_code = next(
        code
        for code in _accept_remaining_workflow_phase.__code__.co_consts
        if getattr(code, "co_name", None) == "read"
    )
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        planning = rig.coordinator._runs[rig.workflow.id]
        await asyncio.shield(planning.driver)
        before = await rig.coordinator.get(rig.workflow.id)
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        stale = decision.model_copy(
            update={"control_revision": decision.control_revision - 1}
        )
        for history in (True, False):
            metadata_release.clear()
            acceptance_release.clear()
            held, accepted = threading.Event(), threading.Event()
            observed = []
            method = (
                WorkflowRunService.get
                if history
                else WorkflowRunService.runtime_snapshot
            )
            original_code = method.__wrapped__.__code__

            def hold_originals(
                frame,
                event,
                result,
                *,
                original_code=original_code,
                held=held,
                observed=observed,
                accepted=accepted,
            ):
                if (
                    event == "return"
                    and frame.f_code is original_code
                    and frame.f_locals.get("self") is rig.service
                    and not held.is_set()
                ):
                    parent = frame.f_back
                    while (
                        parent is not None and parent.f_code.co_name != "read_metadata"
                    ):
                        parent = parent.f_back
                    if parent is None:
                        return
                    held.set()
                    assert metadata_release.wait(10)
                elif event == "call" and frame.f_code is native_read_code:
                    cell = frame.f_locals["cell"]
                    assert cell.decision is stale and cell.controller is rig.controller
                    observed.append(cell)
                    threading.setprofile_all_threads(None)
                    accepted.set()
                    assert acceptance_release.wait(10)

            threading.setprofile_all_threads(hold_originals)
            reader = asyncio.create_task(
                rig.coordinator.get(rig.workflow.id)
                if history
                else rig.coordinator._publish(rig.workflow.id)
            )
            readers.append(reader)
            assert await asyncio.to_thread(held.wait, 10)
            (metadata,) = rig.coordinator._metadata_cells
            assert (
                metadata.run is None
                and metadata.phase is None
                and metadata.issuing is reader
            )
            callback_count = len(seen)
            approval = asyncio.create_task(
                rig.coordinator.approve(stale, command_id=f"superseded-{history}")
            )
            approvals.append(approval)
            assert await asyncio.to_thread(accepted.wait, 10)
            pending = rig.coordinator._runs[rig.workflow.id]
            assert (
                pending.acceptance is observed[0]
                and pending.issuing is observed[0].issuing
            )
            assert (
                pending.phase is None
                and pending.intake is not None
                and pending.token is not None
            )
            metadata_release.set()
            if history:
                with pytest.raises(ValueError, match="workflow_snapshot_superseded"):
                    await reader
            else:
                assert await reader is metadata
            assert len(seen) == callback_count
            assert (
                metadata.disposed.result() is True
                and metadata.proof[1].exception() is None
            )
            assert (
                metadata.proof[2].result() is None
                and metadata not in rig.coordinator._metadata_cells
            )
            acceptance_release.set()
            with pytest.raises(ValueError, match="stale_workflow_decision") as refused:
                await approval
            cell = observed[0]
            assert (
                cell.body_error is refused.value and cell.preparation_entered is False
            )
            assert (
                cell.read_return[1].exception() is None
                and cell.read_return[2].result() is None
            )
            assert cell.owner_return == [True] and not cell.owner.is_admitted
            assert cell.claim.id not in rig.runtime.worktree_activity._claims
            assert rig.workflow.id not in rig.coordinator._runs
        after = await rig.coordinator.get(rig.workflow.id)
        assert (
            after.control_revision == before.control_revision
            and after.plans == before.plans
        )
        assert after.accounting == before.accounting
        assert [key for key, _args in rig.provider.calls] == ["planning"]
        assert rig.runtime.execution_capacity.snapshot().executions == ()
    finally:
        metadata_release.set()
        acceptance_release.set()
        threading.setprofile_all_threads(None)
        for task in (*readers, *approvals):
            if not task.done():
                try:
                    await asyncio.shield(task)
                except Exception as cleanup_error:  # noqa: BLE001 - preserve diagnostic cleanup outcome.
                    print(
                        "G_SUPERSEDED_CLEANUP_ERROR",
                        type(cleanup_error).__name__,
                        flush=True,
                    )
        unsubscribe()
        await rig.close()


def test_actual_new_pending_run_supersedes_original_absence_without_adopting_overlay(
    tmp_path,
):
    _run(
        tmp_path,
        "metadata-superseded",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_direct_coordinator_close_preserves_caller_cancellation(tmp_path):
    _run(
        tmp_path,
        "metadata-direct-close",
        "native",
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )
