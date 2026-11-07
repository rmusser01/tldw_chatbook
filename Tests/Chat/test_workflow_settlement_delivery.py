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
        paused = await rig.coordinator.pause(
            rig.workflow.id, expected_control_revision=settled.control_revision
        )
        with pytest.raises(ValueError, match="resume_requires_completed_planning"):
            await rig.coordinator.resume(
                rig.workflow.id,
                expected_control_revision=paused.control_revision,
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
    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("planning",))
    )
    rig.workflow = await rig.coordinator.start(
        rig.workflow.request, launch_id=rig.workflow.launch_id
    )
    await rig.wait_for_provider_entry()
    original = rig.coordinator._runs[rig.workflow.id]
    owner, claim = original.prepared.primary_owner, original.prepared.primary_claim
    stopped = await rig.coordinator.stop(
        rig.workflow.id, expected_control_revision=rig.workflow.control_revision
    )
    assert stopped.status == "stopping" and owner.is_admitted
    rig.provider_release()
    snapshot = await rig.wait_for_status("recovery_required")
    assert snapshot.accounting.uncertain and snapshot.accounting.used["generation"] == 1
    assert snapshot.plans == () and rig.coordinator._runs[rig.workflow.id] is original
    assert original.prepared.primary_owner is owner and owner.is_admitted
    assert original.prepared.primary_claim is claim
    assert claim.id in rig.runtime.worktree_activity._claims
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage

    with owned_goal_storage(rig.service), rig.service.db.connection() as conn:
        tokens = conn.execute(
            "SELECT state,actual_amount FROM automatic_work_reservations WHERE kind='tokens'"
        ).fetchall()
    assert [tuple(row) for row in tokens] == [("uncertain", None)]
    print("G_ORIGINAL_UNKNOWN_USAGE_AND_CUSTODY_RETAINED", flush=True)
    # This isolated negative deliberately retains actual custody. Process exit
    # is not a retirement observation and the creator-drain marker is forbidden.
    raise RuntimeError("G_REQUIRED_CUSTODY_RETAINED_STOP_UNKNOWN_USAGE")


def test_stop_before_usage_retains_original_recovery_custody(tmp_path):
    with pytest.raises(
        AssertionError, match="G_REQUIRED_CUSTODY_RETAINED_STOP_UNKNOWN_USAGE"
    ):
        _run(
            tmp_path,
            "unknown-usage",
            "native",
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
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
        assert (
            run.status == "recovery_required"
            and rig.coordinator._runs[rig.workflow.id] is run
        )
        assert (
            cell.owner.is_admitted
            and cell.claim.id in rig.runtime.worktree_activity._claims
        )
        assert (
            cell.phase is not None
            and cell.terminal_result is None
            and not cell.returned
        )
        assert not rig.controller.fleet_wake.try_claim_automatic_primary(
            run.session_id, object()
        )
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        command_error = rig.coordinator._commands[("approve", "unknown-initial")][
            1
        ].exception()
        assert isinstance(command_error, RuntimeError)
        with pytest.raises(RuntimeError, match="reader return unproven") as duplicate:
            await rig.coordinator.approve(decision, command_id="unknown-initial")
        assert duplicate.value is command_error
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
            assert (
                cell.owner.is_admitted
                and cell.claim.id in rig.runtime.worktree_activity._claims
            )
        print("G_ORIGINAL_E_WRAPPER_AND_CANCELLATION_RETAINED", flush=True)
    finally:
        release.set()
        threading.setprofile_all_threads(None)
    raise RuntimeError("G_REQUIRED_CUSTODY_RETAINED_E_WRAPPER")


def test_pending_wrapper_failure_plus_repeated_cancel_retains_original(tmp_path):
    with pytest.raises(AssertionError, match="G_REQUIRED_CUSTODY_RETAINED_E_WRAPPER"):
        _run(
            tmp_path,
            "pending-wrapper",
            "native",
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
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
    retained = mode in {"control-unknown", "control-association"}
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
        read, producer, retired, _callback = prepared._planning_read_return

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
    if mode in {"control-unknown", "control-association"}:
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
    replaced_helper = sys.argv[1] == "metadata-helper-held"
    uncertain = late_failure or replaced_helper
    replacement_calls = []

    def replacement(*args, **kwargs):
        replacement_calls.append((args, kwargs))
        raise AssertionError("replacement metadata helper executed")

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
        if replaced_helper:
            rig.coordinator._metadata = replacement
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
        with pytest.raises(
            RuntimeError if uncertain else asyncio.CancelledError
        ) as closed:
            await close
        with pytest.raises(asyncio.CancelledError):
            await original.driver
        if uncertain:
            assert str(closed.value) == "workflow_original_metadata_shutdown_unresolved"
            cell = original.final_metadata
            assert cell.proof[0] is read and cell.proof[1] is read._producer
            assert read._producer.exception() is cell.producer_error
            if late_failure:
                assert cell.producer_error is failure
            else:
                assert isinstance(cell.producer_error, PermissionError)
                assert (
                    str(cell.producer_error)
                    == "original workflow metadata helper changed"
                )
                assert replacement_calls == []
            assert isinstance(cell.caller_error, asyncio.CancelledError)
            assert cell.source_error is not None and cell.disposed.result() is False
            assert rig.coordinator._metadata_cells[cell] is cell.disposed
            assert original.status == "recovery_required"
            assert rig.coordinator._runs[rig.workflow.id] is original
            assert (
                not rig.runtime._disposed
                and not rig.runtime.execution_capacity.snapshot().closed
            )
            assert len(seen) == callbacks and read.retired.result() is None
            with pytest.raises(
                RuntimeError, match="workflow_original_metadata_shutdown_unresolved"
            ):
                await rig.runtime.dispose()
            retained = True
            raise RuntimeError("G_REQUIRED_CUSTODY_RETAINED_METADATA_RETURN")
        assert (
            read._producer.done()
            and not read._producer.cancelled()
            and read._producer.exception() is None
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


def test_final_metadata_preserves_caller_cancellation_and_original_wrapper_error(
    tmp_path,
):
    with pytest.raises(
        AssertionError, match="G_REQUIRED_CUSTODY_RETAINED_METADATA_RETURN"
    ):
        _run(
            tmp_path,
            "metadata-late-failure",
            "native",
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


async def execution_release_retention_journey(tmp_path, monkeypatch):
    """An actual settled execution cannot release from a failed final receiver."""
    import asyncio
    import dis
    import inspect
    import os
    import sys
    import types

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Chat import console_preparation_reads as native

    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(), change_review=True
    )
    mode = sys.argv[2]
    entered, release = threading.Event(), threading.Event()
    saved, notifications, replacement_calls = {}, [], []
    failure = RuntimeError("original execution metadata wrapper failed")
    execute_code = type(rig.coordinator)._execute.__code__
    publish_code = type(rig.coordinator)._publish.__code__
    notify_code = type(rig.coordinator)._notify_execution_release.__code__
    invoke_code = next(
        code
        for code in native.run_preparation_read.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    callback_code = next(
        code
        for code in type(rig.coordinator)._metadata.__wrapped__.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "invoke"
    )
    metadata_code = next(
        code
        for code in type(rig.coordinator)._metadata.__wrapped__.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "read_metadata"
    )
    source_code = next(
        code
        for code in type(rig.coordinator)._metadata.__wrapped__.__code__.co_consts
        if isinstance(code, types.CodeType) and code.co_name == "original_sources"
    )
    approval = None
    primary_error = None

    def replacement(*args, **kwargs):
        replacement_calls.append((args, kwargs))
        raise AssertionError("replacement release receiver executed")

    def observe(frame, event, result):
        code = frame.f_code
        if code is execute_code and event == "call" and "run" not in saved:
            assert frame.f_locals["self"] is rig.coordinator
            run = frame.f_locals["run"]
            assert run.driver is asyncio.current_task()
            saved.update(run=run, driver=run.driver, phase=run.phase)
        elif code is notify_code and event == "call":
            notifications.append((frame.f_locals["self"], frame.f_locals["binding"]))
        elif (
            mode == "successor"
            and code is publish_code
            and event == "return"
            and dis.opname[code.co_code[frame.f_lasti]]
            in {"RETURN_VALUE", "RETURN_CONST"}
            and frame.f_locals.get("_terminal_run") is saved.get("run")
            and "cell" in saved
        ):
            run, cell = saved["run"], saved["cell"]
            caller = frame.f_back
            assert caller.f_code is execute_code
            assert caller.f_locals["self"] is rig.coordinator
            assert caller.f_locals["run"] is run
            assert result is run.final_metadata is cell
            assert asyncio.current_task() is saved["driver"] is run.driver
            binding = caller.f_locals["release_binding"]
            assert binding[1] is run and binding[2] is run.phase
            assert binding[9] is saved["intake"] and binding[10] is saved["token"]
            read, producer, retired, callback = saved["proof"]
            assert cell.proof is saved["proof"] and read.callback is callback
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
            assert (
                cell.body_error
                is cell.caller_error
                is cell.source_error
                is cell.producer_error
                is None
            )
            assert run.phase._retired and run.waves[0] is saved["wave"]
            assert rig.runtime.execution_capacity.snapshot().executions == ()
            wake = rig.controller._fleet_wake
            assert binding[8] is wake
            assert "successor" not in saved
            successor = object()
            # Deliberate native API replacement after positive original metadata;
            # this grants no work and is never G's own release proof.
            assert (
                wake.release_automatic_primary(run.session_id, saved["token"]) is True
            )
            assert wake.try_claim_automatic_primary(run.session_id, successor) is True
            saved["successor"] = successor
        elif (
            code is invoke_code
            and event == "return"
            and dis.opname[code.co_code[frame.f_lasti]]
            in {"RETURN_VALUE", "RETURN_CONST"}
            and "run" in saved
            and not entered.is_set()
        ):
            run = saved["run"]
            read = frame.f_locals["read"]
            cell = run.final_metadata
            if (
                cell is None
                or read.source is not rig.coordinator
                or read.task is not run.driver
            ):
                return
            callback = read.callback
            assert callback.__code__ is callback_code
            closure = inspect.getclosurevars(callback).nonlocals
            assert closure["self"] is rig.coordinator and closure["cell"] is cell
            assert closure["issuing"] is run.driver
            guarded = closure["guarded_read"]
            assert guarded.__wrapped__.__code__ is metadata_code
            body = inspect.getclosurevars(guarded.__wrapped__).nonlocals
            originals = body["original_sources"]
            assert originals.__code__ is source_code
            assert frame.f_locals["require_current"] is originals
            source_closure = inspect.getclosurevars(originals).nonlocals
            assert source_closure["self"] is rig.coordinator
            assert source_closure["cell"] is body["cell"] is cell
            assert body["database"] is rig.service.db
            assert body["service"] is rig.service
            full, view = body["result"]
            assert full is None and view.status == "awaiting_result_review"
            assert cell.run is run and cell.phase is run.phase is saved["phase"]
            assert cell.issuing is saved["driver"] is run.driver
            proof = cell.proof
            assert proof[0] is read and proof[1] is read._producer
            assert proof[2] is read.retired and proof[3] is callback
            assert read.creator is rig.controller
            assert frame.f_locals["callback"] is callback
            binding = frame.f_locals["binding"]
            assert binding.read is read and binding.thread is threading.current_thread()
            assert binding.active is False and result is None
            assert not read._producer.done() and not read.retired.done()
            assert not cell.disposed.done() and cell.body_error is None
            assert run.phase._retired and len(run.waves) == 1
            wave_return = run.waves[0]
            assert wave_return[0].closed and wave_return[1].done()
            assert wave_return[1].exception() is None
            assert wave_return[2].done() and wave_return[3].done()
            assert len(wave_return[6]) == 1
            prepared, receipt = wave_return[6][0]
            authorization = prepared.authorization
            assert prepared.phase is authorization.phase is run.phase
            assert run.phase._task_scopes["task-1"] is authorization
            assert authorization._bridge_preparation[0] is prepared
            assert authorization._native_launch[0] is receipt
            assert receipt.attempt_id == authorization.claim.id
            assert receipt.run_id == authorization.claim.launch_run_id
            assert rig.runtime.execution_capacity.snapshot().executions == ()
            saved.update(
                cell=cell,
                proof=proof,
                view=view,
                wave=wave_return,
                intake=run.intake,
                token=run.token,
                callback=callback,
                prepared=prepared,
                authorization=authorization,
                receipt=receipt,
            )
            # This is the actual wrapper return after callback/source/binding exits.
            threading.setprofile_all_threads(None)
            sys.setprofile(None)
            entered.set()
            assert release.wait(10), "original final execution read hold expired"
            if mode == "unknown":
                raise failure

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await asyncio.shield(rig.coordinator._runs[rig.workflow.id].driver)
        decision = await rig.service.call(
            rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
        )
        threading.setprofile_all_threads(observe)
        approval = asyncio.create_task(
            rig.coordinator.approve(decision, command_id="q1-retained-release")
        )
        while not entered.is_set():
            if (
                approval.done()
                and not approval.cancelled()
                and approval.exception() is not None
            ):
                raise approval.exception()
            if "run" in saved and saved["driver"].done():
                original_cell = saved["run"].final_metadata
                if original_cell is not None and original_cell.proof is not None:
                    original_producer = original_cell.proof[1]
                    if original_producer.done() and not original_producer.cancelled():
                        original_error = original_producer.exception()
                        if original_error is not None:
                            import traceback

                            print(
                                "G_Q1_ORIGINAL_PRODUCER_FAILURE",
                                type(original_error).__name__,
                                str(original_error),
                                [
                                    (frame.filename, frame.name, frame.lineno)
                                    for frame in traceback.extract_tb(
                                        original_error.__traceback__
                                    )
                                ],
                                flush=True,
                            )
                            raise original_error
                await saved["driver"]
                raise AssertionError("original terminal execution read not reached")
            await asyncio.sleep(0.01)
        run, cell = saved["run"], saved["cell"]
        read, producer, retired, callback = saved["proof"]
        assert rig.coordinator._runs[rig.workflow.id] is run
        assert rig.store.received_turn_for_session(run.session_id) is saved["intake"]
        assert (
            rig.controller.fleet_wake._automatic_primary_claims[run.session_id]
            is saved["token"]
        )
        if mode == "receiver":
            rig.coordinator._release = replacement
        if mode in {"buddy-error", "buddy-cancel", "fence-source"}:
            await _original_buddy_disposal_control(rig, saved, approval, release, mode)
            return
        # The worker observer is retired; keep only the host's original notify
        # entry observation through the actual driver return.
        sys.setprofile(observe)
        release.set()
        await asyncio.shield(saved["driver"])
        await asyncio.shield(approval)
        assert cell is run.final_metadata and cell.proof is saved["proof"]
        assert read.callback is callback is saved["callback"]
        assert read.task is saved["driver"] and read.source is rig.coordinator
        assert read._producer is producer and read.retired is retired
        assert producer.done() and not producer.cancelled()
        assert retired.done() and not retired.cancelled() and retired.result() is None
        assert cell.body_error is cell.caller_error is None
        if mode == "unknown":
            assert producer.exception() is cell.producer_error is failure
            assert isinstance(cell.source_error, RuntimeError)
            assert str(cell.source_error) == "original workflow reader return unproven"
            assert cell.disposed.result() is False
        else:
            assert producer.exception() is cell.producer_error is None
            assert isinstance(cell.source_error, PermissionError)
            assert str(cell.source_error) == (
                "original workflow primary token changed"
                if mode == "successor"
                else "original workflow release source changed"
            )
            assert cell.disposed.result() is True
        assert notifications == [] and replacement_calls == []
        assert rig.coordinator._metadata_cells[cell] is cell.disposed
        assert rig.coordinator._runs[rig.workflow.id] is run
        assert run.status == "recovery_required" and run.phase._retired
        assert run.waves[0] is saved["wave"]
        assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
        with owned_goal_storage(rig.service):
            current = rig.service.runtime_snapshot(rig.workflow.id)
        assert current.accounting == saved["view"].accounting
        assert current.status == "awaiting_result_review"
        assert len(current.tasks) == 1 and current.tasks[0].status == "completed"
        assert current.tasks[0].task_id == "task-1"
        prepared, receipt = saved["wave"][6][0]
        assert prepared is saved["prepared"] and receipt is saved["receipt"]
        authorization = saved["authorization"]
        assert (
            prepared.authorization is run.phase._task_scopes["task-1"] is authorization
        )
        assert authorization.phase is run.phase
        assert authorization._bridge_preparation[0] is prepared
        assert authorization._native_launch[0] is receipt
        assert current.tasks[0].run_id == receipt.run_id
        assert current.accounting.used["generation"] == 2
        assert current.accounting.used["child_launch"] == 1
        assert current.accounting.used["model_call"] == 2
        assert not current.accounting.uncertain
        for _ in range(2):
            with pytest.raises(
                RuntimeError, match="workflow_original_metadata_shutdown_unresolved"
            ):
                await rig.runtime.dispose()
            assert rig.coordinator._runs[rig.workflow.id] is run
            assert rig.coordinator._metadata_cells[cell] is cell.disposed
            assert run.intake is saved["intake"] and run.token is saved["token"]
            assert (
                rig.store.received_turn_for_session(run.session_id) is saved["intake"]
            )
            assert rig.controller.fleet_wake._automatic_primary_claims[
                run.session_id
            ] is saved.get("successor", saved["token"])
            assert not rig.runtime._disposed
            assert not rig.runtime.execution_capacity.snapshot().closed
            assert notifications == [] and replacement_calls == []
        print(
            "G_Q1_REQUIRED_RELEASE_CUSTODY_RETAINED",
            mode,
            "; no creator drain",
            flush=True,
        )
        os._exit(17)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        release.set()
        threading.setprofile_all_threads(None)
        sys.setprofile(None)
        if approval is not None and not approval.done():
            try:
                await asyncio.shield(approval)
            except BaseException as error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Original approval also failed: {error!r}")
        try:
            await rig.close()
        except BaseException as error:
            if primary_error is None:
                raise
            primary_error.add_note(f"Original creator close also refused: {error!r}")


@pytest.mark.parametrize("mode", ["unknown", "receiver"])
def test_original_execution_release_requires_final_metadata_and_receiver(
    tmp_path, mode
):
    with pytest.raises(AssertionError, match="G_Q1_REQUIRED_RELEASE_CUSTODY_RETAINED"):
        _run(
            tmp_path,
            "q1-release",
            mode,
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


async def _original_buddy_disposal_control(rig, saved, approval, release, mode):
    """Keep the real G return distinct from a native buddy-fence failure."""
    import asyncio
    import dis
    import inspect
    import sys

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Chat import console_fleet_wake as native_wake
    from tldw_chatbook.Persona_Buddy.console_adapter import PersonaBuddyConsoleAdapter

    runtime, coordinator = rig.runtime, rig.coordinator
    wake = rig.controller._fleet_wake
    sink = runtime._persona_buddy_sink
    assert type(sink) is PersonaBuddyConsoleAdapter
    assert rig.controller._buddy_sink is wake.buddy_sink is sink
    assert sink._controller is vars(rig.controller.app)["_persona_buddy_controller"]
    clear = inspect.getattr_static(PersonaBuddyConsoleAdapter, "clear_wakes")
    assert inspect.getattr_static(sink, "clear_wakes") is clear
    assert clear.__code__.co_freevars == ()
    dispose_code = type(runtime).dispose.__code__
    fence_code = type(wake).dispose.__code__
    close_code = type(coordinator).close.__code__
    drain_code = type(wake).drain_recovery.__code__
    owned_code = type(runtime)._dispose_owned.__code__
    source_record = native_wake._WORKFLOW_WAKE_SOURCE
    source_code = source_record[0].__code__
    retry_code = type(wake).retry_soon.__code__
    notify_code = type(coordinator)._notify_execution_release.__code__
    original_error = RuntimeError("original native buddy clear returned failure")
    injected = asyncio.Event()
    facts = {
        "close": [],
        "drain": [],
        "owned": [],
        "fence": [],
        "fence_return": [],
        "retry": [],
        "notify": [],
    }
    failures = []
    closing = None
    primary_error = None

    def observe(frame, event, result):
        code = frame.f_code
        if code not in {
            clear.__code__,
            close_code,
            drain_code,
            owned_code,
            source_code,
            fence_code,
            retry_code,
            notify_code,
        }:
            return
        normal_return = event == "return" and dis.opname[
            code.co_code[frame.f_lasti]
        ] in {"RETURN_VALUE", "RETURN_CONST"}
        if (
            mode == "fence-source"
            and code is source_code
            and normal_return
            and not injected.is_set()
            and frame.f_back.f_code is dispose_code
        ):
            caller = frame.f_back
            valid = (
                frame.f_locals["wake"] is wake
                and caller.f_locals["self"] is runtime
                and caller.f_locals["recovery_owner"] is wake
                and caller.f_locals["source"] is source_record
                and native_wake._workflow_wake_source is source_record[0]
                and asyncio.current_task() is closing
                and not wake._disposed
                and runtime._workflow_closing
                and result is None
            )
            facts["source"] = valid
            injected.set()
            sys.setprofile(None)
            if valid:
                raise original_error
            return
        if (
            mode != "fence-source"
            and code is clear.__code__
            and normal_return
            and not injected.is_set()
        ):
            caller = frame.f_back
            outer = caller.f_back
            valid = (
                frame.f_locals["self"] is sink
                and frame.f_locals["conversation_id"] is None
                and caller.f_code is fence_code
                and caller.f_locals["self"] is wake
                and outer.f_code is dispose_code
                and outer.f_locals["self"] is runtime
                and outer.f_locals["recovery_owner"] is wake
                and asyncio.current_task() is closing
                and wake._disposed
                and runtime._workflow_closing
                and not wake._pending
                and not sink._wake_owners
                and result is None
            )
            facts["clear"] = valid
            injected.set()
            sys.setprofile(None)
            if valid:
                raise original_error
            return
        try:
            if code is close_code and normal_return:
                facts["close"].append(
                    (
                        frame.f_locals["self"],
                        frame.f_locals["_runtime_disposal"],
                        asyncio.current_task(),
                        result,
                    )
                )
            elif code is drain_code and normal_return:
                facts["drain"].append(
                    (frame.f_locals["self"], asyncio.current_task(), result)
                )
            elif code is owned_code and event == "call":
                facts["owned"].append((frame.f_locals["self"], asyncio.current_task()))
            elif code is fence_code and frame.f_locals["self"] is wake:
                if event == "call":
                    facts["fence"].append(asyncio.current_task())
                elif normal_return:
                    facts["fence_return"].append(
                        (asyncio.current_task(), result, wake._disposed)
                    )
            elif (
                code is retry_code
                and event == "call"
                and frame.f_locals["self"] is wake
            ):
                facts["retry"].append(asyncio.current_task())
            elif (
                code is notify_code
                and event == "call"
                and frame.f_locals["self"] is coordinator
            ):
                facts["notify"].append(
                    (asyncio.current_task(), frame.f_locals["binding"])
                )
        except BaseException as error:  # noqa: BLE001 - observer failures must not alter native cleanup.
            failures.append(error)

    try:
        sys.setprofile(observe)
        closing = asyncio.create_task(runtime.dispose())
        await injected.wait()
        assert facts["source" if mode == "fence-source" else "clear"] is True
        assert not closing.done() and coordinator._closing
        assert wake._disposed is (mode != "fence-source")
        run, cell = saved["run"], saved["cell"]
        read, producer, retired, callback = saved["proof"]
        assert coordinator._runs[rig.workflow.id] is run
        assert not producer.done() and not retired.done() and not cell.disposed.done()
        assert run.intake is saved["intake"] and run.token is saved["token"]
        assert (
            not runtime._disposed and not runtime.execution_capacity.snapshot().closed
        )
        # The deliberate original-return exception disabled only the host profiler.
        sys.setprofile(observe)
        if mode == "buddy-cancel":
            for _ in range(3):
                closing.cancel()
                await asyncio.sleep(0)
                assert not closing.done()
        release.set()
        if mode == "fence-source":
            with pytest.raises(
                PermissionError, match="^workflow wake shutdown fence unproven$"
            ) as notification:
                await asyncio.shield(saved["driver"])
            assert saved["driver"].exception() is notification.value
            traceback = notification.value.__traceback__
            frames = []
            while traceback is not None:
                frames.append(traceback.tb_frame)
                traceback = traceback.tb_next
            assert any(frame.f_code is notify_code for frame in frames)
        else:
            await asyncio.shield(saved["driver"])
        await asyncio.shield(approval)
        try:
            await closing
        except BaseException as error:  # noqa: BLE001 - compare the exact original error or caller cancellation.
            actual_error = error
        else:
            raise AssertionError("original buddy fence failure was lost")
        sys.setprofile(None)
        assert failures == [] and len(facts["close"]) == 1
        same_coordinator, same_runtime, same_task, cancellation = facts["close"][0]
        assert same_coordinator is coordinator and same_runtime is runtime
        assert same_task is closing
        if mode == "buddy-cancel":
            assert isinstance(cancellation, asyncio.CancelledError)
            assert (
                actual_error is cancellation
                and actual_error.__cause__ is original_error
            )
        else:
            assert cancellation is None and actual_error is original_error
        assert facts["drain"] == [(wake, closing, False)]
        assert facts["owned"] == []
        if mode == "fence-source":
            assert facts["fence"] == facts["fence_return"] == facts["retry"] == []
            assert not wake._disposed
            assert len(facts["notify"]) == 1
            issuing, binding = facts["notify"][0]
            assert issuing is saved["driver"] and binding[1] is run
            assert binding[2] is run.phase and binding[8] is wake
            assert run.status != "recovery_required"
        assert cell is run.final_metadata and cell.proof is saved["proof"]
        assert read.callback is callback is saved["callback"]
        assert read.task is saved["driver"] and read.source is coordinator
        assert read._producer is producer and read.retired is retired
        assert (
            producer.done()
            and not producer.cancelled()
            and producer.exception() is None
        )
        assert retired.done() and not retired.cancelled() and retired.result() is None
        assert cell.disposed.done() and cell.disposed.result() is True
        assert (
            cell.body_error
            is cell.caller_error
            is cell.source_error
            is cell.producer_error
            is None
        )
        assert cell not in coordinator._metadata_cells
        assert rig.workflow.id not in coordinator._runs
        assert run.intake is saved["intake"] and run.token is saved["token"]
        assert run.phase._retired
        assert rig.store.received_turn_for_session(run.session_id) is None
        assert run.session_id not in wake._automatic_primary_claims
        assert run.waves[0] is saved["wave"]
        assert runtime.execution_capacity.snapshot().executions == ()
        assert (
            not runtime._disposed and not runtime.execution_capacity.snapshot().closed
        )
        with owned_goal_storage(rig.service):
            current = rig.service.runtime_snapshot(rig.workflow.id)
        assert current.accounting == saved["view"].accounting
        assert current.status == "awaiting_result_review"
        assert len(current.tasks) == 1 and current.tasks[0].status == "completed"
        prepared, receipt = saved["wave"][6][0]
        assert prepared is saved["prepared"] and receipt is saved["receipt"]
        assert prepared.authorization is saved["authorization"]
        assert run.phase._task_scopes["task-1"] is prepared.authorization
        assert prepared.authorization._bridge_preparation[0] is prepared
        assert prepared.authorization._native_launch[0] is receipt
        assert (
            current.tasks[0].task_id == "task-1"
            and current.tasks[0].run_id == receipt.run_id
        )
        assert current.accounting.used["generation"] == 2
        assert current.accounting.used["child_launch"] == 1
        assert current.accounting.used["model_call"] == 2
        assert not current.accounting.uncertain
        assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
        print("G_Q1_BUDDY_FIRST_ERROR_G_RETURNED", mode, flush=True)
        # This is a separately authorized native idempotent disposal, never a
        # claim that the first error completed downstream runtime teardown.
        if mode == "fence-source":
            sys.setprofile(observe)
        await runtime.dispose()
        sys.setprofile(None)
        if mode == "fence-source":
            assert failures == [] and facts["fence"]
            assert facts["fence"][0] is asyncio.current_task()
            assert facts["fence_return"][0] == (asyncio.current_task(), None, True)
        assert runtime._disposed and runtime.execution_capacity.snapshot().closed
        assert runtime.execution_capacity.snapshot().executions == ()
        assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
        print("G_Q1_BUDDY_SECOND_NATIVE_DISPOSAL_RETURNED", mode, flush=True)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        release.set()
        sys.setprofile(None)
        if closing is not None and not closing.done():
            try:
                await asyncio.shield(closing)
            except BaseException as error:
                if primary_error is None:
                    raise
                primary_error.add_note(f"Original disposal also failed: {error!r}")


@pytest.mark.parametrize("mode", ["buddy-error", "buddy-cancel"])
def test_native_buddy_clear_error_preserves_g_close_and_cancellation(tmp_path, mode):
    _run(
        tmp_path,
        "q1-release",
        mode,
        script=Path(__file__).with_name("workflow_coordinator_probe.py").read_text(),
        timeout=60,
    )


def test_native_successor_token_refuses_original_g_release_without_notification(
    tmp_path,
):
    with pytest.raises(AssertionError, match="G_Q1_REQUIRED_RELEASE_CUSTODY_RETAINED"):
        _run(
            tmp_path,
            "q1-release",
            "successor",
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
            timeout=60,
        )


def test_pre_fence_source_error_keeps_positive_g_return_and_refuses_notification(
    tmp_path,
):
    _run(
        tmp_path,
        "q1-release",
        "fence-source",
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


async def replaced_metadata_before_call_journey(tmp_path, monkeypatch):
    import asyncio

    from tldw_chatbook.Agents.workflow_models import WorkflowSnapshot
    from tldw_chatbook.Chat.console_workflow_preparation import _PreparedWorkflowReply

    rig = await build_workflow_rig(
        tmp_path, monkeypatch, WorkflowProvider(hold=("planning",))
    )
    injected, calls = [], []
    original_code = _PreparedWorkflowReply.run_planning.__code__
    retained = False

    async def replacement(*args, **kwargs):
        calls.append((args, kwargs))

    def replace_after_original_f_return(frame, event, result):
        if (
            event == "return"
            and frame.f_code is original_code
            and type(result) is WorkflowSnapshot
            and frame.f_locals.get("self") is original.prepared
        ):
            assert (
                original.prepared._planning_settled
                and not original.prepared.primary_owner.is_admitted
            )
            threading.setprofile_all_threads(None)
            rig.coordinator._publish = replacement
            injected.append(result)

    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        original = rig.coordinator._runs[rig.workflow.id]
        await rig.wait_for_provider_entry()
        threading.setprofile_all_threads(replace_after_original_f_return)
        rig.provider_release()
        await asyncio.shield(original.driver)
        assert len(injected) == 1 and calls == []
        assert rig.coordinator._runs[rig.workflow.id] is original
        assert (
            original.status == "recovery_required" and original.final_metadata is None
        )
        read, producer, retired, _callback = original.prepared._planning_read_return
        assert (
            producer is read._producer
            and producer.exception() is None
            and retired.result() is None
        )
        assert not original.prepared.primary_owner.is_admitted
        assert (
            original.prepared.primary_claim.id
            not in rig.runtime.worktree_activity._claims
        )
        assert (
            original.prepared._planning_intake is None
            and original.prepared._planning_token is None
        )
        for _ in range(2):
            with pytest.raises(
                RuntimeError, match="workflow_original_shutdown_unresolved"
            ):
                await rig.runtime.dispose()
            assert (
                not rig.runtime._disposed
                and not rig.runtime.execution_capacity.snapshot().closed
            )
            assert rig.coordinator._runs[rig.workflow.id] is original and calls == []
        retained = True
        raise RuntimeError("G_REQUIRED_CUSTODY_RETAINED_METADATA_HELPER")
    finally:
        rig.provider_release()
        threading.setprofile_all_threads(None)
        if not retained:
            await rig.close()


@pytest.mark.parametrize("mode", ["metadata-helper-held", "metadata-helper-before"])
def test_replaced_metadata_helper_never_runs_or_releases_original_custody(
    tmp_path, mode
):
    with pytest.raises(AssertionError, match="G_REQUIRED_CUSTODY_RETAINED_METADATA"):
        _run(
            tmp_path,
            mode,
            "native",
            script=Path(__file__)
            .with_name("workflow_coordinator_probe.py")
            .read_text(),
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
