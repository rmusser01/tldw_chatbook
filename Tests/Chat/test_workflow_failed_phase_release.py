"""A failed native phase returns its chat only when no work is left running (TASK-40718)."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = r"""
import asyncio, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install(); real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="false", OPENAI_API_KEY="private-workflow-test")
from Tests.Chat import test_workflow_failed_phase_release as journeys
with user_fixture_default_owner():
    asyncio.run(getattr(journeys, sys.argv[1])(Path(os.environ["XDG_DATA_HOME"]).parent))
print("retired and reopened")
"""


def test_blocked_phase_returns_its_chat_without_a_restart(tmp_path):
    """A phase whose remaining task waits on a task it will never run raised
    "workflow dependencies blocked". Nothing had launched, yet the run stayed
    `recovery_required`, kept the chat's intake claim ("A turn is already
    preparing."), reconcile refused it, and app shutdown could not retire it."""
    _run(tmp_path, "blocked_phase", "native", script=_SCRIPT, timeout=180)


def test_phase_failing_with_a_live_worker_keeps_custody_until_reviewed(tmp_path):
    """A failure while a worker still runs keeps the original custody, and
    reconcile refuses it. Once the worker has returned, reconcile closes the
    run as reviewed history and the chat gets its claim back; on dev 1455ceeb96
    it kept refusing with `workflow_resources_still_live` because execution
    runs have no planning preparation. A refused record leaves the phase able
    to retry, and a claim release that raises is finished by the next
    reconcile instead of wedging the chat."""
    _run(tmp_path, "live_worker_failure", "native", script=_SCRIPT, timeout=180)


def test_finalize_during_a_refused_record_keeps_the_phase_retired(tmp_path):
    """A record that fails must not undo a retirement the finalize path made
    while the record was running."""
    _run(tmp_path, "finalize_during_record", "native", script=_SCRIPT, timeout=180)


def test_ownership_error_in_a_phase_keeps_a_claim_it_does_not_own(tmp_path):
    """The failed-phase path catches every ordinary exception, ownership errors
    included. A primary token that changed hands stays with its new owner:
    the run keeps custody until reconcile, once ownership is restored."""
    _run(tmp_path, "ownership_error", "native", script=_SCRIPT, timeout=180)


def test_phase_failing_before_launch_returns_its_chat_without_reconcile(tmp_path):
    """A phase that failed after preparing task-1 but before launching it kept
    the run retained: the driver's failure handler counted the open preparation
    as live work, so only a reconcile released the chat (TASK-41646). The
    handler now closes that never-launched preparation itself, and the phase
    retires as a resumable failed phase, as `blocked_phase` does."""
    _run(tmp_path, "failed_before_launch", "native", script=_SCRIPT, timeout=180)


def test_open_unlaunched_preparation_is_reconciled_without_effects(tmp_path):
    """A retained run whose wave call was cancelled after preparing task-1 but
    before launching it kept that preparation open: it counted as live work, so
    reconcile refused with `workflow_resources_still_live` and the chat stayed
    held until restart (TASK-41320). Reconcile now closes that never-launched preparation
    without launching or dispatching anything, records the review, and returns
    the chat's claim. A stale revision is refused before anything closes.

    The wave-bound and launching checks here set that state by hand. Real
    launched, wave-bound coverage is TASK-40718's `live_worker_failure`."""
    _run(tmp_path, "unlaunched_preparation", "native", script=_SCRIPT, timeout=180)


def test_unproven_control_return_is_released_only_by_restart(tmp_path):
    """A Stop whose worker returned but whose outcome is unproven keeps the
    chat until restart. Reconcile says so with `workflow_return_unproven`
    instead of "still live, wait", and records nothing (TASK-41320). A plain
    refusal by the phase is proven and is not that case (TASK-41645); here the
    control's storage handle does not return."""
    _run(tmp_path, "unproven_control", "native", script=_SCRIPT, timeout=180)


def test_a_refusal_after_a_stored_decision_stays_unknown(tmp_path):
    """The phase's refusal counts as a refusal only when no decision was
    stored: a Stop whose decision committed before the phase raised keeps the
    chat until restart (TASK-41645)."""
    _run(tmp_path, "refusal_after_decision", "native", script=_SCRIPT, timeout=180)


def test_stop_racing_the_end_of_a_phase_is_a_refusal(tmp_path):
    """Stop passed its revision check, then the phase it targeted retired and
    refused it with `workflow_authority_changed`. That ordinary refusal was
    recorded as an unknown control, so the finished phase kept its chat until
    restart. A refusal whose worker returned, with no decision stored, storage
    returned and sources current, is now just a refusal: the phase releases
    its chat and a new turn is accepted (TASK-41645)."""
    _run(tmp_path, "stop_races_phase_end", "native", script=_SCRIPT, timeout=180)


def test_detail_view_stops_a_running_phase_under_an_unproven_read(tmp_path):
    """Under the unproven-read marker the workflow detail view could not read
    its workflow, so every button stayed disabled and Stop was unreachable.
    The view now reads saved state through the service path the commands use;
    its Stop button stops the running phase and says it was applied
    (TASK-41645)."""
    _run(tmp_path, "detail_stop_under_unproven_read", "native", script=_SCRIPT, timeout=240)


def test_a_failing_close_in_the_failure_handler_keeps_the_original_failure(tmp_path):
    """The driver's failure handler closes never-launched preparations; a close
    that raises must not replace the original failure and push the phase down
    the "did not retire normally" path (a held chat). The live-resource check
    still decides custody (TASK-41645 review)."""
    _run(tmp_path, "close_raises_in_failure_handler", "native", script=_SCRIPT, timeout=180)


def test_detail_view_under_the_marker_refuses_a_closing_or_stale_read(tmp_path):
    """`view` under the unproven-read marker skipped `get`'s closing check and
    its monotonic floor; a screen left open while the coordinator closed could
    render saved state with live Stop buttons (TASK-41645 review)."""
    _run(tmp_path, "view_refuses_closing_and_stale", "native", script=_SCRIPT, timeout=180)


def test_unproven_final_metadata_read_is_released_only_by_restart(tmp_path):
    """A phase's final status read whose storage handles did not provably
    return keeps the chat until restart. Reconcile refuses it with
    `workflow_return_unproven` and the Console names a restart (TASK-41320)."""
    _run(tmp_path, "unproven_final_read", "native", script=_SCRIPT, timeout=180)


def test_an_earlier_unproven_metadata_read_keeps_the_chat_until_restart(tmp_path):
    """A status read of the retained run whose storage handle did not provably
    return leaves an unproven marker even though it was not the run's final
    read. Reconcile refused only the final read's marker, so it closed the
    preparation, stored the run closed and released the chat, then failed
    at its own trailing publication (TASK-41320 review). It now refuses first,
    with `workflow_return_unproven`, and changes nothing. The read that became
    unproven already carries that reason, and another chat's detail view names
    a restart without saying that chat is held."""
    _run(tmp_path, "unproven_earlier_read", "native", script=_SCRIPT, timeout=180)


def test_stop_under_an_unproven_read_applies_and_says_so(tmp_path):
    """Another chat's workflow read lost a storage handle, which leaves the
    coordinator's unproven-read marker. Stop on a running phase then took
    effect, but its trailing read-back was refused, so Stop reported plain
    `workflow_return_unproven` for an action it had applied (TASK-41647). It
    now reports `workflow_command_applied`, and the Console says Stop was
    applied and names a restart for the view."""
    _run(tmp_path, "stop_under_unproven_read", "native", script=_SCRIPT, timeout=180)


def test_unproven_read_refuses_approve_but_stop_still_applies(tmp_path):
    """Under the unproven-read marker an approved phase claimed the chat and
    ran, yet could never publish its terminal state, so it held the chat
    until restart. Approve (like Start and Revise) now refuses with
    `workflow_return_unproven` before any effect, while Stop, which needs no
    proven read, still applies (TASK-41645, ADR-285)."""
    _run(tmp_path, "approve_under_unproven_read", "native", script=_SCRIPT, timeout=180)


def test_a_phase_running_when_the_marker_is_set_fails_at_its_next_wave_boundary(
    tmp_path,
):
    """Owner decision 2026-11-25 (TASK-41946, ADR-285): a phase already running
    when the unproven-read marker is set does not keep running. The task in
    flight finishes, the wave boundary's publication is refused, the phase
    settles `failed`, the next task never runs, and the chat stays held until
    restart (the TASK-40716 rule for an unproven return)."""
    _run(
        tmp_path,
        "phase_running_under_unproven_read",
        "native",
        script=_SCRIPT,
        timeout=180,
    )


async def _started(rig):
    rig.provider.plan["phases"][0]["tasks"].append(
        dict(
            rig.provider.plan["phases"][0]["tasks"][0],
            id="task-2",
            instructions="Execute task-2",
            dependencies=["task-1"],
        )
    )
    rig.workflow = await rig.coordinator.start(
        rig.workflow.request, launch_id=rig.workflow.launch_id
    )
    await rig.wait_for_status("awaiting_phase_review", timeout=30)
    return await rig.service.call(
        rig.service.prepare_decision, rig.workflow.id, phase_id="phase-1"
    )


async def blocked_phase(tmp_path):
    import asyncio
    import functools
    from types import MethodType

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    try:
        decision = await _started(rig)
        service = rig.service
        original = service.phase_task_ids
        reads = []

        @functools.wraps(original)
        def without_first(self, phase):
            # Only the driver's own read (the first) drops task-1; the read
            # inside reviewed_task_ids keeps it. task-1 is then neither
            # pending nor done, and task-2 depends on it: blocked.
            reads.append(phase)
            ids = original(phase)
            if len(reads) > 1:
                return ids
            return tuple(task for task in ids if task != "task-1")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(service, "phase_task_ids", MethodType(without_first, service))
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.wait_for(asyncio.shield(run.driver), 60)
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        assert not rig.runtime.execution_capacity.snapshot().executions
        assert rig.workflow.id not in rig.coordinator._runs
        # The same resumable state as a phase whose task failed.
        assert (await rig.coordinator.get(rig.workflow.id)).status == "paused"
        sent = await rig.controller.submit_draft(
            "A new turn after the failed phase", session_id=session_id
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get(session_id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        await rig.close()


async def live_worker_failure(tmp_path):
    import asyncio
    import contextlib
    import threading
    from types import SimpleNamespace

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery
    from tldw_chatbook.Chat import console_workflow_wave

    from loguru import logger

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider(hold=("task-1",)))
    original = console_workflow_wave.run_workflow_wave
    waves = []
    logged = []
    sink = logger.add(logged.append, format="{message}", level="ERROR")

    async def failing_while_live(phase, selected, **kwargs):
        # The real wave keeps running; only the phase driver's wait fails.
        waves.append(asyncio.ensure_future(original(phase, selected, **kwargs)))
        await rig.wait_for_provider_entry("task-1")
        raise RuntimeError("wave observer failed")

    try:
        decision = await _started(rig)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                console_workflow_wave, "run_workflow_wave", failing_while_live
            )
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.wait_for(asyncio.shield(run.driver), 60)
        assert rig.coordinator._runs.get(rig.workflow.id) is run
        # Never attempted as a returned failure: no cleanup over live work.
        assert not [line for line in logged if "phase failed" in line], logged
        snapshot = await rig.coordinator.get(rig.workflow.id)
        assert snapshot.status == "recovery_required"
        stored = await rig.service.call(rig.service.get, rig.workflow.id)
        try:
            await rig.coordinator.reconcile(
                rig.workflow.id,
                expected_control_revision=snapshot.control_revision,
                decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
                command_id="reconcile-live",
            )
        except ValueError as error:
            assert str(error) == "workflow_resources_still_live", error
        else:
            raise AssertionError("reconcile accepted a phase with a live worker")
        # The refusal recorded nothing.
        assert await rig.service.call(rig.service.get, rig.workflow.id) == stored
        rig.provider_release()
        await asyncio.wait_for(asyncio.gather(*waves, return_exceptions=True), 60)
        review = WorkflowExecutionRecovery(reason="reviewed_prior_effects")
        returned = await rig.service.call(rig.service.get, rig.workflow.id)
        phase, service = run.phase, rig.service
        live_check = service._has_live_workflow_resources
        # A launch racing a record, through the real launch gate
        # (`_commit_task_launch`). Only its fence's identity checks are
        # stubbed; the stub keeps the fence's control lock. The launch starts
        # right after the live check passes and must be refused.
        launches = []

        @contextlib.contextmanager
        def lock_only_fence(operation, *, _launch=None, _control=None):
            with phase._control_lock:
                yield SimpleNamespace(limits=None)

        def racing_launch():
            child = SimpleNamespace(
                claim=SimpleNamespace(id="racing-child", reservation_id="racing-child")
            )
            try:
                phase._commit_task_launch(child, None, None, None)
            except AutomaticWorkRefused as error:
                launches.append(error.reason)
            except BaseException as error:  # noqa: BLE001 - past the gate: bound.
                launches.append("bound:" + type(error).__name__)
            else:
                launches.append("bound")

        def check_then_launch(workflow_id):
            live = live_check(workflow_id)
            if not launches:
                launcher = threading.Thread(target=racing_launch)
                launcher.start()
                launcher.join(10)
            return live

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(phase, "_fence", lock_only_fence)
            patch.setattr(service, "_has_live_workflow_resources", check_then_launch)
            # A record refused inside its transaction keeps the phase retryable.
            with pytest.raises(ValueError, match="revision_conflict"):
                await rig.coordinator.reconcile(
                    rig.workflow.id,
                    expected_control_revision=snapshot.control_revision + 1,
                    decision=review,
                    command_id="reconcile-stale",
                )
        assert launches == ["child_already_launched"], launches
        assert not phase._retired
        assert await service.call(service.get, rig.workflow.id) == returned
        # Two concurrent records; the first fails after the second recorded.
        # Records run one at a time, so the failure cannot undo the second's
        # retirement: the phase ends retired and recorded as returned.
        entered, recorded = threading.Event(), threading.Event()
        checks = []

        def first_check_waits(workflow_id):
            checks.append(workflow_id)
            if len(checks) == 1:
                entered.set()
                recorded.wait(2)  # Serialized records never see the second.
            return live_check(workflow_id)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(service, "_has_live_workflow_resources", first_check_waits)
            first = asyncio.ensure_future(
                service.call(
                    service.record_returned_execution,
                    phase,
                    expected_control_revision=returned.control_revision + 5,
                )
            )
            assert await asyncio.to_thread(entered.wait, 10)
            second = asyncio.ensure_future(
                service.call(
                    service.record_returned_execution,
                    phase,
                    expected_control_revision=returned.control_revision,
                )
            )
            second.add_done_callback(lambda _: recorded.set())
            failed, done = await asyncio.gather(first, second, return_exceptions=True)
        assert str(failed) == "revision_conflict", failed
        assert done.status == "recovery_required", done
        assert phase._retired
        assert (
            await service.call(service.get, rig.workflow.id)
        ).status == "recovery_required"
        # A release that raises is finished by the next reconcile.
        wake = rig.controller._fleet_wake
        release = wake.release_automatic_primary
        raised = []

        def release_once(session_id, token):
            if not raised:
                raised.append(session_id)
                raise RuntimeError("primary release failed")
            return release(session_id, token)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(wake, "release_automatic_primary", release_once)
            current = await service.call(service.get, rig.workflow.id)
            with pytest.raises(RuntimeError, match="primary release failed"):
                await rig.coordinator.reconcile(
                    rig.workflow.id,
                    expected_control_revision=current.control_revision,
                    decision=review,
                    command_id="reconcile-returned",
                )
            assert rig.coordinator._runs.get(rig.workflow.id) is run
            current = await service.call(service.get, rig.workflow.id)
            closed = await rig.coordinator.reconcile(
                rig.workflow.id,
                expected_control_revision=current.control_revision,
                decision=review,
                command_id="reconcile-retry",
            )
        assert raised == ["workflow-test-chat"]
        assert closed.status == "stopped"  # Closed history displays as stopped.
        assert rig.workflow.id not in rig.coordinator._runs
        service, db = rig.service, rig.service.db
        with owned_goal_storage(service), db.connection() as conn:
            row = conn.execute(
                "SELECT status,recovery_reviewed_revision FROM workflow_runs WHERE id=?",
                (rig.workflow.id,),
            ).fetchone()
        assert tuple(row) == ("closed", closed.control_revision)
        sent = await rig.controller.submit_draft(
            "A new turn after the reviewed phase", session_id="workflow-test-chat"
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get("workflow-test-chat")
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        logger.remove(sink)
        rig.provider_release()
        await asyncio.wait_for(asyncio.gather(*waves, return_exceptions=True), 60)
        await rig.close()


async def ownership_error(tmp_path):
    import asyncio
    import functools
    from types import MethodType

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    wake = rig.controller._fleet_wake
    foreign = object()
    try:
        decision = await _started(rig)
        service = rig.service
        original = service.phase_task_ids
        reads = []

        @functools.wraps(original)
        def token_changed_hands(self, phase):
            reads.append(phase)
            if len(reads) == 1:
                with wake._registry_lock:
                    wake._automatic_primary_claims[session_id] = foreign
                raise PermissionError("workflow original primary token changed")
            return original(phase)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                service, "phase_task_ids", MethodType(token_changed_hands, service)
            )
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.wait_for(asyncio.shield(run.driver), 60)
        assert wake._automatic_primary_claims.get(session_id) is foreign
        assert rig.coordinator._runs.get(rig.workflow.id) is run
        assert (await rig.coordinator.get(rig.workflow.id)).status == (
            "recovery_required"
        )
        assert not (
            await rig.controller.submit_draft("Still held", session_id=session_id)
        ).accepted
        # Ownership restored: the review closes the run and returns the claim.
        with wake._registry_lock:
            wake._automatic_primary_claims[session_id] = run.token
        current = await service.call(service.get, rig.workflow.id)
        closed = await rig.coordinator.reconcile(
            rig.workflow.id,
            expected_control_revision=current.control_revision,
            decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            command_id="reconcile-restored",
        )
        assert closed.status == "stopped"
        assert rig.workflow.id not in rig.coordinator._runs
        sent = await rig.controller.submit_draft(
            "A new turn after the review", session_id=session_id
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get(session_id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        with wake._registry_lock:
            if wake._automatic_primary_claims.get(session_id) is foreign:
                del wake._automatic_primary_claims[session_id]
        await rig.close()


async def _failed_before_launch(rig, error, *, on_prepared=None):
    """The wave call raises `error` after task-1 is prepared, before any wave
    binds it; returns once the phase driver has finished."""
    import asyncio

    from tldw_chatbook.Chat import console_workflow_wave

    prepared = []

    async def refused_before_launch(phase, selected, **kwargs):
        prepared.extend(selected)
        if on_prepared is not None:
            on_prepared(selected)
        raise error

    decision = await _started(rig)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(console_workflow_wave, "run_workflow_wave", refused_before_launch)
        await rig.coordinator.approve(decision, command_id="approved-once")
        run = rig.coordinator._runs[rig.workflow.id]
        await asyncio.wait({run.driver}, timeout=60)
    assert run.driver.done()
    # task-1 was prepared and never enrolled in a wave or launched.
    (item,) = prepared
    assert item._wave is None and not item.authorization._native_launch
    return run, item


async def failed_before_launch(tmp_path):
    import asyncio

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    try:
        run, item = await _failed_before_launch(
            rig, RuntimeError("wave refused before launch")
        )
        # The driver closed the preparation itself, without effects.
        assert item._closed and item not in rig.runtime._workflow_preparations
        assert not item.authorization._native_launch
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        assert not rig.runtime.execution_capacity.snapshot().executions
        assert rig.workflow.id not in rig.coordinator._runs
        assert not [key for key in rig.coordinator._commands if key[0] == "reconcile"]
        # The same resumable failed phase as `blocked_phase`.
        assert (await rig.coordinator.get(rig.workflow.id)).status == "paused"
        sent = await rig.controller.submit_draft(
            "A new turn after the failed phase", session_id=session_id
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get(session_id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        await rig.close()


async def _retained_unlaunched(rig):
    """A cancelled wave call leaves task-1's preparation open in a retained
    run: the driver's failure handler takes ordinary exceptions only."""
    import asyncio

    run, item = await _failed_before_launch(rig, asyncio.CancelledError())
    assert item in run.preparations.values()
    assert item in rig.runtime._workflow_preparations and not item._closed
    assert rig.coordinator._runs.get(rig.workflow.id) is run
    assert not (
        await rig.controller.submit_draft("Still held", session_id="workflow-test-chat")
    ).accepted
    return run, item


async def _refused(rig, reason, command_id, *, offset=0):
    """Reconcile refuses with `reason`, recording nothing."""
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery

    stored = await rig.service.call(rig.service.get, rig.workflow.id)
    with pytest.raises(ValueError, match=reason) as refused:
        await rig.coordinator.reconcile(
            rig.workflow.id,
            expected_control_revision=stored.control_revision + offset,
            decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            command_id=command_id,
        )
    assert await rig.service.call(rig.service.get, rig.workflow.id) == stored
    return refused.value


async def unlaunched_preparation(tmp_path):
    import asyncio

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    try:
        run, item = await _retained_unlaunched(rig)
        snapshot = await rig.coordinator.get(rig.workflow.id)
        assert snapshot.status == "recovery_required"
        # A stale revision is refused before any preparation closes.
        await _refused(rig, "revision_conflict", "reconcile-stale", offset=1)
        assert not item._closed and item in rig.runtime._workflow_preparations
        # AC #3: a wave-bound or launching preparation is never closed here;
        # the live check refuses it.
        item._wave = object()  # Bound to a wave.
        try:
            await _refused(rig, "workflow_resources_still_live", "reconcile-waved")
        finally:
            item._wave = None
        item.authorization._native_launch.append(None)  # A launch in progress.
        try:
            await _refused(rig, "workflow_resources_still_live", "reconcile-launching")
        finally:
            item.authorization._native_launch.clear()
        assert not item._closed and item in rig.runtime._workflow_preparations
        closed = await rig.coordinator.reconcile(
            rig.workflow.id,
            expected_control_revision=snapshot.control_revision,
            decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            command_id="reconcile-unlaunched",
        )
        # Closed without effects: nothing launched or dispatched.
        assert item._closed and item not in rig.runtime._workflow_preparations
        assert not item.authorization._native_launch
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        assert not rig.runtime.execution_capacity.snapshot().executions
        assert closed.status == "stopped"  # Closed history displays as stopped.
        assert rig.workflow.id not in rig.coordinator._runs
        service = rig.service
        with owned_goal_storage(service), service.db.connection() as conn:
            row = conn.execute(
                "SELECT status,recovery_reviewed_revision FROM workflow_runs WHERE id=?",
                (rig.workflow.id,),
            ).fetchone()
        assert tuple(row) == ("closed", closed.control_revision)
        sent = await rig.controller.submit_draft(
            "A new turn after the review", session_id=session_id
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get(session_id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        await rig.close()


def _restart_only(error):
    """The Console names a restart, not a wait, for an unproven return."""
    from tldw_chatbook.UI.Console_Modules.workflows import workflow_view_error

    message = workflow_view_error("Recovery", error)
    assert (
        "Restart Chatbook to read workflows again and release held chats" in message
    ), message
    assert "Wait" not in message, message


async def _close_unresolved(rig, reason):
    # Restart-only by design (TASK-41320): shutdown reports the held run.
    with pytest.raises(RuntimeError, match=reason):
        await rig.close()


async def unproven_control(tmp_path):
    import contextlib

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused
    from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
        preparation_read_for,
    )
    from tldw_chatbook.Chat import console_workflow_runs

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    try:
        run, item = await _retained_unlaunched(rig)

        def refused(decision=None, *, expected_control_revision=None):
            raise AutomaticWorkRefused("workflow_authority_changed")

        storage_return = console_workflow_runs._remaining_storage_return

        @contextlib.contextmanager
        def control_handle_lost(databases):
            read = preparation_read_for(rig.controller)
            try:
                with storage_return(databases):
                    yield
            finally:  # As the real check does, even when the control raised.
                if read is not None and read.source is run.phase:
                    raise RuntimeError("original source database handle did not return")

        current = await rig.service.call(rig.service.get, rig.workflow.id)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(run.phase, "_control", refused)
            # A plain refusal is proven (TASK-41645): nothing is retained.
            with pytest.raises(AutomaticWorkRefused, match="workflow_authority_changed"):
                await rig.coordinator.stop(
                    rig.workflow.id, expected_control_revision=current.control_revision
                )
            assert not run.control_unknown and run.control_return.is_set()
            # A refusal whose storage handle did not return is unknown. (Pause:
            # a repeated Stop at this revision replays the recorded command.)
            patch.setattr(
                console_workflow_runs, "_remaining_storage_return", control_handle_lost
            )
            with pytest.raises(RuntimeError, match="control reader return unproven"):
                await rig.coordinator.pause(
                    rig.workflow.id, expected_control_revision=current.control_revision
                )
        # The control's worker returned; its outcome stays unproven.
        assert run.control_unknown and run.control_return.is_set()
        _restart_only(
            await _refused(rig, "workflow_return_unproven", "reconcile-control")
        )
        assert not item._closed and rig.coordinator._runs.get(rig.workflow.id) is run
        assert not (
            await rig.controller.submit_draft(
                "Still held", session_id="workflow-test-chat"
            )
        ).accepted
    finally:
        await _close_unresolved(rig, "workflow_original_shutdown_unresolved")


async def unproven_final_read(tmp_path):
    import asyncio
    import contextlib
    import functools
    from types import MethodType

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
        preparation_read_for,
    )
    from tldw_chatbook.Chat import console_workflow_runs

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    try:
        decision = await _started(rig)
        service = rig.service
        original_ids, original_settle = (
            service.phase_task_ids,
            service.settle_execution_phase,
        )
        storage_return = console_workflow_runs._remaining_storage_return
        reads, settled, lost = [], [], []

        @functools.wraps(original_ids)
        def without_first(self, phase):
            # As in blocked_phase: the phase fails with nothing launched.
            reads.append(phase)
            ids = original_ids(phase)
            return ids if len(reads) > 1 else tuple(t for t in ids if t != "task-1")

        @functools.wraps(original_settle)
        def settle(self, phase, **kwargs):
            settled.append(original_settle(phase, **kwargs))
            return settled[-1]

        @contextlib.contextmanager
        def handle_lost_after_settle(databases):
            # Only the driver's read after settlement: its terminal publication
            # (approve's own trailing read may run then too).
            read = preparation_read_for(rig.controller)
            run = rig.coordinator._runs.get(rig.workflow.id)
            final = settled and run and read and read.task is run.driver
            with storage_return(databases):
                yield
            if final and not lost:
                lost.append(databases)
                raise RuntimeError("original source database handle did not return")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(service, "phase_task_ids", MethodType(without_first, service))
            patch.setattr(
                service, "settle_execution_phase", MethodType(settle, service)
            )
            patch.setattr(
                console_workflow_runs,
                "_remaining_storage_return",
                handle_lost_after_settle,
            )
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.wait_for(asyncio.shield(run.driver), 60)
        assert settled and lost
        cell = run.final_metadata
        # The read's worker returned; whether its handles did is unproven.
        assert cell.disposed.done() and cell.disposed.result() is False
        assert rig.coordinator._metadata_cells[cell] is cell.disposed
        assert rig.coordinator._runs.get(rig.workflow.id) is run
        _restart_only(await _refused(rig, "workflow_return_unproven", "reconcile-read"))
        assert rig.coordinator._runs.get(rig.workflow.id) is run
        assert not (
            await rig.controller.submit_draft(
                "Still held", session_id="workflow-test-chat"
            )
        ).accepted
        # The detail view reads saved state under the marker (TASK-41645): it
        # renders, names a restart and does not offer a refresh.
        notice = await _detail_notice(rig)
        assert "Restart Chatbook to approve or revise work" in notice, notice
        assert "Refresh" not in notice, notice
    finally:
        await _close_unresolved(rig, "workflow_original_metadata_shutdown_unresolved")


async def _detail_notice(rig, conversation_id=None, workflow_id=None):
    """Show the real workflow detail view over the real coordinator; return
    its notice. A bare host: the full Console's workspace tree cannot compose
    while this chat holds a run (a KeyError on its running-attention style)."""
    import asyncio

    from textual.app import App

    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowStatus,
    )

    status = ConsoleWorkflowStatus(
        rig.coordinator,
        conversation_id or rig.workflow.request.conversation_id,
        workflow_id or rig.workflow.id,
        open_changes=lambda *args: None,
        open_fleet=lambda *args: None,
        pending_approvals=lambda *args: False,
        open_approval=lambda *args: None,
    )

    class Host(App):
        def on_mount(self):
            self.push_screen(status)

    def notice():
        found = status.query("#workflow-notice") if status.is_mounted else ()
        text = str(found.first().content) if found else ""
        return text if "unavailable" in text or "Restart Chatbook" in text else ""

    async with Host().run_test(size=(140, 42)) as pilot:
        async with asyncio.timeout(30):
            while not notice():
                await pilot.pause(0.05)
        return notice()


async def unproven_earlier_read(tmp_path):
    import asyncio
    import contextlib

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Chat import console_workflow_runs
    from tldw_chatbook.UI.Console_Modules.workflows import workflow_view_error

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    try:
        run, item = await _retained_unlaunched(rig)
        assert run.final_metadata is None
        with owned_goal_storage(rig.service):
            # Another chat's workflow, which holds nothing.
            other = rig.service.db.workflow_runs.create(
                rig.workflow.request.model_copy(update={"conversation_id": "other"}),
                launch_id="other-chat-workflow",
            )
        storage_return = console_workflow_runs._remaining_storage_return

        @contextlib.contextmanager
        def handle_lost(databases):
            with storage_return(databases):
                yield
            raise RuntimeError("original source database handle did not return")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                console_workflow_runs, "_remaining_storage_return", handle_lost
            )
            with pytest.raises(RuntimeError) as became:
                # Its own task, as a view's read is: shutdown cannot drain a
                # read issued by the task that disposes the runtime.
                await asyncio.create_task(rig.coordinator.get(rig.workflow.id))
        # The read that becomes unproven already names a restart.
        assert became.value.args == ("workflow_return_unproven",), became.value
        _restart_only(became.value)
        assert "Refresh" not in workflow_view_error("Review", became.value)
        # An ordinary status read, not the run's final one, is now unproven.
        assert run.final_metadata is None
        assert [f.result() for f in rig.coordinator._metadata_cells.values()] == [False]
        _restart_only(await _refused(rig, "workflow_return_unproven", "reconcile-read"))
        assert not item._closed and item in rig.runtime._workflow_preparations
        # The marker covers every workflow, so another chat's view names a
        # restart without claiming that chat is held.
        notice = await _detail_notice(rig, "other", other.id)
        assert "Restart Chatbook to approve or revise work" in notice, notice
        assert "this chat" not in notice, notice
        assert rig.coordinator._runs.get(rig.workflow.id) is run
        assert not (
            await rig.controller.submit_draft(
                "Still held", session_id="workflow-test-chat"
            )
        ).accepted
    finally:
        await _close_unresolved(rig, "workflow_original_metadata_shutdown_unresolved")


async def _unproven_other_read(rig):
    """Another chat's workflow read loses a storage handle: the coordinator's
    unproven-read marker is set without touching this workflow's run."""
    import asyncio
    import contextlib

    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
        preparation_read_for,
    )
    from tldw_chatbook.Chat import console_workflow_runs

    with owned_goal_storage(rig.service):
        other = rig.service.db.workflow_runs.create(
            rig.workflow.request.model_copy(update={"conversation_id": "other"}),
            launch_id="other-chat-workflow",
        )
    storage_return = console_workflow_runs._remaining_storage_return
    reader = []

    @contextlib.contextmanager
    def handle_lost(databases):
        read = preparation_read_for(rig.controller)
        with storage_return(databases):
            yield
        if read is not None and read.task in reader:  # Only that read.
            raise RuntimeError("original source database handle did not return")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(console_workflow_runs, "_remaining_storage_return", handle_lost)
        reader.append(asyncio.create_task(rig.coordinator.get(other.id)))
        with pytest.raises(RuntimeError, match="workflow_return_unproven"):
            await reader[0]
    assert rig.coordinator._metadata_unproven()


async def _held_at_wave(rig, gate):
    """Approve phase-1 and hold its running driver at the wave call, with
    task-1 prepared, until `gate` is set; then the real wave runs."""
    import asyncio

    from tldw_chatbook.Chat import console_workflow_wave

    original = console_workflow_wave.run_workflow_wave
    entered = asyncio.Event()

    async def gated(phase, selected, **kwargs):
        entered.set()
        await gate.wait()
        return await original(phase, selected, **kwargs)

    decision = await _started(rig)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(console_workflow_wave, "run_workflow_wave", gated)
        await rig.coordinator.approve(decision, command_id="approved-once")
        run = rig.coordinator._runs[rig.workflow.id]
        await asyncio.wait_for(entered.wait(), 60)
    return run


async def stop_under_unproven_read(tmp_path):
    import asyncio

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.UI.Console_Modules.workflows import workflow_view_error

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    gate = asyncio.Event()
    try:
        run = await _held_at_wave(rig, gate)
        await _unproven_other_read(rig)
        running = await rig.service.call(rig.service.get, rig.workflow.id)
        assert not run.driver.done() and run.control is None  # Still running.
        with pytest.raises(RuntimeError) as applied:
            await rig.coordinator.stop(
                rig.workflow.id, expected_control_revision=running.control_revision
            )
        assert applied.value.args == ("workflow_command_applied",), applied.value
        assert applied.value.__cause__.args == ("workflow_return_unproven",)
        # Stop took effect: the phase is revoked and the stop is stored.
        assert run.control == "stop"
        stopped = await rig.service.call(rig.service.get, rig.workflow.id)
        assert stopped.status == "stopped", stopped.status
        assert stopped.control_revision > running.control_revision
        message = workflow_view_error("Stop", applied.value)
        assert message.startswith("Stop applied"), message
        assert "Restart Chatbook to read workflows again" in message, message
        # A read-back that failed for another reason offers a refresh.
        other = workflow_view_error("Pause", RuntimeError("workflow_command_applied"))
        assert other.startswith("Pause applied") and "Refresh to read" in other
        gate.set()
        await asyncio.wait({run.driver}, timeout=60)
        assert run.driver.done()
        assert [key for key, _ in rig.provider.calls] == ["planning"]
    finally:
        gate.set()
        # The marker stays until restart (TASK-41320).
        await _close_unresolved(rig, "workflow_original_metadata_shutdown_unresolved")


async def detail_stop_under_unproven_read(tmp_path):
    import asyncio

    from textual.app import App
    from textual.widgets import Button

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Widgets.Console.console_workflow_status import (
        ConsoleWorkflowStatus,
    )

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    gate = asyncio.Event()
    try:
        run = await _held_at_wave(rig, gate)
        await _unproven_other_read(rig)
        # The real detail view over the real coordinator, in a bare host (see
        # `_detail_notice`).
        status = ConsoleWorkflowStatus(
            rig.coordinator,
            rig.workflow.request.conversation_id,
            rig.workflow.id,
            open_changes=lambda *args: None,
            open_fleet=lambda *args: None,
            pending_approvals=lambda *args: False,
            open_approval=lambda *args: None,
        )

        class Host(App):
            def on_mount(self):
                self.push_screen(status)

        def button(name):
            return status.query_one("#workflow-" + name, Button)

        def notice():
            return str(status.query_one("#workflow-notice").content)

        async with Host().run_test(size=(140, 42)) as pilot:

            async def until(check):
                async with asyncio.timeout(60):
                    while not (status.is_mounted and check()):
                        await pilot.pause(0.05)

            await until(lambda: status.snapshot is not None and not button("stop").disabled)
            assert status.snapshot.status == "running", status.snapshot.status
            assert "Restart Chatbook to approve or revise work" in notice(), notice()
            button("stop").press()
            await until(lambda: status._action_notice.startswith("Stop applied"))
            await until(lambda: not status._sending and not status._reading)
            assert "Stop applied" in notice(), notice()
        # The Stop took effect: the phase is revoked and the stop is stored.
        assert run.control == "stop"
        stored = await rig.service.call(rig.service.get, rig.workflow.id)
        assert stored.status == "stopped", stored.status
        gate.set()
        await asyncio.wait({run.driver}, timeout=60)
        assert [key for key, _ in rig.provider.calls] == ["planning"]
    finally:
        gate.set()
        await _close_unresolved(rig, "workflow_original_metadata_shutdown_unresolved")


async def refusal_after_decision(tmp_path):
    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    try:
        run, item = await _retained_unlaunched(rig)
        original = run.phase._control

        def stored_then_refused(decision=None, *, expected_control_revision=None):
            original(decision, expected_control_revision=expected_control_revision)
            raise AutomaticWorkRefused("workflow_authority_changed")

        current = await rig.service.call(rig.service.get, rig.workflow.id)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(run.phase, "_control", stored_then_refused)
            with pytest.raises(RuntimeError, match="control refusal unproven"):
                await rig.coordinator.stop(
                    rig.workflow.id, expected_control_revision=current.control_revision
                )
        assert run.control_unknown and run.control_return.is_set()
        stored = await rig.service.call(rig.service.get, rig.workflow.id)
        assert stored.status == "stopped", stored.status  # The decision is stored.
        _restart_only(
            await _refused(rig, "workflow_return_unproven", "reconcile-control")
        )
    finally:
        await _close_unresolved(rig, "workflow_original_shutdown_unresolved")


async def stop_races_phase_end(tmp_path):
    import asyncio

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.automatic_work_budget import AutomaticWorkRefused

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    coordinator = rig.coordinator
    publish = coordinator._publish
    at_release, resume = asyncio.Event(), asyncio.Event()

    async def held_terminal(workflow_id, *, _terminal_run=None):
        # The phase has settled and retired; hold its driver before release.
        if _terminal_run is not None:
            at_release.set()
            await resume.wait()
        return await publish(workflow_id, _terminal_run=_terminal_run)

    try:
        decision = await _started(rig)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(coordinator, "_publish", held_terminal)
            await coordinator.approve(decision, command_id="approved-once")
            run = coordinator._runs[rig.workflow.id]
            try:
                await asyncio.wait_for(at_release.wait(), 60)
                assert run.phase._retired
                current = await rig.service.call(rig.service.get, rig.workflow.id)
                # Stop passes its revision check; the retired phase refuses it.
                with pytest.raises(
                    AutomaticWorkRefused, match="workflow_authority_changed"
                ):
                    await coordinator.stop(
                        rig.workflow.id,
                        expected_control_revision=current.control_revision,
                    )
                assert not run.control_unknown and run.control_return.is_set()
                assert run.control is None and run.status != "recovery_required"
            finally:
                resume.set()
            await asyncio.wait({run.driver}, timeout=60)
        assert run.driver.done() and rig.workflow.id not in coordinator._runs
        sent = await rig.controller.submit_draft(
            "A new turn after the refused Stop", session_id=session_id
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get(session_id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        await rig.close()


async def approve_under_unproven_read(tmp_path):
    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    try:
        decision = await _started(rig)
        await _unproven_other_read(rig)
        stored = await rig.service.call(rig.service.get, rig.workflow.id)
        # Refused: its driver could not release the chat under the marker.
        with pytest.raises(ValueError, match="^workflow_return_unproven$"):
            await rig.coordinator.approve(decision, command_id="approve-under-marker")
        assert rig.workflow.id not in rig.coordinator._runs
        wake = rig.controller._fleet_wake
        assert session_id not in wake._automatic_primary_claims
        assert await rig.service.call(rig.service.get, rig.workflow.id) == stored
        assert [key for key, _ in rig.provider.calls] == ["planning"]
        # Allowed: Stop needs no proven read, so it still applies.
        with pytest.raises(RuntimeError, match="^workflow_command_applied$"):
            await rig.coordinator.stop(
                rig.workflow.id, expected_control_revision=stored.control_revision
            )
        stopped = await rig.service.call(rig.service.get, rig.workflow.id)
        assert stopped.status == "stopped", stopped.status
    finally:
        await _close_unresolved(rig, "workflow_original_metadata_shutdown_unresolved")


async def phase_running_under_unproven_read(tmp_path):
    import asyncio
    import functools
    import threading
    from types import MethodType

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider(hold={"task-1"}))
    try:
        decision = await _started(rig)  # task-2 depends on task-1
        service = rig.service
        original = service.settle_execution_phase
        outcomes = []

        @functools.wraps(original)
        def settle(self, phase, **kwargs):
            outcomes.append(kwargs.get("outcome"))
            return original(phase, **kwargs)

        entered = rig.provider.entered.setdefault("task-1", threading.Event())
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                service, "settle_execution_phase", MethodType(settle, service)
            )
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            # task-1's provider call is in flight (held) when the marker is set.
            assert await asyncio.to_thread(entered.wait, 60)
            await _unproven_other_read(rig)
            rig.provider.released.set()
            await asyncio.wait({run.driver}, timeout=60)
        assert run.driver.done()
        # task-1 finished; the wave boundary's publication was refused, so the
        # phase failed there and task-2 never ran.
        assert [key for key, _ in rig.provider.calls] == ["planning", "task-1"]
        assert outcomes == ["failed"], outcomes
        stored = await service.call(service.get, rig.workflow.id)
        assert [result.status for result in stored.results] == ["completed"]
        # Its terminal publication is refused too: the chat stays held.
        assert rig.coordinator._runs.get(rig.workflow.id) is run
        assert not (
            await rig.controller.submit_draft(
                "Still held", session_id="workflow-test-chat"
            )
        ).accepted
    finally:
        rig.provider.released.set()
        await _close_unresolved(rig, "workflow_original_metadata_shutdown_unresolved")


async def finalize_during_record(tmp_path):
    import asyncio
    import threading

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig
    from tldw_chatbook.Agents.goal_storage_lifetime import owned_goal_storage
    from tldw_chatbook.Agents.workflow_models import WorkflowExecutionRecovery
    from tldw_chatbook.Chat import console_workflow_wave

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider(hold=("task-1",)))
    original = console_workflow_wave.run_workflow_wave
    waves = []

    async def failing_while_live(phase, selected, **kwargs):
        waves.append(asyncio.ensure_future(original(phase, selected, **kwargs)))
        await rig.wait_for_provider_entry("task-1")
        raise RuntimeError("wave observer failed")

    try:
        decision = await _started(rig)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(
                console_workflow_wave, "run_workflow_wave", failing_while_live
            )
            await rig.coordinator.approve(decision, command_id="approved-once")
            run = rig.coordinator._runs[rig.workflow.id]
            await asyncio.wait_for(asyncio.shield(run.driver), 60)
        rig.provider_release()
        await asyncio.wait_for(asyncio.gather(*waves, return_exceptions=True), 60)
        phase, service = run.phase, rig.service
        returned = await service.call(service.get, rig.workflow.id)
        live_check = service._has_live_workflow_resources
        settled = []

        def finalize():
            try:
                with owned_goal_storage(service):
                    settled.append(
                        service.settle_execution_phase(phase, outcome="failed")
                    )
            except BaseException as error:  # noqa: BLE001 - reported below.
                settled.append(error)

        def check_then_finalize(workflow_id):
            live = live_check(workflow_id)
            if not settled:
                finalizer = threading.Thread(target=finalize)
                finalizer.start()
                finalizer.join(10)
            return live

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(service, "_has_live_workflow_resources", check_then_finalize)
            with pytest.raises(ValueError, match="revision_conflict"):
                await service.call(
                    service.record_returned_execution,
                    phase,
                    expected_control_revision=returned.control_revision + 5,
                )
        (outcome,) = settled
        assert not isinstance(outcome, BaseException), outcome
        assert phase._retired, "a refused record undid the finalized retirement"
        current = await service.call(service.get, rig.workflow.id)
        closed = await rig.coordinator.reconcile(
            rig.workflow.id,
            expected_control_revision=current.control_revision,
            decision=WorkflowExecutionRecovery(reason="reviewed_prior_effects"),
            command_id="reconcile-finalized",
        )
        assert closed.status == "stopped"
        assert rig.workflow.id not in rig.coordinator._runs
    finally:
        rig.provider_release()
        await asyncio.wait_for(asyncio.gather(*waves, return_exceptions=True), 60)
        await rig.close()


async def close_raises_in_failure_handler(tmp_path):
    import asyncio

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    session_id = "workflow-test-chat"
    closes = []

    def break_first_close(selected):
        for item in selected:
            real = item.close

            async def close(real=real):
                closes.append(1)
                await real()
                if len(closes) == 1:
                    # Closed, then refused its own bookkeeping: benign.
                    raise PermissionError("launch in flight")

            item.close = close

    try:
        run, item = await _failed_before_launch(
            rig, RuntimeError("wave refused before launch"), on_prepared=break_first_close
        )
        assert closes and run.status != "recovery_required"
        assert (await rig.coordinator.get(rig.workflow.id)).status == "paused"
        sent = await rig.controller.submit_draft(
            "A new turn after the failed phase", session_id=session_id
        )
        assert sent.accepted, sent
        stream = rig.controller._active_stream_tasks.get(session_id)
        if stream is not None:
            await asyncio.wait_for(asyncio.shield(stream), 15)
    finally:
        await rig.close()


async def view_refuses_closing_and_stale(tmp_path):
    import asyncio

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider())
    gate = asyncio.Event()
    coordinator = rig.coordinator
    try:
        await _held_at_wave(rig, gate)
        await _unproven_other_read(rig)
        assert coordinator.reads_unproven
        shown = await coordinator.view(rig.workflow.id)
        # A snapshot older than what this coordinator already published.
        coordinator._metadata_floor[rig.workflow.id] = (
            shown.control_revision + 5,
            shown.progress_cursor + 5,
        )
        with pytest.raises(ValueError, match="workflow_snapshot_superseded"):
            await coordinator.view(rig.workflow.id)
        coordinator._metadata_floor.pop(rig.workflow.id)
        assert (await coordinator.view(rig.workflow.id)).id == shown.id
        coordinator._closing = True
        try:
            with pytest.raises(RuntimeError, match="workflow coordinator closed"):
                await coordinator.view(rig.workflow.id)
        finally:
            coordinator._closing = False
    finally:
        gate.set()
        await _close_unresolved(rig, "workflow_original_metadata_shutdown_unresolved")
