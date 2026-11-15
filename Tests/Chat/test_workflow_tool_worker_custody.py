"""A workflow's tool worker that holds an unproven cleanup never hangs a join (TASK-40716).

Planning and native close used to ``join()`` every tool worker. A worker that
holds an unproven cleanup's scopes waits until a late proof or process exit,
so those joins (and disposal behind them) hung. The phase now fails its
physical-return proof instead, and the lifetime stays held.
"""

import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity, WorkOrigin
from tldw_chatbook.Chat import console_workflow_preparation as preparation


@pytest.fixture
def custody():
    capacity = RuntimeCapacity()
    owner = capacity.begin_execution(origin=WorkOrigin.MANUAL, conversation_id="c")
    retention = owner.reserve_tool().retain_unproven("fs_read")
    holding = threading.Event()

    def hold():
        holding.set()
        retention.hold()

    worker = threading.Thread(target=hold, daemon=True)
    worker.start()
    assert holding.wait(5)
    try:
        yield SimpleNamespace(capacity=capacity, worker=worker)
    finally:
        retention.prove()
        worker.join(5)


def _prepared(workers):
    prepared = object.__new__(preparation._PreparedWorkflowReply)
    prepared._tool_workers = list(workers)
    return prepared


def test_tool_workers_returned_joins_the_rest_but_never_a_retained_worker(custody):
    finished = threading.Thread(target=lambda: None)
    finished.start()
    assert _prepared([finished])._tool_workers_returned()
    assert not _prepared([finished, custody.worker])._tool_workers_returned()
    assert custody.worker.is_alive() and not finished.is_alive()


def test_native_close_refuses_instead_of_waiting_on_a_retained_worker(
    custody, monkeypatch
):
    monkeypatch.setattr(preparation, "_storage_source_current", lambda *a, **k: None)
    prepared = _prepared([custody.worker])
    prepared.source = None
    prepared.authorization = SimpleNamespace(_lock=threading.Lock())
    close_owner = custody.capacity.begin_execution(
        origin=WorkOrigin.MANUAL, conversation_id="c"
    )
    prepared._close_operation = close_owner.reserve_tool()
    prepared._native_closed = False
    with pytest.raises(PermissionError, match="tool cleanup unproven"):
        prepared._close_on_native_owner(close_owner)
    assert not prepared._native_closed


class _Run:
    """The fields of a workflow run the coordinator's shutdown reads."""

    def __init__(self, driver, workers):
        self.driver, self.wake, self.phase, self.control = driver, None, object(), None
        self.prepared = SimpleNamespace(
            _tool_workers=list(workers),
            _planning_recovery_return=None,
            cancel_event=threading.Event(),
        )


def _coordinator(runs):
    from tldw_chatbook.Chat.console_workflow_runs import ConsoleWorkflowCoordinator

    coordinator = object.__new__(ConsoleWorkflowCoordinator)
    runtime = SimpleNamespace(_workflow_waves=(), _workflow_coordinator=coordinator)
    coordinator.runtime = runtime
    coordinator.controller = SimpleNamespace(app=SimpleNamespace(console_runtime=runtime))
    coordinator._subscribers, coordinator._commands = [], {}
    coordinator._metadata_cells, coordinator._pending_planning = {}, set()
    coordinator._runs = {f"w{index}": run for index, run in enumerate(runs)}
    return coordinator


def test_a_run_with_a_live_worker_is_not_retained_only(custody):
    """One retained worker must not hide another that may still act."""
    done = SimpleNamespace(done=lambda: True)
    release = threading.Event()
    live = threading.Thread(target=release.wait, daemon=True)
    live.start()
    run = _Run(done, [custody.worker, live])
    try:
        assert not _coordinator([run])._retained_only()
    finally:
        release.set()
        live.join(5)
    assert _coordinator([run])._retained_only()
    assert not _coordinator([_Run(done, [live])])._retained_only()


@pytest.mark.asyncio
async def test_retained_shutdown_keeps_a_requested_cancellation(custody):
    """The cancellation close would return is chained, not swallowed."""
    import asyncio

    from tldw_chatbook.Chat.console_workflow_runs import WorkflowShutdownRetained

    release = asyncio.Event()
    driver = asyncio.create_task(release.wait())
    coordinator = _coordinator([_Run(driver, [custody.worker])])
    close = asyncio.create_task(
        coordinator.close(_runtime_disposal=coordinator.runtime)
    )
    await asyncio.sleep(0)
    close.cancel()  # the disposer's own task is cancelled while it drains
    await asyncio.sleep(0)
    release.set()
    cancellation = await close
    assert isinstance(cancellation, asyncio.CancelledError)
    assert isinstance(cancellation.__cause__, WorkflowShutdownRetained)
    assert str(cancellation.__cause__) == "workflow_original_shutdown_unresolved"


_PLANNING_SCRIPT = r"""
import asyncio, os, sys
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install(); real_profile_guard.install()
os.environ.update(TLDW_AGENTS_WORKFLOW_RUNS_ENABLED="true", TLDW_AGENTS_AUTOWAKE_ENABLED="false", OPENAI_API_KEY="private-workflow-test")
from Tests.Chat.test_workflow_tool_worker_custody import retained_planning_worker_journey
with user_fixture_default_owner():
    asyncio.run(retained_planning_worker_journey(Path(os.environ["XDG_DATA_HOME"]).parent))
"""


@pytest.mark.bootstrap_profile
def test_planning_keeps_its_lifetime_when_a_tool_worker_is_retained(tmp_path):
    from Tests.Backup_Recovery.test_home_citation_retirement import _run

    _run(tmp_path, "retained-planning", "native", script=_PLANNING_SCRIPT, timeout=90)


async def retained_planning_worker_journey(tmp_path):
    """Drive the real planning ``finally`` with a worker holding unproven custody."""
    import asyncio
    import os
    import traceback

    from Tests.Chat.workflow_rig import WorkflowProvider, build_workflow_rig

    rig = await build_workflow_rig(tmp_path, None, WorkflowProvider(hold=("planning",)))
    retention = None
    try:
        rig.workflow = await rig.coordinator.start(
            rig.workflow.request, launch_id=rig.workflow.launch_id
        )
        await rig.wait_for_provider_entry()
        run = rig.coordinator._runs[rig.workflow.id]
        prepared = run.prepared
        # A planning tool worker whose cleanup is unproven holds its custody.
        retention = prepared.primary_owner.reserve_tool().retain_unproven("fs_read")
        holding = threading.Event()

        def hold():
            holding.set()
            retention.hold()

        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        assert holding.wait(5)
        prepared._tool_workers.append(worker)
        exits, refusals = [], []
        lifetime_exit = prepared._lifetime.__exit__
        prepared._lifetime.__exit__ = lambda *args: (
            exits.append(args),
            lifetime_exit(*args),
        )[1]
        proof = prepared._planning_physical_proof

        def observed_proof():
            try:
                return proof()
            except RuntimeError as error:
                refusals.append(str(error))
                raise

        prepared._planning_physical_proof = observed_proof
        rig.provider_release()
        await asyncio.wait_for(asyncio.gather(run.driver, return_exceptions=True), 30)
        # The planning finally kept the lifetime instead of joining the worker...
        assert exits == [] and not prepared._native_closed
        # ...so the phase failed its physical-return proof.
        assert refusals and worker.is_alive()
        # Disposal returns instead of waiting on the retained worker, closes
        # every other owner, then reports the planning run unresolved rather
        # than claim its retirement.
        from tldw_chatbook.Chat.console_workflow_runs import WorkflowShutdownRetained

        try:
            await asyncio.wait_for(rig.runtime.dispose(), 30)
        except WorkflowShutdownRetained as error:
            assert str(error) == "workflow_original_shutdown_unresolved"
        else:
            raise AssertionError("disposal claimed the retained planning retired")
        capacity = rig.runtime.execution_capacity
        assert capacity.snapshot().closed  # the rest of disposal ran
        assert exits == [] and worker.is_alive()
        assert retention in capacity.retained()
        print("retired and reopened", flush=True)  # the end marker `_run` expects
        os._exit(0)  # The retained custody ends only with this private process.
    except BaseException:
        traceback.print_exc()
        os._exit(17)
