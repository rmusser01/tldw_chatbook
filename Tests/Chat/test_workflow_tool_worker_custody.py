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
        # Disposal returns instead of waiting on the retained worker. It
        # reports the planning run unresolved rather than claim retirement.
        try:
            await asyncio.wait_for(rig.runtime.dispose(), 30)
        except RuntimeError as error:
            assert str(error) == "workflow_original_shutdown_unresolved"
        else:
            raise AssertionError("disposal claimed the retained planning retired")
        assert exits == [] and worker.is_alive()
        print("retired and reopened", flush=True)  # the end marker `_run` expects
        os._exit(0)  # The retained custody ends only with this private process.
    except BaseException:
        traceback.print_exc()
        os._exit(17)
