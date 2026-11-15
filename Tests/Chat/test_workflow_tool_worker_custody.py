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
