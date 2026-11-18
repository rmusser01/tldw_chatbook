"""Queued-read transport controls; these do not claim native retirement proof."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.Chat.test_console_execution_process import _ready_client, _reply, _retrieve
from Tests.Chat.test_console_execution_workspace import _until
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost, ConsoleRemoteFailure


def _frame(client, receipt, kind, payload):
    client._receive(_reply(client, receipt, client._received_sequence + 1, kind, payload))


def _refuse(client, receipt, *, retired=True):
    _frame(client, receipt, "received", {"sequence": receipt.identity.sequence})
    _frame(client, receipt, "failed", {"code": "source_changed", "stage": "operation"})
    _frame(client, receipt, "retired", {
        "operation_id": receipt.identity.request_id, "retired": retired,
    })


async def _cleanup(client, observers):
    client._fail(RuntimeError("transport control cleanup"))
    owners = tuple(client._projection_reads)
    for task in owners:
        task.cancel()
    await asyncio.gather(*observers, *owners, return_exceptions=True)
    if client._projection_lock.locked():
        client._projection_lock.release()  # Only the test's initial hold remains.


@pytest.mark.asyncio
async def test_shared_success_keeps_each_callers_post_read_work(monkeypatch):
    client = _ready_client()
    release = asyncio.Event()
    calls, presented = [], []
    result = object()

    async def original_work(*args, **kwargs):
        # Opaque scheduling seam only: no mock retirement/authority is adopted.
        calls.append((args, kwargs))
        await release.wait()
        return result

    monkeypatch.setattr(client, "_read_projection_locked", original_work)

    async def caller(name):
        value = await client.read_projection("session", "incarnation")
        presented.append(name)
        return value

    await client._projection_lock.acquire()
    observers = [asyncio.create_task(caller("periodic"))]
    try:
        await _until(lambda: len(client._projection_reads) == 1)
        observers.append(asyncio.create_task(caller("terminal")))
        await asyncio.sleep(0)
        client._projection_lock.release()
        await _until(lambda: len(calls) == 1)
        assert not presented
        release.set()
        assert await asyncio.gather(*observers) == [result, result]
        assert sorted(presented) == ["periodic", "terminal"] and len(calls) == 1
    finally:
        release.set()
        await _cleanup(client, observers)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_first", [False, True])
async def test_only_unissued_plain_reads_share_one_original_receipt(cancel_first):
    client = _ready_client()
    observers = []
    await client._projection_lock.acquire()
    try:
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 1)
        owner = next(iter(client._projection_reads))
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert client._projection_reads == {owner}
        assert not client._channel.messages
        if cancel_first:
            observers[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await observers[0]
            assert not owner.done()
        client._projection_lock.release()
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        _frame(client, receipt, "received", {"sequence": receipt.identity.sequence})
        _frame(client, receipt, "failed", {"code": "source_changed", "stage": "operation"})
        await asyncio.sleep(0)
        assert not observers[1].done() and not owner.done()
        assert client._projection_lock.locked()  # Failed Applied still owes Retired.
        _frame(client, receipt, "retired", {
            "operation_id": receipt.identity.request_id, "retired": True,
        })
        results = await asyncio.gather(*observers, return_exceptions=True)
        assert isinstance(results[1], ConsoleRemoteFailure)
        assert isinstance(results[0], asyncio.CancelledError if cancel_first else ConsoleRemoteFailure)
        assert len(client._channel.messages) == 1 and receipt.retired.result() is True
        await asyncio.sleep(0)
        assert not client._projection_reads
    finally:
        await _cleanup(client, observers)


@pytest.mark.asyncio
async def test_issued_full_is_never_reused_by_a_later_terminal_caller():
    client = _ready_client()
    observers = [asyncio.create_task(client.read_projection("session", "incarnation"))]
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        first = next(iter(client._requests.values()))
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 2)
        assert len(client._channel.messages) == 1
        _refuse(client, first)
        await _until(lambda: len(client._channel.messages) == 2)
        second = next(iter(client._requests.values()))
        assert second is not first
        _refuse(client, second)
        results = await asyncio.gather(*observers, return_exceptions=True)
        assert all(isinstance(result, ConsoleRemoteFailure) for result in results)
        assert first.retired.result() is True and second.retired.result() is True
    finally:
        await _cleanup(client, observers)


@pytest.mark.asyncio
async def test_intervening_actual_command_ends_the_unissued_sharing_window():
    client = _ready_client()
    observers = []
    other = None
    await client._projection_lock.acquire()
    try:
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 1)
        other = client.submit("ReadDecisionHead", {"attachment_id": "view", "visit": 1},
                              session_id="session", incarnation_id="incarnation")
        _refuse(client, other)
        await _retrieve(other)
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 2)
        client._projection_lock.release()
        for count in (2, 3):
            await _until(lambda count=count: len(client._channel.messages) == count)
            receipt = next(iter(client._requests.values()))
            assert receipt.operation == "ReadProjection"
            _refuse(client, receipt)
        assert all(isinstance(result, ConsoleRemoteFailure)
                   for result in await asyncio.gather(*observers, return_exceptions=True))
        assert [item.operation for item in client._channel.messages] == [
            "ReadDecisionHead", "ReadProjection", "ReadProjection",
        ]
    finally:
        await _cleanup(client, observers)
        if other is not None:
            await _retrieve(other)


@pytest.mark.asyncio
async def test_cancelled_unissued_owner_does_not_poison_the_next_read():
    client = _ready_client()
    observers = []
    await client._projection_lock.acquire()
    try:
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 1)
        original = next(iter(client._projection_reads))
        original.cancel()
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await asyncio.sleep(0)
        client._projection_lock.release()
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        _refuse(client, receipt)
        results = await asyncio.gather(*observers, return_exceptions=True)
        assert isinstance(results[0], asyncio.CancelledError)
        assert isinstance(results[1], ConsoleRemoteFailure)
        assert receipt.retired.result() is True
    finally:
        await _cleanup(client, observers)


@pytest.mark.asyncio
async def test_queued_new_chat_breaks_plain_read_sharing_before_any_command(monkeypatch):
    client = _ready_client()
    view = object()
    row = SimpleNamespace(id="session", incarnation_id="incarnation")
    client.projection_view = SimpleNamespace(active_session_id=row.id, session=lambda _: row)
    monkeypatch.setattr(client, "require_view", lambda *_: None)
    await client._projection_lock.acquire()
    observers = [asyncio.create_task(client.read_projection(row.id, row.incarnation_id))]
    opening = None
    try:
        await _until(lambda: len(client._projection_reads) == 1)
        opening = asyncio.create_task(client.open_new_chat(view, workspace_id="workspace", title="new"))
        await _until(lambda: bool(client._opening_sessions))
        observers.append(asyncio.create_task(client.read_projection(row.id, row.incarnation_id)))
        await _until(lambda: len(client._projection_reads) == 2)
        assert not client._channel.messages
    finally:
        owners = tuple(client._opening_sessions)
        for task in owners:
            task.cancel()
        await asyncio.gather(*owners, *(() if opening is None else (opening,)), return_exceptions=True)
        await _cleanup(client, observers)


@pytest.mark.asyncio
@pytest.mark.parametrize("barrier", ["mount", "preparation", "history", "inspector", "session"])
async def test_special_or_different_scope_request_breaks_pending_plain_sharing(barrier):
    client = _ready_client()
    observers = []
    await client._projection_lock.acquire()
    try:
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 1)
        options = {
            "mount": {"_checkpoint": "mount"},
            "preparation": {"_preparation": object()},
            "history": {"_run_history": {}},
            "inspector": {"_inspector_refresh": "dictionary"},
            "session": {},
        }[barrier]
        observers.append(asyncio.create_task(client.read_projection(
            "other" if barrier == "session" else "session", "incarnation", **options,
        )))
        await _until(lambda: len(client._projection_reads) == 2)
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await asyncio.sleep(0)
        assert len(client._projection_reads) == 3 and not client._channel.messages
    finally:
        # Cancel unissued owners directly for this queue barrier control. There
        # is no issued receipt or native resource to manufacture settlement for.
        for task in tuple(client._projection_reads):
            task.cancel()
        await _cleanup(client, observers)


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["source", "visit", "attachment"])
async def test_scope_change_does_not_attach_to_old_unissued_owner(changed):
    client = _ready_client()
    observers = []
    await client._projection_lock.acquire()
    try:
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 1)
        if changed == "source":
            client._source_revision += 1
        elif changed == "visit":
            client._visit += 1
        else:
            client._attachment_id = "another-attachment"
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 2)
        assert not client._channel.messages
    finally:
        for task in tuple(client._projection_reads):
            task.cancel()
        await _cleanup(client, observers)


@pytest.mark.asyncio
async def test_negative_retirement_is_not_shared_success_or_a_new_publication():
    client = _ready_client()
    observers = []
    await client._projection_lock.acquire()
    try:
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await _until(lambda: len(client._projection_reads) == 1)
        observers.append(asyncio.create_task(client.read_projection("session", "incarnation")))
        await asyncio.sleep(0)
        client._projection_lock.release()
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        _refuse(client, receipt, retired=False)
        results = await asyncio.gather(*observers, return_exceptions=True)
        assert all(isinstance(result, ConsoleChannelLost) for result in results)
        assert client._projection is None and client._channel_error is not None
        assert len(client._channel.messages) == 1
    finally:
        await _cleanup(client, observers)
