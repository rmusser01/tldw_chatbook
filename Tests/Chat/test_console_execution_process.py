"""Transport receipt controls; these do not qualify a backend/App composition."""

from __future__ import annotations

import asyncio
import io
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_execution_process import (
    ConsoleChannelLost,
    ConsoleExecutionClient,
    ConsoleRemoteFailure,
    _ConsoleFramedChannel,
)
from tldw_chatbook.Chat.console_execution_protocol import (
    ConsoleProtocolError,
    ConsoleWireMessage,
    WireIdentity,
    encode_message,
)
from tldw_chatbook.MCP.permission_process import ProtocolError, write_message


class _SentFrames:
    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


def _ready_client():
    client = ConsoleExecutionClient(None, profile_epoch="profile")
    client._loop = asyncio.get_running_loop()
    client._ready = client._loop.create_future()
    client._ready.set_result(None)
    client._channel = _SentFrames()
    return client


def _reply(client, receipt, sequence, kind, payload):
    source = receipt.identity
    return ConsoleWireMessage(
        WireIdentity(
            source.profile_epoch,
            source.service_epoch,
            sequence,
            source.request_id,
            source.session_id,
            source.incarnation_id,
            source.source_revision,
        ),
        kind,
        receipt.operation,
        payload,
    )


async def _retrieve(receipt):
    for future in (receipt.received, receipt.applied, receipt.retired):
        if future.done():
            await asyncio.gather(future, return_exceptions=True)
        else:
            future.cancel()


@pytest.mark.asyncio
async def test_projection_observer_cancel_keeps_next_read_behind_actual_retirement():
    from Tests.Chat.test_console_execution_workspace import _until

    client = _ready_client()
    first = asyncio.create_task(client.read_projection("session", "incarnation"))
    second = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        client._receive(_reply(client, receipt, 1, "received", {
            "sequence": receipt.identity.sequence,
        }))
        client._receive(_reply(client, receipt, 2, "failed", {
            "code": "source_changed", "stage": "operation",
        }))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.create_task(client.read_projection("session", "incarnation"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(client._channel.messages) == 1
        assert client._projection_lock.locked() and not receipt.retired.done()
        client._receive(_reply(client, receipt, 3, "retired", {
            "operation_id": receipt.identity.request_id, "retired": True,
        }))
        await _until(lambda: len(client._channel.messages) == 2)
        next_receipt = next(iter(client._requests.values()))
        client._receive(_reply(client, next_receipt, 4, "received", {
            "sequence": next_receipt.identity.sequence,
        }))
        client._receive(_reply(client, next_receipt, 5, "failed", {
            "code": "source_changed", "stage": "operation",
        }))
        client._receive(_reply(client, next_receipt, 6, "retired", {
            "operation_id": next_receipt.identity.request_id, "retired": True,
        }))
        with pytest.raises(ConsoleRemoteFailure):
            await second
        await asyncio.sleep(0)
        assert not client._projection_lock.locked() and not client._projection_reads
    finally:
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(*client._projection_reads, return_exceptions=True)
        if second is not None:
            await asyncio.gather(second, return_exceptions=True)
        await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [None, "before-refusal", "with-refusal"])
async def test_projection_admission_refusal_never_waits_for_an_unissued_retirement(cancel):
    from Tests.Chat.test_console_execution_workspace import _until

    client = _ready_client()
    # Cancel the owned operation itself to exercise cleanup's admission race;
    # cancellation of the public shielded observer is covered above.
    task = asyncio.create_task(client._read_projection_locked("session", "incarnation"))
    receipt = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        if cancel == "before-refusal":
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done() and not receipt.applied.done()
        client._receive(_reply(client, receipt, 1, "failed", {
            "code": "command_refused", "stage": "admission",
        }))
        if cancel == "with-refusal":
            task.cancel()
        await _until(task.done)
        if cancel is None:
            with pytest.raises(ConsoleRemoteFailure, match="admission"):
                await task
        else:
            with pytest.raises(asyncio.CancelledError):
                await task
        assert receipt.applied.exception() is receipt.received.exception()
        assert not receipt.retired.done()
        assert client._requests[receipt.identity.request_id] is receipt
        assert client._projection is None and client._channel_error is None
    finally:
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(task, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [True, False])
async def test_received_projection_refusal_still_requires_actual_retirement(retired):
    from Tests.Chat.test_console_execution_workspace import _until

    client = _ready_client()
    task = asyncio.create_task(client.read_projection("session", "incarnation"))
    receipt = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        client._receive(_reply(client, receipt, 1, "received", {
            "sequence": receipt.identity.sequence,
        }))
        # A stage label alone is insufficient once an actual receipt exists.
        client._receive(_reply(client, receipt, 2, "failed", {
            "code": "command_refused", "stage": "admission",
        }))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not task.done() and client._projection_lock.locked()
        client._receive(_reply(client, receipt, 3, "retired", {
            "operation_id": receipt.identity.request_id, "retired": retired,
        }))
        await _until(task.done)
        expected = ConsoleRemoteFailure if retired else ConsoleChannelLost
        with pytest.raises(expected):
            await task
        assert receipt.received.result() is None
        assert receipt.retired.result() is retired
        assert client._projection is None
        assert (client._channel_error is None) is retired
    finally:
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(*client._projection_reads, task, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("retired", [True, False])
async def test_applied_projection_joins_retirement_before_original_value_checks(retired):
    from Tests.Chat.test_console_execution_workspace import _until
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState
    from tldw_chatbook.Chat.console_control_dispatch import ConsoleProjection
    from tldw_chatbook.Chat.console_execution_projection_result import encode_projection_payload

    client = _ready_client()
    task = asyncio.create_task(client.read_projection("session", "incarnation"))
    receipt = None
    try:
        await _until(lambda: len(client._channel.messages) == 1)
        receipt = next(iter(client._requests.values()))
        client._receive(_reply(client, receipt, 1, "received", {
            "sequence": receipt.identity.sequence,
        }))
        frame = ConsoleProjection(0, "session", "incarnation", 0, (), ConsoleRunState(), None)
        client._receive(_reply(client, receipt, 2, "applied", encode_projection_payload(frame, messages=[])))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert receipt.applied.exception() is None
        assert not task.done() and client._projection_lock.locked()
        client._receive(_reply(client, receipt, 3, "retired", {
            "operation_id": receipt.identity.request_id, "retired": retired,
        }))
        await _until(task.done)
        if retired:
            # Successful Applied/Retired is necessary, not permission to skip
            # the unchanged selected-value checks. This frame lacks a sidecar.
            with pytest.raises(ConsoleProtocolError, match="projection_view_missing"):
                await task
        else:
            with pytest.raises(ConsoleChannelLost, match="retirement is unproven"):
                await task
        assert receipt.retired.result() is retired
        assert client._projection is None
        assert (client._channel_error is None) is retired
    finally:
        client._fail(RuntimeError("test cleanup"))
        await asyncio.gather(*client._projection_reads, task, return_exceptions=True)
        if receipt is not None:
            await _retrieve(receipt)


@pytest.mark.asyncio
async def test_pre_admission_failure_ends_receipt_wait_without_inventing_retirement():
    client = _ready_client()
    receipt = client.submit(
        "OpenSession",
        {
            "title": "New chat",
            "workspace_id": None,
            "conversation_id": None,
        },
    )
    client._receive(
        _reply(
            client,
            receipt,
            1,
            "failed",
            {
                "code": "source_changed",
                "stage": "admission",
            },
        )
    )
    try:
        with pytest.raises(ConsoleRemoteFailure, match="admission"):
            await receipt.received
        with pytest.raises(ConsoleRemoteFailure, match="admission"):
            await receipt.applied
        assert not receipt.retired.done()
        assert client._channel_error is None
    finally:
        await _retrieve(receipt)


@pytest.mark.asyncio
async def test_out_of_order_reply_closes_intake_without_positive_retirement():
    client = _ready_client()
    receipt = client.submit("Drain", {})
    client._receive(_reply(client, receipt, 2, "received", {"sequence": 1}))
    try:
        assert client._channel_error is not None
        assert client._intake_open is False
        assert receipt.retired.done()
        assert receipt.retired.exception() is not None
        assert not client.physically_closed
    finally:
        await _retrieve(receipt)


@pytest.mark.asyncio
async def test_observer_cancellation_does_not_cancel_later_operation_retirement():
    client = _ready_client()
    receipt = client.submit("Drain", {})

    async def observe():
        return await asyncio.shield(receipt.retired)

    observer = asyncio.create_task(observe())
    await asyncio.sleep(0)
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    assert not receipt.retired.done()
    client._receive(_reply(client, receipt, 1, "received", {"sequence": 1}))
    client._receive(_reply(client, receipt, 2, "applied", {"applied": True}))
    client._receive(
        _reply(
            client,
            receipt,
            3,
            "retired",
            {
                "operation_id": receipt.identity.request_id,
                "retired": False,
            },
        )
    )
    assert await receipt.retired is False
    assert receipt.applied.result() == {"applied": True}
    assert not client.physically_closed


def test_channel_construction_is_inert_before_owner_publication(monkeypatch):
    started = []

    def unexpected_start(thread):
        started.append(thread)
        raise AssertionError("Unpublished channel started a thread")

    monkeypatch.setattr(threading.Thread, "start", unexpected_start)
    channel = _ConsoleFramedChannel(
        io.BytesIO(),
        io.BytesIO(),
        SimpleNamespace(),
        lambda value: None,
        lambda error: None,
    )
    assert started == []
    assert channel._started == []


def test_partial_pipe_close_still_closes_other_pipes():
    class BrokenClose(io.BytesIO):
        def close(self):
            raise OSError("original close failure")

    errors = []
    client = ConsoleExecutionClient(None)
    client._loop = SimpleNamespace(
        call_soon_threadsafe=lambda callback, error: errors.append(error)
    )
    client._process = SimpleNamespace(
        stdin=BrokenClose(),
        stdout=io.BytesIO(),
        stderr=io.BytesIO(),
    )
    assert client._close_pipes() is False
    assert not client._process.stdin.closed
    assert client._process.stdout.closed
    assert client._process.stderr.closed
    assert len(errors) == 1
    io.BytesIO.close(client._process.stdin)


def _lifecycle_reply_bytes(client, receipt, *, retired):
    stream = io.BytesIO()
    for sequence, (kind, payload) in enumerate(
        (
            ("received", {"sequence": receipt.identity.sequence}),
            ("applied", {"applied": True}),
            (
                "retired",
                {"operation_id": receipt.identity.request_id, "retired": retired},
            ),
        ),
        start=1,
    ):
        write_message(
            stream, encode_message(_reply(client, receipt, sequence, kind, payload))
        )
    return stream.getvalue()


async def _read_reply_bytes(client, value):
    class FragmentedInput(io.BytesIO):
        def read(self, count):
            # Original framing must still accept a short, nonempty read.
            return super().read(min(count, 2))

    channel = _ConsoleFramedChannel(
        FragmentedInput(value),
        io.BytesIO(),
        client._loop,
        client._receive,
        client._channel_failed,
    )
    channel._read()
    # The real reader queues decoded messages and its failure in FIFO order.
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "retired"),
    [("Shutdown", False), ("Shutdown", True), ("Drain", True)],
)
async def test_shutdown_eof_requires_exact_positive_shutdown_retirement(
    operation, retired
):
    client = _ready_client()
    receipt = client.submit(operation, {})
    await _read_reply_bytes(
        client, _lifecycle_reply_bytes(client, receipt, retired=retired)
    )
    expected_eof = operation == "Shutdown" and retired
    assert client._shutdown_acknowledged is expected_eof
    assert (client._channel_error is None) is expected_eof
    if not expected_eof:
        assert isinstance(client._channel_error, EOFError)
    assert await receipt.retired is retired
    assert client._received_sequence == 3
    assert not client.physically_closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tail", "code"),
    [
        (b"\x00", "transport_lost"),
        (b"\x00\x00\x00\x02{", "transport_lost"),
        (b"\x00\x00\x00\x02{}", "transport_lost"),
        (b"\x00\x00\x00\x01{\x00\x00\x00\x00", "invalid_json"),
    ],
    ids=["partial-header", "partial-payload", "missing-terminator", "invalid-json"],
)
async def test_shutdown_does_not_hide_truncated_or_invalid_following_frame(tail, code):
    client = _ready_client()
    receipt = client.submit("Shutdown", {})
    await _read_reply_bytes(
        client, _lifecycle_reply_bytes(client, receipt, retired=True) + tail
    )
    assert client._shutdown_acknowledged
    assert type(client._channel_error) is ProtocolError
    assert str(client._channel_error) == code
    assert await receipt.retired is True
    assert client._received_sequence == 3
    assert not client.physically_closed


def test_uninitialized_process_cleanup_does_not_claim_admitted_work_retired():
    calls = []

    class Child:
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            calls.append("terminate")

        def wait(self, timeout):
            calls.append(("wait", timeout))
            self.returncode = 1

    client = ConsoleExecutionClient(None)
    client._process = Child()
    assert client._close_uninitialized_process() is True
    assert calls == ["terminate", ("wait", 2.0)]
    assert not client.physically_closed
    assert not client._disposed
    client._initialization_attempted = True
    assert client._close_uninitialized_process() is False
    assert calls == ["terminate", ("wait", 2.0)]


@pytest.mark.asyncio
async def test_positive_receipts_cannot_hide_failed_process_wait_after_forced_cleanup():
    import subprocess
    import time

    client = _ready_client()
    failure = subprocess.TimeoutExpired("original-service", 1.0)
    operations = []

    class PositiveLifecycleChannel:
        writer = reader = None
        sequence = 0

        def send(self, message):
            operations.append(message.operation)
            receipt = client._requests[message.identity.request_id]
            for kind, payload in (
                ("received", {"sequence": message.identity.sequence}),
                ("applied", {"applied": True}),
                (
                    "retired",
                    {"operation_id": message.identity.request_id, "retired": True},
                ),
            ):
                self.sequence += 1
                client._receive(_reply(client, receipt, self.sequence, kind, payload))

        def close_after_exit(self, deadline):
            return True

    class Child:
        returncode = None
        stdin, stdout, stderr = io.BytesIO(), io.BytesIO(), io.BytesIO()

        def wait(self, timeout):
            raise failure

        def poll(self):
            return self.returncode

    class Containment:
        def close(self):
            client._process.returncode = -15
            return True

    client._channel = PositiveLifecycleChannel()
    client._process = Child()
    client._tree = Containment()
    result = await client.shutdown(deadline=time.monotonic() + 5)
    assert operations == ["CloseIntake", "Drain", "Shutdown"]
    assert client._shutdown_acknowledged
    assert client._channel_error is failure
    assert client.physically_closed
    assert all(
        stream.closed
        for stream in (
            client._process.stdin,
            client._process.stdout,
            client._process.stderr,
        )
    )
    assert result is False and not client._disposed
