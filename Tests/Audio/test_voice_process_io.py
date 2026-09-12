"""Real private pipes, split writes, bounded receiver wakeups and EOF."""

import io
import asyncio
import os
import sys
import threading

import pytest

from tldw_chatbook.Audio.voice_process_io import PipeReader, PipeWriter, write_record
from tldw_chatbook.Audio.voice_process_protocol import (
    CreditWindow,
    Mailbox,
    ProtocolError,
    ReceiveStream,
    Record,
    StreamKey,
    encode_record,
    read_record,
)


def control(op="lease", sequence=1, **fields):
    return Record(
        dict(
            version=1,
            op=op,
            generation=1,
            request_id="a" * 32,
            sequence=sequence,
            **fields,
        ),
        b"",
    )


@pytest.mark.asyncio
async def test_first_draft_lane_scan_serializes_real_writer_release(monkeypatch):
    from tldw_chatbook.Audio.voice_process_lifetime import LifecyclePipe

    read_fd, peer_write = os.pipe()
    peer_read, write_fd = os.pipe()
    faults = []
    pipe = LifecyclePipe(
        read_fd,
        write_fd,
        generation=1,
        request_id="a" * 32,
        parent=False,
        consume=lambda record: None,
        fault=faults.append,
    )
    active, finish, release_checked = (threading.Event() for _ in range(3))
    original_write = pipe.writer._write
    original_release = pipe.output.release
    release_blocked = []
    paused = False

    def write(data):
        active.set()
        assert finish.wait(2)
        return original_write(data)

    def release(record):
        if record.header["op"] != "session_ready":
            return original_release(record)
        # The actual writer either completes its normal release before the
        # scan resumes, or observes the scan's lock and waits until it exits.
        # Never wait for writer completion while the main thread holds that lock.
        acquired = pipe.output._condition.acquire(blocking=False)
        release_blocked.append(not acquired)
        if acquired:
            try:
                original_release(record)
            finally:
                pipe.output._condition.release()
        release_checked.set()
        if not acquired:
            original_release(record)

    def trace(frame, event, arg):
        nonlocal paused
        if (
            not paused
            and event == "line"
            and frame.f_code.co_name == "<genexpr>"
            and frame.f_back.f_code is Mailbox._lane.__code__
            and "owned_lane" in frame.f_locals
        ):
            paused = True
            finish.set()
            assert release_checked.wait(1)
        return trace

    monkeypatch.setattr(pipe.writer, "_write", write)
    monkeypatch.setattr(pipe.output, "release", release)
    previous_trace = sys.gettrace()
    try:
        pipe.send("session_ready")
        assert active.wait(1)
        sys.settrace(trace)
        try:
            pipe.send("draft", turn_id="new", revision=1, epoch=1, payload=b"draft")
        finally:
            sys.settrace(previous_trace)
        assert paused and release_blocked == [True]
        assert pipe.writer.wait_written(2, 1)
        control_record = read_record(
            lambda count: os.read(peer_read, count), "child_to_parent"
        )
        draft = read_record(lambda count: os.read(peer_read, count), "child_to_parent")
        assert control_record.header["op"] == "session_ready"
        assert draft.header["draft_slot"] == 0 and draft.payload == b"draft"
        assert not faults
    finally:
        sys.settrace(previous_trace)
        finish.set()
        pipe.stop()
        os.close(peer_write)
        assert await pipe.join()
        os.close(peer_read)
        os.close(write_fd)


def test_held_published_frame_survives_admission_close_for_real_pipe_discard():
    read_fd, write_fd = os.pipe()
    key = StreamKey(1, "a" * 32, "provider", "turn", 1, 1)
    outbound = Mailbox("parent_to_child", generation=1, outbound=True)
    inbound = Mailbox("parent_to_child", generation=1)
    inbound.open_stream(key)
    inbound.fence_stream(key)
    window = CreditWindow(key)
    permit = window.try_reserve()
    item = Record(
        dict(
            version=1,
            op="provider_delta",
            generation=1,
            request_id="a" * 32,
            turn_id="turn",
            revision=1,
            epoch=1,
            sequence=1,
        ),
        b"held publication",
    )
    permit.publish(item)
    outbound.put(item)
    window.close()
    faults = []
    writer = PipeWriter(lambda data: os.write(write_fd, data), outbound, faults.append)
    writer.start()
    try:
        assert writer.wait_written(1, 1)
        delivered = read_record(
            lambda count: os.read(read_fd, count), "parent_to_child"
        )
        inbound.put(delivered)
        stream = ReceiveStream(key)
        delivery = stream.receive(inbound.take(), mailbox=inbound)
        assert delivery.discard_only and window.outstanding != (0, 0)
        window.acknowledge(key, *delivery.discard(key))
        assert window.outstanding == (0, 0) and faults == []
    finally:
        writer.close()
        writer.join(1)
        os.close(write_fd)
        os.close(read_fd)


def test_partial_and_interrupted_write_is_one_complete_frame():
    output = bytearray()
    calls = 0

    def write(data):
        nonlocal calls
        calls += 1
        if calls in (1, 4):
            raise InterruptedError("private-sentinel")
        part = bytes(data[:3])
        output.extend(part)
        return len(part)

    write_record(write, control(), "parent_to_child")
    assert (
        read_record(io.BytesIO(output).read, "parent_to_child").header["op"] == "lease"
    )


@pytest.mark.parametrize("cut", [0, 1, 7, 8, 10, -1])
def test_eof_distinguishes_empty_frame_boundary_from_truncated_frame(cut):
    data = encode_record(control(), "parent_to_child")[:cut]
    code = "voice_transport_eof" if cut == 0 else "voice_transport_truncated"
    with pytest.raises(ProtocolError, match=f"^{code}$"):
        read_record(io.BytesIO(data).read, "parent_to_child")


def test_reader_schedules_only_one_batch_and_prioritizes_received_fence():
    read_fd, write_fd = os.pipe()
    pending = []
    callback_ready = threading.Event()
    consumed = []
    faults = []
    box = Mailbox("parent_to_child", generation=1)

    def schedule(callback):
        pending.append(callback)
        callback_ready.set()

    def consume(item):
        consumed.append(item)
        box.release(item)

    reader = PipeReader(read_fd, box, schedule, consume, faults.append)
    try:
        reader.start()
        for n in range(1, 33):
            item = control("start", n, capture_live=True)
            write_record(lambda data: os.write(write_fd, data), item, "parent_to_child")
        write_record(
            lambda data: os.write(write_fd, data),
            control("close", 33, reason="teardown"),
            "parent_to_child",
        )
        assert callback_ready.wait(1)
        assert reader.wait_received(33, 1)
        assert len(pending) == 1
        assert consumed == []
        pending.pop()()
        assert consumed[0].header["op"] == "close"
        assert len(consumed) == 33
    finally:
        os.close(write_fd)
        reader.join(1)
        reader.close()
    assert not reader.alive
    assert all(str(error) == "voice_transport_eof" for error in faults)


def test_writer_finishes_partial_frame_before_pending_fence_and_new_data():
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    window = CreditWindow(StreamKey(1, "a" * 32, "control"))
    first = control("start", 1, capture_live=True)
    window.try_reserve().publish(first)
    box.put(first)
    started = threading.Event()
    resume = threading.Event()
    output = bytearray()
    faults = []

    def write(data):
        if not started.is_set():
            output.extend(data[:5])
            started.set()
            assert resume.wait(1)
            return 5
        output.extend(data)
        return len(data)

    writer = PipeWriter(write, box, faults.append)
    try:
        writer.start()
        assert started.wait(1)
        second = control("start", 2, capture_live=True)
        window.try_reserve().publish(second)
        box.put(second)
        box.put(control("close", 3, reason="teardown"))
        resume.set()
        assert writer.wait_written(3, 1)
    finally:
        resume.set()
        writer.close()
        writer.join(1)
    source = io.BytesIO(output)
    assert [
        read_record(source.read, "parent_to_child").header["op"] for _ in range(3)
    ] == ["start", "close", "start"]
    assert faults == []
    assert not writer.alive


def test_pipe_writer_failure_is_fixed_and_observed():
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    box.put(control())
    failures = []

    def broken(data):
        raise OSError("private-sentinel")

    writer = PipeWriter(broken, box, failures.append)
    writer.start()
    writer.join(1)
    assert not writer.alive
    assert [str(error) for error in failures] == ["voice_transport_failed"]


def test_eof_does_not_overtake_cleanup_received_during_a_batch():
    read_fd, write_fd = os.pipe()
    box = Mailbox("child_to_parent", generation=1)
    pending = []
    events = []
    failures = []

    def consume(item):
        events.append(item.header["op"])
        box.release(item)
        if item.header["op"] == "session_ready":
            write_record(
                lambda data: os.write(write_fd, data),
                control("closed", 2, outcome="clean"),
                "child_to_parent",
            )
            os.close(write_fd)
            assert reader.wait_received(2, 1)
            reader.join(1)

    def failed(error):
        events.append("eof")
        failures.append(error)

    reader = PipeReader(read_fd, box, pending.append, consume, failed)
    try:
        reader.start()
        write_record(
            lambda data: os.write(write_fd, data),
            control("session_ready"),
            "child_to_parent",
        )
        assert reader.wait_received(1, 1)
        pending.pop(0)()
        assert events == ["session_ready"]
        assert len(pending) == 1
        pending.pop(0)()
        assert events == ["session_ready", "closed", "eof"]
    finally:
        try:
            os.close(write_fd)
        except OSError:
            pass
        reader.join(1)
        reader.close()
    assert len(failures) == 1


def test_failed_scheduler_is_observable_without_scheduling_more_callbacks():
    read_fd, write_fd = os.pipe()
    box = Mailbox("parent_to_child", generation=1)

    def broken_schedule(callback):
        raise RuntimeError("private-sentinel")

    reader = PipeReader(
        read_fd, box, broken_schedule, lambda item: None, lambda error: None
    )
    try:
        reader.start()
        write_record(
            lambda data: os.write(write_fd, data), control(), "parent_to_child"
        )
        reader.join(1)
        assert not reader.alive
        assert str(reader.failure) == "voice_transport_failed"
        assert reader.failure.__traceback__ is None
    finally:
        os.close(write_fd)
        reader.close()
        reader.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lane,blocks,size", [("provider", 8, 4096), ("pcm", 8, 65280), ("control", 64, 0)]
)
async def test_private_pipe_backpressure_includes_writer_reader_and_held_consumer(
    lane, blocks, size
):
    read_fd, write_fd = os.pipe()
    outbound = Mailbox("parent_to_child", generation=1, outbound=True)
    inbound = Mailbox("parent_to_child", generation=1)
    turn = "" if lane == "control" else "voice-turn-1"
    key = StreamKey(
        1,
        "a" * 32,
        lane,
        turn,
        0 if lane == "control" else 1,
        0 if lane == "control" else 1,
        1 if lane == "pcm" else 0,
    )
    sender = CreditWindow(key)
    receiver = ReceiveStream(key)
    pending = []
    deliveries = []
    faults = []
    pulls = []

    def consume(item):
        if item.header["op"] == "close":
            inbound.release(item)
        else:
            deliveries.append(receiver.receive(item, mailbox=inbound))

    reader = PipeReader(read_fd, inbound, pending.append, consume, faults.append)
    writer = PipeWriter(
        lambda data: os.write(write_fd, data[:197]), outbound, faults.append
    )

    async def produce():
        for _ in range(blocks + 1):
            permit = await sender.reserve()
            pulls.append(1)
            if lane == "control":
                item = control("start", permit.sequence, capture_live=True)
            else:
                fields = dict(turn_id=turn, revision=1, epoch=1)
                if lane == "pcm":
                    fields["phrase_id"] = 1
                item = Record(
                    dict(
                        control(
                            "pcm" if lane == "pcm" else "provider_delta",
                            permit.sequence,
                            **fields,
                        ).header
                    ),
                    b"x" * size,
                )
            permit.publish(item)
            outbound.put(item)

    task = asyncio.create_task(produce())
    reader.start()
    writer.start()
    try:
        await asyncio.sleep(0)
        assert await asyncio.to_thread(reader.wait_received, blocks, 2)
        assert len(pulls) == blocks
        assert len(pending) == 1
        assert sender.outstanding[0] == blocks
        assert sender.outstanding[1] >= blocks * size
        pending.pop()()
        assert len(deliveries) == blocks
        assert inbound.count == blocks
        await asyncio.sleep(0)
        assert len(pulls) == blocks
        # Reserved close crosses while all ordinary/data consumer custody is full.
        outbound.put(control("close", 1, reason="stop"))
        assert await asyncio.to_thread(reader.wait_received, blocks + 1, 2)
        assert len(pending) == 1
        pending.pop()()
        ack = deliveries[0].consume()
        sender.acknowledge(key, *ack)
        await asyncio.wait_for(task, 2)
        assert len(pulls) == blocks + 1
        assert await asyncio.to_thread(reader.wait_received, blocks + 2, 2)
        pending.pop()()
        assert inbound.count == blocks
    finally:
        sender.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        writer.close()
        await asyncio.to_thread(writer.join, 2)
        os.close(write_fd)
        await asyncio.to_thread(reader.join, 2)
        while pending:
            pending.pop(0)()
        reader.close()
    assert not reader.alive and not writer.alive
    assert [str(error) for error in faults] == ["voice_transport_eof"]


def test_both_draft_slots_cross_private_pipe_and_retire_successive_revisions():
    data_read, data_write = os.pipe()
    credit_read, credit_write = os.pipe()
    outbound = Mailbox("child_to_parent", generation=1, outbound=True)
    inbound = Mailbox("child_to_parent", generation=1)
    keys = [
        StreamKey(1, f"{n + 1:032x}", f"draft_{n}", f"voice-turn-{n + 1}")
        for n in range(2)
    ]
    windows = [CreditWindow(key) for key in keys]
    streams = [ReceiveStream(key) for key in keys]
    pending, received, failures = [], [], []
    for key in keys:
        outbound.retain_turn(key.turn_id)
        inbound.retain_turn(key.turn_id)
    reader = PipeReader(
        data_read, inbound, pending.append, received.append, failures.append
    )
    writer = PipeWriter(
        lambda data: os.write(data_write, data[:97]), outbound, failures.append
    )
    reader.start()
    writer.start()
    try:
        for revision in (1, 2):
            for index, key in enumerate(keys):
                permit = windows[index].try_reserve()
                item = Record(
                    dict(
                        control(
                            "draft",
                            permit.sequence,
                            turn_id=key.turn_id,
                            draft_slot=index,
                            revision=revision,
                            epoch=revision,
                        ).header,
                        request_id=key.request_id,
                    ),
                    b"draft",
                )
                permit.publish(item)
                outbound.put(item)
            assert reader.wait_received(revision * 2, 1)
            assert writer.wait_written(revision * 2, 1)
            pending.pop(0)()
            for index, key in enumerate(keys):
                item = received.pop(0)
                assert item.header["turn_id"] == key.turn_id
                delivery = streams[index].receive(item, mailbox=inbound)
                assert windows[index].outstanding == (1, 5)
                ack = delivery.consume() if revision == 1 else delivery.discard(key)
                assert delivery.record is None
                credit = control(
                    "credit",
                    revision,
                    lane=key.lane,
                    stream_id=key.request_id,
                    stream_turn=key.turn_id,
                    stream_revision=key.revision,
                    stream_epoch=key.epoch,
                    stream_phrase=0,
                    ack_sequence=ack[0],
                    ack_bytes=ack[1],
                )
                write_record(
                    lambda data: os.write(credit_write, data), credit, "parent_to_child"
                )
                windows[index].accept_credit(
                    read_record(
                        lambda count: os.read(credit_read, count), "parent_to_child"
                    )
                )
                assert windows[index].outstanding == (0, 0)
            assert inbound.count == 0
    finally:
        writer.close()
        writer.join(1)
        os.close(data_write)
        reader.join(1)
        while pending:
            pending.pop(0)()
        reader.close()
        os.close(credit_write)
        os.close(credit_read)
    assert not reader.alive and not writer.alive
    assert [str(error) for error in failures] == ["voice_transport_eof"]


@pytest.mark.parametrize("pause_call", [1, 2])
def test_prefix_or_intermediate_header_cannot_issue_a_future_consumption_receipt(
    pause_call,
):
    read_fd, write_fd = os.pipe()
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    window = CreditWindow(key, blocks=1)
    item = Record(
        dict(
            control("provider_delta", turn_id=key.turn_id, revision=1, epoch=1).header
        ),
        b"x" * 4096,
    )
    window.try_reserve().publish(item)
    box.put(item)
    active, resume = threading.Event(), threading.Event()
    failures = []
    calls = 0

    def write(data):
        nonlocal calls
        calls += 1
        if calls == pause_call:
            active.set()
            assert resume.wait(1)
        return os.write(write_fd, data)

    writer = PipeWriter(write, box, failures.append)
    writer.start()
    try:
        assert active.wait(1)
        if pause_call == 1:
            os.set_blocking(read_fd, False)
            with pytest.raises(BlockingIOError):
                os.read(read_fd, 1)
            os.set_blocking(read_fd, True)
        with pytest.raises(ProtocolError, match="voice_protocol_invalid"):
            window.acknowledge(key, 1, 4096)
        assert window.outstanding == (1, 4096)
        assert window.try_reserve() is None
        resume.set()
        assert writer.wait_written(1, 1)
        # Nobody has read the completed frame: the rejected receipt must not
        # silently become valid later and admit another producer pull.
        assert box.count == 0
        assert window.outstanding == (1, 4096)
        assert window.try_reserve() is None
        received = read_record(lambda count: os.read(read_fd, count), "parent_to_child")
        assert received.payload == b"x" * 4096
        receipt = ReceiveStream(key).receive(received).consume()
        window.acknowledge(key, *receipt)
        assert window.outstanding == (0, 0)
    finally:
        resume.set()
        writer.close()
        writer.join(1)
        os.close(write_fd)
        os.close(read_fd)
    assert not writer.alive
    assert failures == []


@pytest.mark.parametrize("result", ["short", "interrupted"])
def test_incomplete_final_write_invalidates_its_impossible_pending_receipt(result):
    read_fd, write_fd = os.pipe()
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    window = CreditWindow(key, blocks=1)
    item = Record(
        dict(
            control("provider_delta", turn_id=key.turn_id, revision=1, epoch=1).header
        ),
        b"x" * 4096,
    )
    window.try_reserve().publish(item)
    box.put(item)
    final_call, resume = threading.Event(), threading.Event()
    calls = 0
    failures = []

    def write(data):
        nonlocal calls
        calls += 1
        if calls == 3:
            final_call.set()
            assert resume.wait(1)
            if result == "interrupted":
                raise InterruptedError("private-sentinel")
            return os.write(write_fd, data[:100])
        return os.write(write_fd, data)

    writer = PipeWriter(write, box, failures.append)
    writer.start()
    try:
        assert final_call.wait(1)
        window.acknowledge(key, 1, 4096)
        assert window.outstanding == (1, 4096)
        resume.set()
        writer.join(1)
        assert not writer.alive
        assert [str(error) for error in failures] == ["voice_protocol_invalid"]
        assert window.outstanding == (1, 4096)
        with pytest.raises(ProtocolError, match="voice_transport_closed"):
            window.try_reserve()
        with pytest.raises(ProtocolError):
            window.acknowledge(key, 1, 4096)
    finally:
        resume.set()
        writer.close()
        writer.join(1)
        os.close(write_fd)
        os.close(read_fd)


@pytest.mark.parametrize("close_owner", [False, True])
@pytest.mark.parametrize("lane", ["provider", "control", "tts_closed"])
def test_real_consumer_receipt_racing_final_write_returns_only_after_writer_exit(
    close_owner,
    lane,
):
    read_fd, write_fd = os.pipe()
    outbound = Mailbox("parent_to_child", generation=1, outbound=True)
    inbound = Mailbox("parent_to_child", generation=1)
    key = (
        StreamKey(1, "a" * 32, lane)
        if lane != "provider"
        else StreamKey(1, "a" * 32, lane, "voice-turn-1", 1, 1)
    )
    window, stream = CreditWindow(key, blocks=1), ReceiveStream(key, blocks=1)
    if lane == "tts_closed":
        item = control(
            "tts_closed",
            turn_id="voice-turn-1",
            revision=1,
            epoch=1,
            phrase_id=1,
            outcome="clean",
            last_sequence=0,
        )
    elif lane == "control":
        item = control("start", capture_live=True)
    else:
        item = Record(
            dict(
                control(
                    "provider_delta", turn_id=key.turn_id, revision=1, epoch=1
                ).header
            ),
            b"x" * 4096,
        )
    window.try_reserve().publish(item)
    held_bytes = (
        4096 if lane == "provider" else len(encode_record(item, "parent_to_child"))
    )
    outbound.put(item)
    pending, failures = [], []
    wrote_body, return_from_write = threading.Event(), threading.Event()
    calls = 0

    def write(data):
        nonlocal calls
        calls += 1
        count = os.write(write_fd, data)
        if calls == (3 if lane == "provider" else 2):
            wrote_body.set()
            assert return_from_write.wait(1)
        return count

    def consume(received):
        receipt = stream.receive(received, mailbox=inbound).consume()
        window.acknowledge(key, *receipt)

    reader = PipeReader(read_fd, inbound, pending.append, consume, failures.append)
    writer = PipeWriter(write, outbound, failures.append)
    reader.start()
    writer.start()
    try:
        assert wrote_body.wait(1)
        assert reader.wait_received(1, 1)
        pending.pop(0)()
        assert inbound.count == 0
        assert window.outstanding == (1, held_bytes)
        assert window.try_reserve() is None
        if close_owner:
            window.close()
        return_from_write.set()
        assert writer.wait_written(1, 1)
        assert window.outstanding == (0, 0)
        if close_owner:
            with pytest.raises(ProtocolError, match="voice_transport_closed"):
                window.try_reserve()
    finally:
        return_from_write.set()
        writer.close()
        writer.join(1)
        os.close(write_fd)
        reader.join(1)
        while pending:
            pending.pop(0)()
        reader.close()
    assert not writer.alive and not reader.alive
    assert [str(error) for error in failures] == ["voice_transport_eof"]


def test_writer_failure_preserves_late_receipt_for_an_earlier_completed_frame():
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    window = CreditWindow(key, blocks=2)
    for sequence in (1, 2):
        item = Record(
            dict(
                control(
                    "provider_delta", sequence, turn_id=key.turn_id, revision=1, epoch=1
                ).header
            ),
            b"x" * 4096,
        )
        window.try_reserve().publish(item)
        box.put(item)
    calls = 0
    failures = []

    def write(data):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("private-sentinel")
        return len(data)

    writer = PipeWriter(write, box, failures.append)
    writer.start()
    try:
        writer.join(1)
        assert not writer.alive
        assert [str(error) for error in failures] == ["voice_transport_failed"]
        window.acknowledge(key, 1, 4096)
        assert window.outstanding == (1, 4096)
        with pytest.raises(ProtocolError):
            window.acknowledge(key, 2, 8192)
    finally:
        writer.close()
        writer.join(1)


@pytest.mark.parametrize("lane", ["provider", "pcm"])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("local_fence", [False, True])
def test_cancel_drains_full_known_stream_by_discard_and_wire_credit(
    lane, partial, local_fence
):
    data_read, data_write = os.pipe()
    credit_read, credit_write = os.pipe()
    outbound = Mailbox("parent_to_child", generation=1, outbound=True)
    inbound = Mailbox("parent_to_child", generation=1)
    key = StreamKey(1, "a" * 32, lane, "voice-turn-1", 1, 1, int(lane == "pcm"))
    window, stream = CreditWindow(key), ReceiveStream(key)
    inbound.open_stream(key)
    size = 65280 if lane == "pcm" else 4096
    pending, received, faults = [], [], []
    active, resume = threading.Event(), threading.Event()

    def make_item(stream_key, sequence):
        fields = dict(
            turn_id=stream_key.turn_id,
            revision=stream_key.revision,
            epoch=stream_key.epoch,
        )
        if lane == "pcm":
            fields["phrase_id"] = stream_key.phrase_id
        return Record(
            dict(
                control(
                    "pcm" if lane == "pcm" else "provider_delta", sequence, **fields
                ).header
            ),
            b"x" * size,
        )

    for _ in range(8):
        permit = window.try_reserve()
        item = make_item(key, permit.sequence)
        permit.publish(item)
        outbound.put(item)
    assert window.try_reserve() is None

    def write(data):
        if partial and not active.is_set():
            count = os.write(data_write, data[:1])
            active.set()
            assert resume.wait(2)
            return count
        return os.write(data_write, data)

    reader = PipeReader(
        data_read, inbound, pending.append, received.append, faults.append
    )
    writer = PipeWriter(write, outbound, faults.append)
    writer_started = False
    reader.start()
    try:
        if partial:
            writer.start()
            writer_started = True
            assert active.wait(1)
        cancel = control(
            "cancel", turn_id=key.turn_id, revision=1, epoch=1, reason="stop"
        )
        if local_fence:
            # Speech detection fences child consumption immediately. Its cancel
            # travels in the opposite direction from the buffered data.
            inbound.fence_stream(key)
            write_record(
                lambda data: os.write(credit_write, data), cancel, "child_to_parent"
            )
            parent_control = Mailbox("child_to_parent", generation=1)
            parent_control.put(
                read_record(
                    lambda count: os.read(credit_read, count), "child_to_parent"
                )
            )
            parent_control.release(parent_control.take())
        else:
            outbound.put(cancel)
        if not partial:
            writer.start()
            writer_started = True
        resume.set()
        count = 8 if local_fence else 9
        assert reader.wait_received(count, 2)
        assert writer.wait_written(count, 2)
        assert len(pending) == 1
        pending.pop(0)()
        if not local_fence:
            assert received.pop(0).header["op"] == "cancel"
        assert window.outstanding == (8, size * 8)
        assert window.try_reserve() is None
        for item in received:
            delivery = stream.receive(item, mailbox=inbound)
            with pytest.raises(ProtocolError):
                delivery.consume()
            ack = delivery.discard(key)
        received.clear()
        credit = control(
            "credit",
            lane=lane,
            stream_id=key.request_id,
            stream_turn=key.turn_id,
            stream_revision=1,
            stream_epoch=1,
            stream_phrase=key.phrase_id,
            ack_sequence=ack[0],
            ack_bytes=ack[1],
        )
        write_record(
            lambda data: os.write(credit_write, data), credit, "child_to_parent"
        )
        window.accept_credit(
            read_record(lambda count: os.read(credit_read, count), "child_to_parent")
        )
        assert window.outstanding == (0, 0)
        newer = StreamKey(1, "a" * 32, lane, "voice-turn-1", 2, 2, key.phrase_id)
        outbound.open_stream(newer)
        inbound.open_stream(newer)
        # Once retired, even the next old sequence is not an admissible discard.
        with pytest.raises(ProtocolError):
            inbound.check_admission(make_item(key, 9).header, size)
        replacement = CreditWindow(newer)
        item = make_item(newer, 1)
        replacement.try_reserve().publish(item)
        outbound.put(item)
        assert reader.wait_received(count + 1, 2)
        assert writer.wait_written(count + 1, 2)
        pending.pop(0)()
        assert ReceiveStream(newer).receive(
            received.pop(0), mailbox=inbound
        ).consume() == (1, size)
    finally:
        resume.set()
        writer.close()
        if writer_started:
            writer.join(2)
        os.close(data_write)
        reader.join(2)
        while pending:
            pending.pop(0)()
        reader.close()
        os.close(credit_write)
        os.close(credit_read)
    assert not writer.alive and not reader.alive
    assert [str(error) for error in faults] == ["voice_transport_eof"]


def test_reader_close_retains_inflight_fd_until_thread_exit(monkeypatch):
    read_fd, write_fd = os.pipe()
    original_read = os.read
    first_byte, resume = threading.Event(), threading.Event()
    calls = []
    replacement = None

    def split_read(fd, count):
        calls.append((fd, count))
        if len(calls) == 1:
            result = original_read(fd, 1)
            first_byte.set()
            assert resume.wait(2)
            return result
        return original_read(fd, count)

    monkeypatch.setattr(os, "read", split_read)
    reader = PipeReader(
        read_fd,
        Mailbox("parent_to_child", generation=1),
        lambda callback: None,
        lambda item: None,
        lambda error: None,
    )
    reader.start()
    os.write(write_fd, b"x")
    try:
        assert first_byte.wait(1)
        reader.close()
        reader.close()
        replacement = os.pipe()
        os.write(replacement[1], b"private")
        # The stopped thread still owns its original descriptor while a read is
        # active. A newly opened endpoint must never acquire that number yet.
        assert replacement[0] != read_fd
        resume.set()
    finally:
        resume.set()
        os.close(write_fd)
        if replacement is not None:
            os.close(replacement[1])
        reader.join(2)
        reader.close()
        if replacement is not None:
            try:
                os.set_blocking(replacement[0], False)
                assert original_read(replacement[0], 7) == b"private"
            finally:
                os.close(replacement[0])
    assert not reader.alive
    with pytest.raises(OSError):
        os.fstat(read_fd)


def test_reader_close_before_start_and_after_eof_never_double_closes_reused_fd():
    for start in (False, True):
        read_fd, write_fd = os.pipe()
        reader = PipeReader(
            read_fd,
            Mailbox("parent_to_child", generation=1),
            lambda callback: None,
            lambda item: None,
            lambda error: None,
        )
        if start:
            reader.start()
        os.close(write_fd)
        if start:
            reader.join(1)
            assert not reader.alive
        reader.close()
        reader.join(1)
        replacement_read, replacement_write = os.pipe()
        try:
            assert replacement_read == read_fd
            reader.close()
            os.fstat(replacement_read)
            if not start:
                with pytest.raises(ProtocolError, match="voice_transport_closed"):
                    reader.start()
        finally:
            os.close(replacement_read)
            os.close(replacement_write)


def test_reader_owns_duplicate_independently_of_file_wrapper():
    read_fd, write_fd = os.pipe()
    wrapper = os.fdopen(read_fd, "rb", buffering=0)
    transferred = os.dup(wrapper.fileno())
    pending, received = [], []
    reader = PipeReader(
        transferred,
        Mailbox("parent_to_child", generation=1),
        pending.append,
        received.append,
        lambda error: None,
    )
    reader.start()
    try:
        wrapper.close()  # The original is not the reader's endpoint.
        write_record(
            lambda data: os.write(write_fd, data), control(), "parent_to_child"
        )
        assert reader.wait_received(1, 1)
        pending.pop(0)()
        assert received[0].header["op"] == "lease"
    finally:
        os.close(write_fd)
        reader.join(1)
        reader.close()
        wrapper.close()
    assert not reader.alive
    with pytest.raises(OSError):
        os.fstat(transferred)
