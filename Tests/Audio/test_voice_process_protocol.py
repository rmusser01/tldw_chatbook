"""Closed wire and ownership regressions; synthetic content only."""

import asyncio
import io
import json
import struct
import subprocess
import sys
import threading

import pytest

from tldw_chatbook.Audio.voice_process_io import PipeWriter
from tldw_chatbook.Audio.voice_process_protocol import (
    CreditWindow,
    Mailbox,
    ProtocolError,
    ReceiveStream,
    Record,
    StreamKey,
    checked_epoch,
    checked_pcm,
    encode_record,
    read_record,
)


def test_bootstrap_omitted_closed_stt_options_preserve_none_defaults():
    from Tests.Chat.test_console_voice_process import bootstrap, console

    original = bootstrap(console())
    header = dict(original.header)
    for key in ("stt_device", "stt_compute_type", "stt_precision"):
        header.pop(key)
    decoded = read_record(
        io.BytesIO(encode_record(Record(header), "parent_to_child")).read,
        "parent_to_child",
    )
    assert all(
        decoded.header.get(key) is None
        for key in ("stt_device", "stt_compute_type", "stt_precision")
    )


@pytest.mark.parametrize("action", ["request", "revoke"])
def test_draft_recovery_is_closed_content_free_child_control(action):
    item = record("draft_recovery", b"", action=action, recovery_id=1)
    wire = encode_record(item, "child_to_parent")
    assert read_record(io.BytesIO(wire).read, "child_to_parent") == item
    with pytest.raises(ProtocolError):
        encode_record(item, "parent_to_child")


@pytest.mark.parametrize(
    "fields,payload",
    [
        ({"recovery_id": 0}, b""),
        ({"recovery_id": True}, b""),
        ({"recovery_id": 2**63}, b""),
        ({"action": "promote"}, b""),
        ({"category": "provider_unavailable"}, b""),
        ({}, b"private transcript"),
    ],
)
def test_draft_recovery_rejects_untyped_authority_and_payload(fields, payload):
    with pytest.raises(ProtocolError):
        encode_record(
            record(
                "draft_recovery",
                payload,
                **(dict(action="request", recovery_id=1) | fields),
            ),
            "child_to_parent",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("stt_device", "cuda:0"),
        ("stt_compute_type", {"callable": "private"}),
        ("stt_precision", True),
        ("stt_options", {}),
    ],
)
def test_invalid_stt_options_rejected_by_parser_before_child_factory(field, value):
    from Tests.Chat.test_console_voice_process import bootstrap, console

    header = dict(bootstrap(console()).header)
    header[field] = value
    body = json.dumps(header).encode()
    wire = struct.pack("!II", len(body), 0) + body
    with pytest.raises(ProtocolError):
        read_record(io.BytesIO(wire).read, "parent_to_child")


def record(op="provider_delta", payload=b"hello", sequence=1, **fields):
    header = dict(
        version=1, op=op, generation=1, request_id="a" * 32, sequence=sequence
    )
    if op not in {"lease", "close", "closed", "fault", "diagnostic", "credit"}:
        header.update(turn_id="voice-turn-1", revision=1, epoch=1)
    if op in {"pcm", "pcm_end", "tts_closed", "synthesize"}:
        header["phrase_id"] = 1
    if op == "draft":
        header["draft_slot"] = 0
    if op in {"prepared", "start_attempt", "terminal_propose"}:
        header["request_handle"] = "c" * 32
    if op in {"cleanup", "tts_closed"}:
        header["last_sequence"] = 0
    if op == "terminal_propose":
        header.update(
            kind="promote", boundary_id=None, answer_sha256="d" * 64, answer_chars=0
        )
    if op == "credit":
        header.update(
            stream_turn="voice-turn-1",
            stream_revision=1,
            stream_epoch=1,
            stream_phrase=0,
        )
    header.update(fields)
    return Record(header, payload)


def wire(header, payload=b""):
    raw = json.dumps(header, ensure_ascii=True).encode()
    return struct.pack("!II", len(raw), len(payload)) + raw + payload


def flush_and_close(outbound, count):
    failures = []
    writer = PipeWriter(lambda part: len(part), outbound, failures.append)
    writer.start()
    try:
        assert writer.wait_written(count, 1)
    finally:
        writer.close()
        writer.join(1)
    assert not writer.alive
    assert failures == []


class ShortReader:
    def __init__(self, data, step=3):
        self.data = io.BytesIO(data)
        self.step = step
        self.requests = []

    def __call__(self, count):
        self.requests.append(count)
        return self.data.read(min(count, self.step))


def test_split_reads_preserve_wire_bytes_and_turn_identifiers():
    item = record(payload="héllo".encode())
    encoded = encode_record(item, "parent_to_child")
    assert struct.unpack("!II", encoded[:8])[1] == 6
    decoded = read_record(ShortReader(encoded), "parent_to_child")
    assert decoded.header["turn_id"] == "voice-turn-1"
    assert decoded.payload == "héllo".encode()


@pytest.mark.parametrize("outcome", ["clean", "force_closed", "detached", "failed"])
def test_resource_close_receipt_has_one_reserved_noncoalescing_slot(outcome):
    header = dict(
        version=1,
        op="resources_closed",
        generation=1,
        request_id="a" * 32,
        sequence=1,
        outcome=outcome,
    )
    receipt = Record(header)
    encoded = encode_record(receipt, "child_to_parent")
    assert (
        read_record(io.BytesIO(encoded).read, "child_to_parent").header["outcome"]
        == outcome
    )
    with pytest.raises(ProtocolError):
        encode_record(receipt, "parent_to_child")
    box = Mailbox("child_to_parent", generation=1)
    box.put(receipt)
    assert box.take() is receipt
    with pytest.raises(ProtocolError, match="voice_capacity_exceeded"):
        box.put(Record(dict(header, sequence=2)))
    box.release(receipt)
    with pytest.raises(ProtocolError):
        box.put(receipt)


@pytest.mark.parametrize("value", [None, True, 1, "unknown", {"outcome": "clean"}])
def test_resource_close_receipt_rejects_unknown_wire_dispositions(value):
    header = dict(
        version=1,
        op="resources_closed",
        generation=1,
        request_id="a" * 32,
        sequence=1,
        outcome=value,
    )
    with pytest.raises(ProtocolError):
        read_record(io.BytesIO(wire(header)).read, "child_to_parent")


@pytest.mark.parametrize("missing", [1, 4, 5])
def test_eof_with_incomplete_payload_is_truncated_not_clean_boundary(missing):
    encoded = encode_record(record(payload=b"hello"), "parent_to_child")
    with pytest.raises(ProtocolError, match="^voice_transport_truncated$"):
        read_record(ShortReader(encoded[:-missing]), "parent_to_child")


@pytest.mark.parametrize("value", [True, False, -1, 2**63, 1.0, "1", None])
def test_integer_boundary_does_not_accept_bool_or_unbounded_values(value):
    with pytest.raises(ProtocolError, match="^voice_protocol_invalid$"):
        checked_epoch(value)


@pytest.mark.parametrize(
    "payload", [b"", b"x", b"x" * 65536, b"x" * 66240, bytearray(960)]
)
def test_pcm_only_accepts_nonempty_whole_frames(payload):
    with pytest.raises(ProtocolError):
        checked_pcm(payload)


def test_pcm_maximum_whole_frame_block_is_accepted():
    assert len(checked_pcm(b"x" * 65280)) == 65280


@pytest.mark.parametrize(
    "change",
    [
        {"op": "run_python"},
        {"version": True},
        {"generation": False},
        {"sequence": -1},
        {"epoch": 2**63},
        {"turn_id": "../secret"},
        {"request_id": "voice-turn-1"},
        {"extra": "private-sentinel"},
        {"turn_id": "\ud800"},
        {"revision": float("nan")},
    ],
)
def test_unknown_or_malformed_metadata_is_categorical(change):
    header = dict(record().header)
    header.update(change)
    with pytest.raises(ProtocolError) as caught:
        read_record(io.BytesIO(wire(header, b"hello")).read, "parent_to_child")
    assert str(caught.value) == "voice_protocol_invalid"
    assert caught.value.__cause__ is None


def test_direction_is_closed_and_header_cannot_be_mutated_after_validation():
    item = record()
    with pytest.raises(ProtocolError):
        encode_record(item, "child_to_parent")
    with pytest.raises(TypeError):
        item.header["op"] = "close"


@pytest.mark.parametrize("raw", [b"{", b'{"op":"lease","op":"close"}', b"\xff", b"[]"])
def test_bad_json_and_duplicate_keys_are_rejected(raw):
    reader = ShortReader(struct.pack("!II", len(raw), 0) + raw)
    with pytest.raises(ProtocolError, match="^voice_protocol_invalid$"):
        read_record(reader, "parent_to_child")


@pytest.mark.parametrize(
    "op,cap",
    [("provider_delta", 4096), ("pcm", 65536), ("prepare", 262144), ("lease", 0)],
)
def test_operation_payload_limit_is_checked_before_body_read(op, cap):
    item = record(op, b"")
    raw = json.dumps(dict(item.header)).encode()
    reader = io.BytesIO(struct.pack("!II", len(raw), cap + 1) + raw)
    with pytest.raises(ProtocolError):
        read_record(
            reader.read, "child_to_parent" if op == "prepare" else "parent_to_child"
        )
    assert reader.tell() == (8 if cap == 262144 else 8 + len(raw))


@pytest.mark.parametrize(
    "prefix", [struct.pack("!II", 4097, 0), struct.pack("!II", 1, 262145)]
)
def test_global_limits_are_checked_before_header_allocation(prefix):
    source = io.BytesIO(prefix + b"private-sentinel")
    with pytest.raises(ProtocolError):
        read_record(source.read, "parent_to_child")
    assert source.tell() == 8


@pytest.mark.parametrize("payload", [b"\xff", b"\xed\xa0\x80"])
def test_invalid_utf8_payload_is_rejected(payload):
    with pytest.raises(ProtocolError):
        read_record(
            io.BytesIO(wire(dict(record().header), payload)).read, "parent_to_child"
        )


def test_transcript_character_limit_is_independent_of_byte_limit():
    with pytest.raises(ProtocolError):
        encode_record(record("prepare", b"a" * 65537), "child_to_parent")
    encoded = encode_record(record("prepare", "😀".encode() * 65536), "child_to_parent")
    assert (
        read_record(io.BytesIO(encoded).read, "child_to_parent").payload
        == "😀".encode() * 65536
    )


def test_diagnostics_have_fixed_categories_and_no_free_text():
    item = record("diagnostic", b"", code="capture_progress", value=1)
    assert encode_record(item, "child_to_parent")
    for fields in (
        {"code": "private-sentinel", "value": 1},
        {"code": "capture_progress", "value": "secret"},
    ):
        with pytest.raises(ProtocolError) as caught:
            encode_record(record("diagnostic", b"", **fields), "child_to_parent")
        assert "secret" not in str(caught.value)
    with pytest.raises(ProtocolError):
        encode_record(record("fault", b"", code="private-sentinel"), "child_to_parent")


def test_mailbox_all_lanes_stay_bounded_with_reserved_revocations():
    box = Mailbox("parent_to_child", generation=1)
    for n in range(64):
        box.put(
            record(
                "prepared",
                b"",
                sequence=n + 1,
                context_handle="b" * 32,
                decision="provisional",
            )
        )
    for n in range(8):
        box.put(record(sequence=n + 1, payload=b"x" * 4096))
        box.put(record("pcm", b"x" * 65280, sequence=n + 1))
    box.put(record("cancel", b"", reason="stop", sequence=1))
    box.put(record("lease", b"", sequence=1))
    box.put(record("fault", b"", code="transport_failed"))
    box.put(
        record(
            "credit",
            b"",
            lane="provider",
            stream_id="a" * 32,
            ack_sequence=1,
            ack_bytes=4096,
        )
    )
    box.put(record("cleanup", b"", outcome="clean"))
    assert box.take().header["op"] == "fault"
    assert box.take().header["op"] == "cancel"
    assert box.count == 85
    for op, payload in (
        ("provider_delta", b"x"),
        ("pcm", b"x" * 960),
        ("prepared", b""),
    ):
        with pytest.raises(ProtocolError, match="voice_capacity_exceeded"):
            box.put(
                record(
                    op,
                    payload,
                    sequence=65,
                    **(
                        {"context_handle": "b" * 32, "decision": "provisional"}
                        if op == "prepared"
                        else {}
                    ),
                )
            )


def test_preparation_terminal_draft_preview_and_diagnostic_budgets():
    box = Mailbox("child_to_parent", generation=1)
    for n in range(2):
        box.retain_turn(f"voice-turn-{n + 1}")
        box.put(
            record("prepare", b"x" * 65536, request_id=f"{n + 1:032x}", sequence=n + 1)
        )
        box.put(record("draft", b"first", turn_id=f"voice-turn-{n + 1}", draft_slot=n))
    with pytest.raises(ProtocolError):
        box.retain_turn("voice-turn-3")
    with pytest.raises(ProtocolError):
        box.put(record("prepare", b"x", request_id="c" * 32))
    box.put(record("terminal_propose", b"", context_handle="b" * 32, last_sequence=0))
    with pytest.raises(ProtocolError, match="voice_capacity_exceeded"):
        box.put(
            record(
                "terminal_propose",
                b"",
                context_handle="b" * 32,
                last_sequence=0,
                sequence=2,
            )
        )
    box.put(record("preview", b"old"))
    box.put(record("preview", b"new", revision=2, sequence=2))
    box.put(record("draft", b"new draft", revision=2, sequence=2))
    for n in range(100):
        box.put(
            record("diagnostic", b"", sequence=n + 1, code="capture_progress", value=n)
        )
    items = []
    while (item := box.take()) is not None:
        items.append(item)
    assert [r.payload for r in items if r.header["op"] == "preview"] == [b"new"]
    assert [r.payload for r in items if r.header["op"] == "draft"] == [
        b"new draft",
        b"first",
    ]
    assert sum(r.header["op"] == "diagnostic" for r in items) == 64


def test_receiver_requires_consumption_not_receipt_and_fences_identity():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    stream = ReceiveStream(key, blocks=2)
    first = stream.receive(record(sequence=1))
    second = stream.receive(record(sequence=2))
    assert stream.ack == (0, 0)
    assert not stream.end(2)
    assert not stream.complete
    assert second.consume() == (0, 0)
    assert first.consume() == (2, 10)
    assert stream.complete
    with pytest.raises(ProtocolError):
        first.consume()
    for bad in (
        record(sequence=2),
        record(sequence=3, generation=2),
        record(sequence=3, epoch=0),
    ):
        with pytest.raises(ProtocolError):
            stream.receive(bad)


def test_pcm_cleanup_is_independent_of_end_and_matching_discard():
    key = StreamKey(1, "a" * 32, "pcm", "voice-turn-1", 1, 1, 1)
    stream = ReceiveStream(key)
    item = stream.receive(record("pcm", b"x" * 960))
    stream.closed("clean")
    assert not stream.complete
    stream.end(1)
    assert not stream.complete
    with pytest.raises(ProtocolError):
        item.discard(StreamKey(1, "b" * 32, "pcm", "voice-turn-1", 1, 1, 1))
    assert item.discard(key) == (1, 960)
    assert stream.complete


@pytest.mark.asyncio
async def test_credit_is_reserved_before_producer_pull_and_only_consumption_returns_it():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    window = CreditWindow(key, blocks=2)
    receiver = ReceiveStream(key, blocks=2)
    outbound = Mailbox("parent_to_child", generation=1, outbound=True)
    pulled = []
    deliveries = []
    reached = asyncio.Event()
    failures = []
    writer = PipeWriter(lambda part: len(part), outbound, failures.append)
    writer.start()

    async def produce():
        for _ in range(3):
            permit = await window.reserve()
            pulled.append(1)
            item = record(sequence=permit.sequence, payload=b"x" * 4096)
            permit.publish(item)
            outbound.put(item)
            assert await asyncio.to_thread(writer.wait_written, len(pulled), 1)
            deliveries.append(receiver.receive(item))
            if len(pulled) == 2:
                reached.set()

    task = asyncio.create_task(produce())
    try:
        await asyncio.wait_for(reached.wait(), 1)
        await asyncio.sleep(0)
        assert len(pulled) == 2
        assert receiver.ack == (0, 0)
        ack = deliveries[0].consume()
        window.acknowledge(key, *ack)
        await asyncio.wait_for(task, 1)
        assert len(pulled) == 3
        assert window.outstanding == (2, 8192)
        with pytest.raises(ProtocolError):
            window.acknowledge(key, *ack)
        with pytest.raises(ProtocolError):
            window.acknowledge(key, 3, 999999)
        with pytest.raises(ProtocolError):
            window.acknowledge(
                StreamKey(2, "a" * 32, "provider", "voice-turn-1", 1, 1), 3, 12288
            )
    finally:
        window.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        writer.close()
        await asyncio.to_thread(writer.join, 1)
    assert not writer.alive
    assert failures == []


@pytest.mark.asyncio
async def test_close_wakes_full_credit_waiter_with_failure_not_receipt():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    window = CreditWindow(key, blocks=1)
    permit = await window.reserve()
    permit.publish(record(payload=b"x"))
    waiter = asyncio.create_task(window.reserve())
    await asyncio.sleep(0)
    window.close()
    with pytest.raises(ProtocolError, match="voice_transport_closed"):
        await asyncio.wait_for(waiter, 1)
    assert window.outstanding == (1, 1)


def test_closed_directions_follow_parent_provider_authority():
    start = record("start_attempt", b"", context_handle="b" * 32)
    claim = record("terminal_claim", b"", context_handle="b" * 32, last_sequence=0)
    assert encode_record(start, "child_to_parent")
    assert encode_record(claim, "parent_to_child")
    with pytest.raises(ProtocolError):
        encode_record(start, "parent_to_child")
    with pytest.raises(ProtocolError):
        encode_record(claim, "child_to_parent")


def test_mailbox_holds_receiver_custody_after_dequeue():
    box = Mailbox("parent_to_child", generation=1)
    for seq in range(1, 9):
        box.put(record(sequence=seq))
    held = [box.take() for _ in range(8)]
    assert box.count == 8
    with pytest.raises(ProtocolError, match="voice_capacity_exceeded"):
        box.put(record(sequence=9))
    box.release(held.pop())
    box.put(record(sequence=9))
    assert box.count == 8


def test_consumed_delivery_relinquishes_payload_and_mailbox_custody():
    box = Mailbox("parent_to_child", generation=1)
    item = record()
    box.put(item)
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    receiver = ReceiveStream(key)
    delivery = receiver.receive(box.take(), mailbox=box)
    assert box.count == 1
    delivery.consume()
    assert delivery.record is None
    assert box.count == 0
    with pytest.raises(ProtocolError):
        box.release(item)


def test_data_end_is_written_after_data_but_cleanup_can_arrive_first():
    box = Mailbox("parent_to_child", generation=1)
    box.put(record("pcm", b"x" * 960))
    box.put(record("pcm_end", b"", last_sequence=1))
    box.put(record("tts_closed", b"", outcome="clean"))
    assert [box.take().header["op"] for _ in range(3)] == [
        "tts_closed",
        "pcm",
        "pcm_end",
    ]


def test_tts_cleanup_has_one_session_credit_charged_for_the_complete_record():
    key = StreamKey(1, "a" * 32, "tts_closed")
    window = CreditWindow(key)
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    item = record("tts_closed", b"", outcome="clean")
    with pytest.raises(ProtocolError):
        box.put(item)
    permit = window.try_reserve()
    permit.publish(item)
    box.put(item)
    size = len(encode_record(item, "parent_to_child"))
    assert window.outstanding == (1, size)
    assert window.try_reserve() is None
    with pytest.raises(ProtocolError):
        window.acknowledge(key, 1, size)  # Not even written yet.
    flush_and_close(box, 1)
    assert window.outstanding == (1, size)  # Ordinary close never refunds it.
    receiver = ReceiveStream(key)
    delivery = receiver.receive(item)
    for wrong_key in (
        StreamKey(2, "a" * 32, "tts_closed"),
        StreamKey(1, "b" * 32, "tts_closed"),
        StreamKey(1, "a" * 32, "tts_closed", "voice-turn-1", 1, 1, 1),
    ):
        with pytest.raises(ProtocolError):
            window.acknowledge(wrong_key, 1, size)
    with pytest.raises(ProtocolError):
        window.acknowledge(key, 1, size - 1)
    window.acknowledge(key, *delivery.consume())
    assert window.outstanding == (0, 0)
    with pytest.raises(ProtocolError):
        window.acknowledge(key, 1, size)


def test_tts_cleanup_receiver_retains_session_sequence_across_phrase_identities():
    receiver = ReceiveStream(StreamKey(1, "a" * 32, "tts_closed"))
    first = record("tts_closed", b"", outcome="clean")
    second = record("tts_closed", b"", sequence=2, phrase_id=2, outcome="failed")
    first_size = len(encode_record(first, "parent_to_child"))
    second_size = len(encode_record(second, "parent_to_child"))
    delivery = receiver.receive(first)
    with pytest.raises(ProtocolError, match="voice_capacity_exceeded"):
        receiver.receive(second)
    assert delivery.consume() == (1, first_size)
    assert receiver.receive(second).consume() == (2, first_size + second_size)
    with pytest.raises(ProtocolError):
        receiver.receive(second)
    with pytest.raises(ProtocolError):
        ReceiveStream(receiver.key).receive(second)


def test_two_retained_draft_slots_are_reusable_without_lifetime_growth():
    box = Mailbox("child_to_parent", generation=1)
    for n in range(100):
        turn = f"voice-turn-{n}"
        box.retain_turn(turn)
        box.put(record("draft", b"x", turn_id=turn, sequence=n + 1, revision=n + 1))
        item = box.take()
        with pytest.raises(ProtocolError):
            box.retire_turn(turn)
        box.release(item)
        box.retire_turn(turn)
        assert box._draft_slots == {}
        assert box._draft_owners == [None, None]
        assert not any(isinstance(key, tuple) for key in box._last_sequences)
    assert box.count == 0


@pytest.mark.asyncio
async def test_outbound_admission_requires_matching_prepull_reservation():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    window = CreditWindow(key)
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    with pytest.raises(ProtocolError):
        box.put(record())
    permit = await window.reserve()
    with pytest.raises(ProtocolError):
        permit.publish(record(epoch=2))
    item = record()
    permit.publish(item)
    box.put(item)
    with pytest.raises(ProtocolError):
        box.put(item)


def test_terminal_can_precede_its_last_data_but_cannot_overtake_consumption():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    stream = ReceiveStream(key, blocks=2)
    assert not stream.end(2)
    one = stream.receive(record(sequence=1))
    one.consume()
    assert not stream.complete
    two = stream.receive(record(sequence=2))
    assert not stream.complete
    two.consume()
    assert stream.complete
    with pytest.raises(ProtocolError):
        stream.receive(record(sequence=3))


@pytest.mark.parametrize(
    "provider",
    [
        "parakeet-onnx",
        "parakeet-mlx",
        "lightning-whisper-mlx",
        "faster-whisper",
        "qwen2audio",
        "parakeet",
        "canary",
    ],
)
def test_bootstrap_preserves_resolved_local_provider_and_nullable_model(provider):
    header = dict(
        version=1,
        op="bootstrap",
        generation=1,
        request_id="a" * 32,
        sequence=1,
        root="/synthetic/source",
        source="d" * 64,
        native_abi=1,
        stt_provider=provider,
        stt_model=None,
        language="en",
        response_eagerness_ms=700,
        aec_enabled=True,
        vad_aggressiveness=2,
        vad_preroll_ms=240,
    )
    result = read_record(
        io.BytesIO(encode_record(Record(header), "parent_to_child")).read,
        "parent_to_child",
    )
    assert result.header["stt_provider"] == provider
    assert result.header["stt_model"] is None


@pytest.mark.parametrize(
    "kind,boundary", [("promote", None), ("promote", "e" * 32), ("accepted", None)]
)
def test_terminal_proposal_carries_seal_and_answer_digest_without_parent_objects(
    kind, boundary
):
    item = record(
        "terminal_propose",
        "😀".encode() * 65536,
        context_handle="b" * 32,
        kind=kind,
        boundary_id=boundary,
        last_sequence=8,
    )
    decoded = read_record(
        io.BytesIO(encode_record(item, "child_to_parent")).read, "child_to_parent"
    )
    assert decoded.header["request_handle"] == "c" * 32
    assert decoded.header["kind"] == kind
    assert decoded.header["boundary_id"] == boundary
    assert len(decoded.payload) == 262144


@pytest.mark.parametrize(
    "change",
    [
        {"answer_sha256": "secret"},
        {"answer_chars": True},
        {"answer_chars": -1},
        {"boundary_id": 1},
        {"kind": "run_tools"},
        {"request_handle": "voice-turn-1"},
    ],
)
def test_terminal_invalid_digest_count_or_authority_handle_is_rejected(change):
    with pytest.raises(ProtocolError):
        encode_record(
            record(
                "terminal_propose",
                b"x",
                context_handle="b" * 32,
                last_sequence=0,
                **change,
            ),
            "child_to_parent",
        )


def test_credit_wire_carries_complete_stream_identity():
    key = StreamKey(1, "a" * 32, "pcm", "voice-turn-1", 1, 1, 2)
    sender = CreditWindow(key)
    permit = sender.try_reserve()
    item = record("pcm", b"x" * 960, phrase_id=2)
    permit.publish(item)
    outbound = Mailbox("parent_to_child", generation=1, outbound=True)
    outbound.put(item)
    flush_and_close(outbound, 1)
    stale = record(
        "credit",
        b"",
        lane="pcm",
        stream_id="a" * 32,
        stream_phrase=1,
        ack_sequence=1,
        ack_bytes=960,
    )
    with pytest.raises(ProtocolError):
        sender.accept_credit(stale)
    current = record(
        "credit",
        b"",
        lane="pcm",
        stream_id="a" * 32,
        stream_phrase=2,
        ack_sequence=1,
        ack_bytes=960,
    )
    sender.accept_credit(current)
    assert sender.outstanding == (0, 0)
    with pytest.raises(ProtocolError):
        sender.accept_credit(current)


def test_published_but_not_enqueued_credit_cannot_be_forged():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    sender = CreditWindow(key)
    permit = sender.try_reserve()
    permit.publish(record(payload=b"x"))
    with pytest.raises(ProtocolError):
        sender.acknowledge(key, 1, 1)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_json_is_not_reinterpreted_as_nullable_boundary(constant):
    header = dict(
        record("terminal_propose", b"", context_handle="b" * 32, last_sequence=0).header
    )
    raw = (
        json.dumps(header)
        .replace('"boundary_id": null', f'"boundary_id": {constant}')
        .encode()
    )
    with pytest.raises(ProtocolError, match="^voice_protocol_invalid$"):
        read_record(
            io.BytesIO(struct.pack("!II", len(raw), 0) + raw).read, "child_to_parent"
        )


def test_preparations_share_one_two_slot_budget_across_distinct_requests():
    key = StreamKey(1, "d" * 32, "prepare")
    sender = CreditWindow(key)
    outbound = Mailbox("child_to_parent", generation=1, outbound=True)
    held = []
    receiver = ReceiveStream(key)
    for n in range(2):
        permit = sender.try_reserve()
        item = record(
            "prepare",
            b"x" * 65536,
            sequence=permit.sequence,
            request_id=f"{n + 1:032x}",
        )
        permit.publish(item)
        outbound.put(item)
        held.append(receiver.receive(item))
    assert sender.try_reserve() is None
    flush_and_close(outbound, 2)
    sender.close()
    sender.acknowledge(key, *held[0].consume())
    assert sender.outstanding == (1, 65536)
    with pytest.raises(ProtocolError, match="voice_transport_closed"):
        sender.try_reserve()


@pytest.mark.asyncio
async def test_mailbox_failure_wakes_owned_credit_waiter():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    sender = CreditWindow(key, blocks=1)
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    permit = await sender.reserve()
    item = record()
    permit.publish(item)
    box.put(item)
    waiter = asyncio.create_task(sender.reserve())
    await asyncio.sleep(0)
    box.close()
    with pytest.raises(ProtocolError, match="voice_transport_closed"):
        await asyncio.wait_for(waiter, 1)


def test_mailbox_rejects_stale_data_before_body_read_after_dequeue():
    box = Mailbox("parent_to_child", generation=1)
    item = record()
    box.put(item)
    box.release(box.take())
    reader = io.BytesIO(encode_record(item, "parent_to_child"))
    with pytest.raises(ProtocolError):
        read_record(reader.read, "parent_to_child", admit=box.check_admission)
    assert reader.tell() == len(reader.getvalue()) - 5


def test_received_revocation_requires_matching_discard_of_known_buffered_data():
    box = Mailbox("parent_to_child", generation=1)
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    box.open_stream(key)
    box.put(record("cancel", b"", epoch=2, reason="stop"))
    reader = io.BytesIO(encode_record(record(epoch=1), "parent_to_child"))
    buffered = read_record(reader.read, "parent_to_child", admit=box.check_admission)
    box.put(buffered)
    box.release(box.take())  # Dispatch the fence before old data.
    delivery = ReceiveStream(key).receive(box.take(), mailbox=box)
    with pytest.raises(ProtocolError):
        delivery.consume()
    with pytest.raises(ProtocolError):
        delivery.discard(StreamKey(1, "b" * 32, "provider", "voice-turn-1", 1, 1))
    assert delivery.discard(key) == (1, 5)
    assert delivery.record is None
    assert box.count == 0


def test_already_held_delivery_becomes_discard_only_when_fence_arrives():
    box = Mailbox("parent_to_child", generation=1)
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    box.put(record())
    delivery = ReceiveStream(key).receive(box.take(), mailbox=box)
    assert not delivery.discard_only
    box.put(record("cancel", b"", reason="stop"))
    assert delivery.discard_only
    with pytest.raises(ProtocolError):
        delivery.consume()
    assert delivery.discard(key) == (1, 5)


def test_local_fence_requires_known_identity_without_changing_wire_sequence():
    box = Mailbox("parent_to_child", generation=1)
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    with pytest.raises(ProtocolError):
        box.fence_stream(key)
    box.open_stream(key)
    with pytest.raises(ProtocolError):
        box.fence_stream(StreamKey(2, "a" * 32, "provider", "voice-turn-1", 1, 1))
    box.fence_stream(key)
    box.fence_stream(key)
    # Local fences never invent a received wire sequence.
    box.put(record("cancel", b"", reason="stop"))
    assert box.take().header["sequence"] == 1


@pytest.mark.parametrize("identity", ["unknown", "replay", "wrong_request"])
def test_fenced_unknown_or_replayed_stream_is_rejected_before_body(identity):
    box = Mailbox("parent_to_child", generation=1)
    if identity != "unknown":
        box.put(record())
        box.release(box.take())
    box.put(record("cancel", b"", epoch=2, reason="stop"))
    item = (
        record(request_id="b" * 32, sequence=2)
        if identity == "wrong_request"
        else record()
    )
    reader = io.BytesIO(encode_record(item, "parent_to_child"))
    with pytest.raises(ProtocolError):
        read_record(reader.read, "parent_to_child", admit=box.check_admission)
    assert reader.tell() == len(reader.getvalue()) - 5


def test_coalesced_fences_have_one_current_and_one_pending_slot():
    box = Mailbox("parent_to_child", generation=1)
    box.put(record("cancel", b"", reason="stop"))
    first = box.take()
    for n in range(2, 100):
        box.put(record("cancel", b"", reason="stop", epoch=n, sequence=n))
        assert box.take() is None
        assert box.count == 2
    box.release(first)
    assert box.take().header["epoch"] == 99


def test_authoritative_cleanup_has_two_reserved_slots_and_never_coalesces():
    box = Mailbox("parent_to_child", generation=1)
    for n in range(64):
        box.put(
            record(
                "prepared",
                b"",
                sequence=n + 1,
                context_handle="b" * 32,
                decision="provisional",
            )
        )
    box.put(record("cleanup", b"", outcome="clean", epoch=1))
    box.put(record("cleanup", b"", outcome="detached", epoch=2, sequence=2))
    with pytest.raises(ProtocolError, match="voice_capacity_exceeded"):
        box.put(record("cleanup", b"", outcome="failed", epoch=3, sequence=3))
    assert [box.take().header["outcome"] for _ in range(2)] == ["clean", "detached"]


def test_outbound_projection_replacement_reuses_queued_credit_and_stops_inflight():
    key = StreamKey(1, "a" * 32, "preview")
    window = CreditWindow(key)
    box = Mailbox("child_to_parent", generation=1, outbound=True)
    box.retain_turn("voice-turn-1")
    first = record("preview", b"old")
    window.try_reserve().publish(first)
    box.put(first)
    assert box.replace_projection(record("preview", b"newest", revision=2))
    assert window.outstanding == (1, 6)
    held = box.take()
    assert held.payload == b"newest"
    assert not box.replace_projection(record("preview", b"too late", revision=3))
    assert window.try_reserve() is None


def test_many_retired_data_streams_keep_only_current_lane_identity():
    box = Mailbox("parent_to_child", generation=1)
    for n in range(1, 100):
        key = StreamKey(1, f"{n:032x}", "provider", f"voice-turn-{n}", 1, n)
        box.open_stream(key)
        box.put(record(request_id=key.request_id, turn_id=key.turn_id, epoch=n))
        box.release(box.take())
    with pytest.raises(ProtocolError):
        box.open_stream(StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1))
    assert box.count == 0


def test_pcm_consumption_barrier_remains_distinct_from_cleanup_receipt():
    key = StreamKey(1, "a" * 32, "pcm", "voice-turn-1", 1, 1, 1)
    stream = ReceiveStream(key)
    item = stream.receive(record("pcm", b"x" * 960))
    stream.end(1)
    item.consume()
    assert stream.data_complete
    assert stream.cleanup_outcome is None
    assert not stream.complete
    stream.closed("clean")
    assert stream.cleanup_outcome == "clean"
    assert stream.complete


def test_direct_malformed_record_never_leaks_private_operation_in_error():
    stream = ReceiveStream(StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1))
    with pytest.raises(ProtocolError, match="^voice_protocol_invalid$"):
        stream.receive(record("private-sentinel", b""))


def test_protocol_and_io_import_without_app_optional_models_or_network():
    script = """
import importlib.abc
import sys
blocked = ('textual', 'sounddevice', 'pyaudio', 'torch', 'mlx', 'numpy',
           'tldw_chatbook.Chat', 'tldw_chatbook.TTS', 'tldw_chatbook.app',
           'tldw_chatbook.config', 'tldw_chatbook.DB', 'tldw_chatbook.LLM_Calls')
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == item or fullname.startswith(item + '.') for item in blocked):
            raise AssertionError('forbidden import')
def no_network(event, args):
    if event in ('socket.connect', 'socket.connect_ex', 'socket.getaddrinfo', 'socket.bind', 'socket.sendto'):
        raise AssertionError('network forbidden')
sys.meta_path.insert(0, Guard())
sys.addaudithook(no_network)
from tldw_chatbook.Audio import voice_process_protocol, voice_process_io
assert all(not any(name == item or name.startswith(item + '.') for item in blocked) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=10
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_control_replay_is_rejected_after_original_custody_retires():
    box = Mailbox("child_to_parent", generation=1)
    item = record("start_attempt", b"", context_handle="b" * 32)
    box.put(item)
    box.release(box.take())
    with pytest.raises(ProtocolError):
        box.put(item)


def test_cleanup_and_closed_receipts_are_owned_by_parent_and_audio_respectively():
    cleanup = record("cleanup", b"", outcome="clean")
    closed = record("closed", b"", outcome="clean")
    assert encode_record(cleanup, "parent_to_child")
    assert encode_record(closed, "child_to_parent")
    with pytest.raises(ProtocolError):
        encode_record(cleanup, "child_to_parent")
    with pytest.raises(ProtocolError):
        encode_record(closed, "parent_to_child")


def test_cancelled_admission_still_writes_original_published_data_for_discard():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    window = CreditWindow(key)
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    permit = window.try_reserve()
    item = record(payload=b"published before cancel")
    permit.publish(item)
    box.put(item)
    window.close()
    with pytest.raises(ProtocolError, match="voice_transport_closed"):
        window.try_reserve()
    failures = []
    writer = PipeWriter(lambda part: len(part), box, failures.append)
    writer.start()
    try:
        assert writer.wait_written(1, 1)
        assert failures == []
        receiver = ReceiveStream(key)
        window.acknowledge(key, *receiver.receive(item).discard(key))
        assert window.outstanding == (0, 0)
    finally:
        writer.close()
        writer.join(1)


def test_queued_unstarted_records_cannot_return_any_sender_credit():
    key = StreamKey(1, "a" * 32, "provider", "voice-turn-1", 1, 1)
    window = CreditWindow(key)
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    for n in range(1, 9):
        permit = window.try_reserve()
        item = record(sequence=n, payload=b"x" * 4096)
        permit.publish(item)
        box.put(item)
    try:
        with pytest.raises(ProtocolError):
            window.acknowledge(key, 8, 32768)
        assert window.outstanding == (8, 32768)
        assert box.count == 8
        assert window.try_reserve() is None
    finally:
        box.close()


def test_projection_replacement_rejects_generation_before_transferring_custody():
    box = Mailbox("child_to_parent", generation=1, outbound=True)
    box.retain_turn("voice-turn-1")
    window = CreditWindow(StreamKey(1, "a" * 32, "preview"))
    old = record("preview", b"old")
    window.try_reserve().publish(old)
    box.put(old)
    with pytest.raises(ProtocolError):
        box.replace_projection(record("preview", b"foreign", generation=2, revision=2))
    assert window.outstanding == (1, 3)
    assert box.take() is old


def test_draft_reservation_cannot_be_relabelled_as_the_other_retained_slot():
    box = Mailbox("child_to_parent", generation=1, outbound=True)
    box.retain_turn("voice-turn-1")
    box.retain_turn("voice-turn-2")
    wrong_slot = CreditWindow(StreamKey(1, "a" * 32, "draft_0", "voice-turn-2", 1, 1))
    item = record("draft", b"x", turn_id="voice-turn-2", draft_slot=1)
    with pytest.raises(ProtocolError):
        wrong_slot.try_reserve().publish(item)
    assert box.count == 0


def test_draft_replacement_preserves_slot_request_and_retained_turn_identity():
    box = Mailbox("child_to_parent", generation=1, outbound=True)
    windows = []
    for index in range(2):
        turn = f"voice-turn-{index + 1}"
        box.retain_turn(turn)
        key = StreamKey(1, f"{index + 1:032x}", f"draft_{index}", turn, 1, 1)
        window = CreditWindow(key)
        item = record(
            "draft", b"old", request_id=key.request_id, turn_id=turn, draft_slot=index
        )
        window.try_reserve().publish(item)
        box.put(item)
        windows.append(window)
    with pytest.raises(ProtocolError):
        box.replace_projection(
            record(
                "draft",
                b"wrong",
                request_id="1".zfill(32),
                turn_id="voice-turn-2",
                revision=2,
            )
        )
    with pytest.raises(ProtocolError):
        box.replace_projection(
            record("draft", b"third", turn_id="voice-turn-3", revision=2)
        )
    assert windows[1].outstanding == (1, 3)
    assert box.replace_projection(
        record("draft", b"current", request_id="1".zfill(32), revision=2)
    )
    assert windows[0].outstanding == (1, 7)


def test_global_preview_can_replace_with_the_other_retained_turn():
    box = Mailbox("child_to_parent", generation=1, outbound=True)
    box.retain_turn("voice-turn-1")
    box.retain_turn("voice-turn-2")
    window = CreditWindow(StreamKey(1, "a" * 32, "preview"))
    old = record("preview", b"old")
    window.try_reserve().publish(old)
    box.put(old)
    assert box.replace_projection(
        record(
            "preview", b"second", request_id="b" * 32, turn_id="voice-turn-2", epoch=2
        )
    )
    with pytest.raises(ProtocolError):
        box.replace_projection(
            record("preview", b"third", turn_id="voice-turn-3", epoch=3)
        )
    assert box.take().payload == b"second"


@pytest.mark.parametrize("slot", [0, 1])
def test_draft_delivery_uses_retained_slot_across_revision_changes(slot):
    turn = f"voice-turn-{slot + 1}"
    key = StreamKey(1, "a" * 32, f"draft_{slot}", turn, 1, 1)
    receiver = ReceiveStream(key)
    first = receiver.receive(record("draft", b"one", turn_id=turn, draft_slot=slot))
    assert first.consume() == (1, 3)
    second = receiver.receive(
        record(
            "draft",
            b"two",
            turn_id=turn,
            sequence=2,
            revision=2,
            epoch=2,
            draft_slot=slot,
        )
    )
    assert second.discard(key) == (2, 6)
    with pytest.raises(ProtocolError):
        receiver.receive(
            record("draft", b"foreign", turn_id="voice-turn-3", sequence=3, revision=3)
        )


def test_received_draft_replacement_cannot_change_the_retained_request_identity():
    box = Mailbox("child_to_parent", generation=1)
    box.retain_turn("voice-turn-1")
    old = record("draft", b"retained")
    box.put(old)
    with pytest.raises(ProtocolError):
        box.replace_projection(
            record("draft", b"foreign", request_id="b" * 32, revision=2)
        )
    assert box.take() is old


@pytest.mark.parametrize("slot", [None, True, False, -1, 2, 0.0, "0", [], {}])
def test_draft_sender_slot_is_mandatory_exact_closed_integer(slot):
    header = dict(record("draft", b"x").header)
    if slot is None:
        header.pop("draft_slot")
    else:
        header["draft_slot"] = slot
    raw = json.dumps(header).encode()
    with pytest.raises(ProtocolError):
        read_record(
            io.BytesIO(struct.pack("!II", len(raw), 1) + raw + b"x").read,
            "child_to_parent",
        )


@pytest.mark.parametrize("slot", [0, 1])
def test_draft_sender_slot_roundtrips_but_is_not_allowed_on_prepare(slot):
    item = record("draft", b"x", draft_slot=slot)
    assert (
        read_record(
            io.BytesIO(encode_record(item, "child_to_parent")).read, "child_to_parent"
        )
        == item
    )
    with pytest.raises(ProtocolError):
        encode_record(record("prepare", b"x", draft_slot=slot), "child_to_parent")


@pytest.mark.parametrize("custody", ["queued", "consumer"])
def test_draft_sender_slot_cannot_replace_other_turn_custody(custody):
    box = Mailbox("child_to_parent", generation=1)
    for turn in ("old", "new"):
        box.retain_turn(turn)
    old = record("draft", b"old", turn_id="old")
    box.put(old)
    if custody == "consumer":
        assert box.take() is old
    with pytest.raises(ProtocolError):
        box.put(record("draft", b"new", turn_id="new", sequence=2, revision=2))
    assert box.count == 1


def test_draft_slot_binding_survives_metadata_order_replay_and_old_retirement():
    box = Mailbox("child_to_parent", generation=1)
    box.retain_turn("old")
    # Prepare/preview metadata may be retained before any draft arrives.
    box.retain_turn("new")
    box.put(record("draft", b"old", turn_id="old", draft_slot=1, sequence=7))
    box.release(box.take())
    with pytest.raises(ProtocolError):
        box.put(record("draft", b"changed", turn_id="old", draft_slot=0, sequence=8))
    box.put(record("draft", b"new", turn_id="new", draft_slot=1, sequence=1))
    box.release(box.take())
    with pytest.raises(ProtocolError):
        box.put(record("draft", b"stale", turn_id="old", draft_slot=1, sequence=8))
    box.retire_turn("old")
    with pytest.raises(ProtocolError):
        box.put(record("draft", b"replay", turn_id="new", draft_slot=1, sequence=1))
    box.put(record("draft", b"next", turn_id="new", draft_slot=1, sequence=2))
    assert box.take().payload == b"next"


@pytest.mark.parametrize(
    "changed",
    [
        {"stream_id": "b" * 32},
        {"stream_turn": "new"},
        {"stream_revision": 2},
        {"stream_epoch": 2},
        {"stream_phrase": 2},
    ],
)
def test_pending_credit_coalescing_requires_full_stream_identity(changed):
    box = Mailbox("parent_to_child", generation=1)
    fields = dict(
        lane="draft_0",
        stream_id="a" * 32,
        stream_turn="old",
        stream_revision=1,
        stream_epoch=1,
        stream_phrase=1,
    )
    old = record("credit", b"", **fields, ack_sequence=1, ack_bytes=3)
    box.put(old)
    with pytest.raises(ProtocolError):
        box.put(
            record("credit", b"", **(fields | changed), ack_sequence=2, ack_bytes=6)
        )
    assert box.take() is old


def test_writer_held_old_credit_allows_one_distinct_pending_successor():
    box = Mailbox("parent_to_child", generation=1, outbound=True)
    fields = dict(lane="draft_0", stream_id="a" * 32, stream_turn="old")
    old = record("credit", b"", **fields, ack_sequence=7, ack_bytes=21)
    box.put(old)
    active, finish = threading.Event(), threading.Event()
    written, faults = bytearray(), []

    def write(data):
        active.set()
        assert finish.wait(2)
        written.extend(data[:7])
        return len(data[:7])

    writer = PipeWriter(write, box, faults.append)
    writer.start()
    try:
        assert active.wait(1)
        successor = record(
            "credit",
            b"",
            **(fields | {"stream_turn": "new"}),
            ack_sequence=1,
            ack_bytes=3,
        )
        box.put(successor)
        assert box.count == 2
        assert box.take() is None
        with pytest.raises(ProtocolError):
            box.put(
                record(
                    "credit",
                    b"",
                    **(fields | {"stream_turn": "third"}),
                    ack_sequence=2,
                    ack_bytes=6,
                )
            )
        finish.set()
        assert writer.wait_written(2, 1)
        source = io.BytesIO(written)
        assert read_record(source.read, "parent_to_child") == old
        assert read_record(source.read, "parent_to_child") == successor
        assert not faults
    finally:
        finish.set()
        writer.close()
        writer.join(1)
        assert not writer.alive
