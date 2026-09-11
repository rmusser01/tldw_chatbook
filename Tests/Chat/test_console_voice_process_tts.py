"""Software-only TTS ownership, real private-pipe custody and cleanup races."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import io
import os
import threading
import wave

import pytest

from tldw_chatbook.Audio.voice_process_io import PipeReader, PipeWriter
from tldw_chatbook.Audio.voice_process_protocol import (
    Mailbox,
    ProtocolError,
    ReceiveStream,
    Record,
    StreamKey,
    read_record,
)
from tldw_chatbook.TTS.adapter_types import TTSAudioResponse
from tldw_chatbook.TTS.pcm_stream import iter_normalized_pcm_frames


class _Owner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        assert self.ready.wait(1)

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.identity = (os.getpid(), threading.get_ident(), self.loop)
        self.ready.set()
        self.loop.run_forever()

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(2)
        assert not self.thread.is_alive()
        self.loop.close()


@pytest.fixture
def owner():
    value = _Owner()
    try:
        yield value
    finally:
        value.close()


async def _until(predicate: Callable[[], bool]):
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0.001)
    pytest.fail("bounded software TTS operation did not settle")


def _key(phrase=1):
    return StreamKey(1, "a" * 32, "pcm", "voice-turn-1", 1, 1, phrase)


def _cleanup_key():
    return StreamKey(1, "a" * 32, "tts_closed")


def _record(key, op, sequence=1, payload=b"", **fields):
    if op == "tts_closed":
        fields.setdefault("last_sequence", 0)
    return Record(
        dict(
            version=1,
            op=op,
            generation=key.generation,
            request_id=key.request_id,
            turn_id=key.turn_id,
            revision=key.revision,
            epoch=key.epoch,
            phrase_id=key.phrase_id,
            sequence=sequence,
            **fields,
        ),
        payload,
    )


def _response(stream, *, cleanup=None, audio_format="pcm", channels=1):
    return TTSAudioResponse(
        provider_id="private-provider",
        model_id="private-model",
        audio_format=audio_format,
        content_type="private-content-type",
        sample_rate=48_000,
        metadata={"channels": channels, "private-source": "private-metadata"},
        byte_stream=stream,
        cleanup=cleanup,
    )


class _Link:
    """Use production pipe framing/custody; no simulated writer acknowledgements."""

    def __init__(self, owner, synthesize, *, on_fault=None):
        from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopPcmProducer

        self.outbound = Mailbox("parent_to_child", generation=1, outbound=True)
        self.inbound = Mailbox("parent_to_child", generation=1)
        self.cleanup_receiver = ReceiveStream(_cleanup_key())
        self.producer = OwnerLoopPcmProducer(
            owner.loop,
            synthesize,
            self.outbound,
            cleanup_key=_cleanup_key(),
            on_fault=on_fault,
        )
        self.records = []
        self.receivers = {}
        self.productions = []
        self.faults = []
        self.pending_credits = []
        self.hold_credits = False
        self.hold_cleanup_credits = False
        self.hold_dispatch = False
        self.pending_dispatch = []
        self.loop = asyncio.get_running_loop()
        read_fd, self.write_fd = os.pipe()
        self.reader = PipeReader(
            read_fd,
            self.inbound,
            self._schedule,
            self._receive,
            self.faults.append,
        )
        self.writer = PipeWriter(
            lambda part: os.write(self.write_fd, part),
            self.outbound,
            self.faults.append,
        )
        self.reader.start()
        self.writer.start()

    def _schedule(self, callback):
        if self.hold_dispatch:
            self.pending_dispatch.append(callback)
        else:
            self.loop.call_soon_threadsafe(callback)

    def resume_dispatch(self):
        self.hold_dispatch = False
        for callback in self.pending_dispatch:
            self.loop.call_soon(callback)
        self.pending_dispatch.clear()

    def _receive(self, record):
        # Keep wire shape and lengths, never retain a second PCM payload queue.
        self.records.append((dict(record.header), len(record.payload)))
        receiver = self.receivers.get(record.header["phrase_id"])
        if receiver is not None:
            receiver.receive(record)
        else:
            assert record.header["op"] == "tts_closed"
            self.inbound.release(record)

    def _credit(self, key, sequence, size):
        if (key.lane == "pcm" and self.hold_credits) or (
            key.lane == "tts_closed" and self.hold_cleanup_credits
        ):
            self.pending_credits.append((key, sequence, size))
        else:
            self.producer.acknowledge(key, sequence, size)

    def release_credits(self):
        self.hold_credits = False
        self.hold_cleanup_credits = False
        for credit in self.pending_credits:
            self.producer.acknowledge(*credit)
        self.pending_credits.clear()

    def start(self, text="test", phrase=1, *, active=True):
        from tldw_chatbook.Audio.voice_phrase_sequencer import ProcessPcmStream

        key = _key(phrase)
        receiver = ProcessPcmStream(
            key,
            self.inbound,
            cleanup_receiver=self.cleanup_receiver,
            on_credit=self._credit,
            on_cancel=self.producer.cancel,
        )
        self.receivers[phrase] = receiver
        if active:
            receiver.activate()
        production = self.producer.submit(key=key, text=text)
        self.productions.append(production)
        return receiver, production

    async def close(self):
        self.release_credits()
        self.resume_dispatch()
        for receiver in self.receivers.values():
            await receiver.aclose()
        await self.producer.aclose()
        for production in self.productions:
            await asyncio.wait_for(production.wait_for_retirement(), 1)
        self.writer.close()
        await asyncio.to_thread(self.writer.join, 1)
        os.close(self.write_fd)
        await asyncio.to_thread(self.reader.join, 1)
        self.reader.close()
        assert not self.reader.alive and not self.writer.alive
        assert all(error.code == "voice_transport_eof" for error in self.faults), (
            self.faults
        )


async def _collect(receiver):
    stream = receiver.normalized_stream()
    try:
        frames = [frame async for frame in stream.frames]
    finally:
        await stream.frames.aclose()
        await stream.cleanup
    return b"".join(frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("audio_format", ["pcm", "wav"])
async def test_parent_decodes_large_audio_with_bounded_real_pipe_credit(
    owner, monkeypatch, audio_format
):
    from tldw_chatbook.Audio import voice_preprocessor

    pcm = b"\x23\x01" * (480 * 600)
    body = io.BytesIO()
    with wave.open(body, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(48_000)
        wav.writeframes(pcm)
    encoded = body.getvalue() if audio_format == "wav" else pcm

    async def chunks():
        for offset in range(0, len(encoded), 120_000):
            yield encoded[offset : offset + 120_000]

    expected = b"".join(
        [
            frame
            async for frame in iter_normalized_pcm_frames(
                audio_format=audio_format,
                sample_rate=48_000,
                channels=1,
                byte_stream=chunks(),
            )
        ]
    )
    assert len(encoded) > 512 * 1024 and expected == pcm
    identities = {}

    def observe(stage):
        identities.setdefault(stage, set()).add(
            (os.getpid(), threading.get_ident(), asyncio.get_running_loop())
        )

    normalize = voice_preprocessor.normalize_pcm16_frames
    normalization_calls = 0

    def observed_normalize(*args, **kwargs):
        nonlocal normalization_calls
        normalization_calls += 1
        observe("normalize")
        return normalize(*args, **kwargs)

    monkeypatch.setattr(
        voice_preprocessor, "normalize_pcm16_frames", observed_normalize
    )

    class Response(TTSAudioResponse):
        async def aclose(self):
            observe("first_close" if not self._closed else "duplicate_close")
            await super().aclose()

    async def synthesize(*, text):
        observe("synthesize")

        async def stream():
            async for chunk in chunks():
                observe("iterate")
                yield chunk

        async def cleanup():
            observe("cleanup")

        response = _response(stream(), audio_format=audio_format, cleanup=cleanup)
        response.__class__ = Response
        return response

    link = _Link(owner, synthesize)
    try:
        receiver, production = link.start()
        await _until(lambda: len(link.records) >= 8)
        await asyncio.sleep(0.02)
        assert len(link.records) == 8  # Neither write nor dequeue replenishes.
        assert normalization_calls == 8  # No ninth decoder pull before credit.
        assert link.producer.outstanding == (8, 8 * 960)
        assert receiver.receiver.ack == (0, 0)
        assert await _collect(receiver) == expected
        await production.wait_for_cleanup()
        await _until(lambda: link.producer.outstanding == (0, 0))
        assert set(identities) == {
            "synthesize",
            "iterate",
            "normalize",
            "first_close",
            "cleanup",
        }
        assert all(value == {owner.identity} for value in identities.values())
        for header, size in link.records:
            assert set(header) <= {
                "version",
                "op",
                "generation",
                "request_id",
                "turn_id",
                "revision",
                "epoch",
                "phrase_id",
                "sequence",
                "last_sequence",
                "outcome",
            }
            if header["op"] == "pcm":
                assert 0 < size <= 65_536 and size % 960 == 0
        assert receiver.receiver.complete
    finally:
        await link.close()


@pytest.mark.asyncio
async def test_first_raw_frame_does_not_wait_for_later_chunk_or_eof(owner):
    release = threading.Event()
    closed = threading.Event()

    async def synthesize(*, text):
        async def stream():
            yield b"\x01\x00" * 480
            while not release.is_set():
                await asyncio.sleep(0.001)
            yield b"\x02\x00" * 480

        async def cleanup():
            closed.set()

        return _response(stream(), cleanup=cleanup)

    link = _Link(owner, synthesize)
    try:
        receiver, production = link.start()
        stream = receiver.normalized_stream()
        assert await asyncio.wait_for(anext(stream.frames), 0.5) == b"\x01\x00" * 480
        assert not closed.is_set()
        await stream.frames.aclose()
        await stream.cleanup
        await production.wait_for_cleanup()
        assert closed.is_set()
    finally:
        release.set()
        await link.close()


@pytest.mark.asyncio
async def test_cancel_before_owner_coroutine_entry_still_settles_its_receipt(owner):
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopPcmProducer

    blocked = threading.Event()
    release = threading.Event()
    calls = []

    def hold_owner():
        blocked.set()
        assert release.wait(1)

    async def synthesize(*, text):
        calls.append(text)
        raise AssertionError("cancelled request must never enter the service")

    mailbox = Mailbox("parent_to_child", generation=1, outbound=True)
    producer = OwnerLoopPcmProducer(
        owner.loop, synthesize, mailbox, cleanup_key=_cleanup_key()
    )
    owner.loop.call_soon_threadsafe(hold_owner)
    await _until(blocked.is_set)
    production = producer.submit(key=_key(), text="cancelled before owner entry")
    production.cancel()
    release.set()
    assert await asyncio.wait_for(production.wait_for_cleanup(), 0.5) == "clean"
    assert calls == []
    await producer.aclose()
    producer.transport_failed(ProtocolError("voice_transport_eof"))
    await production.wait_for_retirement()


@pytest.mark.asyncio
async def test_cancel_full_lane_fences_late_pcm_without_finishing_parent_close(owner):
    close_started = threading.Event()
    release_close = threading.Event()
    close_finished = threading.Event()

    async def synthesize(*, text):
        async def stream():
            yield b"\x01\x00" * (480 * 50)

        async def cleanup():
            close_started.set()
            while not release_close.is_set():
                await asyncio.sleep(0.001)
            close_finished.set()

        return _response(stream(), cleanup=cleanup)

    link = _Link(owner, synthesize)
    try:
        receiver, production = link.start()
        await _until(lambda: len(link.records) == 8)
        observer = asyncio.create_task(production.wait_for_cleanup())
        receiver.fence()
        await _until(close_started.is_set)
        await _until(lambda: link.producer.outstanding == (0, 0))
        assert receiver.receiver.cleanup_outcome is None
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        observer = asyncio.create_task(production.wait_for_cleanup())
        await asyncio.sleep(0.01)
        assert not observer.done() and not close_finished.is_set()
        release_close.set()
        await observer
        await receiver.wait_for_cleanup()
        assert close_finished.is_set()
        assert not any(header["op"] == "pcm_end" for header, _ in link.records)
    finally:
        release_close.set()
        await link.close()


@pytest.mark.asyncio
async def test_natural_first_close_and_withheld_final_credit_block_next_phrase(owner):
    close_started = threading.Event()
    release_close = threading.Event()
    calls = []
    responses = []

    async def synthesize(*, text):
        calls.append(text)

        async def stream():
            yield b"\x01\x00" * 480

        async def cleanup():
            if text == "first":
                close_started.set()
                while not release_close.is_set():
                    await asyncio.sleep(0.001)

        response = _response(stream(), cleanup=cleanup)
        responses.append(response)
        return response

    link = _Link(owner, synthesize)
    try:
        first, production = link.start("first")
        link.hold_credits = True
        consumer = asyncio.create_task(_collect(first))
        await _until(close_started.is_set)
        await _until(lambda: first.receiver.data_complete)
        assert responses[0]._closed  # Real type marks closed before the awaited work.
        assert first.receiver.cleanup_outcome is None and not consumer.done()
        # Queue before old receiver custody clears; only the producer starts now.
        second_production = link.producer.submit(key=_key(2), text="second")
        link.productions.append(second_production)
        production.cancel()
        await asyncio.sleep(0.02)
        assert calls == ["first"]
        release_close.set()
        await consumer
        await production.wait_for_cleanup()
        await asyncio.sleep(0.02)
        assert calls == ["first"]  # Cleanup alone does not retire old PCM.
        from tldw_chatbook.Audio.voice_phrase_sequencer import ProcessPcmStream

        second = ProcessPcmStream(
            _key(2),
            link.inbound,
            cleanup_receiver=link.cleanup_receiver,
            on_credit=link._credit,
            on_cancel=link.producer.cancel,
        )
        link.receivers[2] = second
        second.activate()
        link.release_credits()
        assert await _collect(second) == b"\x01\x00" * 480
        await second_production.wait_for_cleanup()
        assert calls == ["first", "second"]
    finally:
        release_close.set()
        link.release_credits()
        await link.close()


@pytest.mark.asyncio
async def test_cancelled_middle_production_keeps_predecessor_cleanup_barrier(owner):
    started = threading.Event()
    release = threading.Event()
    calls = []

    async def synthesize(*, text):
        calls.append(text)

        async def stream():
            yield b"\x01\x00" * 480

        async def cleanup():
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)

        return _response(stream(), cleanup=cleanup)

    link = _Link(owner, synthesize)
    try:
        first, _ = link.start("first")
        consumer = asyncio.create_task(_collect(first))
        await _until(started.is_set)
        middle_stream, middle = link.start("middle", 2, active=False)
        middle.cancel()
        last_stream, last = link.start("last", 3, active=False)
        last_observer = asyncio.create_task(last.wait_for_cleanup())
        middle_observer = asyncio.create_task(middle.wait_for_cleanup())
        await asyncio.sleep(0.02)
        assert calls == ["first"]
        assert not middle_observer.done() and not last_observer.done()
        last.cancel()
        release.set()
        await consumer
        await middle_observer
        await last_observer
        await middle_stream.wait_for_cleanup()
        await last_stream.wait_for_cleanup()
        assert calls == ["first"]
    finally:
        release.set()
        await link.close()


@pytest.mark.asyncio
async def test_three_queued_cancellations_close_before_stalled_receipt_dispatch():
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsError

    async def synthesize(**kwargs):
        raise AssertionError("queued cancellation must not enter TTS")

    link = _Link(SimpleNamespace(loop=asyncio.get_running_loop()), synthesize)
    link.hold_dispatch = True
    streams, productions = [], []
    try:
        for phrase in (1, 2, 3):
            stream, production = link.start(phrase=phrase, active=False)
            streams.append(stream)
            productions.append(production)
            stream.fence()
        outcomes = await asyncio.gather(
            *(item.wait_for_cleanup() for item in productions)
        )
        assert outcomes == ["clean"] * 3
        await _until(lambda: link.inbound.count == 1)
        assert link.producer.cleanup_outstanding[0] == 1
        assert link.records == []
        with pytest.raises(OwnerLoopTtsError, match="tts_bridge_capacity_exceeded"):
            link.producer.submit(key=_key(4), text="fourth")
        link.resume_dispatch()
        assert (
            await asyncio.gather(*(item.wait_for_retirement() for item in productions))
            == ["clean"] * 3
        )
        assert all(stream.receiver.data_complete for stream in streams)
        assert all(
            header["op"] == "tts_closed" and header["last_sequence"] == 0
            for header, _ in link.records
        )
    finally:
        await link.close()


@pytest.mark.asyncio
async def test_waiting_phrase_retention_is_bounded_during_predecessor_cleanup(owner):
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsError

    started = threading.Event()
    release = threading.Event()
    calls = []

    async def synthesize(*, text):
        calls.append(text)

        async def stream():
            yield bytes(960)

        async def cleanup():
            started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)

        return _response(stream(), cleanup=cleanup)

    link = _Link(owner, synthesize)
    try:
        receiver, _ = link.start("first")
        collecting = asyncio.create_task(_collect(receiver))
        await _until(started.is_set)
        accepted = []
        for phrase in range(2, 100):
            try:
                _, production = link.start("queued", phrase, active=False)
                accepted.append(production)
            except OwnerLoopTtsError as error:
                assert str(error) == "tts_bridge_capacity_exceeded"
                break
        else:
            pytest.fail("queued phrase tasks and text grew without a finite bound")
        assert 2 <= len(accepted) < 64
        assert calls == ["first"]
        for production in accepted:
            production.cancel()
        release.set()
        await collecting
        for production in accepted:
            await production.wait_for_cleanup()
        assert calls == ["first"]
    finally:
        release.set()
        await link.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("hold", ["dispatch", "ack"])
async def test_three_cleanup_owners_survive_stalled_receipt_dispatch_or_credit(
    owner, hold
):
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsError

    close_started = threading.Event()
    release = threading.Event()
    calls = []

    async def synthesize(*, text):
        calls.append(text)

        async def stream():
            yield bytes(960)

        async def cleanup():
            close_started.set()
            while not release.is_set():
                await asyncio.sleep(0.001)

        return _response(stream(), cleanup=cleanup)

    link = _Link(owner, synthesize)
    try:
        first, production = link.start("first")
        collecting = asyncio.create_task(_collect(first))
        await _until(close_started.is_set)
        await _until(lambda: first.receiver.data_complete)
        middle, middle_production = link.start("middle", 2, active=False)
        last, last_production = link.start("last", 3, active=False)
        middle.fence()
        last.fence()
        link.hold_dispatch = hold == "dispatch"
        link.hold_cleanup_credits = hold == "ack"
        release.set()
        for item in (production, middle_production, last_production):
            assert await asyncio.wait_for(item.wait_for_cleanup(), 0.5) == "clean"
        await _until(
            lambda: bool(
                link.pending_dispatch if hold == "dispatch" else link.pending_credits
            )
        )
        # Cancelled observers cannot cancel publication of an already-clean owner.
        production.cancel()
        with pytest.raises(OwnerLoopTtsError, match="tts_bridge_capacity_exceeded"):
            link.producer.submit(key=_key(4), text="over capacity")
        assert calls == ["first"]
        assert middle.receiver.cleanup_outcome is None
        assert last.receiver.cleanup_outcome is None
        assert link.cleanup_receiver.ack[0] == (0 if hold == "dispatch" else 1)
        assert link.producer.cleanup_outstanding[0] == 1
        if hold == "dispatch":
            assert len(link.pending_dispatch) == 1
            assert link.inbound.count == 1
        link.release_credits()
        link.resume_dispatch()
        await collecting
        await middle.wait_for_cleanup()
        await last.wait_for_cleanup()
        for item in (production, middle_production, last_production):
            assert await asyncio.wait_for(item.wait_for_retirement(), 1) == "clean"
        assert link.cleanup_receiver.ack[0] == 3
        assert link.producer.cleanup_outstanding == (0, 0)
    finally:
        release.set()
        await link.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [("request_id", "b" * 32), ("epoch", 2), ("phrase_id", 2), ("revision", 2)],
)
async def test_cleanup_receipt_requires_exact_accepted_phrase_before_credit(
    field, value
):
    from tldw_chatbook.Audio.voice_phrase_sequencer import ProcessPcmStream

    inbox = Mailbox("parent_to_child", generation=1)
    receipts = ReceiveStream(_cleanup_key())
    credits = []
    receiver = ProcessPcmStream(
        _key(),
        inbox,
        cleanup_receiver=receipts,
        on_credit=lambda *args: credits.append(args),
        on_cancel=lambda _: None,
    )
    original = _record(_key(), "tts_closed", outcome="clean")
    inbox.put(Record(dict(original.header, **{field: value})))
    held = inbox.take()
    with pytest.raises(ProtocolError):
        receiver.receive(held)
    assert receiver.receiver.cleanup_outcome is None
    assert receipts.ack == (0, 0) and credits == []
    inbox.release(held)


@pytest.mark.asyncio
async def test_failed_close_reports_fatal_before_held_receipt_credit_retires(owner):
    from tldw_chatbook.Audio.voice_process_types import NormalizedPcmError
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsError

    faults = []
    calls = []

    async def synthesize(*, text):
        calls.append(text)

        async def stream():
            yield bytes(960)

        async def cleanup():
            if text == "bad":
                raise RuntimeError("private cleanup failure")

        return _response(stream(), cleanup=cleanup)

    link = _Link(owner, synthesize, on_fault=lambda error: faults.append(error.code))
    try:
        link.hold_cleanup_credits = True
        first, first_production = link.start("first")
        await _collect(first)
        second, second_production = link.start("bad", 2)
        consumer = asyncio.create_task(_collect(second))
        assert await second_production.wait_for_cleanup() == "failed"
        await _until(lambda: bool(faults))
        assert faults == ["tts_bridge_failed"]
        assert second.receiver.cleanup_outcome is None
        assert await first_production.wait_for_cleanup() == "clean"
        assert not consumer.done()
        with pytest.raises(OwnerLoopTtsError, match="tts_bridge_closed"):
            link.producer.submit(key=_key(3), text="replacement")
        assert calls == ["first", "bad"]
        link.release_credits()
        with pytest.raises(NormalizedPcmError):
            await consumer
        assert await second_production.wait_for_retirement() == "clean"
    finally:
        await link.close()


@pytest.mark.asyncio
async def test_transport_retirement_never_rewrites_a_real_cleanup_result(owner):
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopPcmProducer

    async def synthesize(*, text):
        async def stream():
            yield bytes(960)

        return _response(stream())

    outbound = Mailbox("parent_to_child", generation=1, outbound=True)
    producer = OwnerLoopPcmProducer(
        owner.loop, synthesize, outbound, cleanup_key=_cleanup_key()
    )
    production = producer.submit(key=_key(), text="first")
    assert await production.wait_for_cleanup() == "clean"
    await _until(lambda: producer.cleanup_outstanding[0] == 1)
    retirement = asyncio.create_task(production.wait_for_retirement())
    retirement.cancel()
    with pytest.raises(asyncio.CancelledError):
        await retirement
    producer.transport_failed(ProtocolError("voice_transport_eof"))
    assert await production.wait_for_retirement() == "failed"
    assert await production.wait_for_cleanup() == "clean"
    assert producer.outstanding == (1, 960)
    assert producer.cleanup_outstanding[0] == 1
    await producer.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("broken_observer", [False, True])
async def test_end_admission_failure_retains_fixed_fault_and_actual_cleanup(
    owner, broken_observer
):
    from tldw_chatbook.Chat.console_voice_tts_bridge import (
        OwnerLoopPcmProducer,
        OwnerLoopTtsError,
    )

    class RejectEnd(Mailbox):
        def put(self, record):
            if record.header["op"] == "pcm_end":
                raise ProtocolError("voice_capacity_exceeded")
            return super().put(record)

    closed = threading.Event()
    faults = []

    def fault(error):
        faults.append(error.code)
        if broken_observer:
            raise RuntimeError("private observer content")

    async def synthesize(*, text):
        async def stream():
            yield bytes(960)

        async def cleanup():
            closed.set()

        return _response(stream(), cleanup=cleanup)

    mailbox = RejectEnd("parent_to_child", generation=1, outbound=True)
    producer = OwnerLoopPcmProducer(
        owner.loop, synthesize, mailbox, cleanup_key=_cleanup_key(), on_fault=fault
    )
    production = producer.submit(key=_key(), text="first")
    assert await production.wait_for_cleanup() == "failed"
    assert closed.is_set()
    assert faults == ["voice_capacity_exceeded"]
    assert producer.failure.code == "voice_capacity_exceeded"
    with pytest.raises(OwnerLoopTtsError, match="^tts_bridge_closed$"):
        producer.submit(key=_key(2), text="late")
    await producer.aclose()
    await production.wait_for_retirement()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_failed_predecessor_cleanup_never_enters_queued_synthesis(
    owner, cancel_waiter
):
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopPcmProducer

    entered_close = threading.Event()
    release_close = threading.Event()
    calls = []

    async def synthesize(*, text):
        calls.append(text)

        async def stream():
            yield bytes(960)

        async def cleanup():
            if text == "first":
                entered_close.set()
                while not release_close.is_set():
                    await asyncio.sleep(0.001)
                raise RuntimeError("private uncertain resource cleanup")

        return _response(stream(), cleanup=cleanup)

    # Use real writer completion and a matching receiver discard; no device.
    outbound = Mailbox("parent_to_child", generation=1, outbound=True)
    producer = OwnerLoopPcmProducer(
        owner.loop, synthesize, outbound, cleanup_key=_cleanup_key()
    )
    wire = io.BytesIO()
    writer = PipeWriter(wire.write, outbound, lambda error: None)
    writer.start()
    first = producer.submit(key=_key(), text="first")
    try:
        await _until(entered_close.is_set)
        assert await asyncio.to_thread(writer.wait_written, 2, 1)
        packet = read_record(io.BytesIO(wire.getvalue()).read, "parent_to_child")
        delivery = ReceiveStream(_key()).receive(packet)
        producer.acknowledge(_key(), *delivery.discard(_key()))
        second = producer.submit(key=_key(2), text="second")
        if cancel_waiter:
            await asyncio.sleep(0.01)
            second.cancel()
        release_close.set()
        assert await first.wait_for_cleanup() == "failed"
        assert await second.wait_for_cleanup() == "failed"
        assert calls == ["first"]
    finally:
        release_close.set()
        await producer.aclose()
        producer.transport_failed(ProtocolError("voice_transport_eof"))
        await first.wait_for_retirement()
        await second.wait_for_retirement()
        writer.close()
        await asyncio.to_thread(writer.join, 1)
        assert not writer.alive


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["synthesize", "normalize", "cleanup"])
async def test_failure_wakes_consumer_with_only_categorical_outcome(owner, phase):
    from tldw_chatbook.Audio.voice_process_types import NormalizedPcmError

    async def synthesize(*, text):
        if phase == "synthesize":
            raise RuntimeError("private-provider-content")

        async def stream():
            yield b"odd" if phase == "normalize" else b"\x01\x00" * 480

        async def cleanup():
            if phase == "cleanup":
                raise RuntimeError("private-cleanup-content")

        return _response(stream(), cleanup=cleanup)

    link = _Link(owner, synthesize)
    try:
        receiver, production = link.start()
        with pytest.raises(NormalizedPcmError, match="^synthesis_failed$"):
            await asyncio.wait_for(_collect(receiver), 1)
        assert await production.wait_for_cleanup() == "failed"
        assert receiver.receiver.cleanup_outcome == "failed"
        assert not receiver.receiver.complete
        if phase != "cleanup":
            assert not any(header["op"] == "pcm_end" for header, _ in link.records)
    finally:
        await link.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["synthesize", "source"])
async def test_unrequested_cancellation_fails_cleanup_and_closes_admission(
    owner, phase
):
    from tldw_chatbook.Audio.voice_process_types import NormalizedPcmError
    from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopTtsError

    async def synthesize(*, text):
        if phase == "synthesize":
            raise asyncio.CancelledError

        async def stream():
            yield b"\x01\x00" * 480
            raise asyncio.CancelledError

        return _response(stream())

    faults = []
    link = _Link(owner, synthesize, on_fault=faults.append)
    try:
        receiver, production = link.start()
        assert await production.wait_for_cleanup() == "failed"
        with pytest.raises(NormalizedPcmError, match="^synthesis_failed$"):
            await asyncio.wait_for(_collect(receiver), 1)
        assert receiver.receiver.cleanup_outcome == "failed"
        assert not any(header["op"] == "pcm_end" for header, _ in link.records)
        assert [error.code for error in faults] == ["tts_bridge_failed"]
        assert link.producer.failure.code == "tts_bridge_failed"
        with pytest.raises(OwnerLoopTtsError, match="^tts_bridge_closed$"):
            link.producer.submit(key=_key(2), text="replacement")
    finally:
        await link.close()


@pytest.mark.asyncio
async def test_early_cleanup_receipt_waits_for_last_data_and_consumption():
    from tldw_chatbook.Audio.voice_phrase_sequencer import ProcessPcmStream

    key = _key()
    inbox = Mailbox("parent_to_child", generation=1)
    credits = []
    receiver = ProcessPcmStream(
        key,
        inbox,
        cleanup_receiver=ReceiveStream(_cleanup_key()),
        on_credit=lambda *args: credits.append(args),
        on_cancel=lambda _: None,
    )
    receiver.activate()

    def deliver(record):
        inbox.put(record)
        receiver.receive(inbox.take())

    deliver(_record(key, "tts_closed", outcome="clean", last_sequence=1))
    await receiver.wait_for_cleanup()
    deliver(_record(key, "pcm_end", last_sequence=1))
    stream = receiver.normalized_stream()
    pending = asyncio.create_task(anext(stream.frames))
    await asyncio.sleep(0)
    assert not pending.done() and not receiver.receiver.complete
    deliver(_record(key, "pcm", payload=b"\x05\x00" * 480))
    assert await pending == b"\x05\x00" * 480
    assert all(key.lane == "tts_closed" for key, _, _ in credits)
    assert not receiver.receiver.complete
    with pytest.raises(StopAsyncIteration):
        await anext(stream.frames)
    await stream.cleanup
    assert credits[-1] == (key, 1, 960)
    assert receiver.receiver.complete


@pytest.mark.asyncio
async def test_early_cleanup_cannot_start_next_synthesis_while_normal_end_is_held():
    from types import SimpleNamespace
    from Tests.Audio.test_voice_process_core import Transport
    from tldw_chatbook.Audio.voice_phrase_sequencer import PhraseSpeechSequencer

    async def frames():
        yield b"\x01\x00" * 480

    async def synthesize(**kwargs):
        return _response(frames())

    link = _Link(SimpleNamespace(loop=asyncio.get_running_loop()), synthesize)
    link.hold_dispatch = True
    calls = []

    class Synthesizer:
        async def synthesize_hands_free(self, *, text):
            calls.append(text)
            receiver, _ = link.start(text, phrase=len(calls))
            return receiver.normalized_stream()

    transport = Transport()
    speech = PhraseSpeechSequencer(epoch=1, synthesizer=Synthesizer(), sink=transport)
    try:
        await speech.feed(1, "First sentence. Second sentence. ")
        await _until(lambda: link.inbound.queued == 3)
        first = link.inbound.take()
        assert first.header["op"] == "tts_closed"
        link._receive(first)
        data = link.inbound.take()
        assert data.header["op"] == "pcm"
        link._receive(data)
        await _until(lambda: len(transport.rendered) == 1)
        await asyncio.sleep(0.01)
        assert calls == ["First sentence."]
        assert link.inbound.queued == 1  # The original uncredited pcm_end slot.
        assert not link.receivers[1].receiver.data_complete
        link.resume_dispatch()
        await speech.finish(1)
        await _until(lambda: len(calls) == 2 and len(transport.rendered) == 2)
        assert calls == ["First sentence.", "Second sentence."]
    finally:
        link.resume_dispatch()
        await speech.cancel(1)
        await speech.wait_for_cleanup()
        await link.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["aclose", "fail", "fence"])
@pytest.mark.parametrize("exhausted", [False, True])
async def test_repeated_old_fence_does_not_revoke_replacement_pcm(operation, exhausted):
    from tldw_chatbook.Audio.voice_phrase_sequencer import ProcessPcmStream

    inbox = Mailbox("parent_to_child", generation=1)
    cleanup_receiver = ReceiveStream(_cleanup_key())
    credits = []
    cancellations = []

    def receiver(key):
        return ProcessPcmStream(
            key,
            inbox,
            cleanup_receiver=cleanup_receiver,
            on_credit=lambda *args: credits.append(args),
            on_cancel=cancellations.append,
        )

    def deliver(target, record):
        inbox.put(record)
        target.receive(inbox.take())

    old = receiver(_key())
    old.activate()
    frame = b"\x01\x00" * 480
    deliver(old, _record(_key(), "pcm", payload=frame))
    if exhausted:
        deliver(old, _record(_key(), "pcm_end", last_sequence=1))
        assert await anext(old) == frame
        with pytest.raises(StopAsyncIteration):
            await anext(old)
        assert credits == [(_key(), 1, 960)]
        assert old.receiver.cleanup_outcome is None
    else:
        old.fence()
        # Revocation still accepts/discards matching late data before replacement.
        deliver(old, _record(_key(), "pcm", sequence=2, payload=frame))
        assert credits == [(_key(), 1, 960), (_key(), 2, 1920)]
        deliver(old, _record(_key(), "tts_closed", outcome="clean", last_sequence=2))
        await old.wait_for_cleanup()
        assert (
            old.receiver.data_complete
        )  # Exact cleanup boundary completes discard, not playback.

    new_key = StreamKey(1, "a" * 32, "pcm", "voice-turn-1", 1, 2, 1)
    replacement = receiver(new_key)
    replacement.activate()
    if operation == "aclose":
        await old.aclose()
    else:
        getattr(old, operation)()
    assert cancellations == ([] if exhausted and operation == "aclose" else [_key()])
    if exhausted:
        assert old.receiver.cleanup_outcome is None
        deliver(old, _record(_key(), "tts_closed", outcome="clean", last_sequence=1))
    assert old.receiver.cleanup_outcome == "clean"
    await old.wait_for_cleanup()

    deliver(replacement, _record(new_key, "pcm", payload=frame))
    deliver(replacement, _record(new_key, "pcm_end", sequence=2, last_sequence=1))
    assert await anext(replacement) == frame
    with pytest.raises(StopAsyncIteration):
        await anext(replacement)
    assert credits[-1] == (new_key, 1, 960)
    deliver(
        replacement,
        _record(new_key, "tts_closed", sequence=2, outcome="clean", last_sequence=1),
    )
    await replacement.wait_for_cleanup()
    assert replacement.receiver.complete


@pytest.mark.asyncio
async def test_late_old_pcm_is_discarded_after_local_device_fence():
    from tldw_chatbook.Audio.voice_phrase_sequencer import (
        PhraseSpeechSequencer,
        ProcessPcmStream,
    )

    inbox = Mailbox("parent_to_child", generation=1)
    credits = []
    cancellations = []
    receiver = ProcessPcmStream(
        _key(),
        inbox,
        cleanup_receiver=ReceiveStream(_cleanup_key()),
        on_credit=lambda *args: credits.append(args),
        on_cancel=cancellations.append,
    )
    receiver.activate()

    class Sink:
        fenced = False
        frames = []

        def queue_render(self, frame):
            assert not self.fenced, "old PCM reached the device after its fence"
            self.frames.append(frame)
            return object()

        def fence_output(self):
            self.fenced = True

    class Synthesizer:
        async def synthesize_hands_free(self, *, text):
            return receiver.normalized_stream()

    sink = Sink()
    sequencer = PhraseSpeechSequencer(epoch=1, synthesizer=Synthesizer(), sink=sink)
    await sequencer.feed(1, "First sentence. ")
    await _until(lambda: sequencer.synthesis_in_flight)
    cancelling = asyncio.create_task(sequencer.cancel(1))
    await _until(lambda: bool(cancellations))
    inbox.put(_record(_key(), "pcm", payload=b"\x01\x00" * 480))
    receiver.receive(inbox.take())
    assert sink.frames == [] and credits == [(_key(), 1, 960)]
    assert receiver.receiver.cleanup_outcome is None
    await cancelling
    assert sequencer.supervised_cleanup_count == 1
    inbox.put(_record(_key(), "tts_closed", outcome="clean", last_sequence=1))
    receiver.receive(inbox.take())
    await sequencer.wait_for_cleanup()
    assert sequencer.supervised_cleanup_count == 0
