"""Parent capability custody under fake child/view lifetime races."""

import asyncio
from dataclasses import replace
import hashlib
import importlib
import io
import threading
from uuid import uuid4

import pytest

from Tests.Chat.test_console_voice_attempts import _request
from Tests.Chat.test_console_voice_effect_barrier import _context
from Tests.Chat.test_console_voice_promotion import _winning_case
from tldw_chatbook.Audio.voice_process_protocol import (
    Mailbox,
    ProtocolError,
    Record,
    StreamKey,
    encode_record,
)
from tldw_chatbook.Chat.console_speculative_voice_session import (
    PreparedSpeculativeVoiceAttempt,
)
from tldw_chatbook.Chat.console_voice_promotion import (
    VoicePromotionOwner,
    VoiceWinningPromotion,
)
from tldw_chatbook.Chat.console_voice_supervisor import (
    VoiceDispatchKind,
    VoiceDispatchSupervisor,
    VoiceDispatchQuarantined,
)
from tldw_chatbook.Chat.console_voice_worker import VoiceUiBridge
from tldw_chatbook.Chat.console_provider_gateway import ProviderToolCalls
from tldw_chatbook.Chat.console_trace_models import FrozenTracePolicy
from Tests.Chat.test_console_voice_process_gateway import Source


def key(epoch=1):
    return StreamKey(1, "a" * 32, "provider", "turn-a", epoch, epoch)


def wire(key, op, sequence=1, payload=b"", **fields):
    if op == "draft":
        fields.setdefault("draft_slot", 0)
    return Record(
        dict(
            version=1,
            generation=key.generation,
            request_id=key.request_id,
            turn_id=key.turn_id,
            revision=key.revision,
            epoch=key.epoch,
            op=op,
            sequence=sequence,
            **fields,
        ),
        payload,
    )


class Harness:
    def __init__(
        self,
        *,
        supervisor=None,
        prepare=None,
        gateway=None,
        promote=None,
        accepted=None,
        is_context_current=None,
    ):
        effects_type = importlib.import_module(
            "tldw_chatbook.Chat.console_voice_process_effects"
        ).ProcessVoiceEffects
        self.prepared = PreparedSpeculativeVoiceAttempt(_request(), _context())
        self.controls = []
        self.current = True
        self.entries = []
        self.outbound = Mailbox("parent_to_child", generation=1, outbound=True)
        self.bridge = VoiceUiBridge(asyncio.get_running_loop())
        harness = self

        class Gateway:
            async def stream_chat(self, resolution, prepared, **kwargs):
                assert prepared is harness.prepared.request.prepared
                harness.entries.append(prepared)
                yield "private response"

        async def default_prepare(**kwargs):
            if kwargs["attempt_epoch"] == self.prepared.request.attempt_epoch:
                return self.prepared
            return replace(
                self.prepared,
                request=replace(
                    self.prepared.request, attempt_epoch=kwargs["attempt_epoch"]
                ),
            )

        def send_control(record):
            encode_record(record, "parent_to_child")
            self.controls.append(record)

        self.effects = effects_type(
            generation=1,
            request_id="a" * 32,
            outbound=self.outbound,
            bridge=self.bridge,
            prepare_attempt=prepare or default_prepare,
            gateway=gateway or Gateway(),
            dispatch_supervisor=supervisor or VoiceDispatchSupervisor(),
            is_current=lambda operation, prepared: self.current,
            is_context_current=is_context_current,
            promote=promote,
            submit_accepted_voice_turn=accepted,
            send_control=send_control,
        )

    async def prepare(self, operation=None):
        return await self.effects.prepare(operation or key(), "private transcript")


@pytest.mark.asyncio
async def test_authority_fence_still_publishes_actual_cleanup_to_live_child():
    harness = Harness()
    await harness.prepare()
    harness.effects.fence()
    await harness.effects.aclose()
    assert [record.header["op"] for record in harness.controls] == [
        "prepared",
        "cleanup",
    ]
    receipt = harness.controls[-1]
    assert receipt.header["outcome"] == "clean"
    assert receipt.header["last_sequence"] == 0
    assert harness.entries == []


@pytest.mark.asyncio
async def test_lost_cleanup_delivery_does_not_rewrite_actual_close_result():
    h = Harness()
    await h.prepare()

    def vanished(_record):
        raise BrokenPipeError("private endpoint")

    h.effects._send_control = vanished
    outcome = await h.effects.cancel(key())
    assert outcome.value == "clean"
    assert h.effects.failure.code == "voice_transport_failed"
    assert h.effects._fenced
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_cancel_cleanup_reports_original_published_provider_boundary():
    harness = Harness()
    ready = await harness.prepare()
    harness.effects.start_attempt(
        key(), ready.header["request_handle"], ready.header["context_handle"]
    )
    await harness.effects.wait_attempt(key())
    await harness.effects.cancel(key())
    cleanup = next(
        record for record in harness.controls if record.header["op"] == "cleanup"
    )
    assert cleanup.header["last_sequence"] == 1
    assert harness.outbound.count == 1  # Receipt must not imply payload consumption.
    await harness.effects.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["provider", "accepted"])
@pytest.mark.parametrize("detached", [False, True])
async def test_pending_cleanup_fences_other_facade_after_observer_and_child_close(
    replacement, detached
):
    from tldw_chatbook.Chat.console_voice_attempts import AttemptCleanupManager
    from tldw_chatbook.Chat.console_voice_process import (
        ConsoleVoiceProcess,
        DeviceLease,
    )

    supervisor = VoiceDispatchSupervisor()
    provider_work = asyncio.get_running_loop().create_future()
    entered, disposition = asyncio.Event(), asyncio.Event()

    class HeldProvider:
        async def stream_chat(self, resolution, prepared, *, signals, **kwargs):
            assert signals.register_provider_work(provider_work, lambda: None)
            entered.set()
            yield "private response"
            await asyncio.Event().wait()

    async def held_disposition(completion, seconds):
        await disposition.wait()
        if detached:
            return False
        await asyncio.shield(completion)
        return True

    first = Harness(gateway=HeldProvider(), supervisor=supervisor)
    first.effects._cleanup = AttemptCleanupManager(
        supervisor, wait_for_exit=held_disposition
    )
    ready = await first.prepare()
    first.effects.start_attempt(
        key(), ready.header["request_handle"], ready.header["context_handle"]
    )
    await entered.wait()
    observer = first.effects.cancel(key())
    with pytest.raises(VoiceDispatchQuarantined):
        supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)
    observer.cancel()
    process = ConsoleVoiceProcess(DeviceLease(), effects=first.effects)

    async def closed_child():
        return None

    process._close = closed_child
    await process.close()
    accepted = []
    other = Harness(
        supervisor=supervisor,
        accepted=lambda transcript, context: (
            accepted.append(context) or "original-turn"
        ),
    )
    ready = await other.prepare()
    try:
        assert supervisor.orphan_count == 0  # Still BEFORE detached disposition.
        assert not provider_work.done()
        if replacement == "provider":
            with pytest.raises(VoiceDispatchQuarantined):
                other.effects.start_attempt(
                    key(),
                    ready.header["request_handle"],
                    ready.header["context_handle"],
                )
        else:
            result = await other.effects.terminal(
                wire(
                    key(),
                    "terminal_propose",
                    payload=b"private transcript",
                    kind="accepted",
                    request_handle=ready.header["request_handle"],
                    context_handle=ready.header["context_handle"],
                    last_sequence=0,
                    boundary_id=None,
                    answer_sha256=hashlib.sha256(b"").hexdigest(),
                    answer_chars=0,
                )
            )
            assert result.header["disposition"] == "failed"
            assert not accepted
        assert not other.entries
        with pytest.raises(VoiceDispatchQuarantined):
            supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)
        assert (
            await supervisor.guarded_dispatch(VoiceDispatchKind.TYPED, lambda: "typed")
            == "typed"
        )
        if not detached:
            provider_work.set_result(None)
        disposition.set()
        await process.wait_effects_closed()
        if detached:
            assert supervisor.orphan_count == 1
            with pytest.raises(VoiceDispatchQuarantined):
                supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)
            provider_work.set_result(None)
        for _ in range(100):
            if not supervisor.is_quarantined:
                break
            await asyncio.sleep(0)
        supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)
        allowed = Harness(supervisor=supervisor)
        permitted = await allowed.prepare()
        allowed.effects.start_attempt(
            key(),
            permitted.header["request_handle"],
            permitted.header["context_handle"],
        )
        await allowed.effects.wait_attempt(key())
        assert allowed.entries == [allowed.prepared.request.prepared]
        await allowed.effects.aclose()
    finally:
        if not provider_work.done():
            provider_work.set_result(None)
        disposition.set()
        await process.wait_effects_closed()
        await other.effects.aclose()


@pytest.mark.asyncio
async def test_issued_handles_retain_original_authority_and_never_serialize_it():
    h = Harness()
    prepared = await h.prepare()
    h.effects.start_attempt(
        key(), prepared.header["request_handle"], prepared.header["context_handle"]
    )
    await h.effects.wait_attempt(key())
    assert h.entries == [h.prepared.request.prepared]
    assert b"private fixture" not in encode_record(prepared, "parent_to_child")
    assert b"Frozen system prompt" not in encode_record(prepared, "parent_to_child")
    with pytest.raises(ProtocolError):
        h.effects.start_attempt(
            key(2), prepared.header["request_handle"], prepared.header["context_handle"]
        )
    with pytest.raises(ProtocolError):
        h.effects.start_attempt(key(), "unknown", prepared.header["context_handle"])
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_tools_emit_only_pending_signal_and_keep_arguments_parent_owned():
    class Gateway:
        async def stream_chat(self, *args, **kwargs):
            yield ProviderToolCalls(
                (
                    {
                        "id": "private-call",
                        "type": "function",
                        "function": {
                            "name": "private-tool",
                            "arguments": "private-arguments",
                        },
                    },
                )
            )
            yield "must not become speech"

    h = Harness(gateway=Gateway())
    prepared = await h.prepare()
    h.effects.start_attempt(
        key(), prepared.header["request_handle"], prepared.header["context_handle"]
    )
    await h.effects.wait_attempt(key())
    pending = h.controls[-1]
    assert pending.header["op"] == "tool_pending"
    assert pending.header["last_sequence"] == 0
    assert not pending.payload
    assert h.outbound.count == 0
    assert b"private-arguments" not in b"".join(
        encode_record(record, "parent_to_child") for record in h.controls
    )
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_received_cancel_precedes_start_in_same_parent_batch():
    h = Harness()
    prepared = await h.prepare()
    start = wire(
        key(),
        "start_attempt",
        request_handle=prepared.header["request_handle"],
        context_handle=prepared.header["context_handle"],
    )
    cancel = wire(key(), "cancel", sequence=2, reason="stop")
    with pytest.raises(ProtocolError):
        await h.effects.receive_batch([start, cancel])
    assert not h.entries
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_received_revocation_precedes_accepted_claim_in_same_parent_batch():
    entered = []
    h = Harness(
        accepted=lambda transcript, context: entered.append(context) or "original-turn"
    )
    ready = await h.prepare()
    terminal = wire(
        key(),
        "terminal_propose",
        payload=b"private transcript",
        kind="accepted",
        request_handle=ready.header["request_handle"],
        context_handle=ready.header["context_handle"],
        last_sequence=0,
        boundary_id=None,
        answer_sha256=hashlib.sha256(b"").hexdigest(),
        answer_chars=0,
    )
    with pytest.raises(ProtocolError):
        await h.effects.receive_batch(
            [terminal, wire(key(), "cancel", sequence=2, reason="stop")]
        )
    assert entered == []
    assert not any(r.header["op"] == "terminal_claim" for r in h.controls)
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_original_generic_gateway_and_ipc_share_eight_slots_cancel_wakes_worker(
    monkeypatch,
):
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )

    source = Source()
    loop = asyncio.get_running_loop()
    schedule = loop.call_soon_threadsafe
    callbacks = {"pending": 0, "maximum": 0, "bytes": 0, "maximum_bytes": 0}

    def tracked(callback, *args, **kwargs):
        if isinstance(getattr(callback, "__self__", None), asyncio.Queue):
            callbacks["pending"] += 1
            size = len(args[0].text.encode())
            callbacks["bytes"] += size
            callbacks["maximum"] = max(callbacks["maximum"], callbacks["pending"])
            callbacks["maximum_bytes"] = max(
                callbacks["maximum_bytes"], callbacks["bytes"]
            )

            def deliver():
                callbacks["pending"] -= 1
                callbacks["bytes"] -= size
                callback(*args)

            return schedule(deliver, **kwargs)
        return schedule(callback, *args, **kwargs)

    monkeypatch.setattr(loop, "call_soon_threadsafe", tracked)
    gateway = ConsoleProviderGateway(
        http_client=object(), chat_api_call_fn=lambda **kwargs: source
    )
    h = Harness(gateway=gateway)
    resolution = ConsoleProviderResolution(
        provider="qwencloud",
        base_url="",
        model="private-fake",
        ready=True,
        execution_key="qwencloud",
    )
    original = gateway.prepare_chat_request(
        resolution, [{"role": "user", "content": "private transcript"}]
    )
    h.prepared = replace(
        h.prepared,
        request=replace(h.prepared.request, resolution=resolution, prepared=original),
    )
    prepared = await h.prepare()
    h.effects.start_attempt(
        key(), prepared.header["request_handle"], prepared.header["context_handle"]
    )
    for _ in range(100):
        if source.pulls >= 8:
            break
        await asyncio.sleep(0.002)
    assert h.outbound.count == 7
    assert source.pulls == 8
    assert callbacks["maximum"] == 1
    assert callbacks["maximum_bytes"] <= 4096
    await asyncio.sleep(0.05)
    assert source.pulls == 8
    await h.effects.cancel(key())
    assert source.closed.is_set()
    assert (
        h.effects.context_for(key(), prepared.header["context_handle"])
        is h.prepared.frozen_session_context
    )
    assert any(
        r.header["op"] == "cleanup" and r.header["outcome"] == "clean"
        for r in h.controls
    )
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_cancelled_cleanup_observer_keeps_context_and_global_orphan_custody():
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.Chat.console_voice_attempts import AttemptCleanupManager

    close_entered, release = threading.Event(), threading.Event()

    class HeldClose(Source):
        def close(self):
            close_entered.set()
            release.wait(2)
            super().close()

    source = HeldClose()
    supervisor = VoiceDispatchSupervisor()
    gateway = ConsoleProviderGateway(
        http_client=object(), chat_api_call_fn=lambda **kwargs: source
    )
    h = Harness(gateway=gateway, supervisor=supervisor)
    resolution = ConsoleProviderResolution(
        provider="qwencloud",
        base_url="",
        model="private-fake",
        ready=True,
        execution_key="qwencloud",
    )
    original = gateway.prepare_chat_request(
        resolution, [{"role": "user", "content": "private transcript"}]
    )
    h.prepared = replace(
        h.prepared,
        request=replace(h.prepared.request, resolution=resolution, prepared=original),
    )

    async def observe_exit(task, seconds):
        await asyncio.sleep(0)
        return task.done()

    h.effects._cleanup = AttemptCleanupManager(supervisor, wait_for_exit=observe_exit)
    prepared = await h.prepare()
    h.effects.start_attempt(
        key(), prepared.header["request_handle"], prepared.header["context_handle"]
    )
    try:
        for _ in range(100):
            if source.pulls >= 8:
                break
            await asyncio.sleep(0.002)
        observer = h.effects.cancel(key())
        observer.cancel()
        assert await asyncio.to_thread(close_entered.wait, 1)
        for _ in range(50):
            if supervisor.is_quarantined:
                break
            await asyncio.sleep(0)
        assert supervisor.is_quarantined
        assert not source.closed.is_set()
        assert (
            h.effects.context_for(key(), prepared.header["context_handle"])
            is h.prepared.frozen_session_context
        )
        other = Harness(supervisor=supervisor)
        ready = await other.prepare()
        with pytest.raises(VoiceDispatchQuarantined):
            other.effects.start_attempt(
                key(), ready.header["request_handle"], ready.header["context_handle"]
            )
        await other.effects.aclose()
    finally:
        release.set()
        await h.effects.aclose()
    for _ in range(100):
        if not supervisor.is_quarantined:
            break
        await asyncio.sleep(0.002)
    assert not supervisor.is_quarantined


@pytest.mark.asyncio
@pytest.mark.parametrize("detached", [False, True])
async def test_failed_accepted_terminal_cannot_retire_independent_pending_cleanup(
    detached,
):
    from tldw_chatbook.Chat.console_voice_attempts import AttemptCleanupManager

    entered, disposition = asyncio.Event(), asyncio.Event()
    provider_work = asyncio.get_running_loop().create_future()

    class HeldProvider:
        async def stream_chat(self, resolution, prepared, *, signals, **kwargs):
            assert signals.register_provider_work(provider_work, lambda: None)
            entered.set()
            await asyncio.Event().wait()
            yield ""

    async def held_cleanup(completion, seconds):
        await disposition.wait()
        if detached:
            return False
        await asyncio.shield(completion)
        return True

    accepted = []
    h = Harness(
        gateway=HeldProvider(),
        accepted=lambda *args: accepted.append(args) or "ordinary-turn",
    )
    h.effects._cleanup = AttemptCleanupManager(
        h.effects._supervisor, wait_for_exit=held_cleanup
    )
    ready = await h.prepare()
    h.effects.start_attempt(
        key(), ready.header["request_handle"], ready.header["context_handle"]
    )
    await entered.wait()
    observer = h.effects.cancel(key())
    observer.cancel()
    cleanup = h.effects.cancel(key())
    try:
        await asyncio.gather(h.effects.wait_attempt(key()), return_exceptions=True)
        result = await h.effects.terminal(
            wire(
                key(),
                "terminal_propose",
                payload=b"private transcript",
                kind="accepted",
                request_handle=ready.header["request_handle"],
                context_handle=ready.header["context_handle"],
                last_sequence=0,
                boundary_id=None,
                answer_sha256=hashlib.sha256(b"").hexdigest(),
                answer_chars=0,
            )
        )
        assert result.header["disposition"] == "failed"
        assert not accepted
        assert not cleanup.done()
        assert not provider_work.done()
        with pytest.raises(ProtocolError):
            h.effects.retire_context(key(), ready.header["context_handle"])
        assert (
            h.effects.context_for(key(), ready.header["context_handle"])
            is h.prepared.frozen_session_context
        )
        if not detached:
            provider_work.set_result(None)
        disposition.set()
        outcome = await cleanup
        assert outcome.value == ("detached" if detached else "clean")
        assert provider_work.done() is (not detached)
        h.effects.retire_context(key(), ready.header["context_handle"])
        with pytest.raises(ProtocolError):
            h.effects.context_for(key(), ready.header["context_handle"])
        assert h.effects._supervisor.is_quarantined is detached
    finally:
        if not provider_work.done():
            provider_work.set_result(None)
        disposition.set()
        await cleanup
        await h.effects.aclose()


@pytest.mark.asyncio
async def test_retirement_requires_actual_cleanup_and_explicit_context_release():
    h = Harness()
    first = await h.prepare()
    second = await h.prepare(key(2))
    with pytest.raises(ProtocolError):
        h.effects.retire_context(key(), first.header["context_handle"])
    await h.effects.cancel(key())
    assert (
        h.effects.context_for(key(), first.header["context_handle"])
        is h.prepared.frozen_session_context
    )
    h.effects.retire_context(key(), first.header["context_handle"])
    with pytest.raises(ProtocolError):
        await h.prepare(key())
    third = await h.prepare(key(3))
    assert third.header["request_handle"] != second.header["request_handle"]
    with pytest.raises(ProtocolError):
        h.effects.context_for(key(), first.header["context_handle"])
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_drafts_and_preparations_share_two_retained_turn_slots():
    h = Harness()
    first = await h.prepare()
    second_key = StreamKey(1, "a" * 32, "provider", "turn-b", 2, 2)
    third_key = StreamKey(1, "a" * 32, "provider", "turn-c", 3, 3)
    h.effects.receive_draft(wire(second_key, "draft", payload=b"distinct next turn"))
    with pytest.raises(ProtocolError):
        await h.prepare(third_key)
    with pytest.raises(ProtocolError):
        h.effects.receive_draft(wire(third_key, "draft", payload=b"third turn"))
    with pytest.raises(ProtocolError):
        h.effects.retire_turn("turn-a")
    await h.effects.cancel(key())
    h.effects.retire_context(key(), first.header["context_handle"])
    h.effects.retire_turn("turn-a")
    await h.prepare(third_key)
    assert h.effects.draft_for("turn-b") == "distinct next turn"
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_four_same_turn_interruptions_retire_only_after_new_context_adoption():
    from tldw_chatbook.Audio.voice_process_io import PipeWriter
    from tldw_chatbook.Audio.voice_process_protocol import ReceiveStream, read_record

    h = Harness()
    transport = io.BytesIO()
    writer = PipeWriter(transport.write, h.outbound, lambda error: None)
    writer.start()
    previous = None
    offset = 0
    try:
        for epoch in range(1, 5):
            ready = await h.prepare(key(epoch))
            if previous is not None:
                assert (
                    h.effects.context_for(key(epoch - 1), previous)
                    is h.prepared.frozen_session_context
                )
            h.effects.start_attempt(
                key(epoch),
                ready.header["request_handle"],
                ready.header["context_handle"],
            )
            if previous is not None:
                with pytest.raises(ProtocolError):
                    h.effects.context_for(key(epoch - 1), previous)
            await h.effects.wait_attempt(key(epoch))
            assert await asyncio.to_thread(writer.wait_written, epoch, 1)
            reader = io.BytesIO(transport.getvalue()[offset:])
            packet = read_record(reader.read, "parent_to_child")
            offset += reader.tell()
            delivery = ReceiveStream(key(epoch)).receive(packet)
            assert delivery.discard(key(epoch)) == (1, 16)
            h.effects.accept_credit(
                Record(
                    dict(
                        version=1,
                        op="credit",
                        generation=1,
                        request_id="a" * 32,
                        sequence=epoch,
                        lane="provider",
                        stream_id="a" * 32,
                        stream_turn="turn-a",
                        stream_revision=epoch,
                        stream_epoch=epoch,
                        stream_phrase=0,
                        ack_sequence=1,
                        ack_bytes=16,
                    )
                )
            )
            await h.effects.cancel(key(epoch))
            previous = ready.header["context_handle"]
        assert (
            h.effects.context_for(key(4), previous) is h.prepared.frozen_session_context
        )
    finally:
        writer.close()
        await asyncio.to_thread(writer.join, 1)
        await h.effects.aclose()


@pytest.mark.asyncio
async def test_process_close_fences_effects_and_retains_parent_cleanup_separately():
    from tldw_chatbook.Chat.console_voice_process import (
        ConsoleVoiceProcess,
        DeviceLease,
    )

    h = Harness()
    await h.prepare()
    process = ConsoleVoiceProcess(DeviceLease(), effects=h.effects)

    async def closed_child():
        return None

    process._close = closed_child
    await process.close()
    with pytest.raises(ProtocolError):
        await h.prepare(key(2))
    await process.wait_effects_closed()


@pytest.mark.asyncio
async def test_provider_wire_credit_returns_only_consumed_delivery_then_allows_retirement():
    from tldw_chatbook.Audio.voice_process_io import PipeWriter
    from tldw_chatbook.Audio.voice_process_protocol import ReceiveStream, read_record

    h = Harness()
    ready = await h.prepare()
    h.effects.start_attempt(
        key(), ready.header["request_handle"], ready.header["context_handle"]
    )
    await h.effects.wait_attempt(key())
    credit = Record(
        dict(
            version=1,
            op="credit",
            generation=1,
            request_id="a" * 32,
            sequence=1,
            lane="provider",
            stream_id="a" * 32,
            stream_turn="turn-a",
            stream_revision=1,
            stream_epoch=1,
            stream_phrase=0,
            ack_sequence=1,
            ack_bytes=16,
        )
    )
    with pytest.raises(ProtocolError):
        h.effects.accept_credit(credit)
    transport = io.BytesIO()
    writer = PipeWriter(transport.write, h.outbound, lambda error: None)
    writer.start()
    try:
        assert await asyncio.to_thread(writer.wait_written, 1, 1)
        packet = read_record(io.BytesIO(transport.getvalue()).read, "parent_to_child")
        receiver = ReceiveStream(key())
        delivery = receiver.receive(packet)
        assert delivery.consume() == (1, 16)
        h.effects.accept_credit(credit)
        with pytest.raises(ProtocolError):
            h.effects.accept_credit(credit)
        await h.effects.cancel(key())
        h.effects.retire_context(key(), ready.header["context_handle"])
    finally:
        writer.close()
        await asyncio.to_thread(writer.join, 1)
        assert not writer.alive
        await h.effects.aclose()


@pytest.mark.asyncio
async def test_stale_before_start_abandons_original_gateway_trace_capability():
    from Tests.Chat.test_console_voice_promotion import _post_dispatch_call
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway
    from tldw_chatbook.Chat.console_voice_trace_gateway import (
        ProvisionalTraceRegistry,
        ProvisionalTraceUnavailable,
    )
    from tldw_chatbook.Chat.console_exchange_capture import (
        freeze_provisional_capture_eligibility,
    )

    registry = ProvisionalTraceRegistry()
    promotion_id = str(uuid4())
    trace = registry.begin_attempt(
        policy=FrozenTracePolicy(str(uuid4()), "credentials-v1", False, None),
        promotion_id=promotion_id,
        attempt_id=str(uuid4()),
        eligibility=freeze_provisional_capture_eligibility(
            capture_enabled=True, session_is_saved=True
        ),
    )
    registry._retain_gateway_call(trace, _post_dispatch_call(promotion_id))
    gateway = ConsoleProviderGateway(
        http_client=object(), provisional_trace_registry=registry
    )
    h = Harness(gateway=gateway)
    h.prepared = replace(
        h.prepared, request=replace(h.prepared.request, provisional_trace_attempt=trace)
    )
    ready = await h.prepare()
    assert h.effects._attempts[key()].prepared is h.prepared
    assert (
        h.effects._attempts[key()].prepared.request.provisional_trace_attempt is trace
    )
    h.current = False
    with pytest.raises(ProtocolError):
        h.effects.start_attempt(
            key(), ready.header["request_handle"], ready.header["context_handle"]
        )
    with pytest.raises(ProvisionalTraceUnavailable):
        registry.seal_attempt(trace, expected_call_count=1)
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_cancelled_preparation_observer_cannot_release_slot_before_actual_receipt():
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    h = None

    async def prepare(**kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return h.prepared

    h = Harness(prepare=prepare)
    preparing = asyncio.create_task(h.prepare())
    await entered.wait()
    cleanup = asyncio.ensure_future(h.effects.cancel(key()))
    await cancelled.wait()
    await asyncio.sleep(0)
    assert not cleanup.done()
    cleanup.cancel()
    release.set()
    await asyncio.gather(preparing, cleanup, return_exceptions=True)
    await h.effects.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("result", [True, "private forged assistant", object()])
async def test_unclaimed_lookalike_success_fails_and_next_turn_draft_survives(result):
    h = Harness(promote=lambda **kwargs: result)
    prepared = await h.prepare()
    h.effects.start_attempt(
        key(), prepared.header["request_handle"], prepared.header["context_handle"]
    )
    await h.effects.wait_attempt(key())
    next_key = StreamKey(1, "a" * 32, "provider", "turn-b", 2, 2)
    h.effects.receive_draft(wire(next_key, "draft", payload=b"distinct next turn"))
    proposal = wire(
        key(),
        "terminal_propose",
        payload=b"private transcript",
        kind="promote",
        request_handle=prepared.header["request_handle"],
        context_handle=prepared.header["context_handle"],
        last_sequence=1,
        boundary_id="b" * 32,
        answer_sha256=hashlib.sha256(b"private response").hexdigest(),
        answer_chars=16,
    )
    outcome = await h.effects.terminal(proposal)
    assert outcome.header["disposition"] == "failed"
    assert not any(r.header["op"] == "terminal_claim" for r in h.controls)
    assert h.effects.draft_for("turn-b") == "distinct next turn"
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_two_preparations_bound_retention_even_while_owner_awaits():
    entered = asyncio.Event()
    release = asyncio.Event()
    h = None

    async def prepare(**kwargs):
        entered.set()
        await release.wait()
        return replace(
            h.prepared,
            request=replace(h.prepared.request, attempt_epoch=kwargs["attempt_epoch"]),
        )

    h = Harness(prepare=prepare)
    first = asyncio.create_task(h.prepare(key()))
    await entered.wait()
    second = asyncio.create_task(h.prepare(key(2)))
    await asyncio.sleep(0)
    with pytest.raises(ProtocolError):
        await h.prepare(key(3))
    h.current = False
    release.set()
    results = await asyncio.gather(first, second, return_exceptions=True)
    assert all(isinstance(result, ProtocolError) for result in results)
    assert len(h.controls) == 2
    assert all(
        record.header["op"] == "cleanup" and record.header["last_sequence"] == 0
        for record in h.controls
    )
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_cancel_before_entry_keeps_separate_context_for_accepted_handoff():
    accepted = []

    def handoff(transcript, context):
        accepted.append((transcript, context))
        return "ordinary-owned-turn"

    h = Harness(
        accepted=handoff,
        is_context_current=lambda operation, prepared: operation.revision == 4,
    )
    prepared = await h.prepare()
    await h.effects.cancel(key())
    assert (
        h.effects.context_for(key(), prepared.header["context_handle"])
        is h.prepared.frozen_session_context
    )
    with pytest.raises(ProtocolError):
        h.effects.start_attempt(
            key(), prepared.header["request_handle"], prepared.header["context_handle"]
        )
    assert not h.entries
    result = await h.effects.terminal(
        wire(
            replace(key(), revision=4),
            "terminal_propose",
            payload=b"private transcript with stable continuation",
            kind="accepted",
            request_handle=prepared.header["request_handle"],
            context_handle=prepared.header["context_handle"],
            last_sequence=0,
            boundary_id=None,
            answer_sha256=hashlib.sha256(b"").hexdigest(),
            answer_chars=0,
        )
    )
    assert result.header["disposition"] == "promoted"
    assert result.header["revision"] == 4
    assert (
        next(r for r in h.controls if r.header["op"] == "terminal_claim").header[
            "revision"
        ]
        == 4
    )
    assert key() in h.effects._attempts
    assert accepted == [
        (
            "private transcript with stable continuation",
            h.prepared.frozen_session_context,
        )
    ]
    await h.effects.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", ["view", "configuration", "activation", "revision"])
async def test_revised_accepted_handoff_rechecks_parent_context_lifetime(stale):
    lifetime = dict(view=True, configuration=True, activation=True, revision=4)
    entered = []

    def context_current(operation, prepared):
        assert prepared is h.prepared
        return (
            lifetime["view"]
            and lifetime["configuration"]
            and lifetime["activation"]
            and operation.revision == lifetime["revision"]
        )

    h = Harness(
        accepted=lambda transcript, context: (
            entered.append(context) or "accepted-original"
        ),
        is_context_current=context_current,
    )
    prepared = await h.prepare()
    await h.effects.cancel(key())
    lifetime[stale] = 5 if stale == "revision" else False
    proposal = wire(
        replace(key(), revision=4),
        "terminal_propose",
        payload=b"latest stable transcript",
        kind="accepted",
        request_handle=prepared.header["request_handle"],
        context_handle=prepared.header["context_handle"],
        last_sequence=0,
        boundary_id=None,
        answer_sha256=hashlib.sha256(b"").hexdigest(),
        answer_chars=0,
    )
    with pytest.raises(ProtocolError):
        await h.effects.terminal(proposal)
    assert entered == []
    await h.effects.aclose()


@pytest.mark.asyncio
async def test_global_quarantine_blocks_two_view_facades():
    supervisor = VoiceDispatchSupervisor()
    release = asyncio.get_running_loop().create_future()
    supervisor.retain_orphan(release)
    for _ in range(2):
        h = Harness(supervisor=supervisor)
        prepared = await h.prepare()
        with pytest.raises(VoiceDispatchQuarantined):
            h.effects.start_attempt(
                key(),
                prepared.header["request_handle"],
                prepared.header["context_handle"],
            )
        assert not h.entries
        await h.effects.aclose()
    release.set_result(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("fence_bridge", [False, True])
async def test_promotion_claim_survives_child_death_and_rejects_answer_substitution(
    fence_bridge,
):
    from types import SimpleNamespace
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_voice_process import ConsoleVoiceProcess

    store, _, context, _, registry, _, events = _winning_case()
    release = asyncio.Event()

    async def delayed(call):
        await release.wait()
        return call()

    owner = VoicePromotionOwner(lambda: store, sync_runner=delayed)
    winner = VoiceWinningPromotion(owner, trace_registry=registry)
    h = None

    def promote(*, prepared, snapshot, transcript, assistant_text, on_claim):
        assert prepared is h.prepared

        def claimed(claim):
            on_claim(claim)
            if fence_bridge:
                h.bridge.fence()

        return winner.promote(
            replace(context, user_text=transcript, assistant_text=assistant_text),
            snapshot,
            on_claim=claimed,
        )

    h = Harness(promote=promote)
    prepared = await h.prepare()
    h.effects.start_attempt(
        key(), prepared.header["request_handle"], prepared.header["context_handle"]
    )
    await h.effects.wait_attempt(key())
    proposal = wire(
        key(),
        "terminal_propose",
        payload=b"private transcript",
        kind="promote",
        request_handle=prepared.header["request_handle"],
        context_handle=prepared.header["context_handle"],
        last_sequence=1,
        boundary_id="b" * 32,
        answer_sha256=hashlib.sha256(b"private response").hexdigest(),
        answer_chars=16,
    )
    with pytest.raises(ProtocolError):
        await h.effects.terminal(
            Record(dict(proposal.header, answer_chars=17), proposal.payload)
        )
    observer = asyncio.create_task(h.effects.terminal(proposal))
    for _ in range(30):
        if any(record.header["op"] == "terminal_claim" for record in h.controls):
            break
        await asyncio.sleep(0)
    assert any(record.header["op"] == "terminal_claim" for record in h.controls)
    assert events == []
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    runtime = ConsoleRuntime(SimpleNamespace())
    runtime._chat_store = store
    process = ConsoleVoiceProcess(
        runtime.voice_process_supervisor.device_lease, effects=h.effects
    )
    runtime.voice_process_supervisor.retain(process)
    quitting = asyncio.create_task(runtime.dispose(timeout_seconds=0.01))
    try:
        await asyncio.sleep(0.03)
        assert process.close_task.done()  # Child absence cannot release parent claim.
        assert not quitting.done()
        assert events == []
    finally:
        release.set()
        await quitting
    assert events == ["pair"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_provider_terminal_keeps_sequence_barrier_when_ipc_data_lane_is_full(
    failure,
):
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.Audio.voice_process_io import PipeWriter
    from tldw_chatbook.Audio.voice_process_protocol import ReceiveStream, read_record

    class SevenThenTerminal(Source):
        def __next__(self):
            if self.pulls == 7:
                if failure:
                    raise ValueError("private provider terminal failure")
                raise StopIteration
            return super().__next__()

    source = SevenThenTerminal()
    gateway = ConsoleProviderGateway(
        http_client=object(), chat_api_call_fn=lambda **kwargs: source
    )
    h = Harness(gateway=gateway)
    resolution = ConsoleProviderResolution(
        provider="qwencloud",
        base_url="",
        model="private-fake",
        ready=True,
        execution_key="qwencloud",
    )
    original = gateway.prepare_chat_request(
        resolution, [{"role": "user", "content": "private transcript"}]
    )
    h.prepared = replace(
        h.prepared,
        request=replace(h.prepared.request, resolution=resolution, prepared=original),
    )
    ready = await h.prepare()
    h.effects.start_attempt(
        key(), ready.header["request_handle"], ready.header["context_handle"]
    )
    await h.effects.wait_attempt(key())
    assert source.closed.is_set()
    assert h.outbound.count == 7
    terminal = h.controls[-1]
    assert terminal.header["op"] == ("provider_failure" if failure else "provider_end")
    assert terminal.header["last_sequence"] == 7
    receiver = ReceiveStream(key())
    assert receiver.end(7) is False
    transport = io.BytesIO()
    writer = PipeWriter(transport.write, h.outbound, lambda error: None)
    writer.start()
    try:
        assert await asyncio.to_thread(writer.wait_written, 7, 1)
        reader = io.BytesIO(transport.getvalue())
        deliveries = [
            receiver.receive(read_record(reader.read, "parent_to_child"))
            for _ in range(7)
        ]
        assert receiver.complete is False
        for delivery in deliveries:
            delivery.discard(key())
        assert receiver.complete is True
        assert receiver.ack == (7, 28672)
    finally:
        writer.close()
        await asyncio.to_thread(writer.join, 1)
        await h.effects.aclose()


@pytest.mark.asyncio
async def test_cancelled_terminal_observer_releases_shared_slot_only_after_settlement():
    store, _, context, _, registry, _, events = _winning_case()
    release = asyncio.Event()

    async def delayed(call):
        await release.wait()
        return call()

    winner = VoiceWinningPromotion(
        VoicePromotionOwner(lambda: store, sync_runner=delayed), trace_registry=registry
    )

    def promote(*, prepared, snapshot, transcript, assistant_text, on_claim):
        return winner.promote(
            replace(context, user_text=transcript, assistant_text=assistant_text),
            snapshot,
            on_claim=on_claim,
        )

    h = Harness(
        promote=promote, accepted=lambda transcript, context: "original-ordinary-turn"
    )
    first = await h.prepare()
    h.effects.start_attempt(
        key(), first.header["request_handle"], first.header["context_handle"]
    )
    await h.effects.wait_attempt(key())
    proposal = wire(
        key(),
        "terminal_propose",
        payload=b"private transcript",
        kind="promote",
        request_handle=first.header["request_handle"],
        context_handle=first.header["context_handle"],
        last_sequence=1,
        boundary_id="b" * 32,
        answer_sha256=hashlib.sha256(b"private response").hexdigest(),
        answer_chars=16,
    )
    observer = asyncio.create_task(h.effects.terminal(proposal))
    for _ in range(30):
        if any(r.header["op"] == "terminal_claim" for r in h.controls):
            break
        await asyncio.sleep(0)
    assert any(r.header["op"] == "terminal_claim" for r in h.controls)
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    second_key = StreamKey(1, "a" * 32, "provider", "turn-b", 2, 2)
    second = await h.prepare(second_key)
    next_proposal = wire(
        second_key,
        "terminal_propose",
        payload=b"private transcript",
        kind="accepted",
        request_handle=second.header["request_handle"],
        context_handle=second.header["context_handle"],
        last_sequence=0,
        boundary_id=None,
        answer_sha256=hashlib.sha256(b"").hexdigest(),
        answer_chars=0,
    )
    with pytest.raises(ProtocolError):
        await h.effects.terminal(next_proposal)
    release.set()
    for _ in range(50):
        if h.effects._terminal is None:
            break
        await asyncio.sleep(0)
    assert events == ["pair"]
    assert (await h.effects.terminal(next_proposal)).header["disposition"] == "promoted"
    await h.effects.aclose()
