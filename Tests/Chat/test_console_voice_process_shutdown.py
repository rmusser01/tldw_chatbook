"""App-owned isolated voice closure; all children/devices are private fakes."""

import asyncio
import os
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.console_runtime import ConsoleRuntime


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pending_action",
    ["current", "supersede", "stop", "parent_stop", "parent_supersede", "queued_stop"],
)
async def test_replacement_waits_for_old_published_data_and_parent_credit(
    monkeypatch, pending_action
):
    from contextlib import asynccontextmanager
    from dataclasses import replace
    import threading

    from Tests.Audio.test_voice_process_core import eventually
    from Tests.Chat.test_console_voice_attempts import _request
    from Tests.Chat.test_console_voice_effect_barrier import _context
    from Tests.Chat.test_console_voice_preflight import process_failure_session
    from Tests.Chat.test_console_speculative_voice import _speech, _revision
    from tldw_chatbook.Audio.voice_process_types import ControlKind
    from tldw_chatbook.Audio.voice_process_protocol import ProtocolError
    from Tests.Chat.test_console_voice_process_effects import wire
    from tldw_chatbook.Audio.voice_turn_coordinator import ControlAction
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        PreparedSpeculativeVoiceAttempt,
    )

    @asynccontextmanager
    async def checked_session():
        h = None
        try:
            async with process_failure_session(monkeypatch) as h:
                yield h
        except TimeoutError as error:
            # A transport failure can prevent the helper's expected clean
            # receipt. Surface that original fault instead of its cleanup wait.
            if h is not None and h.parent._failure is not None:
                raise AssertionError(
                    f"parent runtime fault: {h.parent._failure}"
                ) from error
            raise

    async with checked_session() as h:
        held, credits, prepared, starts, promoted = [], [], [], [], []
        release_delta = threading.Event()
        old_text = "published before interruption"

        async def prepare(**kwargs):
            prepared.append(kwargs["attempt_epoch"])
            return PreparedSpeculativeVoiceAttempt(
                replace(_request(), attempt_epoch=kwargs["attempt_epoch"]), _context()
            )

        class Gateway:
            async def stream_chat(self, *args, **kwargs):
                starts.append(h.audio.coordinator.snapshot.current_attempt_epoch)
                if len(starts) == 1:
                    yield old_text
                await asyncio.Future()

        h.parent._effects._prepare_attempt = prepare
        h.parent._effects._gateway = Gateway()
        monkeypatch.setattr(h.child, "promote", lambda **kw: promoted.append(kw))
        take = h.parent.pipe.output.take

        def hold_delta(*, block=False):
            while True:
                if release_delta.is_set() and held:
                    return held.pop()
                record = take(block=block)
                if record is not None and record.header["op"] == "provider_delta":
                    held.append(record)
                    continue
                return record

        monkeypatch.setattr(h.parent.pipe.output, "take", hold_delta)
        accept_credit = h.parent._effects.accept_credit

        def hold_credit(record):
            credits.append(record)

        monkeypatch.setattr(h.parent._effects, "accept_credit", hold_credit)

        async def revise(turn, number):
            frame = _speech(number, started_ns=h.scheduler.now_ns)
            await h.audio.submit(frame)
            revision = _revision(
                turn, number, f"revised words {number}", frame.ended_ns
            )
            h.child.record_revision(revision)
            await h.audio.submit(revision)
            h.scheduler.advance_ms(h.audio.coordinator.response_eagerness_ms)
            await h.audio.coordinator.flush()

        try:
            turn = await h.start()
            await eventually(lambda: bool(held))
            old = next(iter(h.child.attempts.values()))
            old_window = h.parent._effects._attempts[old.key].window
            # Keep the coordinator's original-cleanup observer outstanding while
            # its ordinary one-obsolete allowance dispatches the next revision.
            cleanup_observed, release_cleanup = asyncio.Event(), asyncio.Event()
            cancel = h.child.cancel_attempt

            async def observe_cleanup(epoch):
                outcome = await cancel(epoch)
                if epoch == old.key.epoch:
                    cleanup_observed.set()
                    await release_cleanup.wait()
                return outcome

            monkeypatch.setattr(h.child, "cancel_attempt", observe_cleanup)
            await revise(turn, 2)
            await asyncio.wait_for(cleanup_observed.wait(), 1)
            assert len(h.audio.coordinator._obsolete) == 1
            assert old.cleanup.result().value == "clean" and old.end_sequence == 1
            assert not old.receiver.data_complete and old.receiver.ack == (0, 0)
            assert old_window.outstanding == (1, len(old_text.encode()))
            pending_epoch = h.audio.coordinator.snapshot.current_attempt_epoch
            assert pending_epoch != old.key.epoch
            await asyncio.sleep(0.02)
            assert prepared == [old.key.epoch], (
                "replacement prepared before old data retired"
            )
            assert starts == [old.key.epoch]
            with pytest.raises(ProtocolError):
                h.child.receive(
                    wire(
                        h.child.attempts[pending_epoch].key,
                        "provider_end",
                        last_sequence=0,
                    )
                )
            release_cleanup.set()
            await eventually(lambda: not h.audio.coordinator._obsolete)
            if pending_action == "supersede":
                await revise(turn, 3)
            elif pending_action == "stop":
                await h.audio.submit(ControlAction(ControlKind.STOP))
            expected_epoch = h.audio.coordinator.snapshot.current_attempt_epoch
            # Control/speech admission ran while the writer still owned old data.
            assert old.text == "" and promoted == [] and not h.audio._transport.rendered
            release_delta.set()
            h.parent.pipe.send("cancel", reason="stop", **h.child.fields(old.key))
            await eventually(lambda: bool(credits) or bool(h.faults))
            assert h.faults == [] and h.parent._failure is None
            assert old.receiver.data_complete
            assert old.receiver.ack == (1, len(old_text.encode()))
            assert len(credits) == 1
            receipt = credits[0]
            assert receipt.header["stream_epoch"] == old.key.epoch
            if pending_action != "stop":
                await eventually(lambda: len(prepared) == 2)
                await asyncio.sleep(0.02)
            assert starts == [old.key.epoch], (
                "provider entered before old credit arrived"
            )
            assert h.faults == [] and h.parent._failure is None
            if pending_action == "parent_stop":
                await h.audio.submit(ControlAction(ControlKind.STOP))
                await eventually(
                    lambda: all(
                        s.cancelled for s in h.parent._effects._attempts.values()
                    )
                )
            elif pending_action in {"parent_supersede", "queued_stop"}:
                await revise(turn, 3)
                expected_epoch = h.audio.coordinator.snapshot.current_attempt_epoch
                await eventually(lambda: h.parent._revisions[turn] == 3)
                await asyncio.sleep(0.02)
                assert h.parent._failure is None
                assert len(h.parent._effects._attempts) == 2
                assert len(prepared) == 2
                if pending_action == "queued_stop":
                    await h.audio.submit(ControlAction(ControlKind.STOP))
                    await eventually(
                        lambda: all(s.cleanup.done() for s in h.child.attempts.values())
                    )
            accept_credit(receipt)
            assert old_window.outstanding == (0, 0)
            monkeypatch.setattr(h.parent._effects, "accept_credit", accept_credit)
            if pending_action not in {"stop", "parent_stop", "queued_stop"}:
                await eventually(lambda: len(starts) == 2)
                assert starts == [old.key.epoch, expected_epoch]
                assert prepared == (
                    [old.key.epoch, pending_epoch, expected_epoch]
                    if pending_action == "parent_supersede"
                    else starts
                )
            else:
                await asyncio.sleep(0.02)
                assert starts == [old.key.epoch]
                assert prepared == (
                    [old.key.epoch, pending_epoch]
                    if pending_action in {"parent_stop", "queued_stop"}
                    else [old.key.epoch]
                )
            assert old.text == "" and promoted == [] and not h.audio._transport.rendered
            assert h.faults == [] and h.parent._failure is None
        finally:
            release_delta.set()
            if "release_cleanup" in locals():
                release_cleanup.set()
            if held:
                h.parent.pipe.send("cancel", reason="stop", **h.child.fields(old.key))
            monkeypatch.setattr(h.parent._effects, "accept_credit", accept_credit)
            for receipt in credits:
                if h.parent._effects._attempts.get(old.key, None) is not None:
                    if h.parent._effects._attempts[old.key].window.outstanding != (
                        0,
                        0,
                    ):
                        accept_credit(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("newer_speech", [False, True])
@pytest.mark.parametrize("previous_phrase", [False, True])
async def test_child_local_speech_limit_reaches_text_only_policy(
    monkeypatch, newer_speech, previous_phrase
):
    from dataclasses import replace

    from Tests.Audio.test_voice_process_core import eventually
    from Tests.Chat.test_console_voice_attempts import _request
    from Tests.Chat.test_console_voice_effect_barrier import _context
    from Tests.Chat.test_console_voice_preflight import process_failure_session
    from Tests.Chat.test_console_voice_process_tts import _response
    from Tests.Chat.test_console_speculative_voice import _speech, _revision
    from Tests.Chat.test_console_voice_promotion import _winning_case
    from tldw_chatbook.Chat.console_voice_promotion import (
        VoicePromotionOwner,
        VoiceWinningPromotion,
    )
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        PreparedSpeculativeVoiceAttempt,
    )

    async with process_failure_session(monkeypatch) as h:
        finish, phrase_closed, promoted = asyncio.Event(), asyncio.Event(), []
        entries, boundaries = [], []
        store, _, context, _, registry, _, pair_events = _winning_case()
        winner = VoiceWinningPromotion(
            VoicePromotionOwner(lambda: store), trace_registry=registry
        )
        answer = "[" + "ambiguous" * 1100
        prefix = "Earlier phrase. " if previous_phrase else ""

        async def frames():
            yield b"\x01\x00" * 480

        async def synthesize(**kwargs):
            async def cleanup():
                phrase_closed.set()

            return _response(frames(), cleanup=cleanup)

        async def prepare(**kwargs):
            return PreparedSpeculativeVoiceAttempt(
                replace(_request(), attempt_epoch=kwargs["attempt_epoch"]), _context()
            )

        class Gateway:
            async def stream_chat(self, *args, **kwargs):
                entries.append(h.audio.coordinator.snapshot.current_attempt_epoch)
                if len(entries) > 1:
                    await asyncio.Future()
                if previous_phrase:
                    yield prefix
                    await eventually(
                        lambda: (
                            phrase_closed.is_set()
                            and any(
                                s.speech and not s.speech.synthesis_in_flight
                                for s in h.child.attempts.values()
                            )
                        )
                    )
                yield answer
                await finish.wait()

        def promote(**kwargs):
            promoted.append(kwargs)
            return winner.promote(
                replace(
                    context,
                    user_text=kwargs["transcript"],
                    assistant_text=kwargs["assistant_text"],
                    capture_eligible_at_dispatch=False,
                ),
                kwargs["snapshot"],
                on_claim=kwargs["on_claim"],
            )

        child_promote = h.child.promote

        def observe_boundary(**kwargs):
            boundaries.append(kwargs["terminal_boundary_ns"])
            return child_promote(**kwargs)

        h.parent._effects._prepare_attempt = prepare
        h.parent._effects._gateway = Gateway()
        h.parent._effects._promote = promote
        h.parent._tts._synthesize = synthesize
        monkeypatch.setattr(h.child, "promote", observe_boundary)
        turn = await h.start()
        await eventually(
            lambda: any(
                s.speech and s.speech.failure_code for s in h.child.attempts.values()
            )
        )
        state = next(iter(h.child.attempts.values()))
        assert state.speech.failure_code == "speech_input_limit"
        assert h.faults == [], "local settled speech failure stopped the session"
        await eventually(
            lambda: h.audio.coordinator.snapshot.state.value == "responding_text_only"
        )
        assert promoted == [] and not h.child.phrases
        assert len(h.audio._transport.rendered) == int(previous_phrase)
        if newer_speech:
            frame = _speech(2, started_ns=h.scheduler.now_ns)
            await h.audio.submit(frame)
            revision = _revision(turn, 2, "newer speech", frame.ended_ns)
            h.child.record_revision(revision)
            await h.audio.submit(revision)
        finish.set()
        if newer_speech:
            await asyncio.wait_for(asyncio.shield(state.cleanup), 1)
            await h.audio.coordinator.flush()
            assert promoted == []
        else:
            await eventually(lambda: bool(promoted))
            assert len(promoted) == 1
            assert promoted[0]["assistant_text"] == prefix + answer
            assert boundaries == [None]
            await eventually(lambda: not h.child.attempts and not h.parent._contexts)
            assert state.terminal.result().value == "promoted" and state.claimed
            assert pair_events.count("pair") == 1
            frame = _speech(3, started_ns=h.scheduler.now_ns)
            await h.audio.submit(frame)
            next_turn = h.audio.coordinator.snapshot.turn_id
            revision = _revision(next_turn, 1, "another question", frame.ended_ns)
            h.child.record_revision(revision)
            await h.audio.submit(revision)
            h.scheduler.advance_ms(h.audio.coordinator.response_eagerness_ms)
            await h.audio.coordinator.flush()
            await eventually(lambda: len(entries) == 2)
            assert next_turn != turn and entries[1] > state.key.epoch
        if not newer_speech:
            assert len(h.audio._transport.rendered) == int(previous_phrase)
        assert h.faults == [] and h.parent._failure is None and not h.child.fenced


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_child_speech_limit_with_unsettled_owner_stays_fatal(
    monkeypatch, cleanup_fails
):
    from dataclasses import replace

    from Tests.Audio.test_voice_process_core import eventually
    from Tests.Chat.test_console_voice_attempts import _request
    from Tests.Chat.test_console_voice_effect_barrier import _context
    from Tests.Chat.test_console_voice_preflight import process_failure_session
    from Tests.Chat.test_console_voice_process_tts import _response
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        PreparedSpeculativeVoiceAttempt,
    )

    async with process_failure_session(monkeypatch) as h:
        closing, release, promoted = asyncio.Event(), asyncio.Event(), []

        async def prepare(**kwargs):
            return PreparedSpeculativeVoiceAttempt(
                replace(_request(), attempt_epoch=kwargs["attempt_epoch"]), _context()
            )

        async def frames():
            yield b"\x01\x00" * 480

        async def cleanup():
            closing.set()
            await release.wait()
            if cleanup_fails:
                raise RuntimeError("private cleanup failure")

        async def synthesize(**kwargs):
            return _response(frames(), cleanup=cleanup)

        class Gateway:
            async def stream_chat(self, *args, **kwargs):
                yield "A spoken phrase. "
                await closing.wait()
                yield "[" + "ambiguous" * 1100
                await asyncio.Future()

        h.parent._effects._prepare_attempt = prepare
        h.parent._effects._gateway = Gateway()
        h.parent._tts._synthesize = synthesize
        monkeypatch.setattr(h.child, "promote", lambda **kw: promoted.append(kw))
        try:
            await h.start()
            await eventually(lambda: bool(h.faults))
            state = next(iter(h.child.attempts.values()))
            assert state.speech.failure_code == "speech_input_limit"
            assert h.faults == ["tts_failed"] and promoted == []
            assert all(
                stream.receiver.cleanup_outcome is None
                for stream in h.child.phrases.values()
            )
            release.set()
            if cleanup_fails:
                await eventually(lambda: h.parent._failure is not None)
                assert h.parent._failure.code == "transport_failed"
            await eventually(lambda: not h.parent._productions)
            assert promoted == []
        finally:
            release.set()


@pytest.mark.asyncio
async def test_accepted_handoff_waits_for_cancelled_provider_final_data(monkeypatch):
    from dataclasses import replace
    import threading

    from Tests.Audio.test_voice_process_core import eventually
    from Tests.Chat.test_console_voice_attempts import _request
    from Tests.Chat.test_console_voice_effect_barrier import _context
    from Tests.Chat.test_console_voice_preflight import process_failure_session
    from Tests.Chat.test_console_speculative_voice import _revision
    from tldw_chatbook.Chat.console_speculative_voice_session import (
        PreparedSpeculativeVoiceAttempt,
    )

    async with process_failure_session(monkeypatch) as h:
        original_context = _context()
        held, proposals, accepted = [], [], []
        release_delta = threading.Event()
        draft_consumed = asyncio.Event()

        async def prepare(**kwargs):
            return PreparedSpeculativeVoiceAttempt(
                replace(_request(), attempt_epoch=kwargs["attempt_epoch"]),
                original_context,
            )

        class Gateway:
            async def stream_chat(self, *args, **kwargs):
                yield "last published old delta"
                await asyncio.Future()

        h.parent._effects._prepare_attempt = prepare
        h.parent._effects._gateway = Gateway()
        h.parent._effects._accepted = lambda text, context: (
            accepted.append((text, context)) or "accepted-turn"
        )
        take = h.parent.pipe.output.take

        def hold_published_delta(*, block=False):
            while True:
                if release_delta.is_set() and held:
                    return held.pop()
                record = take(block=block)
                if (
                    record is not None
                    and record.header["op"] == "provider_delta"
                    and not release_delta.is_set()
                ):
                    # Keep the original mailbox delivery/Reservation owned;
                    # allow priority cleanup records to pass this held frame.
                    held.append(record)
                    continue
                return record

        monkeypatch.setattr(h.parent.pipe.output, "take", hold_published_delta)
        send = h.child.pipe.send

        def observe_proposal(op, **fields):
            if op == "terminal_propose":
                proposals.append(fields)
            return send(op, **fields)

        monkeypatch.setattr(h.child.pipe, "send", observe_proposal)
        consume_draft = h.child.pipe.wait_draft_consumed

        async def observe_draft_consumption(turn):
            await consume_draft(turn)
            draft_consumed.set()

        monkeypatch.setattr(
            h.child.pipe, "wait_draft_consumed", observe_draft_consumption
        )
        try:
            turn = await h.start()
            await eventually(lambda: bool(held))
            state = next(iter(h.child.attempts.values()))
            # Original cleanup must return promptly despite the held published
            # frame; it supplies an end boundary, not a consumed-data receipt.
            outcome = await asyncio.wait_for(h.child.cancel_attempt(state.key.epoch), 1)
            assert outcome.value == "clean"
            assert state.cleanup.done() and state.end_sequence == 1
            assert state.receiver.ack[0] == 0 and not state.receiver.data_complete
            revision = _revision(turn, 2, "accepted revised words", 10**18)
            h.child.record_revision(revision)
            await h.audio.submit(revision)
            h.child.submit_accepted_voice_turn(
                "accepted revised words", h.audio.coordinator._turn_context_handle
            )
            await asyncio.wait_for(draft_consumed.wait(), 1)
            assert proposals == [] and accepted == []
            assert (
                h.parent._effects.context_for(
                    state.key, state.prepared.header["context_handle"]
                )
                is original_context
            )

            release_delta.set()
            # Wake the existing writer with an idempotent exact old fence.
            h.parent.pipe.send(
                "cancel",
                reason="stop",
                turn_id=turn,
                revision=state.key.revision,
                epoch=state.key.epoch,
            )
            await eventually(lambda: bool(accepted) or h.parent._failure is not None)
            assert h.parent._failure is None and h.faults == []
            assert accepted == [("accepted revised words", original_context)]
            assert len(proposals) == 1 and proposals[0]["last_sequence"] == 1
            assert state.receiver.data_complete and state.receiver.ack[0] == 1
            await eventually(lambda: not h.parent._contexts)
        finally:
            release_delta.set()
            if held:
                h.parent.pipe.send(
                    "cancel",
                    reason="stop",
                    turn_id=turn,
                    revision=state.key.revision,
                    epoch=state.key.epoch,
                )


def test_lost_recovery_revoke_cannot_interrupt_synchronous_close_fence():
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_process_protocol import ProtocolError, StreamKey

    faults = []

    def lost(*args, **kwargs):
        raise ProtocolError("voice_transport_closed")

    child = _ChildEffects(
        SimpleNamespace(generation=1, request_id="a" * 32, send=lost),
        None,
        lambda: SimpleNamespace(coordinator=SimpleNamespace(snapshot=None)),
        faults.append,
    )
    key = StreamKey(1, "a" * 32, "provider", "turn", 1, 1)
    child.recoveries["turn"] = (key, 1, True)
    child.fence_all()
    assert child.fenced and child.recoveries == {}
    assert faults == ["transport_failed"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "terminal", ["provider_end", "provider_failure", "tool_pending"]
)
async def test_actual_pipe_provider_terminal_waits_last_published_delta(terminal):
    from Tests.Audio.test_voice_process_core import eventually
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_process_lifetime import LifecyclePipe
    from tldw_chatbook.Audio.voice_process_protocol import CreditWindow, Record

    submitted, spoken, faults = [], [], []

    async def submit(event):
        submitted.append(event)

    async def feed(epoch, text):
        spoken.append(text)

    async def finish(epoch):
        spoken.append("EOF")

    audio = SimpleNamespace(
        submit=submit,
        coordinator=SimpleNamespace(snapshot=SimpleNamespace(revision_id=1)),
    )
    child_read, parent_write = os.pipe()
    parent_read, child_write = os.pipe()
    parent = LifecyclePipe(
        parent_read,
        parent_write,
        generation=1,
        request_id="a" * 32,
        parent=True,
        consume=lambda _: None,
        fault=faults.append,
    )
    child = None
    pipe = LifecyclePipe(
        child_read,
        child_write,
        generation=1,
        request_id="a" * 32,
        parent=False,
        consume=lambda record: child.receive(record),
        fault=faults.append,
    )
    child = _ChildEffects(pipe, SimpleNamespace(), lambda: audio, faults.append)
    child.dispatch_attempt(turn_id="turn", attempt_epoch=1, transcript="user")
    state = child.attempts[1]
    state.speech = SimpleNamespace(feed=feed, finish=finish)
    fields = dict(turn_id="turn", revision=1, epoch=1)
    parent.output.retain_turn("turn")
    parent.output.open_stream(state.key)
    window = CreditWindow(state.key)
    permit = window.try_reserve()
    delta = Record(
        dict(
            version=1,
            generation=1,
            request_id="a" * 32,
            sequence=permit.sequence,
            op="provider_delta",
            **fields,
        ),
        b"last words",
    )
    permit.publish(delta)
    try:
        parent.send(
            terminal,
            last_sequence=1,
            **({"code": "provider_failed"} if terminal == "provider_failure" else {}),
            **fields,
        )
        await eventually(lambda: state.provider_end_seen)
        await asyncio.sleep(0.01)
        assert submitted == [] and spoken == []
        parent.output.put(delta)
        await eventually(lambda: len(submitted) == 2)
        assert type(submitted[0]).__name__ == "AttemptOutputDelta"
        assert (
            type(submitted[1]).__name__
            == {
                "provider_end": "AttemptGenerationCompleted",
                "provider_failure": "ProviderAttemptFailed",
                "tool_pending": "AttemptToolPending",
            }[terminal]
        )
        assert state.text == "last words" and state.receiver.data_complete
        assert spoken == (
            ["last words", "EOF"] if terminal == "provider_end" else ["last words"]
        )
        assert faults == []
    finally:
        await asyncio.gather(*tuple(child.tasks), return_exceptions=True)
        parent.stop()
        pipe.stop()
        os.close(parent_write)
        os.close(child_write)
        assert await parent.join()
        assert await pipe.join()


@pytest.mark.asyncio
async def test_failed_terminal_keeps_bounded_draft_slot_until_recovery_really_returns():
    from tldw_chatbook.Chat.console_voice_process import (
        ConsoleVoiceProcess,
        DeviceLease,
    )

    release, retired = asyncio.Event(), []

    async def preserve(text):
        await release.wait()

    parent = ConsoleVoiceProcess(DeviceLease(), preserve_draft=preserve)
    parent._effects = SimpleNamespace(
        draft_for=lambda _: "recoverable", retire_turn=retired.append
    )
    parent.pipe = SimpleNamespace(retire_turn=lambda _: None)
    parent._terminals["turn"] = "failed"
    parent._revisions["turn"] = 1
    parent._preserve_recovery("turn")
    try:
        parent._retire_terminal_contexts()
        assert retired == []
    finally:
        release.set()
        await asyncio.gather(*tuple(parent._parent_tasks))
    parent._retire_terminal_contexts()
    assert retired


@pytest.mark.asyncio
async def test_cancelled_phrase_retains_bounded_identity_for_late_normal_end():
    from Tests.Chat.test_console_voice_process_tts import _record, _key, _cleanup_key
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_phrase_sequencer import ProcessPcmStream
    from tldw_chatbook.Audio.voice_process_protocol import (
        Mailbox,
        ReceiveStream,
        ProtocolError,
        StreamKey,
    )

    inbox = Mailbox("parent_to_child", generation=1)
    pipe = SimpleNamespace(
        generation=1, request_id="a" * 32, input=inbox, credit=lambda *args: None
    )
    effects = _ChildEffects(pipe, None, lambda: None, lambda _: None)
    stream = ProcessPcmStream(
        _key(),
        inbox,
        cleanup_receiver=ReceiveStream(_cleanup_key()),
        on_credit=pipe.credit,
        on_cancel=lambda _: None,
    )
    effects.phrases[_key()] = stream
    stream.activate()

    def deliver(record):
        inbox.put(record)
        value = inbox.take()
        try:
            if effects.receive(value) is not True:
                inbox.release(value)
        except ProtocolError:
            inbox.release(value)
            raise

    # An early cleanup gives a boundary, but is not a normal playback EOF.
    deliver(_record(_key(), "tts_closed", outcome="clean", last_sequence=1))
    assert not stream.receiver.data_complete
    stream.fence()
    deliver(_record(_key(), "pcm", payload=b"\x01\x00" * 480))
    assert effects.phrases == {}
    assert len(effects._retired_phrases) == 1
    inbox.open_stream(StreamKey(1, "a" * 32, "pcm", "voice-turn-1", 2, 2, 1))
    deliver(_record(_key(), "pcm_end", last_sequence=1))
    with pytest.raises(ProtocolError):
        deliver(_record(_key(), "pcm_end", sequence=2, last_sequence=2))
    assert len(effects._retired_phrases) == 1


@pytest.mark.asyncio
async def test_turn_retirement_reuses_actual_draft_credit_slots_and_receiver_state():
    from tldw_chatbook.Audio.voice_process_lifetime import LifecyclePipe

    child_read, parent_write = os.pipe()
    parent_read, child_write = os.pipe()
    faults = []
    parent = LifecyclePipe(
        parent_read,
        parent_write,
        generation=1,
        request_id="a" * 32,
        parent=True,
        consume=lambda _: None,
        fault=faults.append,
    )
    child = LifecyclePipe(
        child_read,
        child_write,
        generation=1,
        request_id="a" * 32,
        parent=False,
        consume=lambda _: None,
        fault=faults.append,
    )
    try:
        for index in range(4):
            turn = f"turn-{index}"
            child.send(
                "draft",
                payload=b"bounded user words",
                turn_id=turn,
                revision=1,
                epoch=index + 1,
            )
            await asyncio.wait_for(child.wait_draft_consumed(turn), 1)
            parent.retire_turn(turn)
            child.retire_turn(turn)
            assert not parent._receivers
            assert len(child._windows) == 1  # Session control only.
        assert faults == []
    finally:
        parent.stop()
        child.stop()
        os.close(parent_write)
        os.close(child_write)
        assert await parent.join()
        assert await child.join()


@pytest.mark.asyncio
async def test_child_preview_coalesces_latest_cumulative_text_until_credit_returns():
    from Tests.Audio.test_voice_process_core import eventually
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_process_protocol import Mailbox, ProtocolError

    busy, previews = True, []

    def send(op, **fields):
        if op == "preview":
            if busy:
                raise ProtocolError("voice_capacity_exceeded")
            previews.append(fields["payload"])

    pipe = SimpleNamespace(
        generation=1,
        request_id="a" * 32,
        input=Mailbox("parent_to_child", generation=1),
        send=send,
    )
    effects = _ChildEffects(
        pipe,
        None,
        lambda: SimpleNamespace(
            coordinator=SimpleNamespace(snapshot=SimpleNamespace(revision_id=1))
        ),
        lambda _: None,
    )
    effects.dispatch_attempt(turn_id="turn", attempt_epoch=1, transcript="user")
    state = effects.attempts[1]
    state.text = "first"
    effects.publish_preview(1, "first")
    state.text = "first plus second"
    effects.publish_preview(1, " plus second")
    await asyncio.sleep(0.01)
    assert previews == []
    busy = False
    try:
        await eventually(lambda: bool(previews))
        assert previews == [b"first plus second"]
    finally:
        for task in tuple(effects.tasks):
            task.cancel()
        await asyncio.gather(*tuple(effects.tasks), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["provider_end", "provider_failure", "tool_pending"])
async def test_retired_never_issued_proposal_rejects_provider_terminal(op):
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_process_protocol import (
        Mailbox,
        ProtocolError,
        Record,
    )

    sent, faults = [], []
    inbox = Mailbox("parent_to_child", generation=1)
    snapshot = SimpleNamespace(revision_id=1, current_attempt_epoch=1, turn_id="turn")
    effects = _ChildEffects(
        SimpleNamespace(
            generation=1,
            request_id="a" * 32,
            input=inbox,
            send=lambda op, **fields: sent.append((op, fields)),
            credit=lambda *args: None,
        ),
        SimpleNamespace(fence_output=lambda: None),
        lambda: SimpleNamespace(coordinator=SimpleNamespace(snapshot=snapshot)),
        faults.append,
    )
    try:
        effects.dispatch_attempt(turn_id="turn", attempt_epoch=1, transcript="old")
        snapshot.current_attempt_epoch = 2
        effects.dispatch_attempt(turn_id="turn", attempt_epoch=2, transcript="pending")
        pending = effects.attempts[2]
        assert not pending.admitted
        effects.fence_attempt(2)
        snapshot.current_attempt_epoch = 3
        effects.dispatch_attempt(turn_id="turn", attempt_epoch=3, transcript="revised")
        assert sorted(effects.attempts) == [1, 3]
        assert [fields["epoch"] for op, fields in sent if op == "prepare"] == [1]

        fields = {"code": "provider_failed"} if op == "provider_failure" else {}
        # Valid wire shape must not authorize a never-issued, now-retired epoch.
        inbox.put(
            Record(
                dict(
                    version=1,
                    generation=1,
                    request_id="a" * 32,
                    turn_id="turn",
                    revision=1,
                    epoch=2,
                    sequence=1,
                    op=op,
                    last_sequence=0,
                    **fields,
                )
            )
        )
        record = inbox.take()
        try:
            with pytest.raises(ProtocolError, match="voice_protocol_invalid"):
                effects.receive(record)
        finally:
            inbox.release(record)
        assert sorted(effects.attempts) == [1, 3]
        assert faults == []
    finally:
        tasks = tuple(effects.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["provider_end", "provider_failure", "tool_pending"])
@pytest.mark.parametrize("last_sequence", [0, 1])
async def test_cancel_cleanup_cannot_retire_unread_provider_data(last_sequence, op):
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_process_protocol import (
        Mailbox,
        Record,
        ProtocolError,
    )

    credits, sent = [], []
    pipe = SimpleNamespace(
        generation=1,
        request_id="a" * 32,
        input=Mailbox("parent_to_child", generation=1),
        send=lambda *args, **kw: sent.append((args, kw)),
        credit=lambda *args: credits.append(args),
    )
    effects = _ChildEffects(
        pipe,
        SimpleNamespace(fence_output=lambda: None),
        lambda: SimpleNamespace(
            coordinator=SimpleNamespace(snapshot=SimpleNamespace(revision_id=1))
        ),
        lambda _: None,
    )
    effects.dispatch_attempt(turn_id="turn", attempt_epoch=1, transcript="user")
    state = effects.attempts[1]
    state.superseded = True
    effects.fence_attempt(1)

    def deliver(op, payload=b"", **fields):
        record = Record(
            dict(
                version=1,
                generation=1,
                request_id="a" * 32,
                turn_id="turn",
                revision=1,
                epoch=1,
                sequence=1,
                op=op,
                **fields,
            ),
            payload,
        )
        pipe.input.put(record)
        record = pipe.input.take()
        retained = effects.receive(record)
        if retained is not True:
            pipe.input.release(record)

    deliver("cleanup", outcome="clean", last_sequence=last_sequence)
    effects._retire_cancelled()
    if last_sequence:
        assert effects.attempts[1] is state
        deliver("provider_delta", b"late published text")
        assert state.receiver.data_complete and credits[-1][1] == 1
        effects._retire_cancelled()
    assert effects.attempts == {}
    fields = {"code": "provider_failed"} if op == "provider_failure" else {}
    late_end = Record(
        dict(
            version=1,
            generation=1,
            request_id="a" * 32,
            turn_id="turn",
            revision=1,
            epoch=1,
            sequence=1,
            op=op,
            last_sequence=last_sequence,
            **fields,
        )
    )
    pipe.input.put(late_end)
    assert pipe.input.take() is late_end
    try:
        effects.receive(late_end)
    finally:
        pipe.input.release(late_end)
    with pytest.raises(ProtocolError):
        effects.receive(late_end)
    effects._end_provider(state, last_sequence)  # Consistent prior EOF is harmless.
    with pytest.raises(ProtocolError):
        effects._end_provider(state, last_sequence + 1)


@pytest.mark.asyncio
async def test_cancelled_pcm_has_exact_end_boundary_before_phrase_retirement():
    from Tests.Chat.test_console_voice_process_tts import _Link, _response, _until

    async def frames():
        yield b"\x01\x00" * 480
        await asyncio.Future()

    async def synthesize(**kwargs):
        return _response(frames())

    link = _Link(SimpleNamespace(loop=asyncio.get_running_loop()), synthesize)
    receiver, production = link.start()
    try:
        await _until(lambda: any(header["op"] == "pcm" for header, _ in link.records))
        receiver.fence()
        await production.wait_for_cleanup()
        await _until(lambda: receiver.receiver.cleanup_outcome is not None)
        assert receiver.receiver.data_complete
        assert link.producer.outstanding == (0, 0)
    finally:
        await link.close()


@pytest.mark.asyncio
async def test_parent_preview_matches_both_lanes_and_revocation_clears_it():
    from tldw_chatbook.Audio.voice_process_protocol import StreamKey
    from tldw_chatbook.Chat.console_voice_process import (
        ConsoleVoiceProcess,
        DeviceLease,
    )

    projections, clears = [], []
    parent = ConsoleVoiceProcess(
        DeviceLease(),
        current=lambda: True,
        project_preview=projections.append,
        clear_preview=lambda: clears.append(True),
    )
    parent._effects = SimpleNamespace(draft_for=lambda _: "current user")
    key = StreamKey(1, "a" * 32, "provider", "turn", 2, 3)
    parent._project(key, "cumulative assistant")
    assert projections == []
    parent._revisions["turn"] = 2
    parent._draft_revisions["turn"] = 2
    parent._project(key)
    assert projections[-1].user_text == "current user"
    assert projections[-1].assistant_text == "cumulative assistant"
    assert projections[-1].status == "responding"
    parent._project(StreamKey(1, "a" * 32, "provider", "turn", 1, 2), "obsolete")
    assert len(projections) == 1
    parent._revoked.add(("turn", 3))
    parent._clear_projection()
    parent._project(key, "late")
    assert len(projections) == 1 and clears == [True]
    assert "current user" not in repr(parent.snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize("claimed", [False, True])
async def test_fatal_notice_and_child_close_do_not_wait_for_draft_preservation(claimed):
    from tldw_chatbook.Chat.console_voice_process import (
        ConsoleVoiceProcess,
        DeviceLease,
    )
    from tldw_chatbook.Audio.voice_process_protocol import ProtocolError

    notices, preserved = [], []
    release = asyncio.Event()

    async def preserve(text):
        preserved.append(text)
        await release.wait()

    class Effects:
        def fence(self):
            pass

        async def aclose(self):
            pass

        def draft_for(self, turn):
            return "unaccepted words"

    parent = ConsoleVoiceProcess(
        DeviceLease(),
        effects=Effects(),
        preserve_draft=preserve,
        on_runtime_failure=notices.append,
    )
    parent._revisions["turn"] = 2
    if claimed:
        parent._claimed.add("turn")
    parent._transport_fault(ProtocolError("voice_transport_failed"))
    try:
        await parent.close()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(notices) == 1
        assert preserved == ([] if claimed else ["unaccepted words"])
        if not claimed:
            observer = asyncio.create_task(parent.wait_effects_closed())
            await asyncio.sleep(0)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
            assert not parent._effects_close_task.done()
        parent._transport_fault()
        await asyncio.sleep(0)
        assert len(notices) == 1
    finally:
        release.set()
        await parent.wait_effects_closed()


@pytest.mark.asyncio
async def test_process_supervisor_fences_synchronously_before_any_quit_wait():
    runtime = ConsoleRuntime(SimpleNamespace())
    supervisor = runtime.voice_process_supervisor
    fenced = []

    class Session:
        def fence_and_close(self, reason):
            fenced.append(reason)
            return asyncio.sleep(0)

        async def wait_effects_closed(self):
            pass

    supervisor.retain(Session())
    supervisor.begin_close()
    assert len(fenced) == 1
    await supervisor.aclose()


@pytest.mark.asyncio
async def test_runtime_quit_keeps_store_and_worker_until_original_orphan_exits():
    runtime = ConsoleRuntime(SimpleNamespace())
    events = []
    survivor = asyncio.get_running_loop().create_future()
    runtime.voice_dispatch_supervisor.retain_orphan(survivor)
    runtime._chat_store = SimpleNamespace(
        sessions=lambda: (), end_app_runtime=lambda: events.append("store")
    )

    class Worker:
        async def aclose(self):
            events.append("worker")

    runtime._voice_worker = Worker()
    owner = asyncio.create_task(runtime.dispose(timeout_seconds=0.01))
    try:
        await asyncio.sleep(0.03)
        assert runtime._disposed
        assert not owner.done()
        assert events == []
        survivor.set_result(None)
        await owner
        assert events == ["worker", "store"]
    finally:
        if not survivor.done():
            survivor.set_result(None)
        await owner


@pytest.mark.asyncio
async def test_ordinary_control_credit_recycles_only_after_actual_consumption():
    from tldw_chatbook.Audio.voice_process_lifetime import LifecyclePipe
    from tldw_chatbook.Audio.voice_process_protocol import ProtocolError

    child_read, parent_write = os.pipe()
    parent_read, child_write = os.pipe()
    retained, faults = [], []
    quit_received = asyncio.Event()

    def consume(record):
        if record.header["op"] == "close":
            quit_received.set()
            return None
        retained.append(record)
        return True

    parent = LifecyclePipe(
        parent_read,
        parent_write,
        generation=1,
        request_id="a" * 32,
        parent=True,
        consume=lambda _: None,
        fault=faults.append,
    )
    child = LifecyclePipe(
        child_read,
        child_write,
        generation=1,
        request_id="a" * 32,
        parent=False,
        consume=consume,
        fault=faults.append,
    )
    try:
        for _ in range(64):
            parent.send(
                "prepared",
                turn_id="b" * 32,
                revision=1,
                epoch=1,
                request_handle="c" * 32,
                context_handle="d" * 32,
                decision="provisional",
            )
        async with asyncio.timeout(2):
            while len(retained) != 64:
                await asyncio.sleep(0.001)
        assert parent.window.outstanding[0] == 64
        with pytest.raises(ProtocolError, match="voice_capacity_exceeded"):
            parent.send(
                "prepared",
                turn_id="b" * 32,
                revision=1,
                epoch=1,
                request_handle="c" * 32,
                context_handle="d" * 32,
                decision="provisional",
            )
        parent.send("close", reason="teardown")
        await asyncio.wait_for(quit_received.wait(), 1)
        assert parent.window.outstanding[0] == 64  # Quit did not invent data credit.
        for record in retained:
            child.release(record)
        async with asyncio.timeout(2):
            while parent.window.outstanding != (0, 0):
                await asyncio.sleep(0.001)
        parent.send(
            "prepared",
            turn_id="b" * 32,
            revision=1,
            epoch=1,
            request_handle="c" * 32,
            context_handle="d" * 32,
            decision="provisional",
        )
        assert faults == []
    finally:
        parent.stop()
        child.stop()
        os.close(parent_write)
        os.close(child_write)
        assert await parent.join()
        assert await child.join()


@pytest.mark.asyncio
async def test_child_claimed_terminal_close_waits_original_result_not_cancel_receipt():
    from tldw_chatbook.Audio.voice_process_entry import _ChildEffects
    from tldw_chatbook.Audio.voice_process_types import VoiceTerminalDisposition

    effects = _ChildEffects(
        SimpleNamespace(generation=1, request_id="a" * 32),
        SimpleNamespace(),
        lambda: None,
        lambda _: None,
    )
    state = SimpleNamespace(
        claimed=True,
        cancelled=False,
        speech=None,
        cleanup=asyncio.get_running_loop().create_future(),
        terminal=asyncio.get_running_loop().create_future(),
    )
    effects.attempts[1] = state
    owner = asyncio.create_task(effects.cancel_attempt(1))
    try:
        await asyncio.sleep(0)
        assert not owner.done()
        state.terminal.set_result(VoiceTerminalDisposition.PROMOTED)
        await asyncio.wait_for(asyncio.shield(owner), 0.1)
        assert not state.cleanup.done()  # Claimed work never fabricated cancellation.
    finally:
        owner.cancel()
        await asyncio.gather(owner, return_exceptions=True)


@pytest.mark.asyncio
async def test_runtime_lazily_shares_device_lease_and_retains_parent_cleanup():
    runtime = ConsoleRuntime(SimpleNamespace())
    supervisor = runtime.voice_process_supervisor
    assert runtime.voice_process_supervisor is supervisor
    entered, release = asyncio.Event(), asyncio.Event()

    class Session:
        def fence_and_close(self, _reason):
            entered.set()
            return asyncio.sleep(0)

        async def wait_effects_closed(self):
            await release.wait()

    session = Session()
    supervisor.retain(session)
    observer = asyncio.create_task(supervisor.aclose())
    await entered.wait()
    observer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await observer
    assert not supervisor.close_task.done()
    release.set()
    await supervisor.aclose()
    assert supervisor.close_task.done()
