"""Private audio entry. No enable flag, app main import, or factory discovery."""

from __future__ import annotations

import os
import stat
import sys


def isolate_standard_streams() -> tuple[int, int]:
    """Privatize fd0/1 and silence fd0/1/2 before later audio imports.

    Non-inheritable duplicates do not cross exec/spawn into model workers.
    Original descriptors become null, including ordinary library stdout/stderr.
    """
    read_fd = write_fd = None
    try:
        if not stat.S_ISFIFO(os.fstat(0).st_mode) or not stat.S_ISFIFO(
            os.fstat(1).st_mode
        ):
            raise OSError("private pipes required")
        read_fd, write_fd = os.dup(0), os.dup(1)
        os.set_inheritable(read_fd, False)
        os.set_inheritable(write_fd, False)
        with open(os.devnull, "r+b", buffering=0) as null:
            for target in (0, 1, 2):
                os.dup2(null.fileno(), target)
        return read_fd, write_fd
    except Exception:
        for fd in (read_fd, write_fd):
            if fd is not None:
                os.close(fd)
        raise


def _production_session(bootstrap):
    return _ProductionSession(bootstrap)


class _ProductionSession:
    """Audio/model composition with no app, provider, or persistence authority."""

    def __init__(self, bootstrap):
        self.settings = bootstrap.header
        self.pipe = None
        self.core = None
        self.stt = None
        self._native = False

    def bind_pipe(self, pipe, fatal):
        self.pipe, self.fatal = pipe, fatal

    async def prepare(self):
        import asyncio
        from .duplex_transport import DuplexAudioTransport
        from .duplex_contracts import DuplexMode
        from .parakeet_voice_worker import LocalVoiceSttProcess, LocalSttOptions
        from .rolling_transcript import TranscriptEngine
        from .voice_preprocessor import VoicePreprocessor, create_webrtc_vad
        from .voice_process_core import ConsoleSpeculativeVoiceSession
        from .voice_transcription import (
            _SerialSttWorker,
            _prepare_streaming_candidate,
            _close_streaming_candidate,
            _UNPREPARED_STREAMING_CANDIDATE,
            _RollingWindowStt,
            _NativeStreamingStt,
            _ROLLING_MIN_WINDOW_NS,
            _ROLLING_DEBOUNCE_SECONDS,
        )

        h = self.settings
        self.stt = LocalVoiceSttProcess(
            provider=h["stt_provider"],
            model=h["stt_model"],
            language=h["language"],
            options=LocalSttOptions(
                device=h.get("stt_device"),
                compute_type=h.get("stt_compute_type"),
                precision=h.get("stt_precision"),
            ),
        )
        serial = _SerialSttWorker()
        candidate = _UNPREPARED_STREAMING_CANDIDATE
        transport = DuplexAudioTransport()
        effects = _ChildEffects(self.pipe, transport, lambda: self.core, self.fatal)
        self.effects = effects
        vad = create_webrtc_vad(aggressiveness=h["vad_aggressiveness"])

        def preprocessor(**kwargs):
            if h["aec_enabled"]:
                return VoicePreprocessor.from_native(
                    vad=vad, vad_preroll_ms=h["vad_preroll_ms"], **kwargs
                )
            return VoicePreprocessor(
                aec=None, vad=vad, vad_preroll_ms=h["vad_preroll_ms"], **kwargs
            )

        async def prepare_stt():
            nonlocal candidate
            candidate, self._native = await _prepare_streaming_candidate(
                self.stt,
                provider=h["stt_provider"],
                model=h["stt_model"],
                language=h["language"],
                serial_worker=serial,
            )

        async def interrupt():
            await asyncio.to_thread(self.stt.close)

        async def close_stt():
            nonlocal candidate
            await interrupt()
            previous, candidate = candidate, _UNPREPARED_STREAMING_CANDIDATE
            await _close_streaming_candidate(previous, serial_worker=serial)
            await serial.close()

        def transcript(turn_id, publish):
            nonlocal candidate
            shared = dict(
                provider=h["stt_provider"],
                model=h["stt_model"],
                language=h["language"],
                serial_worker=serial,
            )
            rolling = _RollingWindowStt(self.stt, **shared)
            kwargs = dict(
                turn_id=turn_id,
                on_revision=publish,
                rolling_min_window_ns=_ROLLING_MIN_WINDOW_NS,
                rolling_debounce_seconds=_ROLLING_DEBOUNCE_SECONDS,
            )
            if not self._native:
                return TranscriptEngine(rolling_adapter=rolling, **kwargs)
            previous, candidate = candidate, _UNPREPARED_STREAMING_CANDIDATE
            return TranscriptEngine(
                live_adapter=_NativeStreamingStt(
                    self.stt,
                    prepared_candidate=previous,
                    quiet_seconds=h["response_eagerness_ms"] / 1000,
                    **shared,
                ),
                fallback_adapter=rolling,
                **kwargs,
            )

        self.core = ConsoleSpeculativeVoiceSession(
            transport=transport,
            effects=effects,
            preprocessor_factory=preprocessor,
            transcript_factory=transcript,
            response_eagerness_ms=h["response_eagerness_ms"],
            initial_duplex_mode=DuplexMode.HALF_DUPLEX,
            deferred_attempt_preparation=True,
            prepare_transcription=prepare_stt,
            interrupt_transcription=interrupt,
            close_transcription=close_stt,
            on_runtime_failure=lambda _: self.fatal("native_failed"),
        )
        await self.core.prepare()
        return self._native

    async def start(self, capture_live):
        del capture_live  # No parent stream can cross the private byte protocol.
        await self.core.enter(capture_live=False)

    def receive(self, record):
        return self.effects.receive(record)

    def begin_close(self):
        import asyncio
        from .voice_process_types import AttemptCleanupOutcome, ControlKind

        if self.core is not None:
            self.core.fence_and_close(ControlKind.TEARDOWN)

        async def native():
            if self.core is not None:
                await self.core._transport.close()
            return True

        async def resources():
            if self.core is not None:
                while self.core._resource_cleanup_task is None:
                    await asyncio.sleep(0)
                await asyncio.shield(self.core._resource_cleanup_task)
            return (
                self.stt.cleanup_outcome
                if self.stt is not None
                else AttemptCleanupOutcome.CLEAN
            )

        return native(), resources()


class _ChildEffects:
    """Bounded child proposals; original provider/context objects stay parent-side."""

    def __init__(self, pipe, transport, core, fatal):
        from .voice_process_protocol import ReceiveStream, StreamKey

        self.pipe, self.transport, self.core, self.fatal = pipe, transport, core, fatal
        self.attempts = {}
        self._retired_provider = {}
        self.phrases = {}
        self._retired_phrases = {}
        self.tasks = set()
        self.drafts = {}
        self.draft_task = None
        self.preview_task = None
        self.preview_pending = None
        self.recoveries = {}
        self._recovery_id = 0
        self.fenced = False
        self.cleanup_receiver = ReceiveStream(
            StreamKey(pipe.generation, pipe.request_id, "tts_closed")
        )

    def own(self, awaitable):
        import asyncio

        task = asyncio.ensure_future(awaitable)
        self.tasks.add(task)

        def finished(done):
            self.tasks.discard(done)
            if not done.cancelled() and done.exception() is not None:
                self.fatal("transport_failed")
            self._retire_cancelled()

        task.add_done_callback(finished)
        return task

    def fields(self, key):
        return dict(turn_id=key.turn_id, revision=key.revision, epoch=key.epoch)

    def credit(self, key, sequence, size):
        from .voice_process_protocol import ProtocolError

        try:
            self.pipe.credit(key, sequence, size)
        except ProtocolError:
            # Lost receipt delivery must never prevent local/native fencing.
            self.fatal("transport_failed")

    def dispatch_attempt(self, *, turn_id, attempt_epoch, transcript):
        import asyncio
        from types import SimpleNamespace
        from .voice_process_protocol import StreamKey, ReceiveStream, ProtocolError

        for state in tuple(self.attempts.values()):
            if (
                state.prepared is None
                and state.cancelled
                and state.cleanup.done()
                and state.receiver.data_complete
                and (state.worker is None or state.worker.done())
            ):
                self._retire_provider(state)
        if self.fenced or len(self.attempts) >= 2:
            raise ProtocolError("voice_capacity_exceeded")
        revision = self.core().coordinator.snapshot.revision_id
        key = StreamKey(
            self.pipe.generation,
            self.pipe.request_id,
            "provider",
            turn_id,
            revision,
            attempt_epoch,
        )
        state = SimpleNamespace(
            key=key,
            transcript=transcript,
            text="",
            prepared=None,
            receiver=ReceiveStream(key),
            speech=None,
            cleanup=asyncio.get_running_loop().create_future(),
            terminal=asyncio.get_running_loop().create_future(),
            cancelled=False,
            claimed=False,
            data=[],
            worker=None,
            ended=None,
            phrase_id=0,
            superseded=False,
            provider_end_seen=False,
            admitted=False,
        )
        state.end_sequence = None
        self.attempts[attempt_epoch] = state
        self.pipe.input.retain_turn(turn_id)
        if self._provider_lane_busy(state):
            self.own(self._prepare_after_retirement(state))
        else:
            self._prepare_provider(state)

    def _provider_lane_busy(self, state):
        return any(
            old is not state
            and old.admitted
            and (
                not old.cleanup.done()
                or not old.receiver.data_complete
                or old.data
                or (old.worker is not None and not old.worker.done())
            )
            for old in self.attempts.values()
        )

    async def _prepare_after_retirement(self, state):
        import asyncio

        while not state.cancelled and not self.fenced:
            snapshot = self.core().coordinator.snapshot
            if (
                snapshot.current_attempt_epoch != state.key.epoch
                or snapshot.turn_id != state.key.turn_id
                or snapshot.revision_id != state.key.revision
            ):
                self.fence_attempt(state.key.epoch)
                return
            if not self._provider_lane_busy(state):
                self._prepare_provider(state)
                return
            await asyncio.sleep(0.001)

    def _prepare_provider(self, state):
        # Actual cleanup can overtake published bytes. Keep the old receive
        # identity until its exact final data has been discarded and credited.
        self.pipe.input.open_stream(state.key)
        state.admitted = True
        self.pipe.send(
            "prepare", payload=state.transcript.encode(), **self.fields(state.key)
        )

    def start_prepared_attempt(self, epoch, request_handle):
        from .voice_phrase_sequencer import PhraseSpeechSequencer, ProcessPcmStream
        from .voice_process_protocol import StreamKey
        from .voice_turn_coordinator import AttemptPlaybackStarted

        state = self.attempts[epoch]
        owner = self

        class Synthesizer:
            async def synthesize_hands_free(self, *, text):
                from .voice_process_protocol import ProtocolError

                owner._retire_phrases()
                if len(owner.phrases) >= 3:
                    raise ProtocolError("voice_capacity_exceeded")
                state.phrase_id += 1
                key = StreamKey(
                    state.key.generation,
                    state.key.request_id,
                    "pcm",
                    state.key.turn_id,
                    state.key.revision,
                    state.key.epoch,
                    state.phrase_id,
                )
                stream = ProcessPcmStream(
                    key,
                    owner.pipe.input,
                    cleanup_receiver=owner.cleanup_receiver,
                    on_credit=owner.credit,
                    on_cancel=lambda _: owner.fence_attempt(epoch),
                )
                owner.phrases[key] = stream
                stream.activate()
                owner.pipe.send(
                    "synthesize",
                    payload=text.encode(),
                    phrase_id=key.phrase_id,
                    **owner.fields(key),
                )
                return stream.normalized_stream()

        state.speech = PhraseSpeechSequencer(
            epoch=epoch,
            synthesizer=Synthesizer(),
            sink=self.transport,
            on_playback_started=lambda _: self.own(
                self.core().submit(AttemptPlaybackStarted(epoch))
            ),
            on_failed=self._speech_failed,
        )
        self.pipe.send(
            "start_attempt",
            request_handle=request_handle.value,
            context_handle=state.prepared.header["context_handle"],
            **self.fields(state.key),
        )
        for old in self.attempts.values():
            if (
                old.key.turn_id == state.key.turn_id
                and old.key.epoch < epoch
                and old.cancelled
            ):
                old.superseded = True
        self._retire_cancelled(except_epoch=epoch)

    def _speech_failed(self, epoch, code):
        from .voice_turn_coordinator import AttemptTtsFailed

        state = self.attempts[epoch]
        if (
            code == "speech_input_limit"
            and not state.speech.synthesis_in_flight
            and all(
                stream.receiver.cleanup_outcome == "clean"
                and stream.receiver.data_complete
                for key, stream in self.phrases.items()
                if key.epoch == epoch
            )
        ):
            if not state.cancelled and not self.fenced:
                self.own(self.core().submit(AttemptTtsFailed(epoch, code)))
        else:
            self.fatal("tts_failed")

    def _retire_cancelled(self, except_epoch=None):
        self._retire_phrases()
        for epoch, state in tuple(self.attempts.items()):
            if (
                epoch != except_epoch
                and state.cancelled
                and state.superseded
                and state.cleanup.done()
                and state.receiver.data_complete
                and not state.data
                and (state.worker is None or state.worker.done())
                and not any(key.epoch == epoch for key in self.phrases)
            ):
                self._retire_provider(state)

    def _retire_phrases(self):
        for key, stream in tuple(self.phrases.items()):
            if (
                stream.receiver.data_complete
                and stream.receiver.cleanup_outcome is not None
            ):
                self.phrases.pop(key)
                self._retired_phrases[key] = (
                    stream.final_sequence,
                    stream.end_received,
                )
        while len(self._retired_phrases) > 3:
            self._retired_phrases.pop(next(iter(self._retired_phrases)))

    def _retire_provider(self, state):
        self.attempts.pop(state.key.epoch, None)
        if not state.admitted:
            # No parent request was issued, so no late terminal record is valid.
            return
        self._retired_provider[state.key] = (
            state.end_sequence,
            state.provider_end_seen,
        )
        while len(self._retired_provider) > 2:
            self._retired_provider.pop(next(iter(self._retired_provider)))

    def _end_provider(self, state, last_sequence):
        from .voice_process_protocol import ProtocolError

        if state.end_sequence is None:
            state.receiver.end(last_sequence)
            state.end_sequence = last_sequence
        elif state.end_sequence != last_sequence:
            raise ProtocolError()

    def receive(self, record):
        from .voice_process_protocol import ProtocolError, StreamKey
        from .voice_process_types import (
            AttemptDispatchPrepared,
            AttemptCleanupOutcome,
            VoiceRequestHandle,
            VoiceTurnContextHandle,
            VoiceSpeculationDecision,
            VoiceTerminalDisposition,
        )

        h = record.header
        op, epoch = h["op"], h.get("epoch")
        if op in {"pcm", "pcm_end", "tts_closed"}:
            key = StreamKey(
                h["generation"],
                h["request_id"],
                "pcm",
                h["turn_id"],
                h["revision"],
                epoch,
                h["phrase_id"],
            )
            stream = self.phrases.get(key)
            if stream is None:
                previous = self._retired_phrases.get(key)
                if op == "pcm_end" and previous == (h["last_sequence"], False):
                    self._retired_phrases[key] = (h["last_sequence"], True)
                    return
                raise ProtocolError()
            stream.receive(record)
            if op == "tts_closed" and h["outcome"] != "clean":
                self.fence_attempt(epoch)
                self.fatal("tts_failed")
            self._retire_phrases()
            return True
        state = self.attempts.get(epoch)
        if state is None and op in {"provider_end", "provider_failure", "tool_pending"}:
            key = StreamKey(
                h["generation"],
                h["request_id"],
                "provider",
                h["turn_id"],
                h["revision"],
                epoch,
            )
            previous = self._retired_provider.get(key)
            if previous is None or previous != (h["last_sequence"], False):
                raise ProtocolError()
            self._retired_provider[key] = (h["last_sequence"], True)
            return
        terminal_matches = (
            state is not None
            and op in {"terminal_claim", "terminal_result"}
            and h["revision"] == getattr(state, "terminal_revision", None)
            and h["turn_id"] == state.key.turn_id
        )
        if (
            state is None
            or not state.admitted
            or not (state.key.matches(record) or terminal_matches)
        ):
            raise ProtocolError()
        if op == "prepared":
            state.prepared = record
            self.own(
                self.core().submit(
                    AttemptDispatchPrepared(
                        epoch,
                        VoiceSpeculationDecision(h["decision"]),
                        VoiceRequestHandle(h["request_handle"]),
                        VoiceTurnContextHandle(h["context_handle"]),
                    )
                )
            )
        elif op == "provider_delta":
            delivery = state.receiver.receive(record, mailbox=self.pipe.input)
            state.data.append(delivery)
            if state.cancelled:
                self._discard(state)
                self._retire_cancelled()
            elif state.worker is None or state.worker.done():
                state.worker = self.own(self._consume_provider(state))
            return True
        elif op in {"provider_end", "provider_failure", "tool_pending"}:
            if state.provider_end_seen:
                raise ProtocolError()
            state.provider_end_seen = True
            self._end_provider(state, h["last_sequence"])
            state.ended = op
            if state.worker is None or state.worker.done():
                state.worker = self.own(self._consume_provider(state))
        elif op == "cleanup":
            self._end_provider(state, h["last_sequence"])
            if not state.cleanup.done():
                state.cleanup.set_result(AttemptCleanupOutcome(h["outcome"]))
        elif op == "terminal_claim":
            state.claimed = True
        elif op == "terminal_result":
            if state.terminal.done():
                raise ProtocolError()
            state.terminal.set_result(VoiceTerminalDisposition(h["disposition"]))
            self.own(self._retire_turn(state))
        elif op == "cancel":
            self.fence_attempt(epoch)
        else:
            raise ProtocolError()
        self._retire_cancelled()

    async def _consume_provider(self, state):
        from .voice_turn_coordinator import (
            AttemptOutputDelta,
            AttemptGenerationCompleted,
        )
        from .voice_process_types import AttemptToolPending, ProviderAttemptFailed

        while state.data:
            delivery = state.data.pop(0)
            if state.cancelled or delivery.discard_only:
                sequence, size = delivery.discard(state.key)
            else:
                text = delivery.record.payload.decode("utf-8")
                state.text += text
                await self.core().submit(AttemptOutputDelta(state.key.epoch, text))
                if not state.cancelled:
                    await state.speech.feed(state.key.epoch, text)
                sequence, size = (
                    delivery.discard(state.key)
                    if delivery.discard_only
                    else delivery.consume()
                )
            self.credit(state.key, sequence, size)
        if state.ended and not state.cancelled and state.receiver.data_complete:
            if state.ended == "provider_end":
                await self.core().submit(
                    AttemptGenerationCompleted(state.key.epoch, state.text)
                )
                await state.speech.finish(state.key.epoch)
            elif state.ended == "tool_pending":
                await self.core().submit(AttemptToolPending(state.key.epoch))
            else:
                await self.core().submit(
                    ProviderAttemptFailed(state.key.epoch, "ProviderFailure")
                )
            state.ended = None

    def _discard(self, state):
        while state.data:
            sequence, size = state.data.pop(0).discard(state.key)
            self.credit(state.key, sequence, size)

    def fence_attempt(self, epoch):
        state = self.attempts.get(epoch)
        if state is None or state.cancelled or state.claimed:
            return
        state.cancelled = True
        self.transport.fence_output()
        if not state.admitted:
            from .voice_process_types import AttemptCleanupOutcome

            # A pending local proposal never gave the parent resource custody.
            self._end_provider(state, 0)
            state.cleanup.set_result(AttemptCleanupOutcome.CLEAN)
            return
        self.pipe.input.fence_stream(state.key)
        self._discard(state)
        if state.speech is not None:
            state.speech.fence(epoch)
        for key, stream in tuple(self.phrases.items()):
            if key.epoch == epoch:
                stream.fence()
        from .voice_process_protocol import ProtocolError

        try:
            self.pipe.send("cancel", reason="stop", **self.fields(state.key))
        except ProtocolError:
            self.fatal("transport_failed")

    def cancel_attempt(self, epoch):
        import asyncio

        state = self.attempts.get(epoch)
        if state is None:

            async def clean():
                from .voice_process_types import AttemptCleanupOutcome

                return AttemptCleanupOutcome.CLEAN

            return clean()
        self.fence_attempt(epoch)

        async def close():
            from .voice_process_types import AttemptCleanupOutcome

            if state.claimed:
                await asyncio.shield(state.terminal)
                outcome = AttemptCleanupOutcome.CLEAN
            else:
                outcome = await asyncio.shield(state.cleanup)
            if state.speech is not None:
                await state.speech.cancel(epoch)
                await state.speech.wait_for_cleanup()
            return outcome

        return close()

    def abort_output(self, epoch):
        self.transport.fence_output()
        return self.transport.abort_output()

    @property
    def assistant_rendering(self):
        return any(
            state.speech is not None
            and not state.cancelled
            and state.speech.first_submission is not None
            for state in self.attempts.values()
        )

    def final_render_submission(self, epoch):
        state = self.attempts.get(epoch)
        return (
            state.speech.final_submission
            if state is not None and state.speech is not None
            else None
        )

    def publish_preview(self, epoch, text):
        from .voice_process_protocol import PAYLOAD_LIMIT

        state = self.attempts.get(epoch)
        if state is None or state.cancelled or self.fenced:
            return
        # state.text is cumulative: dropping a disposable projection cannot
        # delete a delta from the next displayed snapshot.
        payload = state.text.encode()[:PAYLOAD_LIMIT].decode("utf-8", "ignore").encode()
        self.preview_pending = (state.key, payload)
        if self.preview_task is None or self.preview_task.done():
            self.preview_task = self.own(self._publish_preview())

    async def _publish_preview(self):
        import asyncio
        from .voice_process_protocol import ProtocolError

        while self.preview_pending is not None and not self.fenced:
            key, payload = self.preview_pending
            state = self.attempts.get(key.epoch)
            if state is None or state.cancelled:
                self.preview_pending = None
                return
            try:
                self.pipe.send("preview", payload=payload, **self.fields(key))
            except ProtocolError as error:
                if error.code != "voice_capacity_exceeded":
                    raise
                await asyncio.sleep(0.001)
            else:
                if self.preview_pending == (key, payload):
                    self.preview_pending = None

    def clear_preview(self, epoch):
        # Parent clears from authoritative cancel/terminal/close receipts.
        if self.preview_pending is not None and self.preview_pending[0].epoch == epoch:
            self.preview_pending = None

    def record_revision(self, revision):
        from .voice_process_protocol import ProtocolError

        if revision.turn_id not in self.drafts and len(self.drafts) >= 2:
            raise ProtocolError("voice_capacity_exceeded")
        previous = self.drafts.get(revision.turn_id)
        if previous is None or revision.revision_id > previous[0]:
            self.drafts[revision.turn_id] = (
                revision.revision_id,
                revision.stable_text + revision.revisable_text,
                False,
            )
            if self.draft_task is None or self.draft_task.done():
                self.draft_task = self.own(self._publish_drafts())

    async def _publish_drafts(self):
        import asyncio
        from .voice_process_protocol import ProtocolError

        while True:
            pending = [
                (turn, value) for turn, value in self.drafts.items() if not value[2]
            ]
            if not pending:
                return
            for turn, value in pending:
                snapshot = self.core().coordinator.snapshot
                epoch = snapshot.current_attempt_epoch or snapshot.attempt_epoch
                try:
                    self.pipe.send(
                        "draft",
                        turn_id=turn,
                        revision=value[0],
                        epoch=epoch,
                        payload=value[1].encode(),
                    )
                except ProtocolError as error:
                    if error.code != "voice_capacity_exceeded":
                        raise
                    await asyncio.sleep(0.01)
                    continue
                if self.drafts.get(turn) == value:
                    self.drafts[turn] = (value[0], value[1], True)

    def preserve_draft(self, *, turn_id, transcript, **kwargs):
        import asyncio

        snapshot = self.core().coordinator.snapshot
        revision = self.drafts.get(turn_id, (snapshot.revision_id, "", False))[0]
        previous = self.drafts.get(turn_id)
        if previous is None or revision > previous[0]:
            self.drafts[turn_id] = (revision, transcript, False)
        if self.draft_task is None or self.draft_task.done():
            self.draft_task = self.own(self._publish_drafts())
        if kwargs.get("reason") == "provider_failed":
            state = next(
                (
                    state
                    for state in reversed(tuple(self.attempts.values()))
                    if state.key.turn_id == turn_id
                ),
                None,
            )
            from .voice_turn_coordinator import SpeculativeVoiceState

            if (
                state is not None
                and snapshot.turn_id == turn_id
                and snapshot.revision_id == state.key.revision
                and snapshot.state
                is SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
                and not self.fenced
                and turn_id not in self.recoveries
            ):
                self._recovery_id += 1
                self.recoveries[turn_id] = (state.key, self._recovery_id, False)
                self.own(self._request_recovery(state.key, self._recovery_id))
        return asyncio.shield(self.draft_task)

    async def _request_recovery(self, key, token):
        import asyncio

        await asyncio.shield(self.draft_task)
        await self.pipe.wait_draft_consumed(key.turn_id)
        self.observe_recovery_currentness(self.core().coordinator.snapshot)
        if self.recoveries.get(key.turn_id) != (key, token, False):
            return
        self.pipe.send(
            "draft_recovery", action="request", recovery_id=token, **self.fields(key)
        )
        self.recoveries[key.turn_id] = (key, token, True)

    def observe_recovery_currentness(self, snapshot):
        from .voice_process_protocol import ProtocolError
        from .voice_turn_coordinator import SpeculativeVoiceState

        for turn, (key, token, sent) in tuple(self.recoveries.items()):
            if (
                not self.fenced
                and snapshot.turn_id == turn
                and snapshot.revision_id == key.revision
                and snapshot.state
                is SpeculativeVoiceState.LISTENING_AFTER_PROVIDER_FAILURE
            ):
                continue
            self.recoveries.pop(turn)
            if sent:
                try:
                    self.pipe.send(
                        "draft_recovery",
                        action="revoke",
                        recovery_id=token,
                        **self.fields(key),
                    )
                except ProtocolError:
                    # Receipt loss cannot interrupt local/native fencing.
                    self.fatal("transport_failed")

    def promote(self, *, attempt_epoch, transcript, assistant_text, **kwargs):
        import asyncio

        self._terminal(
            self.attempts[attempt_epoch], "promote", transcript, assistant_text
        )
        return asyncio.shield(self.attempts[attempt_epoch].terminal)

    def submit_accepted_voice_turn(self, transcript, context):
        state = next(
            state
            for state in self.attempts.values()
            if state.prepared.header["context_handle"] == context.value
        )
        self._terminal(state, "accepted", transcript, "")
        return context.value

    def _terminal(self, state, kind, transcript, answer):
        state.terminal_revision = (
            self.core().coordinator.snapshot.revision_id
            if kind == "accepted"
            else state.key.revision
        )
        self.own(self._send_terminal(state, kind, transcript, answer))

    async def _send_terminal(self, state, kind, transcript, answer):
        import asyncio
        import hashlib
        from uuid import uuid4

        if self.draft_task is not None:
            await asyncio.shield(self.draft_task)
        await self.pipe.wait_draft_consumed(state.key.turn_id)
        # Cleanup may overtake already-published cancelled deltas. Acceptance
        # must report their actual consumed/discarded boundary, not cleanup alone.
        while kind == "accepted" and not state.receiver.data_complete:
            await asyncio.sleep(0.001)
        self.pipe.send(
            "terminal_propose",
            turn_id=state.key.turn_id,
            revision=state.terminal_revision,
            epoch=state.key.epoch,
            request_handle=state.prepared.header["request_handle"],
            context_handle=state.prepared.header["context_handle"],
            last_sequence=state.receiver.ack[0],
            kind=kind,
            boundary_id=uuid4().hex if kind == "promote" else None,
            answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            answer_chars=len(answer),
            payload=transcript.encode(),
        )

    async def _retire_turn(self, state):
        import asyncio
        from .voice_process_protocol import ProtocolError

        while state.terminal.result().value == "promoted" and not state.claimed:
            await asyncio.sleep(0.01)
        if state.speech is not None:
            await state.speech.wait_for_cleanup()
        while state.data or (state.worker is not None and not state.worker.done()):
            await asyncio.sleep(0.01)
        while any(
            other is not state and other.key.turn_id == state.key.turn_id
            for other in self.attempts.values()
        ):
            self._retire_cancelled()
            await asyncio.sleep(0.001)
        self._retire_provider(state)
        self.drafts.pop(state.key.turn_id, None)
        while True:
            try:
                self.pipe.retire_turn(state.key.turn_id)
            except ProtocolError:
                await asyncio.sleep(0.01)
            else:
                break

    def classify_spoken_command(self, transcript):
        return False

    def voice_dispatch_quarantined(self):
        return False

    def rebuild_audio(self, old, epoch):
        return self.core().rebuild_audio(old, epoch)

    def fence_all(self):
        if self.fenced:
            return
        self.fenced = True
        for epoch in tuple(self.attempts):
            self.fence_attempt(epoch)
        self.observe_recovery_currentness(self.core().coordinator.snapshot)

    async def close(self):
        import asyncio

        self.fence_all()
        await asyncio.gather(
            *(self.cancel_attempt(epoch) for epoch in tuple(self.attempts)),
            return_exceptions=True,
        )


def _native_abi() -> int:
    """Read the installed bridge ABI without constructing/opening a stream."""
    from tldw_voice_aec import DUPLEX_ABI_VERSION

    return DUPLEX_ABI_VERSION


async def run_child(
    read_fd: int,
    write_fd: int,
    *,
    session_factory=None,
    clock=None,
    native_abi_reader=None,
) -> int:
    """Own one child lifecycle; injection is direct Python in test wrappers only.

    Session.prepare() reports native-streaming readiness. Session.start() is the
    first permitted native construction/open. begin_close() synchronously fences
    output and returns independent native-checked-close and resource observers.
    The native observer returns bool; the resource observer returns the exact
    AttemptCleanupOutcome enum. Only timely CLEAN proves graceful cleanup.
    """
    import asyncio
    import threading
    import time

    from .voice_process_lifetime import (
        ChildLease,
        GRACE_SECONDS,
        HANDSHAKE_SECONDS,
        LifecyclePipe,
        NATIVE_CLOSE_SECONDS,
        SourceIdentity,
        VoiceProcessError,
        source_identity,
    )
    from .voice_process_protocol import ProtocolError, read_record
    from .voice_process_types import AttemptCleanupOutcome

    clock = clock or time.monotonic
    loop = asyncio.get_running_loop()
    bootstrap_result = loop.create_future()

    def deliver(value):
        if not bootstrap_result.done():
            bootstrap_result.set_result(value)

    def read_bootstrap():
        try:
            result = read_record(
                lambda count: os.read(read_fd, count), "parent_to_child"
            )
        except Exception:
            result = None
        try:
            loop.call_soon_threadsafe(deliver, result)
        except RuntimeError:
            os.close(read_fd)

    # A daemon owns a potentially incomplete first frame. On startup timeout the
    # process exits; the caller never closes/reuses a still-running read fd.
    bootstrap_thread = threading.Thread(
        target=read_bootstrap, daemon=True, name="voice-bootstrap"
    )
    bootstrap_thread.start()
    deadline = clock() + HANDSHAKE_SECONDS
    while not bootstrap_result.done() and clock() < deadline:
        await asyncio.sleep(0.01)
    if not bootstrap_result.done():
        os.close(write_fd)
        return 2
    bootstrap_thread.join()
    bootstrap = bootstrap_result.result()
    if bootstrap is None:
        os.close(read_fd)
        os.close(write_fd)
        return 2
    try:
        identity = source_identity()
        header = bootstrap.header
        received = SourceIdentity(
            header["root"], header["source"], header["native_abi"], header["version"]
        )
        if header["op"] != "bootstrap" or received != identity:
            raise VoiceProcessError()
        measured_abi = (native_abi_reader or _native_abi)()
        if (
            type(measured_abi) is not int
            or measured_abi != identity.native_abi
            or clock() >= deadline
        ):
            raise VoiceProcessError()
    except Exception:
        os.close(read_fd)
        os.close(write_fd)
        return 2

    lease = ChildLease(clock=clock)
    stop = asyncio.Event()
    start = asyncio.Event()
    parent_lost = False
    integrity_failed = False
    fault_reported = False
    prepared = False
    capture_live = False
    pipe = None
    session = None
    tasks = set()

    def own(awaitable):
        task = asyncio.ensure_future(awaitable)
        tasks.add(task)
        # Keep owners until process exit, including completed observer exceptions.
        task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        return task

    def report_fault(code):
        nonlocal fault_reported
        if fault_reported:
            return
        # Claim the single reserved report before sending: a broken writer or
        # closed mailbox cannot recursively enqueue another diagnostic.
        fault_reported = True
        try:
            pipe.send("fault", code=code)
        except ProtocolError:
            pass

    def fault(error: ProtocolError):
        nonlocal parent_lost, integrity_failed
        parent_lost = True
        stop.set()
        if error.code != "voice_transport_eof":
            integrity_failed = True
            report_fault(
                {
                    "voice_protocol_invalid": "protocol_invalid",
                    "voice_capacity_exceeded": "capacity_exceeded",
                }.get(error.code, "transport_failed")
            )

    def observe_pipe_faults():
        # Thread-retained evidence does not depend on callback delivery order.
        for error in (pipe.reader.failure, pipe.writer.failure):
            if error is not None:
                fault(error)

    def consume(record):
        nonlocal capture_live
        op = record.header["op"]
        if op == "lease":
            if not lease.renew():
                stop.set()
        elif (
            op == "start"
            and prepared
            and not start.is_set()
            and not stop.is_set()
            and lease.alive
        ):
            capture_live = record.header["capture_live"]
            start.set()
        elif op in {"close", "fault"}:
            stop.set()
        elif session is not None and callable(getattr(session, "receive", None)):
            return session.receive(record)
        else:
            stop.set()
            raise ProtocolError()

    pipe = LifecyclePipe(
        read_fd,
        write_fd,
        generation=header["generation"],
        request_id=header["request_id"],
        parent=False,
        consume=consume,
        fault=fault,
        initial_record=bootstrap,
    )

    async def watch_lease():
        while not stop.is_set():
            observe_pipe_faults()
            if not lease.alive:
                stop.set()
                return
            await asyncio.sleep(0.05)

    async def until_stop(task):
        while not task.done():
            if stop.is_set() or not lease.alive:
                raise VoiceProcessError("lease_expired")
            await asyncio.sleep(0.01)
        if stop.is_set() or not lease.alive:
            raise VoiceProcessError("lease_expired")
        return task.result()

    own(watch_lease())
    try:
        pipe.send(
            "hello", root=identity.root, source=identity.source, native_abi=measured_abi
        )
        session = (session_factory or _production_session)(bootstrap)
        if callable(getattr(session, "bind_pipe", None)):

            def fatal(code):
                report_fault(code)
                stop.set()

            session.bind_pipe(pipe, fatal)
        native_streaming = await until_stop(own(session.prepare()))
        prepared = True
        pipe.send("stt_ready", native_streaming=native_streaming)
        await until_stop(own(start.wait()))
        await until_stop(own(session.start(capture_live)))
        pipe.send("session_ready")
        await stop.wait()
    except Exception as error:
        stop.set()
        observe_pipe_faults()
        if isinstance(error, ProtocolError):
            fault(error)
        elif not parent_lost:
            report_fault("startup_failed")
    finally:
        stop.set()
        close_deadline = clock() + GRACE_SECONDS
        native_deadline = clock() + NATIVE_CLOSE_SECONDS
        native = resources = None

        async def observe_native(awaitable):
            result = await awaitable
            # Timestamp actual completion, not the next polling observation;
            # a receipt that first completes after the budget is uncertainty.
            return result is True and clock() < native_deadline

        async def observe_resources(awaitable):
            result = await awaitable
            if type(result) is not AttemptCleanupOutcome:
                return None
            if result is AttemptCleanupOutcome.CLEAN and clock() >= close_deadline:
                return None
            return result

        if session is not None:
            try:
                native_wait, resources_wait = session.begin_close()
                native, resources = (
                    own(observe_native(native_wait)),
                    own(observe_resources(resources_wait)),
                )
            except Exception:
                pass
        while native is not None and not native.done() and clock() < native_deadline:
            await asyncio.sleep(0.01)
        checked = (
            native is not None
            and native.done()
            and not native.cancelled()
            and native.exception() is None
            and native.result() is True
        )
        try:
            pipe.send("closed", outcome="clean" if checked else "failed")
        except ProtocolError as error:
            fault(error)
        while (
            resources is not None and not resources.done() and clock() < close_deadline
        ):
            await asyncio.sleep(0.01)
        disposition = None
        if (
            resources is not None
            and resources.done()
            and not resources.cancelled()
            and resources.exception() is None
        ):
            disposition = resources.result()
        try:
            pipe.send(
                "resources_closed",
                outcome=disposition.value if disposition is not None else "failed",
            )
        except ProtocolError as error:
            fault(error)
        # Final receipts retain a bounded write observation even when resource
        # cleanup consumed its grace deadline. Missing delivery is uncertainty.
        flush_deadline = loop.time() + 0.25
        observe_pipe_faults()
        while pipe.output.count and loop.time() < flush_deadline:
            observe_pipe_faults()
            await asyncio.sleep(0.001)
        pipe.stop()
        await asyncio.to_thread(pipe.writer.join, 0.25)
        observe_pipe_faults()
        if not pipe.writer.alive:
            os.close(write_fd)
        # Parent EOF has removed the only outside POSIX supervisor. Terminate the
        # exact child-led group, including a model that escaped graceful cleanup.
        if parent_lost and os.name == "posix" and os.getpgrp() == os.getpid():
            import signal

            os.killpg(os.getpid(), signal.SIGKILL)
        # No gather-all cleanup: process exit owns unfinished model observers.
        for task in tasks:
            if not task.done():
                task.cancel()
    # A broken output pipe cannot deliver its own fault. Preserve uncertainty
    # in the exit status as well; clean native/resource facts stay independent.
    return 2 if integrity_failed else 0


def main() -> int:
    """Reject all CLI options and require valid private bootstrap pipes."""
    if len(sys.argv) != 1:
        return 2
    try:
        read_fd, write_fd = isolate_standard_streams()
    except Exception:
        return 2
    import asyncio

    # Unlike asyncio.run(), do not wait forever for cancellation-hostile model
    # cleanup at interpreter shutdown. The parent retains whole-tree ownership.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(run_child(read_fd, write_fd))
    except BaseException:
        return 2
    finally:
        loop.close()


if __name__ == "__main__":
    raise SystemExit(main())
