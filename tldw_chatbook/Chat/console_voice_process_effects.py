"""App-owned voice authority; the audio process receives only opaque handles.

This owner runs on the UI loop. The injected controller callbacks are the same
preparation, winning-promotion and synchronous accepted-turn paths used by the
compatibility session. Transport composition supplies bounded control admission;
visible data uses the existing end-to-end protocol credits directly.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Awaitable, Iterable
from dataclasses import dataclass, field
import hashlib
import inspect
from typing import Any
from uuid import uuid4

from tldw_chatbook.Audio.voice_process_protocol import (
    CreditWindow,
    Mailbox,
    ProtocolError,
    Record,
    StreamKey,
    encode_record,
)
from tldw_chatbook.Chat.console_speculative_voice_session import (
    PreparedSpeculativeVoiceAttempt,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnExecutionContext
from tldw_chatbook.Chat.console_voice_attempts import (
    AttemptCleanupManager,
    AttemptCleanupOutcome,
    VoiceAttempt,
    VoiceAttemptDelta,
    VoiceAttemptSnapshot,
)
from tldw_chatbook.Chat.console_voice_eligibility import (
    classify_voice_speculation,
    VoiceSpeculationDecision,
)
from tldw_chatbook.Chat.console_voice_promotion import (
    VoicePromotionClaim,
    VoicePromotionClaimStatus,
    VoicePromotionOutcome,
)
from tldw_chatbook.Chat.console_voice_supervisor import (
    VoiceDispatchKind,
    VoiceDispatchSupervisor,
)
from tldw_chatbook.Chat.console_voice_worker import VoiceUiBridge


@dataclass(slots=True)
class _ContextLease:
    handle: str
    context: ConsoleTurnExecutionContext = field(repr=False)


@dataclass(slots=True)
class _RetainedAttempt:
    key: StreamKey
    transcript: str = field(repr=False)
    request_handle: str = field(default_factory=lambda: uuid4().hex)
    prepared: PreparedSpeculativeVoiceAttempt | None = field(default=None, repr=False)
    context: _ContextLease | None = field(default=None, repr=False)
    attempt: VoiceAttempt | None = field(default=None, repr=False)
    snapshot: VoiceAttemptSnapshot | None = field(default=None, repr=False)
    window: CreditWindow | None = field(default=None, repr=False)
    preparation: asyncio.Task | None = field(default=None, repr=False)
    generation: asyncio.Task | None = field(default=None, repr=False)
    cleanup: asyncio.Future | None = field(default=None, repr=False)
    claim: VoicePromotionClaim | str | None = field(default=None, repr=False)
    terminal: asyncio.Task | None = field(default=None, repr=False)
    settlement: Any = field(default=None, repr=False)
    decision: VoiceSpeculationDecision | None = None
    cancelled: bool = False
    superseded: bool = False
    failed: bool = False
    last_sequence: int = 0
    pending_start: bool = False


class ProcessVoiceEffects:
    """Retain at most two preparations/contexts and one terminal operation.

    ``is_current`` must check the owning view, transcript revision, configuration
    and activation against the original prepared value (None before preparation).
    ``send_control`` performs bounded, synchronous transport admission or raises.
    Neither callback may derive authority from a wire value.
    """

    def __init__(
        self,
        *,
        generation: int,
        request_id: str,
        outbound: Mailbox,
        bridge: VoiceUiBridge,
        prepare_attempt: Callable[..., Awaitable[PreparedSpeculativeVoiceAttempt]],
        gateway: Any,
        dispatch_supervisor: VoiceDispatchSupervisor,
        is_current: Callable[[StreamKey, PreparedSpeculativeVoiceAttempt | None], bool],
        is_context_current: Callable[
            [StreamKey, PreparedSpeculativeVoiceAttempt | None], bool
        ]
        | None = None,
        send_control: Callable[[Record], None],
        promote: Callable[..., Any] | None = None,
        submit_accepted_voice_turn: Callable[[str, ConsoleTurnExecutionContext], str]
        | None = None,
    ) -> None:
        if bridge._loop is not asyncio.get_running_loop():
            raise ValueError("voice_effects_require_ui_owner_loop")
        if not isinstance(dispatch_supervisor, VoiceDispatchSupervisor):
            raise TypeError("dispatch_supervisor must be a VoiceDispatchSupervisor")
        self._session = StreamKey(generation, request_id, "control")
        self._outbound = outbound
        self._bridge = bridge
        self._prepare_attempt = prepare_attempt
        self._gateway = gateway
        self._supervisor = dispatch_supervisor
        self._is_current = is_current
        self._is_context_current = is_context_current or is_current
        self._send_control = send_control
        self._promote = promote
        self._accepted = submit_accepted_voice_turn
        self._cleanup = AttemptCleanupManager(dispatch_supervisor)
        self._attempts: dict[StreamKey, _RetainedAttempt] = {}
        self._drafts: dict[str, tuple[int, str]] = {}
        self._latest_preparation_epoch = -1
        self._sequence = 0
        self._fenced = False
        self._terminal: asyncio.Task | None = None
        self.failure: ProtocolError | None = None

    def _validate_key(self, key: StreamKey) -> None:
        if (
            key.generation != self._session.generation
            or key.request_id != self._session.request_id
            or key.lane != "provider"
            or not key.turn_id
            or key.phrase_id
        ):
            raise ProtocolError()

    def _admit_turn(self, turn_id: str) -> None:
        retained = {key.turn_id for key in self._attempts} | self._drafts.keys()
        if turn_id not in retained and len(retained) >= 2:
            raise ProtocolError("voice_capacity_exceeded")

    def _current(
        self,
        state: _RetainedAttempt,
        *,
        context_only: bool = False,
        operation_key: StreamKey | None = None,
    ) -> bool:
        check = self._is_context_current if context_only else self._is_current
        return (
            not self._fenced
            and not self._bridge.closed
            and not state.superseded
            and (context_only or not state.cancelled)
            and check(operation_key or state.key, state.prepared) is True
        )

    def _require_current(
        self,
        state: _RetainedAttempt,
        *,
        context_only: bool = False,
        operation_key: StreamKey | None = None,
    ) -> None:
        if not self._current(
            state, context_only=context_only, operation_key=operation_key
        ):
            if state.claim is None and state.prepared is not None:
                self._abandon(state.prepared)
            raise ProtocolError()

    def _record(
        self,
        state: _RetainedAttempt,
        op: str,
        *,
        operation_key: StreamKey | None = None,
        **fields: object,
    ) -> Record:
        self._sequence += 1
        key = operation_key or state.key
        return Record(
            dict(
                version=1,
                generation=key.generation,
                request_id=key.request_id,
                sequence=self._sequence,
                op=op,
                turn_id=key.turn_id,
                revision=key.revision,
                epoch=key.epoch,
                **fields,
            )
        )

    def _send(self, record: Record) -> Record:
        encode_record(record, "parent_to_child")
        receipt = record.header["op"] in {
            "cleanup",
            "terminal_claim",
            "terminal_result",
        }
        if not self._fenced or receipt:
            try:
                self._send_control(record)
            except Exception:
                self.failure = ProtocolError("voice_transport_failed")
                self.fence()
                if not receipt:
                    raise self.failure from None
        return record

    def _abandon(self, prepared: PreparedSpeculativeVoiceAttempt) -> None:
        trace = prepared.request.provisional_trace_attempt
        if trace is not None:
            self._gateway.abandon_provisional_voice_trace(trace)

    async def prepare(self, key: StreamKey, transcript: str) -> Record:
        """Issue handles only after original UI preparation and strict eligibility."""
        self._validate_key(key)
        self._retire_superseded()
        # Reuse closed wire validation for text and all identity scalar limits.
        encode_record(
            Record(
                dict(
                    version=1,
                    generation=key.generation,
                    request_id=key.request_id,
                    sequence=1,
                    op="prepare",
                    turn_id=key.turn_id,
                    revision=key.revision,
                    epoch=key.epoch,
                ),
                transcript.encode("utf-8"),
            ),
            "child_to_parent",
        )
        if key.epoch <= self._latest_preparation_epoch:
            raise ProtocolError()
        state = _RetainedAttempt(key, transcript)
        # A received start already adopted its context, even if provider entry
        # still awaits predecessor credit. Its superseded predecessor can free
        # a retained slot only when that real credit arrives. Keep this request
        # in its existing bounded control delivery; do not allocate a third slot.
        try:
            while len(self._attempts) >= 2 and any(
                old.superseded
                and old.window is not None
                and old.window.outstanding != (0, 0)
                for old in self._attempts.values()
            ):
                self._require_current(state)
                await asyncio.sleep(0.001)
                self._retire_superseded()
            self._require_current(state)
        except BaseException:
            if key not in self._attempts:
                self._send(
                    self._record(state, "cleanup", outcome="clean", last_sequence=0)
                )
            raise
        if key.epoch <= self._latest_preparation_epoch:
            raise ProtocolError()
        if key in self._attempts or len(self._attempts) >= 2:
            raise ProtocolError("voice_capacity_exceeded")
        self._admit_turn(key.turn_id)
        self._latest_preparation_epoch = key.epoch
        self._attempts[key] = state

        async def invoke():
            self._require_current(state)
            prepared = await self._prepare_attempt(
                turn_id=key.turn_id,
                attempt_epoch=key.epoch,
                transcript=transcript,
            )
            if type(prepared) is not PreparedSpeculativeVoiceAttempt:
                raise ProtocolError()
            state.prepared = prepared
            try:
                self._require_current(state)
                if prepared.request.attempt_epoch != key.epoch:
                    raise ProtocolError()
            except BaseException:
                self._abandon(prepared)
                raise
            return prepared

        state.preparation = asyncio.create_task(
            self._bridge.prepare(invoke, on_discard=self._abandon)
        )
        try:
            prepared = await asyncio.shield(state.preparation)
            self._require_current(state)
            state.context = _ContextLease(uuid4().hex, prepared.frozen_session_context)
            state.decision = classify_voice_speculation(
                frozen_session_context=prepared.frozen_session_context,
                prepared_request=prepared.request.prepared,
                requires_citation_creation=prepared.requires_citation_creation,
                requires_pre_dispatch_authority=prepared.requires_pre_dispatch_authority,
            )
            return self._send(
                self._record(
                    state,
                    "prepared",
                    request_handle=state.request_handle,
                    context_handle=state.context.handle,
                    decision=state.decision.value,
                )
            )
        except BaseException:
            # The same retained cancellation owner publishes exactly one
            # cleanup receipt even when failure crosses a received fence.
            await self.cancel(key)
            self._attempts.pop(key, None)
            raise

    def _lookup(
        self, key: StreamKey, request_handle: str, context_handle: str
    ) -> _RetainedAttempt:
        self._validate_key(key)
        state = self._attempts.get(key)
        if (
            state is None
            or state.prepared is None
            or state.context is None
            or type(request_handle) is not str
            or request_handle != state.request_handle
            or type(context_handle) is not str
            or context_handle != state.context.handle
        ):
            raise ProtocolError()
        return state

    def context_for(self, key: StreamKey, handle: str) -> ConsoleTurnExecutionContext:
        """Resolve original context independently of provider cancellation."""
        state = self._attempts.get(key)
        if state is None or state.context is None or state.context.handle != handle:
            raise ProtocolError()
        return state.context.context

    def retire_context(self, key: StreamKey, handle: str) -> None:
        """Explicitly retire context only after terminal/cleanup and data custody."""
        self.context_for(key, handle)
        state = self._attempts[key]
        if not self._can_retire(state):
            raise ProtocolError()
        self._attempts.pop(key)

    @staticmethod
    def _can_retire(state: _RetainedAttempt) -> bool:
        dispositions = tuple(
            work for work in (state.terminal, state.cleanup) if work is not None
        )
        return not (
            not dispositions
            or any(
                not work.done() or work.cancelled() or work.exception() is not None
                for work in dispositions
            )
            or (state.preparation is not None and not state.preparation.done())
            or (state.generation is not None and not state.generation.done())
            or (state.window is not None and state.window.outstanding != (0, 0))
        )

    def _retire_superseded(self) -> None:
        for key, state in tuple(self._attempts.items()):
            if state.superseded and self._can_retire(state):
                self._attempts.pop(key)

    def start_attempt(
        self, key: StreamKey, request_handle: str, context_handle: str
    ) -> None:
        """Recheck issued authority and app-wide quarantine before provider entry."""
        state = self._lookup(key, request_handle, context_handle)
        self._require_current(state)
        self._supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)
        if (
            state.attempt is not None
            or state.pending_start
            or state.decision is not VoiceSpeculationDecision.PROVISIONAL
        ):
            raise ProtocolError()
        # Receipt of both issued handles proves same-turn context adoption;
        # provider entry may still wait for the old published-data credit.
        for old in self._attempts.values():
            if (
                old.key.turn_id == key.turn_id
                and old.key.epoch < key.epoch
                and old.cancelled
                and old.claim is None
                and old.terminal is None
            ):
                old.superseded = True
        self._retire_superseded()
        if any(
            old is not state
            and old.window is not None
            and old.window.outstanding != (0, 0)
            for old in self._attempts.values()
        ):
            # The child's discard may precede delivery of its credit. Retain
            # issued authority; resume through the same checks on that receipt.
            state.pending_start = True
            return
        state.window = CreditWindow(key, blocks=7)
        self._outbound.open_stream(key)

        async def sink(delta: VoiceAttemptDelta) -> None:
            self._require_current(state)
            permit = await state.window.reserve()
            try:
                self._require_current(state)
            except BaseException:
                permit.abandon()
                raise
            record = Record(
                dict(
                    version=1,
                    generation=key.generation,
                    request_id=key.request_id,
                    sequence=permit.sequence,
                    op="provider_delta",
                    turn_id=key.turn_id,
                    revision=key.revision,
                    epoch=key.epoch,
                ),
                delta.text.encode("utf-8"),
            )
            permit.publish(record)
            self._outbound.put(record)
            state.last_sequence = permit.sequence

        state.attempt = VoiceAttempt(
            request=state.prepared.request,
            gateway=self._gateway,
            is_epoch_current=lambda epoch: epoch == key.epoch and self._current(state),
            visible_delta_sink=sink,
            on_failed=lambda event: setattr(state, "failed", True),
        )
        state.attempt.start()

        async def observe():
            await state.attempt.wait()
            if not self._current(state):
                return
            state.snapshot = state.attempt.snapshot
            op = (
                "provider_failure"
                if state.failed
                else "tool_pending"
                if state.snapshot.tool_request
                else "provider_end"
            )
            fields = {"last_sequence": state.last_sequence}
            if state.failed:
                fields["code"] = "provider_failed"
            self._send(self._record(state, op, **fields))

        state.generation = asyncio.create_task(observe())
        state.generation.add_done_callback(_consume)

    async def wait_attempt(self, key: StreamKey) -> None:
        """Observe generation without transferring ownership to a view waiter."""
        state = self._attempts[key]
        if state.generation is not None:
            await asyncio.shield(state.generation)

    def cancel(self, key: StreamKey) -> Awaitable[AttemptCleanupOutcome]:
        """Fence immediately; keep original context and actual cleanup custody."""
        state = self._attempts.get(key)
        if state is None:
            raise ProtocolError()
        if state.claim is not None:
            return _clean()
        state.cancelled = True
        state.pending_start = False
        if state.window is not None:
            state.window.close()
        if state.cleanup is None:
            if state.attempt is not None:
                receipt = self._cleanup.cancel(state.attempt)
            else:
                if state.preparation is not None and not state.preparation.done():
                    state.preparation.cancel()
                if state.prepared is not None:
                    self._abandon(state.prepared)

                async def preparation_closed():
                    if state.preparation is not None:
                        try:
                            await asyncio.shield(state.preparation)
                        except (asyncio.CancelledError, Exception):
                            pass
                    return AttemptCleanupOutcome.CLEAN

                receipt = asyncio.create_task(preparation_closed())

            async def publish_cleanup():
                outcome = await receipt
                self._send(
                    self._record(
                        state,
                        "cleanup",
                        outcome=outcome.value,
                        last_sequence=state.last_sequence,
                    )
                )
                return outcome

            state.cleanup = asyncio.create_task(publish_cleanup())
            state.cleanup.add_done_callback(_consume)
        return asyncio.shield(state.cleanup)

    def accept_credit(self, record: Record) -> None:
        """Route exact cumulative provider receipts without inventing credits."""
        encode_record(record, "child_to_parent")
        h = record.header
        if (
            h["op"] != "credit"
            or h["request_id"] != self._session.request_id
            or h["generation"] != self._session.generation
            or h["lane"] != "provider"
        ):
            raise ProtocolError()
        key = StreamKey(
            h["generation"],
            h["stream_id"],
            h["lane"],
            h["stream_turn"],
            h["stream_revision"],
            h["stream_epoch"],
            h["stream_phrase"],
        )
        state = self._attempts.get(key)
        if state is None or state.window is None:
            raise ProtocolError()
        state.window.accept_credit(record)
        self._retire_superseded()
        for pending in tuple(self._attempts.values()):
            if pending.pending_start:
                pending.pending_start = False
                if self._current(pending):
                    self.start_attempt(
                        pending.key, pending.request_handle, pending.context.handle
                    )
                else:
                    self.cancel(pending.key)

    async def terminal(self, record: Record) -> Record:
        """Validate final text against the parent snapshot, then retain settlement."""
        encode_record(record, "child_to_parent")
        h = record.header
        if h["op"] != "terminal_propose":
            raise ProtocolError()
        key = StreamKey(
            h["generation"],
            h["request_id"],
            "provider",
            h["turn_id"],
            h["revision"],
            h["epoch"],
        )
        if h["kind"] == "accepted":
            original = next(
                (
                    state
                    for state in self._attempts.values()
                    if state.request_handle == h["request_handle"]
                    and state.context is not None
                    and state.context.handle == h["context_handle"]
                ),
                None,
            )
            if (
                original is None
                or key
                != StreamKey(
                    original.key.generation,
                    original.key.request_id,
                    "provider",
                    original.key.turn_id,
                    key.revision,
                    original.key.epoch,
                )
                or key.revision < original.key.revision
            ):
                raise ProtocolError()
            state = self._lookup(original.key, h["request_handle"], h["context_handle"])
        else:
            state = self._lookup(key, h["request_handle"], h["context_handle"])
        self._require_current(
            state, context_only=h["kind"] == "accepted", operation_key=key
        )
        if self._terminal is not None or state.terminal is not None:
            raise ProtocolError("voice_capacity_exceeded")
        transcript = record.payload.decode("utf-8")
        if (h["kind"] == "promote" and transcript != state.transcript) or h[
            "last_sequence"
        ] != state.last_sequence:
            raise ProtocolError()
        answer = (
            ""
            if h["kind"] == "accepted"
            else state.snapshot.response_text
            if state.snapshot
            else None
        )
        if (
            answer is None
            or hashlib.sha256(answer.encode()).hexdigest() != h["answer_sha256"]
            or len(answer) != h["answer_chars"]
        ):
            raise ProtocolError()
        if h["kind"] == "promote" and (
            state.failed or state.snapshot.tool_request or state.snapshot.speech_frozen
        ):
            raise ProtocolError()
        state.terminal = asyncio.create_task(
            self._settle(state, h["kind"], answer, transcript, key)
        )
        self._terminal = state.terminal

        def terminal_done(task):
            if self._terminal is task:
                self._terminal = None
            _consume(task)

        state.terminal.add_done_callback(terminal_done)
        try:
            return await asyncio.shield(state.terminal)
        finally:
            if state.terminal.done():
                self._terminal = None

    async def _settle(
        self,
        state: _RetainedAttempt,
        kind: str,
        answer: str,
        transcript: str,
        operation_key: StreamKey,
    ) -> Record:
        def claimed(claim):
            if (
                type(claim) is not VoicePromotionClaim
                or claim.status is not VoicePromotionClaimStatus.CLAIMED
            ):
                raise ProtocolError()
            state.claim = claim

        def invoke():
            self._require_current(
                state, context_only=kind == "accepted", operation_key=operation_key
            )
            if kind == "accepted":
                if self._accepted is None:
                    raise ProtocolError()
                self._supervisor.ensure_dispatch_allowed(VoiceDispatchKind.HANDS_FREE)
                result = self._accepted(transcript, state.context.context)
                if type(result) is not str or not result:
                    raise ProtocolError()
                state.claim = result
                state.settlement = result
                return (result,)
            if self._promote is None:
                raise ProtocolError()
            result = self._promote(
                prepared=state.prepared,
                snapshot=state.snapshot,
                transcript=state.transcript,
                assistant_text=answer,
                on_claim=claimed,
            )
            # Do not await inside the bridge observer: custody already belongs
            # to the original winner and this app-owned settlement operation.
            state.settlement = result
            return (result,)

        disposition = "failed"
        try:
            try:
                (settlement,) = await self._bridge.terminal(invoke)
            except (asyncio.CancelledError, Exception):
                if state.claim is None:
                    raise
                settlement = state.settlement
            if state.claim is not None:
                try:
                    self._send(
                        self._record(
                            state,
                            "terminal_claim",
                            operation_key=operation_key,
                            context_handle=state.context.handle,
                            last_sequence=state.last_sequence,
                        )
                    )
                except ProtocolError:
                    pass  # Delivery failure does not cancel original custody.
            if kind == "accepted":
                disposition = "promoted"
            else:
                if not inspect.isawaitable(settlement):
                    raise ProtocolError()
                outcome = await asyncio.shield(settlement)
                if type(outcome) is not VoicePromotionOutcome:
                    raise ProtocolError()
                if state.claim is not None:
                    if (
                        outcome.promotion_id != state.claim.promotion_id
                        or outcome.session_id != state.claim.session_id
                    ):
                        raise ProtocolError()
                    disposition = outcome.status.value
        except Exception:
            disposition = "failed"
        finally:
            if state.claim is None and state.prepared is not None:
                self._abandon(state.prepared)
        return self._send(
            self._record(
                state,
                "terminal_result",
                operation_key=operation_key,
                context_handle=state.context.handle,
                disposition=disposition,
            )
        )

    def receive_draft(self, record: Record) -> None:
        """Retain each turn's recoverable draft independently during settlement."""
        encode_record(record, "child_to_parent")
        h = record.header
        if h["op"] != "draft" or (h["generation"], h["request_id"]) != (
            self._session.generation,
            self._session.request_id,
        ):
            raise ProtocolError()
        self._admit_turn(h["turn_id"])
        previous = self._drafts.get(h["turn_id"])
        if previous is not None and h["revision"] <= previous[0]:
            raise ProtocolError()
        self._drafts[h["turn_id"]] = (h["revision"], record.payload.decode("utf-8"))

    def draft_for(self, turn_id: str) -> str | None:
        """Return a turn's retained draft without consuming another turn's text."""
        value = self._drafts.get(turn_id)
        return value[1] if value is not None else None

    def retire_turn(self, turn_id: str) -> None:
        """Release a delivered/preserved draft after all its contexts retire."""
        if any(key.turn_id == turn_id for key in self._attempts):
            raise ProtocolError()
        self._drafts.pop(turn_id, None)

    async def receive_batch(self, records: Iterable[Record]) -> None:
        """Apply every already-received revocation before a start or claim."""
        batch = tuple(records)
        revoked = set()
        for record in batch:
            encode_record(record, "child_to_parent")
        for record in batch:
            if record.header["op"] == "cancel":
                h = record.header
                operation = StreamKey(
                    h["generation"],
                    h["request_id"],
                    "provider",
                    h["turn_id"],
                    h["revision"],
                    h["epoch"],
                )
                self.cancel(operation)
                revoked.add(
                    (
                        operation.generation,
                        operation.request_id,
                        operation.turn_id,
                        operation.epoch,
                    )
                )
        for record in batch:
            h = record.header
            if h["op"] == "cancel":
                continue
            operation = StreamKey(
                h["generation"],
                h["request_id"],
                "provider",
                h.get("turn_id", ""),
                h.get("revision", 0),
                h.get("epoch", 0),
            )
            if h["op"] == "start_attempt":
                self.start_attempt(operation, h["request_handle"], h["context_handle"])
            elif h["op"] == "prepare":
                await self.prepare(operation, record.payload.decode("utf-8"))
            elif h["op"] == "terminal_propose":
                if (
                    operation.generation,
                    operation.request_id,
                    operation.turn_id,
                    operation.epoch,
                ) in revoked:
                    raise ProtocolError()
                await self.terminal(record)
            elif h["op"] == "credit":
                self.accept_credit(record)
            elif h["op"] == "draft":
                self.receive_draft(record)
            else:
                raise ProtocolError()

    def fence(self) -> None:
        """Fence child authority without cancelling already claimed operations."""
        self._fenced = True
        for key, state in self._attempts.items():
            if state.claim is None:
                self.cancel(key)

    async def aclose(self) -> None:
        """Observe retained actual cleanup/settlement, isolated from view death."""
        self.fence()
        for state in tuple(self._attempts.values()):
            for work in (
                state.preparation,
                state.cleanup,
                state.generation,
                state.terminal,
            ):
                if work is not None:
                    try:
                        await asyncio.shield(work)
                    except asyncio.CancelledError:
                        if not work.cancelled():
                            raise
                    except Exception:
                        pass


def _clean() -> asyncio.Future[AttemptCleanupOutcome]:
    receipt = asyncio.get_running_loop().create_future()
    receipt.set_result(AttemptCleanupOutcome.CLEAN)
    return receipt


def _consume(task: asyncio.Future) -> None:
    if not task.cancelled():
        task.exception()
