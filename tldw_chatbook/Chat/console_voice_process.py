"""Parent-owned voice process admission and independent teardown evidence."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from tldw_chatbook.Audio.voice_process_lifetime import (
    GRACE_SECONDS,
    KILL_SECONDS,
    LifecyclePipe,
    OwnedProcessTree,
    RENEW_SECONDS,
    SourceIdentity,
    StartupGate,
    TERMINATE_SECONDS,
    VoiceProcessError,
    child_environment,
    source_identity,
)
from tldw_chatbook.Audio.voice_process_protocol import (
    ProtocolError,
    Record,
    StreamKey,
    encode_record,
)
from tldw_chatbook.Audio.voice_process_types import AttemptCleanupOutcome


@dataclass(frozen=True)
class CloseEvidence:
    native_closed: bool
    child_exited: bool
    tree_absent: bool
    pipes_closed: bool
    # A forced attempt (including an uncertain/failed signal), or an observed
    # abnormal child exit, permanently quarantines even with native confirmation.
    forced_termination: bool
    resources_outcome: AttemptCleanupOutcome | None
    transport_failure_code: str | None


@dataclass(frozen=True, slots=True)
class VoiceProcessSnapshot:
    """Content-free parent facts, never a shared child coordinator."""

    phase: str
    ready: bool
    hello_verified: bool
    fenced: bool


class DeviceLease:
    """One app-lifetime instance; independent of provider/promotion custody."""

    def __init__(self) -> None:
        self._owner = None
        self._lock = threading.Lock()
        self.quarantined = False

    def acquire(self, owner) -> None:
        with self._lock:
            if self.quarantined or self._owner is not None:
                raise VoiceProcessError("shutdown_unconfirmed")
            self._owner = owner

    def finish(self, owner, evidence: CloseEvidence) -> None:
        with self._lock:
            if self._owner is not owner:
                return
            if not (
                evidence.native_closed
                and evidence.child_exited
                and evidence.tree_absent
                and evidence.pipes_closed
                and not evidence.forced_termination
                and evidence.resources_outcome is AttemptCleanupOutcome.CLEAN
                and evidence.transport_failure_code is None
            ):
                self.quarantined = True
            else:
                self._owner = None


class VoiceProcessSupervisor:
    """Retain child and independent parent obligations across view replacement."""

    def __init__(self):
        self.device_lease = DeviceLease()
        self._sessions = {}
        self.close_task = None

    def retain(self, session, *, session_id: str | None = None):
        if self.close_task is not None:
            raise VoiceProcessError("stale")
        self._sessions[session] = session_id
        session._on_retired = lambda: self._sessions.pop(session, None)

    def has_live_session(self, session_id: str) -> bool:
        """Include listening and retained cleanup in conversation lifecycle checks."""
        return session_id in self._sessions.values()

    def begin_close(self):
        """Synchronously revoke all current sessions before any quit observer."""
        if self.close_task is None:
            from tldw_chatbook.Audio.voice_process_types import ControlKind

            sessions = tuple(self._sessions)
            receipts = tuple(
                session.fence_and_close(ControlKind.TEARDOWN) for session in sessions
            )
            self.close_task = asyncio.create_task(self._close(sessions, receipts))
            self.close_task.add_done_callback(
                lambda done: None if done.cancelled() else done.exception()
            )

    async def aclose(self):
        self.begin_close()
        await asyncio.shield(self.close_task)

    async def _close(self, sessions, receipts):
        await asyncio.gather(*receipts, return_exceptions=True)
        await asyncio.gather(*(session.wait_effects_closed() for session in sessions))
        self._sessions.clear()


def bootstrap_record(
    *,
    generation: int,
    request_id: str,
    stt_provider: str,
    stt_model: str | None,
    language: str,
    response_eagerness_ms: int,
    aec_enabled: bool,
    vad_aggressiveness: int,
    vad_preroll_ms: int,
    stt_device: str | None = None,
    stt_compute_type: str | None = None,
    stt_precision: str | None = None,
    checkout_identity=None,
) -> Record:
    """Resolve nonsecret settings and attest the actual parent source.

    Task8 passes the existing launcher's identity when active. It remains parent
    context; root/HEAD are checked here, never fabricated for installed wheels.
    """
    identity = source_identity()
    if checkout_identity is not None:
        try:
            if Path(checkout_identity.root).resolve() != Path(identity.root):
                raise VoiceProcessError()
            result = subprocess.run(
                ["git", "-C", identity.root, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=2,
            )
            if result.stdout.strip() != checkout_identity.head:
                raise VoiceProcessError()
        except Exception:
            raise VoiceProcessError() from None
    record = Record(
        dict(
            version=identity.version,
            op="bootstrap",
            generation=generation,
            request_id=request_id,
            sequence=1,
            root=identity.root,
            source=identity.source,
            native_abi=identity.native_abi,
            stt_provider=stt_provider,
            stt_model=stt_model,
            stt_device=stt_device,
            stt_compute_type=stt_compute_type,
            stt_precision=stt_precision,
            language=language,
            response_eagerness_ms=response_eagerness_ms,
            aec_enabled=aec_enabled,
            vad_aggressiveness=vad_aggressiveness,
            vad_preroll_ms=vad_preroll_ms,
        )
    )
    encode_record(record, "parent_to_child")
    return record


class ConsoleVoiceProcess:
    """One-shot facade; retain enter and close owners across caller cancellation.

    No legacy audio fallback exists. Supplying a Popen callable is a direct
    Python test seam; release launch always uses the fixed module command.
    """

    def __init__(
        self,
        device_lease: DeviceLease,
        *,
        popen=subprocess.Popen,
        tree_factory=OwnedProcessTree,
        effects=None,
        bootstrap=None,
        preflight=None,
        current=None,
        prepare_attempt=None,
        gateway=None,
        dispatch_supervisor=None,
        promote=None,
        accepted_handoff=None,
        project_preview=None,
        clear_preview=None,
        preserve_draft=None,
        on_runtime_failure=None,
    ) -> None:
        self.device_lease = device_lease
        self._popen = popen
        self._tree_factory = tree_factory
        self._effects = effects
        self._effects_close_task = None
        self.process = None
        self.tree = None
        self.pipe = None
        self.enter_task = None
        self.close_task = None
        self._monitor_task = None
        self._renew_task = None
        self._stop = asyncio.Event()
        self._failure = None
        self._gate = None
        self.hello_verified = False
        self.ready = False
        self.native_closed = False
        self._native_received = False
        self.resources_outcome: AttemptCleanupOutcome | None = None
        self._resources_received = False
        self._forced_termination = False
        self.transport_failure_code: str | None = None
        self._bootstrap = bootstrap
        self._preflight = preflight
        self._current = current
        self._bindings = (
            prepare_attempt,
            gateway,
            dispatch_supervisor,
            promote,
            accepted_handoff,
        )
        self._project_preview = project_preview
        self._clear_preview = clear_preview
        self._preserve_draft = preserve_draft
        self._on_runtime_failure = on_runtime_failure
        self._failure_notified = False
        self._tts = None
        self._parent_tasks = set()
        self._pending_records = []
        self._dispatch_queued = False
        self._revisions = {}
        self._draft_revisions = {}
        self._contexts = {}
        self._terminals = {}
        self._revoked = set()
        self._preserved = set()
        self._recovery_tasks = {}
        self._recovery_tokens = {}
        self._prepare_failures = {}
        self._claimed = set()
        self._preview = None
        self._projection_key = None
        self._phase = "preparing"

    @property
    def snapshot(self) -> VoiceProcessSnapshot:
        return VoiceProcessSnapshot(
            self._phase, self.ready, self.hello_verified, self._stop.is_set()
        )

    @property
    def state(self):
        from tldw_chatbook.Audio.voice_turn_coordinator import SpeculativeVoiceState

        return (
            SpeculativeVoiceState.LISTENING
            if self.ready
            else SpeculativeVoiceState.IDLE
        )

    def fence_and_close(self, reason):
        self._begin_close()
        return asyncio.shield(self.close_task)

    def _own_parent(self, coroutine):
        task = self._own(coroutine)
        self._parent_tasks.add(task)
        task.add_done_callback(self._parent_tasks.discard)
        return task

    def _own(self, coroutine):
        task = asyncio.ensure_future(coroutine)
        task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        return task

    async def enter(
        self,
        bootstrap: Record | None = None,
        *,
        preflight: Callable[[], Awaitable[bool]] | None = None,
        current: Callable[[], bool] | None = None,
        capture_live: bool = False,
    ) -> None:
        if self.enter_task is not None or self._stop.is_set():
            raise VoiceProcessError("stale")
        if capture_live:
            raise VoiceProcessError("stale")
        self.enter_task = self._own(
            self._enter(
                bootstrap or self._bootstrap,
                preflight or self._preflight,
                current or self._current,
            )
        )
        try:
            await asyncio.shield(self.enter_task)
        except asyncio.CancelledError:
            self._begin_close()
            raise

    async def _enter(self, bootstrap, preflight, current):
        try:
            if not await preflight():
                raise VoiceProcessError()
            if self._stop.is_set() or not current():
                raise VoiceProcessError("stale")
            identity = source_identity()
            h = bootstrap.header
            encode_record(bootstrap, "parent_to_child")
            if (
                SourceIdentity(h["root"], h["source"], h["native_abi"], h["version"])
                != identity
            ):
                raise VoiceProcessError()
            self.device_lease.acquire(self)
            self._gate = StartupGate(identity)
            self.process = self._popen(
                [sys.executable, "-m", "tldw_chatbook.Audio.voice_process_entry"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                start_new_session=os.name == "posix",
                cwd=identity.root,
                env=child_environment(),
                bufsize=0,
            )
            self.tree = self._tree_factory(self.process.pid)
            self.tree.attach()
            self.pipe = LifecyclePipe(
                os.dup(self.process.stdout.fileno()),
                self.process.stdin.fileno(),
                generation=h["generation"],
                request_id=h["request_id"],
                parent=True,
                consume=self._receive,
                fault=self._transport_fault,
            )
            if self._bindings[0] is not None:
                self._compose_parent()
            self._monitor_task = self._own(self._monitor())
            fields = {
                key: value
                for key, value in h.items()
                if key not in {"version", "op", "generation", "request_id", "sequence"}
            }
            self.pipe.send("bootstrap", **fields)
            if not await asyncio.to_thread(self.pipe.writer.wait_written, 1, 0.5):
                raise VoiceProcessError("transport_failed")
            self._renew_task = self._own(self._renew())
            await self._wait(lambda: self._gate.prepared, lambda: self._gate.deadline)
            self._gate.permit(current=not self._stop.is_set() and current())
            # Wire compatibility: this grants new child capture, never adoption.
            self.pipe.send("start", capture_live=True)
            await self._wait(lambda: self.ready, lambda: self._gate.deadline)
        except Exception as error:
            self._begin_close()
            if isinstance(error, VoiceProcessError):
                raise
            raise VoiceProcessError() from None

    async def _wait(self, predicate, deadline):
        while True:
            if self._failure is not None:
                raise self._failure
            if self._stop.is_set():
                raise VoiceProcessError("stale")
            if predicate():
                return
            if time.monotonic() >= deadline():
                raise VoiceProcessError()
            await asyncio.sleep(0.01)

    def _receive(self, record):
        h = record.header
        op = h["op"]
        try:
            if (
                op
                in {
                    "prepare",
                    "start_attempt",
                    "cancel",
                    "terminal_propose",
                    "synthesize",
                    "draft",
                    "draft_recovery",
                    "preview",
                    "credit",
                    "diagnostic",
                }
                and self._effects is not None
            ):
                self._pending_records.append(record)
                if not self._dispatch_queued:
                    self._dispatch_queued = True
                    asyncio.get_running_loop().call_soon(self._dispatch_batch)
                return True
            if op == "closed":
                if self._native_received:
                    raise VoiceProcessError("protocol_invalid")
                self._native_received = True
                self.native_closed = h["outcome"] == "clean"
                self._begin_close()
            elif op == "resources_closed":
                if self._resources_received:
                    raise VoiceProcessError("protocol_invalid")
                self._resources_received = True
                self.resources_outcome = (
                    None
                    if h["outcome"] == "failed"
                    else AttemptCleanupOutcome(h["outcome"])
                )
                if self.resources_outcome is AttemptCleanupOutcome.FORCE_CLOSED:
                    self._forced_termination = True
                self._begin_close()
            elif op == "fault":
                if h["code"] in {
                    "protocol_invalid",
                    "capacity_exceeded",
                    "transport_failed",
                }:
                    self._retain_transport_failure(ProtocolError("voice_" + h["code"]))
                self._failure = VoiceProcessError(h["code"])
                self._begin_close()
            elif self._stop.is_set() and op == "hello":
                # Reserved fault/close may overtake queued hello. Identity is
                # still evidence, but stopped startup can never advance again.
                self.hello_verified = (
                    SourceIdentity(
                        h["root"], h["source"], h["native_abi"], h["version"]
                    )
                    == self._gate.identity
                )
            elif self._stop.is_set():
                return
            elif op == "hello":
                self._gate.hello(
                    SourceIdentity(
                        h["root"], h["source"], h["native_abi"], h["version"]
                    )
                )
                self.hello_verified = True
            elif op == "stt_ready":
                self._gate.stt_ready()
            elif op == "session_ready" and self._gate.permitted and not self.ready:
                self.ready = True
                self._phase = "listening"
            else:
                raise VoiceProcessError("protocol_invalid")
        except VoiceProcessError as error:
            if error.code == "protocol_invalid":
                self._retain_transport_failure(ProtocolError())
            self._failure = error
            self._begin_close()

    def _compose_parent(self):
        from tldw_chatbook.Chat.console_voice_process_effects import ProcessVoiceEffects
        from tldw_chatbook.Chat.console_voice_tts_bridge import OwnerLoopPcmProducer
        from tldw_chatbook.Chat.console_voice_worker import VoiceUiBridge
        from tldw_chatbook.Chat.console_speculative_voice_session import (
            _LazyHandsFreeTts,
        )

        prepare, gateway, supervisor, promote, accepted = self._bindings
        self._bridge = VoiceUiBridge(asyncio.get_running_loop())

        def current(key, prepared, *, context=False):
            return bool(
                not self._stop.is_set()
                and self._current()
                and self._revisions.get(key.turn_id) == key.revision
                and (context or not self._is_revoked(key))
            )

        self._effects = ProcessVoiceEffects(
            generation=self.pipe.generation,
            request_id=self.pipe.request_id,
            outbound=self.pipe.output,
            bridge=self._bridge,
            prepare_attempt=prepare,
            gateway=gateway,
            dispatch_supervisor=supervisor,
            is_current=current,
            is_context_current=lambda key, prepared: current(
                key, prepared, context=True
            ),
            send_control=self._send_control,
            promote=promote,
            submit_accepted_voice_turn=accepted,
        )
        self._tts = OwnerLoopPcmProducer(
            asyncio.get_running_loop(),
            _LazyHandsFreeTts().synthesize_hands_free,
            self.pipe.output,
            cleanup_key=StreamKey(
                self.pipe.generation, self.pipe.request_id, "tts_closed"
            ),
            on_fault=lambda _: None if self._stop.is_set() else self._transport_fault(),
        )
        self._productions = {}

    def _send_control(self, record):
        if record.header["op"] == "provider_failure":
            key = self._key(record)
            self._prepare_failures[key.turn_id] = (key, None, None)
        if record.header["op"] == "terminal_claim":
            self._claimed.add(record.header["turn_id"])
            if (
                self._projection_key is not None
                and self._projection_key.turn_id == record.header["turn_id"]
            ):
                self._clear_projection()
        try:
            return self.pipe.send_record(record)
        except Exception:
            self._transport_fault()
            raise

    def _is_revoked(self, key):
        return any(
            turn == key.turn_id and epoch >= key.epoch for turn, epoch in self._revoked
        )

    def _prune_contexts(self):
        for key, handle in tuple(self._contexts.items()):
            try:
                self._effects.context_for(key, handle)
            except ProtocolError:
                self._contexts.pop(key)

    def _clear_projection(self):
        self._preview = None
        self._projection_key = None
        if self._clear_preview is not None:
            with contextlib.suppress(Exception):
                self._clear_preview()

    def _project(self, key, assistant=None):
        if assistant is None and self._preview is not None:
            preview_key = self._preview[0]
            if (preview_key.turn_id, preview_key.revision) == (
                key.turn_id,
                key.revision,
            ):
                key = preview_key
        if (
            self._stop.is_set()
            or not self._current()
            or key.turn_id in self._terminals
            or key.turn_id in self._claimed
            or self._is_revoked(key)
        ):
            return
        if assistant is not None:
            previous = self._preview
            if previous is not None and (key.revision, key.epoch) < (
                previous[0].revision,
                previous[0].epoch,
            ):
                return
            self._preview = (key, assistant)
        if self._draft_revisions.get(key.turn_id) != key.revision:
            return
        previous = self._preview
        text = previous[1] if previous is not None and previous[0] == key else ""
        if self._project_preview is not None:
            from tldw_chatbook.Widgets.Console import VoicePreviewProjection

            self._phase = "responding" if text else "listening"
            self._project_preview(
                VoicePreviewProjection(
                    turn_id=key.turn_id,
                    attempt_epoch=key.epoch,
                    user_text=self._effects.draft_for(key.turn_id) or "",
                    assistant_text=text,
                    status=self._phase,
                )
            )
            self._projection_key = key

    def _key(self, record, lane="provider"):
        h = record.header
        return StreamKey(
            h["generation"],
            h["request_id"],
            lane,
            h.get("turn_id", ""),
            h.get("revision", 0),
            h.get("epoch", 0),
            h.get("phrase_id", 0) if lane == "pcm" else 0,
        )

    def _dispatch_batch(self):
        self._dispatch_queued = False
        records, self._pending_records = self._pending_records, []
        # Validate in wire order, before scheduling any UI recovery delivery.
        # A later already-received revoke wins over its earlier queued request.
        for record in records:
            h = record.header
            if h["op"] != "draft_recovery" or self._stop.is_set():
                continue
            key, token = self._key(record), h["recovery_id"]
            previous = self._recovery_tokens.get(key.turn_id)
            failed = self._prepare_failures.get(key.turn_id)
            invalid = (
                (
                    h["action"] == "request"
                    and previous is not None
                    and token <= previous[1]
                )
                or (
                    h["action"] == "request"
                    and (
                        failed is None
                        or key.epoch > failed[0].epoch
                        or (key.epoch == failed[0].epoch and key != failed[0])
                    )
                )
                or (h["action"] == "revoke" and previous != (key, token, True))
                or (previous is None and len(self._recovery_tokens) >= 2)
            )
            if invalid:
                self._transport_fault(ProtocolError())
                break
            self._recovery_tokens[key.turn_id] = (key, token, h["action"] == "request")
        revoked = {
            (r.header["turn_id"], r.header["epoch"])
            for r in records
            if r.header["op"] == "cancel"
        }
        for turn, epoch in revoked:
            if self._stop.is_set():
                break
            if (
                not any(old_turn == turn for old_turn, _ in self._revoked)
                and len(self._revoked) >= 2
            ):
                self._transport_fault(ProtocolError("voice_capacity_exceeded"))
                break
            previous = max(
                (old for old_turn, old in self._revoked if old_turn == turn), default=-1
            )
            self._revoked = {item for item in self._revoked if item[0] != turn}
            self._revoked.add((turn, max(epoch, previous)))
        if (
            self._projection_key is not None
            and (self._projection_key.turn_id, self._projection_key.epoch) in revoked
        ):
            self._clear_projection()
        for record in records:
            if record.header["op"] == "cancel":
                with contextlib.suppress(ProtocolError):
                    self._own_parent(self._effects.cancel(self._key(record)))
                for key in tuple(self._productions):
                    if (key.turn_id, key.epoch) in revoked:
                        self._tts.cancel(key)
        for record in records:
            self._own_parent(self._dispatch_record(record, revoked))

    async def _dispatch_record(self, record, revoked):
        h = record.header
        op, key = h["op"], self._key(record)
        try:
            if op == "credit":
                if h["lane"] == "provider":
                    self._effects.accept_credit(record)
                else:
                    self._tts.accept_credit(record)
                self._prune_contexts()
                self._retire_terminal_contexts()
                return
            if op == "draft":
                self._effects.receive_draft(record)
                self._revisions[key.turn_id] = key.revision
                self._draft_revisions[key.turn_id] = key.revision
                self._project(key)
                return
            if op == "synthesize":
                phrase_key = self._key(record, "pcm")
                production = self._tts.submit(
                    key=phrase_key, text=record.payload.decode("utf-8")
                )
                self._productions[phrase_key] = production
                if self._stop.is_set() or self._is_revoked(key):
                    self._tts.cancel(phrase_key)

                async def retire():
                    await production.wait_for_retirement()
                    self._productions.pop(phrase_key, None)

                self._own_parent(retire())
                return
            if self._stop.is_set() or op == "cancel":
                return
            if op == "draft_recovery":
                if h["action"] == "request":
                    await self._deliver_recovery(key, h["recovery_id"])
                return
            if op == "prepare":
                self._revisions[key.turn_id] = max(
                    key.revision, self._revisions.get(key.turn_id, 0)
                )
            if op == "prepare":
                if self._is_revoked(key):
                    self.pipe.send(
                        "cleanup",
                        turn_id=key.turn_id,
                        revision=key.revision,
                        epoch=key.epoch,
                        outcome="clean",
                        last_sequence=0,
                    )
                    return
                try:
                    prepared = await self._effects.prepare(
                        key, record.payload.decode("utf-8")
                    )
                except (asyncio.CancelledError, Exception) as error:
                    from .console_voice_preflight import voice_failure_category

                    category = voice_failure_category(error)
                    current = (
                        not self._stop.is_set()
                        and self._current()
                        and not self._is_revoked(key)
                        and self._revisions.get(key.turn_id) == key.revision
                    )
                    if isinstance(error, ProtocolError) and current:
                        raise
                    if (
                        current
                        and category != "stale"
                        and not isinstance(error, asyncio.CancelledError)
                    ):
                        self._prepare_failures[key.turn_id] = (
                            key,
                            category,
                            type(error).__name__,
                        )
                        self.pipe.send(
                            "provider_failure",
                            code="provider_failed",
                            last_sequence=0,
                            turn_id=key.turn_id,
                            revision=key.revision,
                            epoch=key.epoch,
                        )
                    else:
                        self.pipe.send(
                            "cancel",
                            reason="stop",
                            turn_id=key.turn_id,
                            revision=key.revision,
                            epoch=key.epoch,
                        )
                else:
                    self._contexts[key] = prepared.header["context_handle"]
            elif op == "start_attempt":
                if not self._is_revoked(key):
                    self._effects.start_attempt(
                        key, h["request_handle"], h["context_handle"]
                    )
                    self._prune_contexts()
            elif op == "terminal_propose":
                if (key.turn_id, key.epoch) in revoked:
                    raise ProtocolError()
                result = await self._effects.terminal(record)
                self._terminals[key.turn_id] = result.header["disposition"]
                if (
                    self._projection_key is not None
                    and self._projection_key.turn_id == key.turn_id
                ):
                    self._clear_projection()
                if result.header["disposition"] != "promoted":
                    self._preserve_recovery(key.turn_id)
                self._retire_terminal_contexts()
            elif op == "preview":
                self._project(key, record.payload.decode("utf-8"))
            elif op != "diagnostic":
                raise ProtocolError()
        except (asyncio.CancelledError, Exception):
            if not self._stop.is_set():
                self._transport_fault()
        finally:
            with contextlib.suppress(ProtocolError):
                self.pipe.release(record)
            if self._terminals:
                self._retire_terminal_contexts()

    def _preserve_recovery(self, turn_id):
        if (
            turn_id in self._preserved
            or turn_id in self._claimed
            or self._terminals.get(turn_id) == "promoted"
        ):
            return
        text = self._effects.draft_for(turn_id)
        if text and self._preserve_draft is not None:
            self._preserved.add(turn_id)

            async def preserve():
                import inspect

                result = self._preserve_draft(text)
                if inspect.isawaitable(result):
                    await result

            task = self._own_parent(preserve())
            self._recovery_tasks[turn_id] = task

            def recovered(_done):
                self._recovery_tasks.pop(turn_id, None)
                self._retire_terminal_contexts()

            task.add_done_callback(recovered)

    async def _deliver_recovery(self, key, token):
        def current():
            failed = self._prepare_failures.get(key.turn_id)
            return (
                not self._stop.is_set()
                and self._current()
                and failed is not None
                and failed[0] == key
                and self._recovery_tokens.get(key.turn_id) == (key, token, True)
                # A reader-delivered revoke may still await its scheduled batch
                # when an older UI callback resumes. It already fences delivery.
                and not any(
                    record.header["op"] == "draft_recovery"
                    and record.header["action"] == "revoke"
                    and record.header["recovery_id"] == token
                    and self._key(record) == key
                    for record in self._pending_records
                )
                and self._draft_revisions.get(key.turn_id) == key.revision
                and key.turn_id not in self._claimed
                and key.turn_id not in self._terminals
                and key.turn_id not in self._preserved
            )

        if not current() or self._preserve_draft is None:
            return
        if key.turn_id in self._recovery_tasks:
            raise ProtocolError("voice_capacity_exceeded")
        text = self._effects.draft_for(key.turn_id)
        if text is None:
            raise ProtocolError()
        failure = self._prepare_failures.get(key.turn_id)
        category = failure[1] if failure is not None and failure[0] == key else None
        task = asyncio.current_task()
        self._recovery_tasks[key.turn_id] = task
        try:
            import inspect

            result = self._preserve_draft(
                text,
                preparation_failure_category=category,
                exception_type=failure[2] if category else None,
                is_current=current,
            )
            if inspect.isawaitable(result):
                result = await result
            if result is True:
                self._preserved.add(key.turn_id)
        finally:
            self._recovery_tasks.pop(key.turn_id, None)

    def _retire_terminal_contexts(self):
        for key, handle in tuple(self._contexts.items()):
            if key.turn_id not in self._terminals:
                continue
            try:
                self._effects.context_for(key, handle)
            except ProtocolError:
                # A newer same-turn start may already have retired this exact
                # original context after all its real cleanup/data receipts.
                self._contexts.pop(key)
                continue
            try:
                self._effects.retire_context(key, handle)
            except ProtocolError:
                continue
            self._contexts.pop(key)
        for turn_id in tuple(self._terminals):
            if turn_id in self._recovery_tasks:
                continue
            if any(key.turn_id == turn_id for key in self._contexts):
                continue
            self._effects.retire_turn(turn_id)
            try:
                self.pipe.retire_turn(turn_id)
            except ProtocolError:
                continue
            self._revisions.pop(turn_id, None)
            self._draft_revisions.pop(turn_id, None)
            self._recovery_tokens.pop(turn_id, None)
            self._prepare_failures.pop(turn_id, None)
            self._revoked = {item for item in self._revoked if item[0] != turn_id}
        # Retain only the two most recent terminal tombstones for late drafts.
        while len(self._terminals) > 2:
            oldest = next(iter(self._terminals))
            self._terminals.pop(oldest)
            self._preserved.discard(oldest)
            self._claimed.discard(oldest)

    def _retain_transport_failure(self, error: ProtocolError) -> None:
        code = ProtocolError(error.code).code
        if code != "voice_transport_eof" and self.transport_failure_code is None:
            self.transport_failure_code = code

    def _transport_fault(self, error: ProtocolError | None = None):
        self._retain_transport_failure(error or ProtocolError("voice_transport_failed"))
        self._failure = self._failure or VoiceProcessError("transport_failed")
        self._begin_close()

    async def _renew(self):
        while not self._stop.is_set():
            await asyncio.sleep(RENEW_SECONDS)
            if not self._stop.is_set():
                try:
                    self.pipe.send("lease")
                except ProtocolError as error:
                    self._transport_fault(error)
                except Exception:
                    self._transport_fault()

    async def _monitor(self):
        while not self._stop.is_set():
            if self.process.poll() is not None:
                # PID exit is not a parser failure. Its unread receipts and
                # actual EOF still have to be observed independently below.
                self._failure = self._failure or VoiceProcessError("transport_failed")
                self._begin_close()
                return
            await asyncio.sleep(0.05)

    def _begin_close(self):
        first = not self._stop.is_set()
        self._stop.set()
        self._phase = "off"
        self.ready = False
        if first:
            self._clear_projection()
            if self._tts is not None:
                for key in tuple(self._productions):
                    self._tts.cancel(key)
        if self._effects is not None and self._effects_close_task is None:
            self._effects.fence()
            self._effects_close_task = self._own(self._close_parent())
        if self._gate is not None:
            self._gate.stopped = True
        if self.close_task is None:
            self.close_task = self._own(self._close())
        if self._failure is not None and not self._failure_notified:
            self._failure_notified = True
            if self._on_runtime_failure is not None:
                asyncio.get_running_loop().call_soon(
                    self._on_runtime_failure, self._failure
                )

    async def _close_parent(self):
        await self._effects.aclose()
        if self.close_task is not None:
            await asyncio.shield(self.close_task)
        if self._tts is not None:
            await self._tts.aclose()
        while self._parent_tasks:
            await asyncio.gather(*tuple(self._parent_tasks), return_exceptions=True)
        for turn_id in tuple(self._revisions):
            self._preserve_recovery(turn_id)
        while self._parent_tasks:
            await asyncio.gather(*tuple(self._parent_tasks), return_exceptions=True)
        retired = getattr(self, "_on_retired", None)
        if retired is not None:
            retired()

    async def close(self) -> CloseEvidence:
        self._begin_close()
        return await asyncio.shield(self.close_task)

    async def wait_effects_closed(self) -> None:
        """Observe parent cleanup separately from checked child/device closure."""
        if self._effects_close_task is not None:
            await asyncio.shield(self._effects_close_task)

    def _terminate_tree(self, *, force: bool = False) -> None:
        # Record the requirement before signalling. An uncertain attempt must
        # never become evidence of graceful cleanup if absence is seen later.
        self._forced_termination = True
        self.tree.terminate(force=force)

    async def _observe_until(self, deadline):
        while True:
            child_exited = self.process.poll() is not None
            tree_absent = (
                self.tree.observe_absence()
                if self.tree is not None and self.tree.attached
                else child_exited
            )
            if child_exited and tree_absent:
                return True
            if time.monotonic() >= deadline:
                return False
            # A dead audio owner can no longer clean its surviving model.
            if child_exited and self.tree is not None:
                self._terminate_tree()
            await asyncio.sleep(0.02)

    async def _close(self):
        deadline = time.monotonic() + GRACE_SECONDS
        if self.process is None:
            evidence = CloseEvidence(
                True,
                True,
                True,
                True,
                False,
                AttemptCleanupOutcome.CLEAN,
                self.transport_failure_code,
            )
            self.device_lease.finish(self, evidence)
            if self._effects is None:
                retired = getattr(self, "_on_retired", None)
                if retired is not None:
                    retired()
            return evidence
        if (
            self.pipe is not None
            and self.process.poll() is None
            and not (self._native_received or self._resources_received)
        ):
            try:
                self.pipe.send("close", reason="teardown")
            except ProtocolError as error:
                self._retain_transport_failure(error)
            except Exception:
                self._retain_transport_failure(ProtocolError("voice_transport_failed"))
        if self.tree is None or not self.tree.attached:
            # Failed Windows assignment must tear down the unadmitted PID.
            self._forced_termination = True
            try:
                self.process.kill()
            except ProcessLookupError:
                pass
        if not await self._observe_until(deadline):
            if self.tree is not None:
                self._terminate_tree()
            if not await self._observe_until(time.monotonic() + TERMINATE_SECONDS):
                if self.tree is not None:
                    self._terminate_tree(force=True)
                await self._observe_until(time.monotonic() + KILL_SECONDS)
        child_exited = self.process.poll() is not None
        tree_absent = (
            self.tree.observe_absence()
            if self.tree is not None and self.tree.attached
            else child_exited
        )
        pipes_closed = True
        if self.pipe is not None:
            # Peer/tree absence unblocks the read fd, but is not proof that its
            # final checked-close frame was parsed. Keep reader admission open
            # through its bounded EOF join, then finish scheduled receipt use.
            self.pipe.writer.close()
            pipes_closed = await self.pipe.join()
            drain_deadline = time.monotonic() + 0.25
            while self.pipe.input.count and time.monotonic() < drain_deadline:
                await asyncio.sleep(0.001)
            pipes_closed &= self.pipe.input.count == 0
            # Fault callbacks can still be queued behind clean-looking receipts.
            # The reader/writer retain their categorical evidence independently.
            for failure in (self.pipe.reader.failure, self.pipe.writer.failure):
                if failure is not None:
                    self._retain_transport_failure(failure)
            self.pipe.stop()
        if pipes_closed:
            self.process.stdin.close()
            self.process.stdout.close()
        if self._tts is not None:
            self._tts.transport_failed(ProtocolError("voice_transport_closed"))
        evidence = CloseEvidence(
            self.native_closed,
            child_exited,
            tree_absent,
            pipes_closed,
            self._forced_termination or (child_exited and self.process.returncode != 0),
            self.resources_outcome,
            self.transport_failure_code,
        )
        self.device_lease.finish(self, evidence)
        return evidence
