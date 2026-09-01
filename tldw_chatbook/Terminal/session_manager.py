"""App-global ownership boundary for persistent terminal sessions."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from threading import Event, Lock, RLock
from time import monotonic
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .contracts import (
    AdmissionGate,
    CleanupAttempt,
    CleanupProof,
    CleanupSchedule,
    MAX_SESSION_RECORDS,
    TerminalLaunchRequest,
    TerminalLifecycle,
    TerminalEvent,
    TerminalProjection,
    TerminalReason,
    TerminalReceipt,
    apply_event,
)

if TYPE_CHECKING:
    from .backend import TerminalBackend
    from .io_actors import (
        InputOfferResult,
        OutputOfferResult,
        ParserTurnResult,
        TerminalInputActor,
        TerminalInputEvent,
        TerminalOutputActor,
    )
    from .screen_model import TerminalScreenSnapshot


@dataclass(frozen=True, slots=True)
class TerminalArmResult:
    """Content-free result of a launch-local arm request."""

    armed: bool = False
    reason: TerminalReason | None = None
    disclosure_required: bool = False


@dataclass(frozen=True, slots=True)
class TerminalCreateResult:
    """Immutable result of a terminal-session creation request."""

    admitted: bool = False
    reason: TerminalReason | None = None
    projection: TerminalProjection | None = None


@dataclass(frozen=True, slots=True)
class TerminalCleanupDeadline:
    """Absolute boundaries derived from one cleanup attempt."""

    attempt: CleanupAttempt
    hangup_at: float
    terminate_at: float
    force_kill_at: float
    deadline_at: float


@dataclass(frozen=True, slots=True)
class TerminalViewToken:
    """Opaque monotonically increasing mounted-view generation."""

    generation: int


@dataclass(frozen=True, slots=True)
class TerminalSubscriptionToken:
    """Opaque identity for one content-free change subscription."""

    value: int


@dataclass(frozen=True, slots=True)
class TerminalSessionView:
    """Immutable session and screen values safe for a mounted view."""

    projection: TerminalProjection
    screen: TerminalScreenSnapshot
    shell: str = ""
    start_directory: str = ""
    columns: int = 0
    rows: int = 0
    cleanup_receipt: TerminalReceipt | None = None


@dataclass(frozen=True, slots=True)
class TerminalViewState:
    """Immutable manager state returned to the current view generation."""

    selected_session_id: str | None = None
    sessions: tuple[TerminalSessionView, ...] = ()


@dataclass(frozen=True, slots=True)
class ManagedProcessIdentity:
    """Test-only process identity used for managed-RSS accounting."""

    pid: int
    birth_identity: str

    def __post_init__(self) -> None:
        if type(self.pid) is not int or self.pid <= 0:
            raise ValueError("pid must be a positive integer")
        if type(self.birth_identity) is not str or not self.birth_identity:
            raise ValueError("birth_identity must be non-empty text")


@dataclass(slots=True)
class _SessionRecord:
    """Manager-private mutable ownership for one retained session."""

    projection: TerminalProjection
    request: TerminalLaunchRequest = TerminalLaunchRequest()
    backend: TerminalBackend | None = None
    model: Any | None = None
    model_lock: Any = field(default_factory=Lock, repr=False)
    resize_lock: Any = field(default_factory=asyncio.Lock, repr=False)
    startup_done: Event = field(default_factory=Event, repr=False)
    input_actor: TerminalInputActor | None = None
    output_actor: TerminalOutputActor | None = None
    receipt: TerminalReceipt | None = None
    cleanup_future: Future[None] | None = None
    cleanup_action: str = ""


class TerminalSessionManager:
    """Own terminal authority and sessions independently of mounted views."""

    def __init__(
        self,
        read_permitted: Callable[[], object],
        backend_factory: Callable[[], TerminalBackend],
        *,
        monotonic_clock: Callable[[], float] = monotonic,
        screen_model_factory: Callable[[int, int], Any] | None = None,
    ) -> None:
        self._read_permitted = read_permitted
        self._backend_factory = backend_factory
        self._clock = monotonic_clock
        self._screen_model_factory = screen_model_factory
        self._lock = RLock()
        self._armed = False
        self._disclosure_acknowledged = False
        self._shutting_down = False
        self._shutdown_finalized = False
        self._sessions: dict[str, _SessionRecord] = {}
        self._selected_session_id: str | None = None
        self._view_generation = 0
        self._current_view: TerminalViewToken | None = None
        self._subscription_generation = 0
        self._subscriptions: dict[TerminalSubscriptionToken, Callable[[], None]] = {}
        self._cleanup_executor = ThreadPoolExecutor(
            max_workers=MAX_SESSION_RECORDS,
            thread_name_prefix="terminal-cleanup",
        )

    @property
    def discoverable(self) -> bool:
        """Return whether the Terminal feature is discoverable."""
        return True

    @property
    def permitted(self) -> bool:
        """Return the latest strict persisted unlock state."""
        try:
            permitted = self._read_permitted() is True
        except Exception:
            permitted = False
        if not permitted:
            self._revoke_persisted_unlock()
        return permitted

    @property
    def armed(self) -> bool:
        """Return the launch-local terminal arm bit."""
        with self._lock:
            return self._armed

    @property
    def disclosure_acknowledged(self) -> bool:
        """Return whether this manager observed the full disclosure."""
        with self._lock:
            return self._disclosure_acknowledged

    def arm(self, *, acknowledge_disclosure: bool = False) -> TerminalArmResult:
        """Arm Terminal for this launch after strict unlock and disclosure."""
        with self._lock:
            if not self.permitted:
                self._armed = False
                result = TerminalArmResult(reason=TerminalReason.LOCKED)
            elif not self._disclosure_acknowledged:
                if acknowledge_disclosure is not True:
                    result = TerminalArmResult(disclosure_required=True)
                else:
                    self._disclosure_acknowledged = True
                    self._armed = True
                    result = TerminalArmResult(armed=True)
            else:
                self._armed = True
                result = TerminalArmResult(armed=True)
        self._notify_subscribers()
        return result

    def disarm(self) -> None:
        """Clear authority first, then start one concurrent cleanup cohort."""
        with self._lock:
            self._armed = False
            t0 = self._clock()
            session_ids = tuple(self._sessions)
            for session_id in session_ids:
                self._begin_cleanup_locked(session_id, action="disarm", t0=t0)
        self._notify_subscribers()

    def create_session(self, request: TerminalLaunchRequest) -> TerminalCreateResult:
        """Atomically reserve and start one admitted terminal session.

        Args:
            request: Validated name, shell, directory, and initial dimensions.

        Returns:
            Content-free admission result and immutable projection when admitted.
        """
        from .io_actors import TerminalInputActor, TerminalOutputActor
        from .launch import normalize_session_name

        with self._lock:
            refusal = self._creation_refusal_locked()
            if refusal is not None:
                return TerminalCreateResult(reason=refusal)
            try:
                name = normalize_session_name(
                    request.name,
                    existing_names=(
                        record.projection.name for record in self._sessions.values()
                    ),
                )
            except (TypeError, ValueError):
                return TerminalCreateResult(reason=TerminalReason.INVALID_NAME)

            session_id = uuid4().hex
            projection = TerminalProjection(
                session_id=session_id,
                name=name,
                lifecycle=TerminalLifecycle.RESERVED,
            )
            admitted_request = replace(request, name=name)
            self._sessions[session_id] = _SessionRecord(
                projection=projection,
                request=admitted_request,
            )
            self._replace_lifecycle_locked(session_id, TerminalLifecycle.CREATING)

        try:
            backend = self._backend_factory()
        except Exception:
            self._release_failed_reservation(session_id)
            return TerminalCreateResult(reason=TerminalReason.BACKEND_UNAVAILABLE)

        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return TerminalCreateResult(reason=TerminalReason.ADMISSION_FAILED)
            authority_reason = self._authority_refusal_locked()
            if authority_reason is not None:
                self._sessions.pop(session_id, None)
                self._select_after_removal_locked(session_id)
                return TerminalCreateResult(reason=authority_reason)
            record.backend = backend
            try:
                record.model = self._make_screen_model(
                    admitted_request.columns,
                    admitted_request.rows,
                )
                record.input_actor = TerminalInputActor(clock=self._clock)
                record.output_actor = TerminalOutputActor(clock=self._clock)
            except Exception:
                self._sessions.pop(session_id, None)
                return TerminalCreateResult(reason=TerminalReason.BACKEND_UNAVAILABLE)
            self._replace_lifecycle_locked(session_id, TerminalLifecycle.ADMITTING)
            startup_done = record.startup_done

        admission = AdmissionGate(admitted=True, token=session_id)
        startup_failed = False
        try:
            identity = backend.start(admitted_request, admission)
            if identity.session_id != session_id:
                raise RuntimeError("backend returned a mismatched session identity")
        except Exception:
            startup_failed = True
        finally:
            startup_done.set()

        if startup_failed:
            with self._lock:
                record = self._sessions.get(session_id)
                if (
                    record is not None
                    and record.backend is backend
                    and record.cleanup_future is None
                ):
                    self._begin_cleanup_locked(
                        session_id,
                        action="spawn_failed",
                        t0=self._clock(),
                    )
            self._notify_subscribers()
            return TerminalCreateResult(reason=TerminalReason.SPAWN_FAILED)

        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return TerminalCreateResult(reason=TerminalReason.ADMISSION_FAILED)
            authority_reason = self._authority_refusal_locked()
            if (
                authority_reason is not None
                or record.projection.lifecycle is not TerminalLifecycle.ADMITTING
            ):
                if record.cleanup_future is None:
                    self._begin_cleanup_locked(
                        session_id,
                        action="authority_revoked",
                        t0=self._clock(),
                    )
                return TerminalCreateResult(
                    reason=authority_reason or TerminalReason.ADMISSION_FAILED
                )
            self._replace_lifecycle_locked(session_id, TerminalLifecycle.RUNNING)
            if self._selected_session_id is None:
                self._selected_session_id = session_id
            result = TerminalCreateResult(admitted=True, projection=record.projection)
        self._notify_subscribers()
        return result

    def projections(self) -> tuple[TerminalProjection, ...]:
        """Return immutable UI-safe projections for retained sessions."""
        with self._lock:
            return tuple(record.projection for record in self._sessions.values())

    def projection(self, session_id: str) -> TerminalProjection | None:
        """Return one immutable projection when retained."""
        with self._lock:
            record = self._sessions.get(session_id)
            return None if record is None else record.projection

    def shell_exited(
        self, session_id: str, *, exit_code: int
    ) -> TerminalReceipt | None:
        """Start bounded settlement when the exact shell exits."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or record.projection.lifecycle not in {
                TerminalLifecycle.RUNNING,
                TerminalLifecycle.CLOSING,
                TerminalLifecycle.CLEANUP_UNPROVEN,
            }:
                return None
            record.projection = apply_event(
                record.projection,
                TerminalEvent("shell_exit", exit_code=exit_code),
            )
            receipt = self._begin_cleanup_locked(
                session_id,
                action="shell_exit",
                t0=self._clock(),
                transition=False,
            )
        self._notify_subscribers()
        return receipt

    def cleanup_receipt(self, session_id: str) -> TerminalReceipt | None:
        """Return immutable retained cleanup metadata."""
        with self._lock:
            record = self._sessions.get(session_id)
            return None if record is None else record.receipt

    def output_actor_accounting_for_tests(self, session_id: str) -> tuple[int, int]:
        """Return pending output bytes and bounded next-read credit for tests.

        Args:
            session_id: Opaque retained session identity.

        Returns:
            Pending actor bytes and the maximum safe next backend-read size.
        """
        with self._lock:
            record = self._sessions.get(session_id)
            actor = None if record is None else record.output_actor
        if actor is None:
            return 0, 0
        return actor.pending_bytes, actor.next_read_size

    def cleanup_deadline(self, session_id: str) -> TerminalCleanupDeadline | None:
        """Return absolute stage boundaries for the retained attempt."""
        receipt = self.cleanup_receipt(session_id)
        if receipt is None:
            return None
        schedule = CleanupSchedule()
        t0 = receipt.attempt.t0
        return TerminalCleanupDeadline(
            attempt=receipt.attempt,
            hangup_at=t0 + schedule.hangup_no_later_than,
            terminate_at=t0 + schedule.terminate_no_later_than,
            force_kill_at=t0 + schedule.force_kill_no_later_than,
            deadline_at=t0 + schedule.deadline_seconds,
        )

    def wait_for_cleanup(self, session_id: str, *, timeout_seconds: float) -> bool:
        """Wait only for the record's already-authoritative cleanup task."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or record.cleanup_future is None:
                return True
            future = record.cleanup_future
        try:
            future.result(timeout=timeout_seconds)
        except TimeoutError:
            return False
        return True

    def retry_cleanup(
        self,
        session_id: str,
        *,
        view: TerminalViewToken,
    ) -> TerminalReceipt | None:
        """Start the sole user-authorized fresh cleanup deadline."""
        with self._lock:
            if not self._valid_view_locked(view):
                return None
            record = self._sessions.get(session_id)
            if (
                record is None
                or record.projection.lifecycle is not TerminalLifecycle.CLEANUP_UNPROVEN
                or (
                    record.cleanup_future is not None
                    and not record.cleanup_future.done()
                )
            ):
                return None
            receipt = self._begin_cleanup_locked(
                session_id,
                action="retry",
                t0=self._clock(),
            )
        if receipt is not None:
            self._notify_subscribers()
        return receipt

    async def shutdown(self, *, deadline_seconds: float = 5.0) -> bool:
        """Bound app shutdown while all retained sessions clean concurrently."""
        if not isinstance(deadline_seconds, (int, float)) or isinstance(
            deadline_seconds, bool
        ):
            raise TypeError("deadline_seconds must be a number")
        if deadline_seconds < 0:
            raise ValueError("deadline_seconds must not be negative")
        with self._lock:
            self._shutting_down = True
            self._armed = False
            t0 = self._clock()
            for session_id in tuple(self._sessions):
                self._begin_cleanup_locked(session_id, action="shutdown", t0=t0)
            futures = tuple(
                record.cleanup_future
                for record in self._sessions.values()
                if record.cleanup_future is not None
            )
        if futures:
            wrapped = [asyncio.wrap_future(future) for future in futures]
            done, pending = await asyncio.wait(wrapped, timeout=float(deadline_seconds))
            del done
            if pending:
                return False
        with self._lock:
            return not self._sessions

    def finalize_shutdown(self) -> None:
        """Close remaining app-owned backend handles without another wait."""
        with self._lock:
            if self._shutdown_finalized:
                return
            self._shutdown_finalized = True
            backends = tuple(
                {
                    id(record.backend): record.backend
                    for record in self._sessions.values()
                    if record.backend is not None
                }.values()
            )
        for backend in backends:
            try:
                backend.finalize_shutdown()
            except Exception:
                continue
        self._cleanup_executor.shutdown(wait=False, cancel_futures=True)

    def accepts_input(self, session_id: str) -> bool:
        """Return whether a running record currently accepts input."""
        with self._lock:
            record = self._sessions.get(session_id)
            return (
                record is not None
                and record.projection.lifecycle is TerminalLifecycle.RUNNING
                and not record.projection.parser_failed
                and self._armed
                and self.permitted
            )

    def offer_output(self, session_id: str, data: bytes) -> OutputOfferResult:
        """Offer bounded backend bytes to a healthy retained parser path."""
        from .io_actors import OutputOfferResult

        with self._lock:
            record = self._sessions.get(session_id)
            actor = None if record is None else record.output_actor
            output_refused = (
                record is None
                or record.projection.parser_failed
                or record.projection.stream_closed
            )
        if actor is None or output_refused:
            return OutputOfferResult()
        return actor.offer_output(data)

    def process_output(
        self, session_id: str, *, visible: bool
    ) -> ParserTurnResult | None:
        """Run one bounded parser turn and fail closed on parser exceptions."""
        with self._lock:
            record = self._sessions.get(session_id)
            if (
                record is None
                or record.output_actor is None
                or record.model is None
                or record.projection.parser_failed
            ):
                return None
            actor = record.output_actor
            model = record.model
            model_lock = record.model_lock
        try:
            with model_lock:
                result = actor.process_parser_turn(model.feed, visible=visible)
                replies = self._take_model_replies(model)
                failure_reason = getattr(model, "failure_reason", None)
        except Exception:
            self.parser_failed(session_id)
            return None
        if failure_reason is TerminalReason.TERMINAL_PROTOCOL_FAILED:
            self.parser_failed(session_id)
            return None
        self._queue_model_replies(session_id, model, replies)
        self._notify_subscribers()
        return result

    def parser_failed(self, session_id: str) -> TerminalReceipt | None:
        """Disable input/repaint and request out-of-band cleanup immediately."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None or record.projection.parser_failed:
                return None if record is None else record.receipt
            record.projection = apply_event(
                record.projection,
                TerminalEvent(
                    "parser_failure",
                    reason=TerminalReason.TERMINAL_PROTOCOL_FAILED,
                ),
            )
            backend = record.backend
            cleanup_active = (
                record.cleanup_future is not None and not record.cleanup_future.done()
            )
        if backend is not None and not cleanup_active:
            try:
                backend.request_priority_close()
            except Exception:
                pass
        with self._lock:
            receipt = self._begin_cleanup_locked(
                session_id,
                action="parser_failure",
                t0=self._clock(),
                transition=False,
            )
        self._notify_subscribers()
        return receipt

    @property
    def selected_session_id(self) -> str | None:
        """Return the selected opaque session identity."""
        with self._lock:
            return self._selected_session_id

    def attach_view(self) -> TerminalViewToken:
        """Attach one new view generation, invalidating the previous one."""
        with self._lock:
            self._view_generation += 1
            self._current_view = TerminalViewToken(self._view_generation)
            return self._current_view

    def detach_view(self, view: TerminalViewToken) -> bool:
        """Detach only the current generation."""
        with self._lock:
            if not self._valid_view_locked(view):
                return False
            self._current_view = None
            return True

    def view_state(self, view: TerminalViewToken) -> TerminalViewState | None:
        """Return immutable projections only to the current generation."""
        from .screen_model import TerminalScreenSnapshot

        with self._lock:
            if not self._valid_view_locked(view):
                return None
            candidates = tuple(
                (session_id, record, record.model, record.model_lock)
                for session_id, record in self._sessions.items()
                if record.model is not None
            )

        snapshots: dict[str, TerminalScreenSnapshot] = {}
        for session_id, _record, model, model_lock in candidates:
            try:
                with model_lock:
                    snapshot = model.snapshot()
            except Exception:
                self.parser_failed(session_id)
                return None
            if isinstance(snapshot, TerminalScreenSnapshot):
                snapshots[session_id] = snapshot

        with self._lock:
            if not self._valid_view_locked(view):
                return None
            if any(
                self._sessions.get(session_id) is not record
                or record.model is not model
                for session_id, record, model, _model_lock in candidates
            ):
                return None
            sessions = tuple(
                TerminalSessionView(
                    projection=record.projection,
                    screen=snapshots[session_id],
                    shell=record.request.shell,
                    start_directory=record.request.start_directory,
                    columns=record.request.columns,
                    rows=record.request.rows,
                    cleanup_receipt=record.receipt,
                )
                for session_id, record, _model, _model_lock in candidates
                if session_id in snapshots
            )
            return TerminalViewState(
                selected_session_id=self._selected_session_id,
                sessions=sessions,
            )

    def resize_session(
        self,
        session_id: str,
        *,
        columns: int,
        rows: int,
        view: TerminalViewToken,
    ) -> bool:
        """Offer a coalesced resize from the current mounted generation."""
        with self._lock:
            if not self._valid_view_locked(view):
                return False
            record = self._sessions.get(session_id)
            if (
                record is None
                or record.backend is None
                or record.model is None
                or record.input_actor is None
                or not self._armed
                or not self.permitted
                or record.projection.lifecycle is not TerminalLifecycle.RUNNING
            ):
                return False
            try:
                record.input_actor.offer_resize(columns=columns, rows=rows)
            except (TypeError, ValueError):
                return False
        return True

    async def apply_pending_resize(
        self,
        session_id: str,
        *,
        view: TerminalViewToken,
    ) -> bool:
        """Apply the latest debounced resize without holding manager authority."""
        with self._lock:
            if not self._valid_view_locked(view):
                return False
            record = self._sessions.get(session_id)
            if record is None or record.input_actor is None:
                return False
            actor = record.input_actor
            resize_lock = record.resize_lock

        async with resize_lock:
            with self._lock:
                if not self._valid_view_locked(view):
                    return False
                record = self._sessions.get(session_id)
                if (
                    record is None
                    or record.input_actor is not actor
                    or record.resize_lock is not resize_lock
                ):
                    return False

            resize = await actor.take_resize_debounced()
            if resize is None:
                return False

            with self._lock:
                if not self._valid_view_locked(view):
                    return False
                record = self._sessions.get(session_id)
                if (
                    record is None
                    or record.input_actor is not actor
                    or record.resize_lock is not resize_lock
                    or record.backend is None
                    or record.model is None
                    or not self._armed
                    or not self.permitted
                    or record.projection.lifecycle is not TerminalLifecycle.RUNNING
                ):
                    return False
                backend = record.backend
                model = record.model
                model_lock = record.model_lock

            applied = await asyncio.to_thread(
                self._apply_resize,
                backend,
                model,
                model_lock,
                resize.columns,
                resize.rows,
            )
            if not applied:
                self._fail_resize(session_id, backend=backend, model=model)
                return False

            with self._lock:
                record = self._sessions.get(session_id)
                if (
                    record is None
                    or record.backend is not backend
                    or record.model is not model
                ):
                    return False
                record.request = replace(
                    record.request,
                    columns=resize.columns,
                    rows=resize.rows,
                )
                remains_current = (
                    self._valid_view_locked(view)
                    and record.projection.lifecycle is TerminalLifecycle.RUNNING
                )
        self._notify_subscribers()
        return remains_current

    def focus_session(self, session_id: str, *, view: TerminalViewToken) -> bool:
        """Select a retained session only from the current view generation."""
        with self._lock:
            if not self._valid_view_locked(view) or session_id not in self._sessions:
                return False
            self._selected_session_id = session_id
        self._notify_subscribers()
        return True

    def close_session(
        self,
        session_id: str,
        *,
        view: TerminalViewToken,
    ) -> TerminalReceipt | None:
        """Start bounded close only for the current view generation."""
        with self._lock:
            if not self._valid_view_locked(view):
                return None
            receipt = self._begin_cleanup_locked(
                session_id,
                action="close",
                t0=self._clock(),
            )
        self._notify_subscribers()
        return receipt

    def managed_process_inventory_for_tests(
        self,
    ) -> tuple[ManagedProcessIdentity, ...]:
        """Aggregate only PID/birth identities from retained backends."""
        with self._lock:
            backends = tuple(
                record.backend
                for record in self._sessions.values()
                if record.backend is not None
            )
        inventory: list[ManagedProcessIdentity] = []
        for backend in backends:
            reader = getattr(backend, "managed_process_inventory_for_tests", None)
            if not callable(reader):
                continue
            try:
                identities = reader()
            except Exception:
                continue
            if not isinstance(identities, tuple):
                continue
            inventory.extend(
                identity
                for identity in identities
                if isinstance(identity, ManagedProcessIdentity)
            )
        return tuple(inventory)

    def rename_session(
        self,
        session_id: str,
        name: str,
        *,
        view: TerminalViewToken,
    ) -> bool:
        """Rename a retained record under normalized unique-name policy."""
        from .launch import normalize_session_name

        with self._lock:
            if not self._valid_view_locked(view):
                return False
            record = self._sessions.get(session_id)
            if record is None:
                return False
            try:
                normalized = normalize_session_name(
                    name,
                    existing_names=(
                        other.projection.name
                        for other_id, other in self._sessions.items()
                        if other_id != session_id
                    ),
                )
            except (TypeError, ValueError):
                return False
            record.projection = replace(record.projection, name=normalized)
            record.request = replace(record.request, name=normalized)
        self._notify_subscribers()
        return True

    def send_key(
        self,
        session_id: str,
        data: bytes,
        *,
        view: TerminalViewToken,
    ) -> InputOfferResult:
        """Offer one key event through the manager-owned bounded input actor."""
        from .io_actors import InputOfferResult

        with self._lock:
            actor = self._input_actor_for_view_locked(session_id, view)
            if actor is None:
                return InputOfferResult()
            return actor.offer_key(data)

    def send_paste(
        self,
        session_id: str,
        text: str,
        *,
        bracketed: bool,
        view: TerminalViewToken,
    ) -> InputOfferResult:
        """Offer one atomic paste through the manager-owned input actor."""
        from .io_actors import InputOfferResult

        with self._lock:
            actor = self._input_actor_for_view_locked(session_id, view)
            if actor is None:
                return InputOfferResult()
            return actor.offer_paste(text, bracketed=bracketed)

    def take_input(self, session_id: str) -> TerminalInputEvent | None:
        """Let the owned backend pull one ordered event while input is live."""
        with self._lock:
            record = self._sessions.get(session_id)
            if (
                record is None
                or record.input_actor is None
                or record.projection.lifecycle is not TerminalLifecycle.RUNNING
                or not self._armed
                or not self.permitted
            ):
                return None
            actor = record.input_actor
        return actor.take_nowait()

    def subscribe(self, callback: Callable[[], None]) -> TerminalSubscriptionToken:
        """Register a content-free callback token."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            self._subscription_generation += 1
            token = TerminalSubscriptionToken(self._subscription_generation)
            self._subscriptions[token] = callback
            return token

    def unsubscribe(self, subscription: TerminalSubscriptionToken) -> bool:
        """Remove one content-free change subscription."""
        with self._lock:
            return self._subscriptions.pop(subscription, None) is not None

    def _creation_refusal_locked(self) -> TerminalReason | None:
        if self._shutting_down:
            self._armed = False
            return TerminalReason.UNARMED
        authority_reason = self._authority_refusal_locked()
        if authority_reason is not None:
            return authority_reason
        if len(self._sessions) >= MAX_SESSION_RECORDS:
            return TerminalReason.SESSION_LIMIT
        return None

    def _revoke_persisted_unlock(self) -> None:
        """Apply persisted-lock revocation with the same cleanup as Disarm."""
        with self._lock:
            self._armed = False
            if not self._sessions:
                return
            t0 = self._clock()
            for session_id in tuple(self._sessions):
                self._begin_cleanup_locked(
                    session_id,
                    action="lock_revoked",
                    t0=t0,
                )

    def _authority_refusal_locked(self) -> TerminalReason | None:
        if not self.permitted:
            self._armed = False
            return TerminalReason.LOCKED
        if not self._armed:
            return TerminalReason.UNARMED
        return None

    def _replace_lifecycle_locked(
        self, session_id: str, lifecycle: TerminalLifecycle
    ) -> None:
        record = self._sessions[session_id]
        record.projection = replace(record.projection, lifecycle=lifecycle)

    def _release_failed_reservation(self, session_id: str) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return
            if record.cleanup_future is not None or record.projection.lifecycle in {
                TerminalLifecycle.CLOSING,
                TerminalLifecycle.CLEANUP_UNPROVEN,
            }:
                return
            self._sessions.pop(session_id, None)
            self._select_after_removal_locked(session_id)

    def _make_screen_model(self, columns: int, rows: int) -> Any:
        if self._screen_model_factory is not None:
            return self._screen_model_factory(columns, rows)
        from .screen_model import TerminalScreenModel

        return TerminalScreenModel(columns=columns, rows=rows)

    def _begin_cleanup_locked(
        self,
        session_id: str,
        *,
        action: str,
        t0: float,
        transition: bool = True,
    ) -> TerminalReceipt | None:
        record = self._sessions.get(session_id)
        if record is None:
            return None
        if record.backend is None:
            self._sessions.pop(session_id, None)
            self._select_after_removal_locked(session_id)
            return None
        if (
            record.projection.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN
            and action != "retry"
        ):
            return record.receipt

        current_future = record.cleanup_future
        if current_future is not None and not current_future.done():
            if action != "shell_exit":
                if record.cleanup_action == "shell_exit":
                    self._request_priority_close(record.backend)
                record.cleanup_action = action
                if record.projection.lifecycle is not TerminalLifecycle.CLOSING:
                    record.projection = replace(
                        record.projection,
                        lifecycle=TerminalLifecycle.CLOSING,
                    )
                existing_t0 = (
                    record.receipt.attempt.t0 if record.receipt is not None else t0
                )
                record.receipt = TerminalReceipt(
                    CleanupAttempt(min(existing_t0, t0)),
                    action,
                )
            return record.receipt

        if transition:
            if record.projection.lifecycle is TerminalLifecycle.CLEANUP_UNPROVEN:
                record.projection = replace(
                    record.projection,
                    lifecycle=TerminalLifecycle.CLOSING,
                )
            elif record.projection.lifecycle in {
                TerminalLifecycle.RESERVED,
                TerminalLifecycle.CREATING,
                TerminalLifecycle.ADMITTING,
            }:
                record.projection = replace(
                    record.projection,
                    lifecycle=TerminalLifecycle.CLOSING,
                )
            else:
                record.projection = apply_event(
                    record.projection,
                    TerminalEvent("close"),
                )
        attempt = CleanupAttempt(t0)
        receipt = TerminalReceipt(attempt, action)
        record.receipt = receipt
        record.cleanup_action = action
        if action != "shell_exit" and action != "parser_failure":
            self._request_priority_close(record.backend)
        future = self._cleanup_executor.submit(
            self._run_cleanup,
            session_id,
            record.backend,
            record.startup_done,
            attempt,
        )
        record.cleanup_future = future
        return receipt

    def _run_cleanup(
        self,
        session_id: str,
        backend: TerminalBackend,
        startup_done: Event,
        attempt: CleanupAttempt,
    ) -> None:
        startup_done.wait()
        with self._lock:
            record = self._sessions.get(session_id)
            parser_failed = record is not None and record.projection.parser_failed
        parser_failure_cleanup = getattr(backend, "cleanup_parser_failure", None)
        try:
            if parser_failed and callable(parser_failure_cleanup):
                proof = parser_failure_cleanup(attempt)
            else:
                proof = backend.cleanup(attempt)
            if not isinstance(proof, CleanupProof):
                proof = CleanupProof()
        except Exception:
            proof = CleanupProof()

        if proof.process_dead and proof.stream_closed:
            handoff_complete = self._handoff_cleanup_output_at_eof(
                session_id,
                backend,
            )
            parser_complete = handoff_complete and self._finalize_output_at_eof(
                session_id
            )
            proof = CleanupProof(
                process_dead=True,
                stream_closed=True,
                output_complete=proof.output_complete and parser_complete,
            )

        with self._lock:
            record = self._sessions.get(session_id)
            parser_failed = record is not None and record.projection.parser_failed
        if parser_failed and proof.process_dead and not proof.stream_closed:
            raw_drain = getattr(backend, "cleanup_raw_drain", None)
            if callable(raw_drain):
                try:
                    raw_proof = raw_drain(attempt)
                except Exception:
                    raw_proof = CleanupProof(process_dead=True)
                if isinstance(raw_proof, CleanupProof):
                    proof = CleanupProof(
                        process_dead=proof.process_dead and raw_proof.process_dead,
                        stream_closed=raw_proof.stream_closed,
                        output_complete=False,
                    )
        self._settle_cleanup(session_id, proof)

    def _handoff_cleanup_output_at_eof(
        self,
        session_id: str,
        backend: TerminalBackend,
    ) -> bool:
        """Move bounded backend-preserved bytes into the retained output actor."""
        take_preserved = getattr(backend, "take_preserved_cleanup_output", None)
        if not callable(take_preserved):
            return True
        while True:
            with self._lock:
                record = self._sessions.get(session_id)
                if (
                    record is None
                    or record.output_actor is None
                    or record.model is None
                    or record.projection.parser_failed
                ):
                    return False
                actor = record.output_actor
                maximum = actor.next_read_size
                if maximum:
                    try:
                        chunk = take_preserved(maximum)
                    except Exception:
                        return False
                    if not isinstance(chunk, bytes) or len(chunk) > maximum:
                        return False
                    if not chunk:
                        return True
                    if not actor.offer_output(chunk).accepted:
                        return False
                    continue
            turn = self.process_output(session_id, visible=False)
            if turn is None or turn.processed_bytes <= 0:
                return False

    def _finalize_output_at_eof(self, session_id: str) -> bool:
        """Drain admitted bytes and finalize decoding before claiming completeness."""
        with self._lock:
            record = self._sessions.get(session_id)
            if (
                record is None
                or record.output_actor is None
                or record.model is None
                or record.projection.parser_failed
            ):
                return False
            actor = record.output_actor
            model = record.model
            model_lock = record.model_lock

        actor.close_output()

        with self._lock:
            record = self._sessions.get(session_id)
            if (
                record is None
                or record.output_actor is not actor
                or record.model is not model
                or record.projection.parser_failed
            ):
                return False
            record.projection = replace(record.projection, stream_closed=True)

        try:
            while actor.pending_bytes:
                with model_lock:
                    actor.process_parser_turn(model.feed, visible=False)
                    self._take_model_replies(model)
                    if (
                        getattr(model, "failure_reason", None)
                        is TerminalReason.TERMINAL_PROTOCOL_FAILED
                    ):
                        raise RuntimeError("terminal protocol failed")
            with model_lock:
                finish = getattr(model, "finish", None)
                if callable(finish):
                    finish()
                self._take_model_replies(model)
                if (
                    getattr(model, "failure_reason", None)
                    is TerminalReason.TERMINAL_PROTOCOL_FAILED
                ):
                    raise RuntimeError("terminal protocol failed")
        except Exception:
            self.parser_failed(session_id)
            return False

        with self._lock:
            record = self._sessions.get(session_id)
            return (
                record is not None
                and record.output_actor is actor
                and record.model is model
                and not record.projection.parser_failed
                and actor.pending_bytes == 0
            )

    def _settle_cleanup(self, session_id: str, proof: CleanupProof) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return
            action = record.cleanup_action
            parser_failed = record.projection.parser_failed
            if action == "shell_exit" and proof.process_dead and proof.stream_closed:
                record.projection = replace(
                    record.projection,
                    lifecycle=TerminalLifecycle.EXITED,
                    stream_closed=proof.stream_closed,
                    output_complete=proof.output_complete and not parser_failed,
                )
                record.cleanup_future = None
            elif proof.process_dead and proof.stream_closed:
                self._sessions.pop(session_id, None)
                self._select_after_removal_locked(session_id)
            else:
                record.projection = replace(
                    record.projection,
                    lifecycle=TerminalLifecycle.CLEANUP_UNPROVEN,
                    reason=TerminalReason.CLEANUP_UNPROVEN,
                    stream_closed=proof.stream_closed,
                    output_complete=False,
                )
                record.cleanup_future = None
        self._notify_subscribers()

    def _valid_view_locked(self, view: TerminalViewToken) -> bool:
        return type(view) is TerminalViewToken and view == self._current_view

    @staticmethod
    def _request_priority_close(backend: TerminalBackend) -> None:
        try:
            backend.request_priority_close()
        except Exception:
            pass

    @staticmethod
    def _apply_resize(
        backend: TerminalBackend,
        model: Any,
        model_lock: Any,
        columns: int,
        rows: int,
    ) -> bool:
        """Resize backend and model without blocking manager or event-loop locks."""
        try:
            backend.resize(columns, rows)
            with model_lock:
                model.resize(columns=columns, rows=rows)
        except Exception:
            return False
        return True

    def _fail_resize(
        self,
        session_id: str,
        *,
        backend: TerminalBackend,
        model: Any,
    ) -> None:
        """Fail and clean a session whose backend/model resize did not agree."""
        with self._lock:
            record = self._sessions.get(session_id)
            if (
                record is None
                or record.backend is not backend
                or record.model is not model
                or record.projection.lifecycle is not TerminalLifecycle.RUNNING
            ):
                return
            record.projection = replace(
                record.projection,
                reason=TerminalReason.IO_FAILED,
                output_complete=False,
            )
            self._begin_cleanup_locked(
                session_id,
                action="resize_failure",
                t0=self._clock(),
            )
        self._notify_subscribers()

    def _select_after_removal_locked(self, removed_session_id: str) -> None:
        if self._selected_session_id != removed_session_id:
            return
        self._selected_session_id = next(iter(self._sessions), None)

    def _input_actor_for_view_locked(
        self,
        session_id: str,
        view: TerminalViewToken,
    ) -> TerminalInputActor | None:
        if not self._valid_view_locked(view):
            return None
        record = self._sessions.get(session_id)
        if (
            record is None
            or record.input_actor is None
            or record.projection.lifecycle is not TerminalLifecycle.RUNNING
            or record.projection.parser_failed
            or not self._armed
            or not self.permitted
        ):
            return None
        return record.input_actor

    @staticmethod
    def _take_model_replies(model: Any) -> tuple[bytes, ...]:
        take_replies = getattr(model, "take_pending_replies", None)
        if not callable(take_replies):
            return ()
        replies = take_replies()
        if not isinstance(replies, tuple):
            raise TypeError("terminal model replies must be an immutable tuple")
        if any(type(reply) is not bytes for reply in replies):
            raise TypeError("terminal model replies must contain only bytes")
        return replies

    def _queue_model_replies(
        self,
        session_id: str,
        model: Any,
        replies: tuple[bytes, ...],
    ) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if (
                record is None
                or record.model is not model
                or record.input_actor is None
                or record.projection.lifecycle is not TerminalLifecycle.RUNNING
            ):
                return
            for reply in replies:
                record.input_actor.offer_reply(reply)

    def _notify_subscribers(self) -> None:
        with self._lock:
            callbacks = tuple(self._subscriptions.values())
        for callback in callbacks:
            try:
                callback()
            except Exception:
                continue
