"""Bounded, app-owned Change Review baseline/finalization coordination."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

from tldw_chatbook.Workspaces.change_review_models import (
    ChangeReviewCompletion as ChangeReviewCompletion,
    ChangeReviewReservation as ChangeReviewReservation,
)
from tldw_chatbook.Workspaces.change_turn_tracker import (
    BaselineRootPreparation,
    ChangeTurnTracker,
    TurnChangeRecord,
    TurnHandle,
)


@dataclass(frozen=True)
class ChangeReviewPublicationSnapshot:
    """Payload-free state used by UI polling."""

    revision: int
    pending: int


class ChangeReviewPublicationSignal:
    """Thread-safe content-free revision and pending-work counter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._revision = 0
        self._pending = 0

    def snapshot(self) -> ChangeReviewPublicationSnapshot:
        with self._lock:
            return ChangeReviewPublicationSnapshot(self._revision, self._pending)

    def admitted(self) -> None:
        with self._lock:
            self._pending += 1

    def completed(self, *, published: bool) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            if published:
                self._revision += 1

    def anchor_published(self) -> None:
        """Advance after a durable assistant anchor is written."""
        with self._lock:
            self._revision += 1

    def window_published(self) -> None:
        """Advance after one window publishes while its survivor remains pending."""
        with self._lock:
            self._revision += 1


class ChangeReviewFinalizeResult(Enum):
    """Typed terminal-admission outcome for one review reservation."""

    SCHEDULED = "scheduled"
    OVERLOAD_VISIBLE = "overload_visible"
    REJECTED = "rejected"

    def __bool__(self) -> bool:
        return self is ChangeReviewFinalizeResult.SCHEDULED


@dataclass(frozen=True)
class ChangeReviewPublication:
    """One complete durable-publication request."""

    reservation_id: str
    run_id: str
    kind: str
    records: tuple[TurnChangeRecord, ...]
    roots: tuple[str, ...] = ()


@dataclass
class _ReservationState:
    public: ChangeReviewReservation
    native_source: object = None
    pending_operation: object = None
    pending_reason: str = "ready"
    phase: str = "waiting"
    operation_inflight: bool = False
    final_requested: bool = False
    cancelled: bool = False
    run_id: str = ""
    kind: str = "turn"
    touched_paths: tuple[str, ...] = ()
    survivor_touched_paths: tuple[str, ...] = ()
    end_shas: dict[str, str] | None = None
    survivor_key: str = ""
    has_live_survivors: bool = False
    survivors_settled: bool = False
    active_handle: TurnHandle | None = None
    roots: tuple[str, ...] = ()
    sequence: int = 0
    preparations: tuple[BaselineRootPreparation, ...] = ()
    lane_ordered: bool = True
    admission_error: str = ""
    attribution_invalid_roots: set[str] = field(default_factory=set)
    survivor_window_opened: bool = False
    survivor_window_closed: bool = False
    baseline_timed_out: bool = False
    baseline_timeout_error: str = ""


@dataclass
class _DegradedRootState:
    reservation_ids: set[str] = field(default_factory=set)
    survivor_keys: set[str] = field(default_factory=set)


@dataclass
class _DegradedReservationState:
    roots: tuple[str, ...]
    survivor_key: str
    survivors_settled: bool = False


@dataclass(frozen=True)
class _Operation:
    reservation_id: str
    kind: str
    handle: TurnHandle
    touched_paths: tuple[str, ...] = ()
    end_shas: dict[str, str] | None = None
    preparations: tuple[BaselineRootPreparation, ...] = ()
    ticket: object = None


@dataclass(frozen=True)
class _OperationResult:
    reservation_id: str
    kind: str
    records: tuple[TurnChangeRecord, ...] = ()
    error: str = ""
    preparations: tuple[BaselineRootPreparation, ...] = ()
    ticket: object = None
    started: bool = True


@dataclass(frozen=True)
class _DirectPublication:
    publication: ChangeReviewPublication


_STOP = object()
_RETRY = object()


class ChangeReviewFinalizationCoordinator:
    """Serialize B/E windows per canonical root using fixed daemon workers."""

    def __init__(
        self,
        *,
        tracker: ChangeTurnTracker,
        publish: Callable[[ChangeReviewPublication], None],
        publication_signal: ChangeReviewPublicationSignal | None = None,
        close_publisher: Callable[[], None] | None = None,
        worker_count: int = 2,
        capacity: int = 32,
        native_owner=None,
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._native = native_owner
        self._retry_reasons: set[str] = set()
        self._release_versions = {"activity": 0, "capacity": 0, "queue": 0}
        self._subscriptions = ()
        self._tracker = tracker
        self._publish = publish
        self._close_publisher = close_publisher
        self.publication_signal = publication_signal or ChangeReviewPublicationSignal()
        self._capacity = capacity
        self._lock = threading.Lock()
        self._lanes: dict[str, deque[str]] = {}
        self._states: dict[str, _ReservationState] = {}
        self._native_retirements: deque = deque()
        self._degraded_roots: dict[str, _DegradedRootState] = {}
        self._degraded_reservations: dict[str, _DegradedReservationState] = {}
        self._accepting = True
        self._next_sequence = 0
        self._idle = threading.Event()
        self._idle.set()
        self._operations: queue.Queue[_Operation | object] = queue.Queue(capacity)
        self._results: queue.Queue[_OperationResult | _DirectPublication | object] = (
            queue.Queue(capacity)
        )
        self._direct_results: queue.Queue[_DirectPublication] = queue.Queue(capacity)
        self._direct_slots = threading.BoundedSemaphore(capacity)
        self._direct_pending = 0
        self._publisher_stop_requested = threading.Event()
        self._publisher_closed = False
        self._workers = tuple(
            threading.Thread(
                target=self._worker_loop,
                name=f"change-review-fs-{index + 1}",
                daemon=True,
            )
            for index in range(worker_count)
        )
        self._publisher = threading.Thread(
            target=self._publisher_loop,
            name="change-review-publisher",
            daemon=True,
        )
        self._started = False
        if self._native is not None:
            self._native.coordinator = self

    def _subscribe_native(self, capacity, activity) -> None:
        """Bind once on first admission, before a reserve can miss a release."""
        if not self._subscriptions:
            self._subscriptions = (
                activity.subscribe(lambda: self._availability("activity")),
                capacity.subscribe_tool_availability(
                    lambda: self._availability("capacity")
                ),
            )

    def register(
        self,
        roots: Sequence[Path | str],
        *,
        survivor_key: str = "",
        native_source=None,
        on_unadmitted: Callable[[], None] | None = None,
    ) -> ChangeReviewReservation | None:
        """Atomically admit one reservation to every canonical-root lane.

        ``roots`` are trusted turn-admission outputs from the workspace registry,
        which validates, canonicalizes, and checks them for symlink drift before
        they enter the Console snapshot. This coordinator converts ``Path`` and
        string representations to lane keys only; it is not another user-input
        boundary.

        Args:
            roots: Registry-admitted workspace roots for one Console turn.
            survivor_key: Optional durable assistant key used to retain lineage.

        Returns:
            The all-roots reservation, or ``None`` when no roots were supplied or
            the coordinator no longer accepts work.
        """
        canonical = tuple(dict.fromkeys(str(root) for root in roots))
        if not canonical:
            return None
        handle = (
            TurnHandle([Path(root) for root in canonical])
            if self._native is not None
            else self._tracker.new_turn_handle(canonical)
        )
        source_error = ""
        if self._native is not None:
            try:
                if native_source is None or native_source.issuer is not self._native:
                    raise ValueError("native change-review source unavailable")
                if (
                    tuple(str(c.canonical_root) for c in native_source.chains)
                    != canonical
                ):
                    raise ValueError("native change-review roots changed")
                native_source.check()
            except Exception:  # noqa: BLE001 - fixed visible refusal, no supplied details
                source_error = "native change-review source or roots unavailable"
        with self._lock:
            if not self._accepting:
                return None
            self._start_threads_locked()
            reservation_id = uuid.uuid4().hex
            if source_error:
                handle.errors.update(dict.fromkeys(canonical, source_error))
                handle._baseline_ready.set()
                return ChangeReviewReservation(
                    reservation_id,
                    canonical,
                    handle,
                    source_error,
                    False,
                    _on_unadmitted=on_unadmitted,
                )
            degraded = tuple(root for root in canonical if root in self._degraded_roots)
            if degraded:
                admission_error = (
                    "change-review root is resynchronizing after a baseline "
                    "timeout; filesystem tracking skipped for this turn"
                )
                for root in canonical:
                    handle.errors[root] = admission_error
                    self._degraded_roots.setdefault(
                        root, _DegradedRootState()
                    ).reservation_ids.add(reservation_id)
                self._degraded_reservations[reservation_id] = _DegradedReservationState(
                    canonical, survivor_key
                )
                handle._baseline_ready.set()
                return ChangeReviewReservation(
                    reservation_id,
                    canonical,
                    handle,
                    admission_error,
                    self._native is None and self._direct_slots.acquire(blocking=False),
                    _on_unadmitted=on_unadmitted,
                )
            if len(self._states) >= self._capacity:
                admission_error = (
                    "change-review coordinator is at capacity; "
                    "filesystem tracking skipped for this turn"
                )
                for root in canonical:
                    handle.errors[root] = admission_error
                handle._baseline_ready.set()
                return ChangeReviewReservation(
                    reservation_id,
                    canonical,
                    handle,
                    admission_error,
                    self._native is None and self._direct_slots.acquire(blocking=False),
                    _on_unadmitted=on_unadmitted,
                )
            public = ChangeReviewReservation(
                reservation_id,
                canonical,
                handle,
                _on_unadmitted=on_unadmitted,
            )
            state = _ReservationState(
                public=public,
                native_source=native_source,
                survivor_key=survivor_key,
                active_handle=handle,
                roots=canonical,
                sequence=self._next_sequence,
                phase="registering"
                if getattr(native_source, "workflow_wave", None) is not None
                else "waiting",
            )
            self._next_sequence += 1
            self._states[reservation_id] = state
            for root in canonical:
                self._lanes.setdefault(root, deque()).append(reservation_id)
            self._idle.clear()
            self.publication_signal.admitted()
            self._schedule_ready_locked()
            return public

    def _cancel_unpublished_workflow(self, reservation, source, refusal):
        """Retire only the exact original inert, conclusively unpublished window."""
        from tldw_chatbook.DB.workflow_runs import _WindowUnpublished

        wave = source.workflow_wave
        if (
            type(refusal) is not _WindowUnpublished
            or wave is None
            or refusal.store is not wave.service.db.workflow_runs
            or refusal.phase is not wave.phase
            or refusal.reservation is not reservation
            or source is not wave.source
            or source.issuer is not self._native
        ):
            raise ValueError("original unpublished workflow window required")
        with self._lock:
            state = self._states.get(reservation.id)
            if (
                state is None
                or state.public is not reservation
                or state.native_source is not source
                or state.phase != "registering"
                or state.operation_inflight
                or state.pending_operation is not None
                or state.cancelled
                or reservation._completion.ticket is not None
                or reservation._baseline_completion.ticket is not None
            ):
                raise ValueError("original unused workflow reservation required")
            state.cancelled = True
            self._complete_locked(state, published=False)
            # _complete_locked appends this original source under the same lock.
            # Its callback must return here, not later in another retirement batch.
            retired = self._native_retirements.pop()
            assert retired is source
            close_unused = retired.close_unused
            self._schedule_ready_locked()
        close_unused()

    def _activate_workflow_reservation(self, reservation, source, publication):
        """Activate the same inert native reservation after its FULL publication."""
        from tldw_chatbook.DB.workflow_runs import _WindowPublication

        wave = source.workflow_wave
        if (
            wave is None
            or type(publication) is not _WindowPublication
            or publication.reservation is not reservation
            or publication is not wave.publication
            or publication.store is not wave.service.db.workflow_runs
            or source is not wave.source
            or source.issuer is not self._native
        ):
            raise ValueError("original workflow window publication required")
        source.check()
        with self._lock:
            state = self._states.get(reservation.id)
            if (
                state is None
                or state.public is not reservation
                or state.native_source is not source
                or state.phase != "registering"
            ):
                raise ValueError("original inert workflow reservation required")
            state.phase = "waiting"
            self._schedule_ready_locked()

    def finalize(
        self,
        reservation: ChangeReviewReservation,
        *,
        run_id: str,
        kind: str,
        touched_paths: Sequence[str] = (),
        end_shas: dict[str, str] | None = None,
        has_live_survivors: bool = False,
    ) -> ChangeReviewFinalizeResult:
        """Request E and return without waiting for filesystem work."""
        if reservation.admission_error:
            if not reservation._claim_terminal():
                return ChangeReviewFinalizeResult.REJECTED
            publication = ChangeReviewPublication(
                reservation_id=reservation.id,
                run_id=run_id,
                kind=kind,
                records=tuple(
                    TurnChangeRecord(
                        root=root,
                        tracking_error=reservation.admission_error,
                    )
                    for root in reservation.roots
                ),
                roots=reservation.roots,
            )
            with self._lock:
                self._finish_degraded_reservation_locked(
                    reservation.id,
                    has_live_survivors=has_live_survivors,
                )
                if not self._accepting:
                    if reservation._publication_reserved:
                        self._direct_slots.release()
                    return ChangeReviewFinalizeResult.REJECTED
                if not reservation._publication_reserved:
                    return ChangeReviewFinalizeResult.OVERLOAD_VISIBLE
                self.publication_signal.admitted()
                self._direct_pending += 1
                self._idle.clear()
                try:
                    self._direct_results.put_nowait(_DirectPublication(publication))
                except queue.Full:
                    self._direct_slots.release()
                    self._direct_pending -= 1
                    self.publication_signal.completed(published=False)
                    if not self._states and not self._direct_pending:
                        self._idle.set()
                    return ChangeReviewFinalizeResult.OVERLOAD_VISIBLE
            return ChangeReviewFinalizeResult.SCHEDULED
        with self._lock:
            state = self._states.get(reservation.id)
            if state is None or state.cancelled or state.final_requested:
                return ChangeReviewFinalizeResult.REJECTED
            state.final_requested = True
            state.run_id = run_id
            state.kind = kind
            state.touched_paths = tuple(touched_paths)
            state.end_shas = dict(end_shas) if end_shas is not None else None
            state.has_live_survivors = has_live_survivors
            if has_live_survivors and not state.survivors_settled:
                survivor_token = state.survivor_key or f"unsettleable:{state.public.id}"
                for root in state.attribution_invalid_roots:
                    self._degraded_roots.setdefault(
                        root, _DegradedRootState()
                    ).survivor_keys.add(survivor_token)
            self._schedule_ready_locked()
            return ChangeReviewFinalizeResult.SCHEDULED

    def settle_survivors(self, survivor_key: str) -> None:
        """Declare that one originating turn's last child stopped writing."""
        if not survivor_key:
            return
        with self._lock:
            for state in self._states.values():
                if state.survivor_key == survivor_key:
                    state.survivors_settled = True
            for state in self._degraded_reservations.values():
                if state.survivor_key == survivor_key:
                    state.survivors_settled = True
            for root, degraded in tuple(self._degraded_roots.items()):
                degraded.survivor_keys.discard(survivor_key)
                self._maybe_clear_degraded_locked(root)
            self._schedule_ready_locked()

    def record_survivor_paths(
        self,
        survivor_key: str,
        paths: Sequence[str],
    ) -> None:
        """Attach late child WRITE paths to their originating survivor window."""
        if not survivor_key or not paths:
            return
        with self._lock:
            for state in self._states.values():
                if state.survivor_key != survivor_key:
                    continue
                state.survivor_touched_paths = tuple(
                    dict.fromkeys((*state.survivor_touched_paths, *paths))
                )

    def await_baseline(
        self,
        reservation: ChangeReviewReservation,
        timeout: float = 3.0,
    ) -> bool:
        """Bound one dispatch wait and invalidate overlapping attribution."""
        if reservation.await_baseline(timeout):
            return True
        with self._lock:
            state = self._states.get(reservation.id)
            if state is None or state.cancelled:
                return False
            timed_out = {
                root: error
                for root, error in reservation._handle.errors.items()
                if "baseline snapshot still running" in error
            }
            state.baseline_timed_out = bool(timed_out)
            if timed_out:
                state.baseline_timeout_error = next(iter(timed_out.values()))
            for root, error in timed_out.items():
                self._invalidate_timed_out_root_locked(state, root, error)
            self._schedule_ready_locked()
        return False

    def _invalidate_timed_out_root_locked(
        self,
        state: _ReservationState,
        root: str,
        error: str,
    ) -> None:
        state.attribution_invalid_roots.add(root)
        state.public._handle.errors.setdefault(root, error)
        degraded = self._degraded_roots.setdefault(root, _DegradedRootState())
        degraded.reservation_ids.add(state.public.id)
        lane = self._lanes.get(root, ())
        for predecessor_id in lane:
            if predecessor_id == state.public.id:
                break
            predecessor = self._states.get(predecessor_id)
            if predecessor is None or not self._window_can_claim_changes(predecessor):
                continue
            predecessor.attribution_invalid_roots.add(root)
            active = predecessor.active_handle or predecessor.public._handle
            active.errors[root] = (
                "change attribution invalidated by a successor baseline "
                "timeout while earlier review work was active"
            )
            degraded.reservation_ids.add(predecessor.public.id)
            if (
                predecessor.has_live_survivors
                and predecessor.survivor_key
                and not predecessor.survivors_settled
            ):
                degraded.survivor_keys.add(predecessor.survivor_key)

    @staticmethod
    def _window_can_claim_changes(state: _ReservationState) -> bool:
        if state.cancelled:
            return False
        if state.phase in {"waiting", "discovering", "discovered"}:
            # No B snapshot has started. Keep these reservations globally
            # sequence-ordered when a predecessor discovers another root;
            # treating queued discovery as an active barrier can invert two
            # root lanes and deadlock both reservations.
            return False
        if state.phase == "publishing":
            return (
                state.final_requested
                and state.has_live_survivors
                and not state.survivor_window_closed
            )
        return True

    def cancel(self, reservation: ChangeReviewReservation) -> bool:
        """Tombstone a reservation and wake successors when safe."""
        if reservation.admission_error:
            claimed = reservation._claim_terminal()
            if claimed:
                with self._lock:
                    self._finish_degraded_reservation_locked(
                        reservation.id,
                        has_live_survivors=False,
                    )
                if reservation._publication_reserved:
                    self._direct_slots.release()
            return claimed
        with self._lock:
            state = self._states.get(reservation.id)
            if state is None or state.cancelled:
                return False
            state.cancelled = True
            if not state.operation_inflight:
                self._complete_locked(state, published=False)
                self._schedule_ready_locked()
            return True

    def lane_depth(self, root: Path | str) -> int:
        canonical = str(root)
        with self._lock:
            return len(self._lanes.get(canonical, ()))

    def wait_idle(self, timeout: float | None = None) -> bool:
        return self._idle.wait(timeout)

    def shutdown(self, timeout: float = 2.0) -> bool:
        """Stop admission and bounded-wait for existing pure work."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._lock:
            self._accepting = False
            for state in tuple(self._states.values()):
                state.cancelled = True
                if not state.operation_inflight:
                    self._complete_locked(state, published=False)
            self._schedule_ready_locked()
            started = self._started
        if not started:
            self._close_publisher_once()
            return True
        drained = self._idle.wait(max(0.0, deadline - time.monotonic()))
        if self._native is not None:
            self._publisher_stop_requested.set()
            try:
                self._operations.put_nowait(_STOP)
            except queue.Full:
                pass
            for worker in self._workers:
                worker.join(max(0.0, deadline - time.monotonic()))
            self._publisher.join(max(0.0, deadline - time.monotonic()))
            return (
                drained
                and all(not w.is_alive() for w in self._workers)
                and not self._publisher.is_alive()
            )
        try:
            self._operations.put_nowait(_STOP)
        except queue.Full:
            while True:
                try:
                    self._operations.get_nowait()
                except queue.Empty:
                    break
            self._operations.put_nowait(_STOP)
        for worker in self._workers:
            worker.join(max(0.0, deadline - time.monotonic()))
        self._publisher_stop_requested.set()
        self._publisher.join(max(0.0, deadline - time.monotonic()))
        return (
            drained
            and all(not worker.is_alive() for worker in self._workers)
            and not self._publisher.is_alive()
        )

    def _start_threads_locked(self) -> None:
        """Start the fixed workers once, on first admitted review work."""
        if self._started:
            return
        for worker in self._workers:
            worker.start()
        self._publisher.start()
        self._started = True

    def _heads_all_lanes_locked(self, state: _ReservationState) -> bool:
        reservation_id = state.public.id
        return all(
            self._lanes.get(root) and self._lanes[root][0] == reservation_id
            for root in state.roots
        )

    def _schedule_ready_locked(self) -> None:
        for state in tuple(self._states.values()):
            if (
                state.cancelled
                or state.operation_inflight
                or state.pending_operation is not None
            ):
                continue
            if state.lane_ordered and not self._heads_all_lanes_locked(state):
                continue
            operation: _Operation | None = None
            if state.phase == "waiting":
                operation = _Operation(
                    reservation_id=state.public.id,
                    kind="discover",
                    handle=state.public._handle,
                )
            elif state.phase == "discovered":
                operation = _Operation(
                    reservation_id=state.public.id,
                    kind="baseline",
                    handle=state.public._handle,
                    preparations=state.preparations,
                )
            elif state.phase == "baseline" and state.final_requested:
                operation = _Operation(
                    reservation_id=state.public.id,
                    kind="failure" if state.admission_error else "finalize",
                    handle=state.public._handle,
                    touched_paths=state.touched_paths,
                    end_shas=state.end_shas,
                )
            elif state.phase == "survivor" and state.survivors_settled:
                operation = _Operation(
                    reservation_id=state.public.id,
                    kind="survivor_finalize",
                    handle=state.active_handle or state.public._handle,
                    touched_paths=state.survivor_touched_paths,
                )
            if operation is None:
                continue
            if self._native is not None:
                state.pending_operation = operation
                state.pending_reason = "ready"
                self._request_retry_locked("ready")
                continue
            try:
                self._operations.put_nowait(operation)
            except queue.Full:
                return
            state.operation_inflight = True
            if operation.kind == "discover":
                state.phase = "discovering"
            elif operation.kind == "baseline":
                state.phase = "baselining"
            else:
                state.phase = "finalizing"

    def _request_retry_locked(self, reason):
        self._retry_reasons.add(reason)
        try:
            self._results.put_nowait(_RETRY)
        except queue.Full:
            pass  # The existing publisher work will observe the coalesced flag.

    def _availability(self, reason):
        with self._lock:
            self._release_versions[reason] += 1
            if self._accepting and any(
                s.pending_reason == reason for s in self._states.values()
            ):
                self._request_retry_locked(reason)

    def _admit_pending(self):
        with self._lock:
            reasons, self._retry_reasons = self._retry_reasons, set()
            pending = [
                (state, state.pending_operation)
                for state in self._states.values()
                if state.pending_operation is not None
                and state.pending_reason in reasons
            ]
            versions = dict(self._release_versions)
            for state, _operation in pending:
                state.pending_reason = "admitting"
        for state, operation in pending:
            ticket, reason, error = None, "", ""
            try:
                ticket, reason = state.native_source.admit(operation.kind)
            except Exception:  # noqa: BLE001 - terminal native admission refusal
                error = "native change-review admission unavailable"
            discard = False
            failure = None
            with self._lock:
                if self._states.get(state.public.id) is not state or state.cancelled:
                    discard = True
                elif reason:
                    state.pending_reason = reason
                    if versions[reason] != self._release_versions[reason]:
                        self._request_retry_locked(reason)
                else:
                    state.pending_operation = None
                    state.operation_inflight = True
                    state.phase = {
                        "discover": "discovering",
                        "baseline": "baselining",
                    }.get(operation.kind, "finalizing")
                    if error:
                        failure = _OperationResult(
                            operation.reservation_id, operation.kind, error=error
                        )
                    else:
                        try:
                            self._operations.put_nowait(
                                replace(operation, ticket=ticket)
                            )
                        except queue.Full:
                            state.operation_inflight = False
                            state.pending_operation = operation
                            state.pending_reason = "queue"
                            if versions["queue"] != self._release_versions["queue"]:
                                self._request_retry_locked("queue")
                            discard = True
            if discard and ticket:
                state.native_source.restore_unstarted(operation.kind, ticket)
            if failure is not None:
                self._consume_operation_result(failure)

    def _complete_locked(self, state: _ReservationState, *, published: bool) -> None:
        if (
            state.cancelled
            and state.public._completion.disposition is ChangeReviewCompletion.PENDING
        ):
            state.public._completion.disposition = ChangeReviewCompletion.CANCELLED
        reservation_id = state.public.id
        self._states.pop(reservation_id, None)
        if state.native_source is not None:
            self._native_retirements.append(state.native_source)
        for root in state.roots:
            lane = self._lanes.get(root)
            if lane is None:
                continue
            try:
                lane.remove(reservation_id)
            except ValueError:
                pass
            if not lane:
                self._lanes.pop(root, None)
        self.publication_signal.completed(published=published)
        for root in state.attribution_invalid_roots:
            degraded = self._degraded_roots.get(root)
            if degraded is not None:
                degraded.reservation_ids.discard(reservation_id)
                self._maybe_clear_degraded_locked(root)
        if not self._states and not self._direct_pending:
            self._idle.set()

    def _maybe_clear_degraded_locked(self, root: str) -> None:
        degraded = self._degraded_roots.get(root)
        if degraded is None:
            return
        if degraded.reservation_ids or degraded.survivor_keys:
            return
        self._degraded_roots.pop(root, None)

    def _finish_degraded_reservation_locked(
        self,
        reservation_id: str,
        *,
        has_live_survivors: bool,
    ) -> None:
        state = self._degraded_reservations.pop(reservation_id, None)
        if state is None:
            return
        for root in state.roots:
            degraded = self._degraded_roots.get(root)
            if degraded is None:
                continue
            if has_live_survivors and not state.survivor_key:
                # No settlement identity means there is no safe point at
                # which to resynchronize. Retain the mutation token rather
                # than risk a false attribution window.
                continue
            degraded.reservation_ids.discard(reservation_id)
            if (
                has_live_survivors
                and state.survivor_key
                and not state.survivors_settled
            ):
                degraded.survivor_keys.add(state.survivor_key)
            self._maybe_clear_degraded_locked(root)

    def _worker_loop(self) -> None:
        tracker, operations, results = self._tracker, self._operations, self._results
        while True:
            if (
                self._native is not None
                and self._publisher_stop_requested.is_set()
                and operations.empty()
            ):
                return
            operation = operations.get()
            if self._native is not None:
                self._availability("queue")
            if operation is _STOP:
                try:
                    operations.put_nowait(_STOP)
                except queue.Full:
                    pass
                return
            assert isinstance(operation, _Operation)
            if self._native is not None:
                with self._lock:
                    state = self._states.get(operation.reservation_id)
                    cancelled = state is None or state.cancelled
                if cancelled:
                    results.put(
                        _OperationResult(
                            operation.reservation_id,
                            operation.kind,
                            ticket=operation.ticket,
                            started=False,
                        )
                    )
                    operation.ticket.returned(worker=True)
                    continue
            try:
                with operation.ticket.activate() if operation.ticket else nullcontext():
                    if operation.ticket:
                        operation.ticket.source.check()
                    if operation.kind == "discover":
                        discover = getattr(tracker, "discover_baseline", None)
                        if callable(discover):
                            preparations = tuple(discover(operation.handle))
                            if operation.ticket:
                                preparations = tuple(
                                    replace(
                                        item,
                                        root=operation.ticket.source.validate_root(
                                            item.root
                                        ),
                                    )
                                    for item in preparations
                                )
                            result = _OperationResult(
                                operation.reservation_id,
                                "discover",
                                preparations=preparations,
                            )
                        else:
                            tracker.populate_baseline(operation.handle)
                            result = _OperationResult(
                                operation.reservation_id, "baseline"
                            )
                    elif operation.kind == "baseline":
                        tracker.populate_prepared_baseline(
                            operation.handle, operation.preparations
                        )
                        result = _OperationResult(operation.reservation_id, "baseline")
                    else:
                        if operation.kind == "failure":
                            records = [
                                TurnChangeRecord(root=root, tracking_error=error)
                                for root, error in operation.handle.errors.items()
                            ]
                        else:
                            records = tracker.finish_turn(
                                operation.handle,
                                touched_paths=operation.touched_paths,
                                end_shas=operation.end_shas,
                            )
                        result = _OperationResult(
                            operation.reservation_id,
                            operation.kind,
                            tuple(records),
                        )
            except Exception as exc:  # noqa: BLE001 -- result is disclosed upstream
                result = _OperationResult(
                    operation.reservation_id,
                    operation.kind,
                    error=str(exc)[:400],
                )
            result = replace(result, ticket=operation.ticket)
            try:
                results.put(result)
            finally:
                if operation.ticket:
                    operation.ticket.returned(worker=True)

    def _publisher_loop(self) -> None:
        prefer_direct = True
        try:
            while True:
                if self._native is not None:
                    self._retire_unused_native()
                    self._admit_pending()
                if (
                    self._publisher_stop_requested.is_set()
                    and (
                        self._native is None
                        or all(not w.is_alive() for w in self._workers)
                    )
                    and self._direct_results.empty()
                    and self._results.empty()
                ):
                    return
                result: _OperationResult | _DirectPublication | object | None = None
                waited_for_regular = False
                if prefer_direct:
                    queues = (self._direct_results, self._results)
                    for result_queue in queues:
                        try:
                            result = result_queue.get_nowait()
                            break
                        except queue.Empty:
                            pass
                else:
                    # A bare get_nowait() made the advertised alternation
                    # illusory: a second direct overload record could jump
                    # ahead while a filesystem worker was just about to
                    # report. Give the preferred regular lane one bounded
                    # scheduling quantum before falling back to direct work.
                    waited_for_regular = True
                    try:
                        result = self._results.get(timeout=0.025)
                    except queue.Empty:
                        try:
                            result = self._direct_results.get_nowait()
                        except queue.Empty:
                            pass
                if result is None and not waited_for_regular:
                    try:
                        result = self._results.get(timeout=0.025)
                    except queue.Empty:
                        try:
                            result = self._direct_results.get_nowait()
                        except queue.Empty:
                            continue
                elif result is None:
                    continue
                prefer_direct = not prefer_direct
                if result is _RETRY:
                    continue
                if result is _STOP:
                    return
                if isinstance(result, _DirectPublication):
                    published = False
                    if not self._publisher_stop_requested.is_set():
                        published = self._publish_with_failure_fallback(
                            result.publication
                        ) in {
                            ChangeReviewCompletion.PUBLISHED,
                            ChangeReviewCompletion.DISCLOSED_FAILURE,
                        }
                    with self._lock:
                        self._direct_pending = max(0, self._direct_pending - 1)
                        self.publication_signal.completed(published=published)
                        if not self._states and not self._direct_pending:
                            self._idle.set()
                    self._direct_slots.release()
                    continue
                assert isinstance(result, _OperationResult)
                self._consume_operation_result(result)
        finally:
            self._retire_unused_native()
            self._close_publisher_once()

    def _retire_unused_native(self):
        with self._lock:
            sources = tuple(self._native_retirements)
            self._native_retirements.clear()
        for source in sources:
            source.close_unused()

    def _consume_operation_result(self, result: _OperationResult) -> None:
        ticket = result.ticket
        handoff_returned = False
        with self._lock:
            state = self._states.get(result.reservation_id)
            public = state.public if state is not None else None
            if ticket and state is not None:
                state.public._completion.ticket = ticket
                if result.kind in {"discover", "baseline"}:
                    state.public._baseline_completion.ticket = ticket
        try:
            with ticket.activate() if ticket else nullcontext():
                if (
                    ticket is not None
                    and ticket.source.workflow_wave is not None
                    and result.kind == "discover"
                ):
                    try:
                        ticket.source.workflow_wave.validate_discovered_roots(
                            result.preparations
                        )
                    except ValueError:
                        # No partial root set may become a successful B.
                        error = "workflow snapshot root metadata exceeds its bound"
                        result = replace(
                            result,
                            preparations=tuple(
                                BaselineRootPreparation(root=root, tracking_error=error)
                                for root in public._handle.roots
                            ),
                            error=error,
                        )
                self._consume_result(result)
                handoff_returned = True
        except Exception:  # Scope admission must deliver a terminal handoff.
            if self._native is None:
                raise
            with self._lock:
                state = self._states.get(result.reservation_id)
                if state is not None:
                    for root in state.roots:
                        state.public._handle.errors[root] = (
                            "native change-review handoff unavailable"
                        )
                    state.public._handle._baseline_ready.set()
                    state.public._completion.disposition = (
                        ChangeReviewCompletion.UNRESOLVED
                    )
                    if result.kind in {"discover", "baseline"}:
                        state.public._baseline_completion.disposition = (
                            ChangeReviewCompletion.UNRESOLVED
                        )
                    state.operation_inflight = False
                    self._complete_locked(state, published=False)
                    self._schedule_ready_locked()
        finally:
            if ticket:
                if getattr(ticket.source, "workflow_wave", None) is not None:
                    ticket._workflow_completion = (public, result.kind)
                ticket.returned(
                    handoff=True,
                    durable=public is None
                    or public._completion.disposition
                    is not ChangeReviewCompletion.UNRESOLVED,
                    observation_returned=handoff_returned,
                )
            elif (
                public is not None
                and public._completion.disposition is ChangeReviewCompletion.UNADMITTED
            ):
                public.notify_unadmitted()

    def _consume_result(self, result: _OperationResult) -> None:
        """Advance one filesystem result through durable publication."""
        publication: ChangeReviewPublication | None = None
        with self._lock:
            state = self._states.get(result.reservation_id)
            if state is None:
                return
            if state.cancelled and (
                self._native is None
                or not result.started
                or result.kind in {"discover", "baseline"}
            ):
                state.public._completion.disposition = ChangeReviewCompletion.CANCELLED
                if result.kind in {"discover", "baseline"}:
                    state.public._baseline_completion.disposition = (
                        ChangeReviewCompletion.CANCELLED
                    )
                state.operation_inflight = False
                self._complete_locked(state, published=False)
                self._schedule_ready_locked()
                return
            if self._native is not None and result.ticket is None:
                error = "native change-review admission unavailable"
                state.public._handle.errors.update(dict.fromkeys(state.roots, error))
                state.public._completion.admission_error = error
                state.public._completion.disposition = ChangeReviewCompletion.UNADMITTED
                if result.kind in {"discover", "baseline"}:
                    state.public._baseline_completion.disposition = (
                        ChangeReviewCompletion.UNADMITTED
                    )
                    state.public._handle._baseline_ready.set()
                state.operation_inflight = False
                self._complete_locked(state, published=False)
                self._schedule_ready_locked()
                return
            if result.kind == "discover":
                state.operation_inflight = False
                state.preparations = (
                    result.preparations
                    or tuple(
                        BaselineRootPreparation(
                            root=Path(root), tracking_error=result.error
                        )
                        for root in state.roots
                    )
                    if result.error
                    else result.preparations
                )
                self._enroll_discovered_roots_locked(state)
                state.phase = "discovered"
                self._schedule_ready_locked()
                return
            if result.kind == "baseline":
                state.operation_inflight = False
                state.phase = "baseline"
                if result.error:
                    for root in state.roots:
                        state.public._handle.errors.setdefault(root, result.error)
                    state.public._handle._baseline_ready.set()
                state.public._baseline_completion.disposition = (
                    ChangeReviewCompletion.DISCLOSED_FAILURE
                    if state.public._handle.errors
                    else ChangeReviewCompletion.OBSERVED
                )
                self._schedule_ready_locked()
                return
            if result.kind == "survivor_finalize":
                # The survivor mutation window closes when its E operation
                # returns, not when the resulting durable row finishes
                # publishing. A successor timeout after this point cannot
                # invalidate an already-complete B/E window.
                state.survivor_window_opened = False
                state.survivor_window_closed = True
            state.phase = "publishing"
            records = result.records
            if result.error:
                records = tuple(
                    TurnChangeRecord(root=root, tracking_error=result.error)
                    for root in state.roots
                )
            if state.attribution_invalid_roots:
                records = tuple(
                    record
                    for record in records
                    if record.root not in state.attribution_invalid_roots
                ) + tuple(
                    TurnChangeRecord(
                        root=root,
                        tracking_error=(
                            state.public._handle.errors.get(root)
                            or (
                                state.active_handle.errors.get(root)
                                if state.active_handle
                                else ""
                            )
                            or "change attribution invalidated after baseline timeout"
                        ),
                    )
                    for root in sorted(state.attribution_invalid_roots)
                )
            publication = ChangeReviewPublication(
                reservation_id=state.public.id,
                run_id=state.run_id,
                kind=(
                    "subagent_post_turn"
                    if result.kind == "survivor_finalize"
                    else state.kind
                ),
                records=records,
                roots=state.roots,
            )
        disposition = self._publish_with_failure_fallback(
            publication, publish=result.ticket.publish if result.ticket else None
        )
        published = disposition in {
            ChangeReviewCompletion.PUBLISHED,
            ChangeReviewCompletion.DISCLOSED_FAILURE,
        }
        with self._lock:
            state = self._states.get(result.reservation_id)
            if state is not None:
                state.public._completion.disposition = disposition
                state.operation_inflight = False
                if (
                    published
                    and result.kind == "finalize"
                    and state.has_live_survivors
                    and not state.cancelled
                ):
                    follow_on = self._tracker.continuation(state.public._handle)
                    if follow_on is not None:
                        for root in state.attribution_invalid_roots:
                            follow_on.errors[root] = (
                                state.public._handle.errors.get(root)
                                or "change attribution invalidated after baseline timeout"
                            )
                        self.publication_signal.window_published()
                        state.survivor_window_opened = True
                        state.phase = "survivor"
                        state.public._completion.disposition = (
                            ChangeReviewCompletion.PENDING
                        )
                        state.active_handle = follow_on
                        self._schedule_ready_locked()
                        return
                self._complete_locked(state, published=published)
                self._schedule_ready_locked()

    def _publish_with_failure_fallback(
        self, publication: ChangeReviewPublication, *, publish=None
    ) -> ChangeReviewCompletion:
        """Publish atomically, then make one honest terminal-error attempt."""
        publish = self._publish if publish is None else publish
        try:
            publish(publication)
            return (
                ChangeReviewCompletion.DISCLOSED_FAILURE
                if any(record.tracking_error for record in publication.records)
                else ChangeReviewCompletion.PUBLISHED
            )
        except Exception as exc:  # noqa: BLE001 -- converted to durable disclosure
            detail = f"{type(exc).__name__}: {exc}"[:320]
            roots = publication.roots or tuple(
                dict.fromkeys(record.root for record in publication.records)
            )
            failure = ChangeReviewPublication(
                reservation_id=publication.reservation_id,
                run_id=publication.run_id,
                kind=publication.kind,
                records=tuple(
                    TurnChangeRecord(
                        root=root,
                        tracking_error=("change-review publication failed; " + detail),
                    )
                    for root in roots
                ),
                roots=roots,
            )
            try:
                publish(failure)
                return ChangeReviewCompletion.DISCLOSED_FAILURE
            except Exception:  # noqa: BLE001 -- exactly one fallback attempt
                return ChangeReviewCompletion.UNRESOLVED

    def _close_publisher_once(self) -> None:
        """Close publisher-owned persistence once, after publication stops."""
        with self._lock:
            if self._publisher_closed:
                return
            self._publisher_closed = True
        for unsubscribe in self._subscriptions:
            unsubscribe()
        if self._close_publisher is not None:
            try:
                self._close_publisher()
            except Exception:
                pass

    def _enroll_discovered_roots_locked(self, state: _ReservationState) -> None:
        """Insert discovered roots without overtaking an open change window."""
        reservation_id = state.public.id
        discovered = tuple(dict.fromkeys(str(item.root) for item in state.preparations))
        new_roots = tuple(root for root in discovered if root not in state.roots)
        for root in new_roots:
            lane = self._lanes.setdefault(root, deque())
            barrier_after = 0
            for index, other_id in enumerate(lane):
                other = self._states.get(other_id)
                if other is not None and self._window_can_claim_changes(other):
                    barrier_after = index + 1
            insert_at = len(lane)
            for index in range(barrier_after, len(lane)):
                other_id = lane[index]
                other = self._states.get(other_id)
                if other is None:
                    continue
                if other.sequence > state.sequence:
                    insert_at = index
                    break
            lane.insert(insert_at, reservation_id)
        state.roots = tuple(dict.fromkeys((*state.roots, *new_roots)))
        for root in new_roots:
            if state.baseline_timed_out:
                self._invalidate_timed_out_root_locked(
                    state,
                    root,
                    state.baseline_timeout_error
                    or "baseline snapshot still running after bounded wait",
                )
            elif root in self._degraded_roots:
                error = (
                    "change-review root is resynchronizing after a baseline "
                    "timeout; filesystem tracking skipped for this turn"
                )
                state.attribution_invalid_roots.add(root)
                state.public._handle.errors[root] = error
                self._degraded_roots[root].reservation_ids.add(reservation_id)
