"""Bounded, app-owned Change Review baseline/finalization coordination."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Sequence

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


@dataclass(frozen=True)
class ChangeReviewReservation:
    """Caller-facing handle for one admitted root-ordered window."""

    id: str
    roots: tuple[str, ...]
    _handle: TurnHandle
    admission_error: str = ""
    _publication_reserved: bool = False
    _terminal_lock: threading.Lock = field(
        default_factory=threading.Lock,
        compare=False,
        repr=False,
    )
    _terminal_claimed: threading.Event = field(
        default_factory=threading.Event,
        compare=False,
        repr=False,
    )

    def await_baseline(self, timeout: float = 120.0) -> bool:
        return self._handle.await_baseline(timeout)

    @property
    def baseline_ready(self) -> bool:
        return self._handle._baseline_ready.is_set()

    def _claim_terminal(self) -> bool:
        """Claim the caller-owned terminal path exactly once."""
        with self._terminal_lock:
            if self._terminal_claimed.is_set():
                return False
            self._terminal_claimed.set()
            return True


@dataclass
class _ReservationState:
    public: ChangeReviewReservation
    phase: str = "waiting"
    operation_inflight: bool = False
    final_requested: bool = False
    cancelled: bool = False
    run_id: str = ""
    kind: str = "turn"
    touched_paths: tuple[str, ...] = ()
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


@dataclass(frozen=True)
class _Operation:
    reservation_id: str
    kind: str
    handle: TurnHandle
    touched_paths: tuple[str, ...] = ()
    end_shas: dict[str, str] | None = None
    preparations: tuple[BaselineRootPreparation, ...] = ()


@dataclass(frozen=True)
class _OperationResult:
    reservation_id: str
    kind: str
    records: tuple[TurnChangeRecord, ...] = ()
    error: str = ""
    preparations: tuple[BaselineRootPreparation, ...] = ()


@dataclass(frozen=True)
class _DirectPublication:
    publication: ChangeReviewPublication


_STOP = object()


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
    ) -> None:
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self._tracker = tracker
        self._publish = publish
        self._close_publisher = close_publisher
        self.publication_signal = (
            publication_signal or ChangeReviewPublicationSignal()
        )
        self._capacity = capacity
        self._lock = threading.Lock()
        self._lanes: dict[str, deque[str]] = {}
        self._states: dict[str, _ReservationState] = {}
        self._accepting = True
        self._next_sequence = 0
        self._idle = threading.Event()
        self._idle.set()
        self._operations: queue.Queue[_Operation | object] = queue.Queue(capacity)
        self._results: queue.Queue[
            _OperationResult | _DirectPublication | object
        ] = queue.Queue(capacity)
        self._direct_results: queue.Queue[_DirectPublication] = queue.Queue(
            capacity
        )
        self._direct_slots = threading.BoundedSemaphore(capacity)
        self._direct_pending = 0
        self._publisher_stop_requested = threading.Event()
        self._publisher_closed = False
        self._workers = tuple(
            threading.Thread(
                target=self._worker_loop,
                args=(self._tracker, self._operations, self._results),
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

    def register(
        self,
        roots: Sequence[Path | str],
        *,
        survivor_key: str = "",
    ) -> ChangeReviewReservation | None:
        """Atomically admit one reservation to every canonical-root lane."""
        canonical = tuple(
            dict.fromkeys(str(Path(root).expanduser().resolve()) for root in roots)
        )
        if not canonical:
            return None
        with self._lock:
            if not self._accepting:
                return None
            self._start_threads_locked()
            reservation_id = uuid.uuid4().hex
            handle = self._tracker.new_turn_handle(canonical)
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
                    self._direct_slots.acquire(blocking=False),
                )
            public = ChangeReviewReservation(
                reservation_id,
                canonical,
                handle,
            )
            state = _ReservationState(
                public=public,
                survivor_key=survivor_key,
                active_handle=handle,
                roots=canonical,
                sequence=self._next_sequence,
            )
            self._next_sequence += 1
            self._states[reservation_id] = state
            for root in canonical:
                self._lanes.setdefault(root, deque()).append(reservation_id)
            self._idle.clear()
            self.publication_signal.admitted()
            self._schedule_ready_locked()
            return public

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
                    self._direct_results.put_nowait(
                        _DirectPublication(publication)
                    )
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
            self._schedule_ready_locked()

    def cancel(self, reservation: ChangeReviewReservation) -> bool:
        """Tombstone a reservation and wake successors when safe."""
        if reservation.admission_error:
            claimed = reservation._claim_terminal()
            if claimed and reservation._publication_reserved:
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
        canonical = str(Path(root).expanduser().resolve())
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
            if state.cancelled or state.operation_inflight:
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
                )
            if operation is None:
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

    def _complete_locked(
        self, state: _ReservationState, *, published: bool
    ) -> None:
        reservation_id = state.public.id
        self._states.pop(reservation_id, None)
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
        if not self._states and not self._direct_pending:
            self._idle.set()

    @staticmethod
    def _worker_loop(
        tracker: ChangeTurnTracker,
        operations: queue.Queue[_Operation | object],
        results: queue.Queue[_OperationResult | _DirectPublication | object],
    ) -> None:
        while True:
            operation = operations.get()
            if operation is _STOP:
                try:
                    operations.put_nowait(_STOP)
                except queue.Full:
                    pass
                return
            assert isinstance(operation, _Operation)
            try:
                if operation.kind == "discover":
                    discover = getattr(tracker, "discover_baseline", None)
                    if callable(discover):
                        preparations = tuple(discover(operation.handle))
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
            results.put(result)

    def _publisher_loop(self) -> None:
        prefer_direct = True
        try:
            while True:
                if (
                    self._publisher_stop_requested.is_set()
                    and self._direct_results.empty()
                    and self._results.empty()
                ):
                    return
                result: _OperationResult | _DirectPublication | object | None = None
                queues = (
                    (self._direct_results, self._results)
                    if prefer_direct
                    else (self._results, self._direct_results)
                )
                for result_queue in queues:
                    try:
                        result = result_queue.get_nowait()
                        break
                    except queue.Empty:
                        pass
                if result is None:
                    try:
                        result = self._results.get(timeout=0.025)
                    except queue.Empty:
                        try:
                            result = self._direct_results.get_nowait()
                        except queue.Empty:
                            continue
                prefer_direct = not prefer_direct
                if result is _STOP:
                    return
                if isinstance(result, _DirectPublication):
                    published = False
                    if not self._publisher_stop_requested.is_set():
                        published = self._publish_with_failure_fallback(
                            result.publication
                        )
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
            self._close_publisher_once()

    def _consume_operation_result(self, result: _OperationResult) -> None:
        """Advance one filesystem result through durable publication."""
        publication: ChangeReviewPublication | None = None
        with self._lock:
            state = self._states.get(result.reservation_id)
            if state is None:
                return
            if state.cancelled:
                state.operation_inflight = False
                self._complete_locked(state, published=False)
                self._schedule_ready_locked()
                return
            if result.kind == "discover":
                state.operation_inflight = False
                state.preparations = result.preparations
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
                self._schedule_ready_locked()
                return
            state.phase = "publishing"
            records = result.records
            if result.error:
                records = tuple(
                    TurnChangeRecord(root=root, tracking_error=result.error)
                    for root in state.roots
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
        published = self._publish_with_failure_fallback(publication)
        with self._lock:
            state = self._states.get(result.reservation_id)
            if state is not None:
                state.operation_inflight = False
                if (
                    published
                    and result.kind == "finalize"
                    and state.has_live_survivors
                    and not state.cancelled
                ):
                    follow_on = self._tracker.continuation(state.public._handle)
                    if follow_on is not None:
                        self.publication_signal.window_published()
                        state.phase = "survivor"
                        state.active_handle = follow_on
                        self._schedule_ready_locked()
                        return
                self._complete_locked(state, published=published)
                self._schedule_ready_locked()

    def _publish_with_failure_fallback(
        self, publication: ChangeReviewPublication
    ) -> bool:
        """Publish atomically, then make one honest terminal-error attempt."""
        try:
            self._publish(publication)
            return True
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
                        tracking_error=(
                            "change-review publication failed; " + detail
                        ),
                    )
                    for root in roots
                ),
                roots=roots,
            )
            try:
                self._publish(failure)
                return True
            except Exception:  # noqa: BLE001 -- exactly one fallback attempt
                return False

    def _close_publisher_once(self) -> None:
        """Close publisher-owned persistence once, after publication stops."""
        with self._lock:
            if self._publisher_closed:
                return
            self._publisher_closed = True
        if self._close_publisher is not None:
            try:
                self._close_publisher()
            except Exception:
                pass

    def _enroll_discovered_roots_locked(self, state: _ReservationState) -> None:
        """Insert discovered roots without overtaking an in-flight operation."""
        reservation_id = state.public.id
        discovered = tuple(
            dict.fromkeys(str(item.root.resolve()) for item in state.preparations)
        )
        new_roots = tuple(root for root in discovered if root not in state.roots)
        for root in new_roots:
            lane = self._lanes.setdefault(root, deque())
            insert_at = len(lane)
            for index, other_id in enumerate(lane):
                other = self._states.get(other_id)
                if other is None:
                    continue
                if other.operation_inflight:
                    continue
                if other.sequence > state.sequence:
                    insert_at = index
                    break
            lane.insert(insert_at, reservation_id)
        state.roots = tuple(dict.fromkeys((*state.roots, *new_roots)))
