"""Bounded, root-ordered Change Review finalization coordination."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from tldw_chatbook.Workspaces.change_review_finalization import (
    ChangeReviewFinalizationCoordinator,
    ChangeReviewPublication,
    ChangeReviewPublicationSignal,
)
from tldw_chatbook.Workspaces.change_turn_tracker import (
    BaselineRootPreparation,
    TurnChangeRecord,
    TurnHandle,
)


class _RecordingTracker:
    def __init__(self) -> None:
        self.events: list[str] = []
        self._lock = threading.Lock()
        self.baseline_entered: dict[str, threading.Event] = {}
        self.baseline_release: dict[str, threading.Event] = {}

    def new_turn_handle(self, roots) -> TurnHandle:
        return TurnHandle([Path(root) for root in roots])

    def populate_baseline(self, handle: TurnHandle) -> None:
        label = Path(handle.roots[0]).name
        with self._lock:
            self.events.append(f"B:{label}")
        self.baseline_entered.setdefault(label, threading.Event()).set()
        self.baseline_release.setdefault(label, threading.Event()).wait(timeout=2)
        for root in handle.roots:
            handle.baselines[str(root)] = f"b-{label}"
        handle._baseline_ready.set()

    def finish_turn(self, handle, touched_paths=(), *, end_shas=None):
        label = Path(handle.roots[0]).name
        with self._lock:
            self.events.append(f"E:{label}")
        return [
            TurnChangeRecord(
                root=str(handle.roots[0]),
                baseline_sha=f"b-{label}",
                end_sha=f"e-{label}",
                files_changed=1,
            )
        ]

    def continuation(self, handle):
        follow_on = TurnHandle(list(handle.roots))
        follow_on.baselines = {
            str(root): f"e-{Path(handle.roots[0]).name}" for root in handle.roots
        }
        follow_on._baseline_ready.set()
        return follow_on


def _wait_until(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_workers_are_lazy_until_first_admitted_review():
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=_RecordingTracker(),
        publish=lambda _item: None,
        worker_count=2,
        capacity=2,
    )

    assert not coordinator._started
    assert all(worker.ident is None for worker in coordinator._workers)
    assert coordinator._publisher.ident is None
    assert coordinator.shutdown(timeout=0.01)


def test_shared_root_runs_baseline_to_end_fifo(tmp_path):
    tracker = _RecordingTracker()
    publications: list[ChangeReviewPublication] = []
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publications.append,
        worker_count=2,
        capacity=4,
    )
    root = tmp_path / "shared"
    root.mkdir()

    first = coordinator.register([root])
    second = coordinator.register([root])
    assert first is not None and second is not None
    _wait_until(lambda: tracker.events == ["B:shared"])

    tracker.baseline_release["shared"].set()
    assert first.await_baseline(timeout=1)
    coordinator.finalize(first, run_id="run-1", kind="turn")
    _wait_until(lambda: len(publications) == 1)
    assert second.await_baseline(timeout=1)
    coordinator.finalize(second, run_id="run-2", kind="turn")
    _wait_until(lambda: len(publications) == 2)

    assert tracker.events == ["B:shared", "E:shared", "B:shared", "E:shared"]
    assert [item.run_id for item in publications] == ["run-1", "run-2"]
    coordinator.shutdown(timeout=1)


def test_disjoint_roots_can_prepare_concurrently(tmp_path):
    tracker = _RecordingTracker()
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=lambda _item: None,
        worker_count=2,
        capacity=4,
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    left_reservation = coordinator.register([left])
    right_reservation = coordinator.register([right])
    assert left_reservation is not None and right_reservation is not None
    _wait_until(lambda: set(tracker.events) == {"B:left", "B:right"})

    tracker.baseline_release["left"].set()
    tracker.baseline_release["right"].set()
    coordinator.finalize(left_reservation, run_id="left-run", kind="turn")
    coordinator.finalize(right_reservation, run_id="right-run", kind="turn")
    assert coordinator.wait_idle(timeout=1)
    coordinator.shutdown(timeout=1)


def test_multi_root_reservation_waits_until_it_heads_every_lane(tmp_path):
    tracker = _RecordingTracker()
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=lambda _item: None,
        worker_count=2,
        capacity=4,
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    blocker = coordinator.register([left])
    joined = coordinator.register([left, right])
    assert blocker is not None and joined is not None
    _wait_until(lambda: tracker.events == ["B:left"])
    assert not joined.baseline_ready

    tracker.baseline_release["left"].set()
    coordinator.finalize(blocker, run_id="blocker", kind="turn")
    _wait_until(lambda: len(tracker.events) >= 3)
    assert tracker.events[:3] == ["B:left", "E:left", "B:left"]
    coordinator.cancel(joined)
    coordinator.shutdown(timeout=1)


def test_capacity_rejection_does_not_leave_partial_lane_entries(tmp_path):
    tracker = _RecordingTracker()
    signal = ChangeReviewPublicationSignal()
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=lambda _item: None,
        publication_signal=signal,
        worker_count=1,
        capacity=1,
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    admitted = coordinator.register([left])
    rejected = coordinator.register([left, right])
    assert admitted is not None
    assert rejected is not None
    assert "at capacity" in rejected.admission_error
    assert rejected.await_baseline(timeout=0)
    assert coordinator.lane_depth(left) == 1
    assert coordinator.lane_depth(right) == 0
    assert len(coordinator._states) == 1
    assert signal.snapshot().pending == 1

    assert coordinator.cancel(rejected)
    _wait_until(lambda: "left" in tracker.baseline_release)
    tracker.baseline_release["left"].set()
    coordinator.cancel(admitted)
    assert coordinator.wait_idle(timeout=1)
    coordinator.shutdown(timeout=1)


def test_capacity_failure_publishes_an_honest_tracking_error(tmp_path):
    tracker = _RecordingTracker()
    publications = []
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publications.append,
        worker_count=2,
        capacity=1,
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    admitted = coordinator.register([left])
    rejected = coordinator.register([right])
    assert admitted is not None and rejected is not None
    assert rejected.admission_error

    coordinator.finalize(rejected, run_id="rejected-run", kind="turn")
    _wait_until(lambda: len(publications) == 1)
    assert publications[0].run_id == "rejected-run"
    assert publications[0].records[0].root == str(right.resolve())
    assert "at capacity" in publications[0].records[0].tracking_error

    _wait_until(lambda: "left" in tracker.baseline_release)
    tracker.baseline_release["left"].set()
    coordinator.cancel(admitted)
    assert coordinator.wait_idle(timeout=1)
    coordinator.shutdown(timeout=1)


def test_capacity_error_uses_reserved_channel_while_publisher_is_blocked(tmp_path):
    tracker = _RecordingTracker()
    publish_entered = threading.Event()
    publish_release = threading.Event()
    publications: list[ChangeReviewPublication] = []

    def publish(item: ChangeReviewPublication) -> None:
        publications.append(item)
        if len(publications) == 1:
            publish_entered.set()
            publish_release.wait(timeout=2)

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        worker_count=1,
        capacity=1,
    )
    left = tmp_path / "left"
    right = tmp_path / "right"
    overflow = tmp_path / "overflow"
    left.mkdir()
    right.mkdir()
    overflow.mkdir()

    admitted = coordinator.register([left])
    assert admitted is not None
    _wait_until(lambda: "left" in tracker.baseline_release)
    tracker.baseline_release["left"].set()
    assert admitted.await_baseline(timeout=1)
    assert coordinator.finalize(admitted, run_id="admitted", kind="turn")
    assert publish_entered.wait(timeout=1)

    rejected = coordinator.register([right])
    saturated = coordinator.register([overflow])
    assert rejected is not None and rejected._publication_reserved
    assert saturated is not None and not saturated._publication_reserved
    assert coordinator.finalize(rejected, run_id="rejected", kind="turn")
    assert not coordinator.finalize(saturated, run_id="overflow", kind="turn")

    publish_release.set()
    assert coordinator.wait_idle(timeout=1)
    assert [item.run_id for item in publications] == ["admitted", "rejected"]
    assert "at capacity" in publications[1].records[0].tracking_error
    coordinator.shutdown(timeout=1)


def test_direct_errors_do_not_starve_regular_filesystem_results(tmp_path):
    tracker = _RecordingTracker()
    first_publish_entered = threading.Event()
    first_publish_release = threading.Event()
    baseline_ready_at_second_publish: list[bool] = []
    reservations = []
    publication_count = 0

    def publish(_item: ChangeReviewPublication) -> None:
        nonlocal publication_count
        publication_count += 1
        if publication_count == 1:
            first_publish_entered.set()
            first_publish_release.wait(timeout=2)
        elif publication_count == 2:
            baseline_ready_at_second_publish.append(
                any(item.baseline_ready for item in reservations)
            )

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        worker_count=2,
        capacity=2,
    )
    roots = [tmp_path / name for name in ("left", "right", "r1", "r2")]
    for root in roots:
        root.mkdir()

    reservations.extend(
        [coordinator.register([roots[0]]), coordinator.register([roots[1]])]
    )
    assert all(item is not None for item in reservations)
    _wait_until(lambda: set(tracker.baseline_release) == {"left", "right"})
    rejected = coordinator.register([roots[2]])
    rejected_2 = coordinator.register([roots[3]])
    assert rejected is not None and rejected_2 is not None
    assert coordinator.finalize(rejected, run_id="r1", kind="turn")
    assert first_publish_entered.wait(timeout=1)

    tracker.baseline_release["left"].set()
    tracker.baseline_release["right"].set()
    assert coordinator.finalize(rejected_2, run_id="r2", kind="turn")
    first_publish_release.set()

    _wait_until(lambda: publication_count == 2)
    assert baseline_ready_at_second_publish == [True]
    for reservation in reservations:
        assert reservation is not None
        coordinator.cancel(reservation)
    assert coordinator.wait_idle(timeout=1)
    coordinator.shutdown(timeout=1)


def test_publication_failure_gets_one_terminal_tracking_error_attempt(tmp_path):
    tracker = _RecordingTracker()
    attempts: list[ChangeReviewPublication] = []

    def publish(item: ChangeReviewPublication) -> None:
        attempts.append(item)
        if len(attempts) == 1:
            raise RuntimeError("database unavailable")

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        worker_count=1,
        capacity=2,
    )
    root = tmp_path / "root"
    root.mkdir()
    reservation = coordinator.register([root])
    assert reservation is not None
    _wait_until(lambda: "root" in tracker.baseline_release)
    tracker.baseline_release["root"].set()
    assert reservation.await_baseline(timeout=1)

    assert coordinator.finalize(reservation, run_id="run-1", kind="turn")
    assert coordinator.wait_idle(timeout=1)

    assert len(attempts) == 2
    assert attempts[0].records[0].tracking_error == ""
    assert "publication failed" in attempts[1].records[0].tracking_error
    assert "RuntimeError" in attempts[1].records[0].tracking_error
    coordinator.shutdown(timeout=1)


def test_publication_failure_attempt_is_not_retried_forever(tmp_path):
    tracker = _RecordingTracker()
    attempts: list[ChangeReviewPublication] = []

    def publish(item: ChangeReviewPublication) -> None:
        attempts.append(item)
        raise RuntimeError("still unavailable")

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        worker_count=1,
        capacity=2,
    )
    root = tmp_path / "root"
    root.mkdir()
    reservation = coordinator.register([root])
    assert reservation is not None
    _wait_until(lambda: "root" in tracker.baseline_release)
    tracker.baseline_release["root"].set()
    assert reservation.await_baseline(timeout=1)

    assert coordinator.finalize(reservation, run_id="run-1", kind="turn")
    assert coordinator.wait_idle(timeout=1)

    assert len(attempts) == 2
    coordinator.shutdown(timeout=1)


def test_survivor_window_holds_lane_until_its_turn_children_settle(tmp_path):
    tracker = _RecordingTracker()
    publications: list[ChangeReviewPublication] = []
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publications.append,
        worker_count=1,
        capacity=4,
    )
    root = tmp_path / "shared"
    root.mkdir()

    first = coordinator.register([root], survivor_key="assistant-1")
    second = coordinator.register([root], survivor_key="assistant-2")
    assert first is not None and second is not None
    _wait_until(lambda: "shared" in tracker.baseline_release)
    tracker.baseline_release["shared"].set()
    assert first.await_baseline(timeout=1)
    coordinator.finalize(
        first,
        run_id="run-1",
        kind="turn",
        has_live_survivors=True,
    )
    _wait_until(lambda: len(publications) == 1)

    assert publications[0].kind == "turn"
    assert not second.baseline_ready
    coordinator.settle_survivors("assistant-1")
    _wait_until(lambda: len(publications) == 2)
    assert publications[1].kind == "subagent_post_turn"
    assert second.await_baseline(timeout=1)

    coordinator.cancel(second)
    assert coordinator.wait_idle(timeout=1)
    coordinator.shutdown(timeout=1)


def test_discovered_nested_root_never_overlaps_its_inflight_lane(tmp_path):
    parent = tmp_path / "parent"
    child = parent / "child"
    child.mkdir(parents=True)
    parent_discovery_entered = threading.Event()
    release_parent_discovery = threading.Event()
    child_baseline_entered = threading.Event()
    release_child_baseline = threading.Event()
    parent_baseline_entered = threading.Event()

    class _NestedTracker(_RecordingTracker):
        def discover_baseline(self, handle):
            root = handle.roots[0]
            if root == parent:
                parent_discovery_entered.set()
                release_parent_discovery.wait(timeout=2)
                return (
                    BaselineRootPreparation(root=parent, registered=("child",)),
                    BaselineRootPreparation(root=child),
                )
            return (BaselineRootPreparation(root=child),)

        def populate_prepared_baseline(self, handle, preparations):
            handle.roots[:] = [item.root for item in preparations]
            if preparations[0].root == child:
                child_baseline_entered.set()
                release_child_baseline.wait(timeout=2)
            else:
                parent_baseline_entered.set()
            for item in preparations:
                handle.baselines[str(item.root)] = "b"
            handle._baseline_ready.set()

    tracker = _NestedTracker()
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=lambda _item: None,
        worker_count=2,
        capacity=4,
    )

    parent_reservation = coordinator.register([parent])
    assert parent_discovery_entered.wait(timeout=1)
    child_reservation = coordinator.register([child])
    assert parent_reservation is not None and child_reservation is not None
    assert child_baseline_entered.wait(timeout=1)

    release_parent_discovery.set()
    assert not parent_baseline_entered.wait(timeout=0.05)
    release_child_baseline.set()
    assert child_reservation.await_baseline(timeout=1)
    coordinator.finalize(child_reservation, run_id="child-run", kind="turn")
    assert parent_baseline_entered.wait(timeout=1)

    coordinator.cancel(parent_reservation)
    assert coordinator.wait_idle(timeout=1)
    coordinator.shutdown(timeout=1)


def test_shutdown_is_bounded_and_late_worker_result_never_publishes(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    publications = []

    class _BlockedTracker(_RecordingTracker):
        def discover_baseline(self, handle):
            entered.set()
            release.wait(timeout=2)
            return (BaselineRootPreparation(root=handle.roots[0]),)

        def populate_prepared_baseline(self, handle, preparations):
            raise AssertionError("a cancelled generation must not reach B")

    tracker = _BlockedTracker()
    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publications.append,
        worker_count=1,
        capacity=2,
    )
    root = tmp_path / "root"
    root.mkdir()
    assert coordinator.register([root]) is not None
    assert entered.wait(timeout=1)

    started = time.monotonic()
    drained = coordinator.shutdown(timeout=0.05)
    elapsed = time.monotonic() - started

    assert not drained
    assert elapsed < 0.3
    assert all(worker.daemon for worker in coordinator._workers)
    release.set()
    _wait_until(lambda: all(not worker.is_alive() for worker in coordinator._workers))
    assert publications == []


def test_shutdown_never_closes_publisher_while_database_write_is_inflight(tmp_path):
    tracker = _RecordingTracker()
    publish_entered = threading.Event()
    publish_release = threading.Event()
    events: list[str] = []

    def publish(_item: ChangeReviewPublication) -> None:
        events.append("db-write-enter")
        publish_entered.set()
        publish_release.wait(timeout=2)
        events.append("db-write-exit")

    def close_publisher() -> None:
        events.append("db-close")

    coordinator = ChangeReviewFinalizationCoordinator(
        tracker=tracker,
        publish=publish,
        close_publisher=close_publisher,
        worker_count=1,
        capacity=2,
    )
    root = tmp_path / "root"
    root.mkdir()
    reservation = coordinator.register([root])
    assert reservation is not None
    _wait_until(lambda: "root" in tracker.baseline_release)
    tracker.baseline_release["root"].set()
    assert reservation.await_baseline(timeout=1)
    assert coordinator.finalize(reservation, run_id="run-1", kind="turn")
    assert publish_entered.wait(timeout=1)

    assert not coordinator.shutdown(timeout=0.05)
    assert events == ["db-write-enter"]

    publish_release.set()
    _wait_until(lambda: events == ["db-write-enter", "db-write-exit", "db-close"])
    assert not coordinator._publisher.is_alive()
