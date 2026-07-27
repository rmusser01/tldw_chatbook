from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Condition, Event
from time import monotonic

import pytest

from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    SessionChange,
)


def _wait_for_condition_waiter(
    condition: Condition,
    *,
    timeout: float = 1,
) -> None:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        with condition:
            if condition._waiters:
                return
        Event().wait(0.005)
    raise AssertionError("condition waiter was not registered")


def test_same_root_keeps_session_and_different_root_resets_it(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    first = owner.select_root(tmp_path / "a")
    owner.record_change(first, SessionChange("modified", "one.md"))

    assert owner.select_root(tmp_path / "a") == first
    assert [item.change.relative_path for item in owner.snapshot(first).changes] == [
        "one.md"
    ]

    second = owner.select_root(tmp_path / "b")
    assert second.generation == first.generation + 1
    assert owner.snapshot(second).changes == ()
    assert owner.record_change(first, SessionChange("modified", "late.md")) is False


def test_checked_root_selection_preserves_unexpected_or_same_root_state(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    assert owner.current_binding() is None

    first = owner.try_select_root(
        tmp_path / "a",
        expected_binding=None,
    )
    assert first is not None
    assert owner.record_change(first, SessionChange("modified", "one.md"))

    assert (
        owner.try_select_root(
            tmp_path / "a",
            expected_binding=None,
        )
        == first
    )
    assert (
        owner.try_select_root(
            tmp_path / "b",
            expected_binding=None,
        )
        is None
    )
    assert owner.current_binding() == first
    assert [item.change.relative_path for item in owner.snapshot(first).changes] == [
        "one.md"
    ]

    second = owner.try_select_root(
        tmp_path / "b",
        expected_binding=first,
    )
    assert second is not None
    assert second.generation == first.generation + 1
    assert owner.current_binding() == second
    assert owner.snapshot(second).changes == ()


def test_checked_root_selection_allows_only_legitimate_same_candidate_join(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()

    initial = owner.try_select_root(tmp_path / "a", expected_binding=None)
    assert initial is not None
    assert (
        owner.try_select_root(tmp_path / "a", expected_binding=None)
        == initial
    )

    replacement = owner.try_select_root(
        tmp_path / "b",
        expected_binding=initial,
    )
    assert replacement is not None
    assert replacement.generation == initial.generation + 1
    assert (
        owner.try_select_root(
            tmp_path / "b",
            expected_binding=initial,
        )
        == replacement
    )


def test_checked_root_selection_rejects_same_root_aba_binding(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    first_a = owner.select_root(tmp_path / "a")
    middle_b = owner.select_root(tmp_path / "b")
    current_a = owner.select_root(tmp_path / "a")

    assert current_a.generation == middle_b.generation + 1
    assert (
        owner.try_select_root(
            tmp_path / "a",
            expected_binding=first_a,
        )
        is None
    )
    assert owner.current_binding() == current_a


def test_root_commit_reservation_is_fail_fast_through_synchronous_publication(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    original = owner.select_root(tmp_path / "old")
    candidate = tmp_path / "candidate"
    competing = tmp_path / "competing"
    published_bindings = []

    def publish(binding) -> None:
        published_bindings.append(binding)
        assert owner.current_binding() == binding
        assert (
            owner.try_select_root(
                competing,
                expected_binding=binding,
            )
            is None
        )
        with pytest.raises(RuntimeError, match="root commit is in progress"):
            owner.select_root(competing)

    reservation = owner.try_reserve_root(
        candidate,
        expected_binding=original,
    )
    assert reservation is not None
    try:
        assert (
            owner.try_reserve_root(
                competing,
                expected_binding=original,
            )
            is None
        )
        assert (
            owner.try_select_root(
                competing,
                expected_binding=original,
            )
            is None
        )
        with pytest.raises(RuntimeError, match="root commit is in progress"):
            owner.select_root(competing)

        committed_binding = reservation.commit(publish)

        assert (
            owner.try_select_root(
                competing,
                expected_binding=committed_binding,
            )
            is None
        )
    finally:
        reservation.release()

    assert published_bindings == [committed_binding]
    assert committed_binding.root_key == str((tmp_path / "candidate").resolve())
    competing_binding = owner.select_root(competing)
    assert competing_binding.root_key == str(competing.resolve())


def test_recorder_assigns_one_monotonic_sequence_under_threads(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    with ThreadPoolExecutor(max_workers=4) as pool:
        accepted = list(
            pool.map(
                lambda number: owner.record_change(
                    binding,
                    SessionChange("modified", f"{number}.md"),
                ),
                range(40),
            )
        )

    snapshot = owner.snapshot(binding)
    assert all(accepted)
    assert [item.sequence for item in snapshot.changes] == list(range(1, 41))
    assert len({item.change.relative_path for item in snapshot.changes}) == 40


def test_owner_admits_transitions_mutations_and_status_atomically(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")

    transition = owner.try_acquire_transition(binding, "root")
    assert transition is not None
    assert owner.try_acquire_mutation(binding) is None
    transition.release()

    mutation = owner.try_acquire_mutation(binding)
    assert mutation is not None
    assert owner.try_acquire_transition(binding, "screen") is None
    assert owner.try_acquire_status(binding) is None
    mutation.release()

    status = owner.try_acquire_status(binding)
    assert status is not None
    waiting_mutation = owner.try_acquire_mutation(binding)
    assert waiting_mutation is not None
    assert owner.try_acquire_status(binding) is None
    status.release()
    waiting_mutation.release()


def test_stale_binding_cannot_publish_or_acquire_any_lease(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    stale = owner.select_root(tmp_path / "old")
    current = owner.select_root(tmp_path / "current")
    assert owner.record_change(
        current,
        SessionChange("modified", "current.md"),
    )

    assert not owner.record_change(stale, SessionChange("modified", "late.md"))
    assert owner.snapshot(stale).changes == ()
    assert [
        item.change.relative_path for item in owner.snapshot(current).changes
    ] == ["current.md"]
    assert owner.try_acquire_transition(stale, "path") is None
    assert owner.try_acquire_mutation(stale) is None
    assert owner.try_acquire_status(stale) is None

    current_status = owner.try_acquire_status(current)
    assert current_status is not None
    current_status.release()


def test_leases_release_idempotently_across_root_generation_change(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    first = owner.select_root(tmp_path / "first")
    transition = owner.try_acquire_transition(first, "root")
    assert transition is not None

    second = owner.select_root(tmp_path / "second")
    assert owner.try_acquire_mutation(second) is None
    transition.release()
    transition.release()

    mutation = owner.try_acquire_mutation(second)
    assert mutation is not None
    third = owner.select_root(tmp_path / "third")
    assert owner.try_acquire_transition(third, "screen") is None
    mutation.release()
    mutation.release()

    replacement_mutation = owner.try_acquire_mutation(third)
    assert replacement_mutation is not None
    replacement_mutation.release()

    status = owner.try_acquire_status(third)
    assert status is not None
    fourth = owner.select_root(tmp_path / "fourth")
    assert owner.try_acquire_status(fourth) is None
    status.release()
    status.release()
    replacement = owner.try_acquire_status(fourth)
    assert replacement is not None
    replacement.release()


def test_shutdown_is_idempotent_and_owner_state_is_never_persisted(
    tmp_path: Path,
) -> None:
    class AttachedService:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1

    root = tmp_path / "not-created"
    initial_entries = set(tmp_path.iterdir())
    attached = AttachedService()
    owner = FileNotesSessionOwner()
    owner.attach_git_service(attached)
    binding = owner.select_root(root)
    assert owner.record_change(binding, SessionChange("created", "one.md"))

    owner.shutdown()
    owner.shutdown()

    assert attached.shutdown_calls == 1
    assert not root.exists()
    assert set(tmp_path.iterdir()) == initial_entries
    assert not owner.record_change(binding, SessionChange("modified", "late.md"))
    assert owner.try_acquire_transition(binding, "source") is None
    assert owner.try_acquire_mutation(binding) is None
    assert owner.try_acquire_status(binding) is None


def test_concurrent_shutdown_waits_for_one_cleanup() -> None:
    cleanup_started = Event()
    release_cleanup = Event()
    second_started = Event()
    second_finished = Event()

    class BlockingService:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1
            cleanup_started.set()
            assert release_cleanup.wait(timeout=5)

    service = BlockingService()
    owner = FileNotesSessionOwner()
    owner.attach_git_service(service)

    def call_second_shutdown() -> None:
        second_started.set()
        owner.shutdown()
        second_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.shutdown)
        assert cleanup_started.wait(timeout=1)
        second = pool.submit(call_second_shutdown)
        assert second_started.wait(timeout=1)
        _wait_for_condition_waiter(owner._shutdown_condition)
        assert not second_finished.is_set()
        release_cleanup.set()
        first.result(timeout=1)
        second.result(timeout=1)

    assert second_finished.is_set()
    assert service.shutdown_calls == 1


def test_concurrent_and_later_shutdown_callers_observe_same_cleanup_failure() -> None:
    cleanup_started = Event()
    release_cleanup = Event()
    second_started = Event()
    cleanup_error = RuntimeError("forced cleanup failure")

    class RaisingService:
        def __init__(self) -> None:
            self.shutdown_calls = 0

        def shutdown(self) -> None:
            self.shutdown_calls += 1
            cleanup_started.set()
            assert release_cleanup.wait(timeout=5)
            raise cleanup_error

    service = RaisingService()
    owner = FileNotesSessionOwner()
    owner.attach_git_service(service)

    def call_shutdown(started: Event | None = None) -> BaseException | None:
        if started is not None:
            started.set()
        try:
            owner.shutdown()
        except BaseException as error:
            return error
        return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(call_shutdown)
        assert cleanup_started.wait(timeout=1)
        second = pool.submit(call_shutdown, second_started)
        assert second_started.wait(timeout=1)
        release_cleanup.set()
        assert first.result(timeout=1) is cleanup_error
        assert second.result(timeout=1) is cleanup_error

    with pytest.raises(RuntimeError) as later:
        owner.shutdown()

    assert later.value is cleanup_error
    assert service.shutdown_calls == 1
