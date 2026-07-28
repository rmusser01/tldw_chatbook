from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Condition, Event
from time import monotonic

import pytest

from tldw_chatbook.Notes.file_notes_session_owner import (
    FileSystemIdentity,
    FileNotesSessionOwner,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    RepositoryIdentity,
    SessionChange,
    SessionChangeGroup,
    SessionGitStatus,
    StagingOwnership,
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


def _git_owner_state():
    filesystem_identity = FileSystemIdentity(device=1, inode=2)
    repository = RepositoryIdentity(
        worktree_root="/repo",
        git_dir="/repo/.git",
        git_common_dir="/repo/.git",
        worktree_identity=filesystem_identity,
        git_dir_identity=filesystem_identity,
        git_common_dir_identity=filesystem_identity,
    )
    entry = IndexEntry(
        path="note.md",
        mode="100644",
        object_id="a" * 40,
        stage=0,
    )
    ownership = StagingOwnership(
        repository=repository,
        head=HeadIdentity.attached("refs/heads/main", "b" * 40),
        approved_endpoint_topology=("note.md",),
        approved_move_edges=(),
        approved_current_path="note.md",
        original_baselines={"note.md": IndexBaseline(None)},
        post_stage_entries={"note.md": entry},
    )
    group = SessionChangeGroup(
        group_id=1,
        endpoints=("note.md",),
        source_path="note.md",
        destination_path=None,
        current_path="note.md",
        latest_action="modified",
        latest_sequence=1,
    )
    return repository, ownership, group


def test_same_root_retains_git_state_and_root_change_clears_it_atomically(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    first = owner.select_root(tmp_path / "first")
    repository, ownership, _group = _git_owner_state()
    generation = owner.next_status_generation(first)
    assert generation is not None
    status = SessionGitStatus(
        binding_generation=first.generation,
        status_generation=generation,
        state="ready",
        repository=repository,
    )

    assert owner.publish_trust(first, repository)
    assert owner.publish_status(first, status)
    assert owner.publish_ownership(first, {1: ownership})

    same = owner.select_root(tmp_path / "first")
    snapshot = owner.snapshot(same)
    assert same == first
    assert snapshot.trusted_repository == repository
    assert snapshot.git_status == status
    assert dict(snapshot.staging_ownership) == {1: ownership}

    transition = owner.try_acquire_transition(first, "root")
    assert transition is not None
    second = owner.select_root(tmp_path / "second")
    reset = owner.snapshot(second)
    assert reset.trusted_repository is None
    assert reset.git_status is None
    assert dict(reset.staging_ownership) == {}

    transition.release()
    transition.release()
    replacement = owner.try_acquire_transition(second, "root")
    assert replacement is not None
    replacement.release()


def test_stale_binding_cannot_publish_or_clear_git_state(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    stale = owner.select_root(tmp_path / "old")
    current = owner.select_root(tmp_path / "current")
    repository, ownership, _group = _git_owner_state()
    stale_status = SessionGitStatus(
        binding_generation=stale.generation,
        status_generation=1,
        state="ready",
        repository=repository,
    )

    assert not owner.publish_trust(stale, repository)
    assert not owner.publish_status(stale, stale_status)
    assert not owner.publish_ownership(stale, {1: ownership})
    assert not owner.clear_trust(stale)
    assert not owner.clear_status(stale)
    assert not owner.clear_ownership(stale)
    assert owner.next_status_generation(stale) is None
    assert owner.snapshot(current).trusted_repository is None


def test_status_publication_rejects_older_generation_under_same_binding(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, _group = _git_owner_state()
    assert owner.publish_trust(binding, repository)

    newer = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=10,
        state="ready",
        repository=repository,
    )
    older = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=9,
        state="error",
        message="late failure",
    )

    assert owner.publish_status(binding, newer)
    assert not owner.publish_status(binding, older)
    assert owner.snapshot(binding).git_status == newer


def test_checked_git_clear_methods_and_public_mappings_are_immutable(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, ownership, _group = _git_owner_state()
    status_generation = owner.next_status_generation(binding)
    assert status_generation is not None
    status = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=status_generation,
        state="ready",
        repository=repository,
    )
    assert owner.publish_trust(binding, repository)
    assert owner.publish_status(binding, status)
    assert owner.publish_ownership(binding, {1: ownership})

    snapshot = owner.snapshot(binding)
    with pytest.raises(TypeError):
        snapshot.staging_ownership[2] = ownership  # type: ignore[index]
    with pytest.raises(TypeError):
        ownership.post_stage_entries["other.md"] = IndexEntry(  # type: ignore[index]
            path="other.md",
            mode="100644",
            object_id="c" * 40,
        )

    assert owner.clear_status(binding)
    assert owner.snapshot(binding).git_status is None
    assert owner.snapshot(binding).trusted_repository == repository
    assert owner.clear_ownership(binding)
    assert dict(owner.snapshot(binding).staging_ownership) == {}

    replacement_generation = owner.next_status_generation(binding)
    assert replacement_generation is not None
    replacement_status = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=replacement_generation,
        state="ready",
        repository=repository,
    )
    assert owner.publish_status(binding, replacement_status)
    assert owner.publish_ownership(binding, {1: ownership})
    assert owner.clear_trust(binding)
    cleared = owner.snapshot(binding)
    assert cleared.trusted_repository is None
    assert cleared.git_status is None
    assert dict(cleared.staging_ownership) == {}


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
