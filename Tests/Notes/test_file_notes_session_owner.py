from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tldw_chatbook.Notes.file_notes_session_owner import (
    FileNotesSessionOwner,
    SessionChange,
)


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
