from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Condition, Event
from time import monotonic

import pytest

from tldw_chatbook.Notes.file_notes_git_commit import CommitRecoveryProjection
from tldw_chatbook.Notes.file_notes_session_owner import (
    CommitAuthorityCapture,
    CommitPublication,
    CommitRecoveryCapability,
    FileSystemIdentity,
    FileNotesSessionOwner,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    RepositoryIdentity,
    SessionBinding,
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


def test_root_selection_accepts_valid_shell_metacharacters(
    tmp_path: Path,
) -> None:
    root = tmp_path / "notes;pipe|and&&tick`dollar$(x)${y}"
    root.mkdir()
    expected_root_key = str(root.resolve())

    direct_owner = FileNotesSessionOwner()
    assert direct_owner.select_root(root).root_key == expected_root_key

    conditional_owner = FileNotesSessionOwner()
    conditional = conditional_owner.try_select_root(
        root,
        expected_binding=None,
    )
    assert conditional is not None
    assert conditional.root_key == expected_root_key

    stable_owner = FileNotesSessionOwner()
    stable = stable_owner.acquire_stable_root(root)
    assert stable is not None
    assert stable.binding is not None
    assert stable.binding.root_key == expected_root_key
    stable.release()

    reserved_owner = FileNotesSessionOwner()
    reservation = reserved_owner.try_reserve_root(
        root,
        expected_binding=None,
    )
    assert reservation is not None
    reservation.release()


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


def _ownership_for(
    repository: RepositoryIdentity,
    *,
    path: str,
    object_id: str,
) -> StagingOwnership:
    return StagingOwnership(
        repository=repository,
        head=HeadIdentity.attached("refs/heads/main", "b" * 40),
        approved_endpoint_topology=(path,),
        approved_move_edges=(),
        approved_current_path=path,
        original_baselines={path: IndexBaseline(None)},
        post_stage_entries={
            path: IndexEntry(
                path=path,
                mode="100644",
                object_id=object_id,
            )
        },
    )


def _ready_status(
    owner: FileNotesSessionOwner,
    binding: SessionBinding,
    repository: RepositoryIdentity,
    *,
    state: str = "ready",
) -> SessionGitStatus:
    generation = owner.next_status_generation(binding)
    assert generation is not None
    return SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=generation,
        state=state,  # type: ignore[arg-type]
        repository=repository,
        head=HeadIdentity.attached("refs/heads/main", "b" * 40),
    )


def _prepare_commit_authority(
    owner: FileNotesSessionOwner,
    binding: SessionBinding,
    *,
    include_unowned_change: bool = False,
) -> tuple[
    RepositoryIdentity,
    dict[int, StagingOwnership],
    dict[int, tuple[int, ...]],
    SessionGitStatus,
]:
    repository, first_ownership, _group = _git_owner_state()
    assert owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )
    second_ownership = _ownership_for(
        repository,
        path="second.md",
        object_id="c" * 40,
    )
    assert owner.record_change(
        binding,
        SessionChange("modified", "second.md"),
    )
    if include_unowned_change:
        assert owner.record_change(
            binding,
            SessionChange("modified", "later.md"),
        )
    ownership = {1: first_ownership, 2: second_ownership}
    sequence_ids = {1: (1,), 2: (2,)}
    assert owner.publish_trust(binding, repository)
    status = _ready_status(owner, binding, repository)
    assert owner.publish_status(binding, status)
    assert owner.publish_ownership(
        binding,
        ownership,
        group_sequence_ids=sequence_ids,
    )
    return repository, ownership, sequence_ids, status


def _capture_commit_authority(
    owner: FileNotesSessionOwner,
    binding: SessionBinding,
    repository: RepositoryIdentity,
    sequence_ids: dict[int, tuple[int, ...]],
):
    lease = owner.try_acquire_mutation(binding)
    assert lease is not None
    capture = owner.capture_commit_authority(
        lease,
        binding=binding,
        authority_generation=owner.snapshot(binding).git_authority_generation,
        repository=repository,
        head=HeadIdentity.attached("refs/heads/main", "b" * 40),
        group_sequence_ids=sequence_ids,
    )
    assert isinstance(capture, CommitAuthorityCapture)
    return lease, capture


def _publish_uncertain_commit(
    owner: FileNotesSessionOwner,
    binding: SessionBinding,
) -> tuple[
    RepositoryIdentity,
    dict[int, StagingOwnership],
    CommitRecoveryCapability,
]:
    repository, ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )
    lease, capture = _capture_commit_authority(
        owner,
        binding,
        repository,
        sequence_ids,
    )
    projection = CommitRecoveryProjection(
        message="Commit outcome requires an exact repository check.",
        can_check_again=True,
    )
    publication = owner.publish_commit_outcome(
        lease,
        capture,
        CommitPublication(
            state="uncertain",
            recovery_projection=projection,
        ),
    )
    assert publication.published
    assert isinstance(
        publication.recovery_capability,
        CommitRecoveryCapability,
    )
    lease.release()
    return repository, ownership, publication.recovery_capability


def test_commit_authority_generation_changes_only_for_material_owner_facts(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    selected_generation = owner.snapshot(binding).git_authority_generation
    repository, ownership, _group = _git_owner_state()

    assert owner.select_root(tmp_path / "notes") == binding
    assert owner.snapshot(binding).git_authority_generation == selected_generation

    assert owner.publish_trust(binding, repository)
    trusted_generation = owner.snapshot(binding).git_authority_generation
    assert trusted_generation > selected_generation
    assert owner.publish_trust(binding, repository)
    assert owner.snapshot(binding).git_authority_generation == trusted_generation

    first_status = _ready_status(owner, binding, repository)
    assert owner.publish_status(binding, first_status)
    status_generation = owner.snapshot(binding).git_authority_generation
    equivalent_status = _ready_status(owner, binding, repository)
    equivalent_status = SessionGitStatus(
        binding_generation=equivalent_status.binding_generation,
        status_generation=equivalent_status.status_generation,
        state=equivalent_status.state,
        repository=equivalent_status.repository,
        head=equivalent_status.head,
        message="Presentation-only refresh",
    )
    assert owner.publish_status(binding, equivalent_status)
    assert owner.snapshot(binding).git_authority_generation == status_generation
    materially_changed_status = _ready_status(
        owner,
        binding,
        repository,
        state="stale",
    )
    assert owner.publish_status(binding, materially_changed_status)
    changed_status_generation = owner.snapshot(binding).git_authority_generation
    assert changed_status_generation > status_generation

    assert owner.publish_ownership(binding, {1: ownership})
    ownership_generation = owner.snapshot(binding).git_authority_generation
    assert owner.publish_ownership(binding, {1: ownership})
    assert owner.snapshot(binding).git_authority_generation == ownership_generation
    assert owner.publish_ownership(binding, {})
    assert owner.snapshot(binding).git_authority_generation > ownership_generation

    before_change = owner.snapshot(binding).git_authority_generation
    assert owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )
    assert owner.snapshot(binding).git_authority_generation > before_change

    before_transition = owner.snapshot(binding).git_authority_generation
    transition = owner.try_acquire_transition(binding, "root")
    assert transition is not None
    assert owner.snapshot(binding).git_authority_generation > before_transition
    transition.release()

    before_rebind = owner.snapshot(binding).git_authority_generation
    rebound = owner.select_root(tmp_path / "other")
    assert owner.snapshot(rebound).git_authority_generation > before_rebind


def test_commit_authority_rejects_aba_after_equivalent_state_is_restored(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )
    reviewed_generation = owner.snapshot(binding).git_authority_generation

    assert owner.publish_ownership(binding, {})
    assert owner.publish_ownership(
        binding,
        ownership,
        group_sequence_ids=sequence_ids,
    )
    assert dict(owner.snapshot(binding).staging_ownership) == ownership

    lease = owner.try_acquire_mutation(binding)
    assert lease is not None
    assert (
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=reviewed_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids=sequence_ids,
        )
        is None
    )
    lease.release()


def test_commit_authority_capture_requires_exact_active_owner_state(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )
    authority_generation = owner.snapshot(binding).git_authority_generation
    released = owner.try_acquire_mutation(binding)
    assert released is not None
    released.release()

    assert (
        owner.capture_commit_authority(
            released,
            binding=binding,
            authority_generation=authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids=sequence_ids,
        )
        is None
    )

    lease = owner.try_acquire_mutation(binding)
    assert lease is not None
    assert (
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={1: (1,)},
        )
        is None
    )
    assert (
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={1: (1,), 2: (999,)},
        )
        is None
    )
    assert isinstance(
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids=sequence_ids,
        ),
        CommitAuthorityCapture,
    )
    lease.release()


def test_commit_capture_waits_for_active_status_before_uncertain_outcome(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )
    status_admission = owner.admit_status(binding)
    assert status_admission.lease is not None
    assert status_admission.invalidation_generation is not None
    mutation = owner.try_acquire_mutation(binding)
    assert mutation is not None
    reviewed_generation = owner.snapshot(binding).git_authority_generation

    assert (
        owner.capture_commit_authority(
            mutation,
            binding=binding,
            authority_generation=reviewed_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids=sequence_ids,
        )
        is None
    )

    refreshed_status = _ready_status(
        owner,
        binding,
        repository,
        state="stale",
    )
    assert not owner.publish_status(
        binding,
        refreshed_status,
        invalidation_generation=status_admission.invalidation_generation,
    )
    assert (
        owner.snapshot(binding).git_authority_generation
        == reviewed_generation
    )
    mutation.release()
    assert owner.publish_status(
        binding,
        refreshed_status,
        invalidation_generation=status_admission.invalidation_generation,
    )
    assert (
        owner.snapshot(binding).git_authority_generation
        > reviewed_generation
    )
    status_admission.lease.release()

    mutation = owner.try_acquire_mutation(binding)
    assert mutation is not None
    capture = owner.capture_commit_authority(
        mutation,
        binding=binding,
        authority_generation=owner.snapshot(binding).git_authority_generation,
        repository=repository,
        head=HeadIdentity.attached("refs/heads/main", "b" * 40),
        group_sequence_ids=sequence_ids,
    )
    assert isinstance(capture, CommitAuthorityCapture)
    projection = CommitRecoveryProjection(
        message="Commit outcome requires an exact repository check.",
        can_check_again=True,
    )
    publication = owner.publish_commit_outcome(
        mutation,
        capture,
        CommitPublication(
            state="uncertain",
            recovery_projection=projection,
        ),
    )
    assert publication.published
    assert owner.snapshot(binding).commit_recovery == projection
    mutation.release()
    assert owner.admit_mutation(binding).reason == "recovery_required"


def test_released_status_cannot_invalidate_active_commit_capture(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )
    status_admission = owner.admit_status(binding)
    assert status_admission.lease is not None
    assert status_admission.invalidation_generation is not None
    late_status = _ready_status(
        owner,
        binding,
        repository,
        state="stale",
    )
    status_admission.lease.release()

    mutation, capture = _capture_commit_authority(
        owner,
        binding,
        repository,
        sequence_ids,
    )
    before_late_status = owner.snapshot(binding)
    assert not owner.publish_status(
        binding,
        late_status,
        invalidation_generation=status_admission.invalidation_generation,
    )
    after_late_status = owner.snapshot(binding)
    assert after_late_status.git_status == before_late_status.git_status
    assert (
        after_late_status.git_authority_generation
        == before_late_status.git_authority_generation
    )

    projection = CommitRecoveryProjection(
        message="Commit outcome requires an exact repository check.",
        can_check_again=True,
    )
    publication = owner.publish_commit_outcome(
        mutation,
        capture,
        CommitPublication(
            state="uncertain",
            recovery_projection=projection,
        ),
    )
    assert publication.published
    assert owner.snapshot(binding).commit_recovery == projection
    mutation.release()
    assert owner.admit_mutation(binding).reason == "recovery_required"


def test_commit_authority_record_change_rejects_stale_lineage_capture(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, ownership, _group = _git_owner_state()
    assert owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )
    assert owner.publish_trust(binding, repository)
    assert owner.publish_ownership(
        binding,
        {1: ownership},
        group_sequence_ids={1: (1,)},
    )

    first_lease = owner.try_acquire_mutation(binding)
    assert first_lease is not None
    assert isinstance(
        owner.capture_commit_authority(
            first_lease,
            binding=binding,
            authority_generation=owner.snapshot(binding).git_authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={1: (1,)},
        ),
        CommitAuthorityCapture,
    )
    first_lease.release()

    assert owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )
    current_generation = owner.snapshot(binding).git_authority_generation
    second_lease = owner.try_acquire_mutation(binding)
    assert second_lease is not None
    assert (
        owner.capture_commit_authority(
            second_lease,
            binding=binding,
            authority_generation=current_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={1: (1,)},
        )
        is None
    )
    assert isinstance(
        owner.capture_commit_authority(
            second_lease,
            binding=binding,
            authority_generation=current_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={1: (1, 2)},
        ),
        CommitAuthorityCapture,
    )
    second_lease.release()


def test_commit_authority_rejects_old_root_lease_with_current_root_facts(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    first_binding = owner.select_root(tmp_path / "first")
    stale_lease = owner.try_acquire_mutation(first_binding)
    assert stale_lease is not None

    current_binding = owner.select_root(tmp_path / "current")
    stale_lease.release()
    repository, _ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        current_binding,
    )

    assert (
        owner.capture_commit_authority(
            stale_lease,
            binding=current_binding,
            authority_generation=owner.snapshot(
                current_binding
            ).git_authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids=sequence_ids,
        )
        is None
    )
    stale_lease.release()


def test_commit_authority_rejects_initially_wrong_lineage_sequences(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, first_ownership, _group = _git_owner_state()
    second_ownership = _ownership_for(
        repository,
        path="second.md",
        object_id="c" * 40,
    )
    for path in ("note.md", "second.md", "note.md", "second.md"):
        assert owner.record_change(
            binding,
            SessionChange("modified", path),
        )
    assert owner.publish_trust(binding, repository)
    status = _ready_status(owner, binding, repository)
    assert owner.publish_status(binding, status)
    assert not owner.publish_ownership(
        binding,
        {1: first_ownership, 2: second_ownership},
        group_sequence_ids={1: (1, 4), 2: (2, 3)},
    )
    assert dict(owner.snapshot(binding).staging_ownership) == {}
    exact_sequence_ids = {1: (1, 3), 2: (2, 4)}
    assert owner.publish_ownership(
        binding,
        {1: first_ownership, 2: second_ownership},
        group_sequence_ids=exact_sequence_ids,
    )
    generation = owner.snapshot(binding).git_authority_generation
    lease = owner.try_acquire_mutation(binding)
    assert lease is not None

    assert (
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={1: (1,), 2: (2, 4)},
        )
        is None
    )
    assert (
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={1: (1, 4), 2: (2, 3)},
        )
        is None
    )
    assert isinstance(
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids=exact_sequence_ids,
        ),
        CommitAuthorityCapture,
    )
    lease.release()


def test_commit_authority_uses_only_current_owned_lineages_after_unstage(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )

    assert owner.publish_unstage_result(
        binding,
        repository,
        ownership,
        (1,),
    )
    lease = owner.try_acquire_mutation(binding)
    assert lease is not None
    assert (
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=owner.snapshot(binding).git_authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids=sequence_ids,
        )
        is None
    )
    assert isinstance(
        owner.capture_commit_authority(
            lease,
            binding=binding,
            authority_generation=owner.snapshot(binding).git_authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={2: (2,)},
        ),
        CommitAuthorityCapture,
    )
    lease.release()


def test_commit_publication_success_retires_only_proven_whole_groups(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
        include_unowned_change=True,
    )
    lease, capture = _capture_commit_authority(
        owner,
        binding,
        repository,
        sequence_ids,
    )
    before_publication = owner.snapshot(binding).git_authority_generation

    publication = owner.publish_commit_outcome(
        lease,
        capture,
        CommitPublication(
            state="succeeded",
            new_head=HeadIdentity.attached(
                "refs/heads/main",
                "d" * 40,
            ),
            retired_sequence_ids=(1,),
            divergent_sequence_ids=(2,),
        ),
    )

    assert publication.published
    assert publication.recovery_capability is None
    snapshot = owner.snapshot(binding)
    assert [item.sequence for item in snapshot.changes] == [2, 3]
    assert dict(snapshot.staging_ownership) == {}
    assert snapshot.git_status is None
    assert snapshot.git_authority_generation > before_publication
    lease.release()


def test_commit_publication_rejects_partial_group_retirement(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, ownership, _sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )
    assert owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )
    sequence_ids = {1: (1, 3), 2: (2,)}
    assert owner.publish_ownership(
        binding,
        ownership,
        group_sequence_ids=sequence_ids,
    )
    lease, capture = _capture_commit_authority(
        owner,
        binding,
        repository,
        sequence_ids,
    )

    publication = owner.publish_commit_outcome(
        lease,
        capture,
        CommitPublication(
            state="succeeded",
            new_head=HeadIdentity.attached(
                "refs/heads/main",
                "d" * 40,
            ),
            retired_sequence_ids=(1,),
            divergent_sequence_ids=(2, 3),
        ),
    )

    assert not publication.published
    snapshot = owner.snapshot(binding)
    assert [item.sequence for item in snapshot.changes] == [1, 2, 3]
    assert dict(snapshot.staging_ownership) == ownership
    lease.release()


def test_commit_publication_failed_unchanged_preserves_owner_facts(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, ownership, sequence_ids, status = _prepare_commit_authority(
        owner,
        binding,
    )
    lease, capture = _capture_commit_authority(
        owner,
        binding,
        repository,
        sequence_ids,
    )
    before = owner.snapshot(binding)

    publication = owner.publish_commit_outcome(
        lease,
        capture,
        CommitPublication(state="failed_unchanged"),
    )

    assert publication.published
    after = owner.snapshot(binding)
    assert after.changes == before.changes
    assert after.trusted_repository == repository
    assert after.git_status == status
    assert dict(after.staging_ownership) == ownership
    assert after.git_authority_generation > before.git_authority_generation
    lease.release()


def test_commit_quarantine_moves_exact_ownership_and_blocks_mutations(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    _repository, _ownership, capability = _publish_uncertain_commit(
        owner,
        binding,
    )

    snapshot = owner.snapshot(binding)
    assert snapshot.git_status is None
    assert dict(snapshot.staging_ownership) == {}
    assert snapshot.commit_recovery == CommitRecoveryProjection(
        message="Commit outcome requires an exact repository check.",
        can_check_again=True,
    )
    assert not hasattr(snapshot, "quarantined_ownership")
    assert not hasattr(snapshot, "recovery_capability")

    for _ordinary_action in ("stage", "unstage", "commit"):
        admission = owner.admit_mutation(binding)
        assert admission.lease is None
        assert admission.reason == "recovery_required"

    wrong = CommitRecoveryCapability(owner, object())
    rejected = owner.admit_commit_recovery(binding, wrong)
    assert rejected.lease is None
    assert rejected.capture is None
    assert rejected.reason == "invalid_capability"

    admitted = owner.admit_commit_recovery(binding, capability)
    assert admitted.reason is None
    assert admitted.lease is not None
    assert admitted.capture is not None
    assert dict(admitted.capture.ownership) == _ownership
    admitted.lease.release()


def test_commit_quarantine_exact_recovery_restores_only_captured_ownership(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, ownership, capability = _publish_uncertain_commit(
        owner,
        binding,
    )
    admission = owner.admit_commit_recovery(binding, capability)
    assert admission.lease is not None
    assert admission.capture is not None

    publication = owner.publish_commit_outcome(
        admission.lease,
        admission.capture,
        CommitPublication(state="failed_unchanged"),
    )

    assert publication.published
    snapshot = owner.snapshot(binding)
    assert dict(snapshot.staging_ownership) == ownership
    assert snapshot.commit_recovery is None
    assert owner.admit_mutation(binding).reason == "mutation_active"
    admission.lease.release()
    ordinary = owner.admit_mutation(binding)
    assert ordinary.lease is not None
    assert isinstance(
        owner.capture_commit_authority(
            ordinary.lease,
            binding=binding,
            authority_generation=owner.snapshot(binding).git_authority_generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/main", "b" * 40),
            group_sequence_ids={1: (1,), 2: (2,)},
        ),
        CommitAuthorityCapture,
    )
    ordinary.lease.release()


def test_commit_quarantine_recovery_requires_empty_active_ownership(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    _repository, ownership, capability = _publish_uncertain_commit(
        owner,
        binding,
    )

    assert not owner.publish_ownership(binding, ownership)
    assert dict(owner.snapshot(binding).staging_ownership) == {}
    admission = owner.admit_commit_recovery(binding, capability)
    assert admission.lease is not None
    assert admission.capture is not None
    admission.lease.release()


def test_commit_quarantine_is_discarded_on_rebinding_and_process_exit(
    tmp_path: Path,
) -> None:
    rebound_owner = FileNotesSessionOwner()
    first = rebound_owner.select_root(tmp_path / "first")
    _repository, _ownership, capability = _publish_uncertain_commit(
        rebound_owner,
        first,
    )

    second = rebound_owner.select_root(tmp_path / "second")
    assert rebound_owner.snapshot(second).commit_recovery is None
    assert dict(rebound_owner.snapshot(second).staging_ownership) == {}
    assert (
        rebound_owner.admit_commit_recovery(second, capability).reason
        == "invalid_capability"
    )

    exiting_owner = FileNotesSessionOwner()
    exiting = exiting_owner.select_root(tmp_path / "exiting")
    _repository, _ownership, _capability = _publish_uncertain_commit(
        exiting_owner,
        exiting,
    )
    exiting_owner.shutdown()
    assert exiting_owner.snapshot(exiting).commit_recovery is None
    assert dict(exiting_owner.snapshot(exiting).staging_ownership) == {}


@pytest.mark.asyncio
async def test_settle_git_shutdown_before_shutdown_preserves_quarantine(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    _repository, ownership, capability = _publish_uncertain_commit(
        owner,
        binding,
    )
    projection = owner.snapshot(binding).commit_recovery
    assert projection is not None

    await owner.settle_git_shutdown()

    snapshot = owner.snapshot(binding)
    assert snapshot.commit_recovery == projection
    assert dict(snapshot.staging_ownership) == {}
    assert owner.admit_mutation(binding).reason == "recovery_required"
    recovery = owner.admit_commit_recovery(binding, capability)
    assert recovery.lease is not None
    assert recovery.capture is not None
    assert dict(recovery.capture.ownership) == ownership
    recovery.lease.release()


def test_commit_publication_accepts_exact_active_token_during_shutdown(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )
    lease, capture = _capture_commit_authority(
        owner,
        binding,
        repository,
        sequence_ids,
    )
    publications = []

    class PublishingService:
        def shutdown(self) -> None:
            assert owner.admit_mutation(binding).reason == "shutdown"
            publications.append(
                owner.publish_commit_outcome(
                    lease,
                    capture,
                    CommitPublication(state="failed_unchanged"),
                )
            )

    owner.attach_git_service(PublishingService())
    owner.shutdown()

    assert len(publications) == 1
    assert publications[0].published
    assert owner.admit_mutation(binding).reason == "shutdown"
    lease.release()


@pytest.mark.asyncio
async def test_commit_publication_rejects_token_after_shutdown_settles(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, sequence_ids, _status = _prepare_commit_authority(
        owner,
        binding,
    )
    lease, capture = _capture_commit_authority(
        owner,
        binding,
        repository,
        sequence_ids,
    )
    settlement = asyncio.get_running_loop().create_future()

    class RetainedService:
        def shutdown(self) -> asyncio.Future[None]:
            return settlement

    owner.attach_git_service(RetainedService())
    owner.shutdown()
    settlement.set_result(None)
    await owner.settle_git_shutdown()

    publication = owner.publish_commit_outcome(
        lease,
        capture,
        CommitPublication(state="failed_unchanged"),
    )

    assert not publication.published
    lease.release()


@pytest.mark.asyncio
async def test_commit_quarantine_is_discarded_when_shutdown_settlement_raises(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    _repository, _ownership, capability = _publish_uncertain_commit(
        owner,
        binding,
    )
    admission = owner.admit_commit_recovery(binding, capability)
    assert admission.lease is not None
    assert admission.capture is not None
    projection = owner.snapshot(binding).commit_recovery
    assert projection is not None
    release_settlement = asyncio.Event()
    failure = RuntimeError("retained settlement failed")

    async def fail_after_release() -> None:
        await release_settlement.wait()
        raise failure

    class ReusableRetainedSettlement:
        def __init__(self, task: asyncio.Task[None]) -> None:
            self.task = task

        def __await__(self):
            return asyncio.shield(self.task).__await__()

    class RetainedFailingService:
        def __init__(self, settlement: ReusableRetainedSettlement) -> None:
            self.settlement = settlement

        def shutdown(self) -> ReusableRetainedSettlement:
            return self.settlement

    underlying = asyncio.create_task(fail_after_release())
    owner.attach_git_service(
        RetainedFailingService(ReusableRetainedSettlement(underlying))
    )
    owner.shutdown()
    cancelled_waiter = asyncio.create_task(owner.settle_git_shutdown())
    await asyncio.sleep(0)
    assert not underlying.done()
    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter
    assert not underlying.done()
    pending_projection = CommitRecoveryProjection(
        message="Retained shutdown settlement is still pending.",
        can_check_again=False,
    )
    pending_publication = owner.publish_commit_outcome(
        admission.lease,
        admission.capture,
        CommitPublication(
            state="uncertain",
            recovery_projection=pending_projection,
        ),
    )
    assert pending_publication.published
    assert owner.snapshot(binding).commit_recovery == pending_projection
    assert not owner._commit_publication_closed

    release_settlement.set()
    with pytest.raises(RuntimeError) as raised:
        await owner.settle_git_shutdown()
    assert raised.value is failure

    recovery_projection = owner.snapshot(binding).commit_recovery
    assert owner._commit_publication_closed
    publication = owner.publish_commit_outcome(
        admission.lease,
        admission.capture,
        CommitPublication(state="failed_unchanged"),
    )

    assert (recovery_projection, publication.published) == (None, False)
    assert dict(owner.snapshot(binding).staging_ownership) == {}
    admission.lease.release()


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


def test_status_publication_rejects_invalidation_after_admission(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, _group = _git_owner_state()
    assert owner.publish_trust(binding, repository)
    lease = owner.try_acquire_status(binding)
    assert lease is not None
    admitted_generation = lease.invalidation_generation
    lease.release()

    assert owner.clear_status(binding)
    status_generation = owner.next_status_generation(binding)
    assert status_generation is not None
    late_status = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=status_generation,
        state="ready",
        repository=repository,
    )

    assert not owner.publish_status(
        binding,
        late_status,
        invalidation_generation=admitted_generation,
    )
    assert owner.snapshot(binding).git_status is None


def test_record_change_atomically_invalidates_status_and_preserves_ownership(
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

    assert owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )

    snapshot = owner.snapshot(binding)
    assert tuple(item.change.relative_path for item in snapshot.changes) == (
        "note.md",
    )
    assert snapshot.git_status is None
    assert dict(snapshot.staging_ownership) == {1: ownership}


def test_record_change_rejects_status_admitted_before_the_change(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    binding = owner.select_root(tmp_path / "notes")
    repository, _ownership, _group = _git_owner_state()
    assert owner.publish_trust(binding, repository)
    lease = owner.try_acquire_status(binding)
    assert lease is not None
    admitted_generation = lease.invalidation_generation
    lease.release()

    assert owner.record_change(
        binding,
        SessionChange("modified", "note.md"),
    )
    status_generation = owner.next_status_generation(binding)
    assert status_generation is not None
    late_status = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=status_generation,
        state="ready",
        repository=repository,
    )

    assert not owner.publish_status(
        binding,
        late_status,
        invalidation_generation=admitted_generation,
    )
    assert owner.snapshot(binding).git_status is None


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


def test_mutation_active_is_exact_binding_read_only_query(
    tmp_path: Path,
) -> None:
    owner = FileNotesSessionOwner()
    first = owner.select_root(tmp_path / "first")
    stale_same_path = SessionBinding(first.root_key, first.generation - 1)
    mutation = owner.try_acquire_mutation(first)
    assert mutation is not None

    assert owner.mutation_active(first)
    assert not owner.mutation_active(stale_same_path)

    mutation.release()
    assert not owner.mutation_active(first)


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
