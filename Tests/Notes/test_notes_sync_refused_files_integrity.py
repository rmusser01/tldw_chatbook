"""TASK-34000.16: a refused file never costs a note (data-integrity pins).

One bad file in a kept-in-sync folder -- non-UTF-8 bytes, mixed newlines --
is a NAMED SKIP row, never the folder's refusal, and never a reason to plan
anything against the note that file is bound to. Every test here runs the
production runtime, executor and POSIX filesystem over a real
``NotesDeviceStateStore`` and real files in a temp vault; the bindings, the
plan and the bytes on disk are the evidence.

Moved here from ``test_notes_sync_runtime.py`` (P1-C final fix round, I-D
ruling) so these five guarantees run on every pull request in the
admission-sensitive PR Fast Lane step (``.github/workflows/
derived-artifacts.yml``; pinned by ``Tests/CI/test_ci_queue_pressure_
contract.py``). The runtime file is in no PR lane. Fixtures and helpers stay
in the runtime module and are imported from it, so nothing is duplicated:

- identity-first rename: a renamed bound file whose OLD name is now a bad
  file is a move, not a duplicate note (fix round 1, I-1);
- a renamed AND corrupted bound file is a skip, not a deletion review
  (fix round 1, I-3);
- a bound file that turns non-UTF-8 is never planned as a deletion
  (Review Focus);
- a migrated root with a refused candidate refuses activation BEFORE any
  mutation (fix round 1, I-2);
- a non-gate filesystem error on one file still refuses the whole root
  (fix round 1, I-5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from Tests.Notes.test_notes_sync_runtime import (
    _BAD_BYTES,
    _activate_two_file_vault,
    _migration_root_with_one_refused_candidate,
    _refusing_owner,
    _store,
    _vault_owner,
)
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
)

#: A real SQLite store under ``tmp_path`` (the lasting-sync device state),
#: so the file is ``bootstrap_profile`` like the other real-stack
#: ``test_notes_sync_*.py`` files in the admission-sensitive step.
pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.asyncio
async def test_a_bound_file_that_turns_non_utf8_is_skipped_not_planned_as_a_deletion(
    tmp_path: Path,
) -> None:
    """TASK-34000.16 Review Focus: a Check alone must never plan to delete a
    note whose file exists.

    A refused file is not in the observed set; without the runtime's
    exclusion its ACTIVE binding would reach the planner with ``file=None``
    and be read as a deleted file. The pure model refuses that input; this
    pins it on the real adapter.
    """

    from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationAttentionKind
    from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRootSetup

    root_path = tmp_path / "vault"
    root_path.mkdir()
    (root_path / "ok.md").write_text("# Ok\n", encoding="utf-8")
    (root_path / "note.md").write_text("# Note\n", encoding="utf-8")
    _store(tmp_path)
    owner, _folders, local_notes = _vault_owner(tmp_path)
    await owner.start()
    try:
        review = await owner.review_setup(
            NotesSyncRootSetup(
                display_name="Vault",
                canonical_path=str(root_path),
                note_scope_id="local_note",
                direction=NotesSyncDirection.BIDIRECTIONAL,
            )
        )
        result = await owner.activate_root(review.root_id, review.observation_token)
        assert result.accepted is True and result.applied_count == 2
        bound_before = {
            b.normalized_relative_path: b.state
            for b in owner._store.list_bindings(review.root_id)
        }
        (root_path / "note.md").write_bytes(b"\xe9 legacy\n")  # latin-1 now
        (root_path / "ok.md").write_text("# Ok\n\nedited\n", encoding="utf-8")
        checked = await owner.check_root(review.root_id)
        bound_after = {
            b.normalized_relative_path: b.state
            for b in owner._store.list_bindings(review.root_id)
        }
    finally:
        await owner.shutdown()

    assert [(s.relative_path, s.reason_code) for s in checked.item_skips] == [
        ("note.md", "unsupported_encoding")
    ]
    assert not any(
        item.kind is ReconciliationAttentionKind.DELETION_REVIEW
        for item in checked.attention
    ), checked.attention
    assert checked.deletion_groups == ()
    # The disk edit elsewhere in the folder is still planned (AC#1).
    assert [action.kind for action in checked.safe_actions] == [
        NotesSyncActionKind.UPDATE_NOTE
    ]
    # A Check is mutation-free: the refused file's binding and its note are
    # exactly as activation left them.
    assert bound_after == bound_before
    assert bound_after["note.md"] is NotesSyncBindingState.ACTIVE
    assert (root_path / "note.md").read_bytes() == b"\xe9 legacy\n"


@_BAD_BYTES
@pytest.mark.asyncio
async def test_a_rename_with_a_new_bad_file_at_the_old_name_is_a_move_not_a_duplicate_note(
    tmp_path: Path, payload: bytes, reason: str
) -> None:
    """TASK-34000.16 fix round 1, I-1.

    Skipping the refused path's binding BEFORE the identity fallback hid its
    claim on the inode now at new.md: the automatic pass planned new.md as a
    create, the executor's claimed-identity check (TASK-34789) failed after
    the note insert, the notes DB gained a duplicate "new" note and the root
    wedged on an incomplete operation. The identity lookup now runs first.
    """

    root_path, owner, _folders, local_notes, root_id = await _activate_two_file_vault(
        tmp_path
    )
    try:
        (root_path / "old.md").rename(root_path / "new.md")
        (root_path / "old.md").write_bytes(payload)
        root = owner._store.get_root(root_id)
        plan = await owner._reconcile(root, automatic=True)
        checked = await owner.check_root(root_id)
    finally:
        await owner.shutdown()

    assert [(s.relative_path, s.reason_code) for s in plan.item_skips] == [
        ("old.md", reason)
    ]
    assert NotesSyncActionKind.CREATE_NOTE not in {a.kind for a in plan.safe_actions}
    # The rename IS seen: the bound binding is observed at new.md.
    assert [effect.kind.value for effect in plan.managed_placement_effects] == [
        "file_move"
    ]
    assert owner._store.list_incomplete_operations() == ()
    assert len(local_notes.notes) == 2, sorted(n["title"] for n in local_notes.notes.values())
    assert owner.snapshot().roots[0].next_action != "resolve_cleanup"
    assert checked.item_skips == plan.item_skips
    assert (root_path / "old.md").read_bytes() == payload


@pytest.mark.asyncio
async def test_a_renamed_and_corrupted_bound_file_is_a_skip_not_a_deletion_review(
    tmp_path: Path,
) -> None:
    """TASK-34000.16 fix round 1, I-3: the content sits under the refused
    file's skip row; the review must not offer to delete its note."""

    from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationAttentionKind

    root_path, owner, _folders, local_notes, root_id = await _activate_two_file_vault(
        tmp_path
    )
    try:
        (root_path / "old.md").rename(root_path / "renamed.md")
        (root_path / "renamed.md").write_bytes(b"caf\xe9\n")
        checked = await owner.check_root(root_id)
    finally:
        await owner.shutdown()

    assert [(s.relative_path, s.reason_code) for s in checked.item_skips] == [
        ("renamed.md", "unsupported_encoding")
    ]
    assert not any(
        item.kind is ReconciliationAttentionKind.DELETION_REVIEW
        for item in checked.attention
    ), checked.attention
    assert checked.deletion_groups == ()
    assert NotesSyncActionKind.CREATE_NOTE not in {a.kind for a in checked.safe_actions}
    assert len(local_notes.notes) == 2


@pytest.mark.asyncio
async def test_a_migrated_root_with_a_refused_candidate_refuses_activation_before_any_mutation(
    tmp_path: Path,
) -> None:
    """TASK-34000.16 fix round 1, I-2.

    The refused candidate has no action, so ``activate_root`` found a
    candidate id missing from the reviewed set -- AFTER it had created the
    "Migrated notes" folder and recorded activation recovery, leaving the
    root paused with ``check_root`` raising ``sync_root_not_active``. The
    refusal now comes first and mutates nothing; fixing the file lets the
    next review activate.
    """

    store, root_path, root_id, folders, owner = _migration_root_with_one_refused_candidate(
        tmp_path
    )
    await owner.start()
    try:
        review = await owner.check_root(root_id)
        assert [(s.relative_path, s.reason_code) for s in review.item_skips] == [
            ("b.md", "mixed_newlines")
        ]
        result = await owner.activate_root(root_id, review.observation_token)
        assert result.accepted is False
        assert folders.created == []
        root = store.get_root(root_id)
        assert root.state is NotesSyncRootState.PAUSED
        assert root.last_status_code == "migration_review_required"
        assert root.logical_folder_id is None
        # The root is not wedged: the next review still builds.
        again = await owner.check_root(root_id)
        assert again.item_skips == review.item_skips

        (root_path / "b.md").write_bytes(b"a\nb\nc\n")
        fixed = await owner.check_root(root_id)
        assert fixed.item_skips == ()
        activated = await owner.activate_root(root_id, fixed.observation_token)
    finally:
        await owner.shutdown()

    assert activated.accepted is True
    assert store.get_root(root_id).state is NotesSyncRootState.ACTIVE
    assert len(folders.created) == 1


@pytest.mark.asyncio
async def test_a_non_gate_filesystem_error_on_one_file_still_refuses_the_whole_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TASK-34000.16 fix round 1, I-5: the non-gate arm of the walk is pinned.

    ``root_identity_changed`` means the folder itself moved under the walk;
    that is the root's refusal, carried with its own reason code, never a
    per-file skip and never silently dropped.
    """

    from tldw_chatbook.Notes.notes_sync_filesystem import NotesSyncFilesystemError
    from tldw_chatbook.Notes.notes_sync_runtime import (
        NotesSyncRootRefused,
        NotesSyncRootSetup,
    )

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "ok.md").write_text("body", encoding="utf-8")
    (vault / "note.md").write_text("body", encoding="utf-8")
    real_observe = PosixNotesSyncFilesystem.observe

    def observe(self, relative_path, **kwargs):
        if Path(relative_path).as_posix() == "note.md":
            raise NotesSyncFilesystemError("root_identity_changed")
        return real_observe(self, relative_path, **kwargs)

    monkeypatch.setattr(PosixNotesSyncFilesystem, "observe", observe)
    owner = _refusing_owner(tmp_path)
    await owner.start()
    try:
        with pytest.raises(NotesSyncRootRefused) as raised:
            await owner.review_setup(
                NotesSyncRootSetup(
                    display_name="Vault",
                    canonical_path=str(vault),
                    note_scope_id="local_note",
                    direction=NotesSyncDirection.BIDIRECTIONAL,
                )
            )
    finally:
        await owner.shutdown()

    assert str(raised.value) == "root_discovery_incomplete"
    assert raised.value.reason_code == "root_identity_changed"

