"""TASK-34000.49 (TASK-34000 wave 1c, Task 1).

A note whose ``version`` moved without a content change -- a soft-delete and
restore, a keywords-only save, a title-only edit -- used to wedge every later
file-to-note update. ``NotesSyncExecutor._validate_initial`` refused the
``update_note`` as ``stale_observation`` because the binding's recorded
``note_version`` no longer matched the live note, even though the content
baseline (``_note_matches_baseline``) still did. The reconciler is
digest-only, so nothing ever re-based the binding, and the root published
"needs_attention / review_changes" with nothing to review.

The rule these tests pin: ``binding.note_version`` is the version at the last
baseline commit and is compared only against journal-recorded binding facts;
the content baseline and the exact fresh re-observe carry "the note is still
what we synced". They run the production runtime owner, the real executor and
the POSIX filesystem over a real ``CharactersRAGDB`` and a real ``.md`` in a
temp vault. Bytes on disk, rows in the notes DB and the binding row in the
sync store are the evidence. The negative control pins the invariant the
version proxy was standing in for: a note whose CONTENT moved is still
refused and goes to a two-sided review, with both texts intact.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from Tests.Notes.notes_sync_tail_edit_support import (
    VAULT_TEXT,
    Vault,
    build_owner,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationAttentionKind

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]

VAULT_BYTES = VAULT_TEXT.encode("utf-8")
#: The disk edit every test makes after the version-only move.
APPENDED = VAULT_TEXT + "from the vault\n"
APPENDED_BYTES = APPENDED.encode("utf-8")


@pytest.fixture
def vault(tmp_path: Path):
    selected = Vault(tmp_path)
    try:
        yield selected
    finally:
        selected.close()


def _binding_version(vault: Vault) -> int:
    store = NotesDeviceStateStore(vault.state_path)
    try:
        return int(store.get_binding("binding-1").note_version)
    finally:
        store.close()


def _operation_count(vault: Vault) -> int:
    """Every operation the store ever journaled, complete or not."""

    with sqlite3.connect(vault.state_path) as connection:
        row = connection.execute(
            "SELECT COUNT(*) FROM notes_sync_operations"
        ).fetchone()
    return int(row[0])


async def _delete(vault: Vault) -> int:
    """Soft-delete note-1 through the seam the Library's Delete uses."""

    version = int(vault.note()["version"])
    deleted = await vault.scope_service.delete_note(
        scope="local_note", note_id="note-1", version=version, user_id="user-1"
    )
    assert deleted
    state = vault.database.get_note_version_states(["note-1"])["note-1"]
    assert state["deleted"] is True
    return int(state["version"])


async def _restore(vault: Vault, tombstone_version: int) -> None:
    """Restore note-1 through the seam the receipt's Undo and Trash use."""

    record = await vault.scope_service.restore_note(
        scope="local_note",
        note_id="note-1",
        version=tombstone_version,
        user_id="user-1",
    )
    assert record["id"] == "note-1"
    state = vault.database.get_note_version_states(["note-1"])["note-1"]
    assert state["deleted"] is False


async def _assert_signal_leaves_root_healthy_without_an_operation(
    vault: Vault, owner
) -> None:
    """The version-only move is no change to the reconciler: nothing runs."""

    operations_before = _operation_count(vault)
    assert await owner.note_changed("note-1") == ("root-1",)
    await owner.settle()
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("up_to_date", "sync_now")
    assert root.action_id is None
    assert vault.incomplete() == []
    assert _operation_count(vault) == operations_before
    assert vault.file.read_bytes() == VAULT_BYTES


async def _assert_next_disk_edit_flows_into_the_note(vault: Vault, owner) -> None:
    """The automatic pass applies the file edit and re-bases the binding."""

    vault.file.write_bytes(APPENDED_BYTES)
    assert owner.schedule_hint("root-1") is not None
    await owner.settle()
    note = vault.note()
    assert note["content"] == APPENDED
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("up_to_date", "sync_now")
    assert vault.incomplete() == []
    assert vault.file.read_bytes() == APPENDED_BYTES
    assert _binding_version(vault) == int(note["version"])


async def test_a_keywords_only_save_leaves_the_folder_able_to_apply_the_next_disk_edit(
    vault: Vault,
) -> None:
    """AC#2: a keywords-only edit moves the version, not the content."""

    owner = build_owner(vault)
    await owner.start()
    try:
        before = vault.note()
        await vault.scope_service.save_note(
            scope="local_note",
            title=before["title"],
            content=before["content"],
            note_id="note-1",
            version=int(before["version"]),
            user_id="user-1",
            keywords=["alpha"],
        )
        after = vault.note()
        assert int(after["version"]) > int(before["version"])
        assert after["content"] == VAULT_TEXT
        # Nothing wrote a baseline: the binding still carries the old version.
        assert _binding_version(vault) == int(before["version"])

        await _assert_signal_leaves_root_healthy_without_an_operation(vault, owner)
        await _assert_next_disk_edit_flows_into_the_note(vault, owner)
    finally:
        await owner.shutdown()


async def test_a_delete_and_restore_leaves_the_folder_able_to_apply_the_next_disk_edit(
    vault: Vault,
) -> None:
    """AC#1: the restore moves the version twice; the folder keeps syncing."""

    owner = build_owner(vault)
    await owner.start()
    try:
        baseline_version = _binding_version(vault)
        tombstone_version = await _delete(vault)
        await owner.note_changed("note-1")
        await owner.settle()
        assert owner.snapshot().roots[0].status == "needs_attention"
        await _restore(vault, tombstone_version)
        assert int(vault.note()["version"]) > baseline_version
        assert vault.note()["content"] == VAULT_TEXT
        assert _binding_version(vault) == baseline_version

        await _assert_signal_leaves_root_healthy_without_an_operation(vault, owner)
        await _assert_next_disk_edit_flows_into_the_note(vault, owner)
    finally:
        await owner.shutdown()


async def test_a_title_only_edit_leaves_the_folder_able_to_apply_the_next_disk_edit(
    vault: Vault,
) -> None:
    """Review focus: a title edit moves the version; the title is kept, too.

    The planner's digest covers content only, so a title-only edit plans no
    change and never re-bases the binding -- the same wedge shape. The later
    ``update_note`` writes the LIVE note's title (``desired_title`` is lifted
    from the observed note), so the rename survives the file edit: no side
    wins silently on the title either.
    """

    owner = build_owner(vault)
    await owner.start()
    try:
        before = vault.note()
        renamed = "quotes (renamed in the app)"
        assert vault.database.update_note(
            "note-1",
            {"title": renamed, "content": before["content"]},
            int(before["version"]),
        )
        after = vault.note()
        assert (after["title"], after["content"]) == (renamed, VAULT_TEXT)
        assert int(after["version"]) > int(before["version"])
        assert _binding_version(vault) == int(before["version"])

        await _assert_signal_leaves_root_healthy_without_an_operation(vault, owner)
        await _assert_next_disk_edit_flows_into_the_note(vault, owner)
        assert vault.note()["title"] == renamed
    finally:
        await owner.shutdown()


async def test_a_membership_only_change_leaves_the_folder_able_to_apply_the_next_disk_edit(
    vault: Vault,
) -> None:
    """Review focus: a folder-membership change is a membership-row fact.

    Attaching the note to a second folder by hand moves the membership row's
    version and leaves the note row's alone (``_ensure_manual_membership``
    only ever updates ``note_folder_memberships``), so this is the control
    that proves the fix is not needed here -- and that a membership change
    on a bound note does not disturb the root either way.
    """

    owner = build_owner(vault)
    await owner.start()
    try:
        before = vault.note()
        folders = LocalNoteFolderRepository(vault.database)
        folders.create_folder(name="Manual", parent_id=None, folder_id="folder-2")
        membership = folders.attach_manual(
            folder_id="folder-2",
            note_id="note-1",
            expected_note_version=int(before["version"]),
        )
        assert membership.note_id == "note-1"
        after = vault.note()
        assert int(after["version"]) == int(before["version"])
        assert after["content"] == VAULT_TEXT
        assert _binding_version(vault) == int(before["version"])

        await _assert_signal_leaves_root_healthy_without_an_operation(vault, owner)
        await _assert_next_disk_edit_flows_into_the_note(vault, owner)
    finally:
        await owner.shutdown()


async def test_a_note_whose_content_moved_is_still_refused_and_reviewed(
    vault: Vault,
) -> None:
    """Negative control: version AND content moved while the file changed too.

    Dropping the version proxy must not open a path for a content change:
    the content baseline still refuses the note, the planner holds the pair
    as ``both_sides_changed``, no operation opens, and both texts are intact
    where they were typed. No side wins.
    """

    owner = build_owner(vault)
    await owner.start()
    try:
        baseline_version = _binding_version(vault)
        typed = VAULT_TEXT + "typed in the app\n"
        vault.edit_note(typed)
        assert int(vault.note()["version"]) > baseline_version
        vault.file.write_bytes(APPENDED_BYTES)

        operations_before = _operation_count(vault)
        assert owner.schedule_hint("root-1") is not None
        await owner.settle()
        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("needs_attention", "review_changes")
        assert root.action_id is None
        assert vault.incomplete() == []
        assert _operation_count(vault) == operations_before

        plan = await owner.check_root("root-1")
        assert [(item.kind, item.reason_code) for item in plan.attention] == [
            (ReconciliationAttentionKind.CONFLICT, "both_sides_changed")
        ]
        assert plan.safe_actions == ()
        assert _operation_count(vault) == operations_before
        assert vault.note()["content"] == typed
        assert vault.file.read_bytes() == APPENDED_BYTES
        assert _binding_version(vault) == baseline_version
    finally:
        await owner.shutdown()
