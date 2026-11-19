"""TASK-34000.2 (review finding N-02): an end-of-note edit must not wedge sync.

A vault whose files end with a newline is the ordinary case. Ctrl+End in the
note editor parks the caret AFTER that newline, so the first word typed there
leaves the note's content without a trailing newline. ``serialize`` re-adds
the newline the file's captured profile carries, and the write postcondition
then compared the file's text with the RAW note content -- a correctly written
file never matched its own note, the operation was fenced at
``postcondition_failed``, and the whole folder stopped syncing both ways.

Every test here runs the production executor, runtime and POSIX filesystem
over a real ``CharactersRAGDB`` and a real file in a temp vault. The bytes on
disk and the rows in the stores are the evidence; nothing is mocked.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Notes.notes_sync_tail_edit_support import (
    TAIL_EDIT as _TAIL_EDIT,
    VAULT_TEXT as _VAULT_TEXT,
    Vault as _Vault,
    build_owner as _owner,
    wedge_root as _wedge,
)
from tldw_chatbook.Notes.notes_device_state_store import NotesDeviceStateStore
from tldw_chatbook.Notes.notes_scope_service import ScopeType
from tldw_chatbook.Notes.notes_sync_authority import NotesScopeSyncAuthority
from tldw_chatbook.Notes.notes_sync_executor import (
    NotesSyncExecutionRequest,
    NotesSyncExecutor,
)
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncActionKind,
    NotesSyncDirection,
    NotesSyncOperationState,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.fixture
def vault(tmp_path: Path):
    selected = _Vault(tmp_path)
    try:
        yield selected
    finally:
        selected.close()


async def test_executor_completes_a_tail_edit_under_a_final_newline_profile(
    vault: _Vault,
) -> None:
    """AC#5: the executor-level pin. No runtime, no planner -- one request."""

    vault.edit_note(_TAIL_EDIT)
    store = NotesDeviceStateStore(vault.state_path)
    filesystem = PosixNotesSyncFilesystem(vault.root)
    filesystem.__enter__()
    try:
        authority = NotesScopeSyncAuthority(
            vault.scope_service,
            scope=ScopeType.LOCAL_NOTE,
            user_id="user-1",
            note_scope_id="local_note",
        )
        note = await authority.observe("note-1")
        file = filesystem.observe("quotes.md")
        assert not note.content.endswith("\n")
        executor = NotesSyncExecutor(
            store, authority, filesystem, recovery_capacity_bytes=1024 * 1024
        )
        result = await executor.execute(
            NotesSyncExecutionRequest(
                operation_id="operation-tail",
                root_id="root-1",
                logical_folder_id="folder-1",
                direction=NotesSyncDirection.BIDIRECTIONAL,
                binding_id="binding-1",
                observation_token="observation-tail",
                action_kind=NotesSyncActionKind.UPDATE_FILE,
                note=note,
                file=file,
                desired_title=note.title,
                recovery_id="recovery-operation-tail",
                recovery_expires_at=2**62,
            )
        )

        assert (result.state, result.reason_code) == (
            NotesSyncOperationState.COMPLETED,
            None,
        )
        # The file keeps its own convention: the note's words, then the
        # newline the vault's profile carries.
        assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")
        assert store.list_incomplete_operations() == ()
        written = filesystem.observe("quotes.md")
        binding = store.get_binding("binding-1")
        assert binding.content_digest == written.observation.content_digest
        assert binding.serialization.final_newline is True
    finally:
        filesystem.__exit__(None, None, None)
        store.close()


async def test_a_tail_edit_reaches_the_file_and_disk_edits_keep_flowing(
    vault: _Vault,
) -> None:
    """AC#1, end to end through the production runtime."""

    owner = _owner(vault)
    await owner.start()
    try:
        vault.edit_note(_TAIL_EDIT)
        await owner.note_changed("note-1")
        await owner.settle()

        assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")
        assert vault.incomplete() == []

        # The folder still syncs in both directions afterwards: a later disk
        # append arrives in the note ...
        appended = _TAIL_EDIT + "\nfrom the vault\n"
        vault.file.write_bytes(appended.encode("utf-8"))
        assert owner.schedule_hint("root-1") is not None
        await owner.settle()
        assert vault.note()["content"] == appended
        assert vault.incomplete() == []

        # ... and the next tail edit in Notes still reaches the file.
        vault.edit_note(appended + "and back")
        await owner.note_changed("note-1")
        await owner.settle()
        assert vault.file.read_bytes() == (appended + "and back\n").encode("utf-8")
        assert vault.incomplete() == []
    finally:
        await owner.shutdown()


async def test_recovery_heals_a_root_wedged_by_the_old_comparison(
    vault: _Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#2: Recovery closes the entry; nothing is lost and sync resumes."""

    owner = _owner(vault)
    await owner.start()
    try:
        operation_id = await _wedge(vault, owner, monkeypatch)

        await owner.resolve_cleanup("root-1", operation_id)

        assert vault.incomplete() == []
        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("up_to_date", "sync_now")
        # Neither side was rewritten by the heal.
        assert vault.note()["content"] == _TAIL_EDIT
        assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")
        # Checks work again, and a later disk edit reaches the note.
        await owner.check_root("root-1")
        appended = _TAIL_EDIT + "\nafter the heal\n"
        vault.file.write_bytes(appended.encode("utf-8"))
        assert owner.schedule_hint("root-1") is not None
        await owner.settle()
        assert vault.note()["content"] == appended
        assert vault.incomplete() == []
    finally:
        await owner.shutdown()


async def test_recovery_heals_a_wedge_the_user_kept_typing_past(
    vault: _Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#2: the typing that continued after the wedge reaches the file."""

    owner = _owner(vault)
    await owner.start()
    try:
        operation_id = await _wedge(vault, owner, monkeypatch)
        later = _TAIL_EDIT + " that kept going"
        vault.edit_note(later)
        # The held root ignores the save's hint -- this is the old lie.
        assert await owner.note_changed("note-1") == ()

        await owner.resolve_cleanup("root-1", operation_id)

        assert vault.incomplete() == []
        assert vault.note()["content"] == later
        assert vault.file.read_bytes() == (later + "\n").encode("utf-8")
        assert owner.snapshot().roots[0].status == "up_to_date"
    finally:
        await owner.shutdown()


async def test_a_mid_write_disk_edit_ends_in_a_conflict_review_not_a_loop(
    vault: _Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#3: the app/disk race lands on an ordinary conflict, both sides kept."""

    owner = _owner(vault)
    await owner.start()
    disk_side = _VAULT_TEXT + "| riley-disk | 2 |\n"
    original_admit = NotesSyncExecutor._admit

    def racing_admit(self, request):
        admitted = original_admit(self, request)
        # The disk edit lands after the plan and before the write.
        vault.file.write_bytes(disk_side.encode("utf-8"))
        return admitted

    try:
        with monkeypatch.context() as patch:
            patch.setattr(NotesSyncExecutor, "_admit", racing_admit)
            vault.edit_note(_VAULT_TEXT + "| riley-app | 1 | app2")
            await owner.note_changed("note-1")
            await owner.settle()
        assert vault.incomplete() == [
            ("update_file", "needs_attention", "stale_observation")
        ]
        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("needs_attention", "resolve_cleanup")

        await owner.resolve_cleanup("root-1", root.action_id)

        # The entry is closed; the two edits now wait as a conflict for an
        # explicit Keep file / Keep note / Keep both. Nothing was overwritten.
        assert vault.incomplete() == []
        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("needs_attention", "review_changes")
        assert vault.note()["content"] == _VAULT_TEXT + "| riley-app | 1 | app2"
        assert vault.file.read_bytes() == disk_side.encode("utf-8")
        plan = await owner.check_root("root-1")
        assert [(item.kind.value, item.reason_code) for item in plan.attention] == [
            ("conflict", "both_sides_changed")
        ]
    finally:
        await owner.shutdown()


async def test_a_raw_note_baseline_still_reads_as_unchanged(tmp_path: Path) -> None:
    """Baselines kept before this fix (and migrated ones) hold the RAW digest.

    A migrated binding carries a placeholder profile (no final newline) and
    the raw note digest. Measuring its note only in represented form would
    report every note ending in a newline as changed and plan a write nobody
    asked for; either form matching the baseline must read as unchanged.
    """

    vault = _Vault(tmp_path)
    try:
        store = NotesDeviceStateStore(vault.state_path)
        binding = store.get_binding("binding-1")
        placeholder = replace(
            binding,
            serialization=replace(binding.serialization, final_newline=False),
        )
        with store.transaction(immediate=True) as connection:
            connection.execute(
                "UPDATE notes_sync_bindings SET final_newline = 0 WHERE binding_id = ?",
                ("binding-1",),
            )
        assert store.get_binding("binding-1") == placeholder
        store.close()
        owner = _owner(vault)
        await owner.start()
        try:
            root = NotesDeviceStateStore(vault.state_path).get_root("root-1")
            observations = await owner._adapter.observe_root(root)
            [observed] = observations.bindings
            assert observed.note_digest == observed.baseline_note_digest
            assert vault.incomplete() == []
        finally:
            await owner.shutdown()
    finally:
        vault.close()


async def test_status_listeners_hear_every_publication_and_never_break_a_pass(
    vault: _Vault,
) -> None:
    """Fix round 1 (review Important #2): the ``_publish`` seam has listeners.

    Registration is idempotent per object, a raising listener is swallowed
    (the pass still writes the file), and a removed listener hears nothing.
    """

    owner = _owner(vault)
    await owner.start()
    seen: list[tuple[str, str]] = []

    def listener(snapshot) -> None:
        seen.append((snapshot.root_id, snapshot.status))

    def exploding(_snapshot) -> None:
        raise RuntimeError("listener failure must not reach the pass")

    try:
        owner.add_status_listener(listener)
        owner.add_status_listener(listener)
        owner.add_status_listener(exploding)
        with pytest.raises(TypeError):
            owner.add_status_listener("not callable")  # type: ignore[arg-type]

        vault.edit_note(_TAIL_EDIT)
        await owner.note_changed("note-1")
        await owner.settle()

        assert vault.file.read_bytes() == (_TAIL_EDIT + "\n").encode("utf-8")
        assert seen, "the pass published at least once"
        assert seen[-1] == ("root-1", "up_to_date")
        assert len(seen) == len([s for s in seen if s[0] == "root-1"])
        published = len(seen)

        owner.remove_status_listener(listener)
        owner.remove_status_listener(listener)  # unknown listeners are ignored
        vault.file.write_bytes((_TAIL_EDIT + "\nfrom disk\n").encode("utf-8"))
        assert owner.schedule_hint("root-1") is not None
        await owner.settle()
        assert vault.note()["content"] == _TAIL_EDIT + "\nfrom disk\n"
        assert len(seen) == published
    finally:
        await owner.shutdown()
