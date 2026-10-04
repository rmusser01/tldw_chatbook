"""TASK-32633 slice (TASK-34000 wave 1, Task 4; review finding N-03).

Deleting a synced note left its root reading "✓ Up to date" while the file
was still on disk, because the delete and restore seams never told lasting
sync anything. What a note-side deletion DOES to the file is not this
slice's to decide: the reconciler already classifies it as a deletion
review (``note_missing``) -- the root is held for attention and no file is
removed on its own. These tests pin that the signal reaches the runtime and
that the runtime answers it the way the design says, on the production
runtime, executor and POSIX filesystem over a real ``CharactersRAGDB`` and
a real ``.md`` in a temp vault. Bytes on disk and rows in the stores are the
evidence.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import time
from pathlib import Path

import pytest

from Tests.Notes.notes_sync_tail_edit_support import (
    VAULT_TEXT,
    Vault,
    build_owner,
    wedge_root,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncOperationRecord,
    NotesSyncRootRecord,
)
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncOperationState,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_reconciler import ReconciliationAttentionKind
from tldw_chatbook.Notes.notes_sync_runtime import (
    NotesSyncRuntimeOwner,
    _ProductionRuntimeAdapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]

VAULT_BYTES = VAULT_TEXT.encode("utf-8")


@pytest.fixture
def vault(tmp_path: Path):
    selected = Vault(tmp_path)
    try:
        yield selected
    finally:
        selected.close()


def _tombstone(vault: Vault) -> dict:
    """``(version, deleted)`` of note-1, including a soft-deleted row."""

    return vault.database.get_note_version_states(["note-1"])["note-1"]


async def _delete(vault: Vault) -> int:
    """Soft-delete note-1 through the seam the Library's Delete uses."""

    version = int(vault.note()["version"])
    deleted = await vault.scope_service.delete_note(
        scope="local_note", note_id="note-1", version=version, user_id="user-1"
    )
    assert deleted
    state = _tombstone(vault)
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
    assert _tombstone(vault)["deleted"] is False


async def test_deleting_a_bound_note_holds_the_root_and_leaves_its_file_on_disk(
    vault: Vault,
) -> None:
    """The design, pinned: a note-side deletion is a review, never a file delete."""

    owner = build_owner(vault)
    await owner.start()
    try:
        await _delete(vault)
        assert await owner.note_changed("note-1") == ("root-1",)
        await owner.settle()

        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("needs_attention", "review_changes")
        # A planner hold, not Recovery's: no operation was opened.
        assert root.action_id is None
        assert vault.incomplete() == []
        # The file is exactly what it was. Nothing chose a winner.
        assert vault.file.exists()
        assert vault.file.read_bytes() == VAULT_BYTES

        plan = await owner.check_root("root-1")
        assert [(item.kind, item.reason_code) for item in plan.attention] == [
            (ReconciliationAttentionKind.DELETION_REVIEW, "note_missing")
        ]
        assert vault.file.read_bytes() == VAULT_BYTES
    finally:
        await owner.shutdown()


async def test_restoring_the_deleted_note_releases_the_hold_and_the_root_is_healthy(
    vault: Vault,
) -> None:
    """Undo of a delete signals too, and the root returns to up to date.

    The hold the deletion produced is the planner's own classification; the
    note coming back is the resolution, so the restore's signal re-plans the
    root (the ``_run_settled_pass`` precedent) instead of being refused as a
    hint on a blocked root -- which is what left "⚠ Needs attention" standing
    over a folder with nothing to review.
    """

    owner = build_owner(vault)
    await owner.start()
    try:
        tombstone_version = await _delete(vault)
        assert await owner.note_changed("note-1") == ("root-1",)
        await owner.settle()
        held = owner.snapshot().roots[0]
        assert held.status == "needs_attention"

        await _restore(vault, tombstone_version)
        assert await owner.note_changed("note-1") == ("root-1",)
        await owner.settle()

        root = owner.snapshot().roots[0]
        assert (root.status, root.next_action) == ("up_to_date", "sync_now")
        assert vault.incomplete() == []
        assert vault.note()["content"] == VAULT_TEXT
        assert vault.file.read_bytes() == VAULT_BYTES
    finally:
        await owner.shutdown()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TASK-34000.49: the restore moved the note's version without a content "
        "change, and the executor's update_note precondition compares the "
        "binding's recorded version, so the next file-to-note update is refused "
        "as stale_observation (needs_attention, nothing to review). Pre-existing; "
        "found by this slice."
    ),
)
async def test_a_disk_edit_after_a_restore_still_flows_into_the_note(
    vault: Vault,
) -> None:
    """The folder must keep syncing both ways after a delete and restore."""

    owner = build_owner(vault)
    await owner.start()
    try:
        tombstone_version = await _delete(vault)
        await owner.note_changed("note-1")
        await owner.settle()
        await _restore(vault, tombstone_version)
        await owner.note_changed("note-1")
        await owner.settle()
        assert owner.snapshot().roots[0].status == "up_to_date"

        appended = VAULT_TEXT + "from the vault\n"
        vault.file.write_bytes(appended.encode("utf-8"))
        assert owner.schedule_hint("root-1") is not None
        await owner.settle()
        assert vault.note()["content"] == appended
        assert owner.snapshot().roots[0].status == "up_to_date"
    finally:
        await owner.shutdown()


async def test_a_note_change_never_releases_a_hold_recovery_owns(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: an open entry is Recovery's, whatever the note does."""

    owner = build_owner(vault)
    await owner.start()
    try:
        operation_id = await wedge_root(vault, owner, monkeypatch)
        before = owner.snapshot().roots[0]
        assert before.action_id == operation_id

        assert await owner.note_changed("note-1") == ()
        await owner.settle()

        after = owner.snapshot().roots[0]
        assert (after.status, after.next_action, after.action_id) == (
            "needs_attention",
            "resolve_cleanup",
            operation_id,
        )
        assert vault.incomplete() == [
            ("update_file", "needs_attention", "postcondition_failed")
        ]
    finally:
        await owner.shutdown()


async def test_every_published_root_status_says_when_it_was_confirmed(
    vault: Vault,
) -> None:
    """The healthy copy's time comes from the publication, not the paint."""

    owner = build_owner(vault)
    before = time.time()
    await owner.start()
    try:
        root = owner.snapshot().roots[0]
        assert root.status == "up_to_date"
        assert isinstance(root.published_at, float)
        assert before <= root.published_at <= time.time()

        # A later publication carries its own, later, time (review Minor 6:
        # strictly later, so a time that never moved cannot pass).
        await asyncio.sleep(0.005)
        await owner.check_root("root-1")
        later = owner.snapshot().roots[0]
        assert later.status == "up_to_date"
        assert later.published_at > root.published_at
    finally:
        await owner.shutdown()


def _persist_root_status(vault: Vault, code: str) -> None:
    store = NotesDeviceStateStore(vault.state_path)
    try:
        store.update_root_status("root-1", code)
    finally:
        store.close()


async def test_restore_in_a_later_session_releases_the_startup_hold(
    vault: Vault,
) -> None:
    """Fix round 1 (review Important 2): Recently deleted ▸ Restore is used in
    a later session. Startup re-publishes the persisted hold as durable and
    never leases the root; the restore's signal must lease it, re-plan, and
    leave the watcher running -- no Check."""

    first = build_owner(vault)
    await first.start()
    try:
        tombstone_version = await _delete(vault)
        assert await first.note_changed("note-1") == ("root-1",)
        await first.settle()
        assert first.snapshot().roots[0].status == "needs_attention"
    finally:
        await first.shutdown()

    second = build_owner(vault)
    await second.start()
    try:
        held = second.snapshot().roots[0]
        assert (held.status, held.next_action, held.action_id) == (
            "needs_attention",
            "review_changes",
            None,
        )
        # Held since startup: no lease, so no hint can reach it yet.
        assert second.schedule_hint("root-1") is None

        await _restore(vault, tombstone_version)
        assert await second.note_changed("note-1") == ("root-1",)
        await second.settle()

        root = second.snapshot().roots[0]
        assert (root.status, root.next_action) == ("up_to_date", "sync_now")
        assert vault.file.read_bytes() == VAULT_BYTES
        assert vault.note()["content"] == VAULT_TEXT
        # The lease is held and the watcher is running: hints are admitted.
        assert second.schedule_hint("root-1") is not None
        await second.settle()
    finally:
        await second.shutdown()


async def _held_in_a_later_session(vault: Vault) -> int:
    """Delete + signal in one session, shut down; return the tombstone version."""

    first = build_owner(vault)
    await first.start()
    try:
        tombstone_version = await _delete(vault)
        assert await first.note_changed("note-1") == ("root-1",)
        await first.settle()
        assert first.snapshot().roots[0].status == "needs_attention"
    finally:
        await first.shutdown()
    return tombstone_version


@pytest.mark.parametrize("look_first", ["check_root", "request_sync_now"])
async def test_restore_releases_the_hold_after_review_or_check_in_a_later_session(
    vault: Vault, look_first: str
) -> None:
    """Fix round 2 (review 2, Open 1): the held row's own actions -- Review
    (``check_root``) and Check changes (``request_sync_now``) -- lease the root
    and re-block it without starting a watcher. A restore after that must still
    heal the folder on its own: the released pass runs whenever no hint can be
    scheduled, leased or not."""

    tombstone_version = await _held_in_a_later_session(vault)
    second = build_owner(vault)
    await second.start()
    try:
        plan = await getattr(second, look_first)("root-1")
        assert [item.reason_code for item in plan.attention] == ["note_missing"]
        held = second.snapshot().roots[0]
        assert (held.status, held.next_action) == ("needs_attention", "review_changes")
        # Leased by the look, but nothing watches it: a hint is refused.
        assert second.schedule_hint("root-1") is None

        await _restore(vault, tombstone_version)
        assert await second.note_changed("note-1") == ("root-1",)
        await second.settle()

        root = second.snapshot().roots[0]
        assert (root.status, root.next_action) == ("up_to_date", "sync_now")
        assert vault.file.read_bytes() == VAULT_BYTES
        assert vault.note()["content"] == VAULT_TEXT
        assert second.schedule_hint("root-1") is not None
        await second.settle()
    finally:
        await second.shutdown()


async def test_the_released_pass_returns_promptly_and_is_joined_by_settle(
    vault: Vault,
) -> None:
    """Review 2 deferred Minor 2: a restore on a startup-held root must not
    await a whole pass inline; the pass is admitted like a hint and ``settle``
    joins it (Task 2's post-save bounded wait relies on exactly that)."""

    tombstone_version = await _held_in_a_later_session(vault)
    second = build_owner(vault)
    await second.start()
    try:
        await _restore(vault, tombstone_version)
        assert await second.note_changed("note-1") == ("root-1",)
        # The signal returned before the pass finished: the hold still shows
        # (review 3 Minor 7: this status assertion is the proof; no wall clock).
        assert second.snapshot().roots[0].status == "needs_attention"
        await second.settle()
        assert second.snapshot().roots[0].status == "up_to_date"
    finally:
        await second.shutdown()


async def test_a_closed_admission_runtime_neither_leases_nor_runs_from_a_signal(
    vault: Vault,
) -> None:
    """Review 2 deferred Minor 1: maintenance fences the released pass too."""

    tombstone_version = await _held_in_a_later_session(vault)
    second = build_owner(vault)
    await second.start()
    try:
        await _restore(vault, tombstone_version)
        runtime = second._runtime if hasattr(second, "_runtime") else second
        runtime._maintenance_close_admission()
        try:
            assert await second.note_changed("note-1") == ()
            await second.settle()
            held = second.snapshot().roots[0]
            assert (held.status, held.next_action) == (
                "needs_attention",
                "review_changes",
            )
            assert second.schedule_hint("root-1") is None
            assert "root-1" not in runtime._leases
        finally:
            runtime._maintenance_resume()
        # Admission reopened: the same signal now heals the folder.
        assert await second.note_changed("note-1") == ("root-1",)
        await second.settle()
        assert second.snapshot().roots[0].status == "up_to_date"
    finally:
        await second.shutdown()


async def test_a_planner_shaped_hold_with_an_open_operation_is_refused(
    vault: Vault,
) -> None:
    """Review 2 deferred Minor 4: the incomplete-operation gate on its own.
    The status is the planner's shape (no ``action_id``), only the journal
    says Recovery owns the root."""

    owner = build_owner(vault)
    await owner.start()
    try:
        tombstone_version = await _delete(vault)
        assert await owner.note_changed("note-1") == ("root-1",)
        await owner.settle()
        held = owner.snapshot().roots[0]
        assert (held.status, held.next_action, held.action_id) == (
            "needs_attention",
            "review_changes",
            None,
        )
        store = NotesDeviceStateStore(vault.state_path)
        try:
            store.create_operation(
                NotesSyncOperationRecord(
                    operation_id="operation-open",
                    root_id="root-1",
                    binding_id=None,
                    kind="create_file",
                    state=NotesSyncOperationState.PENDING,
                    reason_code=None,
                    observation_token="review-open",
                    expected_note_version=None,
                    expected_file_digest=None,
                )
            )
        finally:
            store.close()

        await _restore(vault, tombstone_version)
        assert await owner.note_changed("note-1") == ()
        await owner.settle()
        after = owner.snapshot().roots[0]
        assert (after.status, after.next_action) == ("needs_attention", "review_changes")
        # Still blocked: a hint is refused for the held root.
        assert owner.schedule_hint("root-1") is None
        assert vault.file.read_bytes() == VAULT_BYTES
    finally:
        await owner.shutdown()


async def test_a_startup_hold_that_is_not_the_planners_is_never_released(
    vault: Vault,
) -> None:
    """Negative control: ``activation_recovery_required`` is Review settings'."""

    _persist_root_status(vault, "activation_recovery_required")
    owner = build_owner(vault)
    await owner.start()
    try:
        before = owner.snapshot().roots[0]
        assert (before.status, before.next_action) == (
            "needs_attention",
            "review_settings",
        )
        vault.edit_note(VAULT_TEXT + "typed anyway")
        assert await owner.note_changed("note-1") == ()
        await owner.settle()
        after = owner.snapshot().roots[0]
        assert (after.status, after.next_action) == (
            "needs_attention",
            "review_settings",
        )
        assert owner.schedule_hint("root-1") is None
        assert vault.file.read_bytes() == VAULT_BYTES
    finally:
        await owner.shutdown()


async def test_a_durable_hold_with_an_open_operation_is_never_released(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: Recovery's entry survives a restart and owns the hold."""

    first = build_owner(vault)
    await first.start()
    try:
        operation_id = await wedge_root(vault, first, monkeypatch)
    finally:
        await first.shutdown()

    second = build_owner(vault)
    await second.start()
    try:
        before = second.snapshot().roots[0]
        assert before.status == "needs_attention"
        assert before.action_id == operation_id
        assert await second.note_changed("note-1") == ()
        await second.settle()
        after = second.snapshot().roots[0]
        assert (after.status, after.action_id) == ("needs_attention", operation_id)
        assert vault.incomplete() == [
            ("update_file", "needs_attention", "postcondition_failed")
        ]
    finally:
        await second.shutdown()


# --- Fix round 3 (review 3): hints during the released pass, refusals, failures


OTHER_TEXT = "> A second folder, kept healthy so the watcher is already running.\n"


def _add_second_root(vault: Vault, tmp_path: Path) -> Path:
    """A second bound note/file pair in its own root, like ``Vault`` builds one."""

    root = tmp_path / "vault2"
    root.mkdir()
    file = root / "other.md"
    file.write_bytes(OTHER_TEXT.encode("utf-8"))
    assert vault.database.add_note("other", OTHER_TEXT, "note-2") == "note-2"
    folders = LocalNoteFolderRepository(vault.database)
    folders.create_folder(name="VSync2", parent_id=None, folder_id="folder-2")
    folders.reconcile_managed(owner_id="root-2", desired=(("folder-2", "note-2"),))
    with PosixNotesSyncFilesystem(root) as filesystem:
        baseline = filesystem.observe("other.md")
    note = vault.database.get_note_by_id("note-2")
    store = NotesDeviceStateStore(vault.state_path)
    try:
        store.create_root(
            NotesSyncRootRecord(
                root_id="root-2",
                note_scope_id="local_note",
                logical_folder_id="folder-2",
                canonical_path=str(root.resolve()),
                direction=NotesSyncDirection.BIDIRECTIONAL,
                state=NotesSyncRootState.ACTIVE,
            )
        )
        store.create_binding(
            NotesSyncBindingRecord(
                binding_id="binding-2",
                root_id="root-2",
                note_scope_id="local_note",
                note_id="note-2",
                normalized_relative_path="other.md",
                stable_identity_digest=NotesSyncExecutor.stable_identity_digest(
                    baseline
                ),
                state=NotesSyncBindingState.ACTIVE,
                serialization=baseline.observation.serialization,
                content_digest=hashlib.sha256(OTHER_TEXT.encode("utf-8")).hexdigest(),
                note_version=int(note["version"]),
            )
        )
    finally:
        store.close()
    return file


class _ObservationGate:
    """Hold the released pass right after its first observation of root-1."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.observed = asyncio.Event()
        self.release = asyncio.Event()
        self.count = 0
        real = _ProductionRuntimeAdapter.observe_root
        gate = self

        async def gated(adapter, root):
            result = await real(adapter, root)
            if root.root_id == "root-1":
                gate.count += 1
                if gate.count == 1:
                    gate.observed.set()
                    await gate.release.wait()
            return result

        monkeypatch.setattr(_ProductionRuntimeAdapter, "observe_root", gated)


LATE_SAVE = VAULT_TEXT + "saved while the released pass ran"


@pytest.mark.parametrize("second_folder", [True, False], ids=["two-folders", "one-folder"])
async def test_a_save_during_the_released_pass_is_run_after_it(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, second_folder: bool
) -> None:
    """Review 3 Important 1 (two folders: the hint coalesced onto the released
    pass must be re-run, not discarded) and Minor 3 (one folder: no watcher
    runs during the pass, so the signal must mark the root dirty itself). In
    both shapes the save the runtime acknowledged has to reach the file."""

    if second_folder:
        _add_second_root(vault, tmp_path)
    tombstone_version = await _held_in_a_later_session(vault)
    gate = _ObservationGate(monkeypatch)
    second = build_owner(vault)
    await second.start()
    try:
        if second_folder:
            assert second.schedule_hint("root-2") is not None, "the watcher runs for root-2"
            await second.settle()
        await _restore(vault, tombstone_version)
        assert await second.note_changed("note-1") == ("root-1",)
        await asyncio.wait_for(gate.observed.wait(), 10)

        # The pass has observed the folder; a save lands now and is signalled.
        vault.edit_note(LATE_SAVE)
        try:
            assert await second.note_changed("note-1") == ("root-1",)
        finally:
            gate.release.set()
        await second.settle()

        assert gate.count >= 2, "the folder was observed again for the late save"
        assert vault.file.read_bytes() == (LATE_SAVE + "\n").encode("utf-8")
        assert second.snapshot().roots[0].status == "up_to_date"
        assert vault.incomplete() == []
    finally:
        await second.shutdown()


def _runtime(owner) -> NotesSyncRuntimeOwner:
    assert isinstance(owner, NotesSyncRuntimeOwner)
    return owner


async def test_a_refused_schedule_puts_the_hold_back_so_a_later_signal_heals(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 3 Minor 1 (and the Minor 4 pin of the gate in
    ``_schedule_released_pass``): admission closes inside the release's own
    await; the schedule is refused, the signal reports nothing hinted, and the
    hold is back so the next signal after resume heals the folder."""

    tombstone_version = await _held_in_a_later_session(vault)
    second = build_owner(vault)
    await second.start()
    real = NotesSyncRuntimeOwner._release_planner_hold

    async def closing_after_release(runtime, root_id):
        released = await real(runtime, root_id)
        if released:
            runtime._maintenance_close_admission()
        return released

    try:
        await _restore(vault, tombstone_version)
        monkeypatch.setattr(
            NotesSyncRuntimeOwner, "_release_planner_hold", closing_after_release
        )
        assert await second.note_changed("note-1") == ()
        await second.settle()
        runtime = _runtime(second)
        assert "root-1" in runtime._blocked_roots
        assert "root-1" not in runtime._leases
        held = second.snapshot().roots[0]
        assert (held.status, held.next_action) == ("needs_attention", "review_changes")

        monkeypatch.setattr(NotesSyncRuntimeOwner, "_release_planner_hold", real)
        runtime._maintenance_resume()
        assert await second.note_changed("note-1") == ("root-1",)
        await second.settle()
        assert second.snapshot().roots[0].status == "up_to_date"
    finally:
        await second.shutdown()


async def test_a_fence_that_closes_before_the_admitted_pass_runs_puts_the_hold_back(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 3 Minor 4: pins the ``operation()`` wrap and the hold put-back.
    Admission closes after the task is created and before it runs: the fence
    refuses the pass (without the wrap it would run and publish up to date),
    and the hold is back (without the put-back the root would be unblocked,
    unleased and unreachable by any later signal)."""

    tombstone_version = await _held_in_a_later_session(vault)
    second = build_owner(vault)
    await second.start()
    real = NotesSyncRuntimeOwner._schedule_released_pass

    def closing_after_schedule(runtime, root_id):
        task = real(runtime, root_id)
        if task is not None:
            runtime._maintenance_close_admission()
        return task

    try:
        await _restore(vault, tombstone_version)
        monkeypatch.setattr(
            NotesSyncRuntimeOwner, "_schedule_released_pass", closing_after_schedule
        )
        assert await second.note_changed("note-1") == ("root-1",)
        await second.settle()
        runtime = _runtime(second)
        assert "root-1" in runtime._blocked_roots
        assert "root-1" not in runtime._leases
        held = second.snapshot().roots[0]
        assert (held.status, held.next_action) == ("needs_attention", "review_changes")

        monkeypatch.setattr(NotesSyncRuntimeOwner, "_schedule_released_pass", real)
        runtime._maintenance_resume()
        assert await second.note_changed("note-1") == ("root-1",)
        await second.settle()
        assert second.snapshot().roots[0].status == "up_to_date"
    finally:
        await second.shutdown()


async def test_a_failure_inside_the_released_pass_blocks_the_root_and_publishes_failed(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 3 Minor 2: a raw store error is handled like ``_run_hint`` handles
    one -- the root is blocked and ``failed`` is published -- instead of
    becoming an unretrieved task exception with the hold silently cleared."""

    tombstone_version = await _held_in_a_later_session(vault)
    second = build_owner(vault)
    await second.start()
    real = NotesDeviceStateStore.get_root
    failures = {"left": 1}

    def failing_get_root(store, root_id):
        if root_id == "root-1" and failures["left"]:
            failures["left"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real(store, root_id)

    try:
        await _restore(vault, tombstone_version)
        monkeypatch.setattr(NotesDeviceStateStore, "get_root", failing_get_root)
        assert await second.note_changed("note-1") == ("root-1",)
        await second.settle()
        runtime = _runtime(second)
        failed = second.snapshot().roots[0]
        assert (failed.status, failed.next_action) == ("failed", "review_changes")
        assert "root-1" in runtime._blocked_roots
        assert failures["left"] == 0
        # A failed pass is not a planner hold: the next signal does not pretend.
        assert await second.note_changed("note-1") == ()
        assert vault.file.read_bytes() == VAULT_BYTES
    finally:
        await second.shutdown()
