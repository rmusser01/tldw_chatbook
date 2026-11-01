"""TASK-34000.50 (TASK-34000 wave 1c, Task 2; review finding N-03 follow-up).

A folder held at startup is published as a durable hold without a lease and
without a watcher. Healing it through Review (``check_root``) or Recovery
(``resolve_cleanup``) leased the root and re-planned it, and the row turned to
"✓ Up to date as of HH:MM" -- while no watcher ran: a later disk edit was never
seen and a note edit was never carried. Only Check changes started one.

These tests run the production runtime owner, executor and POSIX filesystem
over a real ``CharactersRAGDB`` and a real ``.md`` in a temp vault, with a
restart in the middle. The evidence is the note row and the bytes on disk
after the heal, in both directions, with no extra action.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from pathlib import Path

import pytest

from Tests.Notes.notes_sync_tail_edit_support import (
    TAIL_EDIT,
    VAULT_TEXT,
    Vault,
    build_owner,
    wedge_root,
)
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncRootRecord,
)
from tldw_chatbook.Notes.notes_sync_coordinator import NotesSyncRootCoordinator
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]

VAULT_BYTES = VAULT_TEXT.encode("utf-8")


@pytest.fixture
def vault(tmp_path: Path):
    selected = Vault(tmp_path)
    try:
        yield selected
    finally:
        selected.close()


def _runtime(owner) -> NotesSyncRuntimeOwner:
    assert isinstance(owner, NotesSyncRuntimeOwner)
    return owner


def _tombstone(vault: Vault) -> dict:
    return vault.database.get_note_version_states(["note-1"])["note-1"]


async def _delete(vault: Vault) -> int:
    """Soft-delete note-1 through the seam the Library's Delete uses."""

    version = int(vault.note()["version"])
    assert await vault.scope_service.delete_note(
        scope="local_note", note_id="note-1", version=version, user_id="user-1"
    )
    state = _tombstone(vault)
    assert state["deleted"] is True
    return int(state["version"])


async def _restore(vault: Vault, tombstone_version: int) -> None:
    """Restore note-1 at the DB level only -- no signal to the runtime."""

    record = await vault.scope_service.restore_note(
        scope="local_note",
        note_id="note-1",
        version=tombstone_version,
        user_id="user-1",
    )
    assert record["id"] == "note-1"
    assert _tombstone(vault)["deleted"] is False


async def _held_by_a_missing_file(vault: Vault) -> None:
    """Session 1: the file vanishes, the watcher's pass holds the root; quit.

    The identical bytes are put back while the runtime is down, so the next
    session starts with a persisted ``needs_attention`` over a clean folder.
    """

    first = build_owner(vault)
    await first.start()
    try:
        vault.file.unlink()
        assert first.schedule_hint("root-1") is not None
        await first.settle()
        held = first.snapshot().roots[0]
        assert (held.status, held.next_action, held.action_id) == (
            "needs_attention",
            "review_changes",
            None,
        )
        plan = await first.check_root("root-1")
        assert [item.reason_code for item in plan.attention] == ["file_missing"]
    finally:
        await first.shutdown()
    vault.file.write_bytes(VAULT_BYTES)


async def _held_by_a_deleted_note(vault: Vault) -> int:
    """Session 1: the note is deleted and signalled; quit. Returns the tombstone."""

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


def _assert_startup_hold(owner, *, action_id: str | None = None) -> None:
    """The root is held since startup and no hint can reach it.

    A planner hold (no ``action_id``) is re-published without a lease. A
    Recovery hold IS leased at startup -- ``_resume_incomplete`` leases the
    root to try the entry -- and the startup watcher runs for that lease, so
    only the block keeps hints out; the heal must leave it watched either way.
    """

    held = owner.snapshot().roots[0]
    assert held.status == "needs_attention"
    assert held.action_id == action_id
    assert owner.schedule_hint("root-1") is None
    if action_id is None:
        assert "root-1" not in _runtime(owner)._leases


async def _assert_both_directions_flow(vault: Vault, owner, *, note_text: str) -> None:
    """The heal left the folder syncing both ways, with no further action.

    ``note_text`` is the note's content right after the heal; the file holds
    it under the binding's final-newline profile.
    """

    runtime = _runtime(owner)
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("up_to_date", "sync_now")
    # The lease is held AND a watcher runs for it: hints are admitted.
    assert "root-1" in runtime._leases
    assert owner.schedule_hint("root-1") is not None, (
        "the heal leased the root but started no watcher"
    )
    assert root.watching is True
    await owner.settle()

    # Disk -> note: an edit on disk reaches the note on the watcher's pass.
    appended = note_text + ("" if note_text.endswith("\n") else "\n") + "from the vault\n"
    vault.file.write_bytes(appended.encode("utf-8"))
    assert owner.schedule_hint("root-1") is not None
    await owner.settle()
    assert vault.note()["content"] == appended
    assert vault.incomplete() == []

    # Note -> file: an in-app edit is signalled and reaches the file.
    typed = appended + "typed in chatbook"
    vault.edit_note(typed)
    assert await owner.note_changed("note-1") == ("root-1",)
    await owner.settle()
    assert vault.file.read_bytes() == (typed + "\n").encode("utf-8")
    assert vault.incomplete() == []
    root = owner.snapshot().roots[0]
    assert (root.status, root.next_action) == ("up_to_date", "sync_now")
    assert root.watching is True


@pytest.mark.parametrize("hold", ["file_missing", "delete_restore"])
async def test_review_on_a_startup_held_root_leaves_the_watcher_running(
    vault: Vault, hold: str
) -> None:
    """AC#1/#3: Review heals the hold AND the folder keeps syncing both ways.

    ``file_missing`` is independent of TASK-34000.49. ``delete_restore``
    restores the note at the DB level in the later session WITHOUT a signal
    (the restore moved the note's version only), then Reviews; it is green
    only with TASK-34000.49's version-proxy removal in place.
    """

    if hold == "file_missing":
        await _held_by_a_missing_file(vault)
    else:
        tombstone_version = await _held_by_a_deleted_note(vault)
    second = build_owner(vault)
    await second.start()
    try:
        _assert_startup_hold(second)
        if hold == "delete_restore":
            await _restore(vault, tombstone_version)
        plan = await second.check_root("root-1")
        assert plan.attention == ()
        assert vault.note()["content"] == VAULT_TEXT
        assert vault.file.read_bytes() == VAULT_BYTES
        await _assert_both_directions_flow(vault, second, note_text=VAULT_TEXT)
    finally:
        await second.shutdown()


async def test_recovery_on_a_startup_held_root_leaves_the_watcher_running(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC#1: Recovery on an entry that survived a restart leaves it watched."""

    first = build_owner(vault)
    await first.start()
    try:
        operation_id = await wedge_root(vault, first, monkeypatch)
    finally:
        await first.shutdown()

    second = build_owner(vault)
    await second.start()
    try:
        _assert_startup_hold(second, action_id=operation_id)
        await second.resolve_cleanup("root-1", operation_id)
        assert vault.incomplete() == []
        # Neither side was rewritten by the heal.
        assert vault.note()["content"] == TAIL_EDIT
        assert vault.file.read_bytes() == (TAIL_EDIT + "\n").encode("utf-8")
        await _assert_both_directions_flow(vault, second, note_text=TAIL_EDIT)
    finally:
        await second.shutdown()


# --- The "watching" fact is read live, never frozen at publish time ---------


async def test_the_watching_fact_follows_the_watcher_not_the_publication(
    vault: Vault,
) -> None:
    """AC#2: a healthy status over a stopped watcher reads ``watching False``.

    A backup's maintenance fence stops the watcher without publishing
    anything; the snapshot still says so, and the restart is re-announced to
    listeners so a row that read "stopped" is re-projected without a user
    action.
    """

    owner = build_owner(vault)
    await owner.start()
    runtime = _runtime(owner)
    announced: list[tuple[str, bool]] = []
    try:
        healthy = owner.snapshot().roots[0]
        assert (healthy.status, healthy.watching) == ("up_to_date", True)

        owner.add_status_listener(
            lambda snapshot: announced.append((snapshot.status, snapshot.watching))
        )
        runtime._maintenance_close_admission()
        runtime._maintenance_quiesce_task = asyncio.create_task(
            runtime._maintenance_quiesce()
        )
        await runtime._maintenance_quiesce_task
        assert not runtime._watcher_running()
        fenced = owner.snapshot().roots[0]
        # The last PUBLISHED status is unchanged; the fact is read live.
        assert (fenced.status, fenced.published_at) == (
            healthy.status,
            healthy.published_at,
        )
        assert fenced.watching is False
        assert announced == []

        runtime._maintenance_resume()
        assert runtime._watcher_running()
        resumed = owner.snapshot().roots[0]
        assert (resumed.status, resumed.published_at, resumed.watching) == (
            healthy.status,
            healthy.published_at,
            True,
        )
        # The restart re-announced the root with the live fact.
        assert announced == [("up_to_date", True)]
        # Fix round 1 (review Minor 2): the tree's folder-level read of the
        # same fact -- empty while watched, the root's folder while not.
        assert await owner.unwatched_folder_ids() == frozenset()
        task = runtime._watcher_task
        assert task is not None
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert owner.snapshot().status == "active"
        assert await owner.unwatched_folder_ids() == frozenset({"folder-1"})
        assert await owner.attention_folder_ids() == frozenset()
        runtime._start_watcher()
        assert await owner.unwatched_folder_ids() == frozenset()
    finally:
        await owner.shutdown()


# --- Races: shutdown or Pause while a heal is leasing the root ---------------


class _AcquireGate:
    """Hold the heal's coordinator acquire (on its worker thread) until released."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.calls = 0
        real = NotesSyncRootCoordinator.try_acquire
        gate = self

        def gated(coordinator, candidate, **validation):
            gate.calls += 1
            if gate.calls == 1:
                gate.entered.set()
                gate.proceed.wait(10)
            return real(coordinator, candidate, **validation)

        monkeypatch.setattr(NotesSyncRootCoordinator, "try_acquire", gated)


async def test_shutdown_during_a_heal_leaves_no_watcher_running(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lease that lands after shutdown began must not start a watcher."""

    await _held_by_a_missing_file(vault)
    second = build_owner(vault)
    await second.start()
    runtime = _runtime(second)
    assert not runtime._watcher_running(), "held at startup: nothing to watch"
    gate = _AcquireGate(monkeypatch)
    shutdown: asyncio.Task[None] | None = None
    try:
        heal = asyncio.create_task(second.check_root("root-1"))
        assert await asyncio.to_thread(gate.entered.wait, 10)
        shutdown = asyncio.create_task(second.shutdown())
        await asyncio.sleep(0.05)
        assert runtime._closing
        gate.proceed.set()
        try:
            await asyncio.wait_for(heal, 30)
        except RuntimeError:
            pass  # a heal refused by the closing runtime is fine; a watcher is not
        await asyncio.wait_for(shutdown, 30)
    finally:
        gate.proceed.set()
        if shutdown is None:
            await second.shutdown()

    assert runtime._watcher_task is None or runtime._watcher_task.done()
    assert not runtime._watcher_running()
    assert second.snapshot().status == "stopped"
    assert all(root.watching is False for root in second.snapshot().roots)


async def test_pause_during_a_heal_never_starts_the_watcher_for_the_closing_root(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root being closed by Pause gets no watcher from its late lease.

    Resume afterwards leases it again and that lease IS watched.
    """

    await _held_by_a_missing_file(vault)
    second = build_owner(vault)
    await second.start()
    runtime = _runtime(second)
    gate = _AcquireGate(monkeypatch)
    try:
        heal = asyncio.create_task(second.check_root("root-1"))
        assert await asyncio.to_thread(gate.entered.wait, 10)
        pause = asyncio.create_task(second.pause_root("root-1"))
        await asyncio.sleep(0.05)
        assert "root-1" in runtime._closed_roots
        gate.proceed.set()
        try:
            await asyncio.wait_for(heal, 30)
        except RuntimeError:
            pass
        result = await asyncio.wait_for(pause, 30)
        await asyncio.wait_for(second.settle(), 30)

        assert (result.status, result.next_action) == ("paused", "resume_sync")
        assert not runtime._watcher_running(), "a watcher was started for a paused root"
        assert "root-1" not in runtime._leases
        paused = second.snapshot().roots[0]
        assert (paused.status, paused.watching) == ("paused", False)

        resumed = await second.resume_root("root-1")
        assert (resumed.status, resumed.next_action) == ("up_to_date", "sync_now")
        assert second.schedule_hint("root-1") is not None
        await second.settle()
        assert second.snapshot().roots[0].watching is True
    finally:
        gate.proceed.set()
        await second.shutdown()


# --- Negative control: another root's heal never watches a Recovery hold ----


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


async def test_a_recovery_held_root_is_not_watched_by_another_roots_heal(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: Recovery's hold is Recovery's, whatever else is leased."""

    first = build_owner(vault)
    await first.start()
    try:
        operation_id = await wedge_root(vault, first, monkeypatch)
    finally:
        await first.shutdown()
    _add_second_root(vault, tmp_path)

    second = build_owner(vault)
    await second.start()
    try:
        by_id = {root.root_id: root for root in second.snapshot().roots}
        held, healthy = by_id["root-1"], by_id["root-2"]
        assert (held.status, held.next_action, held.action_id) == (
            "needs_attention",
            "resolve_cleanup",
            operation_id,
        )
        assert held.watching is False
        assert (healthy.status, healthy.watching) == ("up_to_date", True)

        await second.check_root("root-2")
        assert second.schedule_hint("root-2") is not None
        await second.settle()

        by_id = {root.root_id: root for root in second.snapshot().roots}
        held = by_id["root-1"]
        assert (held.status, held.next_action, held.action_id) == (
            "needs_attention",
            "resolve_cleanup",
            operation_id,
        )
        assert held.watching is False
        assert second.schedule_hint("root-1") is None
        assert await second.note_changed("note-1") == ()
        await second.settle()
        assert vault.incomplete() == [
            ("update_file", "needs_attention", "postcondition_failed")
        ]
        assert vault.file.read_bytes() == (TAIL_EDIT + "\n").encode("utf-8")
    finally:
        await second.shutdown()
