"""An update fenced only because its note moved on settles on its own (TASK-34000.51).

Lasting sync requires a note to stay unchanged from the moment a pass admits
its ``update_file`` until that write completes. A second save of the same note
inside that window used to fence the folder -- ``stale_observation`` before
admission (a planner-shaped hold with nothing to review) or
``postcondition_failed`` after the write (a Recovery entry) -- and the folder
then synced in neither direction until the user pressed Recovery. The click
added no decision: the settle Recovery performs commits only a baseline proven
against the journal digest and the bytes on disk, and the next pass writes the
newer note. The runtime now performs that settle itself and re-plans, bounded.

A genuine two-sided change (the file also changed) is still a review, and no
side ever wins: the two negative controls below pin it at both fence points and
during the settle itself.

Everything here is the production stack: the production runtime owner, the
real executor and POSIX filesystem, a real ChaChaNotes database and a real
``.md`` in a temp vault. The gates only delay a pass at a named executor point;
they replace nothing. Assertions read the database row, the bytes on disk and
the sync store rows.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest

from Tests.Notes.notes_sync_tail_edit_support import VAULT_TEXT, Vault, build_owner
from tldw_chatbook.Library.library_notes_session import (
    DatabaseNoteSessionCoordinator,
    NoteLoadOutcomeKind,
    NoteSaveOutcomeKind,
)
from tldw_chatbook.Notes import notes_sync_runtime
from tldw_chatbook.Notes.note_folder_repository import LocalNoteFolderRepository
from tldw_chatbook.Notes.notes_device_state_store import (
    NotesDeviceStateStore,
    NotesSyncBindingRecord,
    NotesSyncRootRecord,
)
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor
from tldw_chatbook.Notes.notes_sync_filesystem import PosixNotesSyncFilesystem
from tldw_chatbook.Notes.notes_sync_models import (
    NotesSyncBindingState,
    NotesSyncDirection,
    NotesSyncRootState,
)
from tldw_chatbook.Notes.notes_sync_runtime import NotesSyncRuntimeOwner
from tldw_chatbook.UI.Library_Modules import library_notes_sync_attention as attention
from tldw_chatbook.UI.Library_Modules.note_session_port import (
    _LibraryDatabaseNoteSessionPort,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: What the max-wait autosave commits while the user is still typing...
FIRST_SAVE = VAULT_TEXT + "typed up to the max wait"
#: ...and the keys that landed during it, committed inside the first save's pass.
SECOND_SAVE = FIRST_SAVE + " and one more key"
#: An edit made on disk by hand: the other side of a genuine conflict.
DISK_EDIT = VAULT_TEXT + "| edited on disk while sync was working |\n"
#: The second folder of the AC#6 control.
OTHER_TEXT = "> A second folder, whose pass is held open the whole time.\n"

#: Statuses under which a folder is not syncing until the user acts.
HELD = frozenset({"failed", "needs_attention", "partial"})


def _file_bytes(note_text: str) -> bytes:
    """The vault file for ``note_text`` under its final-newline profile."""

    return (note_text if note_text.endswith("\n") else note_text + "\n").encode("utf-8")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def vault(tmp_path: Path):
    selected = Vault(tmp_path)
    try:
        yield selected
    finally:
        selected.close()


class _Gate:
    """Hold one root's first automatic ``update_file`` at a named executor point.

    ``execute``: before admission -- the pass has observed the note and built
    its request, no journal row exists yet. ``_require_desired``: right after
    the write -- the first save's bytes are on disk and the postcondition has
    not been checked yet. Only the first call for the root is held.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, point: str, *, root_id: str = "root-1"
    ) -> None:
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self.count = 0
        real = getattr(NotesSyncExecutor, point)
        gate = self

        async def gated(executor, request):
            if request.root_id == root_id:
                gate.count += 1
                if gate.count == 1:
                    gate.reached.set()
                    await gate.release.wait()
            return await real(executor, request)

        monkeypatch.setattr(NotesSyncExecutor, point, gated)


def _recorded(owner: NotesSyncRuntimeOwner) -> list[tuple[str, str, str]]:
    """Every status the runtime publishes from now on, in order."""

    published: list[tuple[str, str, str]] = []
    owner.add_status_listener(
        lambda snapshot: published.append(
            (snapshot.root_id, snapshot.status, snapshot.next_action)
        )
    )
    return published


def _root(owner: NotesSyncRuntimeOwner, root_id: str = "root-1"):
    return next(root for root in owner.snapshot().roots if root.root_id == root_id)


def _binding(vault: Vault, binding_id: str = "binding-1") -> NotesSyncBindingRecord:
    store = NotesDeviceStateStore(vault.state_path)
    try:
        return store.get_binding(binding_id)
    finally:
        store.close()


async def _second_save(vault: Vault, owner: NotesSyncRuntimeOwner) -> None:
    """The keys that landed during the first save, committed and signalled."""

    vault.edit_note(SECOND_SAVE)
    assert await owner.note_changed("note-1") == ("root-1",)


async def _fence(
    vault: Vault,
    owner: NotesSyncRuntimeOwner,
    gate: _Gate,
    *,
    at_gate: Callable[[], Awaitable[None]],
) -> None:
    """First save, its pass held at the gate, ``at_gate`` inside the window, then settle."""

    vault.edit_note(FIRST_SAVE)
    assert await owner.note_changed("note-1") == ("root-1",)
    await asyncio.wait_for(gate.reached.wait(), 10)
    await at_gate()
    gate.release.set()
    await asyncio.wait_for(owner.settle(), 30)


def _healthy_at(vault: Vault, owner: NotesSyncRuntimeOwner, note_text: str) -> None:
    """No open entry, no hold, nothing to click, and the file equals the note."""

    root = _root(owner)
    assert vault.incomplete() == [], "an operation was left open for Recovery"
    assert (root.status, root.next_action, root.action_id) == (
        "up_to_date",
        "sync_now",
        None,
    ), "the folder was left held"
    assert vault.note()["content"] == note_text
    assert vault.file.read_bytes() == _file_bytes(note_text)


# --- AC#2 / AC#4: a save inside the pass settles on the next pass ---------------


@pytest.mark.parametrize(
    "point",
    ["_require_desired", "execute"],
    ids=["after_write", "at_admission"],
)
async def test_a_save_during_the_write_settles_on_its_own(
    vault: Vault, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    gate = _Gate(monkeypatch, point)
    owner = build_owner(vault)
    await owner.start()
    published = _recorded(owner)
    try:
        await _fence(vault, owner, gate, at_gate=lambda: _second_save(vault, owner))

        _healthy_at(vault, owner, SECOND_SAVE)
        binding = _binding(vault)
        assert binding.content_digest == _digest(_file_bytes(SECOND_SAVE)), (
            "the binding baseline is not the second save's"
        )
        assert binding.note_version == int(vault.note()["version"])
        held = [entry for entry in published if entry[1] in HELD]
        assert held == [], f"the save held the folder: {held}"
    finally:
        gate.release.set()
        await owner.shutdown()


async def test_a_source_moved_entry_persisted_across_a_restart_is_settled_at_startup(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restart in the middle: the runtime's own settle is refused once (the
    store's compare-and-set raced), the entry stays open across a shutdown, and
    the next launch settles it before leasing the root -- no Recovery click."""

    gate = _Gate(monkeypatch, "_require_desired")
    refused: list[str] = []
    real_settle = NotesSyncExecutor.settle_attention

    async def refusing_settle(executor, operation_id):
        if not refused:
            refused.append(operation_id)
            raise RuntimeError("stale_observation")
        return await real_settle(executor, operation_id)

    with monkeypatch.context() as patch:
        patch.setattr(NotesSyncExecutor, "settle_attention", refusing_settle)
        owner = build_owner(vault)
        await owner.start()
        try:
            await _fence(vault, owner, gate, at_gate=lambda: _second_save(vault, owner))
            # Held for Recovery, exactly as the refusal leaves it; the first
            # save's bytes are on disk and the reason names the cause.
            assert vault.incomplete() == [
                ("update_file", "needs_attention", "source_moved_on")
            ]
            root = _root(owner)
            assert (root.status, root.next_action) == (
                "needs_attention",
                "resolve_cleanup",
            )
            assert root.action_id is not None
            assert vault.file.read_bytes() == _file_bytes(FIRST_SAVE)
            assert refused == [root.action_id]
        finally:
            gate.release.set()
            await owner.shutdown()

    relaunched = build_owner(vault)
    await relaunched.start()
    published = _recorded(relaunched)
    try:
        await asyncio.wait_for(relaunched.settle(), 30)

        _healthy_at(vault, relaunched, SECOND_SAVE)
        assert [entry for entry in published if entry[1] in HELD] == []
        # And the root is watched again (TASK-34000.50): a disk edit flows.
        appended = SECOND_SAVE + "\nafter the restart\n"
        vault.file.write_bytes(appended.encode("utf-8"))
        assert relaunched.schedule_hint("root-1") is not None
        await asyncio.wait_for(relaunched.settle(), 30)
        assert vault.note()["content"] == appended
        assert vault.incomplete() == []
    finally:
        await relaunched.shutdown()


async def test_the_re_plan_is_bounded_and_the_hint_loop_carries_the_rest(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuous typing: the note moves on at EVERY write, one more time than
    one pass may re-plan. The pass never loops forever: at the bound it leaves
    the root unblocked with the dirty mark set, the hint loop carries the rest,
    and the file ends up equal to the last save with nothing held."""

    bound = notes_sync_runtime._SOURCE_MOVED_REPLANS
    moves = bound + 2
    moved: list[str] = []
    seen: set[str] = set()
    executes = {"count": 0}
    per_reconcile: list[int] = []
    real_desired = NotesSyncExecutor._require_desired
    real_execute = NotesSyncExecutor.execute
    real_reconcile = NotesSyncRuntimeOwner._reconcile_locked
    owner = build_owner(vault)

    async def moving(executor, request):
        # The first postcondition of each new write: the user typed meanwhile.
        if request.root_id == "root-1" and request.operation_id not in seen:
            seen.add(request.operation_id)
            if len(moved) < moves:
                text = f"{FIRST_SAVE} + key {len(moved) + 1}"
                moved.append(text)
                vault.edit_note(text)
                await owner.note_changed("note-1")
        return await real_desired(executor, request)

    async def counted_execute(executor, request):
        executes["count"] += 1
        return await real_execute(executor, request)

    async def counted_reconcile(runtime, root, *, automatic):
        before = executes["count"]
        try:
            return await real_reconcile(runtime, root, automatic=automatic)
        finally:
            per_reconcile.append(executes["count"] - before)

    monkeypatch.setattr(NotesSyncExecutor, "_require_desired", moving)
    monkeypatch.setattr(NotesSyncExecutor, "execute", counted_execute)
    monkeypatch.setattr(NotesSyncRuntimeOwner, "_reconcile_locked", counted_reconcile)
    await owner.start()
    published = _recorded(owner)
    try:
        vault.edit_note(FIRST_SAVE)
        assert await owner.note_changed("note-1") == ("root-1",)
        await asyncio.wait_for(owner.settle(), 60)

        assert len(moved) == moves
        _healthy_at(vault, owner, moved[-1])
        held = [entry for entry in published if entry[1] in HELD]
        assert held == [], f"typing held the folder: {held}"
        # One write per move plus the one that finally completed...
        assert executes["count"] == moves + 1
        # ...no single locked pass ran more than the bound allows...
        assert max(per_reconcile) <= bound + 1, per_reconcile
        # ...and the remainder was carried by a later pass, not by looping.
        assert len([count for count in per_reconcile if count]) >= 2, per_reconcile
    finally:
        await owner.shutdown()


# --- AC#3: a genuine two-sided change still stops for review -------------------


@pytest.mark.parametrize(
    "point, entry",
    [("execute", None), ("_require_desired", "postcondition_failed")],
    ids=["at_admission", "after_write"],
)
async def test_a_file_edited_on_disk_during_the_pass_still_stops_for_review(
    vault: Vault, monkeypatch: pytest.MonkeyPatch, point: str, entry: str | None
) -> None:
    """Negative control: the note moved on AND the file changed. The fence
    keeps today's reason (never ``source_moved_on``), nothing is written over
    either side, and the plan after Recovery is the ordinary conflict review."""

    gate = _Gate(monkeypatch, point)
    owner = build_owner(vault)
    await owner.start()

    async def both_sides() -> None:
        vault.file.write_bytes(DISK_EDIT.encode("utf-8"))
        await _second_save(vault, owner)

    try:
        await _fence(vault, owner, gate, at_gate=both_sides)

        root = _root(owner)
        if entry is None:
            # Before admission: the planner's own hold, no entry to settle.
            assert vault.incomplete() == []
            assert (root.status, root.next_action, root.action_id) == (
                "needs_attention",
                "review_changes",
                None,
            )
        else:
            assert vault.incomplete() == [("update_file", "needs_attention", entry)]
            assert (root.status, root.next_action) == (
                "needs_attention",
                "resolve_cleanup",
            )
            assert root.action_id is not None
        assert vault.note()["content"] == SECOND_SAVE
        assert vault.file.read_bytes() == DISK_EDIT.encode("utf-8")

        if entry is not None:
            await owner.resolve_cleanup("root-1", root.action_id)
            assert vault.incomplete() == []
            root = _root(owner)
            assert (root.status, root.next_action) == (
                "needs_attention",
                "review_changes",
            )
        plan = await owner.check_root("root-1")
        assert [(item.kind.value, item.reason_code) for item in plan.attention] == [
            ("conflict", "both_sides_changed")
        ]
        assert vault.note()["content"] == SECOND_SAVE
        assert vault.file.read_bytes() == DISK_EDIT.encode("utf-8")
    finally:
        gate.release.set()
        await owner.shutdown()


async def test_a_file_edited_on_disk_during_the_settle_is_not_overwritten(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control on the settle itself: the fence is the one-sided one
    (the note moved on, the write is on disk), but a hand edit lands before the
    runtime's settle reads the file. The settle cannot prove the write any more,
    keeps the reviewed baseline, and the re-plan is a conflict: no write."""

    gate = _Gate(monkeypatch, "_require_desired")
    settled: list[str] = []
    real_settle = NotesSyncExecutor.settle_attention

    async def edit_then_settle(executor, operation_id):
        settled.append(operation_id)
        vault.file.write_bytes(DISK_EDIT.encode("utf-8"))
        return await real_settle(executor, operation_id)

    monkeypatch.setattr(NotesSyncExecutor, "settle_attention", edit_then_settle)
    original_version = int(vault.note()["version"])
    owner = build_owner(vault)
    await owner.start()
    try:
        await _fence(vault, owner, gate, at_gate=lambda: _second_save(vault, owner))

        assert len(settled) == 1, "the runtime did not settle the fenced entry"
        assert vault.incomplete() == []
        root = _root(owner)
        assert (root.status, root.next_action, root.action_id) == (
            "needs_attention",
            "review_changes",
            None,
        )
        assert vault.note()["content"] == SECOND_SAVE
        assert vault.file.read_bytes() == DISK_EDIT.encode("utf-8")
        binding = _binding(vault)
        assert binding.content_digest == _digest(VAULT_TEXT.encode("utf-8")), (
            "the settle committed a baseline it could not prove"
        )
        assert binding.note_version == original_version
        plan = await owner.check_root("root-1")
        assert [(item.kind.value, item.reason_code) for item in plan.attention] == [
            ("conflict", "both_sides_changed")
        ]
    finally:
        gate.release.set()
        await owner.shutdown()


# --- AC#6: a save never waits for another folder's pass -------------------------


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
    assert note is not None
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
                content_digest=_digest(OTHER_TEXT.encode("utf-8")),
                note_version=int(note["version"]),
            )
        )
    finally:
        store.close()
    return file


def _port(vault: Vault, owner) -> _LibraryDatabaseNoteSessionPort:
    """The shipped Library save seam over the vault's real Notes authorities."""

    return _LibraryDatabaseNoteSessionPort(
        run_service_call=LibraryScreen._run_library_service_call,
        notes_scope_service=vault.scope_service,
        notes_service=vault.interop,
        user_id="user-1",
        clock=lambda: datetime.now(timezone.utc),
        notes_sync_runtime=lambda: owner,
    )


async def _session(vault: Vault, owner, note_id: str = "note-1"):
    session = DatabaseNoteSessionCoordinator(
        _port(vault, owner), clock=lambda: datetime.now(timezone.utc)
    )
    opened = await session.open_session(note_id)
    assert opened.kind is NoteLoadOutcomeKind.LOADED, opened
    return session


async def _until(predicate: Callable[[], bool], what: str, timeout: float = 10) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        await asyncio.sleep(0.05)


async def test_a_save_in_one_folder_never_waits_for_another_folders_pass(
    vault: Vault, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review focus "a pass on another folder". Root-2's pass is held open the
    whole time; a re-save of root-1's note through the shipped session and
    port returns at once and its file is written, without the port ever
    joining the runtime's work. On the mitigation this save waited the full
    bound (patched to 2 s) for root-2's pass."""

    monkeypatch.setattr(attention, "RESAVE_SYNC_PASS_WAIT_SECONDS", 2.0, raising=False)
    other_file = _add_second_root(vault, tmp_path)
    gate = _Gate(monkeypatch, "execute", root_id="root-2")
    owner = build_owner(vault)
    await owner.start()
    settles: list[float] = []
    real_settle = owner.settle

    async def counted_settle():
        settles.append(time.monotonic())
        await real_settle()

    monkeypatch.setattr(owner, "settle", counted_settle)
    other_text = OTHER_TEXT + "edited in the other folder"
    try:
        other = vault.database.get_note_by_id("note-2")
        assert other is not None and vault.database.update_note(
            "note-2",
            {"title": other["title"], "content": other_text},
            int(other["version"]),
        )
        assert await owner.note_changed("note-2") == ("root-2",)
        await asyncio.wait_for(gate.reached.wait(), 10)  # root-2's pass is held

        session = await _session(vault, owner)
        assert session.mutate(body=FIRST_SAVE)
        assert (await session.request_save(explicit=False)).kind is (
            NoteSaveOutcomeKind.SAVED
        )
        assert session.mutate(body=SECOND_SAVE)
        started = time.monotonic()
        outcome = await asyncio.wait_for(session.request_save(explicit=False), 5)
        took = time.monotonic() - started

        assert outcome.kind is NoteSaveOutcomeKind.SAVED
        assert not gate.release.is_set(), "the control needs root-2's pass still held"
        assert took < 1.0, f"the save waited {took:.2f}s for another folder's pass"
        assert settles == [], "the save joined the runtime's work"
        await _until(
            lambda: vault.file.read_bytes() == _file_bytes(SECOND_SAVE),
            "root-1's file while root-2's pass is still held",
        )
        assert not gate.release.is_set()

        gate.release.set()
        await asyncio.wait_for(real_settle(), 30)
        _healthy_at(vault, owner, SECOND_SAVE)
        second = _root(owner, "root-2")
        assert (second.status, second.next_action) == ("up_to_date", "sync_now")
        assert other_file.read_bytes() == _file_bytes(other_text)
    finally:
        gate.release.set()
        await owner.shutdown()
