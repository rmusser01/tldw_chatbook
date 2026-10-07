"""Saves of a synced note and the sync pass they hint (final review I1 and I4).

I1. The sync executor requires a note to stay unchanged from the moment a pass
admits its ``update_file`` until that write completes. A second save that
commits inside that window fences the folder (``stale_observation`` or
``postcondition_failed``), and it then syncs in neither direction until the
user presses Recovery. TASK-34000.1 made that second save routine: an autosave
now fires while the user is still typing, and the note session re-saves at once
for keys that landed during it. The mitigation: a save of a note whose previous
save hinted a pass waits, bounded, for that pass before it commits. A note in
no synced folder never waits. TASK-34000.51 owns the durable runtime fix.

I4. Ctrl+Q flushed the note and exited without waiting for the pass that flush
hinted, so the file missed the last edit until the next launch. A clean quit
flush now waits, bounded, for that pass.

Everything here is the production stack: the shipped note session coordinator
and its Library port, the production sync runtime, executor and POSIX
filesystem, a real ChaChaNotes database and a real ``.md`` in a temp vault.
The gates below only delay a pass at a named point; they replace nothing.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.Notes.notes_sync_tail_edit_support import (
    TAIL_EDIT,
    VAULT_TEXT,
    Vault,
    build_owner,
)
from tldw_chatbook.Library.library_notes_session import (
    DatabaseNoteSessionCoordinator,
    NoteFlushOutcomeKind,
    NoteLoadOutcomeKind,
    NoteSaveOutcomeKind,
)
from tldw_chatbook.Notes.notes_sync_executor import NotesSyncExecutor
from tldw_chatbook.UI.Library_Modules import library_notes_sync_attention as attention
from tldw_chatbook.UI.Library_Modules import library_pending_work
from tldw_chatbook.UI.Library_Modules.note_session_port import (
    _LibraryDatabaseNoteSessionPort,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

pytestmark = [pytest.mark.unit, pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: What the max-wait autosave commits while the user is still typing...
FIRST_SAVE = VAULT_TEXT + "typed up to the max wait"
#: ...and the keys that landed during it, which the session re-saves at once.
SECOND_SAVE = FIRST_SAVE + " and one more key"


def _file_bytes(note_text: str) -> bytes:
    """The vault file for ``note_text`` under its final-newline profile."""

    return (note_text if note_text.endswith("\n") else note_text + "\n").encode("utf-8")


@pytest.fixture
def vault(tmp_path: Path):
    selected = Vault(tmp_path)
    try:
        yield selected
    finally:
        selected.close()


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


class _PassGate:
    """Hold root-1's first ``update_file`` just before the executor runs it.

    The pass has observed the note and built its request by then, which is the
    window the executor fences: a save that commits while the gate is closed
    is exactly the "second save inside the pass" of the finding.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reached = asyncio.Event()
        self.release = asyncio.Event()
        self.count = 0
        real = NotesSyncExecutor.execute
        gate = self

        async def gated(executor, request):
            gate.count += 1
            if gate.count == 1:
                gate.reached.set()
                await gate.release.wait()
            return await real(executor, request)

        monkeypatch.setattr(NotesSyncExecutor, "execute", gated)


def _healthy(vault: Vault, owner, note_text: str) -> None:
    """No open operation, no hold, and the file equals the note."""

    root = owner.snapshot().roots[0]
    assert vault.incomplete() == [], "a save left an operation open for Recovery"
    assert (root.status, root.next_action) == ("up_to_date", "sync_now"), (
        "a save left the synced folder held"
    )
    assert vault.note()["content"] == note_text
    assert vault.file.read_bytes() == _file_bytes(note_text)


# --- I1: a save waits for the pass the previous save hinted ----------------------


async def test_a_second_save_waits_for_the_pass_the_first_save_hinted(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _PassGate(monkeypatch)
    owner = build_owner(vault)
    await owner.start()
    try:
        session = await _session(vault, owner)
        assert session.mutate(body=FIRST_SAVE)
        first = await session.request_save(explicit=False)
        assert first.kind is NoteSaveOutcomeKind.SAVED
        await asyncio.wait_for(gate.reached.wait(), 10)  # pass 1 is mid-flight

        assert session.mutate(body=SECOND_SAVE)
        second = asyncio.create_task(session.request_save(explicit=False))
        await asyncio.sleep(0.3)
        committed_inside_the_pass = vault.note()["content"] != FIRST_SAVE
        snapshot = session.snapshot
        waiting_reads_as_saving = snapshot is not None and snapshot.saving
        gate.release.set()
        outcome = await asyncio.wait_for(second, 10)
        await owner.settle()

        assert outcome.kind is NoteSaveOutcomeKind.SAVED
        _healthy(vault, owner, SECOND_SAVE)
        assert not committed_inside_the_pass
        assert waiting_reads_as_saving, "the waiting save must read as in flight"
    finally:
        gate.release.set()
        await owner.shutdown()


async def test_keys_landing_during_a_save_are_resaved_after_its_pass(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session's own immediate re-save (``_drive_saves``), one save chain."""

    gate = _PassGate(monkeypatch)
    owner = build_owner(vault)
    await owner.start()
    try:
        session = await _session(vault, owner)
        port = session._port
        real_persist = port._persist_note
        saves_started: list[str] = []

        async def persist_then_type(note_id, expected_version, payload):
            saves_started.append(payload.body)
            if len(saves_started) == 2:
                # Natural timing puts this commit a few ms after the first
                # save's signal. Pin the bad case: pass 1 already holds the
                # first save's ``update_file`` when the re-save commits.
                await asyncio.wait_for(gate.reached.wait(), 10)
            reply = await real_persist(note_id, expected_version, payload)
            if len(saves_started) == 1:
                # A key lands while the first save is still in flight.
                assert session.mutate(body=SECOND_SAVE)
            return reply

        monkeypatch.setattr(port, "_persist_note", persist_then_type)
        assert session.mutate(body=FIRST_SAVE)
        chain = asyncio.create_task(session.request_save(explicit=False))
        await asyncio.wait_for(gate.reached.wait(), 10)  # pass 1 is mid-flight
        await asyncio.sleep(0.3)
        committed_inside_the_pass = vault.note()["content"] != FIRST_SAVE
        gate.release.set()
        outcome = await asyncio.wait_for(chain, 10)
        await owner.settle()

        assert outcome.kind is NoteSaveOutcomeKind.SAVED
        assert saves_started == [FIRST_SAVE, SECOND_SAVE]
        _healthy(vault, owner, SECOND_SAVE)
        assert not committed_inside_the_pass
    finally:
        gate.release.set()
        await owner.shutdown()


async def test_keys_typed_while_a_save_waits_are_saved_with_it(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wait must not make its own draft stale. A key typed while a save
    waits for the pass goes out with that save: one save, not a save of the
    older draft followed at once by another (which would repeat for as long
    as the user typed faster than a pass)."""

    gate = _PassGate(monkeypatch)
    owner = build_owner(vault)
    await owner.start()
    try:
        session = await _session(vault, owner)
        assert session.mutate(body=FIRST_SAVE)
        assert (await session.request_save(explicit=False)).kind is (
            NoteSaveOutcomeKind.SAVED
        )
        await asyncio.wait_for(gate.reached.wait(), 10)  # pass 1 is mid-flight
        version = int(vault.note()["version"])

        assert session.mutate(body=SECOND_SAVE)
        second = asyncio.create_task(session.request_save(explicit=False))
        await asyncio.sleep(0.2)  # the save is waiting for pass 1
        latest = SECOND_SAVE + ", then another"
        assert session.mutate(body=latest)
        gate.release.set()
        outcome = await asyncio.wait_for(second, 10)
        await owner.settle()

        assert outcome.kind is NoteSaveOutcomeKind.SAVED
        assert int(vault.note()["version"]) == version + 1, (
            "the save committed a draft its own wait had made stale"
        )
        _healthy(vault, owner, latest)
        assert not session.snapshot.dirty and not session.snapshot.saving
    finally:
        gate.release.set()
        await owner.shutdown()


@pytest.mark.parametrize("delay", [0.0, 0.025, 0.05, 0.1, 0.2])
async def test_a_second_save_shortly_after_the_first_leaves_no_hold(
    vault: Vault, delay: float
) -> None:
    """The reviewer's probe, ungated: natural timing, 0 to 200 ms apart."""

    owner = build_owner(vault)
    await owner.start()
    try:
        session = await _session(vault, owner)
        assert session.mutate(body=FIRST_SAVE)
        assert (await session.request_save(explicit=False)).kind is (
            NoteSaveOutcomeKind.SAVED
        )
        await asyncio.sleep(delay)
        assert session.mutate(body=SECOND_SAVE)
        assert (await session.request_save(explicit=False)).kind is (
            NoteSaveOutcomeKind.SAVED
        )
        await owner.settle()

        _healthy(vault, owner, SECOND_SAVE)
    finally:
        await owner.shutdown()


async def test_a_note_in_no_synced_folder_never_waits(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: with root-1's pass held open, an unbound note's re-save
    is not delayed, and its port asks the runtime to settle nothing."""

    assert vault.database.add_note("loose", "outside any vault", "note-free")
    gate = _PassGate(monkeypatch)
    owner = build_owner(vault)
    await owner.start()
    settles: list[float] = []
    real_settle = owner.settle

    async def counted_settle():
        settles.append(time.monotonic())
        await real_settle()

    monkeypatch.setattr(owner, "settle", counted_settle)
    try:
        bound = await _session(vault, owner)
        assert bound.mutate(body=FIRST_SAVE)
        assert (await bound.request_save(explicit=False)).kind is (
            NoteSaveOutcomeKind.SAVED
        )
        await asyncio.wait_for(gate.reached.wait(), 10)  # root-1's pass is held open

        port = _port(vault, owner)
        loose = DatabaseNoteSessionCoordinator(
            port, clock=lambda: datetime.now(timezone.utc)
        )
        assert (
            await loose.open_session("note-free")
        ).kind is NoteLoadOutcomeKind.LOADED
        started = time.monotonic()
        for text in ("first loose save", "second loose save"):
            assert loose.mutate(body=text)
            outcome = await asyncio.wait_for(loose.request_save(explicit=False), 2.0)
            assert outcome.kind is NoteSaveOutcomeKind.SAVED
        took = time.monotonic() - started

        assert not gate.release.is_set(), "the control needs the pass still held"
        assert settles == [], "an unbound note's save waited on the sync runtime"
        assert took < 2.0
        row = vault.database.get_note_by_id("note-free")
        assert row is not None and row["content"] == "second loose save"
        # Not even one event-loop turn: the hook finishes without suspending.
        probe = port.settle_before_save("note-free")
        with pytest.raises(StopIteration) as finished:
            probe.send(None)
        assert finished.value.value is False
    finally:
        gate.release.set()
        await owner.shutdown()


async def test_a_pass_that_outlasts_the_wait_does_not_block_the_save(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On timeout the save proceeds. The folder may then be held, as before the
    mitigation, and Recovery heals it with nothing lost."""

    monkeypatch.setattr(attention, "RESAVE_SYNC_PASS_WAIT_SECONDS", 0.2, raising=False)
    gate = _PassGate(monkeypatch)
    owner = build_owner(vault)
    await owner.start()
    try:
        session = await _session(vault, owner)
        assert session.mutate(body=FIRST_SAVE)
        assert (await session.request_save(explicit=False)).kind is (
            NoteSaveOutcomeKind.SAVED
        )
        await asyncio.wait_for(gate.reached.wait(), 10)

        assert session.mutate(body=SECOND_SAVE)
        started = time.monotonic()
        outcome = await asyncio.wait_for(session.request_save(explicit=False), 5.0)
        waited = time.monotonic() - started

        assert outcome.kind is NoteSaveOutcomeKind.SAVED
        assert not gate.release.is_set(), "the save must not have needed the pass"
        assert vault.note()["content"] == SECOND_SAVE
        assert waited >= 0.2, "the save did not wait for the pass at all"

        gate.release.set()
        await owner.settle()
        root = owner.snapshot().roots[0]
        if root.status != "up_to_date":
            # Visible and healable, exactly as today.
            assert root.status == "needs_attention"
            if root.action_id is not None:
                await owner.resolve_cleanup("root-1", root.action_id)
            else:
                await owner.note_changed("note-1")
            await owner.settle()
        _healthy(vault, owner, SECOND_SAVE)
    finally:
        gate.release.set()
        await owner.shutdown()


def test_the_resave_wait_fits_inside_the_flush_bound() -> None:
    """A navigation or quit flush that has to wait must still have time to save."""

    assert (
        0
        < attention.RESAVE_SYNC_PASS_WAIT_SECONDS
        < library_pending_work._DEFAULT_FLUSH_TIMEOUT_SECONDS
    )
    assert attention.RESAVE_SYNC_PASS_WAIT_SECONDS <= attention.SYNC_PASS_WAIT_SECONDS


# --- I4: a clean quit flush waits for the pass it hinted -------------------------


def _quitting_screen(session: DatabaseNoteSessionCoordinator, owner) -> SimpleNamespace:
    """What ``confirm_library_quit`` reads of the Library screen.

    The flush is the real session's flush over the real port; the runtime is
    the app's own attribute. No widget is involved in what this pins.
    """

    async def flush_pending_work(*, quitting: bool = False) -> bool:
        assert quitting
        return (await session.flush()).kind is NoteFlushOutcomeKind.PERMITTED

    return SimpleNamespace(
        flush_pending_work=flush_pending_work,
        app=SimpleNamespace(NAVIGATION_FLUSH_TIMEOUT_SECONDS=5.0),
        app_instance=SimpleNamespace(notes_sync_runtime_owner=owner),
        _library_note_session=session,
    )


async def test_ctrl_q_waits_for_the_pass_its_flush_hinted_and_a_relaunch_is_in_step(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Save, quit, relaunch: the file is in step when the quit may exit."""

    gate = _PassGate(monkeypatch)
    owner = build_owner(vault)
    await owner.start()
    session = await _session(vault, owner)
    assert session.mutate(body=TAIL_EDIT)

    async def a_slow_folder() -> None:
        await gate.reached.wait()
        await asyncio.sleep(0.4)
        gate.release.set()

    slow = asyncio.create_task(a_slow_folder())
    try:
        assert await library_pending_work.confirm_library_quit(
            _quitting_screen(session, owner)
        )
        # The approved quit may exit now. Read what it would leave on disk.
        file_at_exit = vault.file.read_bytes()
        note_at_exit = vault.note()["content"]
    finally:
        gate.release.set()
        slow.cancel()
        await owner.shutdown()

    assert note_at_exit == TAIL_EDIT, "the quit flush lost the save"
    assert file_at_exit == _file_bytes(TAIL_EDIT), (
        "the quit exited before the sync pass its flush hinted had run"
    )
    relaunched = build_owner(vault)
    await relaunched.start()
    try:
        await relaunched.settle()
        _healthy(vault, relaunched, TAIL_EDIT)
    finally:
        await relaunched.shutdown()


async def test_a_quit_past_a_stuck_pass_is_bounded_and_the_relaunch_catches_up(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exception the guide names: the quit never hangs on a slow folder."""

    monkeypatch.setattr(attention, "SYNC_PASS_WAIT_SECONDS", 0.3)
    gate = _PassGate(monkeypatch)
    owner = build_owner(vault)
    await owner.start()
    session = await _session(vault, owner)
    assert session.mutate(body=TAIL_EDIT)
    try:
        started = time.monotonic()
        assert await asyncio.wait_for(
            library_pending_work.confirm_library_quit(_quitting_screen(session, owner)),
            10,
        )
        waited = time.monotonic() - started
        assert vault.note()["content"] == TAIL_EDIT
        assert waited >= 0.3, "the quit did not wait for the pass at all"
        assert not gate.release.is_set()
    finally:
        gate.release.set()
        await owner.shutdown()

    relaunched = build_owner(vault)
    await relaunched.start()
    try:
        await relaunched.settle()
        root = relaunched.snapshot().roots[0]
        if root.action_id is not None:
            await relaunched.resolve_cleanup("root-1", root.action_id)
            await relaunched.settle()
        _healthy(vault, relaunched, TAIL_EDIT)
    finally:
        await relaunched.shutdown()
