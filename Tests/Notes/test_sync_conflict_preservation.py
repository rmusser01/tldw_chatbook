"""Conflict-path preservation and strategy-application pins (task-19554).

Why this module exists — the incident, not the rule. The 2026-08-21 holistic
review (Lane 3, F4) found two live defects in ``NotesSyncEngine``'s conflict
path, both reachable from the Library sync panel on an unattended 300-second
timer:

1. **Unrecoverable overwrite.** On a ``both_changed`` conflict the losing side
   was overwritten wholesale. ``SyncConflict`` *carried* ``db_content`` and
   ``disk_content``, but ``_record_conflict`` persisted only the two hashes and
   ``sync_conflicts`` had no content columns at all. No ``.bak``, no history
   row: the discarded text was gone.
2. **``disk_wins`` was a lie.** ``ConflictResolution.DISK_WINS`` appeared in
   ``sync_engine.py`` exactly once — its own enum definition. Selecting "Disk
   wins" recorded the conflict, applied nothing, and reported as resolved.

The two born-red pins for those are
``test_newer_wins_db_newer_preserves_the_losing_disk_copy`` and
``test_disk_wins_actually_applies_the_disk_copy``; every other test here guards
a way the fix could be hollowed out (a preserved copy that is really the
winner's text, a sidecar that gets re-ingested as a note next pass, a
preservation failure that still lets the overwrite through, a run that reports
"resolved" for a conflict it left alone).

``_recoverable_copies`` deliberately looks in BOTH places a losing copy can
live (sidecar on disk, ``sync_conflicts.losing_content`` in the DB) and
tolerates the column being absent, so at base these tests go red on the
behaviour — an empty recovery set — rather than on a missing-column
``OperationalError``.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Notes.Notes_Library import NotesInteropService
from tldw_chatbook.Notes.sync_engine import (
    ConflictResolution,
    NotesSyncEngine,
    SyncDirection,
)

USER = "test_user"
NOTE_NAME = "conflict_test.md"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def temp_dir():
    temp_path = Path(tempfile.mkdtemp()).resolve(strict=True)
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def test_db(temp_dir):
    db = CharactersRAGDB(str(temp_dir / "test_notes.db"), "test_client")
    yield db
    db.close_connection()


@pytest.fixture
def notes_service(test_db, temp_dir):
    service = NotesInteropService(
        base_db_directory=temp_dir,
        api_client_id="test_api",
        global_db_to_use=test_db,
    )
    yield service
    service.close_all_user_connections()


@pytest.fixture
def sync_engine(notes_service, test_db):
    return NotesSyncEngine(notes_service, test_db)


@pytest.fixture
def sync_dir(temp_dir):
    path = temp_dir / "sync_root"
    path.mkdir()
    return path


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _version(notes_service: NotesInteropService, note_id: str) -> int:
    return notes_service.get_note_by_id(USER, note_id)["version"]


def _content(notes_service: NotesInteropService, note_id: str) -> str:
    return notes_service.get_note_by_id(USER, note_id)["content"]


def _make_conflicted_note(
    notes_service: NotesInteropService,
    sync_engine: NotesSyncEngine,
    sync_dir: Path,
    *,
    db_content: str,
    disk_content: str,
    disk_is_newer: bool,
    baseline: str = "Original content",
) -> str:
    """Build a genuine ``both_changed`` conflict with a real sync baseline.

    A real baseline matters: with ``last_synced_disk_file_hash`` left NULL both
    sides compare "changed" against ``None`` and a conflict is detected for the
    wrong reason. Here the baseline hash/mtime are actually stored, then both
    sides are diverged from it.

    ``disk_is_newer`` sets the file's mtime an hour either side of the note's
    ``last_modified`` (which SQLite stamps at one-second resolution), which is
    what ``NEWER_WINS`` compares.
    """
    note_id = notes_service.add_note(
        user_id=USER, title="Conflict Test", content=baseline
    )
    file_path = sync_dir / NOTE_NAME
    file_path.write_text(baseline, encoding="utf-8")

    assert notes_service.update_note_sync_metadata(
        user_id=USER,
        note_id=note_id,
        sync_metadata={
            "file_path_on_disk": str(file_path),
            "relative_file_path_on_disk": NOTE_NAME,
            "sync_root_folder": str(sync_dir),
            "is_externally_synced": 1,
            "file_extension": ".md",
        },
        expected_version=_version(notes_service, note_id),
    )

    file_info = sync_engine._get_file_info(file_path, sync_dir)
    assert file_info is not None
    assert notes_service.update_note_sync_metadata(
        user_id=USER,
        note_id=note_id,
        sync_metadata={
            "last_synced_disk_file_hash": file_info.content_hash,
            "last_synced_disk_file_mtime": file_info.mtime,
        },
        expected_version=_version(notes_service, note_id),
    ), "the baseline hash must actually be stored, or the conflict is bogus"

    # Diverge both sides from that baseline.
    file_path.write_text(disk_content, encoding="utf-8")
    assert notes_service.update_note(
        user_id=USER,
        note_id=note_id,
        update_data={"content": db_content},
        expected_version=_version(notes_service, note_id),
    )

    stamp = time.time() + 3600 if disk_is_newer else time.time() - 3600
    os.utime(file_path, (stamp, stamp))
    return note_id


def _recoverable_copies(db: CharactersRAGDB, sync_dir: Path, session_id: str) -> list[str]:
    """Every text the losing side could be recovered from after a run.

    Looks in both preservation surfaces and tolerates the DB one not existing,
    so a base-version run fails on an EMPTY recovery set rather than on an
    ``OperationalError`` about a missing column.
    """
    found: list[str] = []
    for path in sorted(sync_dir.rglob("*")):
        if path.is_file() and ".conflict-" in path.name:
            found.append(path.read_text(encoding="utf-8"))
    with db.transaction() as conn:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(sync_conflicts)")
        }
        if "losing_content" in columns:
            for row in conn.execute(
                "SELECT losing_content FROM sync_conflicts "
                "WHERE session_id = ? AND losing_content IS NOT NULL",
                (session_id,),
            ):
                found.append(row[0])
    return found


def _sidecars(sync_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in sync_dir.rglob("*")
        if path.is_file() and ".conflict-" in path.name
    )


def _conflict_rows(db: CharactersRAGDB, session_id: str) -> list[dict[str, Any]]:
    with db.transaction() as conn:
        columns = [
            row[1] for row in conn.execute("PRAGMA table_info(sync_conflicts)")
        ]
        selected = ", ".join(columns)
        return [
            dict(zip(columns, row))
            for row in conn.execute(
                f"SELECT {selected} FROM sync_conflicts WHERE session_id = ?",
                (session_id,),
            )
        ]


async def _run(
    sync_engine: NotesSyncEngine,
    sync_dir: Path,
    resolution: ConflictResolution,
    direction: SyncDirection = SyncDirection.BIDIRECTIONAL,
):
    return await sync_engine.sync(
        root_path=sync_dir,
        user_id=USER,
        direction=direction,
        conflict_resolution=resolution,
    )


# --------------------------------------------------------------------------
# Defect 1 — the overwrite destroyed the losing side (BORN RED)
# --------------------------------------------------------------------------
class TestLosingSideIsRecoverable:
    @pytest.mark.asyncio
    async def test_newer_wins_db_newer_preserves_the_losing_disk_copy(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        """BORN RED (defect 1). DB newer under NEWER_WINS rewrites the file.

        At base the run overwrites ``conflict_test.md`` with the note body and
        the disk text is gone: no sidecar, no history row, only a SHA-256 of it
        in ``sync_conflicts.disk_content_hash``.
        """
        _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )

        session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.NEWER_WINS
        )

        assert len(progress.conflicts) == 1
        # The winner really was applied (this half passes at base).
        assert (sync_dir / NOTE_NAME).read_text(encoding="utf-8") == (
            "Modified in database"
        )
        # The loser must still be reachable.
        recoverable = _recoverable_copies(test_db, sync_dir, session_id)
        assert "Modified on disk" in recoverable, (
            "NEWER_WINS overwrote the on-disk file and left no recoverable "
            f"copy of it; recovery surfaces held {recoverable!r}"
        )

    @pytest.mark.asyncio
    async def test_newer_wins_disk_newer_preserves_the_losing_db_copy(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        """The mirror image: disk newer replaces the note body via update_note."""
        note_id = _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=True,
        )

        session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.NEWER_WINS
        )

        assert len(progress.conflicts) == 1
        assert _content(notes_service, note_id) == "Modified on disk"
        recoverable = _recoverable_copies(test_db, sync_dir, session_id)
        assert "Modified in database" in recoverable, (
            "NEWER_WINS replaced the note content and left no recoverable "
            f"copy of it; recovery surfaces held {recoverable!r}"
        )

    @pytest.mark.asyncio
    async def test_preserved_copy_is_the_loser_not_the_winner(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        """Mutation control: preserving the WINNER would satisfy a naive pin."""
        _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )
        session_id, _progress = await _run(
            sync_engine, sync_dir, ConflictResolution.NEWER_WINS
        )
        recoverable = _recoverable_copies(test_db, sync_dir, session_id)
        assert recoverable, "nothing was preserved at all"
        assert all(text == "Modified on disk" for text in recoverable), (
            f"a preserved copy holds the winner's text: {recoverable!r}"
        )

    @pytest.mark.asyncio
    async def test_sidecar_is_named_for_the_note_and_the_losing_side(
        self, sync_engine, notes_service, sync_dir
    ):
        """The on-disk copy has to be findable by a human, next to the file."""
        _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )
        await _run(sync_engine, sync_dir, ConflictResolution.NEWER_WINS)

        sidecars = _sidecars(sync_dir)
        assert len(sidecars) == 1, sidecars
        sidecar = sidecars[0]
        assert sidecar.parent == sync_dir
        assert sidecar.name.startswith(NOTE_NAME + ".conflict-")
        assert sidecar.name.endswith("-disk.bak")
        assert sidecar.read_text(encoding="utf-8") == "Modified on disk", (
            "the sidecar must be byte-exact so recovery is a rename"
        )

    @pytest.mark.asyncio
    async def test_sidecars_are_never_re_ingested_as_notes(
        self, sync_engine, notes_service, sync_dir
    ):
        """A preserved copy must not become a note (or a sync loop) next pass."""
        _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )
        await _run(sync_engine, sync_dir, ConflictResolution.NEWER_WINS)
        assert _sidecars(sync_dir)

        before = {note["id"] for note in notes_service.list_notes(USER)}
        _session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.NEWER_WINS
        )
        after = {note["id"] for note in notes_service.list_notes(USER)}

        assert after == before, "the conflict sidecar was ingested as a note"
        assert not progress.created_notes

    @pytest.mark.asyncio
    async def test_preservation_failure_blocks_the_overwrite(
        self, sync_engine, notes_service, test_db, sync_dir, monkeypatch
    ):
        """Fail-closed: if the losing copy cannot be saved, nothing is destroyed."""
        note_id = _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )

        def _explode(*_args, **_kwargs):
            raise OSError("preservation is unavailable")

        monkeypatch.setattr(
            NotesSyncEngine, "_write_conflict_sidecar", _explode, raising=True
        )

        _session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.NEWER_WINS
        )

        assert (sync_dir / NOTE_NAME).read_text(encoding="utf-8") == (
            "Modified on disk"
        ), "the file was overwritten even though its copy could not be saved"
        assert _content(notes_service, note_id) == "Modified in database"
        assert len(progress.conflicts) == 1
        assert progress.conflicts[0].applied is False
        assert progress.errors, "a blocked resolution must be reported"


# --------------------------------------------------------------------------
# Defect 2 — disk_wins applied nothing (BORN RED)
# --------------------------------------------------------------------------
class TestEveryOfferedStrategyApplies:
    @pytest.mark.asyncio
    async def test_disk_wins_actually_applies_the_disk_copy(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        """BORN RED (defect 2). ``DISK_WINS`` had no branch anywhere.

        At base the note keeps its database body and the run still reports the
        conflict as resolved by the "Disk wins" policy.
        """
        note_id = _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            # deliberately the side NEWER_WINS would NOT pick, so a pass here
            # cannot come from newer-wins logic leaking in.
            disk_is_newer=False,
        )

        session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.DISK_WINS
        )

        assert len(progress.conflicts) == 1
        assert _content(notes_service, note_id) == "Modified on disk", (
            "DISK_WINS did not apply the disk copy to the note"
        )
        assert (sync_dir / NOTE_NAME).read_text(encoding="utf-8") == (
            "Modified on disk"
        )
        recoverable = _recoverable_copies(test_db, sync_dir, session_id)
        assert "Modified in database" in recoverable

    @pytest.mark.asyncio
    async def test_db_wins_actually_applies_the_db_copy(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        """``DB_WINS`` had no bidirectional branch either — same class."""
        note_id = _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=True,
        )

        session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.DB_WINS
        )

        assert len(progress.conflicts) == 1
        assert (sync_dir / NOTE_NAME).read_text(encoding="utf-8") == (
            "Modified in database"
        ), "DB_WINS did not write the note body to disk"
        assert _content(notes_service, note_id) == "Modified in database"
        recoverable = _recoverable_copies(test_db, sync_dir, session_id)
        assert "Modified on disk" in recoverable

    @pytest.mark.asyncio
    async def test_ask_still_applies_nothing_and_says_so(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        """Control: ``ASK`` is the one policy that must NOT touch either side."""
        note_id = _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )

        session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.ASK
        )

        assert (sync_dir / NOTE_NAME).read_text(encoding="utf-8") == (
            "Modified on disk"
        )
        assert _content(notes_service, note_id) == "Modified in database"
        assert not _sidecars(sync_dir), "nothing was discarded, so nothing to save"
        assert len(progress.conflicts) == 1
        assert progress.conflicts[0].applied is False
        rows = _conflict_rows(test_db, session_id)
        assert len(rows) == 1
        assert rows[0]["resolution"] is None, (
            "an unresolved conflict must stay open for later resolution"
        )

    @pytest.mark.asyncio
    async def test_db_to_disk_newer_wins_pushes_when_the_note_is_newer(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        """The one-way push path had no NEWER_WINS branch at all."""
        _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )

        session_id, progress = await _run(
            sync_engine,
            sync_dir,
            ConflictResolution.NEWER_WINS,
            direction=SyncDirection.DB_TO_DISK,
        )

        assert len(progress.conflicts) == 1
        assert (sync_dir / NOTE_NAME).read_text(encoding="utf-8") == (
            "Modified in database"
        )
        assert "Modified on disk" in _recoverable_copies(
            test_db, sync_dir, session_id
        )

    @pytest.mark.asyncio
    async def test_disk_to_db_conflict_preserves_the_note_body(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        """The pull path overwrote the note with no conflict record at all."""
        note_id = _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=True,
        )

        session_id, progress = await _run(
            sync_engine,
            sync_dir,
            ConflictResolution.DISK_WINS,
            direction=SyncDirection.DISK_TO_DB,
        )

        assert _content(notes_service, note_id) == "Modified on disk"
        assert len(progress.conflicts) == 1
        assert "Modified in database" in _recoverable_copies(
            test_db, sync_dir, session_id
        )


# --------------------------------------------------------------------------
# The report must match what actually happened (AC #4)
# --------------------------------------------------------------------------
class TestReportedOutcomeMatchesReality:
    @pytest.mark.asyncio
    async def test_applied_flag_and_persisted_resolution_agree(
        self, sync_engine, notes_service, test_db, sync_dir
    ):
        _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )
        session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.DISK_WINS
        )

        conflict = progress.conflicts[0]
        assert conflict.applied is True
        assert conflict.resolution == "use_disk"
        assert conflict.preserved_path is not None

        rows = _conflict_rows(test_db, session_id)
        assert len(rows) == 1
        assert rows[0]["resolution"] == "use_disk"
        assert rows[0]["losing_side"] == "db"
        assert rows[0]["losing_content"] == "Modified in database"
        assert rows[0]["preserved_file_path"] == str(conflict.preserved_path)
        assert rows[0]["resolved_at"] is not None

    @pytest.mark.asyncio
    async def test_a_run_that_changed_nothing_reports_no_applied_conflict(
        self, sync_engine, notes_service, sync_dir
    ):
        """AC #4, stated as the engine-level fact the UI line is built from."""
        _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )
        _session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.ASK
        )
        assert progress.conflicts
        assert not [c for c in progress.conflicts if c.applied]
        assert not progress.updated_files
        assert not progress.updated_notes


# --------------------------------------------------------------------------
# Qodo review round on PR #1922 — defects inside the shipped design
# --------------------------------------------------------------------------
class TestSidecarRecognitionDoesNotSwallowRealNotes:
    """Finding 1: the sidecar filter must not un-sync legitimate notes.

    ``is_conflict_sidecar`` matched the ``.conflict-`` marker ANYWHERE in a
    filename, and ``_scan_directory`` drops what it matches. A user's own note
    called ``meeting.conflict-notes.md`` was therefore silently excluded from
    every sync — no error, no skip row, it simply stopped being mirrored.
    Sidecars always end in ``.bak``, so requiring both keeps the
    never-re-ingest guarantee without eating real notes.
    """

    @pytest.mark.asyncio
    async def test_a_note_whose_name_contains_the_marker_still_syncs(
        self, sync_engine, notes_service, sync_dir
    ):
        decoy = sync_dir / "meeting.conflict-notes.md"
        decoy.write_text("Notes about a merge conflict at work", encoding="utf-8")

        _session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.NEWER_WINS
        )

        assert len(progress.created_notes) == 1, (
            "a legitimate note containing '.conflict-' was silently dropped "
            f"from the scan; skipped={progress.skipped_items}"
        )
        titles = {note["title"] for note in notes_service.list_notes(USER)}
        assert "meeting.conflict-notes" in titles

    def test_recognition_requires_both_the_marker_and_the_bak_suffix(self):
        recognized = NotesSyncEngine.is_conflict_sidecar
        # Real sidecars, in every shape the writer can produce.
        assert recognized(Path("note.md.conflict-20260821T203015Z-disk.bak"))
        assert recognized(Path("note.md.conflict-20260821T203015Z-2-db.bak"))
        assert recognized(Path("sub/note.txt.conflict-20260821T203015Z-db.bak"))
        # Real notes that merely mention the marker, or merely end in .bak.
        assert not recognized(Path("meeting.conflict-notes.md"))
        assert not recognized(Path("the.conflict-of-1914.txt"))
        assert not recognized(Path("archive.bak"))
        assert not recognized(Path("notes.md"))


class TestSidecarNameIsClaimedAtomically:
    """Finding 2: an exists-then-write sidecar can destroy another copy.

    ``_write_conflict_sidecar`` used to test ``Path.exists()`` and then call
    ``PinnedSyncRoot.write_text``, which REPLACES whatever is at the target.
    Two sync runs resolving the same note in the same second both saw "free"
    and the second one's rename destroyed the first one's preserved copy —
    losing exactly the data this task exists to preserve, in the one code path
    whose whole job is not to.

    The window is opened deterministically here through ``_before_create``,
    the module's existing test-seam idiom (see ``_before_replace``, used the
    same way in ``test_sync_containment.py``): a competitor claims the name
    after the name was chosen and before this run's create.
    """

    @pytest.mark.asyncio
    async def test_a_competitor_claiming_the_name_first_is_never_clobbered(
        self, sync_engine, notes_service, sync_dir, monkeypatch
    ):
        from tldw_chatbook.Notes.sync_paths import PinnedSyncRoot

        _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )

        competitor = "another run's preserved copy"
        claimed: list[Path] = []

        def _claim_the_name_first(self, relative_path: Path) -> None:
            # Fire once: a competitor that re-claimed every candidate would
            # exhaust the name space instead of demonstrating the clobber.
            if claimed:
                return
            claimed.append(relative_path)
            (self.canonical_root / relative_path).write_text(
                competitor, encoding="utf-8"
            )

        monkeypatch.setattr(
            PinnedSyncRoot, "_before_create", _claim_the_name_first, raising=True
        )

        _session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.NEWER_WINS
        )

        assert claimed, "the seam never fired -- the test proves nothing"
        sidecars = _sidecars(sync_dir)
        contents = {path.read_text(encoding="utf-8") for path in sidecars}
        assert competitor in contents, (
            "this run overwrote a preserved copy that already held the name; "
            f"surviving sidecars: {sorted(contents)}"
        )
        assert "Modified on disk" in contents, (
            "this run's own copy must land too, under the next free name"
        )
        assert len(sidecars) == 2, sorted(path.name for path in sidecars)
        # ...and the resolution still went through, on the second name.
        assert progress.conflicts[0].applied is True
        assert (sync_dir / NOTE_NAME).read_text(encoding="utf-8") == (
            "Modified in database"
        )

    @pytest.mark.asyncio
    async def test_an_exhausted_name_space_fails_closed(
        self, sync_engine, notes_service, sync_dir, monkeypatch
    ):
        """Raise rather than overwrite when every candidate is taken."""
        from tldw_chatbook.Notes.sync_paths import PinnedSyncRoot

        note_id = _make_conflicted_note(
            notes_service,
            sync_engine,
            sync_dir,
            db_content="Modified in database",
            disk_content="Modified on disk",
            disk_is_newer=False,
        )

        def _always_claim(self, relative_path: Path) -> None:
            (self.canonical_root / relative_path).write_text(
                "squatter", encoding="utf-8"
            )

        monkeypatch.setattr(
            PinnedSyncRoot, "_before_create", _always_claim, raising=True
        )

        _session_id, progress = await _run(
            sync_engine, sync_dir, ConflictResolution.NEWER_WINS
        )

        assert (sync_dir / NOTE_NAME).read_text(encoding="utf-8") == (
            "Modified on disk"
        ), "nothing may be overwritten when no copy could be saved"
        assert _content(notes_service, note_id) == "Modified in database"
        assert progress.conflicts[0].applied is False
        assert progress.errors
