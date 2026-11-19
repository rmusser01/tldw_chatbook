"""Ten consecutive Move note operations never report a false conflict.

TASK-34000.28 (N-24, AC#5), review notes-library-ux-review-2026-10-02: the
verifier saw 4 of 7 moves fail with "That folder changed elsewhere — refresh
and try aga…" and nothing written. Measured cause (this file, on the base
commit, failing at move 1): the repository's DEFERRED mutation transaction
read, then another thread's finite worker closed its connection with
``PRAGMA wal_checkpoint(TRUNCATE)``, which restarted the WAL, and the
mutation's first write failed with SQLITE_BUSY_SNAPSHOT (517) -- no other
write anywhere. The deterministic pin of that cause is
``Tests/Notes/test_note_folder_repository.py::
test_a_wal_restart_after_the_first_read_cannot_fail_a_mutation``; this file
is the end-to-end guard, outside the PR lane (about 3 s a move).

Same harness and seed as ``test_library_notes_move_never_false_conflicts.py``:
the real ``LibraryScreen`` with production CSS over a real ChaChaNotes
database, every move a real row press, a press of Move note, typed picker
keys and Down + Enter, each started as soon as the previous mutation returns
(not when the tree has settled).
"""

from __future__ import annotations

import pytest

from Tests.UI.library_quit_guard_support import SIZE
from Tests.UI.test_library_notes_move_never_false_conflicts import (
    MOVES,
    UPDATED,
    _mounted,
    _move,
    _note_rows,
    _notes_ready,
    _placements,
    _record_conflict_sites,
    _record_mutations,
    _row_for,
    _wait,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_ten_consecutive_moves_never_report_a_false_conflict(
    tmp_path, monkeypatch
) -> None:
    """AC#5: ten moves in a row, Unfiled and foldered, all succeed.

    The recorder holds every repository site that translated an error into
    ``FolderConflictError`` with the SQLite code it saw, so a RED run names
    the cause, not just the notice.
    """
    sites = _record_conflict_sites(monkeypatch)
    host, seeded = _mounted(tmp_path)
    async with host.run_test(size=SIZE) as pilot:
        screen = await _notes_ready(host, pilot)
        outcomes = _record_mutations(screen, monkeypatch)
        for index, move in enumerate(MOVES, start=1):
            title, source, _query, destination = move
            operation, asked, ok, notice = await _move(
                host, pilot, screen, outcomes, seeded, move
            )
            destination_id = seeded.folder_ids[destination.split(" / ")[-1]]
            assert (operation, asked["destination_folder_id"]) == (
                "move_placement",
                destination_id,
            ), outcomes
            assert (ok, notice) == (True, UPDATED), (
                f"move {index} ({title!r} -> {destination!r}): {notice!r}; "
                f"conflict sites: {sites}"
            )
            if source is None:
                note_id = seeded.note_ids[title]
                # Ruling 4 (routed from Task 3's review): the moved note
                # leaves the Unfiled group without any further key, and the
                # destination lists it.
                await _wait(
                    pilot,
                    lambda: _row_for(screen, note_id, None) is None
                    and _row_for(screen, note_id, destination_id) is not None,
                    lambda: f"{title!r} gone from Unfiled and listed in "
                    f"{destination!r}: "
                    f"{[(r.note_id, r.folder_id) for r in _note_rows(screen)]}",
                )

    assert sites == []
    placements = _placements(tmp_path)
    for title, source, _query, destination in MOVES:
        assert placements[seeded.note_ids[title]] == {
            seeded.folder_ids[destination.split(" / ")[-1]]
        }, (title, source, destination)
