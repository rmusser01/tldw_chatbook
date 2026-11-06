"""Bulk Add/Remove keyword commits each note on its own (PR #3423 N-1/N-2/N-3).

The re-review of the I-2 fix found the bulk keyword worker holding ONE
``BEGIN IMMEDIATE`` Notes transaction across every checked note. On a ready
Sync v2 local-first profile each note pays 0.12-0.2 s of Sync v2 profile
reads inside it, so the whole ChaChaNotes database -- conversations and
messages too, not only notes -- stayed write-locked for the entire run: a
concurrent ``add_note`` failed at once with "database is locked", and
``add_conversation`` waited out the run (N-1). A note whose change raised
after its first write was committed half-done while the receipt said it
"could not be changed" (N-2), and a refused change dropped its reason (N-3).

Ruling (controller, 2026-11-06): one ``note_transaction`` per note, still
IMMEDIATE; a note whose change raises rolls back alone; the refusal receipt
names the reason. Each pin boots the real ``LibraryScreen`` with the
production stylesheet over a real ``CharactersRAGDB`` behind the real Notes
scope service (``library_notes_bulk_support``), with a ready Sync v2 profile
wired by the app's own ``_wire_notes_sync_services``:

- Concurrent writers. An event gate opens once, at the first moment after
  the first note's change is in: the bulk's request for the next note's
  transaction or, failing that, the next note's Sync v2 write. There the gate
  runs ``add_note`` and then ``add_conversation`` on a thread of their own
  (so their own connection) and holds the bulk until both return, for at
  most ``GATE_HOLD`` seconds. Both must succeed within ``WRITER_BOUND`` seconds.
  This pins that the bulk lets go of the database between notes. It does
  NOT show that a writer arriving at any moment gets in: each note's ~0.12 s
  of Sync v2 profile reads still runs inside that note's write transaction,
  and a writer that arrives mid-note is starved (``add_conversation`` waits
  out the rest of the run, ``add_note`` fails at once) -- measured and
  tracked in TASK-34000.116.
- A fault after the first note's writes (the organization service's own
  ``failure_injector``, called after a note's keyword, link and intents are
  written) rolls that note back whole: no link, no intents, and the receipt
  counts it as not changed with the note still checked.
- A refusal names its reason: the runtime policy's own sentence, and the Sync
  v2 group not being ready.

Base (the shared transaction, the ``PR #3423`` review-fix head): the writers
fail (``add_note``: "database is locked" at once; ``add_conversation``: held
past the bound), the faulted note keeps its link and intents, and the
receipts carry no reason.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from Tests.UI.library_notes_bulk_support import (
    enter_select_mode,
    mounted,
    notes_ready,
    painted,
    until,
)
from Tests.UI.test_library_notes_bulk_keyword_sync_v2 import (
    DATASET,
    PICKED,
    SERVER,
    WIDE,
    _DenyNoteUpdates,
    _intents,
    _ready_sync_v2,
)
from Tests.UI.test_library_notes_select_mode_counts_and_bulk_extended import (
    _check,
    _keyword,
)
from Tests.UI.test_library_notes_trash_pages_and_purges_extended import (
    _painted_toasts,
)
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: Each concurrent writer must succeed within this many seconds of the gate.
WRITER_BOUND = 3.0
#: The longest the gate holds the bulk while the writers run. Longer than the
#: bound, so a writer that can only finish after the bulk lets go fails it.
GATE_HOLD = 6.0
#: The not-ready group's sentence, pinned as the receipt paints it.
SYNC_NOT_READY = (
    "Sync v2 is not ready to carry note keywords yet — see Manual Sync in "
    "Settings."
)


class _MidRunWriters:
    """Run two writers once, mid-run, and hold the bulk meanwhile.

    One thread of their own (so their own connection) runs them one after
    the other: ``add_note``'s deferred transaction fails at once on ANY held
    write lock, so two writers started together can refuse each other
    (measured: "database is locked after 0.04 s" at load ~3, with the bulk
    holding nothing) -- that is not the bulk's doing.
    """

    def __init__(self, db) -> None:
        self.db = db
        self.opened = threading.Event()
        self.results: dict[str, tuple[str, float]] = {}
        self.changes_before: int | None = None
        self.at = ""
        self._lock = threading.Lock()

    def open(self, at: str, changes_so_far: int) -> None:
        with self._lock:
            if self.opened.is_set():
                return
            self.opened.set()
        self.at = at
        self.changes_before = changes_so_far
        writer = threading.Thread(target=self._write_both, daemon=True)
        writer.start()
        writer.join(GATE_HOLD)

    def _write_both(self) -> None:
        for kind in ("add_note", "add_conversation"):
            self._write(kind)

    def _write(self, kind: str) -> None:
        start = time.monotonic()
        try:
            if kind == "add_note":
                self.db.add_note("Concurrent note", "written mid-bulk")
            else:
                self.db.add_conversation(
                    {"title": "Concurrent chat", "character_id": None}
                )
        except Exception as exc:  # noqa: BLE001 - the pin reports it
            self.results[kind] = (
                f"{type(exc).__name__}: {exc}",
                time.monotonic() - start,
            )
            return
        self.results[kind] = ("ok", time.monotonic() - start)


async def test_concurrent_writers_succeed_while_a_bulk_keyword_change_runs(
    tmp_path,
) -> None:
    host, seed = mounted(tmp_path)
    _ready_sync_v2(seed, tmp_path)
    ids = [seed.note_ids[title] for title in PICKED]
    scope_service = seed.profile.scope_service
    organization = scope_service.organization_sync_service
    interop = seed.profile.interop
    writers = _MidRunWriters(seed.profile.db)
    changes: list[str] = []
    real_transaction = interop.note_transaction
    real_change = scope_service.change_local_note_keyword

    def change(**kwargs):
        changes.append(str(kwargs.get("note_id")))
        return real_change(**kwargs)

    def note_transaction(user_id):
        if changes:
            # The bulk asks for another transaction after a note changed:
            # nothing of the bulk's is held here.
            writers.open("between notes", len(changes))
        return real_transaction(user_id)

    def after_note_write(stage: str) -> None:
        if stage == "after_notes_mutation_and_intent" and len(changes) >= 2:
            # The next note's Sync v2 write, inside whatever the bulk holds.
            writers.open("inside the next note's write", len(changes) - 1)

    scope_service.change_local_note_keyword = change
    interop.note_transaction = note_transaction
    organization.failure_injector = after_note_write
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, PICKED)
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "travel",
            "Add a keyword to 3 notes",
        )
        # Both writers report before anything else is judged: a held bulk
        # also starves the Library's own reads, which would otherwise be the
        # first thing to time out.
        await until(
            pilot,
            lambda: len(writers.results) == 2,
            lambda: f"both mid-run writers to return: {writers.results}",
        )
        assert writers.opened.is_set(), "the gate never opened mid-run"
        # Mid-run: the first note had changed and the others had not.
        assert writers.changes_before == 1, (writers.at, writers.changes_before)
        for kind in ("add_note", "add_conversation"):
            outcome, seconds = writers.results[kind]
            assert outcome == "ok" and seconds <= WRITER_BOUND, (
                f"{kind} during the bulk ({writers.at}): {outcome} after "
                f"{seconds:.2f} s (bound {WRITER_BOUND} s)"
            )
        await until(
            pilot,
            lambda: "Added 'travel' to 3 notes" in painted(host),
            lambda: f"the add receipt: {painted(host)}",
        )
    # The bulk itself is whole: every note linked, one intent per link.
    assert seed.keyword_links("travel") == set(ids)
    links = [row for row in _intents(seed) if row[0] == "notes.keyword_link"]
    assert sorted(row[2] for row in links) == sorted(ids), links
    with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM notes WHERE title = 'Concurrent note'"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE title = 'Concurrent chat'"
        ).fetchone() == (1,)


async def test_a_fault_after_a_notes_first_write_rolls_that_note_back(
    tmp_path,
) -> None:
    host, seed = mounted(tmp_path)
    _ready_sync_v2(seed, tmp_path)
    scope_service = seed.profile.scope_service
    organization = scope_service.organization_sync_service
    order: list[str] = []
    real_change = scope_service.change_local_note_keyword

    def change(**kwargs):
        order.append(str(kwargs.get("note_id")))
        return real_change(**kwargs)

    def fault_the_first_note(stage: str) -> None:
        # Called after the note's keyword, link and intents are written.
        if stage == "after_notes_mutation_and_intent" and len(order) == 1:
            raise RuntimeError("injected after the first note's writes")

    scope_service.change_local_note_keyword = change
    organization.failure_injector = fault_the_first_note
    async with host.run_test(size=WIDE, notifications=True) as pilot:
        screen = await notes_ready(host, pilot)
        screen.app_instance.notify = host.notify
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, PICKED)
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "travel",
            "Add a keyword to 3 notes",
        )
        receipt = (
            "Added 'travel' to 2 notes · 1 could not be changed — still "
            "checked, so you can try again"
        )
        await until(
            pilot,
            lambda: screen._notes_state.notice == receipt,
            lambda: f"the partial receipt: {screen._notes_state.notice!r}",
        )
        faulted = order[0]
        # The faulted note stays checked, alone.
        assert screen._notes_state.select_mode
        assert screen._notes_state.row_selection.count == 1
        assert screen._notes_state.row_selection.is_selected(faulted)
    others = {seed.note_ids[title] for title in PICKED} - {faulted}
    assert seed.keyword_links("travel") == others
    intents = _intents(seed)
    # Nothing of the faulted note's change survived: no link intent for it...
    assert faulted not in {row[2] for row in intents if row[0] == "notes.keyword_link"}
    assert sorted(
        row[2] for row in intents if row[0] == "notes.keyword_link"
    ) == sorted(others), intents
    # ...and the keyword it created went with it, so the next note created it
    # again and recorded its resource once.
    assert [row for row in intents if row[0] == "notes.keyword"] == [
        ("notes.keyword", "upsert", "travel")
    ], intents


async def test_a_refused_bulk_keyword_change_names_its_reason(tmp_path) -> None:
    host, seed = mounted(tmp_path)
    state = _ready_sync_v2(seed, tmp_path)
    assert state is not None
    async with host.run_test(size=WIDE, notifications=True) as pilot:
        screen = await notes_ready(host, pilot)
        screen.app_instance.notify = host.notify
        policy = _DenyNoteUpdates()
        seed.profile.scope_service.policy_enforcer = policy
        before = seed.keyword_state()
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, PICKED)
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "food",
            "Add a keyword to 3 notes",
        )
        refused = (
            "Added 'food' to 0 notes · 3 could not be changed — still checked, "
            "so you can try again. Local note changes are not allowed right now."
        )
        await until(
            pilot,
            lambda: screen._notes_state.notice == refused,
            lambda: f"the policy receipt: {screen._notes_state.notice!r}",
        )
        await until(
            pilot,
            lambda: any(
                "Local note changes are not allowed right now." in toast
                for toast in _painted_toasts(host)
            ),
            lambda: f"the policy toast: {_painted_toasts(host)!r}",
        )
        assert seed.keyword_state() == before

        # The policy lets go; the Sync v2 group is now not ready. The same
        # three notes are still checked, so the same action retries them.
        seed.profile.scope_service.policy_enforcer = None
        with seed.profile.db.transaction() as cursor:
            cursor.execute(
                "UPDATE notes_organization_sync_checkpoints SET local_state = "
                "'pulling' WHERE server_profile_id = ? AND dataset_id = ?",
                (SERVER, DATASET),
            )
        assert screen._notes_state.row_selection.count == 3
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "food",
            "Add a keyword to 3 notes",
        )
        not_ready = (
            "Added 'food' to 0 notes · 3 could not be changed — still checked, "
            f"so you can try again. {SYNC_NOT_READY}"
        )
        await until(
            pilot,
            lambda: screen._notes_state.notice == not_ready,
            lambda: f"the not-ready receipt: {screen._notes_state.notice!r}",
        )
        assert seed.keyword_state() == before
