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
  This pins that the bulk lets go of the database between notes.
- A fault after the first note's writes (the organization service's own
  ``failure_injector``, called after a note's keyword, link and intents are
  written) rolls that note back whole: no link, no intents, and the receipt
  counts it as not changed with the note still checked.
- A refusal names its reason: the runtime policy's own sentence, and the Sync
  v2 group not being ready.

TASK-34000.116 (the rest of N-1): each note also used to resolve the Sync v2
profile inside its own write transaction -- four ``SyncStateRepository``
reads, about 0.12 s -- so the write lock was held for nearly the whole run,
and a writer that arrived mid-note waited out the rest of the run. The bulk
now resolves the profile, and checks its group's readiness, once before the
first note's transaction:

- Profile reads. A counting seam on the wired ``SyncStateRepository`` counts
  every profile read the bulk worker makes, and those made while it holds a
  note's transaction: none inside, and the same few for 5 notes as for 25.
- An un-gated writer. On a 50-note and a 120-note bulk, ``add_conversation``
  starts on its own thread the moment the middle note commits, and nothing
  waits for it: it must succeed within ``UNGATED_BOUND`` seconds.

Base (the shared transaction, the ``PR #3423`` review-fix head): the writers
fail (``add_note``: "database is locked" at once; ``add_conversation``: held
past the bound), the faulted note keeps its link and intents, and the
receipts carry no reason. Base for the last two (``fcdebff32c``, a profile
resolution per note): four profile reads inside every note's transaction,
and the un-gated writer waits out the remaining notes (measured 4.1 s at 50
notes; at 120 the bulk outlasts the wait).
"""

from __future__ import annotations

import sqlite3
import sys
import threading
import time
from contextlib import contextmanager

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
from tldw_chatbook.Widgets.Library.library_note_folder_dialog import (
    LibraryNoteKeywordDialog,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: Each concurrent writer must succeed within this many seconds of the gate.
WRITER_BOUND = 3.0
#: The longest the gate holds the bulk while the writers run. Longer than the
#: bound, so a writer that can only finish after the bulk lets go fails it.
GATE_HOLD = 6.0
#: TASK-34000.116 AC #4: a writer started at any moment of a bulk run gets in
#: within this many seconds.
UNGATED_BOUND = 1.0
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


#: The bulk keyword worker's database closure: a read made with this frame on
#: the stack is the bulk's.
_BULK_WRITE = "_run_library_notes_bulk_keyword.<locals>.write"


class _ProfileReads:
    """Count the bulk's ``SyncStateRepository`` profile reads (TASK-34000.116).

    Wraps ``list_sync_v2_profile_states`` and ``get_sync_v2_profile_state``
    on the wired repository (the list's own per-row read is counted too), and
    ``note_transaction`` to know when the bulk's thread holds a note's write
    transaction. A read counts only when the bulk worker's ``write`` closure
    is on the stack, so the Library's own reads stay out of it.
    """

    def __init__(self, state, interop) -> None:
        self.inside = 0
        self.outside = 0
        self._holder: int | None = None
        for name in ("list_sync_v2_profile_states", "get_sync_v2_profile_state"):
            setattr(state, name, self._counted(getattr(state, name)))
        real_transaction = interop.note_transaction

        @contextmanager
        def note_transaction(user_id):
            with real_transaction(user_id) as cursor:
                self._holder = threading.get_ident()
                try:
                    yield cursor
                finally:
                    self._holder = None

        interop.note_transaction = note_transaction

    def _counted(self, method):
        def counted(*args, **kwargs):
            frame = sys._getframe(1)
            while frame is not None and not frame.f_code.co_qualname.endswith(
                _BULK_WRITE
            ):
                frame = frame.f_back
            if frame is not None:
                if self._holder == threading.get_ident():
                    self.inside += 1
                else:
                    self.outside += 1
            return method(*args, **kwargs)

        return counted


class _UngatedWriter:
    """Start ``add_conversation`` once note ``at`` has committed; never wait.

    ``note_transaction`` is wrapped to count the bulk's commits and time each
    one's hold (lock acquired to commit). The writer runs on a thread of its
    own (so its own connection), started from the commit of note ``at``
    without holding the bulk: it races whatever the bulk does next.
    """

    def __init__(self, db, interop, *, at: int) -> None:
        self.db = db
        self.at = at
        self.committed = 0
        self.holds: list[float] = []
        self.result: tuple[str, float] | None = None
        real_transaction = interop.note_transaction

        @contextmanager
        def note_transaction(user_id):
            with real_transaction(user_id) as cursor:
                acquired = time.monotonic()
                yield cursor
            self.holds.append(time.monotonic() - acquired)
            self.committed += 1
            if self.committed == self.at:
                threading.Thread(target=self._write, daemon=True).start()

        interop.note_transaction = note_transaction

    def _write(self) -> None:
        start = time.monotonic()
        try:
            self.db.add_conversation({"title": "Concurrent chat", "character_id": None})
        except Exception as exc:  # noqa: BLE001 - the pin reports it
            self.result = (f"{type(exc).__name__}: {exc}", time.monotonic() - start)
            return
        self.result = ("ok", time.monotonic() - start)


async def _bulk_add(host, pilot, screen, note_ids, keyword: str) -> None:
    """Add ``keyword`` to ``note_ids`` through the real dialog and worker.

    The tree renders one page of rows, so a selection larger than a page is
    handed to the controller's own entry -- the one the Actions chooser
    calls with the checked ids -- and the dialog takes it from there.
    """
    count = len(note_ids)
    screen._notes_controller._ask_library_notes_bulk_keyword(
        tuple(note_ids), remove=False
    )
    await until(
        pilot,
        lambda: isinstance(host.screen, LibraryNoteKeywordDialog)
        and f"Add a keyword to {count} notes" in painted(host),
        lambda: f"the keyword dialog for {count} notes: {painted(host)}",
    )
    await pilot.press(*keyword, "enter")
    await until(pilot, lambda: host.screen is screen, "the dialog to apply")


async def test_a_bulk_keyword_change_reads_the_profile_once_not_per_note(
    tmp_path,
) -> None:
    host, seed = mounted(tmp_path)
    db = seed.profile.db
    ids = [db.add_note(f"Bulk {index:02d}", "bulk body") for index in range(25)]
    state = _ready_sync_v2(seed, tmp_path)
    reads = _ProfileReads(state, seed.profile.interop)
    per_run: list[tuple[int, int, int]] = []
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        fenced = screen._notes_controller._library_notes_mutation_fenced
        for keyword, picked in (("travel", ids[:5]), ("food", ids)):
            await until(pilot, lambda: not fenced(), "the Notes writer to be free")
            inside, outside = reads.inside, reads.outside
            await _bulk_add(host, pilot, screen, picked, keyword)
            receipt = f"Added '{keyword}' to {len(picked)} notes"
            await until(
                pilot,
                lambda: receipt in str(screen._notes_state.notice),
                lambda: f"{receipt!r}: {screen._notes_state.notice!r}",
            )
            assert seed.keyword_links(keyword) == set(picked)
            per_run.append(
                (len(picked), reads.inside - inside, reads.outside - outside)
            )
    # No profile read while a note holds the write lock, at any size...
    assert [inside for _, inside, _ in per_run] == [0, 0], (
        f"(notes, profile reads inside note transactions, outside): {per_run}"
    )
    # ...and one resolution per bulk: the same reads for 5 notes as for 25.
    (_, _, small), (_, _, large) = per_run
    assert 0 < small == large, (
        f"(notes, profile reads inside note transactions, outside): {per_run}"
    )


@pytest.mark.parametrize("bulk", [50, 120])
async def test_a_writer_started_mid_run_gets_in_within_a_second(
    tmp_path, bulk: int
) -> None:
    """A BOUND, not a fairness guard (PR #3423 re-review 2, m-2).

    The un-gated ``add_conversation`` must succeed within ``UNGATED_BOUND``
    seconds. It may well wait out the rest of the run -- at 50 notes it did
    in 3 of 3 measured runs -- since the busy-timeout writer waits at most the short
    locked run. A few extra milliseconds per note pass this pin; the
    profile-read counting pin above is what guards the seam.
    """
    host, seed = mounted(tmp_path)
    db = seed.profile.db
    ids = [db.add_note(f"Bulk {index:03d}", "bulk body") for index in range(bulk)]
    _ready_sync_v2(seed, tmp_path)
    writer = _UngatedWriter(db, seed.profile.interop, at=bulk // 2)
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await _bulk_add(host, pilot, screen, ids, "travel")
        # The writer reports before anything else is judged: on the old code
        # it waits out the remaining notes, and the receipt would time out
        # first.
        await until(
            pilot,
            lambda: writer.result is not None,
            lambda: f"the mid-run writer to return ({writer.committed} committed)",
        )
        holds = sorted(writer.holds)
        outcome, seconds = writer.result
        assert outcome == "ok" and seconds <= UNGATED_BOUND, (
            f"add_conversation started after note {writer.at} of {bulk} "
            f"committed: {outcome} after {seconds:.2f} s (bound {UNGATED_BOUND} "
            f"s); median note hold {holds[len(holds) // 2] * 1000:.0f} ms over "
            f"{len(holds)} notes so far"
        )
        receipt = f"Added 'travel' to {bulk} notes"
        await until(
            pilot,
            lambda: receipt in str(screen._notes_state.notice),
            lambda: f"the add receipt: {screen._notes_state.notice!r}",
        )
    assert writer.committed == bulk
    # The bulk itself is whole: every note linked, one link intent per note,
    # one resource for the new keyword (TASK-34000.116 AC #2).
    assert seed.keyword_links("travel") == set(ids)
    intents = _intents(seed)
    assert sorted(row[2] for row in intents if row[0] == "notes.keyword_link") == (
        sorted(ids)
    )
    assert [row for row in intents if row[0] == "notes.keyword"] == [
        ("notes.keyword", "upsert", "travel")
    ]
    with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM conversations WHERE title = 'Concurrent chat'"
        ).fetchone() == (1,)


# -- PR #3423 Qodo 3 and 4: no half-written note, no unused keyword ----------
#
# Without Sync v2 (the plain local path, ``_sync_local_note_keywords``), a
# database failure injected by a trigger in the test's own database file, so
# it fires on the worker thread's own connection too. Base (``bd01c0e46b``):
# ONE ``note_transaction`` around every note, each call's exception caught --
# a link whose ``note_keywords`` row was written before its ``sync_log`` row
# failed was committed (Qodo 3), and the keyword was created before any link
# was tried (Qodo 4).


def _fault(tmp_path, ddl: str) -> None:
    with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
        connection.execute(ddl)


def _sync_log_link_rows(tmp_path, note_id: str) -> int:
    with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM sync_log WHERE entity = 'note_keywords' "
            "AND substr(entity_id, 1, length(?) + 1) = ? || '_'",
            (note_id, note_id),
        ).fetchone()[0]


async def test_a_link_whose_sync_log_write_fails_commits_nothing_for_that_note(
    tmp_path,
) -> None:
    """Qodo 3: the note's link row is written, then its sync-log row raises.

    The note's own transaction rolls back whole: no link and no sync-log row
    for it, while the other two notes are linked and logged; the receipt
    counts it as not changed, and it stays checked.
    """
    host, seed = mounted(tmp_path)
    faulted = seed.note_ids["Reading list"]
    others = {seed.note_ids[title] for title in PICKED} - {faulted}
    # A uuid, safe to inline: trigger DDL takes no bound parameters.
    _fault(
        tmp_path,
        "CREATE TRIGGER qodo3_sync_log_fault BEFORE INSERT ON sync_log "
        "WHEN NEW.entity = 'note_keywords' "
        f"AND substr(NEW.entity_id, 1, {len(faulted)}) = '{faulted}' "
        "BEGIN SELECT RAISE(ABORT, 'injected sync-log failure'); END",
    )
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
        receipt = (
            "Added 'travel' to 2 notes · 1 could not be changed — still "
            "checked, so you can try again"
        )
        await until(
            pilot,
            lambda: screen._notes_state.notice == receipt,
            lambda: f"the partial receipt: {screen._notes_state.notice!r}",
        )
        assert screen._notes_state.row_selection.count == 1
        assert screen._notes_state.row_selection.is_selected(faulted)
    assert seed.keyword_links("travel") == others
    assert _sync_log_link_rows(tmp_path, faulted) == 0
    assert all(_sync_log_link_rows(tmp_path, note_id) == 1 for note_id in others)


async def test_a_new_keyword_that_no_note_takes_leaves_no_keyword_row(
    tmp_path,
) -> None:
    """Qodo 4: every selected note refused, so the new keyword is not created.

    The keyword is created inside each note's own transaction, after the
    policy gate, so a note whose link is refused rolls its keyword back with
    it: no keyword row, no link, and every note still checked.
    """
    host, seed = mounted(tmp_path)
    _fault(
        tmp_path,
        "CREATE TRIGGER qodo4_link_refused BEFORE INSERT ON note_keywords "
        "BEGIN SELECT RAISE(ABORT, 'injected link refusal'); END",
    )
    before = seed.keyword_state()
    async with host.run_test(size=WIDE) as pilot:
        screen = await notes_ready(host, pilot)
        await enter_select_mode(host, pilot, screen)
        await _check(host, pilot, screen, PICKED)
        await _keyword(
            host,
            pilot,
            screen,
            "library-notes-select-action-add-keyword",
            "lagoon",
            "Add a keyword to 3 notes",
        )
        receipt = (
            "Added 'lagoon' to 0 notes · 3 could not be changed — still "
            "checked, so you can try again"
        )
        await until(
            pilot,
            lambda: screen._notes_state.notice == receipt,
            lambda: f"the refused receipt: {screen._notes_state.notice!r}",
        )
        assert screen._notes_state.row_selection.count == 3
    assert seed.keyword_state() == before
    with sqlite3.connect(tmp_path / "notes.sqlite3") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM keywords WHERE keyword = 'lagoon' COLLATE NOCASE"
        ).fetchone() == (0,)
