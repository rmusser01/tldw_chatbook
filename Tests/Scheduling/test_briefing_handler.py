"""Tests for the scheduled briefing generation handler (briefings phase 4,
task 3).

Locked Decision 3: `SchedulerLoop.tick` awaits every handler serially,
inline -- a briefing generation is a multi-minute LLM call, so
`BriefingJobHandler.handle` must claim-check and spawn, never await the
generation itself. The non-blocking tests below prove this structurally
(ordering through a test-controlled gate), never through a sleep or a
wall-clock timeout -- a handler that regressed to awaiting inline would
still "pass" a sleep-based test if the sleep were long enough, which is
exactly the failure mode a gate avoids.

`generate_briefing`/`GenerationInFlightError`/`active_briefing_claims` are
the real briefings phase 4 task 1 objects -- the claim registry genuinely
is shared, in-process, module-level state (Locked Decision 1), so the
claimed-skip tests below use the real `_claim_briefing` context manager
rather than faking claim state, the same way
`Tests/Subscriptions/test_briefing_service.py` does for its own claim
tests.
"""

from __future__ import annotations

import asyncio
import functools
import threading
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler import (
    BriefingJobHandler,
)
from tldw_chatbook.Subscriptions import briefing_service
from tldw_chatbook.Subscriptions.briefing_keep import KeepRefused
from tldw_chatbook.Subscriptions.briefing_service import (
    GenerationInFlightError,
    generate_briefing,
)
from tldw_chatbook.Subscriptions.item_persist import persist_subscription_item
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


def _task(watchlist_id: int | str = 7) -> dict[str, Any]:
    return {
        "id": f"briefing:{watchlist_id}",
        "title": "Some Watchlist",
        "type": "briefing_job",
        "status": "waiting",
        "next_run_at": None,
        "owner_id": "local",
    }


def _db_with_default_preset(preset_id: int | None):
    """A MagicMock standing in for `SubscriptionsDB`, answering exactly the
    one query `_default_preset_id` makes.

    `_default_preset_id` reads through `with subscriptions_db.transaction()
    as conn:` (Qodo rule 1011851), not `subscriptions_db.conn.execute`
    directly -- `transaction()` is wired here to hand back `db.conn`
    itself, mirroring the real `SubscriptionsDB.transaction`, which yields
    `self.conn`. Without this, `conn` inside the handler's `with` block
    would resolve to an unrelated, unconfigured child mock instead of the
    one `db.conn.execute` was set up on above.
    """
    db = MagicMock()
    db.conn.execute.return_value.fetchone.return_value = (
        {"default_briefing_preset_id": preset_id} if preset_id is not None else None
    )
    db.transaction.return_value.__enter__.return_value = db.conn
    return db


async def _drain(handler: BriefingJobHandler) -> None:
    """Await every currently-pending spawned generation to completion.

    `_run_generation` never raises (that is the whole point of the
    containment it implements), so `asyncio.gather` here would only ever
    re-raise if containment had a hole in it -- which is exactly what the
    containment tests below rely on.
    """
    pending = list(handler._pending_generations)
    if pending:
        await asyncio.gather(*pending)


# --- non-blocking: the tick must not stall on a slow generation -------------


@pytest.mark.asyncio
async def test_handle_returns_before_a_slow_generation_has_even_started():
    """Proof by ordering, not by timing: `create_task` schedules the
    coroutine but does not run any of its body until the event loop is
    given a chance to -- so if `handle` returns without awaiting anything
    itself, the spawned coroutine cannot have executed even its first line
    yet."""
    order: list[str] = []
    gate = asyncio.Event()

    async def slow_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        order.append("generate-start")
        await gate.wait()
        order.append("generate-end")
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=slow_generate
    )

    await handler.handle(_task(7))
    order.append("handle-returned")

    # If `handle` awaited the generation inline, "generate-start" (and,
    # since the gate is never set here, a hang) would appear before
    # "handle-returned" ever could.
    assert order == ["handle-returned"]
    assert len(handler._pending_generations) == 1

    gate.set()
    await _drain(handler)

    assert order == ["handle-returned", "generate-start", "generate-end"]
    assert handler._pending_generations == set()


@pytest.mark.asyncio
async def test_handle_passes_the_watchlists_default_preset_id():
    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append((watchlist_id, preset_id))
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(42), generate=fake_generate
    )

    await handler.handle(_task(9))
    await _drain(handler)

    assert calls == [(9, 42)]


@pytest.mark.asyncio
async def test_handle_passes_none_when_the_watchlist_has_no_default_preset():
    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append((watchlist_id, preset_id))
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    await handler.handle(_task(9))
    await _drain(handler)

    assert calls == [(9, None)]


@pytest.mark.asyncio
async def test_default_preset_id_read_runs_off_the_event_loop_thread():
    """Review round 1: `_default_preset_id`'s raw SQLite read must run via
    `asyncio.to_thread`, never directly on the event loop.
    `SubscriptionsDB` sets no `busy_timeout` (SQLite's 5s default applies),
    and this handler's own spawned generations write to the very same
    `watchlists`/`briefings` connection from `to_thread` workers -- a
    direct, synchronous read here could block on a lock its own spawned
    work is holding, self-inflicting the tick stall Locked Decision 3
    exists to prevent.

    Same technique as
    `test_briefing_service.py::test_the_db_work_runs_off_the_event_loop_thread`:
    a mutation that drops `asyncio.to_thread` and reads directly passes
    every other test in this file unchanged (the end state -- the fake
    generate call receiving `preset_id=5` -- is identical either way); only
    watching which thread actually executed the read can tell the two
    apart.

    `db.transaction()` is wired to hand back `db.conn` on `__enter__` --
    see `_db_with_default_preset`'s docstring for why: `_default_preset_id`
    reads through `with subscriptions_db.transaction() as conn:` (Qodo
    rule 1011851), so the spy has to sit behind that same seam to observe
    anything.
    """
    loop_thread_id = threading.get_ident()
    read_thread_ids: list[int] = []

    db = MagicMock()

    def _spy_execute(*args, **kwargs):
        read_thread_ids.append(threading.get_ident())
        cursor = MagicMock()
        cursor.fetchone.return_value = {"default_briefing_preset_id": 5}
        return cursor

    db.conn.execute.side_effect = _spy_execute
    db.transaction.return_value.__enter__.return_value = db.conn

    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append(preset_id)
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(subscriptions_db=db, generate=fake_generate)

    await handler.handle(_task(7))
    await _drain(handler)

    assert calls == [5]  # the read genuinely happened and reached generate()
    assert len(read_thread_ids) == 1
    assert read_thread_ids[0] != loop_thread_id, (
        "_default_preset_id must run via asyncio.to_thread, not directly on "
        "the event loop"
    )


# --- malformed id -------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_ignores_a_malformed_task_id():
    called = False

    async def fake_generate(*args, **kwargs):
        nonlocal called
        called = True
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    await handler.handle({"id": "watchlist:9"})
    await _drain(handler)

    assert called is False
    assert handler._pending_generations == set()


# --- claimed watchlist: skip, do not queue a second attempt -----------------


@pytest.mark.asyncio
async def test_a_claimed_watchlist_is_skipped_and_never_generated():
    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append(watchlist_id)
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    with briefing_service._claim_briefing(7):
        await handler.handle(_task(7))
        await _drain(handler)

    assert calls == []
    assert handler._pending_generations == set()


@pytest.mark.asyncio
async def test_an_unclaimed_watchlist_is_generated_normally():
    """The claim check must not refuse everything -- only the claimed one."""
    calls = []

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        calls.append(watchlist_id)
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    with briefing_service._claim_briefing(999):  # a different watchlist
        await handler.handle(_task(7))
        await _drain(handler)

    assert calls == [7]


# --- containment: failures inside the spawned task must never escape -------


@pytest.mark.asyncio
async def test_generation_in_flight_error_is_contained_and_logged_at_debug():
    """Losing the claim race between the handler's snapshot check and
    `generate_briefing`'s own atomic check-then-add is harmless -- it must
    not propagate, and (what this test's name claims, now actually pinned)
    it must log at DEBUG rather than a louder level that would page someone
    for a race this handler already treats as routine.

    No `briefings` row is asserted here: `GenerationInFlightError` is raised
    by `_claim_briefing` BEFORE `generate_briefing` ever inserts a row
    (Task 1's pre-row-insert contract) -- there is nothing to write for
    THIS specific failure. `test_a_real_provider_failure_writes_a_failed_row_end_to_end`
    below is the test that carries the actual failed-row claim, through the
    real service.
    """

    async def racing_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        raise GenerationInFlightError(f"already claimed: {watchlist_id}")

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=racing_generate
    )

    debug_lines: list[str] = []
    warning_or_louder: list[str] = []
    debug_sink = logger.add(debug_lines.append, level="DEBUG", catch=False)
    warning_sink = logger.add(warning_or_louder.append, level="WARNING", catch=False)
    try:
        await handler.handle(_task(7))
        await _drain(handler)  # would raise if the exception were not contained
    finally:
        logger.remove(debug_sink)
        logger.remove(warning_sink)

    assert handler._pending_generations == set()
    assert any("lost the generation claim race" in line for line in debug_lines)
    # Genuinely DEBUG, not merely ALSO logged at debug: a sink that only
    # forwards WARNING and louder must see nothing from this path.
    assert warning_or_louder == []


@pytest.mark.asyncio
async def test_an_unexpected_exception_inside_the_spawned_task_does_not_propagate():
    """Wrapper-only containment: a database error escaping `generate_briefing`
    (documented as possible in its own docstring, for any `to_thread` call
    outside the chat try/except) must not become an unhandled task
    exception. The fake `generate` here never touches a DB, so there is no
    `briefings` row to assert -- this test's whole claim is "the exception
    does not propagate", nothing more. See
    `test_a_real_provider_failure_writes_a_failed_row_end_to_end` below for
    the row-level claim, through the real service.
    """
    import sqlite3

    async def boom_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=boom_generate
    )

    await handler.handle(_task(7))
    await _drain(handler)  # would raise (via asyncio.gather) if uncontained

    assert handler._pending_generations == set()


@pytest.mark.asyncio
async def test_a_real_provider_failure_writes_a_failed_row_end_to_end(tmp_path):
    """The actual failed-row claim `_run_generation`'s docstring makes ("the
    failed briefing row already records it -- service behaviour"), proven
    through the REAL `generate_briefing` -- not a fake standing in for it --
    with only the `chat` seam faked, this project's "fake exactly three
    seams" testing rule
    (`Tests/Subscriptions/test_briefing_service.py`'s own module docstring).
    Mirrors `test_llm_failure_is_honest_and_loses_nothing`
    (`test_briefing_service.py:253`) end to end through the scheduler path
    rather than only through `generate_briefing` called directly.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = WatchlistBundleService(db).create(name="Flaky Provider")["id"]
    source_id = db.add_subscription(
        name="acme", type="rss", source="https://acme.example/feed.xml"
    )
    WatchlistBundleService(db).add_source(watchlist_id, source_id)
    with db.transaction() as conn:
        persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://items.example/acme/1",
                "title": "Something Happened",
                "content": "body of something",
                "content_hash": "hash-1",
                "content_kind": "article",
                "content_format": "text",
            },
            run_id=None,
            now=datetime.now(timezone.utc).isoformat(),
        )

    def boom_chat(**kwargs):
        raise RuntimeError("provider exploded: 503 upstream")

    handler = BriefingJobHandler(
        subscriptions_db=db,
        generate=functools.partial(generate_briefing, chat=boom_chat),
    )

    await handler.handle(_task(watchlist_id))
    await _drain(handler)

    rows = db.list_briefings(watchlist_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert "provider exploded: 503 upstream" in rows[0]["error"]
    assert handler._pending_generations == set()


# --- metrics: the completion label must name what it measures ---------------


@pytest.mark.asyncio
async def test_successful_generation_emits_a_completed_status_not_dispatched():
    """Review round 1: the metric label previously read `"dispatched"` but
    fired only in `_run_generation`'s `finally`, at COMPLETION -- an
    in-flight generation, or one whose process is killed before `finally`
    runs, was never counted at all, so that name claimed the opposite of
    what it measured. Pins the corrected label."""

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        return {"id": 1, "status": "complete"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )

    with patch(
        "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.log_counter"
    ) as counter:
        await handler.handle(_task(7))
        await _drain(handler)

    counter.assert_called_once_with(
        "briefing_schedule_runs", labels={"status": "completed"}
    )


# --- empty end to end: the real service, a real DB, no chat call ------------


@pytest.mark.asyncio
async def test_empty_result_writes_an_empty_row_end_to_end(tmp_path):
    """The spec's "empty rows when nothing is new" invariant, pinned all the
    way through the scheduler path: a real `SubscriptionsDB`, the real
    (un-injected) `generate_briefing`, and a watchlist with nothing to
    brief. No `chat` seam is faked because the empty path never calls one
    -- `generate_briefing` returns before the chat step when selection
    finds no items, so this is a genuine, network-free end-to-end path.

    File-backed, not `:memory:`, for the same reason
    `test_briefing_service.py`'s own `_db` helper is: `generate_briefing`
    offloads its DB work to `asyncio.to_thread`, and `SubscriptionsDB`
    connections are thread-local, so an in-memory DB would be invisible on
    the worker thread.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = WatchlistBundleService(db).create(name="Quiet")["id"]
    db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

    handler = BriefingJobHandler(subscriptions_db=db)  # real generate_briefing

    await handler.handle(_task(watchlist_id))
    await _drain(handler)

    rows = db.list_briefings(watchlist_id)
    assert len(rows) == 1
    assert rows[0]["status"] == "empty"
    assert rows[0]["error"] is None


# --- Task 3: auto-keep on scheduled completion -------------------------------
#
# `BriefingJobHandler` gains an optional `chachanotes_db_getter`; once a
# spawned generation resolves `complete`, `_run_generation` mirrors it into
# ChaChaNotes via `briefing_keep.keep_briefing(..., origin="scheduled")`
# (spec: Keep-service "Auto path"). Every real DB here is file-backed at
# `tmp_path`, never `:memory:` and never the live user data directory --
# matching both this file's own convention above and
# `Tests/Subscriptions/test_briefing_keep.py`'s.
#
# Review round 1: the handle is a GETTER, resolved fresh inside `_auto_keep`
# every time, not a plain instance captured once at construction -- `app.py`
# builds this handler before its own `self.chachanotes_db` attribute even
# exists, so capturing the instance directly would freeze `None` into the
# handler forever. Most tests below pass `lambda: chacha_db` (a closure over
# an already-real db, standing in for "the attribute already resolved by
# call time"); `test_a_late_bound_chachanotes_handle_is_still_auto_kept`
# below is the one that specifically proves resolution happens at CALL time,
# not at construction time.


def _chacha_db(tmp_path) -> CharactersRAGDB:
    """A real, file-backed ChaChaNotes handle for auto-keep assertions.

    Mirrors `Tests/Subscriptions/test_briefing_keep.py`'s own `_chacha_db`
    helper.
    """
    return CharactersRAGDB(tmp_path / "chacha.sqlite", client_id="briefing-handler-test")


def _seed_complete_briefing(
    db: SubscriptionsDB,
    watchlist_id: int,
    *,
    body: str = "# Digest\n\nSomething happened.\n",
) -> int:
    """A real `complete` `briefings` row, seeded directly (bypassing
    generation) so a test can drive the handler's auto-keep wiring through
    a fake `generate` without a real or faked chat call. Mirrors
    `Tests/Subscriptions/test_briefing_keep.py::_complete_briefing`.
    """
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(briefing_id, status="complete", body_markdown=body)
    return briefing_id


def _canned_chat(**kwargs) -> str:
    """A stand-in for `chat_api_call` that always succeeds with a
    non-empty reply -- the one faked seam, same convention as
    `test_briefing_service.py`'s own `_FakeChat`."""
    return "# Weekly Digest\n\nAcme shipped a new thing this week.\n"


@pytest.mark.asyncio
async def test_a_complete_scheduled_generation_is_auto_kept_with_scheduled_origin(
    tmp_path,
):
    """Task 3's headline path, through the REAL `generate_briefing` (only
    `chat` faked, this stream's "fake exactly three seams" rule): a
    scheduled generation that resolves `complete` is mirrored into
    ChaChaNotes with `origin="scheduled"`, unprompted -- the spec's
    Keep-service "Auto path"."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = WatchlistBundleService(db).create(name="Acme Watch")["id"]
        source_id = db.add_subscription(
            name="acme", type="rss", source="https://acme.example/feed.xml"
        )
        WatchlistBundleService(db).add_source(watchlist_id, source_id)
        with db.transaction() as conn:
            persist_subscription_item(
                conn,
                source_id,
                {
                    "url": "https://items.example/acme/1",
                    "title": "Something Happened",
                    "content": "body of something",
                    "content_hash": "hash-1",
                    "content_kind": "article",
                    "content_format": "text",
                },
                run_id=None,
                now=datetime.now(timezone.utc).isoformat(),
            )

        handler = BriefingJobHandler(
            subscriptions_db=db,
            generate=functools.partial(generate_briefing, chat=_canned_chat),
            chachanotes_db_getter=lambda: chacha_db,
        )

        await handler.handle(_task(watchlist_id))
        await _drain(handler)

        rows = db.list_briefings(watchlist_id)
        assert len(rows) == 1
        assert rows[0]["status"] == "complete"
        briefing_id = rows[0]["id"]

        kept = chacha_db.list_kept_briefings()
        assert len(kept) == 1
        assert kept[0]["source_briefing_id"] == briefing_id
        assert kept[0]["origin"] == "scheduled"
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_auto_keep_skips_empty_scheduled_results(tmp_path):
    """Named invariant (plan Task 3 / spec): auto-keep must never mirror
    an `empty` scheduled row. `_auto_keep` branches on the row's own
    `status` before ever touching a thread hop or ChaChaNotes, so an
    empty window never reaches `keep_briefing` at all -- through the REAL
    `generate_briefing`, exactly like this file's own
    `test_empty_result_writes_an_empty_row_end_to_end` above, plus the
    ChaChaNotes assertion Task 3 adds."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = WatchlistBundleService(db).create(name="Quiet Watch")["id"]
        db.set_watchlist_briefing_settings(watchlist_id, briefing_cadence_seconds=3600)

        handler = BriefingJobHandler(
            subscriptions_db=db, chachanotes_db_getter=lambda: chacha_db
        )

        await handler.handle(_task(watchlist_id))
        await _drain(handler)

        rows = db.list_briefings(watchlist_id)
        assert len(rows) == 1
        assert rows[0]["status"] == "empty"
        assert chacha_db.list_kept_briefings() == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_an_auto_keep_failure_never_escapes_and_leaves_the_row_untouched(
    tmp_path,
):
    """Containment discipline extended to auto-keep: an exception escaping
    `keep_briefing` (anything other than its own `KeepRefused`) must never
    reach the spawned task -- `asyncio.gather` in `_drain` would raise if
    it did -- must never flip `_run_generation`'s own metric away from
    `"completed"` (the GENERATION did not fail; only the best-effort
    mirror did), and must never touch the `briefings` row, pinned by full
    equality rather than just a status check.

    Mutation check performed by hand (Edit-revert cycle, not committed):
    removing `_auto_keep`'s outer `except Exception` -- or widening the
    inner `except KeepRefused` to a bare `except Exception` so it
    swallows this `RuntimeError` too before the *outer* handler notices --
    both reproduce a real regression; the first is caught by `_drain`
    re-raising, the second by `counter.assert_called_once_with(...,
    labels={"status": "completed"})` still passing when it should (a
    swallow-everywhere mutation does not change the metric, so that
    assertion alone would not catch it -- the `_drain` re-raise is what
    catches it). Both were verified RED against the un-mutated fix and
    restored; `git status --short` was clean before and after.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = WatchlistBundleService(db).create(name="Acme Watch")["id"]
        briefing_id = _seed_complete_briefing(db, watchlist_id)
        row_before = db.get_briefing(briefing_id)

        async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
            return dict(row_before)

        handler = BriefingJobHandler(
            subscriptions_db=db,
            generate=fake_generate,
            chachanotes_db_getter=lambda: chacha_db,
        )

        with (
            patch(
                "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.keep_briefing",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.log_counter"
            ) as counter,
        ):
            await handler.handle(_task(watchlist_id))
            await _drain(handler)  # would raise (via asyncio.gather) if uncontained

        counter.assert_called_once_with(
            "briefing_schedule_runs", labels={"status": "completed"}
        )
        assert db.get_briefing(briefing_id) == row_before
        assert chacha_db.list_kept_briefings() == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_a_keep_refused_race_is_treated_as_benign_not_an_error(tmp_path):
    """The belt-and-braces branch: `_auto_keep`'s own local status check
    already keeps `empty`/`failed` rows from ever reaching `keep_briefing`,
    but the keep service refuses independently too (its own re-read can
    legitimately disagree, e.g. a genuine race) -- and that must land as
    the same silent, DEBUG-only no-op as any other skip, not the louder
    WARNING path a truly unexpected keep failure gets. Patches
    `keep_briefing` directly to raise `KeepRefused`, rather than
    engineering a real race, mirroring how
    `test_an_auto_keep_failure_never_escapes_and_leaves_the_row_untouched`
    faked an arbitrary failure -- the two tests together pin both of
    `_auto_keep`'s `except` branches."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = WatchlistBundleService(db).create(name="Acme Watch")["id"]
        briefing_id = _seed_complete_briefing(db, watchlist_id)
        row_before = db.get_briefing(briefing_id)

        async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
            return dict(row_before)

        handler = BriefingJobHandler(
            subscriptions_db=db,
            generate=fake_generate,
            chachanotes_db_getter=lambda: chacha_db,
        )

        debug_lines: list[str] = []
        warning_or_louder: list[str] = []
        debug_sink = logger.add(debug_lines.append, level="DEBUG", catch=False)
        warning_sink = logger.add(warning_or_louder.append, level="WARNING", catch=False)
        try:
            with patch(
                "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.keep_briefing",
                side_effect=KeepRefused("raced"),
            ):
                await handler.handle(_task(watchlist_id))
                await _drain(handler)  # would raise if uncontained
        finally:
            logger.remove(debug_sink)
            logger.remove(warning_sink)

        assert any("Auto-keep refused" in line for line in debug_lines)
        assert warning_or_louder == []
        assert db.get_briefing(briefing_id) == row_before
        assert chacha_db.list_kept_briefings() == []
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_no_chachanotes_handle_skips_keep_without_crashing():
    """`chachanotes_db_getter=None` -- the default -- must never crash the
    handler and must never attempt a keep; generation still completes
    normally. `test_a_getter_returning_none_at_call_time_skips_keep_
    without_crashing` below covers the sibling case: a getter that IS
    configured but itself returns `None` when called."""

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        return {"id": 1, "status": "complete", "body_markdown": "irrelevant"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )
    assert handler._chachanotes_db_getter is None

    with (
        patch(
            "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.keep_briefing"
        ) as keep_mock,
        patch(
            "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.log_counter"
        ) as counter,
    ):
        await handler.handle(_task(7))
        await _drain(handler)

    keep_mock.assert_not_called()
    counter.assert_called_once_with(
        "briefing_schedule_runs", labels={"status": "completed"}
    )


@pytest.mark.asyncio
async def test_a_getter_returning_none_at_call_time_skips_keep_without_crashing():
    """A `chachanotes_db_getter` IS configured (unlike the sibling test
    above), but returns `None` when actually called -- the genuinely
    "no handle available right now" case `_auto_keep`'s own docstring
    documents. Must skip cleanly, exactly like no getter at all."""

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        return {"id": 1, "status": "complete", "body_markdown": "irrelevant"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None),
        generate=fake_generate,
        chachanotes_db_getter=lambda: None,
    )

    with (
        patch(
            "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.keep_briefing"
        ) as keep_mock,
        patch(
            "tldw_chatbook.Scheduling.scheduler.handlers.briefing_handler.log_counter"
        ) as counter,
    ):
        await handler.handle(_task(7))
        await _drain(handler)

    keep_mock.assert_not_called()
    counter.assert_called_once_with(
        "briefing_schedule_runs", labels={"status": "completed"}
    )


@pytest.mark.asyncio
async def test_a_late_bound_chachanotes_handle_is_still_auto_kept(tmp_path):
    """Review round 1's headline liveness proof: a getter that would have
    returned `None` if called AT CONSTRUCTION TIME, but resolves to a real
    ChaChaNotes handle by the time the generation actually completes,
    still gets auto-kept -- exactly `app.py`'s real shape, where
    `self.chachanotes_db` does not exist yet when
    `BriefingJobHandler` is built, but does exist by the time any
    scheduled job actually fires later in the process's life.

    This is the test that REDs against the OLD (pre-review-round-1)
    instance-param wiring: capturing `chachanotes_db` once at construction
    would have frozen in whatever `holder.db` was at `BriefingJobHandler(
    ...)` time -- `None` -- and no later mutation of `holder.db` could
    ever reach it. The getter, resolved fresh inside `_auto_keep`, sees
    whatever `holder.db` has become by the time the generation completes.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = WatchlistBundleService(db).create(name="Acme Watch")["id"]
        briefing_id = _seed_complete_briefing(db, watchlist_id)
        row = db.get_briefing(briefing_id)

        class _LateBoundHolder:
            """Stands in for `self` in `app.py`: `.db` is `None` at the
            moment the handler is constructed, mutated to the real
            ChaChaNotes handle afterward -- before the generation
            actually completes -- exactly mirroring `self.chachanotes_db`
            being assigned later in `TldwCli.__init__`, strictly after
            `BriefingJobHandler` is already built."""

            db: CharactersRAGDB | None = None

        holder = _LateBoundHolder()
        assert holder.db is None  # the state at construction time

        async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
            # By the time generation "completes", the attribute has been
            # assigned -- simulating `__init__` continuing past the point
            # where the scheduler wiring block ran.
            holder.db = chacha_db
            return dict(row)

        handler = BriefingJobHandler(
            subscriptions_db=db,
            generate=fake_generate,
            chachanotes_db_getter=lambda: holder.db,
        )

        await handler.handle(_task(watchlist_id))
        await _drain(handler)

        kept = chacha_db.list_kept_briefings()
        assert len(kept) == 1
        assert kept[0]["source_briefing_id"] == briefing_id
        assert kept[0]["origin"] == "scheduled"
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_a_second_scheduled_run_auto_keeps_the_new_briefing_too(tmp_path):
    """No idempotency confusion across DISTINCT briefings for the same
    watchlist: two scheduled runs, each producing its own `complete` row,
    must each be auto-kept. `keep_briefing`'s additive-idempotency (Task
    2) is about RE-keeping the SAME `source_briefing_id`; it must not be
    mistaken here for collapsing two genuinely different briefings into
    one kept row."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    chacha_db = _chacha_db(tmp_path)
    try:
        watchlist_id = WatchlistBundleService(db).create(name="Acme Watch")["id"]
        first_id = _seed_complete_briefing(
            db, watchlist_id, body="# Digest 1\n\nFirst week.\n"
        )
        second_id = _seed_complete_briefing(
            db, watchlist_id, body="# Digest 2\n\nSecond week.\n"
        )
        rows_by_call = iter([db.get_briefing(first_id), db.get_briefing(second_id)])

        async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
            return dict(next(rows_by_call))

        handler = BriefingJobHandler(
            subscriptions_db=db,
            generate=fake_generate,
            chachanotes_db_getter=lambda: chacha_db,
        )

        await handler.handle(_task(watchlist_id))
        await _drain(handler)
        await handler.handle(_task(watchlist_id))
        await _drain(handler)

        kept = chacha_db.list_kept_briefings()
        assert len(kept) == 2
        kept_source_ids = {row["source_briefing_id"] for row in kept}
        assert kept_source_ids == {first_id, second_id}
        assert {row["origin"] for row in kept} == {"scheduled"}
    finally:
        chacha_db.close_connection()


@pytest.mark.asyncio
async def test_the_real_app_wiring_getter_reads_chachanotes_db_live_not_frozen_at_boot():
    """Review round 1's production-liveness proof, through `app.py`'s
    REAL wiring (`_wire_watchlists_and_notifications_services`), not a
    reimplementation of it -- the other tests in this section prove
    `BriefingJobHandler` itself resolves the getter lazily; this one
    proves the specific closure `app.py` builds does too.

    `Tests/UI/app_factory._build_test_app` stubs `self.notes_service` to
    `None` and `get_chachanotes_db_lazy` to return `None` for every test
    app it builds (by design, for speed and hermeticity across the 90+
    modules that share it) -- so `app.chachanotes_db` really is `None`
    immediately after boot in this harness, honestly. Asserting it
    resolves non-`None` here would therefore be dishonest, not a genuine
    liveness proof (the premise this test's name would otherwise imply).
    What CAN be proven honestly, using only the real wiring code the app
    actually ran: the getter `app.py` built
    (`lambda: getattr(self, "chachanotes_db", None)`) reads
    `self.chachanotes_db` FRESH on every call, not whatever it captured at
    `BriefingJobHandler(...)` construction time. Mutating
    `app.chachanotes_db` after boot and calling the SAME getter again
    demonstrates exactly that: the old (pre-review-round-1) instance-param
    wiring would still report `None` here no matter what `app.chachanotes_db`
    became afterward.
    """
    from Tests.UI.app_factory import _build_test_app

    app = _build_test_app()
    async with app.run_test(size=(120, 40)):
        # The harness's own honest starting state -- see the docstring above
        # for why this is stubbed to None rather than a real handle.
        assert app.chachanotes_db is None

        handler = app.scheduler_loop.handlers["briefing_job"]
        getter = handler._chachanotes_db_getter
        assert getter is not None
        assert getter() is None  # agrees with app.chachanotes_db right now

        sentinel = object()
        app.chachanotes_db = sentinel  # the real, later __init__ assignment, simulated

        assert getter() is sentinel  # the SAME getter now reads the new value


# --- Task 3 (daily reports): completion notifications -------------------------
#
# `_run_generation` auto-keeps completed briefings but never told the user
# anything; these tests pin the optional `dispatch_service` /
# `notification_app_getter` collaborators (same optional-collaborator
# discipline as `chachanotes_db_getter`, and the same `ReminderHandler`
# dispatch seam): one `category="briefing"` notification per generation
# completion/failure, nothing on a claim-race skip, and silence when no
# dispatch service is configured.


class _DispatchSpy:
    """Records dispatch kwargs; mirrors NotificationDispatchService.dispatch."""

    def __init__(self):
        self.calls: list[dict] = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return {"persisted": True}


def _seeded_watchlist(db: SubscriptionsDB, name: str = "Daily Brief") -> int:
    """A watchlist whose next generation can resolve `complete`/`failed`.

    Plan deviation (documented in task-3-report.md): the brief's tests
    created a bare watchlist, but `generate_briefing` returns `empty`
    BEFORE the chat seam is ever invoked when selection finds no items --
    so `complete`/`failed` (and therefore the `"information"`/`"warning"`
    severities the brief's assertions pin) are unreachable without a
    source carrying one item. Same seeding shape as this file's own
    `test_a_real_provider_failure_writes_a_failed_row_end_to_end`.
    """
    watchlist_id = int(WatchlistBundleService(db).create(name)["id"])
    source_id = db.add_subscription(
        name="acme", type="rss", source="https://acme.example/feed.xml"
    )
    WatchlistBundleService(db).add_source(watchlist_id, source_id)
    with db.transaction() as conn:
        persist_subscription_item(
            conn,
            source_id,
            {
                "url": "https://items.example/acme/1",
                "title": "Something Happened",
                "content": "body of something",
                "content_hash": "hash-1",
                "content_kind": "article",
                "content_format": "text",
            },
            run_id=None,
            now=datetime.now(timezone.utc).isoformat(),
        )
    return watchlist_id


def _notify_handler(db, spy, *, generate=None, app_marker=object()):
    return BriefingJobHandler(
        subscriptions_db=db,
        generate=generate or functools.partial(generate_briefing, chat=_canned_chat),
        chachanotes_db_getter=lambda: None,
        dispatch_service=spy,
        notification_app_getter=lambda: app_marker,
    )


@pytest.mark.asyncio
async def test_complete_scheduled_generation_dispatches_briefing_notification(tmp_path):
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = _seeded_watchlist(db)
    spy = _DispatchSpy()
    app_marker = object()

    handler = _notify_handler(db, spy, app_marker=app_marker)
    await handler._run_generation(watchlist_id)

    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["category"] == "briefing"
    assert call["severity"] == "information"
    assert "Daily Brief" in call["message"]
    assert call["source_entity_kind"] == "briefing"
    assert int(call["source_entity_id"]) >= 1
    assert call["app"] is app_marker


@pytest.mark.asyncio
async def test_failed_generation_dispatches_warning_with_error(tmp_path):
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = _seeded_watchlist(db)

    def _failing_chat(**kwargs):
        raise RuntimeError("401 unauthorized")

    spy = _DispatchSpy()
    handler = _notify_handler(
        db, spy, generate=functools.partial(generate_briefing, chat=_failing_chat)
    )
    await handler._run_generation(watchlist_id)

    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["category"] == "briefing"
    assert call["severity"] == "warning"
    assert "401 unauthorized" in call["message"]


@pytest.mark.asyncio
async def test_no_dispatch_service_configured_stays_silent_and_safe(tmp_path):
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = _seeded_watchlist(db)
    handler = BriefingJobHandler(
        subscriptions_db=db,
        generate=functools.partial(generate_briefing, chat=_canned_chat),
        chachanotes_db_getter=lambda: None,
    )
    await handler._run_generation(watchlist_id)  # must not raise
    row = db.list_briefings(watchlist_id)[0]
    assert row["status"] == "complete"


@pytest.mark.asyncio
async def test_claim_race_dispatches_nothing(tmp_path):
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])

    async def _raced_generate(*args, **kwargs):
        raise GenerationInFlightError("claim lost")

    spy = _DispatchSpy()
    handler = _notify_handler(db, spy, generate=_raced_generate)
    await handler._run_generation(watchlist_id)
    assert spy.calls == []
