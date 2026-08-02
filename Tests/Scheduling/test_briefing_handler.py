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
# `BriefingJobHandler` gains an optional `chachanotes_db`; once a spawned
# generation resolves `complete`, `_run_generation` mirrors it into
# ChaChaNotes via `briefing_keep.keep_briefing(..., origin="scheduled")`
# (spec: Keep-service "Auto path"). Every real DB here is file-backed at
# `tmp_path`, never `:memory:` and never the live user data directory --
# matching both this file's own convention above and
# `Tests/Subscriptions/test_briefing_keep.py`'s.


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
            chachanotes_db=chacha_db,
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

        handler = BriefingJobHandler(subscriptions_db=db, chachanotes_db=chacha_db)

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
            subscriptions_db=db, generate=fake_generate, chachanotes_db=chacha_db
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
            subscriptions_db=db, generate=fake_generate, chachanotes_db=chacha_db
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
    """`chachanotes_db=None` -- the default, and what `app.py` passes when
    its own ChaChaNotes handle is not yet constructed at the point this
    handler is wired up -- must never crash the handler and must never
    attempt a keep; generation still completes normally."""

    async def fake_generate(db, watchlist_id, *, preset_id=None, **kwargs):
        return {"id": 1, "status": "complete", "body_markdown": "irrelevant"}

    handler = BriefingJobHandler(
        subscriptions_db=_db_with_default_preset(None), generate=fake_generate
    )
    assert handler._chachanotes_db is None

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
            subscriptions_db=db, generate=fake_generate, chachanotes_db=chacha_db
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
