"""TASK-1383: a scheduled check must produce the run row the Runs pane reads.

The scheduled path used to sink its results into
`SubscriptionsDB.record_check_result` only, which writes `subscription_stats`
-- a daily aggregate whose one reader has no callers. The Runs pane reads
`local_watchlist_runs`, written only by `LocalWatchlistsService.launch_run`.
So a source checked only by the scheduler produced nothing on the one screen
built to show what a check did.

These tests drive the REAL `URLMonitor` through the REAL handler with only the
HTTP fetch faked (AC#2): every pre-existing scheduled-path test passes
`url_monitor=AsyncMock()`, so none of them ever exercised `check_url`'s actual
contract -- its return shape, its dispositions, its snapshot writes -- through
the scheduled caller.

Threading: every test here is same-thread `asyncio` (`await loop.tick()` /
`await handler.handle(...)`), so an in-memory DB would be safe. They still use
file-backed DBs under `tmp_path`, matching the rest of `Tests/Scheduling`,
because `SubscriptionsDB` builds its schema on the constructing thread and
keeps thread-local connections -- a later thread-hop would otherwise turn
these into silent no-ops against an empty schema rather than failures.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from loguru import logger as loguru_logger

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler import (
    WatchlistCheckHandler,
)
from tldw_chatbook.Scheduling.scheduler.loop import SchedulerLoop
from tldw_chatbook.Scheduling.services.watchlist_projection import WatchlistProjection
from tldw_chatbook.Subscriptions import LocalWatchlistsService
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    EXECUTABLE_SOURCE_TYPES,
)

pytestmark = pytest.mark.unit


# --- harness ---------------------------------------------------------------


def _response(text: str, url: str, *, content_type: str = "text/html"):
    """Stand in for the `httpx.Response` `guarded_fetch_httpx_async` returns.

    Adapted from `Tests/Subscriptions/test_watchlist_content_kind_producer.py`,
    with `final_url` following the requested URL so a multi-URL source's
    responses stay distinguishable.
    """
    return SimpleNamespace(
        status_code=200,
        headers={"content-type": content_type},
        text=text,
        final_url=url,
        raise_for_status=lambda: None,
    )


def _serve(monkeypatch, pages: dict[str, str], *, sitemap: bool = False) -> list[str]:
    """Serve `pages` by URL from the real fetch seam. Returns the fetch log.

    Args:
        monkeypatch: pytest fixture.
        pages: URL -> body. A URL not present raises, which surfaces as a
            failed check rather than a silent empty one.
        sitemap: Also patch `local_watchlists_service`'s own import of the
            fetch, which is what `_urls_for_sitemap` calls to read the index.

    Returns:
        The list of URLs fetched, in order -- what "checked every URL" is
        asserted against.
    """
    fetched: list[str] = []

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        fetched.append(url)
        if url not in pages:
            raise AssertionError(f"unexpected fetch: {url}")
        content_type = (
            "application/xml" if sitemap and url.endswith(".xml") else "text/html"
        )
        return _response(pages[url], url, content_type=content_type)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )
    if sitemap:
        monkeypatch.setattr(
            "tldw_chatbook.Subscriptions.local_watchlists_service."
            "guarded_fetch_httpx_async",
            fake_guarded,
        )
    return fetched


def _serve_failure(monkeypatch, error: Exception) -> None:
    """Make every fetch raise, to drive the failure path."""

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        raise error

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )


def _page(body: str) -> str:
    return f"<html><body><article>{body}</article></body></html>"


def _sitemap(urls: list[str]) -> str:
    entries = "".join(f"<url><loc>{url}</loc></url>" for url in urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{entries}</urlset>"
    )


def _add_due_source(subs_db: SubscriptionsDB, **kwargs) -> int:
    """Add an active source whose cadence has already elapsed."""
    kwargs.setdefault("check_frequency", 3600)
    subscription_id = subs_db.add_subscription(**kwargs)
    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    with subs_db.transaction() as conn:
        conn.execute(
            "UPDATE subscriptions SET last_checked = ? WHERE id = ?",
            (stale, subscription_id),
        )
    return subscription_id


def _handler(subs_db: SubscriptionsDB, **kwargs) -> WatchlistCheckHandler:
    return WatchlistCheckHandler(subscriptions_db=subs_db, **kwargs)


def _task(subscription_id: int) -> dict:
    """The projected task shape the scheduler dispatches."""
    return {
        "id": f"watchlist:{subscription_id}",
        "type": "watchlist_job",
        "title": "Scheduled source",
        "owner_id": "local",
    }


def _service(subs_db: SubscriptionsDB) -> LocalWatchlistsService:
    return LocalWatchlistsService(db_factory=lambda: subs_db)


def _loop(subs_db: SubscriptionsDB, handler: WatchlistCheckHandler) -> SchedulerLoop:
    """A real loop whose only source of work is the watchlist projection."""
    tasks_db = MagicMock()
    tasks_db.list_reminder_tasks.return_value = []
    return SchedulerLoop(
        tasks_db,
        handlers={"watchlist_job": handler},
        watchlist_projection=WatchlistProjection(subs_db),
    )


def _run_rows(subs_db: SubscriptionsDB) -> list[dict]:
    with subs_db.transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM local_watchlist_runs ORDER BY id ASC"
        ).fetchall()
    return [dict(row) for row in rows]


def _count_items(subs_db: SubscriptionsDB, subscription_id: int) -> int:
    with subs_db.transaction() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM subscription_items WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
    return row["n"]


def _count_snapshots(subs_db: SubscriptionsDB, subscription_id: int) -> int:
    with subs_db.transaction() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM url_snapshots WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchone()
    return row["n"]


# --- AC#1 + AC#2 -----------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduled_url_check_creates_a_run_the_runs_pane_can_read(
    tmp_path, monkeypatch
):
    """The whole point: a scheduler-only source becomes a visible run.

    Drives the real `URLMonitor` (AC#2) across two checks so the second one
    carries a non-trivial disposition -- `changed`, not just the first check's
    baseline -- and asserts it both in the stored `stats_json` (AC#1) and in
    the flattened shape `list_runs` hands the Runs pane.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    url = "https://example.com/watched"
    subscription_id = _add_due_source(
        subs_db, name="Watched page", type="url", source=url
    )
    handler = _handler(subs_db)

    _serve(monkeypatch, {url: _page("original text")})
    await handler.handle(_task(subscription_id))

    _serve(monkeypatch, {url: _page("completely different text now")})
    await handler.handle(_task(subscription_id))

    rows = _run_rows(subs_db)
    assert len(rows) == 2, "each scheduled check must record its own run row"
    assert [row["source_id"] for row in rows] == [subscription_id] * 2

    first_stats = json.loads(rows[0]["stats_json"])
    second_stats = json.loads(rows[1]["stats_json"])
    assert first_stats["dispositions"]["baseline"] == 1, (
        "the first check of a URL stores a baseline and must say so"
    )
    assert second_stats["dispositions"]["changed"] == 1, (
        "the scheduled path dropped the disposition entirely before TASK-1383"
    )

    runs = await _service(subs_db).list_runs(source_id=subscription_id)
    assert len(runs) == 2
    latest = runs[0]
    assert latest["status"] == "completed"
    assert latest["dispositions"]["changed"] == 1, (
        "the Runs pane reads dispositions off the run's top level"
    )
    assert latest["stats"]["new_items_found"] == 1


@pytest.mark.asyncio
async def test_scheduled_check_produces_the_item_and_snapshot(tmp_path, monkeypatch):
    """The real monitor's side effects reach the DB through the scheduler."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    url = "https://example.com/watched"
    subscription_id = _add_due_source(
        subs_db, name="Watched page", type="url", source=url
    )
    handler = _handler(subs_db)

    _serve(monkeypatch, {url: _page("original text")})
    await handler.handle(_task(subscription_id))
    _serve(monkeypatch, {url: _page("a materially different body")})
    await handler.handle(_task(subscription_id))

    assert _count_snapshots(subs_db, subscription_id) == 2, (
        "the real URLMonitor persists a snapshot per check"
    )

    with subs_db.transaction() as conn:
        items = conn.execute(
            "SELECT * FROM subscription_items WHERE subscription_id = ?",
            (subscription_id,),
        ).fetchall()
    assert len(items) == 1, "the detected change is stored as an item"
    assert items[0]["run_id"] == _run_rows(subs_db)[1]["id"], (
        "the item is attributed to the run that found it"
    )


# --- the two deleted divergences -------------------------------------------


@pytest.mark.asyncio
async def test_scheduled_sitemap_source_is_actually_checked(tmp_path, monkeypatch):
    """Regression: the handler's URL tuple omitted `sitemap` entirely.

    A scheduled sitemap source took the "unknown subscription type" branch and
    was never checked, while the same source checked by hand worked.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    index = "https://example.com/sitemap.xml"
    pages = [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]
    subscription_id = _add_due_source(
        subs_db, name="Sitemap source", type="sitemap", source=index
    )
    fetched = _serve(
        monkeypatch,
        {index: _sitemap(pages), **{url: _page(f"body {url}") for url in pages}},
        sitemap=True,
    )

    await _handler(subs_db).handle(_task(subscription_id))

    rows = _run_rows(subs_db)
    assert len(rows) == 1, "a scheduled sitemap check must record a run"
    assert rows[0]["status"] == "completed"
    stats = json.loads(rows[0]["stats_json"])
    assert sum(stats["dispositions"].values()) == len(pages), (
        "every URL in the sitemap must be checked"
    )
    assert [url for url in fetched if url != index] == pages


@pytest.mark.asyncio
async def test_scheduled_url_list_checks_every_url(tmp_path, monkeypatch):
    """Regression: `url_list` was handed whole to ONE `check_url` call.

    A scheduled 50-URL source checked a single URL -- and, because the whole
    subscription was passed through, the one it checked was whatever `source`
    happened to hold.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    urls = [f"https://example.com/page-{index}" for index in range(4)]
    subscription_id = _add_due_source(
        subs_db,
        name="URL list source",
        type="url_list",
        source=urls[0],
        extraction_rules=json.dumps({"urls": urls}),
    )
    fetched = _serve(monkeypatch, {url: _page(f"body {url}") for url in urls})

    await _handler(subs_db).handle(_task(subscription_id))

    rows = _run_rows(subs_db)
    assert len(rows) == 1
    stats = json.loads(rows[0]["stats_json"])
    assert sum(stats["dispositions"].values()) == len(urls), (
        f"expected one disposition per URL, got {stats['dispositions']}"
    )
    assert fetched == urls, "every configured URL must be fetched, in order"


# --- shadow mode -----------------------------------------------------------


@pytest.mark.asyncio
async def test_shadow_mode_creates_no_run_row_item_or_snapshot(tmp_path, monkeypatch):
    """Shadow mode stays a no-mutation probe: it must not join the run table."""
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    url = "https://example.com/watched"
    subscription_id = _add_due_source(
        subs_db, name="Watched page", type="url", source=url
    )
    before = subs_db.get_subscription(subscription_id)["last_checked"]
    fetched = _serve(monkeypatch, {url: _page("original text")})

    await _handler(subs_db, shadow_mode=True).handle(_task(subscription_id))

    assert fetched == [url], "shadow mode still performs the fetch"
    assert _run_rows(subs_db) == [], "shadow mode must not create a run row"
    assert _count_items(subs_db, subscription_id) == 0, (
        "shadow mode wrote to subscription_items"
    )
    assert _count_snapshots(subs_db, subscription_id) == 0, (
        "shadow mode wrote to url_snapshots"
    )
    after = subs_db.get_subscription(subscription_id)["last_checked"]
    assert after == before, "shadow mode must not record a check result"


# --- failure path and auto-pause parity ------------------------------------


@pytest.mark.asyncio
async def test_failed_fetch_marks_run_failed_and_bumps_failures_once(
    tmp_path, monkeypatch
):
    """A dead source records a failed run AND keeps the auto-pause counter.

    Auto-pause is driven by `subscriptions.consecutive_failures` (its only
    implementation is `record_check_result`'s error branch,
    `DB/Subscriptions_DB.py:1318-1341`). The handler used to bump that counter
    via `record_check_error`; the service path reaches the identical call from
    `record_run_failure` (`local_watchlists_service.py:509`), so the counter
    must still advance -- exactly once, not twice.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = _add_due_source(
        subs_db, name="Dead page", type="url", source="https://example.com/gone"
    )
    _serve_failure(monkeypatch, RuntimeError("connection refused"))
    handler = _handler(subs_db)

    await handler.handle(_task(subscription_id))

    rows = _run_rows(subs_db)
    assert len(rows) == 1, "a failed scheduled check must still record a run"
    assert rows[0]["status"] == "failed"
    assert "connection refused" in (rows[0]["error_msg"] or "")

    row = subs_db.get_subscription(subscription_id)
    assert row["consecutive_failures"] == 1, (
        "auto-pause input must advance exactly once per failed check"
    )
    assert "connection refused" in (row["last_error"] or "")

    await handler.handle(_task(subscription_id))
    assert subs_db.get_subscription(subscription_id)["consecutive_failures"] == 2, (
        "repeated failures must keep accumulating toward auto_pause_threshold"
    )


@pytest.mark.asyncio
async def test_failure_around_execution_still_records_run_and_error(
    tmp_path, monkeypatch
):
    """A failure that escapes `execute_run` must not leave a `queued` orphan.

    `execute_run` resolves the run and its subscription *before* its own
    `try`, so a source deleted between launch and execution -- or any other
    fault around the fetch -- raises straight out. This is the TASK-1090
    shape, now reachable from the scheduler because the scheduler launches
    runs at all.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = _add_due_source(
        subs_db, name="Watched page", type="url", source="https://example.com/watched"
    )
    service = _service(subs_db)

    async def exploding_execute_run(run_id):
        raise RuntimeError("source vanished mid-run")

    monkeypatch.setattr(service, "execute_run", exploding_execute_run)

    await _handler(subs_db, watchlists_service=service).handle(_task(subscription_id))

    rows = _run_rows(subs_db)
    assert len(rows) == 1
    assert rows[0]["status"] == "failed", "the launched run must not stay queued"
    assert "source vanished mid-run" in (rows[0]["error_msg"] or "")
    row = subs_db.get_subscription(subscription_id)
    assert row["consecutive_failures"] == 1
    assert "source vanished mid-run" in (row["last_error"] or "")


@pytest.mark.asyncio
async def test_failure_recorder_itself_raising_still_records_the_check_error(
    tmp_path, monkeypatch
):
    """The last-resort fallback: even the failure recorder can fail.

    `_record_failure` prefers `record_run_failure` so the launched row is
    marked, and falls back to `record_check_error` when that call itself
    raises. Without the fallback the original error reaches no durable surface
    at all -- the exact swallowed-failure shape this whole path exists to
    prevent -- and the exception would escape `handle` into the scheduler loop.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = _add_due_source(
        subs_db, name="Watched page", type="url", source="https://example.com/watched"
    )
    service = _service(subs_db)

    async def exploding_execute_run(run_id):
        raise RuntimeError("source vanished mid-run")

    async def exploding_record_run_failure(run_id, **kwargs):
        raise RuntimeError("the run table is unwritable")

    monkeypatch.setattr(service, "execute_run", exploding_execute_run)
    monkeypatch.setattr(service, "record_run_failure", exploding_record_run_failure)

    handler = _handler(subs_db, watchlists_service=service)
    loop = _loop(subs_db, handler)
    loop.queue.load()

    # Through the real loop: an escaping exception here is what stops the
    # scheduler dispatching every later task in the tick.
    await loop.tick()

    row = subs_db.get_subscription(subscription_id)
    assert "source vanished mid-run" in (row["last_error"] or ""), (
        "the ORIGINAL error must survive, not the recorder's own failure"
    )
    assert row["consecutive_failures"] == 1


@pytest.mark.asyncio
async def test_failed_scheduled_check_is_logged_as_a_warning_with_the_error(
    tmp_path, monkeypatch
):
    """A failed run must say so in the log, with its error text.

    It used to be reported at INFO as "check complete" with no error at all --
    so an unattended check of a dead source left a metric counter as its only
    trace, which is the same invisibility the run row was added to fix.

    Captured with a loguru sink rather than `caplog`: this repo logs through
    loguru, which does not propagate to the stdlib `logging` handlers `caplog`
    installs, so a `caplog`-based assertion here would pass vacuously.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    subscription_id = _add_due_source(
        subs_db, name="Dead page", type="url", source="https://example.com/gone"
    )
    _serve_failure(monkeypatch, RuntimeError("connection refused"))

    records: list[tuple[str, str]] = []
    sink_id = loguru_logger.add(
        lambda message: records.append(
            (message.record["level"].name, message.record["message"])
        ),
        level="INFO",
    )
    try:
        await _handler(subs_db).handle(_task(subscription_id))
    finally:
        loguru_logger.remove(sink_id)

    warnings = [text for level, text in records if level in ("WARNING", "ERROR")]
    assert any("connection refused" in text for text in warnings), (
        f"the failure's error text must reach the log at WARNING+; got {records}"
    )
    assert not any("check complete" in text.lower() for _, text in records), (
        "a failed run must not also be announced as a completed one"
    )


@pytest.mark.asyncio
async def test_shadow_reports_unsupported_types_distinctly_from_unknown_ones(
    tmp_path, monkeypatch
):
    """Shadow mode must not call a real, checkable source unrecognised.

    Unification made the real path execute `sitemap` and `api`, but the shadow
    probe cannot reach either. Reporting them as `unknown_type` would have
    shadow mode -- a diagnostic -- assert that a perfectly valid source is not
    recognised by the scheduler, which is a false clean bill of health.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    sitemap_id = _add_due_source(
        subs_db,
        name="Sitemap source",
        type="sitemap",
        source="https://example.com/sitemap.xml",
    )
    fetched = _serve(monkeypatch, {}, sitemap=True)

    statuses: list[str] = []
    monkeypatch.setattr(
        "tldw_chatbook.Scheduling.scheduler.handlers.watchlist_check_handler.log_counter",
        lambda name, labels=None: statuses.append((labels or {}).get("status")),
    )

    handler = _handler(subs_db, shadow_mode=True)
    await handler.handle(_task(sitemap_id))

    assert statuses == ["shadow_unsupported"], (
        "an executable type the probe cannot reach is not an unknown type"
    )
    assert fetched == [], "shadow mode must not pretend to have checked it"
    assert _run_rows(subs_db) == [], "shadow mode still writes no run row"

    # A type nothing can execute is still reported as genuinely unknown.
    statuses.clear()
    unknown = _subscription_of_unsupported_type(subs_db)
    await handler.handle(_task(unknown))
    assert statuses == ["unknown_type"]


def _subscription_of_unsupported_type(subs_db: SubscriptionsDB) -> int:
    """Insert a row whose type no executor handles.

    Written past `add_subscription` deliberately: the `subscriptions.type` CHECK
    constraint accepts exactly `EXECUTABLE_SOURCE_TYPES`, so a genuinely unknown
    type cannot be stored through the normal API -- which is itself the
    invariant `test_executable_types_match_every_type_the_db_accepts` pins.
    """
    subscription_id = _add_due_source(
        subs_db, name="Odd source", type="url", source="https://example.com/odd"
    )
    with subs_db.transaction() as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute(
            "UPDATE subscriptions SET type = 'gopher' WHERE id = ?",
            (subscription_id,),
        )
        conn.execute("PRAGMA ignore_check_constraints = OFF")
    return subscription_id


# --- guards the handler keeps ----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("gate", [{"is_paused": 1}, {"is_active": 0}])
async def test_paused_or_inactive_source_creates_no_run_row(
    tmp_path, monkeypatch, gate
):
    """The handler's own gate still short-circuits before anything is launched.

    The gate goes on through `update_subscription`, whose own allowlist decides
    which columns may be written, rather than by interpolating a column name
    into SQL here.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    url = "https://example.com/watched"
    subscription_id = _add_due_source(
        subs_db, name="Watched page", type="url", source=url
    )
    assert subs_db.update_subscription(subscription_id, **gate) is True
    fetched = _serve(monkeypatch, {url: _page("original text")})

    await _handler(subs_db).handle(_task(subscription_id))

    assert fetched == [], "a paused/inactive source must not be fetched"
    assert _run_rows(subs_db) == [], "a skipped source must not create a run row"


def test_executable_types_match_every_type_the_db_accepts(tmp_path):
    """The handler's executable set must not drift from the schema's, again.

    This is the bug of TASK-1383 stated as an invariant: the handler carried
    its own `_URL_TYPES = ("url", "url_list")` while the `subscriptions.type`
    CHECK constraint had long since accepted `sitemap` too, so a row the DB
    happily stored was one the scheduler declared an unknown type and refused
    to check. Anything storable must be executable.
    """
    subs_db = SubscriptionsDB(tmp_path / "subs.db")
    with subs_db.transaction() as conn:
        schema = conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'subscriptions'"
        ).fetchone()["sql"]
    constraint = re.search(r"type\s+IN\s*\(([^)]*)\)", schema)
    assert constraint, "the subscriptions.type CHECK constraint moved or vanished"
    accepted = {value.strip().strip("'\"") for value in constraint.group(1).split(",")}

    assert accepted == set(EXECUTABLE_SOURCE_TYPES), (
        "every source type the database accepts must have a run executor; "
        f"unexecutable: {accepted - set(EXECUTABLE_SOURCE_TYPES)}, "
        f"unstorable: {set(EXECUTABLE_SOURCE_TYPES) - accepted}"
    )
