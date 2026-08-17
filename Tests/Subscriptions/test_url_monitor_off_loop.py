"""task-15764: URLMonitor's extraction and diff work runs off the event loop.

task-15463 moved the watchlist check path's sqlite hops and the feed-body
parse off the loop, and deliberately left `URLMonitor`'s own CPU work inline
as "a separate, larger change". That work is this task's scope, and it is
pure CPU with no sqlite involvement:

* `ContentExtractor.extract_text_from_html` -- BeautifulSoup over a page of
  up to `MAX_FETCH_BYTES_PAGE` (10 MB), run on EVERY check of a
  `url`/`url_list`/`sitemap` source, including the common unchanged case;
* the difflib work on the changed path: `calculate_change_percentage`
  (`SequenceMatcher.ratio` over both full texts -- the largest single
  difflib cost here, though task-15764's AC enumeration missed it),
  `_segment_for_diff` twice, `build_change_diff`, `added_and_removed_text`,
  and `classify_change_type`.

Reconciliation note (task-596 playbook, 4th duplicate-implementation event):
two independent sessions implemented task-15764; the merged-first
implementation (commits 8f638f815 + 8fbd1426d, PR #1650) stands, and this
file is the ported delta from the reviewed second implementation (verdict
MERGE; semantically byte-identical to the incumbent, verified across 11
baseline+change cycles). It adds what the incumbent's own tests in
`test_watchlists_db_instance_and_off_loop.py` do not cover: the whole-check
extraction path through `check_url` (theirs probes `_fetch_url_content`
directly), and -- the incumbent's AC#3 had no test at all -- a `url_list`
source driven through the real `launch_run`/`execute_run` path, two URLs x
two runs, proving every URL's extraction AND diff work leaves the loop, not
just the first URL's.

Like task-15463's `test_watchlists_db_instance_and_off_loop.py`, everything
below is mechanical thread identity -- no durations, nothing timing-flaky.
The spies call straight through to the real functions, so every test also
asserts the OUTPUT it always produced (baseline row, diff body, added and
removed text, dispositions): a hop that changed what the check computes
would redden here, not just a hop that never happened.

Threading note: the file-backed DBs under `tmp_path` are load-bearing --
`SubscriptionsDB` keeps thread-local connections, so `URLMonitor`'s
`run_db_off_loop` hops need a database whose file another thread can open.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import tldw_chatbook.Subscriptions.monitoring_engine as monitoring_engine
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.local_watchlists_service import (
    LocalWatchlistsService,
)
from tldw_chatbook.Subscriptions.monitoring_engine import (
    DISPOSITION_BASELINE_STORED,
    DISPOSITION_CHANGED,
    ContentExtractor,
    URLMonitor,
)
from tldw_chatbook.Subscriptions.watchlist_rule_matching import (
    RULE_MATCH_ADDED_TEXT_KEY,
    RULE_MATCH_REMOVED_TEXT_KEY,
)

pytestmark = pytest.mark.unit


_PAGE_ONE = (
    "<html><body><p>alpha bravo charlie.</p>\n<p>delta echo foxtrot.</p></body></html>"
)
_PAGE_TWO = (
    "<html><body><p>alpha bravo charlie.</p>\n<p>golf hotel india.</p></body></html>"
)


def _url_subscription(db: SubscriptionsDB) -> dict:
    """A real subscriptions row (`url_snapshots.subscription_id` is a
    foreign key under `PRAGMA foreign_keys = ON`), as the dict `check_url`
    takes."""
    source_id = db.add_subscription(
        name="Watched page", type="url", source="https://example.com/page"
    )
    return {
        "id": source_id,
        "name": "Watched page",
        "type": "url",
        "source": "https://example.com/page",
    }


#: The difflib work `check_url` owes the changed path, exactly as the task's
#: AC enumerates it -- plus `calculate_change_percentage`, which is also
#: difflib (`SequenceMatcher.ratio` over both full texts) and also ran inline.
_DIFF_WORK = (
    "_segment_for_diff",
    "build_change_diff",
    "added_and_removed_text",
    "classify_change_type",
)


def _response(text: str, url: str):
    """Stand in for the `httpx.Response` `guarded_fetch_httpx_async` returns."""
    return SimpleNamespace(
        status_code=200,
        headers={"content-type": "text/html"},
        text=text,
        final_url=url,
        raise_for_status=lambda: None,
    )


def _serve(monkeypatch, body_for) -> list[str]:
    """Serve HTML from the real fetch seam. Returns the fetch log.

    ``body_for(url, call_number)`` picks the body, with ``call_number``
    starting at 1, so one test can serve a baseline and then a changed page.
    """
    fetched: list[str] = []

    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        fetched.append(url)
        return _response(body_for(url, len(fetched)), url)

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )
    return fetched


def _spy_extraction(monkeypatch) -> list[tuple[str, int]]:
    """Record ``(html, thread_ident)`` per extraction; call the real one."""
    extractions: list[tuple[str, int]] = []
    real_extract = ContentExtractor.extract_text_from_html

    def spy(html, ignore_selectors=None):
        extractions.append((html, threading.get_ident()))
        return real_extract(html, ignore_selectors)

    monkeypatch.setattr(ContentExtractor, "extract_text_from_html", staticmethod(spy))
    return extractions


def _spy_diff_work(monkeypatch) -> dict[str, list[int]]:
    """Record the thread each diff-work call ran on; call the real ones.

    Module-global lookups are late-bound, so these spies see the calls
    whether `check_url` makes them inline (the pre-task-15764 shape) or from
    inside a `to_thread` hop -- which is what lets the same test be born red.
    """
    threads: dict[str, list[int]] = {}

    for name in _DIFF_WORK:
        real = getattr(monitoring_engine, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            threads.setdefault(_name, []).append(threading.get_ident())
            return _real(*args, **kwargs)

        monkeypatch.setattr(monitoring_engine, name, spy)

    real_percentage = ContentExtractor.calculate_change_percentage

    def percentage_spy(old_content, new_content):
        threads.setdefault("calculate_change_percentage", []).append(
            threading.get_ident()
        )
        return real_percentage(old_content, new_content)

    monkeypatch.setattr(
        ContentExtractor, "calculate_change_percentage", staticmethod(percentage_spy)
    )
    return threads


# --- AC#1: the HTML extraction ------------------------------------------------


@pytest.mark.asyncio
async def test_html_extraction_runs_off_the_event_loop_thread(tmp_path, monkeypatch):
    """BeautifulSoup over a fetched page is not loop work.

    This is the cost EVERY url-family check pays, changed or not, so it is
    asserted on the plain first-check path with nothing else in play.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    monitor = URLMonitor(db)
    subscription = _url_subscription(db)
    _serve(monkeypatch, lambda url, n: _PAGE_ONE)
    extractions = _spy_extraction(monkeypatch)

    result, disposition = await monitor.check_url(subscription)

    # Never vacuous: the check must have done its usual first-check work.
    assert result is None and disposition["kind"] == DISPOSITION_BASELINE_STORED
    with db.transaction() as conn:
        rows = conn.execute(
            "SELECT extracted_content FROM url_snapshots WHERE subscription_id = ?",
            (subscription["id"],),
        ).fetchall()
    assert len(rows) == 1 and "alpha bravo charlie." in rows[0]["extracted_content"], (
        "the baseline snapshot must hold the extracted text, or the spy "
        "proves nothing about where the extraction ran"
    )

    assert len(extractions) == 1, "the extraction must have run exactly once"
    assert extractions[0][1] != threading.get_ident(), (
        "extract_text_from_html must run under asyncio.to_thread, not inline "
        "on the event loop"
    )


# --- AC#2: the difflib work ---------------------------------------------------


@pytest.mark.asyncio
async def test_the_diff_work_runs_off_the_event_loop_thread(tmp_path, monkeypatch):
    """Every difflib call on the changed path leaves the loop -- and still
    produces byte-for-byte the diff, rule-match text and classification it
    always did."""
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    monitor = URLMonitor(db)
    subscription = _url_subscription(db)
    bodies = {1: _PAGE_ONE, 2: _PAGE_TWO}
    _serve(monkeypatch, lambda url, n: bodies[n])
    threads = _spy_diff_work(monkeypatch)

    baseline, first_disposition = await monitor.check_url(subscription)
    assert baseline is None
    assert first_disposition["kind"] == DISPOSITION_BASELINE_STORED
    assert not threads, "no diff work exists on a first check"

    change, disposition = await monitor.check_url(subscription)

    # Semantics pinned first: same item the inline code produced.
    assert disposition["kind"] == DISPOSITION_CHANGED
    assert change is not None and change["type"] == "url_change"
    assert "+golf hotel india." in change["content"]
    assert "-delta echo foxtrot." in change["content"]
    assert change["diff_summary"] == "1 line(s) added, 1 removed"
    assert change["change_type"] == "content"
    assert change[RULE_MATCH_ADDED_TEXT_KEY] == "golf hotel india."
    assert change[RULE_MATCH_REMOVED_TEXT_KEY] == "delta echo foxtrot."

    expected = set(_DIFF_WORK) | {"calculate_change_percentage"}
    assert set(threads) == expected, (
        f"every difflib call must have happened (saw {sorted(threads)})"
    )
    # 4 = 2 in the change-percentage hop (TASK-16839 rebased the percentage
    # onto the same segment basis as the diff) + 2 in the details hop, where
    # the Qodo segment-once rule still holds: `build_change_diff` and
    # `added_and_removed_text` share one segmentation per side. A 5th call
    # means that sharing broke.
    assert len(threads["_segment_for_diff"]) == 4, (
        "each hop segments each side exactly once (percentage hop + details "
        "hop) -- an extra call means the segment-once sharing broke"
    )
    loop_thread = threading.get_ident()
    for name, idents in threads.items():
        assert all(ident != loop_thread for ident in idents), (
            f"{name} ran on the event-loop thread during a URL check"
        )


# --- AC#3: url_list -- every URL, not just the first --------------------------


@pytest.mark.asyncio
async def test_url_list_runs_extraction_and_diff_off_loop_for_every_url(
    tmp_path, monkeypatch
):
    """The real `url_list` execution path, two URLs, two runs.

    Run one baselines both URLs (extraction only); run two changes both
    (extraction + diff). Every one of those per-URL costs must be off the
    loop -- a `url_list` source multiplies them by its URL count, which is
    exactly why the task calls it out.
    """
    db = SubscriptionsDB(tmp_path / "subs.db", "test")
    source_id = db.add_subscription(
        name="Two pages",
        type="url_list",
        source="https://a.example/one\nhttps://b.example/two",
    )
    service = LocalWatchlistsService(db_factory=lambda: db)

    def body_for(url, call_number):
        version = 1 if call_number <= 2 else 2
        return f"<html><body><p>page {url} version {version}.</p></body></html>"

    fetched = _serve(monkeypatch, body_for)
    extractions = _spy_extraction(monkeypatch)
    threads = _spy_diff_work(monkeypatch)

    first = await service.launch_run(source_id=source_id)
    first_run = await service.execute_run(first["run_id"])
    second = await service.launch_run(source_id=source_id)
    second_run = await service.execute_run(second["run_id"])

    # Never vacuous: both runs completed and every URL took its real path.
    assert first_run["status"] == "completed"
    assert first_run["stats"]["dispositions"]["baseline"] == 2
    assert second_run["status"] == "completed"
    assert second_run["stats"]["dispositions"]["changed"] == 2
    assert fetched == [
        "https://a.example/one",
        "https://b.example/two",
        "https://a.example/one",
        "https://b.example/two",
    ]

    loop_thread = threading.get_ident()
    assert len(extractions) == 4, "two URLs, two runs: four extractions"
    for html, ident in extractions:
        assert ident != loop_thread, (
            f"extraction ran on the event-loop thread for {html[:60]!r}"
        )
    # Each URL's own page is what was extracted -- the per-URL loop, not one
    # URL four times.
    assert [
        ("a.example/one" in html, "b.example/two" in html) for html, _ in extractions
    ] == [
        (True, False),
        (False, True),
        (True, False),
        (False, True),
    ]

    assert len(threads.get("build_change_diff", [])) == 2, (
        "run two must diff BOTH URLs, not just the first"
    )
    for name, idents in threads.items():
        assert all(ident != loop_thread for ident in idents), (
            f"{name} ran on the event-loop thread during a url_list run"
        )
