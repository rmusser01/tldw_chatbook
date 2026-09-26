# Tests/Dreams/test_cycle_service.py
"""Cycle orchestration: date claims, budgets, degradation, catch-up.

Real ``DreamsDB`` on ``tmp_path``; optional DB getters stubbed to ``None``;
the Task 3 fake shapes are reused for ``perform`` (``perform_websearch``
payload contract) and chat (OpenAI-shaped ``chat_api_call`` reply). ``now``
is fixed so the local date is deterministic on any machine's timezone.
"""
import asyncio
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.cycle_service import CycleDeps, run_catchup_if_due, run_cycle
from tldw_chatbook.Utils.timestamps import to_utc_iso

NOW = datetime(2026, 9, 22, 8, 0, tzinfo=UTC)


@pytest.fixture()
def dreams_db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "test-client")
    yield database
    database.close()


@pytest.fixture()
def settings(monkeypatch):
    """Deterministic [dreams] settings; defaults flow through when unset."""
    overrides: dict = {}

    def get(section, key, default=None):
        if section == "dreams" and key in overrides:
            return overrides[key]
        return default

    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting", get)
    return overrides


def _seed_topics(db):
    db.upsert_profile_entry("topic", "rust tui", weight=1.0, searchable=1,
                            source="user")
    db.upsert_profile_entry("topic", "jazz guitar", weight=0.8, searchable=1,
                            source="user")


def _fake_chat(**kwargs):
    system = str(kwargs.get("system_message") or "")
    if "search queries" in system:  # query_synthesis's system prompt
        return {"choices": [{"message": {"content": "rust tui news\n"
                                          "jazz guitar seattle\n"
                                          "surprising robots research"}}]}
    return {"choices": [{"message": {
        "content": "A story about this find, written for you because it "
                   "matches what you have been reading lately."}}]}


def _fake_perform(search_engine, search_query, **kwargs):
    slug = search_query.replace(" ", "-")
    return {"results": [{"title": f"{search_query} finding {i}",
                         "url": f"https://dreams.test/{slug}/{i}",
                         "content": "rust tui and jazz guitar details"}
                        for i in range(3)]}


def _deps(db, *, perform=None, chat=None):
    return CycleDeps(
        dreams_db=db,
        chachanotes_db_getter=lambda: None,
        media_db_getter=lambda: None,
        subs_db_getter=lambda: None,
        pc_service_getter=lambda: None,
        chat_getter=lambda: chat if chat is not None else _fake_chat,
        perform_search=perform if perform is not None else _fake_perform,
        now=lambda: NOW,
    )


def _today() -> str:
    return NOW.astimezone().strftime("%Y-%m-%d")


@pytest.mark.asyncio
async def test_run_cycle_creates_one_dated_collection_with_complete_stories(
        dreams_db, settings):
    _seed_topics(dreams_db)
    result = await run_cycle(_deps(dreams_db), trigger="manual")
    assert result["status"] == "complete" and result["stories"] == 5
    row = dreams_db.get_collection_by_date(_today())
    assert row is not None and row["status"] == "complete"
    assert row["trigger"] == "manual" and row["degradation_notes"] is None
    stories = dreams_db.list_stories(row["id"])
    assert len(stories) == 5
    assert all(s["status"] == "complete" for s in stories)
    assert {s["source"] for s in stories} == {"web"}
    # Searches, the one synthesis call, and one call per story are recorded
    # against the daily budget; every surfaced URL lands in the seen ledger.
    assert dreams_db.usage_get(_today()) == {"searches": 3, "llm_calls": 6}
    assert dreams_db.seen_filter_unseen([s["url"] for s in stories]) == set()


@pytest.mark.asyncio
async def test_second_run_same_date_returns_inflight_and_does_not_duplicate(
        dreams_db, settings):
    _seed_topics(dreams_db)
    entered = threading.Event()
    release = threading.Event()

    def slow_perform(search_engine, search_query, **kwargs):
        entered.set()
        release.wait(timeout=10)
        return _fake_perform(search_engine, search_query, **kwargs)

    deps = _deps(dreams_db, perform=slow_perform)
    first = asyncio.create_task(run_cycle(deps, trigger="manual"))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while not entered.is_set() and loop.time() < deadline:
        await asyncio.sleep(0.01)
    assert entered.is_set(), "first cycle never reached the search stage"

    second = await run_cycle(deps, trigger="manual")
    assert second == {"collection_id": None, "status": "inflight", "stories": 0}

    release.set()
    result = await first
    assert result["status"] == "complete"
    row = dreams_db.get_collection_by_date(_today())
    assert row is not None and row["status"] == "complete"
    assert len(dreams_db.list_stories(row["id"])) == result["stories"] == 5


@pytest.mark.asyncio
async def test_stale_generating_row_is_reclaimed_then_regenerated(
        dreams_db, settings):
    stale_date = "2026-09-01"
    stale_id = dreams_db.create_collection(stale_date, "scheduled", "old-digest")
    with dreams_db.transaction() as conn:
        conn.execute(
            "UPDATE dreams_collections SET created_at = '2020-01-01T00:00:00+00:00'"
            " WHERE id = ?", (stale_id,))

    _seed_topics(dreams_db)
    result = await run_cycle(_deps(dreams_db), trigger="catchup")
    assert result["status"] == "complete"

    stale = dreams_db.get_collection_by_date(stale_date)
    assert stale["status"] == "failed"
    assert "reclaimed" in (stale["degradation_notes"] or "")

    today_row = dreams_db.get_collection_by_date(_today())
    assert today_row is not None and today_row["status"] == "complete"
    assert len(dreams_db.list_stories(today_row["id"])) == 5


@pytest.mark.asyncio
async def test_budget_cap_trims_queries_and_records_degradation(
        dreams_db, settings):
    settings["max_searches_per_day"] = 1
    _seed_topics(dreams_db)
    result = await run_cycle(_deps(dreams_db), trigger="scheduled")
    assert result["status"] == "complete"

    usage = dreams_db.usage_get(_today())
    assert usage["searches"] <= settings["max_searches_per_day"] == 1
    # One query ran, so only that query's candidates exist (3 rows), and the
    # trim is visible on the collection row.
    assert result["stories"] == 3
    row = dreams_db.get_collection_by_date(_today())
    assert row["degradation_notes"] and "queries" in row["degradation_notes"]


@pytest.mark.asyncio
async def test_synthesis_llm_call_counts_toward_daily_budget(
        dreams_db, settings):
    _seed_topics(dreams_db)
    # A day whose llm budget allows exactly one call: synthesis spends it,
    # the story cap sees zero left, and the cycle records the degradation.
    settings["max_llm_calls_per_day"] = 1
    result = await run_cycle(_deps(dreams_db), trigger="scheduled")
    assert dreams_db.usage_get(_today()) == {"searches": 3, "llm_calls": 1}
    row = dreams_db.get_collection_by_date(_today())
    assert row["degradation_notes"] and "llm budget" in row["degradation_notes"]
    assert result["stories"] == 0 and result["status"] == "failed"


@pytest.mark.asyncio
async def test_unresolved_chat_makes_no_synthesis_budget_spend(
        dreams_db, settings):
    _seed_topics(dreams_db)

    def broken_getter():
        raise RuntimeError("Dreams provider/model unavailable")

    deps = CycleDeps(
        dreams_db=dreams_db,
        chachanotes_db_getter=lambda: None,
        media_db_getter=lambda: None,
        subs_db_getter=lambda: None,
        pc_service_getter=lambda: None,
        chat_getter=broken_getter,
        perform_search=_fake_perform,
        now=lambda: NOW,
    )
    result = await run_cycle(deps, trigger="manual")
    assert result["status"] == "failed" and result["stories"] == 0
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}


@pytest.mark.asyncio
async def test_second_trigger_same_date_appends_up_to_cap_without_duplicates(
        dreams_db, settings):
    settings["stories_per_cycle"] = 4
    _seed_topics(dreams_db)

    def sparse_perform(search_engine, search_query, **kwargs):
        slug = search_query.replace(" ", "-")
        return {"results": [{"title": f"{search_query} finding",
                             "url": f"https://dreams.test/{slug}/0",
                             "content": "rust tui and jazz guitar details"}]}

    first = await run_cycle(_deps(dreams_db, perform=sparse_perform),
                            trigger="manual")
    assert first["status"] == "complete" and first["stories"] == 3
    row = dreams_db.get_collection_by_date(_today())
    assert row["trigger"] == "manual"
    first_urls = {s["url"] for s in dreams_db.list_stories(row["id"])}

    # Overlapping pool: the SAME urls plus new ones per query.
    def mixed_perform(search_engine, search_query, **kwargs):
        slug = search_query.replace(" ", "-")
        return {"results": [
            {"title": f"{search_query} finding",
             "url": f"https://dreams.test/{slug}/0",  # already in run 1
             "content": "rust tui and jazz guitar details"},
            {"title": f"{search_query} extra 1",
             "url": f"https://dreams.test/{slug}/new-1",
             "content": "rust tui and jazz guitar details"},
            {"title": f"{search_query} extra 2",
             "url": f"https://dreams.test/{slug}/new-2",
             "content": "rust tui and jazz guitar details"},
        ]}

    second = await run_cycle(_deps(dreams_db, perform=mixed_perform),
                             trigger="refresh")
    # Total reaches the configured cap exactly; nothing duplicated.
    assert second["stories"] == 4 and second["status"] == "complete"
    row = dreams_db.get_collection_by_date(_today())
    assert row["trigger"] == "manual"  # creating trigger is kept
    stories = dreams_db.list_stories(row["id"])
    assert len(stories) == 4
    urls = [s["url"] for s in stories]
    assert len(set(urls)) == 4  # no duplicate rows
    assert first_urls <= set(urls)  # run 1's rows survive verbatim
    # Six overlapping candidates were offered; only the unseen, in-budget
    # remainder was appended (1 synthesis + 1 story call re-spent here).
    assert dreams_db.usage_get(_today()) == {"searches": 6, "llm_calls": 6}


@pytest.mark.asyncio
async def test_append_with_exhausted_budget_records_note_and_appends_nothing(
        dreams_db, settings):
    _seed_topics(dreams_db)

    def sparse_perform(search_engine, search_query, **kwargs):
        slug = search_query.replace(" ", "-")
        return {"results": [{"title": f"{search_query} finding",
                             "url": f"https://dreams.test/{slug}/0",
                             "content": "rust tui and jazz guitar details"}]}

    first = await run_cycle(_deps(dreams_db, perform=sparse_perform),
                            trigger="manual")
    assert first["stories"] == 3  # 1 synthesis + 3 story calls spent below
    # Exhaust the day's llm budget exactly: the append may add no stories.
    settings["max_llm_calls_per_day"] = 4

    second = await run_cycle(_deps(dreams_db), trigger="refresh")
    assert second["status"] == "complete" and second["stories"] == 3
    row = dreams_db.get_collection_by_date(_today())
    assert row["degradation_notes"] and "llm budget" in row["degradation_notes"]
    assert len(dreams_db.list_stories(row["id"])) == 3
    # The budget is checked BEFORE synthesis: nothing re-spent at all.
    assert dreams_db.usage_get(_today()) == {
        "searches": 6, "llm_calls": 4}


@pytest.mark.asyncio
async def test_search_failure_falls_back_to_llm_source_stories_marked_degraded(
        dreams_db, settings):
    _seed_topics(dreams_db)

    def dead_perform(search_engine, search_query, **kwargs):
        return {"results": [], "processing_error": "Search provider failed.",
                "error_kind": "response"}

    result = await run_cycle(_deps(dreams_db, perform=dead_perform),
                             trigger="scheduled")
    assert result["status"] == "complete" and result["stories"] == 3

    row = dreams_db.get_collection_by_date(_today())
    assert row["degradation_notes"] and "web search" in row["degradation_notes"]
    stories = dreams_db.list_stories(row["id"])
    assert stories and all(s["source"] == "llm" for s in stories)
    assert all(s["status"] == "complete" for s in stories)
    assert all(s["url"].startswith("dreams://llm/") for s in stories)


@pytest.mark.asyncio
async def test_run_catchup_if_due_runs_only_when_no_collection_today(
        dreams_db, settings, tmp_path):
    settings["enabled"] = True
    settings["catchup_enabled"] = True
    _seed_topics(dreams_db)

    assert await run_catchup_if_due(_deps(dreams_db)) is True
    row = dreams_db.get_collection_by_date(_today())
    assert row is not None and row["trigger"] == "catchup"
    assert row["status"] == "complete"

    # Today already has a collection: not due again.
    assert await run_catchup_if_due(_deps(dreams_db)) is False

    other = DreamsDB(tmp_path / "dreams-other.sqlite", "test-client")
    try:
        # Disabled Dreams never catches up, even with no collection today.
        settings["enabled"] = False
        assert await run_catchup_if_due(_deps(other)) is False
        assert other.get_collection_by_date(_today()) is None

        # Enabled, but yesterday's cycle completed within the cadence window.
        settings["enabled"] = True
        yesterday = (NOW - timedelta(days=1)).astimezone().strftime("%Y-%m-%d")
        prior_id = other.create_collection(yesterday, "scheduled", "d")
        other.set_collection_status(prior_id, "complete",
                                    completed_at=NOW.isoformat())
        assert await run_catchup_if_due(_deps(other)) is False

        # A failed latest collection (e.g. reclaimed after a crash) is due.
        other.set_collection_status(prior_id, "failed")
        assert await run_catchup_if_due(_deps(other)) is True
        today_on_other = other.get_collection_by_date(_today())
        assert today_on_other is not None
        assert today_on_other["trigger"] == "catchup"
    finally:
        other.close()


def test_library_urls_reads_media_urls_and_tolerates_missing_db(
        dreams_db, settings):
    from types import SimpleNamespace

    from tldw_chatbook.Dreams.cycle_service import _library_urls

    assert _library_urls(_deps(dreams_db)) == set()

    class FakeMediaDB:
        def __init__(self, urls):
            self._rows = [{"url": u} for u in urls]
            self.queries = []

        def execute_query(self, query, params=None):
            self.queries.append((query, params))
            return SimpleNamespace(
                fetchall=lambda: list(self._rows))

    media_db = FakeMediaDB(["https://lib/1", "https://lib/2", None])
    deps = CycleDeps(
        dreams_db=dreams_db,
        chachanotes_db_getter=lambda: None,
        media_db_getter=lambda: media_db,
        subs_db_getter=lambda: None,
        pc_service_getter=lambda: None,
        chat_getter=lambda: _fake_chat,
        perform_search=_fake_perform,
        now=lambda: NOW,
    )
    assert _library_urls(deps) == {"https://lib/1", "https://lib/2"}
    query, params = media_db.queries[0]
    assert "SELECT url FROM Media" in query
    assert "?" in query and params is not None


# --- Profile-signal refresh (Ruling R18) + feedback loop (Ruling R19) -------


class _FakeNotesDB:
    """Minimal notes DB: one execute_query returning keyword/uses rows."""

    def __init__(self, keyword_uses):
        self._rows = [{"keyword": k, "uses": u} for k, u in keyword_uses]

    def execute_query(self, query, params=None):
        return SimpleNamespace(fetchall=lambda: list(self._rows))


class _FakeMediaDB:
    """Minimal media DB: serves both reader queries the cycle issues.

    The keyword-aggregation query (``read_media_topics``) gets the seeded
    keyword/score rows; the library-dedupe query (``_library_urls``) gets a
    URL list — here empty, so nothing is filtered as already-ingested.
    """

    def __init__(self, keyword_scores, library_urls=()):
        self._keyword_rows = [
            {"keyword": k, "score": s} for k, s in keyword_scores]
        self._url_rows = [{"url": u} for u in library_urls]

    def execute_query(self, query, params=None):
        if "SELECT url FROM Media" in query:
            rows = self._url_rows
        else:
            rows = self._keyword_rows
        return SimpleNamespace(fetchall=lambda: list(rows))


class _FakePCRecord:
    def __init__(self, subject, kind="note"):
        self.kind = kind
        self.payload = SimpleNamespace(subject=subject)


class _FakePCService:
    def __init__(self, records):
        self._records = records

    def list_records(self, *, scope_ids, include_archived=False):
        return list(self._records)


def _profile_row(db, text, facet="topic"):
    with db.connection() as conn:
        row = conn.execute(
            "SELECT * FROM dream_interest_profile WHERE facet = ? AND text = ?",
            (facet, text),
        ).fetchone()
    return dict(row) if row is not None else None


def _seed_story_with_feedback(db, *, matched, kind, created_at=None):
    """One story plus one feedback row with a CONTROLLED created_at."""
    collection = db.get_collection_by_date("2026-09-20")
    if collection is None:
        collection_id = db.create_collection(
            "2026-09-20", "scheduled", "digest")
    else:
        collection_id = int(collection["id"])
    story_id = db.insert_story(
        collection_id,
        title="t",
        url=f"https://dreams.test/feedback/"
             f"{len(db.list_stories(collection_id))}",
        snippet="s",
        body="b",
        status="complete",
        source="web",
        kind="content",
        event_date=None,
        location=None,
        matched_topics=matched,
        query="q",
    )
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO dream_feedback (story_id, kind, created_at)"
            " VALUES (?, ?, ?)",
            (story_id, kind, created_at or NOW.isoformat()),
        )
    return story_id


def _profile_facts(db):
    """Stable per-key facts (weight/source/boost) for idempotence checks."""
    return {
        (row["facet"], row["text"]): (
            round(float(row["weight"]), 6), row["source"],
            row["last_boosted_at"],
        )
        for row in db.list_profile()
    }


@pytest.mark.asyncio
async def test_cycle_refreshes_profile_signals_from_all_three_sources(
        dreams_db, settings, tmp_path, monkeypatch):
    from tldw_chatbook.Dreams import cycle_service

    monkeypatch.setattr(
        cycle_service, "_pc_distillate_cache_path",
        lambda: tmp_path / "pc_distillate.json")
    # A user-owned row must survive the refresh untouched (ruling R18).
    dreams_db.upsert_profile_entry("topic", "user topic", weight=0.7,
                                   searchable=1, source="user")
    deps = CycleDeps(
        dreams_db=dreams_db,
        chachanotes_db_getter=lambda: _FakeNotesDB([("rust tui", 2)]),
        media_db_getter=lambda: _FakeMediaDB([("jazz guitar", 2)]),
        subs_db_getter=lambda: None,
        pc_service_getter=lambda: _FakePCService(
            [_FakePCRecord("visit japan")]),
        chat_getter=lambda: _fake_chat,
        perform_search=_fake_perform,
        now=lambda: NOW,
    )
    first = await run_cycle(deps, trigger="manual")
    assert first["status"] == "complete"

    facts = _profile_facts(dreams_db)
    # Two note touches -> 0.5; merged rows land per origin with fresh decay
    # clocks (last_boosted_at stamped) so snapshot() does not floor them.
    assert facts[("topic", "rust tui")] == (0.5, "notes", to_utc_iso(NOW))
    assert facts[("topic", "jazz guitar")] == (0.5, "media", to_utc_iso(NOW))
    assert facts[("topic", "visit japan")] == (
        1.0, "personal_context", to_utc_iso(NOW))
    # The pre-seeded user row keeps its weight, source, and boost stamp.
    assert facts[("topic", "user topic")] == (0.7, "user", None)

    # A second cycle neither duplicates rows nor disturbs the user row.
    second = await run_cycle(deps, trigger="refresh")
    assert second["status"] == "complete"
    assert _profile_facts(dreams_db) == facts
    keys = [(row["facet"], row["text"]) for row in dreams_db.list_profile()]
    assert len(keys) == len(set(keys))


@pytest.mark.asyncio
async def test_profile_source_failure_degrades_but_cycle_completes(
        dreams_db, settings):
    class BrokenDB:
        def execute_query(self, query, params=None):
            raise sqlite3.OperationalError("database is locked")

    deps = CycleDeps(
        dreams_db=dreams_db,
        chachanotes_db_getter=lambda: BrokenDB(),
        media_db_getter=lambda: None,
        subs_db_getter=lambda: None,
        pc_service_getter=lambda: None,
        chat_getter=lambda: _fake_chat,
        perform_search=_fake_perform,
        now=lambda: NOW,
    )
    result = await run_cycle(deps, trigger="manual")
    assert result["status"] == "complete"
    row = dreams_db.get_collection_by_date(_today())
    assert row["degradation_notes"] and "notes" in row["degradation_notes"]


@pytest.mark.asyncio
async def test_pc_cache_path_failure_skips_source_with_note(
        dreams_db, settings, monkeypatch):
    from tldw_chatbook.Dreams import cycle_service

    def broken_path():
        raise OSError("no user data dir")

    monkeypatch.setattr(cycle_service, "_pc_distillate_cache_path",
                        broken_path)
    deps = CycleDeps(
        dreams_db=dreams_db,
        chachanotes_db_getter=lambda: _FakeNotesDB([("rust tui", 1)]),
        media_db_getter=lambda: None,
        subs_db_getter=lambda: None,
        pc_service_getter=lambda: _FakePCService([]),
        chat_getter=lambda: _fake_chat,
        perform_search=_fake_perform,
        now=lambda: NOW,
    )
    result = await run_cycle(deps, trigger="manual")
    assert result["status"] == "complete"
    row = dreams_db.get_collection_by_date(_today())
    assert row["degradation_notes"] and "personal context" in \
        row["degradation_notes"]
    # The healthy notes source still landed.
    assert _profile_row(dreams_db, "rust tui") is not None


# --- Feedback loop (Ruling R19) ----------------------------------------------
#
# Feedback is netted per topic over the trailing window and applied by the
# snapshot as an OFFSET -- the stored weight is never rewritten, so a
# reaction counts once per cycle and never compounds.


def _snapshot_weight(db, text, feedback):
    from tldw_chatbook.Dreams.interest_profile import snapshot

    snap = snapshot(db, now_epoch=NOW.timestamp(), feedback=feedback)
    return {t["text"]: t["weight"] for t in snap["topics"]}.get(text)


def _net(db):
    from tldw_chatbook.Dreams.cycle_service import _feedback_net

    return _feedback_net(db, now=NOW)


def _fresh_topic(db, text, weight, *, source="notes", facet="topic"):
    """A topic whose decay clock is NOW, so the snapshot keeps its weight."""
    db.upsert_profile_entry(facet, text, weight=weight, searchable=1,
                            source=source)
    with db.transaction() as conn:
        conn.execute("UPDATE dream_interest_profile SET last_boosted_at = ?"
                     " WHERE facet = ? AND text = ?",
                     (NOW.isoformat(), facet, text))


def test_feedback_more_and_less_offset_the_snapshot_weight(dreams_db, settings):
    _fresh_topic(dreams_db, "rust tui", 0.5)
    _fresh_topic(dreams_db, "jazz guitar", 0.5)
    _seed_story_with_feedback(dreams_db, matched=["rust tui"], kind="more")
    _seed_story_with_feedback(dreams_db, matched=["Jazz Guitar"], kind="less")
    net = _net(dreams_db)
    assert net == {"rust tui": 1, "jazz guitar": -1}
    assert _snapshot_weight(dreams_db, "rust tui", net) == pytest.approx(0.6)
    assert _snapshot_weight(dreams_db, "jazz guitar", net) == \
        pytest.approx(0.4)
    # The stored weight is untouched.
    assert _profile_row(dreams_db, "rust tui")["weight"] == pytest.approx(0.5)


def test_feedback_adjusts_user_topics_but_never_goals(dreams_db, settings):
    from tldw_chatbook.Dreams.interest_profile import snapshot

    _fresh_topic(dreams_db, "rust tui", 0.5, source="user")
    _fresh_topic(dreams_db, "rust tui", 0.9, source="seed", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=["rust tui"], kind="more")
    snap = snapshot(dreams_db, now_epoch=NOW.timestamp(),
                    feedback=_net(dreams_db))
    assert [t["facet"] for t in snap["topics"]] == ["topic"]
    assert snap["topics"][0]["weight"] == pytest.approx(0.6)
    assert _profile_row(dreams_db, "rust tui", facet="goal")["weight"] == \
        pytest.approx(0.9)


def test_feedback_offset_is_clamped(dreams_db, settings):
    _fresh_topic(dreams_db, "ceil topic", 0.95)
    _fresh_topic(dreams_db, "floor topic", 0.1)
    _seed_story_with_feedback(dreams_db, matched=["ceil topic"], kind="more")
    _seed_story_with_feedback(dreams_db, matched=["floor topic"], kind="less")
    net = _net(dreams_db)
    assert _snapshot_weight(dreams_db, "ceil topic", net) == pytest.approx(1.0)
    assert _snapshot_weight(dreams_db, "floor topic", net) == \
        pytest.approx(0.05)


def test_feedback_ignores_old_and_neutral_reactions(dreams_db, settings):
    _seed_story_with_feedback(
        dreams_db, matched=["rust tui"], kind="more",
        created_at=(NOW - timedelta(days=15)).isoformat())
    _seed_story_with_feedback(dreams_db, matched=["rust tui"],
                              kind="exported")
    assert _net(dreams_db) == {}


@pytest.mark.asyncio
async def test_feedback_never_compounds_across_cycles(dreams_db, settings):
    """Regression: one reaction used to re-add +0.1 on EVERY cycle."""
    dreams_db.upsert_profile_entry("topic", "rust tui", weight=0.5,
                                   searchable=1, source="user")
    _seed_story_with_feedback(dreams_db, matched=["rust tui"], kind="more")
    for _ in range(3):
        await run_cycle(_deps(dreams_db), trigger="manual")
    assert _profile_row(dreams_db, "rust tui")["weight"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_cycle_feeds_feedback_into_the_snapshot(dreams_db, settings,
                                                      monkeypatch):
    from tldw_chatbook.Dreams import interest_profile

    seen = []
    real = interest_profile.snapshot

    def spy(db, *, now_epoch, feedback=None):
        seen.append(feedback)
        return real(db, now_epoch=now_epoch, feedback=feedback)

    monkeypatch.setattr(interest_profile, "snapshot", spy)
    dreams_db.upsert_profile_entry("topic", "rust tui", weight=0.5,
                                   searchable=1, source="user")
    _seed_story_with_feedback(dreams_db, matched=["rust tui"], kind="more")
    result = await run_cycle(_deps(dreams_db), trigger="manual")
    assert result["status"] == "complete"
    assert seen == [{"rust tui": 1}]


# --- Budget / pool / ledger regressions (Qodo review) -------------------------


@pytest.mark.asyncio
async def test_full_collection_refresh_spends_nothing(dreams_db, settings):
    _seed_topics(dreams_db)
    first = await run_cycle(_deps(dreams_db), trigger="manual")
    assert first["stories"] == 5
    before = dreams_db.usage_get(_today())
    calls = []

    def counting_chat(**kwargs):
        calls.append(kwargs)
        return _fake_chat(**kwargs)

    second = await run_cycle(_deps(dreams_db, chat=counting_chat),
                             trigger="refresh")
    assert second == {"collection_id": first["collection_id"],
                      "status": "complete", "stories": 5}
    assert calls == []
    assert dreams_db.usage_get(_today()) == before


@pytest.mark.asyncio
async def test_zero_llm_budget_makes_no_synthesis_call(dreams_db, settings):
    _seed_topics(dreams_db)
    settings["max_llm_calls_per_day"] = 0
    calls = []

    def counting_chat(**kwargs):
        calls.append(kwargs)
        return _fake_chat(**kwargs)

    await run_cycle(_deps(dreams_db, chat=counting_chat), trigger="manual")
    assert calls == []
    assert dreams_db.usage_get(_today())["llm_calls"] == 0
    row = dreams_db.get_collection_by_date(_today())
    assert "llm budget exhausted" in row["degradation_notes"]


@pytest.mark.asyncio
async def test_watchlist_pool_survives_an_exhausted_search_budget(
        dreams_db, settings, monkeypatch):
    from tldw_chatbook.Dreams import cycle_service, discovery

    _seed_topics(dreams_db)
    settings["max_searches_per_day"] = 0
    item = discovery.Candidate("rust tui watch item",
                               "https://watch.test/1", "", "watchlist")
    monkeypatch.setattr(discovery, "fetch_watchlist_candidates",
                        lambda *a, **k: [item])
    deps = _deps(dreams_db)
    deps.subs_db_getter = lambda: object()
    await cycle_service.run_cycle(deps, trigger="manual")
    row = dreams_db.get_collection_by_date(_today())
    assert [s["url"] for s in dreams_db.list_stories(row["id"])] == \
        ["https://watch.test/1"]


@pytest.mark.asyncio
async def test_cycle_prunes_the_seen_ledger_by_ttl(dreams_db, settings):
    settings["seen_item_ttl_days"] = 30
    dreams_db.seen_upsert([("https://old.test/a", "d")])
    with dreams_db.transaction() as conn:
        conn.execute("UPDATE dream_seen_items SET last_seen = ?",
                     ((NOW - timedelta(days=31)).isoformat(),))
    await run_cycle(_deps(dreams_db), trigger="manual")
    assert dreams_db.seen_filter_unseen(["https://old.test/a"]) == \
        {"https://old.test/a"}


@pytest.mark.asyncio
async def test_seen_ledger_catches_url_variants(dreams_db, settings):
    _seed_topics(dreams_db)
    dreams_db.seen_upsert([("https://dreams.test/rust-tui-news/0", "d")])

    def variant_perform(search_engine, search_query, **kwargs):
        return {"results": [{
            "title": "rust tui variant",
            "url": "https://DREAMS.test/rust-tui-news/0/?utm_source=x",
            "content": "rust tui"}]}

    await run_cycle(_deps(dreams_db, perform=variant_perform),
                    trigger="manual")
    row = dreams_db.get_collection_by_date(_today())
    assert dreams_db.list_stories(row["id"]) == []


@pytest.mark.asyncio
async def test_llm_fallback_story_does_not_repeat_next_day(dreams_db, settings):
    _seed_topics(dreams_db)

    def dead_perform(search_engine, search_query, **kwargs):
        return {"results": [], "processing_error": "down"}

    await run_cycle(_deps(dreams_db, perform=dead_perform), trigger="manual")
    day_one = dreams_db.get_collection_by_date(_today())
    urls = {s["url"] for s in dreams_db.list_stories(day_one["id"])}
    assert urls and all(u.startswith("dreams://llm/") for u in urls)

    tomorrow = NOW + timedelta(days=1)
    deps = _deps(dreams_db, perform=dead_perform)
    deps.now = lambda: tomorrow
    await run_cycle(deps, trigger="manual")
    day_two = dreams_db.get_collection_by_date(
        tomorrow.astimezone().strftime("%Y-%m-%d"))
    assert not urls & {s["url"] for s in dreams_db.list_stories(day_two["id"])}
