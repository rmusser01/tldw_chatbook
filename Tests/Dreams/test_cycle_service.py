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
    database.close_all()  # every to_thread hop opened its own connection


@pytest.fixture()
def settings(monkeypatch):
    """Deterministic [dreams] settings; defaults flow through when unset.

    ``enabled`` starts True: ``run_cycle`` itself refuses to run -- and
    stops a run in flight -- while the master switch is off (Qodo #15,
    PR #2921), so the shipped default (off) would make every cycle test
    a no-op. Tests about the switch set it explicitly.
    """
    overrides: dict = {"enabled": True}

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
    # ... and with no llm slot left for a story, nothing is searched
    # either (review round 3, item 4).
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 1}
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
    # The budget is checked BEFORE synthesis: nothing re-spent at all --
    # no model call, and (review round 3, item 4) no search whose results
    # could not become a story.
    assert dreams_db.usage_get(_today()) == {
        "searches": 3, "llm_calls": 4}


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


def _seed_story_with_feedback(db, *, matched, kind, created_at=None,
                              story_kind="content", goal=None):
    """One story plus one feedback row with a CONTROLLED created_at.

    ``goal`` is the story's goal provenance (the goal whose query surfaced
    it), kept apart from the topics it merely matched.
    """
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
        kind=story_kind,
        event_date=None,
        location=None,
        matched_topics=matched,
        query="q",
        goal=goal,
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


def test_feedback_tracked_reaction_is_positive(dreams_db, settings):
    """Phase 2 Track flip: ``tracked`` now nets +1 (+0.1 snapshot weight)."""
    _fresh_topic(dreams_db, "tracked topic", 0.5)
    _seed_story_with_feedback(dreams_db, matched=["tracked topic"],
                              kind="tracked")
    net = _net(dreams_db)
    assert net == {"tracked topic": 1}
    assert _snapshot_weight(dreams_db, "tracked topic", net) == \
        pytest.approx(0.6)
    # The stored weight is untouched (offset, never rewrite).
    assert _profile_row(dreams_db, "tracked topic")["weight"] == \
        pytest.approx(0.5)


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


# --- Goal query-angle feedback (task-33165) -----------------------------------
#
# The OTHER feedback half: goals never change weight; feedback on a story
# a GOAL's own query surfaced (the story row's ``goal`` provenance) steers
# that goal's QUERY ANGLE ("avoid: {story kind}" on less, "prefer: {story
# kind}" on any positive kind; the newest non-neutral reaction replaces
# whatever came before).


def _angles(db):
    from tldw_chatbook.Dreams.cycle_service import _apply_goal_angles

    _apply_goal_angles(db, now=NOW)


def _goal_angle(db, text):
    row = _profile_row(db, text, facet="goal")
    return None if row is None else row["query_angle"]


def test_goal_less_on_event_story_writes_avoid_angle_weight_immune(
        dreams_db, settings):
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="less",
                              story_kind="event")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") == "avoid: event"
    # The goal's weight is untouched: angles steer, weights stay immune.
    assert _profile_row(dreams_db, "visit japan",
                        facet="goal")["weight"] == pytest.approx(0.9)


def test_goal_positive_kinds_write_prefer_angle(dreams_db, settings):
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="more",
                              story_kind="deal")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") == "prefer: deal"


def test_goal_angle_positive_replaces_negative_and_back(dreams_db, settings):
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="less",
                              story_kind="event")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="kept",
                              story_kind="deal")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") == "prefer: deal"
    # And a negative after a positive replaces it right back.
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="less",
                              story_kind="social_opportunity")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") == "avoid: social_opportunity"


def test_goal_angle_second_less_with_different_kind_replaces(
        dreams_db, settings):
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="less",
                              story_kind="event")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="less",
                              story_kind="deal")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") == "avoid: deal"


def test_goal_angle_unknown_story_kind_maps_to_content(dreams_db, settings):
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="less",
                              story_kind="unknown")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") == "avoid: content"


def test_goal_angle_exported_reaction_is_neutral(dreams_db, settings):
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan",
                              kind="exported", story_kind="event")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") is None


def test_topic_only_stories_never_touch_goal_angles(dreams_db, settings):
    """A story matching only TOPIC rows steers no goal, ever."""
    _fresh_topic(dreams_db, "rust tui", 0.5, source="user")
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=["rust tui"], kind="less",
                              story_kind="event")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") is None
    assert _profile_row(dreams_db, "visit japan",
                        facet="goal")["weight"] == pytest.approx(0.9)


def test_goal_angle_joins_on_the_exact_stored_goal_text(dreams_db, settings):
    """Provenance joins on the goal's exact stored text -- the profile's own
    key, and what the cycle records on the story (review round 3, N3). A
    recorded goal that differs in case is a different goal (or one that no
    longer exists) and steers nothing."""
    _fresh_topic(dreams_db, "Visit Japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="more",
                              story_kind="event")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "Visit Japan") is None

    _seed_story_with_feedback(dreams_db, matched=[], goal="Visit Japan", kind="more",
                              story_kind="event")
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "Visit Japan") == "prefer: event"


def test_goal_angle_ignores_feedback_outside_the_window(dreams_db, settings):
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(
        dreams_db, matched=[], goal="visit japan", kind="less", story_kind="event",
        created_at=(NOW - timedelta(days=15)).isoformat())
    _angles(dreams_db)
    assert _goal_angle(dreams_db, "visit japan") is None


@pytest.mark.asyncio
async def test_cycle_writes_goal_angle_and_goal_weight_stays_immune(
        dreams_db, settings):
    """End to end: the cycle's feedback stage steers the goal's angle."""
    _seed_topics(dreams_db)
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="less",
                              story_kind="event")
    result = await run_cycle(_deps(dreams_db), trigger="manual")
    assert result["status"] == "complete"
    goal = _profile_row(dreams_db, "visit japan", facet="goal")
    assert goal["query_angle"] == "avoid: event"
    assert goal["weight"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_goal_angle_failure_degrades_without_killing_the_cycle(
        dreams_db, settings, monkeypatch):
    """A dead angle write is a note; the topic offset and cycle survive."""
    from tldw_chatbook.Dreams import interest_profile

    _seed_topics(dreams_db)
    _fresh_topic(dreams_db, "rust tui", 0.5, source="user")
    _fresh_topic(dreams_db, "visit japan", 0.9, source="user", facet="goal")
    _seed_story_with_feedback(dreams_db, matched=["rust tui"], kind="more")
    _seed_story_with_feedback(dreams_db, matched=[], goal="visit japan", kind="less",
                              story_kind="event")

    def boom(*args, **kwargs):
        raise RuntimeError("angle write refused")

    monkeypatch.setattr(DreamsDB, "set_goal_query_angle", boom)
    seen = []
    real = interest_profile.snapshot

    def spy(db, *, now_epoch, feedback=None):
        seen.append(feedback)
        return real(db, now_epoch=now_epoch, feedback=feedback)

    monkeypatch.setattr(interest_profile, "snapshot", spy)
    result = await run_cycle(_deps(dreams_db), trigger="manual")
    assert result["status"] == "complete"
    row = dreams_db.get_collection_by_date(_today())
    assert "goal angle feedback failed" in row["degradation_notes"]
    # The topic net still reached the snapshot: the angle failure took
    # nothing else down with it. (Goal provenance is its own column, so a
    # goal's text is no longer part of the topic net at all.)
    assert seen == [{"rust tui": 1}]


# --- Goal attribution on generated stories (review of #2921, item 1) -----------
#
# A story is attributed to a goal by PROVENANCE: the goal whose own search
# query surfaced it, carried from query synthesis through the search into
# the story row's ``goal`` column. It used to be inferred by fuzzy overlap
# between the goal's text and the story (any shared word, stopwords
# included), so "plan a trip to Japan in the spring" matched "How to fix a
# flat bike tire in the rain" and one reaction to any story rewrote the
# angle of every goal containing a/to/the/in.

_GOAL_JAPAN = "plan a trip to Japan in the spring"
_GOAL_CELLO = "learn to play the cello"


def _seed_two_goals(db) -> None:
    _seed_topics(db)
    _fresh_topic(db, _GOAL_JAPAN, 0.9, source="user", facet="goal")
    _fresh_topic(db, _GOAL_CELLO, 0.9, source="user", facet="goal")


def _tagging_chat(**kwargs):
    """Synthesis tags the one line it wrote for a goal; stories are plain.

    Goals reach the model heaviest-first, ties by text, so "learn to play
    the cello" is goal 1 and the Japan goal is goal 2.
    """
    if "search queries" in str(kwargs.get("system_message") or ""):
        return {"choices": [{"message": {"content":
            "bike repair tips\n"
            "[g2] japan spring travel deals\n"
            "surprising robots research"}}]}
    return _fake_chat(**kwargs)


def _bike_and_japan_perform(search_engine, search_query, **kwargs):
    if "japan" in search_query.lower():
        return {"results": [{
            "title": "Japan rail pass sale",
            "url": "https://dreams.test/japan",
            "content": "Discounted tickets for a spring trip."}]}
    if "bike" in search_query:
        return {"results": [{
            "title": "How to fix a flat bike tire in the rain",
            "url": "https://dreams.test/bike",
            "content": "A guide to patching a tube on the road."}]}
    return {"results": []}


def _stories_by_url(db, collection_id) -> dict:
    return {story["url"]: story for story in db.list_stories(collection_id)}


def _both_goal_angles(db) -> tuple:
    return (_goal_angle(db, _GOAL_JAPAN), _goal_angle(db, _GOAL_CELLO))


@pytest.mark.asyncio
async def test_unrelated_story_is_attributed_to_no_goal_and_steers_none(
        dreams_db, settings):
    """The reviewer's probe, as a regression: a story no goal's query
    surfaced names no goal -- however many words it shares with one -- and
    feedback on it changes no goal's angle."""
    from tldw_chatbook.Dreams.cycle_service import _apply_goal_angles

    _seed_two_goals(dreams_db)
    result = await run_cycle(
        _deps(dreams_db, perform=_bike_and_japan_perform, chat=_tagging_chat),
        trigger="manual")
    bike = _stories_by_url(dreams_db, result["collection_id"])[
        "https://dreams.test/bike"]

    assert bike["goal"] is None
    assert _GOAL_JAPAN.lower() not in bike["matched_topics"]
    assert _GOAL_CELLO.lower() not in bike["matched_topics"]
    assert bike["query"] == "bike repair tips", (
        "the story records the query that actually surfaced it"
    )

    dreams_db.record_feedback(bike["id"], "less")
    _apply_goal_angles(dreams_db, now=datetime.now(UTC))

    assert _both_goal_angles(dreams_db) == (None, None)


@pytest.mark.asyncio
async def test_story_from_a_goals_query_steers_exactly_that_goal(
        dreams_db, settings):
    from tldw_chatbook.Dreams.cycle_service import _apply_goal_angles

    _seed_two_goals(dreams_db)
    result = await run_cycle(
        _deps(dreams_db, perform=_bike_and_japan_perform, chat=_tagging_chat),
        trigger="manual")
    japan = _stories_by_url(dreams_db, result["collection_id"])[
        "https://dreams.test/japan"]

    assert japan["goal"] == _GOAL_JAPAN
    assert japan["query"] == "japan spring travel deals", (
        "the provenance tag never reaches the search or the story row"
    )
    assert _GOAL_JAPAN.lower() not in japan["matched_topics"], (
        "goal provenance is not a topic match: it stays off the topics line"
    )

    dreams_db.record_feedback(japan["id"], "less")
    _apply_goal_angles(dreams_db, now=datetime.now(UTC))

    assert _both_goal_angles(dreams_db) == (f"avoid: {japan['kind']}", None)
    assert _profile_row(dreams_db, _GOAL_JAPAN,
                        facet="goal")["weight"] == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_no_goal_mark_or_list_marker_ever_reaches_the_search(
        dreams_db, settings):
    """Review round 3, item 1 -- the reviewer's probe as a regression. A
    model that numbers and bullets its lines used to have ``1. rust tui
    news`` and ``2. [g2] japan rail pass deals`` sent to the search provider
    verbatim, and the goal's story lost its goal."""
    _seed_two_goals(dreams_db)
    searched: list[str] = []

    def chat(**kwargs):
        if "search queries" in str(kwargs.get("system_message") or ""):
            return {"choices": [{"message": {"content":
                "1. bike repair tips\n"
                '2. "[g2] japan rail pass deals"\n'
                "- **[g1]** beginner cello lessons"}}]}
        return _fake_chat(**kwargs)

    def perform(search_engine, search_query, **kwargs):
        searched.append(search_query)
        return _bike_and_japan_perform(search_engine, search_query, **kwargs)

    result = await run_cycle(
        _deps(dreams_db, perform=perform, chat=chat), trigger="manual")

    assert searched == [
        "bike repair tips", "japan rail pass deals",
        "beginner cello lessons"]
    stories = _stories_by_url(dreams_db, result["collection_id"])
    japan = stories["https://dreams.test/japan"]
    assert (japan["query"], japan["goal"]) == (
        "japan rail pass deals", _GOAL_JAPAN)
    assert stories["https://dreams.test/bike"]["goal"] is None


@pytest.mark.asyncio
async def test_page_found_by_a_topic_query_and_a_goal_query_names_the_goal(
        dreams_db, settings):
    """Review round 3, N2, through the cycle: the topic query comes first
    and finds the same page the goal's query finds."""
    _seed_two_goals(dreams_db)

    def chat(**kwargs):
        if "search queries" in str(kwargs.get("system_message") or ""):
            return {"choices": [{"message": {"content":
                "japan travel news\n"
                "[g2] japan spring travel deals"}}]}
        return _fake_chat(**kwargs)

    result = await run_cycle(
        _deps(dreams_db, perform=_bike_and_japan_perform, chat=chat),
        trigger="manual")
    japan = _stories_by_url(dreams_db, result["collection_id"])[
        "https://dreams.test/japan"]

    assert japan["goal"] == _GOAL_JAPAN
    assert japan["query"] == "japan spring travel deals"


@pytest.mark.asyncio
async def test_fallback_goal_query_carries_its_goal_without_any_tag(
        dreams_db, settings):
    """With synthesis down the cycle searches the deterministic lines, and
    a goal's line IS that goal's query: provenance is exact, no model
    involved."""
    _seed_two_goals(dreams_db)

    def chat(**kwargs):
        if "search queries" in str(kwargs.get("system_message") or ""):
            raise RuntimeError("provider down")
        return _fake_chat(**kwargs)

    def perform(search_engine, search_query, **kwargs):
        if search_query.startswith(_GOAL_CELLO):
            return {"results": [{
                "title": "Beginner cello workshop", "content": "Six weeks.",
                "url": "https://dreams.test/cello"}]}
        return {"results": []}

    settings["queries_per_cycle"] = 6
    result = await run_cycle(
        _deps(dreams_db, perform=perform, chat=chat), trigger="manual")

    cello = _stories_by_url(dreams_db, result["collection_id"])[
        "https://dreams.test/cello"]
    assert cello["goal"] == _GOAL_CELLO


@pytest.mark.asyncio
async def test_topic_match_never_steers_a_goal_with_the_same_text(
        dreams_db, settings):
    """A topic and a goal may carry the same text. A story that merely
    matched the TOPIC must not steer the goal: provenance is kept apart
    from topic matches for exactly this reason."""
    from tldw_chatbook.Dreams.cycle_service import _apply_goal_angles

    _seed_topics(dreams_db)  # includes the topic "rust tui"
    _fresh_topic(dreams_db, "rust tui", 0.9, source="user", facet="goal")
    result = await run_cycle(_deps(dreams_db), trigger="manual")
    story = dreams_db.list_stories(result["collection_id"])[0]
    assert "rust tui" in story["matched_topics"], "fixture: the topic matched"
    assert story["goal"] is None

    dreams_db.record_feedback(story["id"], "less")
    _apply_goal_angles(dreams_db, now=datetime.now(UTC))

    assert _goal_angle(dreams_db, "rust tui") is None


@pytest.mark.asyncio
async def test_generated_story_never_records_an_unsearchable_goal(
        dreams_db, settings):
    """The privacy gate holds here too: a goal whose text may not leave the
    machine is never sent to synthesis, so the model cannot tag it, the
    fallback writes no line for it, and no story row carries its text."""
    _seed_topics(dreams_db)
    dreams_db.upsert_profile_entry(
        "goal", "secretly visit japan", weight=1.0, searchable=0,
        source="user")
    result = await run_cycle(
        _deps(dreams_db, perform=_bike_and_japan_perform, chat=_tagging_chat),
        trigger="manual")

    stories = dreams_db.list_stories(result["collection_id"])
    assert stories, "fixture: the cycle wrote stories"
    for story in stories:
        assert story["goal"] is None, (
            "the tag named a goal position the model was never given"
        )
        assert "secretly visit japan" not in story["matched_topics"]


# --- The master switch reaches a cycle already running (Qodo #15, PR #2921) ----
#
# Every caller checks ``[dreams] enabled`` before it STARTS a cycle. A cycle
# takes minutes (searches, one model call per story), and the Settings
# toggle only wrote the flag: a cycle already in flight kept searching and
# calling the model after the user had switched Dreams off.


class _CountingChat:
    """``chat_api_call``-shaped fake counting synthesis and story calls."""

    def __init__(self, on_synthesis=None, on_story=None):
        self.synthesis_calls = 0
        self.story_calls = 0
        self._on_synthesis = on_synthesis
        self._on_story = on_story

    def __call__(self, **kwargs):
        if "search queries" in str(kwargs.get("system_message") or ""):
            self.synthesis_calls += 1
            if self._on_synthesis is not None:
                self._on_synthesis()
        else:
            self.story_calls += 1
            if self._on_story is not None:
                self._on_story()
        return _fake_chat(**kwargs)


@pytest.mark.asyncio
async def test_run_cycle_does_not_start_while_dreams_is_disabled(
        dreams_db, settings):
    settings["enabled"] = False
    _seed_topics(dreams_db)
    chat = _CountingChat()
    searches: list[str] = []

    def perform(search_engine, search_query, **kwargs):
        searches.append(search_query)
        return _fake_perform(search_engine, search_query, **kwargs)

    result = await run_cycle(
        _deps(dreams_db, perform=perform, chat=chat), trigger="scheduled")

    assert result == {"collection_id": None, "status": "disabled",
                      "stories": 0}
    assert dreams_db.get_collection_by_date(_today()) is None
    assert (chat.synthesis_calls, chat.story_calls, searches) == (0, 0, [])
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}


@pytest.mark.asyncio
async def test_disabling_dreams_during_the_search_stops_before_any_story(
        dreams_db, settings):
    _seed_topics(dreams_db)
    chat = _CountingChat()

    searches: list[str] = []

    def perform(search_engine, search_query, **kwargs):
        searches.append(search_query)
        settings["enabled"] = False  # the user flips the switch mid-search
        return _fake_perform(search_engine, search_query, **kwargs)

    result = await run_cycle(
        _deps(dreams_db, perform=perform, chat=chat), trigger="scheduled")

    # Review round 3, item 2: the switch is read before EACH search, so the
    # one in flight finishes and the rest of the batch never goes out --
    # and a search that was never sent is never counted.
    assert len(searches) == 1, "the rest of the batch stays unsent"
    assert dreams_db.usage_get(_today())["searches"] == 1
    assert chat.story_calls == 0, "no model call after the switch went off"
    # Review of #2921, item 7: nothing was produced, so the run is reported
    # as not run -- not as a cycle that failed.
    assert result == {"collection_id": None, "status": "disabled",
                      "stories": 0}
    assert dreams_db.get_collection_by_date(_today()) is None, (
        "no collection is left behind, neither generating nor failed"
    )
    assert dreams_db.usage_get(_today())["llm_calls"] == 1, (
        "only the synthesis call that preceded the switch was spent"
    )


@pytest.mark.asyncio
async def test_disabling_dreams_between_stories_stops_the_remaining_ones(
        dreams_db, settings):
    _seed_topics(dreams_db)
    chat = _CountingChat(
        on_story=lambda: settings.__setitem__("enabled", False))

    result = await run_cycle(_deps(dreams_db, chat=chat), trigger="scheduled")

    assert chat.story_calls == 1, "the story in flight finishes; no more start"
    assert result["stories"] == 1
    row = dreams_db.get_collection_by_date(_today())
    assert "disabled mid-cycle" in row["degradation_notes"]


@pytest.mark.asyncio
async def test_disabling_dreams_during_synthesis_stops_before_any_search(
        dreams_db, settings):
    _seed_topics(dreams_db)
    chat = _CountingChat(
        on_synthesis=lambda: settings.__setitem__("enabled", False))
    searches: list[str] = []

    def perform(search_engine, search_query, **kwargs):
        searches.append(search_query)
        return _fake_perform(search_engine, search_query, **kwargs)

    result = await run_cycle(
        _deps(dreams_db, perform=perform, chat=chat), trigger="scheduled")

    assert searches == [], "no search after the switch went off"
    assert chat.story_calls == 0, "and no llm-knowledge stories either"
    assert result == {"collection_id": None, "status": "disabled",
                      "stories": 0}
    assert dreams_db.get_collection_by_date(_today()) is None


@pytest.mark.asyncio
async def test_reenabling_after_a_halted_cycle_does_not_wait_out_a_cadence(
        dreams_db, settings):
    """Review of #2921, item 7. A cycle switched off before its first story
    used to close as ``failed``. The projection retries a failed cycle one
    cadence after the attempt and the boot catch-up skips a date that owns
    a row, so switching Dreams back on found nothing due for a day."""
    from tldw_chatbook.Dreams.cycle_service import _catchup_due
    from tldw_chatbook.Scheduling.services.dreams_projection import (
        DreamsProjection,
    )

    _seed_topics(dreams_db)
    chat = _CountingChat(
        on_synthesis=lambda: settings.__setitem__("enabled", False))
    await run_cycle(_deps(dreams_db, chat=chat), trigger="scheduled")

    settings["enabled"] = True  # switched back on
    wall_now = datetime.now(UTC)
    cycle_task = DreamsProjection(lambda: dreams_db).tasks(wall_now)[0]
    assert cycle_task.next_run_at <= wall_now, "due on the next tick"
    assert _catchup_due(
        dreams_db, local_date=_today(), cadence_hours=24, now=NOW)

    # And the day is still open: the next cycle runs normally.
    result = await run_cycle(_deps(dreams_db), trigger="scheduled")
    assert result["status"] == "complete" and result["stories"] > 0


@pytest.mark.asyncio
async def test_halted_append_leaves_the_existing_collection_as_it_was(
        dreams_db, settings):
    """A later trigger for a date that already has stories, switched off
    before it adds one: the collection it would have appended to is not
    re-stamped (a fresh ``completed_at`` would push the next run out by a
    cadence) and gains no note about a run that added nothing."""
    _seed_topics(dreams_db)
    settings["stories_per_cycle"] = 1
    first = await run_cycle(_deps(dreams_db), trigger="scheduled")
    assert first["status"] == "complete" and first["stories"] == 1
    before = dreams_db.get_collection_by_date(_today())
    settings["stories_per_cycle"] = 3
    chat = _CountingChat(
        on_synthesis=lambda: settings.__setitem__("enabled", False))

    second = await run_cycle(_deps(dreams_db, chat=chat), trigger="refresh")

    assert chat.story_calls == 0
    assert second == {"collection_id": before["id"], "status": "complete",
                      "stories": 1}
    assert dreams_db.get_collection_by_date(_today()) == before


@pytest.mark.asyncio
async def test_a_cycle_does_not_write_into_a_collection_that_took_its_id(
        dreams_db, settings):
    """Review round 4, I5. Two app processes on one profile are possible.
    While this cycle is writing its first story, "the other process" --
    which owned the still-empty row -- deletes it, and a new collection is
    created that gets the SAME id. The late story used to land in it."""
    import sqlite3

    _seed_topics(dreams_db)
    taken_over: list[int] = []

    def chat(**kwargs):
        system = str(kwargs.get("system_message") or "")
        if "search queries" in system:
            return _fake_chat(**kwargs)
        if not taken_over:
            row = dreams_db.get_collection_by_date(_today())
            assert dreams_db.delete_empty_collection(int(row["id"])) is True
            with dreams_db.transaction() as conn:
                conn.execute(
                    "INSERT INTO dreams_collections (id, local_date, status,"
                    " profile_digest, trigger, created_at) VALUES"
                    " (?, ?, 'generating', 'other', 'catchup',"
                    " '2000-01-01T00:00:00.000Z')", (int(row["id"]), _today()))
            taken_over.append(int(row["id"]))
        return _fake_chat(**kwargs)

    with pytest.raises(sqlite3.IntegrityError):
        await run_cycle(_deps(dreams_db, chat=chat), trigger="manual")

    assert taken_over, "fixture: the id was taken over mid-cycle"
    other = dreams_db.get_collection_by_date(_today())
    assert other["profile_digest"] == "other" and other["status"] == "generating"
    assert dreams_db.list_stories(taken_over[0]) == [], (
        "the other collection got none of this cycle's stories"
    )


def test_delete_empty_collection_never_removes_a_collection_with_stories(
        dreams_db):
    """The halted-cycle cleanup can only ever remove an EMPTY row that is
    still ``generating`` -- or that another cycle's stale sweep has turned
    into ``failed`` meanwhile (round 3, N5) -- never one holding stories,
    and never one closed as ``complete`` or ``partial``."""
    empty = dreams_db.create_collection("2026-09-20", "manual", "digest")
    assert dreams_db.delete_empty_collection(empty) is True
    assert dreams_db.get_collection_by_date("2026-09-20") is None

    holding = dreams_db.create_collection("2026-09-21", "manual", "digest")
    dreams_db.insert_story(
        holding, title="t", url="https://dreams.test/s", snippet="", body="b",
        status="complete", source="web", kind="content", event_date=None,
        location=None, matched_topics=[], query="q")
    assert dreams_db.delete_empty_collection(holding) is False

    closed = dreams_db.create_collection("2026-09-19", "manual", "digest")
    dreams_db.set_collection_status(closed, "complete", story_count=0)
    assert dreams_db.delete_empty_collection(closed) is False
    assert dreams_db.get_collection_by_date("2026-09-21") is not None
    assert dreams_db.get_collection_by_date("2026-09-19") is not None

    # Round 3, N5: the halted cycle's own row may have been reclaimed to
    # ``failed`` by another cycle's stale sweep in the meantime; empty, it
    # is still the same nothing and is handed back just the same.
    reclaimed = dreams_db.create_collection("2026-09-18", "manual", "digest")
    dreams_db.fail_stale_generating("9999-01-01T00:00:00+00:00")
    assert dreams_db.get_collection_by_date("2026-09-18")["status"] == "failed"
    assert dreams_db.delete_empty_collection(reclaimed) is True
    assert dreams_db.get_collection_by_date("2026-09-18") is None
    # ... and the sweep that reclaimed it also failed the row holding a
    # story: failed or not, a row with a story is never removed.
    assert dreams_db.get_collection_by_date("2026-09-21")["status"] == "failed"
    assert dreams_db.delete_empty_collection(holding) is False


@pytest.mark.asyncio
async def test_halted_cycle_hands_its_date_back_even_after_a_stale_reclaim(
        dreams_db, settings):
    """Round 3, N5 through the cycle: a long-running cycle's row is
    reclaimed to ``failed`` (another date's cycle swept it as stale), then
    the user switches Dreams off before its first story. The failed row
    must not stay behind -- it would bring the cadence wait back."""
    from tldw_chatbook.Dreams.cycle_service import _catchup_due

    _seed_topics(dreams_db)

    def perform(search_engine, search_query, **kwargs):
        dreams_db.fail_stale_generating("9999-01-01T00:00:00+00:00")
        settings["enabled"] = False
        return _fake_perform(search_engine, search_query, **kwargs)

    result = await run_cycle(
        _deps(dreams_db, perform=perform), trigger="scheduled")

    assert result == {"collection_id": None, "status": "disabled",
                      "stories": 0}
    assert dreams_db.get_collection_by_date(_today()) is None
    settings["enabled"] = True
    assert _catchup_due(
        dreams_db, local_date=_today(), cadence_hours=24, now=NOW)


@pytest.mark.asyncio
async def test_disabling_dreams_before_synthesis_spends_nothing_at_all(
        dreams_db, settings):
    _seed_topics(dreams_db)
    chat = _CountingChat()
    searches: list[str] = []

    def perform(search_engine, search_query, **kwargs):
        searches.append(search_query)
        return _fake_perform(search_engine, search_query, **kwargs)

    deps = _deps(dreams_db, perform=perform, chat=chat)

    def notes_db_getter():
        settings["enabled"] = False  # switched off during the profile stage
        return None

    deps.chachanotes_db_getter = notes_db_getter

    result = await run_cycle(deps, trigger="scheduled")

    assert (chat.synthesis_calls, chat.story_calls, searches) == (0, 0, [])
    assert result["stories"] == 0
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}


# --- A cycle and a track check share one daily budget (review item 6) -----------
#
# Track checks reserve their budget slots atomically. The cycle still read
# the day's usage, did the work, and bumped the counter afterwards -- so a
# check running while the cycle was mid-search (or mid-story) saw the slots
# as free, took one, and the day ended over both caps.


class _ParkedOnce:
    """A sync callable that parks its FIRST call in the worker thread."""

    def __init__(self, inner):
        self._inner = inner
        self.started = threading.Event()
        self.release = threading.Event()

    def __call__(self, *args, **kwargs):
        if not self.started.is_set():
            self.started.set()
            assert self.release.wait(timeout=10)
        return self._inner(*args, **kwargs)


async def _until_parked(parked: _ParkedOnce) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5
    while not parked.started.is_set() and loop.time() < deadline:
        await asyncio.sleep(0.01)
    assert parked.started.is_set(), "the cycle never reached the parked call"


def _check_search(search_engine, search_query, **kwargs):
    return {"results": [{"title": "Flights round-up", "content": "new dates",
                         "url": "https://dreams.test/watched"}]}


@pytest.mark.asyncio
async def test_cycle_and_check_cannot_both_spend_the_last_search(
        dreams_db, settings):
    from tldw_chatbook.Dreams.track_service import (
        run_track_check,
        track_question,
    )

    settings["max_searches_per_day"] = 1  # one slot for the whole day
    _seed_topics(dreams_db)
    item_id = await track_question(
        dreams_db, query_template="flights to kyoto", intent="deal")
    parked = _ParkedOnce(_fake_perform)

    cycle = asyncio.create_task(
        run_cycle(_deps(dreams_db, perform=parked), trigger="manual"))
    await _until_parked(parked)  # the cycle is inside the day's one search
    check = await run_track_check(
        _deps(dreams_db, perform=_check_search), item_id)
    parked.release.set()
    await cycle

    assert check == {"status": "skipped", "notified": False}, (
        "the slot was the cycle's from the moment its search started"
    )
    assert dreams_db.usage_get(_today())["searches"] == 1


@pytest.mark.asyncio
async def test_cycle_and_check_together_never_search_past_the_cap(
        dreams_db, settings):
    """Review round 3, item 2: the cycle takes ONE search slot right before
    each search (it used to reserve the whole batch up front). A check that
    runs while the cycle is mid-batch may take a slot the cycle had planned
    on; what must hold is that every search made was counted first and the
    two of them together stop at the cap."""
    from tldw_chatbook.Dreams.track_service import (
        run_track_check,
        track_question,
    )

    settings["max_searches_per_day"] = 3  # the cycle plans all three
    _seed_topics(dreams_db)
    item_id = await track_question(
        dreams_db, query_template="flights to kyoto", intent="deal")
    cycle_searches: list[str] = []
    check_searches: list[str] = []

    def cycle_search(search_engine, search_query, **kwargs):
        cycle_searches.append(search_query)
        return _fake_perform(search_engine, search_query, **kwargs)

    def check_search(search_engine, search_query, **kwargs):
        check_searches.append(search_query)
        return _check_search(search_engine, search_query, **kwargs)

    parked = _ParkedOnce(cycle_search)
    cycle = asyncio.create_task(
        run_cycle(_deps(dreams_db, perform=parked), trigger="manual"))
    await _until_parked(parked)
    await run_track_check(_deps(dreams_db, perform=check_search), item_id)
    parked.release.set()
    await cycle

    assert len(check_searches) == 1, "a slot was free: the check took it"
    assert len(cycle_searches) == 2, "so the cycle stops one search short"
    assert dreams_db.usage_get(_today())["searches"] == 3
    row = dreams_db.get_collection_by_date(_today())
    assert "search budget" in row["degradation_notes"]


@pytest.mark.asyncio
async def test_no_search_is_made_when_no_llm_slot_is_left_for_a_story(
        dreams_db, settings):
    """Review round 3, item 4: with the day's llm budget gone the cycle
    used to run all its searches, count them, and throw the results away
    -- a story needs a model call. The track check has the same guard."""
    settings["max_llm_calls_per_day"] = 0
    _seed_topics(dreams_db)
    sent: list[str] = []

    def perform(search_engine, search_query, **kwargs):
        sent.append(search_query)
        return _fake_perform(search_engine, search_query, **kwargs)

    await run_cycle(_deps(dreams_db, perform=perform), trigger="manual")

    assert sent == []
    assert dreams_db.usage_get(_today()) == {"searches": 0, "llm_calls": 0}
    row = dreams_db.get_collection_by_date(_today())
    assert "no searches made" in row["degradation_notes"]
    assert "web search disabled" not in row["degradation_notes"]


@pytest.mark.asyncio
async def test_cycle_searches_only_what_the_cap_allows_after_its_plan(
        dreams_db, settings, monkeypatch):
    """Review round 3, item 3a: the plan (how many queries to aim for) is
    read once; another Dreams task then spends two of the three slots the
    plan saw. The cycle must perform exactly the one search the cap still
    allows -- not the three it planned."""
    settings["max_searches_per_day"] = 3
    _seed_topics(dreams_db)
    sent: list[str] = []

    def perform(search_engine, search_query, **kwargs):
        sent.append(search_query)
        return _fake_perform(search_engine, search_query, **kwargs)

    plan_read = dreams_db.usage_get
    spent_once: list[int] = []

    def read_then_another_task_spends(local_date):
        usage = plan_read(local_date)
        if not spent_once:
            spent_once.append(1)
            dreams_db.usage_bump(local_date, searches=2)
        return usage

    monkeypatch.setattr(dreams_db, "usage_get", read_then_another_task_spends)

    await run_cycle(_deps(dreams_db, perform=perform), trigger="manual")

    assert len(sent) == 1
    assert plan_read(_today())["searches"] == 3, "never past the cap"
    row = dreams_db.get_collection_by_date(_today())
    assert "search budget: stopped after 1 of 3" in row["degradation_notes"]


@pytest.mark.asyncio
async def test_cancelled_cycle_counts_only_the_searches_it_sent(
        dreams_db, settings):
    """Review round 3, N7: a cycle cancelled inside its first search used to
    leave the two searches it never sent counted against the day."""
    _seed_topics(dreams_db)
    sent: list[str] = []

    def perform(search_engine, search_query, **kwargs):
        sent.append(search_query)
        return _fake_perform(search_engine, search_query, **kwargs)

    parked = _ParkedOnce(perform)
    cycle = asyncio.create_task(
        run_cycle(_deps(dreams_db, perform=parked), trigger="manual"))
    await _until_parked(parked)
    cycle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cycle
    parked.release.set()
    # The search in flight cannot be recalled: its worker thread finishes.
    for _ in range(500):
        if sent:
            break
        await asyncio.sleep(0.01)

    assert len(sent) == 1
    assert dreams_db.usage_get(_today())["searches"] == 1


@pytest.mark.asyncio
async def test_cycle_and_check_cannot_both_spend_the_last_llm_call(
        dreams_db, settings):
    from tldw_chatbook.Dreams.track_service import (
        run_track_check,
        track_question,
    )

    settings["max_llm_calls_per_day"] = 2  # synthesis + exactly one story
    _seed_topics(dreams_db)
    item_id = await track_question(
        dreams_db, query_template="flights to kyoto", intent="deal")
    dreams_db.insert_track_run(
        item_id, status="baseline", digest_hash="an-older-result-set")
    judge_calls: list[dict] = []

    def judge(**kwargs):
        judge_calls.append(kwargs)
        return {"choices": [{"message": {
            "content": '{"changed": false, "note": "same"}'}}]}

    def cycle_chat(**kwargs):
        if "search queries" in str(kwargs.get("system_message") or ""):
            return _fake_chat(**kwargs)
        return story_call(**kwargs)

    story_call = _ParkedOnce(_fake_chat)

    cycle = asyncio.create_task(
        run_cycle(_deps(dreams_db, chat=cycle_chat), trigger="manual"))
    await _until_parked(story_call)
    check = await run_track_check(
        _deps(dreams_db, perform=_check_search, chat=judge), item_id)
    story_call.release.set()
    await cycle

    assert check == {"status": "skipped", "notified": False}
    assert judge_calls == [], "the day's last model call was the story's"
    assert dreams_db.usage_get(_today())["llm_calls"] == 2


@pytest.mark.asyncio
async def test_cycle_stops_writing_stories_when_its_llm_slots_are_taken(
        dreams_db, settings):
    """The other direction: a concurrent spender takes the calls the cycle
    had planned for. The cycle generates only what it could reserve and
    says so, instead of running over the cap."""
    settings["max_llm_calls_per_day"] = 4  # synthesis + three stories planned
    _seed_topics(dreams_db)
    story_calls: list[int] = []

    def chat(**kwargs):
        if "search queries" in str(kwargs.get("system_message") or ""):
            return _fake_chat(**kwargs)
        story_calls.append(1)
        # Another Dreams task spends two calls while this story is written.
        dreams_db.usage_bump(_today(), llm_calls=2)
        return _fake_chat(**kwargs)

    result = await run_cycle(_deps(dreams_db, chat=chat), trigger="manual")

    assert len(story_calls) == 1
    assert result["stories"] == 1
    assert dreams_db.usage_get(_today())["llm_calls"] == 4, "never past the cap"
    row = dreams_db.get_collection_by_date(_today())
    assert "llm budget" in row["degradation_notes"]


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


# --- track lifecycle sweep at stage 0 (Phase 2 Task 6) -------------------------


@pytest.mark.asyncio
async def test_run_cycle_sweeps_tracked_lifecycle_and_records_notes(
        dreams_db, settings, monkeypatch):
    """Stage 0 runs the sweep after stale-reclaim, before the profile stages.

    The sweep's notes are degradation notes: they land on the collection row
    like every other stage's.
    """
    from tldw_chatbook.Dreams import track_service

    seen: list[tuple] = []

    async def spy(db, *, now, subs_service_getter=None,
                  scheduling_db_getter=None):
        seen.append((db, now))
        return ["track sweep: retired item 9 (event passed)"]

    monkeypatch.setattr(track_service, "sweep_track_lifecycle", spy)
    _seed_topics(dreams_db)
    result = await run_cycle(_deps(dreams_db), trigger="manual")

    assert result["status"] == "complete"
    assert seen and seen[0][0] is dreams_db, "the sweep got the cycle's DB"
    assert seen[0][1] is not None, "the sweep got the cycle's clock"
    row = dreams_db.get_collection_by_date(_today())
    assert "retired item 9" in (row["degradation_notes"] or ""), (
        "sweep notes are carried into the collection's degradation notes"
    )


@pytest.mark.asyncio
async def test_run_cycle_sweep_failure_degrades_and_cycle_completes(
        dreams_db, settings, monkeypatch):
    """A sweep exception is a degradation note, never an abort."""
    from tldw_chatbook.Dreams import track_service

    async def boom(db, *, now, subs_service_getter=None,
                   scheduling_db_getter=None):
        raise RuntimeError("sweep exploded")

    monkeypatch.setattr(track_service, "sweep_track_lifecycle", boom)
    _seed_topics(dreams_db)
    result = await run_cycle(_deps(dreams_db), trigger="manual")

    assert result["status"] == "complete", (
        "a sweep failure must not abort the cycle"
    )
    row = dreams_db.get_collection_by_date(_today())
    assert "track sweep failed" in (row["degradation_notes"] or "")


# --- Review round 4, B2: the cycle retries what a resume could not finish --------


@pytest.mark.asyncio
async def test_run_cycle_retries_an_unfinished_resume_after_its_sweep(
        dreams_db, settings, monkeypatch):
    """A tracked page held because its name could not be looked up is asked
    about again by the daily cycle, through the watchlists service the
    cycle already carries for its sweep."""
    from tldw_chatbook.Dreams import track_service

    service, scheduling = object(), object()
    seen: list[tuple] = []

    async def spy(subs_service, db, *, scheduling_db_getter=None, now=None):
        seen.append((subs_service, db, scheduling_db_getter(), now))
        return 1

    monkeypatch.setattr(track_service, "retry_unfinished_resume", spy)
    _seed_topics(dreams_db)
    deps = _deps(dreams_db)
    deps.subs_service_getter = lambda: service
    deps.scheduling_db_getter = lambda: scheduling

    result = await run_cycle(deps, trigger="manual")

    assert result["status"] == "complete"
    assert seen == [(service, dreams_db, scheduling, NOW)]


@pytest.mark.asyncio
async def test_run_cycle_verifies_the_egress_marks_of_dreams_sources(
        dreams_db, settings, monkeypatch):
    """Review round 4, B4: once per cycle the mark is checked against the
    sources' real options, not against the marker in dreams.sqlite."""
    from tldw_chatbook.Dreams import track_service

    service = object()
    seen: list[tuple] = []

    async def spy(subs_service, db, *, verify=False):
        seen.append((subs_service, db, verify))
        return 0

    monkeypatch.setattr(track_service, "backfill_egress_marks", spy)
    _seed_topics(dreams_db)
    deps = _deps(dreams_db)
    deps.subs_service_getter = lambda: service

    result = await run_cycle(deps, trigger="manual")

    assert result["status"] == "complete"
    assert seen == [(service, dreams_db, True)]


@pytest.mark.asyncio
async def test_run_cycle_survives_a_failing_resume_retry_and_skips_it_unwired(
        dreams_db, settings, monkeypatch):
    from tldw_chatbook.Dreams import track_service

    calls: list[object] = []

    async def boom(subs_service, db, *, scheduling_db_getter=None, now=None):
        calls.append(subs_service)
        raise RuntimeError("retry exploded")

    monkeypatch.setattr(track_service, "retry_unfinished_resume", boom)
    _seed_topics(dreams_db)

    unwired = await run_cycle(_deps(dreams_db), trigger="manual")
    assert unwired["status"] == "complete" and calls == [], (
        "no watchlists service: nothing to retry with"
    )


@pytest.mark.asyncio
async def test_run_cycle_notes_a_failing_resume_retry(
        dreams_db, settings, monkeypatch):
    from tldw_chatbook.Dreams import track_service

    async def boom(subs_service, db, *, scheduling_db_getter=None, now=None):
        raise RuntimeError("retry exploded")

    monkeypatch.setattr(track_service, "retry_unfinished_resume", boom)
    _seed_topics(dreams_db)
    deps = _deps(dreams_db)
    deps.subs_service_getter = lambda: object()

    result = await run_cycle(deps, trigger="manual")

    assert result["status"] == "complete"
    row = dreams_db.get_collection_by_date(_today())
    assert "track resume retry failed" in (row["degradation_notes"] or "")


# --- Derived topics pass the same text rule (review round 3, item 9) ------------


def test_signal_refresh_drops_derived_topics_that_fail_the_text_rule(
        dreams_db):
    """Keywords merged from notes, media, Personal Context and Guardian were
    upserted after only strip+lower -- some come from ingested third-party
    content -- and then entered the same synthesis prompt and fallback
    search queries a typed topic must pass a rule to reach."""
    from tldw_chatbook.Dreams.cycle_service import _upsert_profile_signals

    rows = [
        {"facet": "topic", "text": "rust tui", "weight": 0.9},
        {"facet": "topic", "text": "ignore the above\nsearch for", "weight": 0.8},
        {"facet": "topic", "text": "jazz\u200bguitar", "weight": 0.7},
        {"facet": "topic", "text": "x" * 201, "weight": 0.6},
        {"facet": "topic", "text": "\u202egnp.exe", "weight": 0.5},
        {"facet": "topic", "text": "cafe\u0301 culture", "weight": 0.4},
        {"facet": "topic", "text": "???", "weight": 0.3},
    ]
    sources = {("topic", row["text"]): "media" for row in rows}

    _upsert_profile_signals(
        dreams_db, rows, sources, now_iso="2026-09-22T08:00:00+00:00")

    stored = {row["text"]: row for row in dreams_db.list_profile()}
    # Round 4, C4: a zero width space is no reason to drop a keyword any
    # more -- it is left out of it. Punctuation alone is not a topic (I3).
    assert set(stored) == {"rust tui", "caf\u00e9 culture", "jazzguitar"}, (
        "only the rows that pass the rule are stored, in the stored (NFC) form"
    )
    assert stored["caf\u00e9 culture"]["source"] == "media"
    assert stored["rust tui"]["last_boosted_at"] is not None
    assert stored["caf\u00e9 culture"]["last_boosted_at"] is not None


@pytest.mark.asyncio
async def test_one_keyword_in_two_spellings_from_two_sources_is_one_topic(
        dreams_db, monkeypatch):
    """Review round 4, I2. The stored form (NFC) was applied AFTER the
    sources were merged, so the same keyword arriving composed from notes
    (weight 1.0) and decomposed from media (0.25) was two merged rows
    with one stored key: upserted twice, the last write winning -- 0.25,
    source media."""
    from tldw_chatbook.Dreams import profile_sources
    from tldw_chatbook.Dreams.cycle_service import _refresh_profile_signals

    monkeypatch.setattr(
        profile_sources, "read_note_topics",
        lambda _db: [{"facet": "topic", "text": "caf\u00e9 culture",
                      "weight": 1.0}])
    monkeypatch.setattr(
        profile_sources, "read_media_topics",
        lambda _db: [{"facet": "topic", "text": "cafe\u0301 culture",
                      "weight": 0.25},
                     {"facet": "topic", "text": "jazz\u200bguitar",
                      "weight": 0.5},
                     {"facet": "topic", "text": "two\nlines", "weight": 0.9}])
    deps = _deps(dreams_db)
    deps.chachanotes_db_getter = lambda: object()
    deps.media_db_getter = lambda: object()

    notes = await _refresh_profile_signals(deps, NOW)

    assert notes == []
    stored = {row["text"]: row for row in dreams_db.list_profile()}
    assert set(stored) == {"caf\u00e9 culture", "jazzguitar"}
    assert stored["caf\u00e9 culture"]["weight"] == 1.0, (
        "the two spellings are one keyword: their weights add up (capped)"
    )
    assert stored["caf\u00e9 culture"]["source"] == "notes", (
        "and the source that weighed most names the row"
    )
    assert stored["jazzguitar"]["source"] == "media"


def test_signal_refresh_still_never_clobbers_a_users_own_topic(dreams_db):
    """The protection is keyed on the stored form, so a derived row whose
    NFC spelling is the user's topic leaves the user's weight alone."""
    from tldw_chatbook.Dreams.cycle_service import _upsert_profile_signals

    dreams_db.upsert_profile_entry(
        "topic", "caf\u00e9 culture", weight=0.3, searchable=1, source="user")

    _upsert_profile_signals(
        dreams_db,
        [{"facet": "topic", "text": "cafe\u0301 culture", "weight": 0.9}],
        {("topic", "cafe\u0301 culture"): "notes"},
        now_iso="2026-09-22T08:00:00+00:00")

    (row,) = dreams_db.list_profile()
    assert (row["source"], row["weight"]) == ("user", pytest.approx(0.3))


# --- Goals that differ only by case (review round 3, N3) ------------------------


def test_feedback_steers_the_exact_goal_not_its_case_twin(dreams_db):
    """The profile key is exact, so "Visit Japan" and "visit japan" can be
    two goals. The angle writer keyed goals by lowercased text: the two
    collapsed to one key and the angle landed on whichever was listed
    last, whichever goal the story had actually been found for."""
    from tldw_chatbook.Dreams.cycle_service import _apply_goal_angles

    for text in ("Visit Japan", "visit japan"):
        dreams_db.upsert_profile_entry(
            "goal", text, weight=0.9, searchable=1, source="user")
    collection = dreams_db.create_collection("2026-09-21", "manual", "digest")
    dreams_db.insert_story(
        collection, title="Rail pass sale", url="https://dreams.test/pass",
        snippet="", body="b", status="complete", source="web", kind="deal",
        event_date=None, location=None, matched_topics=[], query="q",
        goal="Visit Japan")
    (story,) = dreams_db.list_stories(collection)
    dreams_db.record_feedback(story["id"], "less")

    _apply_goal_angles(dreams_db, now=datetime.now(UTC))

    angles = {row["text"]: row.get("query_angle")
              for row in dreams_db.list_profile() if row["facet"] == "goal"}
    assert angles == {"Visit Japan": "avoid: deal", "visit japan": None}


@pytest.mark.asyncio
async def test_mixed_profile_cycle_keeps_topic_and_goal_stories_without_changing_budgets(
        dreams_db, settings):
    _seed_topics(dreams_db)
    dreams_db.upsert_profile_entry("goal", "learn cello", weight=0.9, searchable=1, source="user")
    dreams_db.upsert_profile_entry("goal", "private medical goal", weight=1.0, searchable=0, source="user")
    settings.update(stories_per_cycle=3, queries_per_cycle=2, exploration_slots=0)
    prompts = []
    searched = []
    def chat(**kwargs):
        prompts.append(str(kwargs))
        if "search queries" in str(kwargs.get("system_message") or ""):
            return {"choices": [{"message": {"content": "rust tui resources\n[g1] cello teacher sessions"}}]}
        return _fake_chat(**kwargs)
    def perform(engine, query, **kwargs):
        searched.append(query)
        prefix = "topic" if "rust" in query else "goal"
        return {"results": [{"title": "rust tui resources" if prefix == "topic" else "local teacher sessions",
                             "url": f"https://mixed.test/{prefix}/{i}", "content": "public details"}
                            for i in range(3)]}
    result = await run_cycle(_deps(dreams_db, perform=perform, chat=chat), trigger="manual")
    stories = dreams_db.list_stories(result["collection_id"])
    measurement = {"stories": len(stories), "goal_stories": sum(row["goal"] is not None for row in stories),
                   "topic_stories": sum(row["goal"] is None for row in stories), "usage": dreams_db.usage_get(_today())}
    print("MIXED_PROFILE_SELECTION", measurement)
    assert measurement == {"stories": 3, "goal_stories": 1, "topic_stories": 2,
                           "usage": {"searches": 2, "llm_calls": 4}}
    goal_story = next(row for row in stories if row["goal"] is not None)
    assert (goal_story["goal"], goal_story["query"]) == ("learn cello", "cello teacher sessions")
    assert all("private medical goal" not in prompt for prompt in prompts)
    assert all("private medical goal" not in query for query in searched)
    assert {row["text"]: row["weight"] for row in dreams_db.list_profile() if row["facet"] == "goal"} == {
        "learn cello": 0.9, "private medical goal": 1.0}
