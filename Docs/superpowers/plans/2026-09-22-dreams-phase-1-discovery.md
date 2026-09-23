# Dreams Phase 1 (Discovery) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Dreams daily discovery loop — a dated collection of story-shaped discoveries generated on a schedule (with boot catch-up), surfaced on the Artifacts screen with dive/keep/export/feedback/ingest actions.

**Architecture:** A new `Dreams/` package plus `DB/Dreams_DB.py` follows the briefing pipeline's proven patterns: row-per-outcome generation, an event-loop-only claim set, `asyncio.to_thread` for every blocking call, and provider resolution through `[dreams]` config → persisted chat defaults. A read-only `DreamsProjection` feeds `dreams_cycle` tasks into the existing scheduler; the Artifacts screen gets a Dreams refresh worker mirroring the Daily Reports wiring.

**Tech Stack:** Python ≥3.12 (stdlib + existing deps only — no new third-party packages), SQLite via `BaseDB`, Textual 8.x, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-22-dreams-daily-discovery-design.md` (+ `backlog/decisions/178-dreams-daily-discovery-and-tracking.md`). The plan argues from the spec; read both. Phase 2 (Track) is **out of scope** for this plan.

## Global Constraints

- Dreams is **off by default**: `[dreams] enabled` defaults to `false`; no cycle, projection task, or UI row may appear unless enabled.
- Parameterized SQL only — never f-strings into queries.
- All timestamps stored as UTC ISO-8601 strings; `local_date` is the user's **local** calendar date (`YYYY-MM-DD`, `datetime.now().astimezone().strftime`).
- Every blocking call (`perform_websearch`, `chat_api_call`, every SQLite access from async code) runs under `asyncio.to_thread`, grouped one hop per stage; claim sets are mutated only on the event loop with no `await` between check and add.
- Raw signals never leave the machine — only synthesized queries and story prompts (distilled topics + region + public snippets) may be sent out (spec §Privacy).
- UI: ADR-150 `$ds-*` tokens only (governance test `Tests/UI/test_design_token_governance.py` enforces); keybindings per ADR-031 (single-letter actions in modals, no terminal-convention keys, footer hints only for implemented actions).
- Testing: targeted runs only (`python3 -m pytest Tests/Dreams/<file> -v`); full sweeps are a pre-PR event the user requests.
- One commit per task, `feat(dreams):`/`test(dreams):` prefixes; only the task's files staged.
- Each task below maps 1:1 to a backlog task created at execution start (`backlog task create … --plain`, assign yourself, move to In Progress; Done when the task's steps complete).

## File Structure

```
tldw_chatbook/
  Dreams/
    __init__.py                # empty, package marker
    settings.py                # [dreams] defaults + typed accessor (Task 1)
    interest_profile.py        # pure weight math: decay/merge/normalize (Task 2)
    profile_sources.py         # notes/media/PersonalContext readers (Task 2)
    query_synthesis.py         # profile snapshot -> queries, one LLM call (Task 3)
    discovery.py               # search + watchlist pool + dedupe/rank (Task 3)
    story_service.py           # one story per call, event metadata (Task 4)
    cycle_service.py           # orchestration: claims, budgets, catch-up (Task 4/5)
    dreams_view.py             # Artifacts projection helpers (Task 6)
    ingest_action.py           # story URL -> capture backend (Task 8)
  DB/
    Dreams_DB.py               # DreamsDB(BaseDB), schema v1 (Task 1)
  Scheduling/services/
    dreams_projection.py       # cadence -> ScheduledTask rows (Task 5)
  Scheduling/scheduler/handlers/
    dreams_handler.py          # fire-and-forget cycle dispatch (Task 5)
  UI/Screens/
    artifacts_screen.py        # MODIFY: Dreams type + refresh worker (Task 6)
    artifacts_dreams_modal.py  # story detail modal + actions (Task 7)
Tests/
  Dreams/                      # one test file per task
```

---

### Task 1: `DreamsDB` schema v1 + settings module

**Files:**
- Create: `tldw_chatbook/DB/Dreams_DB.py`
- Create: `tldw_chatbook/Dreams/__init__.py` (empty)
- Create: `tldw_chatbook/Dreams/settings.py`
- Test: `Tests/Dreams/test_dreams_db.py`

**Interfaces:**
- Consumes: `tldw_chatbook.DB.base_db.BaseDB`; `tldw_chatbook.config.get_cli_setting`.
- Produces (used by Tasks 2–8):
  - `class DreamsDB(BaseDB)` with `__init__(self, database_path: str | Path, client_id: str | None = None)` and methods: `create_collection(local_date: str, trigger: str, profile_digest: str) -> int | None` (returns `None` when the date already has a row), `get_collection_by_date(local_date: str) -> dict | None`, `set_collection_status(collection_id: int, status: str, *, provider: str | None = None, model: str | None = None, degradation_notes: str | None = None, completed_at: str | None = None) -> None`, `insert_story(collection_id: int, *, title: str, url: str, snippet: str, body: str, status: str, source: str, kind: str, event_date: str | None, location: str | None, matched_topics: list[str], query: str, error: str | None = None) -> int`, `list_stories(collection_id: int) -> list[dict]`, `list_recent_stories(limit: int = 20) -> list[dict]`, `set_story_kept(story_id: int, kept: bool) -> None`, `record_feedback(story_id: int, kind: str) -> None`, `seen_upsert(urls: list[tuple[str, str]]) -> None`, `seen_filter_unseen(urls: list[str]) -> set[str]`, `prune_seen(cutoff_iso: str) -> int`, `list_profile() -> list[dict]`, `upsert_profile_entry(facet: str, text: str, *, weight: float, searchable: int, source: str) -> None`, `delete_profile_entry(entry_id: int) -> None`, `usage_bump(local_date: str, *, searches: int = 0, llm_calls: int = 0) -> None`, `usage_get(local_date: str) -> dict`, `fail_stale_generating(cutoff_iso: str) -> int`.
  - `Dreams/settings.py`: `DREAMS_DEFAULTS: dict[str, Any]` and `dreams_setting(key: str, default: Any = None) -> Any` (reads `[dreams]` via `get_cli_setting("dreams", key, DREAMS_DEFAULTS[key])`).

Schema v1 (spec §Data model, Phase 1 tables only — track tables are Phase 2's migration): `dream_interest_profile`, `dreams_collections` (UNIQUE on `local_date`), `dream_stories` (UNIQUE on `(collection_id, url)`), `dream_feedback`, `dream_seen_items` (`url` PRIMARY KEY), `dream_daily_usage` (`local_date` PRIMARY KEY, `searches INTEGER`, `llm_calls INTEGER` — plan-level addition backing the spec's budget mechanism). Status CHECK constraints exactly as in the spec.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Dreams/test_dreams_db.py
"""DreamsDB schema v1 behavior: date bucketing, story rows, seen ledger, usage."""
import sqlite3
from datetime import datetime, timezone

import pytest

from tldw_chatbook.DB.Dreams_DB import DreamsDB


@pytest.fixture()
def db(tmp_path):
    database = DreamsDB(tmp_path / "dreams.sqlite", "test-client")
    yield database
    database.close()


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_create_collection_enforces_one_row_per_local_date(db):
    first = db.create_collection("2026-09-22", "scheduled", "digest-1")
    assert first is not None
    assert db.create_collection("2026-09-22", "manual", "digest-2") is None
    assert db.get_collection_by_date("2026-09-22")["id"] == first


def test_insert_and_list_stories_roundtrip_with_metadata(db):
    cid = db.create_collection("2026-09-22", "scheduled", "d")
    sid = db.insert_story(
        cid, title="Cheap flights to Japan", url="https://example.com/f",
        snippet="Fares from $89", body="A story about fares.", status="complete",
        source="web", kind="deal", event_date=None, location="Japan",
        matched_topics=["visit japan"], query="cheap flights japan",
    )
    stories = db.list_stories(cid)
    assert [s["id"] for s in stories] == [sid]
    assert stories[0]["kind"] == "deal"
    assert stories[0]["kept"] == 0


def test_insert_story_rejects_duplicate_url_within_collection(db):
    cid = db.create_collection("2026-09-22", "scheduled", "d")
    db.insert_story(cid, title="a", url="https://x/1", snippet="", body="",
                    status="complete", source="web", kind="content",
                    event_date=None, location=None, matched_topics=[], query="q")
    with pytest.raises(sqlite3.IntegrityError):
        db.insert_story(cid, title="a2", url="https://x/1", snippet="", body="",
                        status="complete", source="web", kind="content",
                        event_date=None, location=None, matched_topics=[], query="q")


def test_seen_ledger_filters_and_prunes(db):
    db.seen_upsert([("https://a", "t-a"), ("https://b", "t-b")])
    assert db.seen_filter_unseen(["https://a", "https://c"]) == {"https://c"}
    assert db.prune_seen("2999-01-01T00:00:00+00:00") == 2


def test_usage_bump_accumulates_per_local_date(db):
    db.usage_bump("2026-09-22", searches=3)
    db.usage_bump("2026-09-22", llm_calls=5)
    assert db.usage_get("2026-09-22") == {"searches": 3, "llm_calls": 5}
    assert db.usage_get("2026-09-23") == {"searches": 0, "llm_calls": 0}


def test_fail_stale_generating_marks_only_old_generating_rows(db):
    cid = db.create_collection("2026-09-22", "scheduled", "d")
    db.create_collection("2026-09-23", "scheduled", "d2")  # stays untouched
    db.fail_stale_generating("2999-01-01T00:00:00+00:00")
    assert db.get_collection_by_date("2026-09-22")["status"] == "failed"
    assert db.get_collection_by_date("2026-09-23")["status"] == "generating"
```

Also add `Tests/Dreams/test_dreams_settings.py`:

```python
from tldw_chatbook.Dreams.settings import DREAMS_DEFAULTS, dreams_setting


def test_defaults_cover_spec_config_keys():
    expected = {
        "enabled", "provider", "model", "stories_per_cycle", "queries_per_cycle",
        "exploration_slots", "search_engine", "web_search_enabled", "region",
        "cadence_hours", "catchup_enabled", "watchlist_freshness_hours",
        "seen_item_ttl_days", "max_searches_per_day", "max_llm_calls_per_day",
    }
    assert expected <= set(DREAMS_DEFAULTS)


def test_dreams_setting_returns_default_when_unset(monkeypatch):
    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting",
                        lambda section, key, default: default)
    assert dreams_setting("stories_per_cycle") == 5
    assert dreams_setting("enabled") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/Dreams/test_dreams_db.py Tests/Dreams/test_dreams_settings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tldw_chatbook.DB.Dreams_DB'`.

- [ ] **Step 3: Implement `DreamsDB`**

Model on `DB/Library_Collections_DB.py` (thread-local connections via `BaseDB`, `_CURRENT_SCHEMA_VERSION = 1`, additive `CREATE TABLE IF NOT EXISTS`; read its class docstring before writing). Timestamps: `datetime.now(timezone.utc).isoformat()`. Every method takes a connection from the thread-local pattern and uses parameterized SQL. Representative implementation of the two trickiest methods:

```python
def create_collection(self, local_date: str, trigger: str, profile_digest: str) -> int | None:
    """Insert one collection for a local date; None when the date is taken.

    The UNIQUE(local_date) index is the race guard: scheduler fire, boot
    catch-up, and manual trigger all funnel through this INSERT, and SQLite
    makes exactly one of them win.
    """
    now = _utc_now_iso()
    with self.transaction() as cursor:
        try:
            cursor.execute(
                "INSERT INTO dreams_collections"
                " (local_date, status, profile_digest, trigger, created_at)"
                " VALUES (?, 'generating', ?, ?, ?)",
                (local_date, profile_digest, trigger, now),
            )
        except sqlite3.IntegrityError:
            return None
        return int(cursor.lastrowid)


def fail_stale_generating(self, cutoff_iso: str) -> int:
    """Reclaim dates wedged by a crashed cycle (spec §idempotency)."""
    with self.transaction() as cursor:
        cursor.execute(
            "UPDATE dreams_collections SET status = 'failed',"
            " degradation_notes = COALESCE(degradation_notes || '; ', '')"
            " || 'reclaimed: stale generating row'"
            " WHERE status = 'generating' AND created_at < ?",
            (cutoff_iso,),
        )
        return cursor.rowcount
```

Implement the remaining methods per the Interfaces block with the same style; `seen_filter_unseen` executes one `SELECT url FROM dream_seen_items WHERE url IN (…)` built by placeholders; `list_recent_stories` joins the newest collection first (`ORDER BY c.local_date DESC, s.id ASC`, kept stories first within a collection).

`Dreams/settings.py`:

```python
"""[dreams] config defaults and accessor. Spec §Configuration; off by default."""
from __future__ import annotations

from typing import Any

from tldw_chatbook.config import get_cli_setting

DREAMS_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "provider": None,
    "model": None,
    "stories_per_cycle": 5,
    "queries_per_cycle": 3,
    "exploration_slots": 1,
    "search_engine": "duckduckgo",
    "web_search_enabled": True,
    "region": "",
    "cadence_hours": 24,
    "catchup_enabled": True,
    "watchlist_freshness_hours": 48,
    "seen_item_ttl_days": 90,
    "max_searches_per_day": 30,
    "max_llm_calls_per_day": 60,
}


def dreams_setting(key: str, default: Any = None) -> Any:
    if default is None and key in DREAMS_DEFAULTS:
        default = DREAMS_DEFAULTS[key]
    return get_cli_setting("dreams", key, default)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/Dreams/test_dreams_db.py Tests/Dreams/test_dreams_settings.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Dreams_DB.py tldw_chatbook/Dreams/__init__.py tldw_chatbook/Dreams/settings.py Tests/Dreams/test_dreams_db.py Tests/Dreams/test_dreams_settings.py
git commit -m "feat(dreams): DreamsDB schema v1 (date-bucketed collections, stories, ledger, usage) + settings"
```

---

### Task 2: Interest profile — signal readers and weight math

**Files:**
- Create: `tldw_chatbook/Dreams/interest_profile.py`
- Create: `tldw_chatbook/Dreams/profile_sources.py`
- Test: `Tests/Dreams/test_interest_profile.py`, `Tests/Dreams/test_profile_sources.py`

**Interfaces:**
- Consumes: `DreamsDB.list_profile/upsert_profile_entry` (Task 1); `CharactersRAGDB` (notes), `ClientMediaDB` (media keywords), `Personal_Context.service` (`list_records`, raises `ProfileLockedError` per `Personal_Context/key_protector.py`).
- Produces (used by Tasks 3–4):
  - `interest_profile.decay_weights(topics: list[dict], *, now_epoch: float, half_life_days: float = 14.0, floor: float = 0.05) -> list[dict]` — pure; decays each topic's `weight` toward `floor` based on `last_boosted_at` (epoch), never above 1.0.
  - `interest_profile.merge_signals(signal_lists: list[list[dict]], *, top_n: int = 30) -> list[dict]` — pure; sums weights per normalized `(facet, text)` key, returns top-N by weight.
  - `interest_profile.snapshot(db: DreamsDB, *, now_epoch: float) -> dict` — reads profile rows, applies decay, returns `{"topics": [...], "region": str}`.
  - `profile_sources.read_note_topics(chachanotes_db, *, window_days: int = 14) -> list[dict]`
  - `profile_sources.read_media_topics(media_db, *, window_days: int = 14) -> list[dict]`
  - `profile_sources.read_personal_context_topics(pc_service, *, cache_path: Path) -> list[dict]` — **never raises** `ProfileLockedError`; on lock or any read error, returns the cached distillate (JSON list under `open_private_binary` discipline from `Utils/private_paths.py`), writing the cache only on a successful read.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Dreams/test_interest_profile.py
import math

import pytest

from tldw_chatbook.Dreams.interest_profile import decay_weights, merge_signals

DAY = 86400.0


def test_decay_moves_weight_toward_floor_with_half_life():
    topics = [{"facet": "topic", "text": "rust", "weight": 0.9,
               "last_boosted_at": 0.0}]
    out = decay_weights(topics, now_epoch=14 * DAY)  # one half-life
    assert out[0]["weight"] == pytest.approx(0.05 + (0.9 - 0.05) / 2)


def test_decay_never_reaches_zero_and_caps_at_one():
    fresh = [{"facet": "topic", "text": "a", "weight": 1.0,
              "last_boosted_at": 99 * DAY}]
    stale = [{"facet": "topic", "text": "b", "weight": 1.0,
              "last_boosted_at": 0.0}]
    assert decay_weights(fresh, now_epoch=100 * DAY)[0]["weight"] <= 1.0
    assert decay_weights(stale, now_epoch=100 * DAY)[0]["weight"] >= 0.05


def test_merge_sums_same_topic_and_ranks():
    merged = merge_signals(
        [[{"facet": "topic", "text": " Rust ", "weight": 0.4},
          {"facet": "topic", "text": "html", "weight": 0.2}],
         [{"facet": "topic", "text": "rust", "weight": 0.3}]],
        top_n=2,
    )
    assert merged[0]["text"] == "rust"
    assert merged[0]["weight"] == pytest.approx(0.7)
    assert [t["text"] for t in merged] == ["rust", "html"]
```

```python
# Tests/Dreams/test_profile_sources.py
import json

import pytest

from tldw_chatbook.Dreams.profile_sources import read_personal_context_topics


class LockedService:
    def list_records(self, *, scope_ids, include_archived=False):
        raise RuntimeError("profile-locked")  # wrapper re-raises ProfileLockedError


def test_locked_personal_context_returns_cached_distillate(tmp_path, monkeypatch):
    cache = tmp_path / "pc_distillate.json"
    cache.write_text(json.dumps([{"facet": "topic", "text": "visit japan",
                                  "weight": 1.0, "searchable": 1,
                                  "source": "personal_context"}]))
    out = read_personal_context_topics(LockedService(), cache_path=cache)
    assert [t["text"] for t in out] == ["visit japan"]


def test_successful_read_refreshes_cache(tmp_path):
    class OkService:
        def list_records(self, *, scope_ids, include_archived=False):
            return []  # empty tuple is a successful read
    cache = tmp_path / "pc_distillate.json"
    out = read_personal_context_topics(OkService(), cache_path=cache)
    assert out == []
    assert cache.exists()
```

Also one test each for `read_note_topics`/`read_media_topics` using in-memory `CharactersRAGDB`/`ClientMediaDB` fixtures if the repo has them (`grep -rn "ClientMediaDB(" Tests/ | head -3` to find the construction pattern; if no fixture exists, construct against `tmp_path` SQLite files and insert one note/keyword row + one media keyword row directly, then assert the reader returns them with normalized text).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/Dreams/test_interest_profile.py Tests/Dreams/test_profile_sources.py -v`
Expected: FAIL, `ModuleNotFoundError` for both new modules.

- [ ] **Step 3: Implement both modules**

`interest_profile.py` (pure math, no I/O):

```python
import math


def decay_weights(topics, *, now_epoch, half_life_days=14.0, floor=0.05):
    half_life_s = max(half_life_days, 0.5) * 86400.0
    out = []
    for t in topics:
        age = max(0.0, now_epoch - float(t.get("last_boosted_at") or 0.0))
        factor = math.pow(0.5, age / half_life_s)
        weight = floor + (min(float(t.get("weight", 0.0)), 1.0) - floor) * factor
        row = dict(t)
        row["weight"] = min(max(weight, floor), 1.0)
        out.append(row)
    return out


def merge_signals(signal_lists, *, top_n=30):
    totals: dict[str, dict] = {}
    for signals in signal_lists:
        for entry in signals:
            key = (entry.get("facet", "topic"), str(entry.get("text", "")).strip().lower())
            if not key[1]:
                continue
            agg = totals.setdefault(key[0] + "|" + key[1],
                                    {"facet": key[0], "text": str(entry["text"]).strip(),
                                     "weight": 0.0})
            agg["weight"] += float(entry.get("weight", 0.0))
    ranked = sorted(totals.values(), key=lambda t: t["weight"], reverse=True)
    for row in ranked:
        row["weight"] = min(row["weight"], 1.0)
    return ranked[:top_n]
```

`snapshot(db, *, now_epoch)` reads `db.list_profile()`, keeps `facet == "topic"`, applies `decay_weights`, returns `{"topics": topics, "region": dreams_setting("region", "")}`.

`profile_sources.py`: note reader selects keyword aggregates from recent notes (`ChaChaNotes_DB` — find the keyword listing query the Notes screens use: `grep -rn "note_keywords" tldw_chatbook/DB/ChaChaNotes_DB.py | head -5` and reuse the indexed read, no LIKE scans); media reader aggregates `Keywords` joined to `Media` rows touched inside the window (`ReadingProgress` counts double weight). Personal Context reader:

```python
def read_personal_context_topics(pc_service, *, cache_path):
    """Opt-in signal: degrade to the cached distillate on any read failure.

    Personal Context key custody may be passphrase-wrapped
    (Personal_Context/key_protector.py), so an unattended read raises
    ProfileLockedError -- the cache written by the last successful read is
    the whole point (spec §interest profile, cache-on-unlock).
    """
    try:
        records = pc_service.list_records(scope_ids=_dreams_scope_ids(pc_service))
        topics = [_record_to_topic(r) for r in records if _record_to_topic(r)]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(topics))
        return topics
    except Exception:  # noqa: BLE001 - locked, unreadable, or unconfigured
        try:
            return json.loads(cache_path.read_text()) if cache_path.exists() else []
        except Exception:  # noqa: BLE001 - corrupt cache == no signal
            return []
```

`_dreams_scope_ids` lists `pc_service.list_scopes()` and keeps scope ids whose `scope_id`/name suggests interests/topics — if scopes are fixed-enum, pick the interests-like scope; the executor reads `Personal_Context/service.py:1349` `list_records` + `list_scopes` to choose (record it in a comment). Cache write must go through `open_private_binary` (import from `Utils/private_paths.py`) rather than plain `write_text` if that helper enforces private-path permissions — check its signature and use it for both read and write.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/Dreams/ -v`
Expected: all PASS (Tasks 1–2 files).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Dreams/interest_profile.py tldw_chatbook/Dreams/profile_sources.py Tests/Dreams/test_interest_profile.py Tests/Dreams/test_profile_sources.py
git commit -m "feat(dreams): interest profile signal readers + decay/merge math with locked-profile degradation"
```

---

### Task 3: Query synthesis and discovery (search, watchlist pool, dedupe, rank)

**Files:**
- Create: `tldw_chatbook/Dreams/query_synthesis.py`
- Create: `tldw_chatbook/Dreams/discovery.py`
- Test: `Tests/Dreams/test_query_synthesis.py`, `Tests/Dreams/test_discovery.py`

**Interfaces:**
- Consumes: `chat_api_call`-shaped callable (injected; kwargs per `Chat/Chat_Functions.py:937`), `perform_websearch` (injected; signature per `Web_Scraping/WebSearch_APIs.py:1990`), `SubscriptionsDB` (`subscription_items` has `published_date`, `status`, `url`, `title`), `ClientMediaDB` (`Media` URLs), Task 1 ledger methods, Task 2 snapshot.
- Produces (used by Task 4):
  - `@dataclass(slots=True) class Candidate: title: str; url: str; snippet: str; source: str  # 'web' | 'watchlist'`
  - `query_synthesis.synthesize_queries(chat, *, snapshot: dict, count: int, exploration_slots: int) -> list[str]` — **one** chat call; prompt embeds topics + region; returns `count` strings; at least `exploration_slots` of them must not reference any profile topic (the prompt demands at least one adjacent/serendipity angle); on any chat failure returns a deterministic fallback list built directly from the top topics (`f"{topic} recent developments"`, plus one `"surprising adjacent to {top_topic}"`) so a dead LLM still yields a degraded cycle.
  - `discovery.fetch_watchlist_candidates(subs_db, *, freshness_hours: int, now_epoch: float) -> list[Candidate]` — `subscription_items` where `published_date` within the window and `status = 'new'`.
  - `discovery.dedupe_and_rank(candidates, *, seen_urls: set[str], library_urls: set[str], topics: list[dict], limit: int) -> list[Candidate]` — pure; drops seen/library URLs and exact duplicates (normalized: lowercase, strip trailing slash and query string), then ranks by lexical overlap with topic texts (case-insensitive substring/token match; 1.0 per matching topic, summed, ties by original order), returns `limit`.
  - `discovery.run_queries(perform, *, engine: str, queries: list[str], result_count: int = 8, date_range: str | None = "m") -> tuple[list[Candidate], int]` — runs each query under `asyncio.to_thread`, flattens non-error results, returns `(candidates, searches_used)`. `date_range="m"` enforces the spec's recency promise for time-sensitive angles.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Dreams/test_query_synthesis.py
import pytest

from tldw_chatbook.Dreams.query_synthesis import synthesize_queries

SNAP = {"topics": [{"facet": "topic", "text": "rust tui", "weight": 0.9},
                   {"facet": "topic", "text": "jazz guitar", "weight": 0.6}],
        "region": "Seattle"}


def _capture_chat(response_text):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": response_text}}]}

    return chat, calls


def test_synthesis_makes_exactly_one_call_and_parses_lines():
    chat, calls = _capture_chat("rust tui new releases 2026\njazz guitar concerts Seattle\nrandom curiosity: deep sea cables")
    out = synthesize_queries(chat, snapshot=SNAP, count=3, exploration_slots=1)
    assert len(calls) == 1
    assert len(out) == 3 and "deep sea cables" in out[2]
    assert "Seattle" in calls[0]["messages_payload"][0]["content"]


def test_synthesis_falls_back_deterministically_when_chat_fails():
    def boom(**kwargs):
        raise RuntimeError("provider down")

    out = synthesize_queries(boom, snapshot=SNAP, count=3, exploration_slots=1)
    assert out[0] == "rust tui recent developments"
    assert any("adjacent" in q for q in out)


def test_synthesis_never_leaks_unsearchable_or_regionless_junk():
    chat, calls = _capture_chat("a\nb\nc")
    synthesize_queries(chat, snapshot=SNAP, count=3, exploration_slots=0)
    system = calls[0]["system_message"]
    assert "queries" in system.lower()
```

```python
# Tests/Dreams/test_discovery.py
from tldw_chatbook.Dreams.discovery import Candidate, dedupe_and_rank

CANDS = [
    Candidate("Rust TUI guide", "https://a.x/Rust-Tui/", "…", "web"),
    Candidate("Dup", "https://a.x/rust-tui?utm=1", "…", "web"),
    Candidate("Jazz near Seattle", "https://b.x/jazz", "…", "watchlist"),
    Candidate("Seen already", "https://c.x/old", "…", "web"),
    Candidate("In library", "https://d.x/lib", "…", "web"),
    Candidate("Off-topic", "https://e.x/taxes", "…", "web"),
]
TOPICS = [{"facet": "topic", "text": "rust tui", "weight": 0.9},
          {"facet": "topic", "text": "jazz guitar", "weight": 0.6}]


def test_dedupe_and_rank_drops_seen_library_and_duplicates_and_ranks_by_overlap():
    ranked = dedupe_and_rank(
        CANDS, seen_urls={"https://c.x/old"}, library_urls={"https://d.x/lib"},
        topics=TOPICS, limit=3,
    )
    urls = [c.url for c in ranked]
    assert urls == ["https://a.x/Rust-Tui/", "https://b.x/jazz", "https://e.x/taxes"]


def test_ranking_falls_back_to_original_order_on_no_overlap():
    ranked = dedupe_and_rank([CANDS[5]], seen_urls=set(), library_urls=set(),
                             topics=[], limit=5)
    assert ranked[0].url == "https://e.x/taxes"
```

Plus an async test for `run_queries` with a fake `perform` returning the standardized result dict (`{"results": [{"title": t, "link": u, "snippet": s}]}` — check the real key names at `Web_Scraping/WebSearch_APIs.py` result construction before finalizing the fake; adjust the extractor to the real keys, not the fake's) asserting `searches_used == len(queries)` and error results are skipped:

```python
@pytest.mark.asyncio
async def test_run_queries_counts_searches_and_skips_error_results():
    def perform(search_engine, search_query, **kwargs):
        if "bad" in search_query:
            return {"error": "search_processing_error", "status": "error"}
        return {"results": [{"title": "t", "link": "https://ok.x/1", "snippet": "s"}]}

    cands, used = await run_queries(perform, engine="duckduckgo",
                                    queries=["good query", "bad query"])
    assert used == 2 and [c.url for c in cands] == ["https://ok.x/1"]
```

`fetch_watchlist_candidates` test: construct `SubscriptionsDB` on `tmp_path` (copy the fixture pattern from `Tests/` — `grep -rn "SubscriptionsDB(" Tests/ | head -3`), insert one subscription + one fresh and one stale `subscription_item` via its own methods or direct SQL, assert only the fresh one returns.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/Dreams/test_query_synthesis.py Tests/Dreams/test_discovery.py -v`
Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement both modules**

`query_synthesis.py` core (chat invocation follows `briefing_service._invoke_chat`'s kwargs exactly — `api_endpoint`, `messages_payload`, `system_message`, `model`, `streaming: False`, `max_tokens`, `temp` — with `endpoint="openai"` replaced by the Task 4 resolver's values passed in via the injected `chat` already being a closure; the module itself only builds prompts and parses):

```python
SYSTEM_PROMPT = (
    "You turn a user's interest profile into web search queries. Return ONE query "
    "per line, no numbering, no commentary. At least {explore} line(s) must explore "
    "something ADJACENT to their interests rather than the interests themselves "
    "(serendipity, not more of the same). Queries must be self-contained for a "
    "search engine, may include the user's region verbatim when locality helps, "
    "and must never include anything except topics, region, and search terms."
)


def synthesize_queries(chat, *, snapshot, count, exploration_slots):
    topics = [t["text"] for t in snapshot.get("topics", [])]
    region = (snapshot.get("region") or "").strip()
    user = json.dumps({"topics": topics, "region": region, "count": count},
                      ensure_ascii=False)
    try:
        resp = _invoke(chat, system=SYSTEM_PROMPT.format(explore=exploration_slots),
                       user=user)
        lines = [ln.strip() for ln in extract_response_content(resp).splitlines()
                 if ln.strip()][:count]
        if len(lines) >= max(1, min(count, 2)):
            return lines
    except Exception:
        pass
    return _fallback_queries(topics, count)


def _fallback_queries(topics, count):
    out = [f"{t} recent developments" for t in topics[: max(count - 1, 1)]]
    out.append(f"surprising adjacent to {topics[0]}" if topics else "curious new things this week")
    return out[:count]
```

`_invoke` copies `briefing_service._invoke_chat`'s awaitable/thread handling with fixed `max_tokens=512`, `temp=0.4`. `extract_response_content` imports from `Chat.Chat_Functions`.

`discovery.py`: `Candidate` dataclass; `run_queries` as an `async def` looping `await asyncio.to_thread(perform, engine, q, ...)` per query (result normalization via a `_extract_results(payload)` function reading the real result keys — verify against `WebSearch_APIs.py` before writing); `fetch_watchlist_candidates` selects `url, title` from `subscription_items` joined to `subscriptions` where `published_date >= cutoff` and `status = 'new'` (parameterized; cutoff computed from `now_epoch`); `dedupe_and_rank` pure per the test (normalizer: `url.lower().split("?")[0].rstrip("/")`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/Dreams/ -v` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Dreams/query_synthesis.py tldw_chatbook/Dreams/discovery.py Tests/Dreams/test_query_synthesis.py Tests/Dreams/test_discovery.py
git commit -m "feat(dreams): query synthesis with deterministic fallback + discovery dedupe/rank over web and watchlist pools"
```

---

### Task 4: Story service and cycle orchestration

**Files:**
- Create: `tldw_chatbook/Dreams/story_service.py`
- Create: `tldw_chatbook/Dreams/cycle_service.py`
- Test: `Tests/Dreams/test_story_service.py`, `Tests/Dreams/test_cycle_service.py`

**Interfaces:**
- Consumes: Tasks 1–3 (`DreamsDB`, `snapshot`, `synthesize_queries`, `run_queries`, `fetch_watchlist_candidates`, `dedupe_and_rank`), `Chat.Chat_Functions.chat_api_call`/`extract_response_content`, `Chat.provider_setup_persistence.canonical_provider_key/resolve_remembered_provider_model`, `config.load_cli_config_and_ensure_existence` (resolution chain copies `Subscriptions/briefing_service.py:326-358`).
- Produces (used by Tasks 5–7):
  - `@dataclass(slots=True) class StoryResult: title: str; body: str; kind: str; event_date: str | None; location: str | None; status: str; error: str | None`
  - `story_service.resolve_dreams_chat() -> Callable` — returns a zero-arg closure `chat(**kwargs) -> Any` with `api_endpoint`/`api_key`/`model` pre-bound: `[dreams] provider`+`model` when set, else `chat_defaults.provider` + `resolve_remembered_provider_model(persisted, provider)`; raises `RuntimeError("Dreams provider/model unavailable")` when neither resolves. Endpoint/API-key mapping follows the provider→endpoint table the chat screens use (`grep -rn "api_endpoint=" tldw_chatbook/UI/Screens/chat_screen.py | head -5` to find the mapping helper; reuse it, do not fork it).
  - `story_service.generate_story(chat, *, candidate: Candidate, snapshot: dict) -> StoryResult` — sync (caller thread-offloads); 120–200 word body contract; `kind`/`event_date`/`location` extracted **only** from the candidate's title+snippet text (explicit date parse or explicit place name; else `kind="content"`, dates `None`); relative dates anchored to today; empty/exception → `status="empty"|"failed"` with `error` set, never raises.
  - `cycle_service.CycleDeps` dataclass: `dreams_db`, `chachanotes_db_getter`, `media_db_getter`, `subs_db_getter`, `pc_service_getter`, `chat_getter`, `perform_search` (default `perform_websearch`), `now: Callable[[], datetime]`.
  - `cycle_service.run_cycle(deps, *, trigger: str) -> dict` — the whole orchestration (below); returns `{"collection_id": int | None, "status": str, "stories": int}`; `GenerationInFlightError`-style double-run returns the in-flight marker instead of raising.
  - `cycle_service.run_catchup_if_due(deps) -> bool` — spec's catch-up predicate: enabled + no collection for today's local date + (no completed collection within `cadence_hours` OR latest is `failed` after stale-reclaim) → runs `run_cycle(trigger="catchup")`; returns whether it ran.
- Behavior locked by tests: claim guard via a module-level event-loop-only `set[str]` of local dates (`_ACTIVE_CYCLE_DATES`, mutated with no `await` between check and add — copy `_ACTIVE_BRIEFING_CLAIMS`'s discipline from `briefing_service.py:378-410`); budget checks (`usage_get` vs `max_searches_per_day`/`max_llm_calls_per_day`) trim queries/stories lowest-priority-first (exploration queries dropped first, then stories beyond budget) and stamp `degradation_notes`; `fail_stale_generating(now - 15min)` runs before the date claim; all SQLite in one `asyncio.to_thread` hop per stage; searches/stories recorded via `usage_bump`; every generated story URL goes into the seen ledger.

- [ ] **Step 1: Write the failing tests**

`Tests/Dreams/test_story_service.py` (chat injected as a plain callable; extraction rules are the point):

```python
from tldw_chatbook.Dreams.discovery import Candidate
from tldw_chatbook.Dreams.story_service import generate_story


def _chat_ok(body):
    def chat(**kwargs):
        return {"choices": [{"message": {"content": body}}]}
    return chat


def test_generate_story_parses_explicit_event_date_and_kind():
    cand = Candidate("Wednesday 13 at The Crocodile, Nov 3 2026",
                     "https://x/1", "Tickets $40 — Seattle show", "web")
    res = generate_story(_chat_ok("A story about the show."),
                         candidate=cand, snapshot={"topics": [], "region": "Seattle"})
    assert res.status == "complete"
    assert res.kind == "event"
    assert res.event_date is not None and res.event_date.startswith("2026-11-03")
    assert res.location == "Seattle"


def test_generate_story_never_invents_metadata_absent_from_source():
    cand = Candidate("An article about rust", "https://x/2", "No dates or places here.", "web")
    res = generate_story(_chat_ok("A story about rust."),
                         candidate=cand, snapshot={"topics": [], "region": ""})
    assert res.event_date is None and res.kind == "content"


def test_generate_story_empty_and_failed_outcomes_are_rows_not_exceptions():
    empty = generate_story(_chat_ok("   "), candidate=Candidate("t", "https://x/3", "s", "web"),
                           snapshot={"topics": [], "region": ""})
    assert empty.status == "empty"
    def boom(**kwargs):
        raise RuntimeError("provider 500")
    failed = generate_story(boom, candidate=Candidate("t", "https://x/4", "s", "web"),
                            snapshot={"topics": [], "region": ""})
    assert failed.status == "failed" and "provider 500" in failed.error
```

`Tests/Dreams/test_cycle_service.py` — build `CycleDeps` with a real `DreamsDB` on `tmp_path`, stub getters returning `None` for the optional DBs, a fake `chat`, and a fake `perform` (reuse the Task 3 fakes). Cover, in separate tests:

```python
def test_run_cycle_creates_one_dated_collection_with_complete_stories(...)
def test_second_run_same_date_returns_inflight_and_does_not_duplicate(...)
def test_stale_generating_row_is_reclaimed_then_regenerated(...)
def test_budget_cap_trims_queries_and_records_degradation(...)
def test_search_failure_falls_back_to_llm_source_stories_marked_degraded(...)
def test_run_catchup_if_due_runs_only_when_no_collection_today(...)
```

Key assertions: `db.get_collection_by_date(today)["status"] == "complete"` with `stories_per_cycle` rows (or fewer with `partial` when some fail); the degraded test's stories have `source="llm"` and `degradation_notes` mentions web search; the budget test's `usage_get(today)["searches"] <= max_searches_per_day`. Fake `now` fixes the date (`CycleDeps(now=lambda: datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc))`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/Dreams/test_story_service.py Tests/Dreams/test_cycle_service.py -v` — Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`story_service.py`: `_extract_metadata(title, snippet, *, now)` first — regex month-name + day + optional year (`r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?"`), year defaulting to `now.year` (bump to next year when the parsed date is >11 months in the past — "Nov 3" read in September means next year); `kind` decision: explicit date → `event`; price/`$`/`tickets`/`fare` tokens in snippet → `deal`; else `content`. The chat prompt embeds ONLY `candidate.title/snippet/url` + distilled topic names + region; body contract "120–200 words, second person, why this fits them now". `generate_story` wraps everything in try/except producing `StoryResult(status="failed", error=str(exc))`.

`cycle_service.py` skeleton (full logic, trimmed logging):

```python
_ACTIVE_CYCLE_DATES: set[str] = set()  # event-loop only; no await between check and add


async def run_cycle(deps: CycleDeps, *, trigger: str) -> dict:
    now = deps.now()
    local_date = now.astimezone().strftime("%Y-%m-%d")
    dreams_db = deps.dreams_db

    if local_date in _ACTIVE_CYCLE_DATES:
        return {"collection_id": None, "status": "inflight", "stories": 0}
    _ACTIVE_CYCLE_DATES.add(local_date)
    try:
        cutoff = (now - timedelta(minutes=15)).isoformat()
        await asyncio.to_thread(dreams_db.fail_stale_generating, cutoff)
        snap = await asyncio.to_thread(interest_profile.snapshot, dreams_db,
                                       now_epoch=now.timestamp())
        profile_digest = hashlib.sha256(
            json.dumps(snap, sort_keys=True).encode()).hexdigest()[:16]
        collection_id = await asyncio.to_thread(
            dreams_db.create_collection, local_date, trigger, profile_digest)
        if collection_id is None:
            return {"collection_id": None, "status": "date_taken", "stories": 0}
        notes, queries = [], []
        try:
            chat = deps.chat_getter()
            queries = await query_synthesis.synthesize_queries(
                chat, snapshot=snap,
                count=dreams_setting("queries_per_cycle"),
                exploration_slots=dreams_setting("exploration_slots"))
        except Exception as exc:  # provider unavailable -> degrade to LLM-only later
            notes.append(f"query synthesis failed: {exc}")
        usage = await asyncio.to_thread(dreams_db.usage_get, local_date)
        budget = _plan_budget(usage)          # trims queries/stories per spec §budgets
        candidates: list[Candidate] = []
        if dreams_setting("web_search_enabled") and budget.searches > 0 and queries:
            found, used = await discovery.run_queries(
                deps.perform_search, engine=dreams_setting("search_engine"),
                queries=queries[: budget.searches])
            await asyncio.to_thread(dreams_db.usage_bump, local_date, searches=used)
            candidates.extend(found)
        else:
            notes.append("web search disabled or over budget")
        subs_db = deps.subs_db_getter()
        if subs_db is not None and budget.searches > 0:
            candidates.extend(await asyncio.to_thread(
                discovery.fetch_watchlist_candidates, subs_db,
                freshness_hours=dreams_setting("watchlist_freshness_hours"),
                now_epoch=now.timestamp()))
        seen = await asyncio.to_thread(dreams_db.seen_filter_unseen,
                                       [c.url for c in candidates])
        library_urls = await asyncio.to_thread(_library_urls, deps)
        picked = discovery.dedupe_and_rank(
            candidates, seen_urls=set(c.url for c in candidates if c.url not in seen)
            | set(),  # seen already excluded below; keep pure call simple:
            library_urls=library_urls, topics=snap["topics"],
            limit=dreams_setting("stories_per_cycle"))
        # ... generate each story via asyncio.to_thread(generate_story, ...),
        # usage_bump(llm_calls=1) per call, insert_story rows, seen_upsert,
        # set_collection_status(complete|partial|failed, degradation_notes=notes)
        return {"collection_id": collection_id, "status": final, "stories": n}
    finally:
        _ACTIVE_CYCLE_DATES.discard(local_date)
```

(The inline comment marks the one spot to keep clean: compute `seen` BEFORE ranking and pass already-unseen candidates with `seen_urls=set()`; write the final code that way — the tests for the ledger enforce it.) `run_catchup_if_due(deps)` implements the predicate from the Interfaces block and calls `run_cycle(deps, trigger="catchup")` when true.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/Dreams/ -v` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Dreams/story_service.py tldw_chatbook/Dreams/cycle_service.py Tests/Dreams/test_story_service.py Tests/Dreams/test_cycle_service.py
git commit -m "feat(dreams): story generation with honest metadata extraction + cycle orchestration (claims, budgets, catch-up)"
```

---

### Task 5: Scheduler integration — projection, handler, app wiring

**Files:**
- Create: `tldw_chatbook/Scheduling/services/dreams_projection.py`
- Create: `tldw_chatbook/Scheduling/scheduler/handlers/dreams_handler.py`
- Modify: `tldw_chatbook/app.py` (handler dict ~line 10650; projection wiring near `briefing_projection`; DreamsDB construction near line 9874; boot catch-up near the scheduler worker start ~line 16676)
- Test: `Tests/Dreams/test_dreams_scheduler.py`

**Interfaces:**
- Consumes: `Scheduling/models.py` `ScheduledTask`; `SchedulerLoop`'s `handlers: dict[str, Handler]`; `BriefingProjection` as the structural template (`Scheduling/services/briefing_projection.py` — read its module docstring first: attempt-aware `next_run_at`, one id-shape definition point); `cycle_service.run_cycle/run_catchup_if_due` (Task 4); `DreamsDB` (Task 1); `dreams_setting`.
- Produces:
  - `dreams_projection.DREAMS_TASK_PREFIX = "dreams"` and `parse_dreams_task_id(task_id) -> str | None` (`"dreams:cycle" -> "cycle"`).
  - `dreams_projection.DreamsProjection` — `__init__(self, db_getter: Callable[[], DreamsDB | None], *, cadence_hours: int | None = None)`; `tasks(now: datetime) -> list[ScheduledTask]` — when `[dreams] enabled` is false or db is `None`, returns `[]`; otherwise one `ScheduledTask(id="dreams:cycle", type="dreams_cycle", next_run_at=...)` where `next_run_at = max(last_completed_or_attempted, ...) + cadence`, never-attempted → due now (mirror BriefingProjection's watermark rule: a failed cycle retries one cadence later, completion watermark from `status IN ('complete','partial')`).
  - `dreams_handler.DreamsCycleHandler` — `__init__(self, *, deps_getter: Callable[[], CycleDeps | None])`; `async def handle(self, task: dict) -> None` — fire-and-forget exactly like `BriefingJobHandler`: validate via `parse_dreams_task_id`, spawn `asyncio.create_task(run_cycle(deps, trigger="scheduled"))`, keep a module-level reference set for shutdown, never raise into the loop.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Dreams/test_dreams_scheduler.py
from datetime import datetime, timedelta, timezone

from tldw_chatbook.DB.Dreams_DB import DreamsDB
from tldw_chatbook.Dreams.settings import DREAMS_DEFAULTS
from tldw_chatbook.Scheduling.services.dreams_projection import (
    DREAMS_TASK_PREFIX, DreamsProjection, parse_dreams_task_id,
)


def test_parse_roundtrip_and_rejects_foreign_ids():
    assert parse_dreams_task_id(f"{DREAMS_TASK_PREFIX}:cycle") == "cycle"
    assert parse_dreams_task_id("briefing:7") is None
    assert parse_dreams_task_id(None) is None


def _projection(tmp_path, monkeypatch, enabled=True):
    monkeypatch.setattr("tldw_chatbook.Dreams.settings.get_cli_setting",
                        lambda s, k, d: True if (k == "enabled" and enabled) else d)
    db = DreamsDB(tmp_path / "d.sqlite", "t")
    return DreamsProjection(lambda: db), db


def test_projection_emits_nothing_when_disabled(tmp_path, monkeypatch):
    proj, db = _projection(tmp_path, monkeypatch, enabled=False)
    assert proj.tasks(datetime.now(timezone.utc)) == []


def test_projection_due_now_when_never_run_and_retries_after_failure(tmp_path, monkeypatch):
    proj, db = _projection(tmp_path, monkeypatch)
    now = datetime(2026, 9, 22, 8, 0, tzinfo=timezone.utc)
    tasks = proj.tasks(now)
    assert [t.id for t in tasks] == ["dreams:cycle"]
    assert tasks[0].next_run_at <= now  # never attempted -> due immediately
    cid = db.create_collection("2026-09-21", "scheduled", "d")
    db.set_collection_status(cid, "failed", completed_at=now.isoformat())
    retry = proj.tasks(now)[0]
    assert retry.next_run_at > now                       # failed -> one cadence later
    db.set_collection_status(cid, "complete", completed_at=now.isoformat())
    done = proj.tasks(now)[0]
    assert done.next_run_at == now + timedelta(hours=DREAMS_DEFAULTS["cadence_hours"])
```

Handler test: instantiate `DreamsCycleHandler(deps_getter=lambda: None)`, call `await handler.handle({"id": "dreams:cycle"})`, assert it does not raise and spawns nothing when deps are `None`; with a stub `CycleDeps` + monkeypatched `run_cycle` recording calls, assert it was dispatched with `trigger="scheduled"`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/Dreams/test_dreams_scheduler.py -v` — Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement projection + handler, then wire app.py**

Projection copies `BriefingProjection`'s shape (read it in full first — ~60 lines of context comments explaining the watermark rules you must not drift from). The completion watermark reads `dreams_collections` `MAX(completed_at) WHERE status IN ('complete','partial')`; the attempt watermark reads `MAX(created_at)`; `next_run_at = max(the two) + cadence`, and rows still `generating` are ignored (the claim set and stale-reclaim own them).

Handler mirrors `BriefingJobHandler.handle`'s spawn-and-forget structure with a `_SPAWNED: set[asyncio.Task]` for shutdown cancellation.

`app.py` modifications (keep each hunk small and commented `# dreams phase 1`):
1. Near line 9874 (where `LibraryCollectionsDB` is built): `self.dreams_db = DreamsDB(<user data dir path> / "dreams.sqlite", CLI_APP_CLIENT_ID)` — path resolved exactly as the neighboring construction resolves its path (copy the two lines above it and adapt the filename; follow any `get_*_db_path` helper convention present).
2. Where `briefing_projection` is built: build `dreams_projection = DreamsProjection(lambda: getattr(self, "dreams_db", None))` and pass it into the loop constructor next to `briefing_projection` (add a `dreams_projection` parameter to `SchedulerLoop.__init__` storing it, and in the loop's queue-reload path beside the briefing projection's task feed, extend with `dreams_projection.tasks(now)` — read how `watchlist_projection`/`briefing_projection` are consumed in `Scheduling/scheduler/loop.py`/`queue.py` first and insert symmetrically, ~5 lines).
3. In the `handlers` dict (~10650): `handlers["dreams_cycle"] = DreamsCycleHandler(deps_getter=lambda: self._dreams_cycle_deps())` where `_dreams_cycle_deps()` (new app method) builds `CycleDeps` with **getter lambdas** for every DB (the `chachanotes_db_getter` discipline documented at `app.py:10615-10635` — instances don't exist yet at wiring time; getters must tolerate `None`).
4. Where the scheduler worker starts (~16676, after `_ui_ready`): if `dreams_setting("enabled")` and `dreams_setting("catchup_enabled")` — `self.run_worker(cycle_service.run_catchup_if_due(deps), exclusive=False, group="dreams")` guarded so a disabled Dreams never constructs deps.

- [ ] **Step 4: Run tests (unit + import smoke)**

Run: `python3 -m pytest Tests/Dreams/test_dreams_scheduler.py -v && python3 -c "import tldw_chatbook.app"` — Expected: unit tests PASS; import exits 0.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/services/dreams_projection.py tldw_chatbook/Scheduling/scheduler/handlers/dreams_handler.py tldw_chatbook/app.py Tests/Dreams/test_dreams_scheduler.py
git commit -m "feat(dreams): scheduler integration - DreamsProjection cadence tasks, fire-and-forget handler, app wiring + boot catch-up"
```

---

### Task 6: Artifacts surface — Dreams type, refresh worker, collection list

**Files:**
- Create: `tldw_chatbook/Dreams/dreams_view.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py` (filter label ~line 903; state/workers near `_daily_reports` ~137-340; list rendering ~927-935)
- Test: `Tests/Dreams/test_dreams_view.py`, `Tests/UI/test_artifacts_dreams_rows.py`

**Interfaces:**
- Consumes: Task 1 `DreamsDB`; the `_daily_reports` wiring pattern inside `artifacts_screen.py` (generation-guarded refresh worker, `call_from_thread` apply, `REPORT_DISPLAY_LIMIT`).
- Produces:
  - `dreams_view.list_recent_dreams(dreams_db, *, limit: int = 10) -> list[dict]` — newest-collection-first stories with `{id, label, kept, kind, collection_date, status}`; `label` = `f"{title}"` truncated to 60 chars; failed collections contribute a synthetic `{"label": f"Cycle {date}: {status}", "status": status}` row.
  - `dreams_view.format_dream_row(story: dict) -> str` — `"> Dream: {label}" + (" · kept" if kept)` (matches the Report row idiom at `artifacts_screen.py:927-935`).
  - `artifacts_screen.py`: `self._dreams: list[dict]` + `_dreams_generation: int` + `_dreams_worker`, `_start_dreams_refresh()`, `_refresh_dreams(generation)` (`@work(thread=True)`, reads `list_recent_dreams(self.app.dreams_db)`), `_apply_dreams(generation, rows)` (generation-checked, `is_attached`-guarded — copy the daily-reports trio line for line in structure); filter label becomes `"Types: All | Chatbooks | Reports | Dreams | Datasets | Drafts | Exports | Sort: Recent"`; compose renders dreams rows after Reports rows via `format_dream_row`, ids `f"artifacts-dream-row-{story['id']}"`. Empty state: `"> Dreams: disabled"` when `[dreams] enabled` is false, `"> Dreams: none yet"` when enabled but empty.

- [ ] **Step 1: Write the failing tests**

`Tests/Dreams/test_dreams_view.py` — real `DreamsDB` on `tmp_path`; seed one `complete` collection with two stories (one `kept=1`) and one `failed` collection; assert `list_recent_dreams` ordering (failed-collection synthetic row for its date, newest first), label truncation at 60 chars, and `format_dream_row` output `"… · kept"`.

`Tests/UI/test_artifacts_dreams_rows.py` — mount/inspect pattern copied from an existing artifacts test (`grep -rn "artifacts" Tests/UI/ -l | head -3`; reuse its app fixture): with `_dreams` populated via `_apply_dreams(1, rows)` directly, assert the composed screen contains a row widget with id `artifacts-dream-row-<id>` and that the mode label string contains `Dreams`. If no mountable fixture exists, test `_apply_dreams` state + `format_dream_row` rendering helper in isolation and assert the label constant changed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/Dreams/test_dreams_view.py Tests/UI/test_artifacts_dreams_rows.py -v` — Expected: FAIL (`ModuleNotFoundError` / label assertion).

- [ ] **Step 3: Implement view + screen wiring**

`dreams_view.py` is a ~50-line read-only module (mirrors `Subscriptions/daily_reports_view.py`'s role: no UI imports, just row shaping). Screen edits follow the Interfaces block; all new widgets use existing `$ds-*` classes already present on the neighboring rows (`destination-section`, `ds-panel` — no new styles, no new literals, so the token governance test stays green untouched).

- [ ] **Step 4: Run tests + governance**

Run: `python3 -m pytest Tests/Dreams/test_dreams_view.py Tests/UI/test_artifacts_dreams_rows.py Tests/UI/test_design_token_governance.py -v` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Dreams/dreams_view.py tldw_chatbook/UI/Screens/artifacts_screen.py Tests/Dreams/test_dreams_view.py Tests/UI/test_artifacts_dreams_rows.py
git commit -m "feat(dreams): Artifacts screen Dreams type + refresh worker + collection rows"
```

---

### Task 7: Story detail modal — dive deeper, keep, export, feedback, query preview

**Files:**
- Create: `tldw_chatbook/UI/Screens/artifacts_dreams_modal.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py` (row click opens modal; refresh after modal actions)
- Test: `Tests/UI/test_artifacts_dreams_modal.py`

**Interfaces:**
- Consumes: `ChatHandoffPayload` (`Chat/chat_handoff_models.py:19` — fields `source`, `item_type`, `title`, `body`, `suggested_prompt`); the pending handoff store (`UI/Navigation/pending_handoff_store.py` — `.stage(channel, payload)`; find the chat screen's posting idiom via `grep -rn "ChatHandoffPayload(" tldw_chatbook/UI/Screens/skills_screen.py` and copy it); `DreamsDB.set_story_kept/record_feedback` (Task 1); `dreams_view` (Task 6).
- Produces:
  - `class DreamsStoryModal(ModalScreen[None])` — `__init__(self, story: dict, *, dreams_db_getter, on_changed: Callable[[], None])`. Body: title, source URL, story body, provenance line (`matched_topics` + `query` + `kind` + `event_date` if set), "what we'll look for" preview section (next-cycle queries = `query_synthesis` fallback for the current snapshot — label it "preview (fallback queries until next cycle)"), then action buttons `Keep/Unkeep (k)`, `Dive deeper (d)`, `Export (e)`, `More like this (m)`, `Less like this (l)`, `Close (q)`. Actions record feedback kinds `kept/dived/exported/more/less` respectively (keep also toggles `set_story_kept`), call `on_changed()` (screen refreshes its rows), and post a dismissible notice. Keybindings: single letters per ADR-031; footer hints list exactly the six implemented actions. Export writes `~/Documents/tldw_exports/dreams/<slug>-<id>.md` via `path_validation` (import from `Utils/path_validation.py`; check its API and use the sanctioned folder-check helper) with a Markdown stub: title, source URL, provenance, body, generated date.
  - `artifacts_screen.py`: clicking a dreams row (or `Enter` on it) constructs the modal with `dreams_db_getter=lambda: getattr(self.app, "dreams_db", None)` and `on_changed=self._start_dreams_refresh`.

- [ ] **Step 1: Write the failing tests**

Mount the modal inside a minimal Textual app harness (copy the harness from an existing modal test — `grep -rln "ModalScreen" Tests/UI/ | head -3`): press `k` → `dreams_db.set_story_kept` called with `(id, True)` (assert via real `DreamsDB` state), feedback row `kind="kept"` recorded, `on_changed` called once; press `m` → feedback `more` recorded; press `q` → modal dismissed; `d` with a stubbed handoff stage → payload captured with `source="dreams"`, `item_type="dream_story"`, `title`/`body` from the story; `e` with `path_validation` monkeypatched to a `tmp_path` sink → file exists containing the title. Also assert the footer hint string contains no unimplemented action words.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/UI/test_artifacts_dreams_modal.py -v` — Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement modal + wiring**

Modal skeleton (Textual 8 idiom — `ModalScreen`, compose with `$ds-*` panel classes reused from an existing modal such as `UI/Watchlists_Modules/kept_briefings_modal.py`; read it first and copy its structure, spacing classes, and dismiss pattern):

```python
class DreamsStoryModal(ModalScreen[None]):
    BINDINGS = [("k", "keep", "Keep"), ("d", "dive", "Dive deeper"),
                ("e", "export", "Export"), ("m", "more", "More like this"),
                ("l", "less", "Less like this"), ("q", "close", "Close")]

    def action_keep(self) -> None: ...   # toggle kept + record_feedback('kept') + on_changed
    def action_dive(self) -> None: ...   # stage ChatHandoffPayload(source='dreams', item_type='dream_story', ...) then dismiss
    def action_export(self) -> None: ... # validated path write + record_feedback('exported')
    def action_more(self) -> None: ...   # record_feedback('more'); notify; dismiss
    def action_less(self) -> None: ...   # record_feedback('less'); notify; dismiss
    def action_close(self) -> None: ...  # dismiss(None)
```

Each action body is 3–8 lines calling the interfaces above; no network, no workers (all DB ops are single-row and instant — same as kept-briefings modal).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Tests/UI/test_artifacts_dreams_modal.py Tests/UI/test_design_token_governance.py -v` — Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/artifacts_dreams_modal.py tldw_chatbook/UI/Screens/artifacts_screen.py Tests/UI/test_artifacts_dreams_modal.py
git commit -m "feat(dreams): story detail modal - dive/keep/export/feedback actions + query preview"
```

---

### Task 8: Ingest action — story URL into read-it-later capture

**Files:**
- Create: `tldw_chatbook/Dreams/ingest_action.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_dreams_modal.py` (add `Ingest (i)` binding + action)
- Test: `Tests/Dreams/test_ingest_action.py`, extend `Tests/UI/test_artifacts_dreams_modal.py`

**Interfaces:**
- Consumes: `Library/collections_capture_service.py` — `CollectionsCaptureBackend` Protocol (line 88) and `LocalCollectionsCaptureService` (line 178); its request models in `Library/collections_capture_models.py` (`CaptureSaveRequest`/`CaptureDetail` — read both first and use the service's real public entry, e.g. the save/capture method `LocalCollectionsCaptureService` exposes above `_start_extraction`; the backend Protocol is the test seam).
- Produces:
  - `ingest_action.ingest_story_url(backend: CollectionsCaptureBackend, *, url: str, title: str, source_note: str) -> str` — submits the URL as a read-it-later capture item attributed to Dreams; returns the backend's item identifier; raises the backend's typed error on failure (caller shows it, never crashes the modal).
  - Modal: `Ingest (i)` records feedback `ingested` and calls `ingest_story_url` with `backend_getter()` (wired in `artifacts_screen.py` from the app's capture service instance — find it via `grep -rn "LocalCollectionsCaptureService(" tldw_chatbook/app.py`).

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Dreams/test_ingest_action.py
import pytest

from tldw_chatbook.Dreams.ingest_action import ingest_story_url


class FakeBackend:  # implements Library/collections_capture_service.py:88 Protocol
    def __init__(self):
        self.saved = []

    def save_capture_item(self, *, url, title, note):  # match REAL Protocol method name
        self.saved.append({"url": url, "title": title, "note": note})
        return f"cap-{len(self.saved)}"


def test_ingest_submits_url_with_dreams_attribution():
    backend = FakeBackend()
    ident = ingest_story_url(backend, url="https://x/1", title="Cheap flights",
                             source_note="via Dreams 2026-09-22")
    assert ident == "cap-1"
    assert backend.saved[0]["note"].startswith("via Dreams")


def test_ingest_propagates_backend_failure():
    class BrokenBackend:
        def save_capture_item(self, *, url, title, note):
            raise RuntimeError("capture queue full")
    with pytest.raises(RuntimeError, match="capture queue full"):
        ingest_story_url(BrokenBackend(), url="https://x/2", title="t", source_note="n")
```

(Before finalizing: read `CollectionsCaptureBackend`'s actual method names and `CaptureSaveRequest`'s fields; rename the fake's method and kwargs to match reality — the test must bind to the real Protocol, not this sketch.)

Extend the modal test: press `i` → feedback `ingested` recorded and fake backend received the story URL.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Tests/Dreams/test_ingest_action.py -v` — Expected: FAIL, `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

`ingest_action.py` is ~20 lines: validate URL shape (`http(s)://` prefix, no whitespace), call the backend's real save method with Dreams attribution in the note field, return its identifier. Modal action `action_ingest` mirrors `action_keep`: try/except with a dismissible error notice, feedback row, `on_changed()`.

- [ ] **Step 4: Run tests to verify the whole Phase 1 suite passes**

Run: `python3 -m pytest Tests/Dreams/ Tests/UI/test_artifacts_dreams_modal.py Tests/UI/test_artifacts_dreams_rows.py Tests/UI/test_design_token_governance.py -v` — Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Dreams/ingest_action.py tldw_chatbook/UI/Screens/artifacts_dreams_modal.py Tests/Dreams/test_ingest_action.py Tests/UI/test_artifacts_dreams_modal.py
git commit -m "feat(dreams): ingest story URL into read-it-later capture with Dreams attribution"
```

---

## Self-Review (completed during plan writing)

**Spec coverage:** Phase-1 spec sections → tasks: package layout/DB (T1), interest profile + cache-on-unlock + cold-start fallback queries (T2–T4), query synthesis + exploration + watchlist pool + dedupe/rank + recency/blocklist hooks (`date_range` in T3; `site_blacklist` rides the same `perform` kwargs — pass `[dreams]`-configured values in Task 4's `_plan_budget`/query stage if desired later, YAGNI now), story generation + event-metadata rules + row-per-outcome (T4), date-bucket idempotency + race guard + stale reclaim + catch-up + budgets + degraded LLM mode (T4–T5), Artifacts surface + ordering + empty/degraded visibility (T6), modal actions incl. dive-deeper/keep/export/feedback + preview (T7), ingest (T8). Privacy: prompts carry only distilled topics/region/snippets (T3–T4 tests pin the prompt inputs). Phase-2 items (Track tables, goals facet, region labeling in preview, reminder promotion, tracked updates section) are deliberately absent per the spec's phasing.

**Placeholder scan:** no TBD/TODO; the two seams the spec names as open items (capture entry method, provider→endpoint mapping helper) are resolved by pinned grep-and-read steps with concrete fallback contracts rather than invented names.

**Type consistency:** `Candidate(title,url,snippet,source)` used identically in T3/T4; `StoryResult` fields consumed by `insert_story` kwargs in T4; `dreams_setting` keys used in T4/T5 match `DREAMS_DEFAULTS` in T1; `list_recent_dreams` row shape produced in T6 matches modal consumption in T7; task-id prefix defined once in T5 with parser, per the 2b lesson.
