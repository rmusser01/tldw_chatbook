# Daily Reports Surface and Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing watchlist-briefing pipeline into a user-facing "Daily Reports" experience: a Reports list on the Artifacts screen, completion notifications for scheduled briefings, and a one-click live demo that seeds a real "Daily Brief" watchlist and runs it immediately (text brief + TTS audio when possible).

**Architecture:** "Daily Report" is a lens over the existing briefing tables — a new read-only DB join + view module feeds the Artifacts screen's Reports slot; a new `DailyReportDemoService` seeds real rows (watchlist, sources, preset, cadence) and drives the *real* run-now seams (`LocalWatchlistsService` → `generate_briefing` → `generate_script_audio`); `BriefingJobHandler` gains a dispatch-service param mirroring `ReminderHandler`. No new tables, no new generation code, no scheduler changes.

**Tech Stack:** Python ≥3.11, Textual 8.x, SQLite (SubscriptionsDB / ChaChaNotes), pytest (asyncio_mode=auto), optional pydub (audio stitching).

**Spec:** `Docs/superpowers/specs/2026-08-29-daily-reports-demo-design.md` (commit `c437ccadf`) — the plan argues from the spec; executors read both.

## Global Constraints

- **No schema changes**: zero new tables or columns anywhere (ADR-079; ADR-078 "no second universal artifact database"). If a task seems to need one, stop and re-read the spec.
- **No new dependencies.** pydub stays optional (`audio` extra); the demo must succeed text-only without it.
- **Notification category string is exactly `briefing`** (open-ended category, policy defaults all-True; no registration step exists or is needed).
- **Config keys** (both under the existing `[scheduling]` section, template lives in `tldw_chatbook/config.py` around line 3142): `daily_report_demo_banner_dismissed` (default `false`). Read with `get_cli_setting(section, key, default)`; write with `save_setting_to_cli_config(section, key, value)` (`config.py:5844`). `set_cli_setting` does NOT exist — never call it.
- **Fixed names**: demo watchlist `"Daily Brief"`, demo preset `"Daily Brief"`, cadence `86400` seconds, demo sources exactly the three in `DEMO_SOURCES` below.
- **Testing policy**: targeted runs only — run the specific files each task names, never a full sweep unless the user asks. Markers are strict (`--strict-markers`): service/DB tests use `pytestmark = pytest.mark.unit`, screen tests `pytest.mark.ui`. DBs in tests are **file-backed under `tmp_path`** (`SubscriptionsDB(tmp_path / "subs.db", "test")`), never `:memory:` — `SubscriptionsDB.conn` is thread-local and `asyncio.to_thread` hops would see an empty DB. Fake exactly one named seam per test (chat via DI `chat=...`; fetch via monkeypatching `monitoring_engine.guarded_fetch_httpx_async`).
- **Off-event-loop discipline** (task-15463): DB work from async code goes through `asyncio.to_thread`; UI workers use `exclusive=True` + a named group; the single `SubscriptionsDB` app instance is passed around, never rebuilt.
- **Scheduled-generation safety**: the demo must call `generate_briefing` directly (its claim machinery serializes against the scheduler); never insert briefing rows by hand outside tests.
- **Ignore `.worktrees/` and `build/` in every grep** — stale copies of every file exist there.
- **Keybindings**: add no keybindings; new UI is buttons only (backlog/decisions/031).
- **Mutation-proof every guard**: for each new guard test, the TDD "verify it fails" step must show the test failing for the reason you wrote it (red) before the implementation step.
- **Commits**: one per task step where indicated, messages matching repo history style (`feat:`, `test:`, `docs:`).

## Spec deviations this plan locks in (all narrow, all reasoned)

1. **Roster/voice correction (spec §3 seed step said "remote backends' default voices")**: `resolve_roster_voices` (`Subscriptions/briefing_voices.py:149`) *raises* for a speaker with no `voice_profile_id` — there is no default-voice fallback. Therefore the demo builds its cast roster from the user's existing TTS profiles at seed time (`TTSProfileService.list_profiles()`), and when zero profiles exist it seeds a one-speaker roster with `voice_profile_id: None` and skips the audio stage with a Settings hint. This matches the spec's error table ("No TTS backend / pydub missing → text brief succeeds; audio skipped").
2. **Provider preflight mechanism (spec: "preflight aborts if no LLM provider")**: there is no cheap verified "provider key configured" seam; `default_briefing_provider()` always returns a non-empty endpoint (config owns the fallback). The demo preflights what is checkable (services available, no existing schedule) and treats a `failed` briefing row as the authoritative provider-failure signal, dispatching a guidance notification that names the row's error and points at Settings (F9) → API. Spec intent (no silent burn, clear guidance) preserved.
3. **Demo sources are three RSS feeds** (Hacker News via `https://hnrss.org/frontpage`, BBC World, Ars Technica), all `source_type="rss"` — the most-supported monitoring path. The HN scraper's subscription-type wiring is unverified; HN's official RSS gets the same content with zero new seams.
4. **Phase 3 (first-run onboarding card) is out of this plan** — its surface doesn't exist yet; it gets its own follow-up plan once Home onboarding lands (spec phasing already orders it last).

## File Structure

- Create: `tldw_chatbook/Subscriptions/daily_reports_view.py` — read-only derivation of UI-shaped report rows from briefing tables.
- Create: `tldw_chatbook/Subscriptions/daily_report_demo.py` — `DailyReportDemoService` (preflight, seed, run-now orchestration, stage notifications).
- Create: `backlog/decisions/079-daily-reports-surface-and-demo-seeding.md` — ADR.
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` — add `list_recent_briefings(limit)` after `list_briefing_schedules` (~line 3383).
- Modify: `tldw_chatbook/Scheduling/scheduler/handlers/briefing_handler.py` — dispatch params + completion notification.
- Modify: `tldw_chatbook/app.py` — wire dispatch into `BriefingJobHandler`; construct `daily_report_demo_service`.
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py` — Reports refresh worker, rows, play/open/demo handlers.
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` — demo banner + handlers.
- Modify: `tldw_chatbook/config.py` — `[scheduling]` template key.
- Test create: `Tests/Subscriptions/test_daily_reports_view.py`, `Tests/Subscriptions/test_daily_report_demo.py`, `Tests/UI/test_artifacts_screen_reports.py`, `Tests/Watchlists/test_watchlists_demo_banner.py`.
- Test modify: `Tests/Scheduling/test_briefing_handler.py` (append notification tests).

---

### Task 1: ADR-079 and backlog task bookkeeping

**Files:**
- Create: `backlog/decisions/079-daily-reports-surface-and-demo-seeding.md`

**Interfaces:**
- Consumes: the approved spec (path above).
- Produces: ADR-079 (linked later from the backlog task's plan/notes); a backlog task id referenced by Tasks 2-8 commit messages as `TASK-<id>` (substitute the id the CLI prints).

- [ ] **Step 1: Create the backlog task and record its id**

```bash
backlog task create "Daily Reports surface and demo" \
  -d "Surface scheduled watchlist briefings as 'Daily Reports' on the Artifacts screen, notify on scheduled briefing completion, and add a one-click live demo that seeds a real Daily Brief watchlist (RSS sources, preset, daily cadence) and runs it immediately - text brief plus TTS audio when a voice profile exists. Spec: Docs/superpowers/specs/2026-08-29-daily-reports-demo-design.md; ADR-079." \
  --ac "Artifacts Reports slot lists recent briefings across watchlists with play/open actions and an empty-state demo CTA" \
       "Scheduled briefing completion dispatches a 'briefing' notification through NotificationDispatchService" \
       "One-click demo seeds watchlist+sources+preset+24h cadence idempotently and generates a text brief live" \
       "Demo synthesizes audio when a TTS voice profile + pydub exist; otherwise skips audio with a Settings hint and still succeeds" \
       "Watchlists screen shows a dismissible demo banner only while no briefing schedule exists" \
       "No new tables, columns, or dependencies" \
  -s "In Progress"
backlog task list -s "In Progress" --plain
```

Record the printed task id (e.g. `task-19700`) — call it `$TASK` below; use it in commit messages as `($TASK)`.

- [ ] **Step 2: Write ADR-079**

Create `backlog/decisions/079-daily-reports-surface-and-demo-seeding.md` with exactly this content (number 079 is verified free: 077 is reserved by the pending server-offload rename per task-19610, 078 exists):

```markdown
# ADR-079: Daily Reports surface and demo seeding

- **Status:** Accepted
- **Date:** 2026-08-29
- **Task:** `$TASK` — Daily Reports surface and demo
- **Spec:** [2026-08-29 daily-reports-demo-design](../../Docs/superpowers/specs/2026-08-29-daily-reports-demo-design.md)
- **Amends:** nothing; aligns with [ADR-015](015-shell-destinations.md) (Artifacts charter already includes "reports") and [ADR-078](078-research-workspace-authority-and-screen-boundaries.md) (no second universal artifact database).

## Context

The watchlist-briefing pipeline already produces scheduled text briefs with cast
scripts and synthesized audio, but it is invisible as a product: the Artifacts
screen's Reports slot is a hardcoded "none available" placeholder, a new user
must hand-wire watchlist + preset + schedule before anything runs, and scheduled
briefings complete silently (only reminders dispatch notifications).

## Decision

1. **"Daily Report" is a briefing.** The Artifacts screen's Reports slot is fed
   by a read-only view over the existing `briefings`/`briefing_scripts`/
   `briefing_audio` tables across all watchlists. No new artifact store, no new
   tables, no new scheduler task types (ADR-078 direction: presentation adapters
   over canonical owners).
2. **The demo writes real, persistent data by design.** The one-click demo seeds
   a real "Daily Brief" watchlist (three RSS sources incl. Hacker News via
   hnrss.org), a briefing preset, and a 24h cadence, then drives the existing
   run-now seams (claim-path `generate_briefing`, `generate_script`,
   `generate_script_audio`). The seeded setup *is* the user's first daily
   report; it keeps running via the existing `BriefingProjection`/`BriefingJobHandler`.
   Idempotency keys on configured briefing schedules (`list_briefing_schedules`),
   never on names.
3. **Audio is a progressive enhancement.** The cast roster is built from the
   user's existing TTS voice profiles; with zero profiles (or without pydub) the
   demo completes text-only and records an "audio skipped, here's how to enable
   it" hint. `resolve_roster_voices` has no default-voice fallback.
4. **Scheduled briefing completion dispatches one notification** (category
   `"briefing"`, success or attention) through `NotificationDispatchService`,
   policy-gated like `"reminder"`. Stage-by-stage notifications exist only
   during the interactive demo run.
5. **Demo discovery**: Artifacts-screen empty-state CTA and a dismissible
   Watchlists banner (hidden while any briefing schedule exists; dismissal
   persists at `scheduling.daily_report_demo_banner_dismissed`). First-run
   onboarding is follow-up work.

## Consequences

- One artifact authority (SubscriptionsDB briefing tables); the Artifacts
  Reports slot can never disagree with the Watchlists artifacts pane.
- The demo consumes real API quota (LLM + TTS); CTA copy says so.
- Briefing schedules remain rolling cadences (`briefing_cadence_seconds`) — a
  "preferred time of day" remains follow-up work (spec Follow-ups).
```

Fix the `Task:` line to the real `$TASK` id, and fix the two relative ADR links if `015`/`078` filenames differ (`ls backlog/decisions/ | grep -E "^(015|078)"`).

- [ ] **Step 3: Commit**

```bash
git add backlog/decisions/079-daily-reports-surface-and-demo-seeding.md
git commit -m "docs: ADR-079 daily reports surface and demo seeding ($TASK)"
```

---

### Task 2: Daily Reports read path (DB join + view module)

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` (add method after `list_briefing_schedules`, which ends around line 3383)
- Create: `tldw_chatbook/Subscriptions/daily_reports_view.py`
- Test: `Tests/Subscriptions/test_daily_reports_view.py`

**Interfaces:**
- Consumes: `SubscriptionsDB` DDL tables `briefings`/`briefing_scripts`/`briefing_audio`/`watchlists`; `briefing_audio.audio_file_path_is_safe(file_path) -> bool`.
- Produces:
  - `SubscriptionsDB.list_recent_briefings(self, limit: int = 20) -> List[Dict[str, Any]]` — rows keyed `briefing_id, watchlist_id, watchlist_name, status, created_at, item_count, model_used, complete_script_count, complete_audio_count, latest_audio_file_path`, newest first.
  - `daily_reports_view.list_recent_reports(db: Any, *, limit: int = 20) -> list[dict[str, Any]]` — rows keyed `id, watchlist_id, watchlist_name, status, created_at, item_count, model_used, has_audio, audio_file_path, label`. `audio_file_path` is non-None **only** when a complete audio row exists AND the path passes the safety guard.

- [ ] **Step 1: Write the failing tests**

Create `Tests/Subscriptions/test_daily_reports_view.py`:

```python
"""Daily Report rows: the cross-watchlist DB join and the view derivation.

Real `SubscriptionsDB` under `tmp_path`, real write paths for seeding (the
persist seams are the ones production uses); the only faked collaborator is
the path-safety guard, and only in the test that pins its effect.
"""

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import daily_reports_view
from tldw_chatbook.Subscriptions.daily_reports_view import list_recent_reports
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService

pytestmark = pytest.mark.unit


def _db(tmp_path) -> SubscriptionsDB:
    """File-backed DB -- thread-local connections make `:memory:` unusable."""
    return SubscriptionsDB(tmp_path / "subs.db", "test")


def _watchlist(db, name: str) -> int:
    return int(WatchlistBundleService(db).create(name)["id"])


def _briefing(db, watchlist_id: int, *, status: str = "complete") -> int:
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        briefing_id,
        status=status,
        body_markdown="## Daily Brief\n\nOne story [item 1].",
        item_count=1,
    )
    return briefing_id


def _complete_script_with_audio(db, briefing_id: int, file_path: str) -> None:
    script_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Daily Brief",
        roster_snapshot_json="[]",
    )
    db.update_briefing_script(script_id, status="complete", turns_json="[]")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    db.update_briefing_audio(
        audio_id, status="complete", file_path=file_path,
        duration_seconds=1.0, turn_count=1,
    )


def test_list_recent_briefings_orders_newest_first_across_watchlists(tmp_path):
    db = _db(tmp_path)
    w1, w2 = _watchlist(db, "Tech"), _watchlist(db, "World")
    b1 = _briefing(db, w1)
    b2 = _briefing(db, w2, status="empty")
    b3 = _briefing(db, w1)

    rows = db.list_recent_briefings(limit=10)

    assert [r["briefing_id"] for r in rows] == [b3, b2, b1]  # same-second ties break on id DESC
    by_id = {r["briefing_id"]: r for r in rows}
    assert by_id[b2]["watchlist_name"] == "World"
    assert by_id[b2]["status"] == "empty"
    assert by_id[b1]["item_count"] == 1


def test_list_recent_briefings_rejects_bad_limit(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(ValueError):
        db.list_recent_briefings(limit=0)
    with pytest.raises(ValueError):
        db.list_recent_briefings(limit=True)


def test_list_recent_briefings_honors_limit(tmp_path):
    db = _db(tmp_path)
    w = _watchlist(db, "Tech")
    for _ in range(3):
        _briefing(db, w)
    assert len(db.list_recent_briefings(limit=2)) == 2


def test_report_rows_surface_audio_only_through_the_safety_guard(tmp_path, monkeypatch):
    db = _db(tmp_path)
    w = _watchlist(db, "Tech")
    b1 = _briefing(db, w)
    _complete_script_with_audio(db, b1, "/armored/briefing_audio/script-1-audio-1.wav")
    b2 = _briefing(db, w)  # text-only report

    monkeypatch.setattr(daily_reports_view, "audio_file_path_is_safe", lambda p: True)
    rows = list_recent_reports(db, limit=10)
    by_id = {r["id"]: r for r in rows}
    assert by_id[b1]["has_audio"] is True
    assert by_id[b1]["audio_file_path"] == "/armored/briefing_audio/script-1-audio-1.wav"
    assert by_id[b2]["has_audio"] is False
    assert by_id[b2]["audio_file_path"] is None

    monkeypatch.setattr(daily_reports_view, "audio_file_path_is_safe", lambda p: False)
    rows = list_recent_reports(db, limit=10)
    by_id = {r["id"]: r for r in rows}
    assert by_id[b1]["has_audio"] is False
    assert by_id[b1]["audio_file_path"] is None  # unsafe path never reaches the UI


def test_report_rows_label_watchlist_status_and_audio(tmp_path, monkeypatch):
    db = _db(tmp_path)
    w = _watchlist(db, "Daily Brief")
    b1 = _briefing(db, w)
    _complete_script_with_audio(db, b1, "/x/y.wav")
    _briefing(db, w, status="failed")

    monkeypatch.setattr(daily_reports_view, "audio_file_path_is_safe", lambda p: True)
    rows = list_recent_reports(db, limit=10)

    assert rows[1]["label"].startswith("Daily Brief — ")
    assert "audio" in rows[1]["label"]
    assert "(failed)" in rows[0]["label"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Subscriptions/test_daily_reports_view.py -v`
Expected: FAIL — `AttributeError: 'SubscriptionsDB' object has no attribute 'list_recent_briefings'` (and `ModuleNotFoundError` for `daily_reports_view` if DB method is attempted first; create an empty `daily_reports_view.py` with just the docstring below if the import error blocks the DB test from even collecting — that empty file is Step 3's first lines anyway).

- [ ] **Step 3: Implement the DB method and the view module**

In `tldw_chatbook/DB/Subscriptions_DB.py`, directly after `list_briefing_schedules`'s `return` (~line 3383), add:

```python
    def list_recent_briefings(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Recent briefings across ALL watchlists, newest first.

        The Artifacts screen's Reports slot reads through this -- one bounded
        query instead of per-watchlist fan-out (ADR-079). Narrow projection on
        purpose (task-15464 pattern): no ``body_markdown`` blobs; the body is
        read on open in the Watchlists artifacts pane.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        with self.transaction() as conn:
            rows = conn.execute(
                """
                SELECT
                    b.id AS briefing_id,
                    b.watchlist_id AS watchlist_id,
                    w.name AS watchlist_name,
                    b.status AS status,
                    b.created_at AS created_at,
                    b.item_count AS item_count,
                    b.model_used AS model_used,
                    (
                        SELECT COUNT(*) FROM briefing_scripts AS s
                        WHERE s.briefing_id = b.id AND s.status = 'complete'
                    ) AS complete_script_count,
                    (
                        SELECT COUNT(*) FROM briefing_audio AS a
                        JOIN briefing_scripts AS s ON s.id = a.script_id
                        WHERE s.briefing_id = b.id
                          AND a.status = 'complete'
                          AND a.file_path IS NOT NULL
                    ) AS complete_audio_count,
                    (
                        SELECT a.file_path FROM briefing_audio AS a
                        JOIN briefing_scripts AS s ON s.id = a.script_id
                        WHERE s.briefing_id = b.id
                          AND a.status = 'complete'
                          AND a.file_path IS NOT NULL
                        ORDER BY a.id DESC
                        LIMIT 1
                    ) AS latest_audio_file_path
                FROM briefings AS b
                JOIN watchlists AS w ON w.id = b.watchlist_id
                ORDER BY b.created_at DESC, b.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]
```

Create `tldw_chatbook/Subscriptions/daily_reports_view.py`:

```python
"""Read-only aggregation of watchlist briefings into Daily Report rows.

A "Daily Report" is a briefing (ADR-079). This module is the thin derivation
layer between the `briefings` tables and the Artifacts screen's Reports slot:
no writes, no new tables, no caching. Callers own thread discipline -- call
from a worker thread or wrap in `asyncio.to_thread`.
"""

from __future__ import annotations

from typing import Any, Mapping

from tldw_chatbook.Subscriptions.briefing_audio import audio_file_path_is_safe

_STATUS_MARKERS = {
    "complete": "",
    "empty": " (empty)",
    "failed": " (failed)",
    "generating": " (writing…)",
}


def list_recent_reports(db: Any, *, limit: int = 20) -> list[dict[str, Any]]:
    """Recent briefings across all watchlists, newest first, UI-shaped.

    Args:
        db: The single ``SubscriptionsDB`` instance.
        limit: Maximum rows.

    Returns:
        One dict per briefing with keys ``id``, ``watchlist_id``,
        ``watchlist_name``, ``status``, ``created_at``, ``item_count``,
        ``model_used``, ``has_audio``, ``audio_file_path`` (non-None only for
        a complete audio row whose path passes the safety guard), ``label``.
    """
    rows = db.list_recent_briefings(limit=limit)
    return [_to_report_row(row) for row in rows]


def _to_report_row(row: Mapping[str, Any]) -> dict[str, Any]:
    audio_path = row.get("latest_audio_file_path")
    has_audio = bool(audio_path) and audio_file_path_is_safe(audio_path)
    return {
        "id": int(row["briefing_id"]),
        "watchlist_id": int(row["watchlist_id"]),
        "watchlist_name": str(
            row.get("watchlist_name") or f"Watchlist {row['watchlist_id']}"
        ),
        "status": str(row.get("status") or ""),
        "created_at": row.get("created_at"),
        "item_count": row.get("item_count") or 0,
        "model_used": row.get("model_used"),
        "has_audio": has_audio,
        "audio_file_path": str(audio_path) if has_audio else None,
        "label": _label(row),
    }


def _label(row: Mapping[str, Any]) -> str:
    name = row.get("watchlist_name") or f"Watchlist {row.get('watchlist_id')}"
    marker = _STATUS_MARKERS.get(
        str(row.get("status") or ""), f" ({row.get('status')})"
    )
    audio = " · audio" if row.get("complete_audio_count") else ""
    return f"{name} — {row.get('created_at', '')}{marker}{audio}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Subscriptions/test_daily_reports_view.py -v`
Expected: 5 PASS.

Then prove the ordering guard discriminates (lessons-testing-evidence: a guard must be *proven* able to go red): temporarily change `ORDER BY b.created_at DESC, b.id DESC` to `ASC, ASC`, re-run `test_list_recent_briefings_orders_newest_first_across_watchlists` — it must FAIL — then revert and re-run to green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/DB/Subscriptions_DB.py tldw_chatbook/Subscriptions/daily_reports_view.py Tests/Subscriptions/test_daily_reports_view.py
git commit -m "feat: cross-watchlist daily report read path ($TASK)"
```

---

### Task 3: Scheduled-briefing completion notifications

**Files:**
- Modify: `tldw_chatbook/Scheduling/scheduler/handlers/briefing_handler.py`
- Modify: `tldw_chatbook/app.py` (the `BriefingJobHandler(...)` construction, ~line 6869)
- Test: `Tests/Scheduling/test_briefing_handler.py` (append)

**Interfaces:**
- Consumes: `briefing_service` status constants (`STATUS_COMPLETE`, `STATUS_EMPTY`, `STATUS_FAILED` at `briefing_service.py:69-75`); `NotificationDispatchService.dispatch(*, app=None, category, title, message, severity="information", source_backend=None, source_entity_kind=None, source_entity_id=None, payload=None, timeout=None)`.
- Produces: `BriefingJobHandler.__init__(self, subscriptions_db, generate=generate_briefing, chachanotes_db_getter=None, dispatch_service: Any | None = None, notification_app_getter: Callable[[], Any | None] | None = None)` — fully backward-compatible (both new params optional, no dispatch when omitted).

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Scheduling/test_briefing_handler.py` (reuse the file's existing imports/fixtures style — it already imports `BriefingJobHandler`, `generate_briefing`, `SubscriptionsDB`; add `WatchlistBundleService` to its imports if absent):

```python
class _DispatchSpy:
    """Records dispatch kwargs; mirrors NotificationDispatchService.dispatch."""

    def __init__(self):
        self.calls: list[dict] = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return {"persisted": True}


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
    watchlist_id = int(
        WatchlistBundleService(db).create("Daily Brief")["id"]
    )
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
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])

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
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])
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
```

If `_canned_chat` / `functools` / `GenerationInFlightError` are not already imported in that file, add them to the imports (they are used by existing tests there — copy the import lines from the top of the file rather than redefining).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Scheduling/test_briefing_handler.py -v -k "dispatches or stays_silent"`
Expected: FAIL — `TypeError: BriefingJobHandler.__init__() got an unexpected keyword argument 'dispatch_service'`.

- [ ] **Step 3: Implement the handler changes**

In `briefing_handler.py`:

3a. Extend the status-constant import from `briefing_service` (currently only `STATUS_COMPLETE` is imported, ~line 26):

```python
from tldw_chatbook.Subscriptions.briefing_service import (
    STATUS_COMPLETE,
    STATUS_EMPTY,
    STATUS_FAILED,
    GenerationInFlightError,
    active_briefing_claims,
    generate_briefing,
)
```

3b. Extend `__init__` (currently at :87-92) — keep the existing docstring, add the params and attributes:

```python
    def __init__(
        self,
        subscriptions_db: Any,
        generate: Callable[..., Awaitable[dict[str, Any]]] = generate_briefing,
        chachanotes_db_getter: Callable[[], CharactersRAGDB | None] | None = None,
        dispatch_service: Any | None = None,
        notification_app_getter: Callable[[], Any | None] | None = None,
    ) -> None:
        """Initialize the handler.

        ``dispatch_service``/``notification_app_getter`` follow the same
        optional-collaborator discipline as ``chachanotes_db_getter``: absent
        means headless/tests and every notification path is a no-op. The app
        is a *getter* for the same late-binding reason ``chachanotes_db`` is.
        """
        self.subscriptions_db = subscriptions_db
        self._generate = generate
        self._chachanotes_db_getter = chachanotes_db_getter
        self.dispatch_service = dispatch_service
        self._notification_app_getter = notification_app_getter
        #: Strong references to spawned generation tasks ...
        self._pending_generations: set[asyncio.Task[Any]] = set()
```

3c. In `_run_generation` (:203-264), add notification calls on the two paths — inside the `except Exception` block after the `log_warning`, add `await self._notify_error(watchlist_id)`; and change the `else:` branch from `await self._auto_keep(result)` to:

```python
        else:
            await self._auto_keep(result)
            await self._notify_result(watchlist_id, result)
```

3d. Add the two notification methods after `_auto_keep` (before `__call__`):

```python
    async def _notify_result(self, watchlist_id: int, result: dict[str, Any]) -> None:
        """Dispatch one completion notification for a finished generation.

        No-op without a dispatch service; never raises (same containment rule
        as `_auto_keep` -- a notification failure must never surface as a
        scheduling failure).
        """
        if self.dispatch_service is None:
            return
        try:
            status = str(result.get("status") or "")
            if status not in (STATUS_COMPLETE, STATUS_EMPTY, STATUS_FAILED):
                return
            name = await asyncio.to_thread(self._watchlist_name, watchlist_id)
            briefing_id = result.get("id")
            if status == STATUS_COMPLETE:
                title = "Daily brief ready"
                message = f"{name} finished its scheduled brief."
                severity = "information"
            else:
                title = "Daily brief needs attention"
                error = str(result.get("error") or "").strip()
                message = (
                    f"{name} finished its scheduled brief with status "
                    f"'{status}'" + (f": {error}" if error else "") + "."
                )
                severity = "warning"
            app = (
                self._notification_app_getter()
                if self._notification_app_getter is not None
                else None
            )
            self.dispatch_service.dispatch(
                app=app,
                category="briefing",
                title=title,
                message=message,
                severity=severity,
                source_entity_kind="briefing",
                source_entity_id=(
                    str(briefing_id) if briefing_id is not None else None
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Briefing completion notification for watchlist "
                f"{watchlist_id} failed: {type(exc).__name__}"
            )

    async def _notify_error(self, watchlist_id: int) -> None:
        """Dispatch one attention notification for a crashed generation."""
        if self.dispatch_service is None:
            return
        try:
            name = await asyncio.to_thread(self._watchlist_name, watchlist_id)
            app = (
                self._notification_app_getter()
                if self._notification_app_getter is not None
                else None
            )
            self.dispatch_service.dispatch(
                app=app,
                category="briefing",
                title="Daily brief failed",
                message=(
                    f"{name}'s scheduled brief failed outside the briefing "
                    "service's own handling. See the Watchlists artifacts "
                    "pane for the failed row."
                ),
                severity="error",
                source_entity_kind="watchlist",
                source_entity_id=str(watchlist_id),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                f"Briefing error notification for watchlist {watchlist_id} "
                f"failed: {type(exc).__name__}"
            )

    def _watchlist_name(self, watchlist_id: int) -> str:
        """The watchlist's name, or a stable fallback (same read pattern as
        `_default_preset_id`)."""
        with self.subscriptions_db.transaction() as conn:
            row = conn.execute(
                "SELECT name FROM watchlists WHERE id = ?", (watchlist_id,)
            ).fetchone()
        if row is None:
            return f"Watchlist {watchlist_id}"
        return str(row["name"] or f"Watchlist {watchlist_id}")
```

3e. Wire in `app.py` — find the `BriefingJobHandler(` construction (~line 6869) and change it to:

```python
            briefing_handler = BriefingJobHandler(
                subscriptions_db=subscriptions_db,
                chachanotes_db_getter=lambda: getattr(self, "chachanotes_db", None),
                dispatch_service=self.notification_dispatch_service,
                notification_app_getter=lambda: self,
            )
```

(`self.notification_dispatch_service` is constructed earlier in the same method at ~:6790, so a direct reference is safe — unlike `chachanotes_db`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Scheduling/test_briefing_handler.py -v`
Expected: ALL PASS (existing + 4 new).

Also run the neighboring suite the change can reach: `python -m pytest Tests/Scheduling/test_scheduler_loop.py Tests/Scheduling/test_briefing_projection.py -v` — expected PASS (handler contract unchanged).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Scheduling/scheduler/handlers/briefing_handler.py tldw_chatbook/app.py Tests/Scheduling/test_briefing_handler.py
git commit -m "feat: briefing completion notifications via dispatch service ($TASK)"
```

---

### Task 4: Artifacts screen Reports slot

**Files:**
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Test: `Tests/UI/test_artifacts_screen_reports.py`

**Interfaces:**
- Consumes: `daily_reports_view.list_recent_reports(db, limit=20)` (Task 2); `play_audio_file` from `tldw_chatbook.TTS.audio_player`; `NavigateToScreen("watchlists_collections")`; app services via `getattr(self.app_instance, "subscriptions_db", None)` and (from Task 7) `daily_report_demo_service`.
- Produces: screen attributes `_daily_reports: list[dict]`, `_daily_reports_generation: int`; method `_start_daily_reports_refresh(self) -> None`; widget ids `artifacts-daily-report-demo` (Button), `artifacts-open-watchlists` (Button), `artifacts-report-row-{id}` (Static), `artifacts-report-play-{id}` (Button), `artifacts-list-reports` (Static), `artifacts-reports-more` (Static).

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_artifacts_screen_reports.py`:

```python
"""Artifacts screen Reports slot: empty-state CTA and seeded report rows.

Real app via `_build_test_app`, real `SubscriptionsDB`, real write paths for
seeding; no LLM/fetch involved (no briefing generation here, just rows).
"""

from contextlib import asynccontextmanager

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions.watchlist_bundle_service import WatchlistBundleService
from tldw_chatbook.UI.Screens.artifacts_screen import ArtifactsScreen

pytestmark = pytest.mark.ui


def _seed_report(app, *, status: str = "complete") -> int:
    db: SubscriptionsDB = app.subscriptions_db
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])
    briefing_id = db.insert_briefing(watchlist_id)
    db.update_briefing(
        briefing_id, status=status,
        body_markdown="## Daily Brief\n\nOne story [item 1].", item_count=1,
    )
    return briefing_id


@asynccontextmanager
async def _open_artifacts(app, *, size=(160, 50)):
    host = DestinationHarness(app, "artifacts")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.1)
        screen = host.screen_stack[-1]
        assert isinstance(screen, ArtifactsScreen)
        yield screen, pilot


@pytest.mark.asyncio
async def test_empty_state_offers_demo_cta():
    app = _build_test_app(configured_default="artifacts")
    async with _open_artifacts(app) as (screen, pilot):
        cta = screen.query_one("#artifacts-daily-report-demo", Button)
        assert cta.region.height >= 1, "CTA must paint, not just mount"
        assert screen.query_one("#artifacts-list-reports", Static)


@pytest.mark.asyncio
async def test_seeded_reports_list_rows_with_open_button():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        for _ in range(50):
            await pilot.pause(0.05)
            if screen._daily_reports:
                break
        assert screen._daily_reports, "refresh worker must land rows"
        row = screen.query_one(f"#artifacts-report-row-{briefing_id}", Static)
        assert row.region.height >= 1, "report row must paint"
        assert screen.query_one("#artifacts-open-watchlists", Button)
        # CTA belongs to the empty state only
        assert not screen.query("#artifacts-daily-report-demo")


@pytest.mark.asyncio
async def test_audio_row_shows_play_button():
    app = _build_test_app(configured_default="artifacts")
    briefing_id = _seed_report(app)
    db: SubscriptionsDB = app.subscriptions_db
    script_id = db.insert_briefing_script(
        briefing_id, preset_id=None, preset_name="Daily Brief",
        roster_snapshot_json="[]",
    )
    db.update_briefing_script(script_id, status="complete", turns_json="[]")
    audio_id = db.create_briefing_audio(script_id, voice_snapshot_json="[]")
    # A lexically-safe path: under the real briefing_audio_dir the guard passes
    # without touching disk (the play handler itself checks existence).
    from tldw_chatbook.Subscriptions.briefing_audio import briefing_audio_dir
    db.update_briefing_audio(
        audio_id, status="complete",
        file_path=str(briefing_audio_dir() / f"script-{script_id}-audio-{audio_id}.wav"),
        duration_seconds=1.0, turn_count=1,
    )
    async with _open_artifacts(app) as (screen, pilot):
        screen._start_daily_reports_refresh()
        for _ in range(50):
            await pilot.pause(0.05)
            if screen._daily_reports:
                break
        play = screen.query_one(f"#artifacts-report-play-{briefing_id}", Button)
        assert play.region.height >= 1
```

Note: writing a `file_path` string under the *real* `briefing_audio_dir()` writes nothing to disk — it is a lexical string in a test DB under `tmp_path`; only the prefix check runs. Do not create the file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/UI/test_artifacts_screen_reports.py -v`
Expected: FAIL — `TooManyMatches`/`NoMatches` on `#artifacts-daily-report-demo` (the screen still renders the old `Reports: none available` placeholder).

- [ ] **Step 3: Implement the screen changes**

In `artifacts_screen.py`:

3a. Imports (top of file, matching its existing relative-import style):

```python
from pathlib import Path

from loguru import logger

from ...Subscriptions.daily_reports_view import list_recent_reports
from ...TTS.audio_player import play_audio_file
```

(Skip `Path`/`logger` lines if already imported — check first.)

3b. `__init__` additions (after `self._chatbook_unmounted = True`, ~line 87):

```python
        self._daily_reports: list[dict[str, Any]] = []
        self._daily_reports_generation = 0
        self._daily_reports_worker: Worker[Any] | None = None
```

3c. In `on_mount` (:90-94) and `on_screen_resume` (:96-104), add `self._start_daily_reports_refresh()` next to the existing `self._start_chatbook_refresh()` call.

3d. Add the refresh machinery after `_start_chatbook_refresh` (~line 157):

```python
    def _start_daily_reports_refresh(self) -> None:
        """Re-read recent briefings off the UI thread, then repaint."""
        self._daily_reports_generation += 1
        self.refresh(recompose=True)
        self._daily_reports_worker = self._refresh_daily_reports(
            self._daily_reports_generation
        )

    @work(exclusive=True, thread=True)
    def _refresh_daily_reports(self, generation: int) -> None:
        db = getattr(self.app_instance, "subscriptions_db", None)
        reports: list[dict[str, Any]] = []
        if db is not None:
            try:
                reports = list_recent_reports(db, limit=20)
            except Exception:  # noqa: BLE001 - an Artifacts refresh must never crash the app
                reports = []
        self.app.call_from_thread(self._apply_daily_reports, generation, reports)

    def _apply_daily_reports(
        self, generation: int, reports: list[dict[str, Any]]
    ) -> None:
        if generation != self._daily_reports_generation:
            return  # a newer refresh superseded this one
        self._daily_reports = reports
        self.refresh(recompose=True)
```

3e. In `compose_content`, replace the single Reports placeholder Static:

```python
                    yield Static(
                        "  Reports: none available", id="artifacts-list-reports"
                    )
```

with:

```python
                    if self._daily_reports:
                        for report in self._daily_reports[:5]:
                            yield Static(
                                self._literal_text(f"> Report: {report['label']}"),
                                id=f"artifacts-report-row-{report['id']}",
                            )
                            if report.get("has_audio"):
                                yield Button(
                                    "Play",
                                    id=f"artifacts-report-play-{report['id']}",
                                    tooltip="Play this report's audio brief.",
                                )
                        if len(self._daily_reports) > 5:
                            yield Static(
                                self._literal_text(
                                    f"  + {len(self._daily_reports) - 5} more in Watchlists"
                                ),
                                id="artifacts-reports-more",
                            )
                        yield Button(
                            "Open Watchlists",
                            id="artifacts-open-watchlists",
                            tooltip="Read, play, keep, or export daily reports.",
                        )
                    else:
                        yield Static(
                            "  Reports: none yet", id="artifacts-list-reports"
                        )
                        yield Button(
                            "Create Your First Daily Report",
                            id="artifacts-daily-report-demo",
                            tooltip=(
                                "Seeds a 'Daily Brief' watchlist from live RSS, drafts "
                                "a text brief with your configured LLM provider, and "
                                "records audio when a TTS voice profile exists. Uses "
                                "live sources and your provider's API quota."
                            ),
                        )
```

3f. Add handlers after the existing `@on(Button.Pressed, ...)` block (~line 783):

```python
    @on(Button.Pressed, "#artifacts-open-watchlists")
    def open_watchlists(self) -> None:
        self.post_message(NavigateToScreen("watchlists_collections"))

    @on(Button.Pressed, "#artifacts-daily-report-demo")
    def start_daily_report_demo(self) -> None:
        service = getattr(self.app_instance, "daily_report_demo_service", None)
        if service is None:
            self.app_instance.notify(
                "The Daily Report demo is unavailable in this runtime.",
                severity="warning",
            )
            return
        self.run_worker(
            self._run_daily_report_demo(service),
            exclusive=True,
            group="artifacts-daily-report-demo",
        )

    async def _run_daily_report_demo(self, service: Any) -> None:
        """Worker body: run the demo, then refresh whatever landed."""
        try:
            outcome = await service.run_demo()
        except Exception:  # noqa: BLE001 - a worker crash exits the app
            logger.warning("Daily report demo failed unexpectedly")
            self.app_instance.notify(
                "The Daily Report demo failed unexpectedly.",
                severity="error",
            )
        else:
            if str(outcome.get("status")) == "complete":
                self.app_instance.notify(
                    "Your first daily report is ready — see Reports below."
                )
        finally:
            if self.is_attached:
                self._start_daily_reports_refresh()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Dynamic-id dispatch for per-report Play buttons.

        `@on` selectors cannot express the `artifacts-report-play-{id}` family,
        so prefix-match here; unrelated buttons fall through untouched.
        """
        button_id = event.button.id or ""
        if not button_id.startswith("artifacts-report-play-"):
            return
        event.stop()
        try:
            briefing_id = int(button_id.rsplit("-", 1)[-1])
        except ValueError:
            return
        report = next(
            (r for r in self._daily_reports if r.get("id") == briefing_id), None
        )
        if report is None or not report.get("has_audio"):
            return
        path = Path(str(report["audio_file_path"]))
        if not path.exists():
            self.app_instance.notify(
                "This audio file no longer exists on disk.", severity="warning"
            )
            return
        play_audio_file(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/UI/test_artifacts_screen_reports.py -v`
Expected: 3 PASS. Then neighbors: `python -m pytest Tests/UI/test_destination_shells.py -v` — expected PASS (destination chrome untouched).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/artifacts_screen.py Tests/UI/test_artifacts_screen_reports.py
git commit -m "feat: artifacts screen daily reports slot with demo CTA ($TASK)"
```

---

### Task 5: Demo service — preflight, seed, run-now text brief

**Files:**
- Create: `tldw_chatbook/Subscriptions/daily_report_demo.py`
- Test: `Tests/Subscriptions/test_daily_report_demo.py`

**Interfaces:**
- Consumes: `LocalWatchlistsService.resolve_or_create_watchlist(name) -> (dict, bool)` / `.create_source(payload)` (payload keys `source_type`, `url`, `name`) / `.add_source_to_watchlist(watchlist_id=, source_id=)` / `.launch_run(source_id=)` -> dict with `run_id` / `.execute_run(run_id)`; `SubscriptionsDB.insert_briefing_preset` / `set_watchlist_briefing_settings` / `list_briefing_schedules` / `insert_briefing`-family; `generate_briefing(db, watchlist_id, *, chat=..., preset_id=...)` + `STATUS_COMPLETE`/`STATUS_FAILED`/`GenerationInFlightError`; `briefing_cast.validate_roster` / `dump_roster`; `TTSProfileService.list_profiles() -> ProfileStoreResult[TTSProfilePage]` (`.value.profiles`, each with `.profile_id: UUID`); `NotificationDispatchService.dispatch`.
- Produces: `DailyReportDemoService(subscriptions_db, *, local_watchlists_getter, dispatch_service, app_getter, tts_service_getter=None, tts_profile_service_getter=None, chat=chat_api_call, synthesize=None)` with `async run_demo(self) -> dict` returning `{"status": "complete" | "briefing_failed" | "fetch_failed" | "unavailable" | "in_flight" | "error", "watchlist_id": int | None, "briefing_id": int | None, "audio": "complete" | "skipped" | "failed" | None, "reasons": list[str]}`. Tasks 6-7 rely on exactly these keys and the constants `DEMO_WATCHLIST_NAME = "Daily Brief"`, `DEMO_PRESET_NAME = "Daily Brief"`, `DEMO_CADENCE_SECONDS = 86400`, `DEMO_SOURCES` (3 RSS payloads).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Subscriptions/test_daily_report_demo.py`:

```python
"""DailyReportDemoService: real DBs and services, faked seams only.

Faked seams per test: the chat callable (DI, the service's own parameter) and
the HTTP fetch (monkeypatched `monitoring_engine.guarded_fetch_httpx_async`,
the convention `test_url_monitor_off_loop.py` set). Everything else -- watchlist
creation, subscription rows, run rows, item upserts, briefing lifecycle -- is
the real production path.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
from tldw_chatbook.Subscriptions import daily_report_demo
from tldw_chatbook.Subscriptions.daily_report_demo import (
    DEMO_CADENCE_SECONDS,
    DEMO_SOURCES,
    DEMO_WATCHLIST_NAME,
    DailyReportDemoService,
)
from tldw_chatbook.Subscriptions.local_watchlists_service import LocalWatchlistsService
from tldw_chatbook.TTS.profile_types import (
    ProfileStoreResult,
    TTSProfilePage,
)

pytestmark = pytest.mark.unit

_RSS = """<?xml version="1.0"?>
<rss version="2.0"><channel><title>Demo Feed</title>
<item><title>Demo story</title><link>https://example.com/1</link>
<description>Body of the demo story.</description>
<pubDate>Thu, 28 Aug 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


def _db(tmp_path) -> SubscriptionsDB:
    return SubscriptionsDB(tmp_path / "subs.db", "test")


class _FakeChat:
    """Stand-in for `Chat_Functions.chat_api_call` (the one faked seam)."""

    def __init__(self, *, error: Exception | None = None):
        self.reply = "## Daily Brief\n\nOne story [item 1].\n"
        self.error = error

    def __call__(self, **kwargs):
        if self.error is not None:
            raise self.error
        return self.reply


class _DispatchSpy:
    def __init__(self):
        self.calls: list[dict] = []

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        return {"persisted": True}


class _ProfileService:
    """Mirrors `TTSProfileService.list_profiles`'s real return shape."""

    def __init__(self, profiles=()):
        self._profiles = tuple(profiles)

    async def list_profiles(self, search=None, limit=50, offset=0):
        return ProfileStoreResult(
            generation=1,
            value=TTSProfilePage(profiles=self._profiles, total=len(self._profiles)),
        )


def _serve_rss(monkeypatch, *, fail_all: bool = False):
    async def fake_guarded(url, *, client, max_bytes, **kwargs):
        if fail_all:
            raise RuntimeError("network unreachable")
        return SimpleNamespace(
            status_code=200,
            headers={"content-type": "application/rss+xml"},
            text=_RSS,
            final_url=url,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(
        "tldw_chatbook.Subscriptions.monitoring_engine.guarded_fetch_httpx_async",
        fake_guarded,
    )


def _service(tmp_path, monkeypatch, *, chat=None, profiles=(), fail_fetch=False):
    db = _db(tmp_path)
    local = LocalWatchlistsService(db_factory=lambda: db)
    spy = _DispatchSpy()
    _serve_rss(monkeypatch, fail_all=fail_fetch)
    service = DailyReportDemoService(
        subscriptions_db=db,
        local_watchlists_getter=lambda: local,
        dispatch_service=spy,
        app_getter=lambda: None,
        tts_service_getter=lambda: None,
        tts_profile_service_getter=lambda: _ProfileService(profiles),
        chat=chat if chat is not None else _FakeChat(),
    )
    return service, db, spy


def _titles(spy):
    return [c["title"] for c in spy.calls]


@pytest.mark.asyncio
async def test_run_demo_seeds_watchlist_preset_schedule_and_briefs(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch)

    outcome = await service.run_demo()

    assert outcome["status"] == "complete"
    watchlist_id = outcome["watchlist_id"]
    assert watchlist_id is not None
    assert outcome["briefing_id"] is not None
    # Seeded setup: sources attached, preset bound, daily cadence set.
    schedules = db.list_briefing_schedules()
    assert len(schedules) == 1
    assert schedules[0]["watchlist_id"] == watchlist_id
    assert schedules[0]["briefing_cadence_seconds"] == DEMO_CADENCE_SECONDS
    with db.transaction() as conn:
        n_sources = conn.execute(
            "SELECT COUNT(*) AS n FROM watchlist_sources WHERE watchlist_id = ?",
            (watchlist_id,),
        ).fetchone()["n"]
    assert n_sources == len(DEMO_SOURCES)
    presets = db.list_briefing_presets()
    assert any(p["name"] == DEMO_WATCHLIST_NAME for p in presets)
    # The live fetch + generation actually ran through the real seams.
    briefing = db.get_briefing(outcome["briefing_id"])
    assert briefing["status"] == "complete"
    assert briefing["item_count"] >= 1
    # Stage trail + one completion dispatch, all under the briefing category.
    assert "Fetching today's stories" in _titles(spy)
    assert "Writing your brief" in _titles(spy)
    assert spy.calls[-1]["category"] == "briefing"


@pytest.mark.asyncio
async def test_run_demo_is_idempotent_when_a_schedule_exists(tmp_path, monkeypatch):
    service, db, _ = _service(tmp_path, monkeypatch)
    await service.run_demo()
    service2, db2, _ = _service(tmp_path, monkeypatch)  # same tmp DB file
    outcome = await service2.run_demo()

    assert outcome["status"] == "complete"
    with db2.transaction() as conn:
        n_watchlists = conn.execute("SELECT COUNT(*) AS n FROM watchlists").fetchone()["n"]
    assert n_watchlists == 1, "second run must reuse, not re-seed"
    assert len(db2.list_briefing_schedules()) == 1
    assert len(db2.list_briefings(outcome["watchlist_id"])) == 2  # ran again, once per demo


@pytest.mark.asyncio
async def test_run_demo_without_local_service_reports_unavailable(tmp_path, monkeypatch):
    db = _db(tmp_path)
    spy = _DispatchSpy()
    service = DailyReportDemoService(
        subscriptions_db=db,
        local_watchlists_getter=lambda: None,
        dispatch_service=spy,
        app_getter=lambda: None,
        chat=_FakeChat(),
    )
    outcome = await service.run_demo()
    assert outcome["status"] == "unavailable"
    with db.transaction() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM watchlists").fetchone()["n"] == 0


@pytest.mark.asyncio
async def test_run_demo_all_sources_failing_aborts_with_fetch_failed(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch, fail_fetch=True)
    outcome = await service.run_demo()
    assert outcome["status"] == "fetch_failed"
    assert db.list_briefings(outcome["watchlist_id"]) == [], "no briefing row on total fetch failure"


@pytest.mark.asyncio
async def test_failed_briefing_dispatches_provider_guidance(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch, chat=_FakeChat(error=RuntimeError("401 unauthorized")))
    outcome = await service.run_demo()
    assert outcome["status"] == "briefing_failed"
    last = spy.calls[-1]
    assert last["severity"] == "warning"
    assert "401 unauthorized" in last["message"]
    assert "Settings" in last["message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Subscriptions/test_daily_report_demo.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tldw_chatbook.Subscriptions.daily_report_demo'`.

- [ ] **Step 3: Implement the demo service (core; audio stage lands in Task 6)**

Create `tldw_chatbook/Subscriptions/daily_report_demo.py`:

```python
"""One-click Daily Report demo: seed a real watchlist, run it live.

The demo IS the product (ADR-079): everything it creates -- watchlist, RSS
sources, briefing preset, 24h cadence -- is real, persistent, user-owned
setup, and the run-now path is the same claim-guarded machinery the scheduler
uses tomorrow. Idempotency keys on configured briefing schedules
(`list_briefing_schedules`), never on names.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from loguru import logger

from tldw_chatbook.Subscriptions.briefing_cast import dump_roster, validate_roster
from tldw_chatbook.Subscriptions.briefing_service import (
    STATUS_COMPLETE,
    GenerationInFlightError,
    default_briefing_provider,
    generate_briefing,
)
from ..Chat.Chat_Functions import chat_api_call
from ..DB.Subscriptions_DB import SubscriptionsDB

DEMO_WATCHLIST_NAME = "Daily Brief"
DEMO_PRESET_NAME = "Daily Brief"
DEMO_CADENCE_SECONDS = 86_400
DEMO_STYLE_NOTES = (
    "A crisp daily news brief: top stories first, one short paragraph each, "
    "plain language, no filler."
)
#: Three stable RSS sources, all on the best-supported monitoring path.
#: Hacker News arrives via its official RSS feed (hnrss.org).
DEMO_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "name": "Hacker News (front page)",
        "source_type": "rss",
        "url": "https://hnrss.org/frontpage",
    },
    {
        "name": "BBC World News",
        "source_type": "rss",
        "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
    },
    {
        "name": "Ars Technica",
        "source_type": "rss",
        "url": "https://feeds.arstechnica.com/arstechnica/index",
    },
)
_AUDIO_SETTINGS_HINT = (
    "Audio skipped: add a TTS voice profile (Settings → Speech/TTS) and "
    "install the audio extra to hear tomorrow's brief."
)
_PROVIDER_GUIDANCE = (
    " Check your provider in Settings (F9) → API Keys, then run the demo again."
)


class DailyReportDemoService:
    """Seed-and-run orchestration for the Daily Report demo.

    All collaborators are injected: late-bound app services arrive as getters
    (the app wires this service before some of them exist), the chat callable
    and (Task 6) the synthesize callable are DI seams for tests.
    """

    def __init__(
        self,
        subscriptions_db: SubscriptionsDB,
        *,
        local_watchlists_getter: Callable[[], Any | None],
        dispatch_service: Any | None,
        app_getter: Callable[[], Any | None],
        tts_service_getter: Callable[[], Any | None] | None = None,
        tts_profile_service_getter: Callable[[], Any | None] | None = None,
        chat: Callable[..., Any] = chat_api_call,
        synthesize: Callable[..., Any] | None = None,
    ) -> None:
        self._db = subscriptions_db
        self._local_getter = local_watchlists_getter
        self._dispatch = dispatch_service
        self._app_getter = app_getter
        self._tts_getter = tts_service_getter or (lambda: None)
        self._tts_profiles_getter = tts_profile_service_getter or (lambda: None)
        self._chat = chat
        self._synthesize = synthesize

    async def run_demo(self) -> dict[str, Any]:
        """Run the whole demo; never raises (failures land in the outcome)."""
        outcome: dict[str, Any] = {
            "status": "error",
            "watchlist_id": None,
            "briefing_id": None,
            "audio": None,
            "reasons": [],
        }
        try:
            await self._run(outcome)
        except Exception as exc:  # noqa: BLE001 - the UI shows the outcome, not a traceback
            logger.warning(f"Daily report demo crashed: {type(exc).__name__}")
            outcome["reasons"].append(f"error:{type(exc).__name__}")
        return outcome

    async def _run(self, outcome: dict[str, Any]) -> None:
        local = self._local_getter()
        if local is None:
            outcome["status"] = "unavailable"
            outcome["reasons"].append("no-local-watchlists-service")
            return
        if not str(default_briefing_provider()).strip():
            outcome["status"] = "unavailable"
            outcome["reasons"].append("no-provider")
            return

        schedules = await asyncio.to_thread(self._db.list_briefing_schedules)
        if schedules:
            # Someone already has a daily report: reuse it, never re-seed.
            watchlist_id = int(schedules[0]["watchlist_id"])
            preset_id = await self._default_preset_id(watchlist_id)
            outcome["watchlist_id"] = watchlist_id
            outcome["reasons"].append("existing-schedule")
        else:
            watchlist_id, preset_id = await self._seed(local)
            outcome["watchlist_id"] = watchlist_id
            outcome["reasons"].append("seeded")

        await self._notify(
            "Fetching today's stories",
            "Checking your Daily Brief sources…",
        )
        fetched = await self._check_sources(local, watchlist_id, outcome)
        if fetched == 0:
            outcome["status"] = "fetch_failed"
            await self._notify(
                "Daily brief could not fetch sources",
                "None of the Daily Brief sources could be reached. Check your "
                "network and try again -- your schedule is saved and will "
                "retry on its own.",
                severity="warning",
            )
            return

        await self._notify(
            "Writing your brief", "Your LLM provider is drafting today's brief…"
        )
        try:
            row = await generate_briefing(
                self._db, watchlist_id, preset_id=preset_id, chat=self._chat
            )
        except GenerationInFlightError:
            outcome["status"] = "in_flight"
            await self._notify(
                "A briefing is already being written",
                "Nothing else was started; watch the Watchlists artifacts pane.",
                severity="warning",
            )
            return
        outcome["briefing_id"] = row.get("id")
        if str(row.get("status")) != STATUS_COMPLETE:
            outcome["status"] = "briefing_failed"
            await self._notify(
                "Daily brief failed to generate",
                "The LLM provider refused or failed"
                + (f": {row.get('error')}" if row.get("error") else "")
                + "." + _PROVIDER_GUIDANCE,
                severity="warning",
            )
            return

        # Task 6 fills this in (audio stage); until then the brief is the product.
        outcome["audio"] = "skipped"
        outcome["status"] = "complete"
        await self._notify(
            "Daily brief ready",
            "Your first daily report is ready -- see Artifacts → Reports. It "
            "refreshes daily from now on.",
        )

    # -- seeding ---------------------------------------------------------

    async def _seed(self, local: Any) -> tuple[int, int | None]:
        """Create the watchlist, sources, preset, and daily cadence."""
        watchlist, _created = await local.resolve_or_create_watchlist(
            DEMO_WATCHLIST_NAME
        )
        watchlist_id = int(watchlist["id"])
        for payload in DEMO_SOURCES:
            row = await local.create_source(dict(payload))
            await local.add_source_to_watchlist(
                watchlist_id=watchlist_id, source_id=int(row["id"])
            )
        roster, _audio_ready = await self._build_roster()
        preset_id = await asyncio.to_thread(
            self._db.insert_briefing_preset,
            DEMO_PRESET_NAME,
            roster_json=dump_roster(validate_roster(roster)),
            style_notes=DEMO_STYLE_NOTES,
        )
        await asyncio.to_thread(
            self._db.set_watchlist_briefing_settings,
            watchlist_id,
            selection_mode="auto_featured",
            default_preset_id=preset_id,
            briefing_cadence_seconds=DEMO_CADENCE_SECONDS,
        )
        return watchlist_id, preset_id

    async def _build_roster(self) -> tuple[list[dict[str, Any]], bool]:
        """Cast roster from the user's real voice profiles.

        `resolve_roster_voices` raises for speakers without a profile id --
        there is no default-voice fallback -- so zero profiles means a
        placeholder single-speaker roster and audio skipped (spec deviation 1).
        """
        profile_service = self._tts_profiles_getter()
        if profile_service is None:
            return [{"name": "Host", "voice_profile_id": None}], False
        try:
            result = await profile_service.list_profiles()
            profiles = list(result.value.profiles)
        except Exception:  # noqa: BLE001 - audio is optional; never fail the demo here
            return [{"name": "Host", "voice_profile_id": None}], False
        usable = [p for p in profiles if getattr(p, "profile_id", None) is not None]
        if not usable:
            return [{"name": "Host", "voice_profile_id": None}], False
        speakers = [
            {"name": "Host", "voice_profile_id": str(usable[0].profile_id)}
        ]
        if len(usable) > 1:
            speakers.append(
                {"name": "Analyst", "voice_profile_id": str(usable[1].profile_id)}
            )
        return speakers, True

    # -- run-now ---------------------------------------------------------

    async def _check_sources(
        self, local: Any, watchlist_id: int, outcome: dict[str, Any]
    ) -> int:
        """Run every source's check now; return how many produced items."""
        source_ids = await asyncio.to_thread(
            self._watchlist_source_ids, watchlist_id
        )
        for source_id in source_ids:
            launched = await local.launch_run(source_id=source_id)
            await local.execute_run(launched["run_id"])
        return await asyncio.to_thread(self._count_items, watchlist_id)

    def _watchlist_source_ids(self, watchlist_id: int) -> list[int]:
        with self._db.transaction() as conn:
            rows = conn.execute(
                "SELECT subscription_id FROM watchlist_sources "
                "WHERE watchlist_id = ? ORDER BY subscription_id",
                (watchlist_id,),
            ).fetchall()
        return [int(r["subscription_id"]) for r in rows]

    def _count_items(self, watchlist_id: int) -> int:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM subscription_items AS i "
                "JOIN watchlist_sources AS ws ON ws.subscription_id = i.subscription_id "
                "WHERE ws.watchlist_id = ?",
                (watchlist_id,),
            ).fetchone()
        return int(row["n"])

    def _default_preset_id(self, watchlist_id: int) -> int | None:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT default_briefing_preset_id FROM watchlists WHERE id = ?",
                (watchlist_id,),
            ).fetchone()
        return row["default_briefing_preset_id"] if row else None

    # -- notifications ----------------------------------------------------

    async def _notify(
        self, title: str, message: str, *, severity: str = "information"
    ) -> None:
        if self._dispatch is None:
            return
        try:
            self._dispatch.dispatch(
                app=self._app_getter(),
                category="briefing",
                title=title,
                message=message,
                severity=severity,
                source_entity_kind="daily_report_demo",
            )
        except Exception as exc:  # noqa: BLE001 - notifications must never fail the demo
            logger.warning(
                f"Demo stage notification failed: {type(exc).__name__}"
            )
```

Import note: check how sibling modules import `chat_api_call` (`briefing_service.py` does `from ..Chat.Chat_Functions import chat_api_call`) and copy that form exactly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Subscriptions/test_daily_report_demo.py -v`
Expected: 5 PASS. If `test_run_demo_is_idempotent_when_a_schedule_exists` fails because the second `_service(...)` opened a *new* `SubscriptionsDB` on the same file — that is fine and intended (file-backed DBs share the file); the assertion is on `db2`.

Mutation check: in `_run`, temporarily comment out the `if schedules:` reuse branch so the second run re-seeds — `test_run_demo_is_idempotent_when_a_schedule_exists` must FAIL (watchlist count 2). Revert.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/daily_report_demo.py Tests/Subscriptions/test_daily_report_demo.py
git commit -m "feat: daily report demo service - seed and live text brief ($TASK)"
```

---

### Task 6: Demo service — audio stage

**Files:**
- Modify: `tldw_chatbook/Subscriptions/daily_report_demo.py`
- Test: `Tests/Subscriptions/test_daily_report_demo.py` (append)

**Interfaces:**
- Consumes: `generate_script(db, briefing_id, *, preset_id, chat=...)` and `generate_script_audio(db, script_id, *, tts_service, profile_service, synthesize=...)` (both imported into the demo module's namespace so tests can monkeypatch them there); roster readiness from Task 5's `_build_roster`.
- Produces: `outcome["audio"]` becomes `"complete" | "failed"` on the audio paths (still `"skipped"` when not ready); stage notifications "Recording audio…" / audio-skip hint with `_AUDIO_SETTINGS_HINT`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Subscriptions/test_daily_report_demo.py`:

```python
@pytest.mark.asyncio
async def test_run_demo_skips_audio_without_voice_profiles(tmp_path, monkeypatch):
    service, db, spy = _service(tmp_path, monkeypatch, profiles=())
    outcome = await service.run_demo()
    assert outcome["status"] == "complete"
    assert outcome["audio"] == "skipped"
    assert db.list_briefing_scripts(outcome["briefing_id"]) == [], \
        "no cast script should be generated when it could not be voiced"
    assert any("Audio skipped" in t for t in _titles(spy))


@pytest.mark.asyncio
async def test_run_demo_generates_audio_when_ready(tmp_path, monkeypatch):
    from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

    def _profile(pid: uuid.UUID) -> TTSGenerationProfile:
        now = datetime.now(timezone.utc)
        return TTSGenerationProfile(
            profile_id=pid, display_name="Host voice", normalized_name="host voice",
            provider_id="openai", model_id="tts-1", voice_id="alloy",
            response_format="wav", speed=1.0, options={}, revision=1,
            created_at=now, updated_at=now,
        )

    profiles = (_profile(uuid.uuid4()),)
    service, db, spy = _service(tmp_path, monkeypatch, profiles=profiles)

    # Orchestration pin: the demo module's own seams, faked here because
    # `briefing_audio`/`briefing_cast` internals have their own suites.
    scripted: list[tuple[int, int]] = []

    async def _fake_generate_script(db_, briefing_id, *, preset_id, **kwargs):
        script_id = db_.insert_briefing_script(
            briefing_id, preset_id=preset_id, preset_name="Daily Brief",
            roster_snapshot_json="[]",
        )
        db_.update_briefing_script(script_id, status="complete", turns_json="[]")
        scripted.append((briefing_id, script_id))
        return db_.get_briefing_script(script_id)

    audio_calls: list[dict] = []

    async def _fake_generate_script_audio(db_, script_id, **kwargs):
        audio_calls.append({"script_id": script_id, **kwargs})
        return {"id": 1, "script_id": script_id, "status": "complete"}

    monkeypatch.setattr(daily_report_demo, "generate_script", _fake_generate_script)
    monkeypatch.setattr(daily_report_demo, "generate_script_audio", _fake_generate_script_audio)

    outcome = await service.run_demo()

    assert outcome["status"] == "complete"
    assert outcome["audio"] == "complete"
    assert scripted == [(outcome["briefing_id"], audio_calls[0]["script_id"])]
    assert "Recording audio" in _titles(spy)
    assert not any("Audio skipped" in t for t in _titles(spy))


@pytest.mark.asyncio
async def test_run_demo_audio_failure_degrades_to_text_success(tmp_path, monkeypatch):
    from tldw_chatbook.TTS.profile_types import TTSGenerationProfile

    def _profile(pid: uuid.UUID) -> TTSGenerationProfile:
        now = datetime.now(timezone.utc)
        return TTSGenerationProfile(
            profile_id=pid, display_name="Host voice", normalized_name="host voice",
            provider_id="openai", model_id="tts-1", voice_id="alloy",
            response_format="wav", speed=1.0, options={}, revision=1,
            created_at=now, updated_at=now,
        )

    service, db, spy = _service(
        tmp_path, monkeypatch, profiles=(_profile(uuid.uuid4()),)
    )

    async def _fake_generate_script(db_, briefing_id, *, preset_id, **kwargs):
        script_id = db_.insert_briefing_script(
            briefing_id, preset_id=preset_id, preset_name="Daily Brief",
            roster_snapshot_json="[]",
        )
        db_.update_briefing_script(script_id, status="complete", turns_json="[]")
        return db_.get_briefing_script(script_id)

    async def _failing_audio(db_, script_id, **kwargs):
        return {"id": 1, "script_id": script_id, "status": "failed",
                "error": "pydub is not installed"}

    monkeypatch.setattr(daily_report_demo, "generate_script", _fake_generate_script)
    monkeypatch.setattr(daily_report_demo, "generate_script_audio", _failing_audio)

    outcome = await service.run_demo()
    assert outcome["status"] == "complete", "audio failure never fails the demo"
    assert outcome["audio"] == "failed"
    assert any("Audio could not be synthesized" in t for t in _titles(spy))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Subscriptions/test_daily_report_demo.py -v -k audio`
Expected: FAIL — `AttributeError: module 'tldw_chatbook.Subscriptions.daily_report_demo' has no attribute 'generate_script'` (first two) and `outcome["audio"] == "skipped"` mismatch (third).

- [ ] **Step 3: Implement the audio stage**

3a. In `daily_report_demo.py` imports add:

```python
from tldw_chatbook.Subscriptions.briefing_audio import generate_script_audio
from tldw_chatbook.Subscriptions.briefing_cast import generate_script
```

3b. Thread roster readiness through seeding: in `_run`, capture the audio-ready flag. Change the seeding call site to keep it:

```python
        else:
            watchlist_id, preset_id, audio_ready = await self._seed(local)
```

and in the `if schedules:` branch set `audio_ready = await self._audio_ready_now()` (add a small method that re-runs `_build_roster` and returns the flag — the existing-schedule path may belong to a user who configured voices after seeding):

```python
    async def _audio_ready_now(self) -> bool:
        _roster, ready = await self._build_roster()
        return ready
```

Change `_seed`'s signature to `async def _seed(self, local: Any) -> tuple[int, int | None, bool]:`, capture `roster, audio_ready = await self._build_roster()`, and `return watchlist_id, preset_id, audio_ready`.

3c. Replace the success tail of `_run` (the block after the failed-briefing `return`) with:

```python
        if audio_ready:
            await self._notify(
                "Recording audio", "Synthesizing your audio brief…"
            )
            outcome["audio"] = await self._generate_audio(
                row, preset_id, outcome
            )
        else:
            outcome["audio"] = "skipped"
            await self._notify("Audio skipped", _AUDIO_SETTINGS_HINT)

        outcome["status"] = "complete"
        await self._notify(
            "Daily brief ready",
            "Your first daily report is ready -- see Artifacts → Reports. It "
            "refreshes daily from now on.",
        )
```

3d. Add the audio orchestrator (next to `_check_sources`):

```python
    async def _generate_audio(
        self,
        briefing_row: dict[str, Any],
        preset_id: int | None,
        outcome: dict[str, Any],
    ) -> str:
        """Cast + synthesize; any failure degrades to a text-only success."""
        briefing_id = int(briefing_row["id"])
        try:
            script = await generate_script(
                self._db,
                briefing_id,
                preset_id=int(preset_id) if preset_id is not None else 0,
                chat=self._chat,
            )
            if str(script.get("status")) != "complete":
                outcome["reasons"].append(
                    f"script:{script.get('status')}:{script.get('error')}"
                )
                await self._notify(
                    "Audio skipped", _AUDIO_SETTINGS_HINT, severity="warning"
                )
                return "skipped"
            kwargs: dict[str, Any] = {
                "tts_service": self._tts_getter(),
                "profile_service": self._tts_profiles_getter(),
            }
            if self._synthesize is not None:
                kwargs["synthesize"] = self._synthesize
            audio = await generate_script_audio(
                self._db, int(script["id"]), **kwargs
            )
            if str(audio.get("status")) != "complete":
                outcome["reasons"].append(
                    f"audio:{audio.get('status')}:{audio.get('error')}"
                )
                await self._notify(
                    "Audio could not be synthesized",
                    "Today's text brief is ready. "
                    + _AUDIO_SETTINGS_HINT,
                    severity="warning",
                )
                return "failed"
            return "complete"
        except Exception as exc:  # noqa: BLE001 - audio is optional, never fatal
            outcome["reasons"].append(f"audio-error:{type(exc).__name__}")
            await self._notify(
                "Audio could not be synthesized",
                "Today's text brief is ready. "
                + _AUDIO_SETTINGS_HINT,
                severity="warning",
            )
            return "failed"
```

(`preset_id=0` never happens in practice — `generate_script` requires a preset; if `preset_id` is `None` on the existing-schedule path the script stage records its own failure row and the degradation path runs.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Subscriptions/test_daily_report_demo.py -v`
Expected: 8 PASS (5 from Task 5 + 3 new). If `TTSGenerationProfile(options={}, ...)` is rejected by its validators, adjust only the `options`/`speed` literals until the REAL class accepts it — never substitute a dict for the profile (the fake must match the real seam).

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Subscriptions/daily_report_demo.py Tests/Subscriptions/test_daily_report_demo.py
git commit -m "feat: demo audio stage with graceful degradation ($TASK)"
```

---

### Task 7: App wiring, Artifacts CTA activation, Watchlists banner

**Files:**
- Modify: `tldw_chatbook/app.py` (`_wire_watchlists_and_notifications_services`, after the `SchedulerLoop(...)` construction ~line 6904)
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Modify: `tldw_chatbook/config.py` (`[scheduling]` template, ~line 3177)
- Test: `Tests/Watchlists/test_watchlists_demo_banner.py` (create), `Tests/UI/test_artifacts_screen_reports.py` (append)

**Interfaces:**
- Consumes: `DailyReportDemoService` (Tasks 5-6); `save_setting_to_cli_config` / `get_cli_setting`; app services `self.local_watchlists_service` (late-bound → getter), `self.notification_dispatch_service`, `self.tts_service`, `self._tts_profile_service`.
- Produces: `app.daily_report_demo_service` (consumed by Task 4's handlers); banner widget ids `watchlists-daily-report-banner`, `watchlists-daily-report-demo`, `watchlists-daily-report-banner-dismiss`; config key `scheduling.daily_report_demo_banner_dismissed` (default `false` in the TOML template).

- [ ] **Step 1: Write the failing tests**

Create `Tests/Watchlists/test_watchlists_demo_banner.py`:

```python
"""Watchlists demo banner: shown only while no briefing schedule exists.

Real app + real SubscriptionsDB via `_build_test_app`; visibility comes from
real `list_briefing_schedules` rows, dismissal from the real config file the
test conftest isolates.
"""

import os
import tomllib

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import DestinationHarness
from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
    WatchlistsCollectionsScreen,
)

pytestmark = pytest.mark.ui


def _seed_schedule(app) -> None:
    from tldw_chatbook.Subscriptions.watchlist_bundle_service import (
        WatchlistBundleService,
    )
    db = app.subscriptions_db
    watchlist_id = int(WatchlistBundleService(db).create("Daily Brief")["id"])
    db.set_watchlist_briefing_settings(
        watchlist_id, briefing_cadence_seconds=86_400
    )


async def _open(app, *, size=(180, 50)):
    host = DestinationHarness(app, "watchlists_collections")
    async with host.run_test(size=size) as pilot:
        await pilot.pause(0.2)
        screen = host.screen_stack[-1]
        assert isinstance(screen, WatchlistsCollectionsScreen)
        # Give the banner-resolution worker its turn(s).
        for _ in range(50):
            await pilot.pause(0.05)
            if screen.query("#watchlists-daily-report-banner"):
                break
        return screen, pilot


@pytest.mark.asyncio
async def test_banner_mounts_when_no_schedule_exists():
    app = _build_test_app(configured_default="watchlists_collections")
    screen, pilot = await _open(app)
    assert screen.query_one("#watchlists-daily-report-banner")


@pytest.mark.asyncio
async def test_banner_absent_when_a_schedule_exists():
    app = _build_test_app(configured_default="watchlists_collections")
    _seed_schedule(app)
    screen, pilot = await _open(app)
    for _ in range(20):
        await pilot.pause(0.05)
    assert not screen.query("#watchlists-daily-report-banner")


@pytest.mark.asyncio
async def test_dismiss_persists_and_removes_banner():
    app = _build_test_app(configured_default="watchlists_collections")
    screen, pilot = await _open(app)
    banner = screen.query_one("#watchlists-daily-report-banner")
    await screen.query_one("#watchlists-daily-report-banner-dismiss").press()
    for _ in range(20):
        await pilot.pause(0.05)
    assert not screen.query("#watchlists-daily-report-banner")
    config_path = os.environ["TLDW_CONFIG_PATH"]
    with open(config_path, "rb") as fh:
        data = tomllib.load(fh)
    assert data["scheduling"]["daily_report_demo_banner_dismissed"] is True
```

Append to `Tests/UI/test_artifacts_screen_reports.py`:

```python
@pytest.mark.asyncio
async def test_demo_cta_runs_the_wired_service():
    app = _build_test_app(configured_default="artifacts")

    class _StubDemo:
        def __init__(self):
            self.calls = 0

        async def run_demo(self):
            self.calls += 1
            return {"status": "complete"}

    stub = _StubDemo()
    app.daily_report_demo_service = stub
    async with _open_artifacts(app) as (screen, pilot):
        await screen.query_one("#artifacts-daily-report-demo").press()
        for _ in range(50):
            await pilot.pause(0.05)
            if stub.calls:
                break
        assert stub.calls == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest Tests/Watchlists/test_watchlists_demo_banner.py Tests/UI/test_artifacts_screen_reports.py -v`
Expected: banner tests FAIL (`NoMatches: #watchlists-daily-report-banner`); CTA test FAILS (`AttributeError: no attribute 'daily_report_demo_service'` → stub never called because the screen's handler hit the `service is None` notify path).

- [ ] **Step 3: Implement**

3a. `app.py` — add to the imports near the other Scheduling/Subscriptions imports (~line 396-409):

```python
from .Subscriptions.daily_report_demo import DailyReportDemoService
```

Then in `_wire_watchlists_and_notifications_services`, immediately after the `self.scheduler_loop = SchedulerLoop(...)` block (~line 6904), add:

```python
        # The demo reuses the single subscriptions_db (task-15463) and the
        # notification service above; every late-bound app service arrives as
        # a getter because several are constructed after this method runs.
        self.daily_report_demo_service = DailyReportDemoService(
            subscriptions_db=subscriptions_db,
            local_watchlists_getter=lambda: getattr(
                self, "local_watchlists_service", None
            ),
            dispatch_service=self.notification_dispatch_service,
            app_getter=lambda: self,
            tts_service_getter=lambda: getattr(self, "tts_service", None),
            tts_profile_service_getter=lambda: getattr(
                self, "_tts_profile_service", None
            ),
        )
```

3b. `config.py` — in the `[scheduling]` template (~line 3177, after `briefing_schedules_enabled = true`), add:

```python
# Dismiss the Watchlists "daily brief demo" banner permanently.
daily_report_demo_banner_dismissed = false
```

3c. `watchlists_collections_screen.py`:

- Extend the config import (the screen already imports `get_cli_setting`; find that import line and add `save_setting_to_cli_config` beside it).
- Add the banner resolver + handlers. Place the resolver near the other workers; handlers near the other `@on(Button.Pressed, ...)` handlers:

```python
    @work(exclusive=True)
    async def _resolve_daily_report_banner(self) -> None:
        """Mount the demo banner only when it can teach something.

        No compose-time DB reads (the screen's predicates must stay
        side-effect free): the banner mounts after this worker resolves
        dismissal + schedules.
        """
        if bool(
            get_cli_setting(
                "scheduling", "daily_report_demo_banner_dismissed", False
            )
        ):
            return
        db = self._briefings_db()
        if db is None:
            return
        try:
            schedules = await asyncio.to_thread(db.list_briefing_schedules)
        except Exception:  # noqa: BLE001 - a banner must never break the screen
            return
        if schedules:
            return
        banner = Horizontal(
            id="watchlists-daily-report-banner",
            classes="destination-filter-strip",
        )
        await banner.mount(
            Static(
                "Turn your watchlists into a daily brief — text and audio.",
                id="watchlists-daily-report-banner-text",
            ),
            Button("Try the demo", id="watchlists-daily-report-demo"),
            Button("Dismiss", id="watchlists-daily-report-banner-dismiss"),
        )
        await self.mount(banner, before="#watchlists-header-bar")

    @on(Button.Pressed, "#watchlists-daily-report-banner-dismiss")
    def dismiss_daily_report_banner(self, event: Button.Pressed) -> None:
        event.stop()
        save_setting_to_cli_config(
            "scheduling", "daily_report_demo_banner_dismissed", True
        )
        self.query_one("#watchlists-daily-report-banner").remove()

    @on(Button.Pressed, "#watchlists-daily-report-demo")
    def start_daily_report_demo(self, event: Button.Pressed) -> None:
        event.stop()
        service = getattr(self.app_instance, "daily_report_demo_service", None)
        if service is None:
            self._notify_watchlists(
                "The Daily Report demo is unavailable in this runtime.",
                severity="warning",
                markup=False,
            )
            return
        self.run_worker(
            self._run_daily_report_demo(service),
            exclusive=True,
            group="wl-daily-report-demo",
        )

    async def _run_daily_report_demo(self, service: Any) -> None:
        """Worker body: run the demo, report, take the banner down."""
        try:
            outcome = await service.run_demo()
        except Exception:  # noqa: BLE001 - a worker crash exits the app
            logger.warning("Daily report demo failed unexpectedly (banner)")
            self._notify_watchlists(
                "The Daily Report demo failed unexpectedly.",
                severity="error",
                markup=False,
            )
            return
        if str(outcome.get("status")) == "complete":
            self._notify_watchlists(
                "Your first daily report is ready — see Artifacts → Reports.",
                markup=False,
            )
        banner = self.query("#watchlists-daily-report-banner")
        if banner:
            banner.first().remove()
```

- Start the resolver from `on_mount` (find the screen's existing `on_mount` and append):

```python
        self._resolve_daily_report_banner()
```

(`@work` methods are invoked as workers by calling them — same idiom the artifacts screen uses for `_refresh_chatbook_context`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest Tests/Watchlists/test_watchlists_demo_banner.py Tests/UI/test_artifacts_screen_reports.py -v`
Expected: all PASS (3 banner + 4 artifacts). Then the reachable neighbors: `python -m pytest Tests/Watchlists/test_no_side_effecting_predicates.py Tests/UI/test_destination_shells.py -v` — expected PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/app.py tldw_chatbook/config.py tldw_chatbook/UI/Screens/watchlists_collections_screen.py Tests/Watchlists/test_watchlists_demo_banner.py Tests/UI/test_artifacts_screen_reports.py
git commit -m "feat: wire daily report demo, artifacts CTA, watchlists banner ($TASK)"
```

---

### Task 8: Live verification and task hygiene

**Files:**
- No code files. Updates: the backlog task (`$TASK`), possibly `backlog/docs/lessons-*.md`.

**Interfaces:**
- Consumes: everything above, running for real.
- Produces: live-evidence notes in the task's Implementation Notes; task status Done.

- [ ] **Step 1: Isolate a scratch profile (never the real one)**

This plan bumps no schema (Task 2 adds a method, not a migration) — but the demo *writes real rows and makes real API calls*, so it must run against a scratch profile regardless:

```bash
VERIFY=/tmp/daily-reports-verify && rm -rf "$VERIFY" && mkdir -p "$VERIFY/data"
cat > "$VERIFY/config.toml" <<'EOF'
[paths]
data_dir = "/tmp/daily-reports-verify/data"
EOF
```

Then export a provider key into the scratch environment (the config file is the only thing `TLDW_CONFIG_PATH` redirects — put the key in that TOML's `[API]` section rather than the environment, following the config's own key naming).

- [ ] **Step 2: Run the app headless and drive the demo at the user's surface**

Per `backlog/docs/lessons-live-verification.md` — verify at the surface the user touches, stderr stays attached:

```bash
TLDW_CONFIG_PATH="$VERIFY/config.toml" tmux -L verify new-session -d -x 235 -y 52 '.venv/bin/python -m tldw_chatbook.app'
tmux -L verify capture-pane -p | tail -40   # navigate to Artifacts; confirm empty-state CTA paints
```

Drive it: press the **Create Your First Daily Report** button, then watch notifications and the Reports slot fill. Confirm:
1. Stage notifications appear (fetching → writing → recording or audio-skipped).
2. The Reports slot lists the new brief row; **Open Watchlists** jumps to the watchlist artifacts pane; **Play** works when audio was synthesized (needs a voice profile + pydub in the venv — `pip install -e ".[audio]"` in the scratch run only).
3. The Watchlists banner is gone (a schedule now exists).
4. `sqlite3 "$VERIFY/data/..."` — one `Daily Brief` watchlist, three RSS subscriptions, one complete briefing row; `list_briefing_schedules` cadence 86400.
5. Kill the app, relaunch with the same profile: the Reports slot still lists the brief (persistence across remount, not just first paint — lessons-live-verification #5).

Capture the pane at each step (`tmux -L verify capture-pane -p > "$VERIFY/step-N.txt"`) — these are the evidence artifacts.

```bash
tmux -L verify kill-session
```

- [ ] **Step 3: Finish the backlog task**

```bash
backlog task edit $TASK_ID --notes "Implemented per Docs/superpowers/plans/2026-08-29-daily-reports-demo.md; spec Docs/superpowers/specs/2026-08-29-daily-reports-demo-design.md; ADR-079. Live verification transcript in /tmp/daily-reports-verify (pane captures). All AC checked."
backlog task edit $TASK_ID -s Done
```

Manually check every AC checkbox in the task file (`- [ ]` → `- [x]`), add the `## Implementation Notes` section (approach, modified files, the two spec deviations and why), and verify the ADR-079 link is present.

- [ ] **Step 4: Lessons check**

Most tasks produce nothing here. Two candidates — record only what the implementation actually cost:
- If the `TTSGenerationProfile` construction in Task 6's tests needed non-obvious literals, that's a "read the validator before writing the double" data point for `lessons-testing-evidence.md`.
- If the live run surfaced anything the pilot tests couldn't (paint vs. mount, notification timing), that's a `lessons-live-verification.md` entry. State the incident, not the rule.

- [ ] **Step 5: Final commit**

```bash
git add backlog/tasks/ backlog/docs/ 2>/dev/null || true
git commit -m "docs: close out daily reports task with live evidence ($TASK)" || echo "nothing to commit"
```

---

## Self-Review (completed during planning)

- **Spec coverage**: view §1 → Task 2; Artifacts slot §2 → Tasks 4+7; demo §3 → Tasks 5+6; banner §4 → Task 7; notifications §5 → Task 3; error table → Tasks 5/6 tests (unavailable/fetch_failed/briefing_failed/in-flight/audio-degraded); testing §→ per-task TDD + Task 8 live run; governance → Task 1 ADR-079; phasing ①=Tasks 2-4, ②=Tasks 5-7, ③=follow-up plan (documented deviation 4). Spec Follow-ups stay out of scope.
- **Placeholders**: none — every code step carries real code; the two `$TASK`/`$TASK_ID` tokens are CLI-printed ids with explicit instructions.
- **Type consistency**: `list_recent_briefings`/`list_recent_reports` keys match between Tasks 2 and 4; `DailyReportDemoService` ctor and `run_demo` outcome keys are identical in Tasks 5, 6, 7; handler ids (`artifacts-daily-report-demo`, `artifacts-report-play-{id}`, `watchlists-daily-report-banner*`) consistent across tasks and tests; category string `briefing` used everywhere.
