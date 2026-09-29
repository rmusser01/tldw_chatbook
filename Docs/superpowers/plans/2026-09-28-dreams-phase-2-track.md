# Dreams Phase 2 (Track) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Dreams Track loop — "Track this" on a story becomes either a watched page (existing subscription/URLMonitor/alert machinery) or a recurring question (search + LLM change-judgment with run rows), surfaced as tracked updates with reminder promotion and self-retiring lifecycle.

**Architecture:** Phase 2 adds a `track_service.py` to the existing `Dreams/` package plus two tables in `DreamsDB` (schema v2). The page mechanism creates rows through `LocalWatchlistsService`'s real APIs (subscriptions ARE the scheduling registration — `WatchlistProjection` fabricates `watchlist:<id>` tasks from subscription rows; no stored task rows exist); the question mechanism reuses the cycle's search/chat seams under a new `dream_track_check` task type emitted by the existing `DreamsProjection`. All app wiring goes through the existing post-`_ui_ready` `_wire_dreams_scheduler_integration` seam (ADR-097).

**Tech Stack:** Python ≥3.12 (stdlib + existing deps only), SQLite via `BaseDB`, Textual 8.x, pytest.

**Spec:** `Docs/superpowers/specs/2026-09-22-dreams-daily-discovery-design.md` (§The track loop, §Data model, §Phasing tasks 9–13) + `backlog/decisions/196-dreams-daily-discovery-and-tracking.md`. ADR-196 already covers Phase 2's decisions (attach-not-duplicate, budgets shared, goals immune, disposition vocabulary) — no new ADR needed.

**Worktree:** `.worktrees/feat-dreams-phase-2` (branch `feat/dreams-phase-2` off `origin/dev` @ `e5ac111967`). Tests: `../../.venv/bin/python -m pytest` from worktree root.

## Global Constraints

- Dreams remains off by default; tracked-item machinery may run only when `[dreams] enabled` is true (projection gate covers track tasks too).
- Parameterized SQL only; all timestamps UTC ISO-8601 (`_utc_now_iso` idiom); blocking calls (`perform_websearch`, `chat_api_call`, every SQLite access from async code) under `asyncio.to_thread`, one hop per stage.
- **Privacy (spec §Privacy, binding):** goal text enters an outbound query or prompt ONLY when the goal's `searchable` flag is 1. Unsearchable goals contribute zero outbound text. Every test that builds an outbound payload asserts this.
- **Budgets are shared with the cycle** (spec §budgets): every track check bumps `dream_daily_usage` (`searches`/`llm_calls`) and respects `max_searches_per_day`/`max_llm_calls_per_day`; over-budget checks record a `skipped` run, never spend.
- **Disposition vocabulary shared with watchlists** (spec): track runs use exactly `changed / unchanged / baseline / rebaselined / withheld / error / skipped` (the `local_watchlists_runs` vocabulary).
- **Attach, not duplicate** (ADR-196 contract 4): page-tracking reuses an existing subscription for the same source URL; untrack retires the wrapper and disables/removes the subscription ONLY if Dreams created it (`created_by_dreams`).
- **`tracked` feedback flips to positive** this phase (spec carry note): `_feedback_net`'s +1 set becomes `("more", "kept", "dived", "ingested", "tracked")` with the module comment updated.
- ADR-097 boot census: NO new module-scope Dreams/Scheduling-track imports in `app.py` — all wiring extends the existing `_wire_dreams_scheduler_integration` post-`_ui_ready` seam.
- UI: ADR-150 `$ds-*` classes only (reuse the existing dream-row idiom — bare Statics, `artifacts-dream-track-row-{id}` ids); ADR-031 single-letter bindings; footer hints only implemented actions; new app-mounting tests carry `@pytest.mark.bootstrap_profile`.
- Targeted test runs only; pristine output; one commit per task, explicit paths only — never `git add .` / `-A`; `feat(dreams-track):`/`test(dreams-track):` prefixes.
- Baseline gate before every commit: `../../.venv/bin/python -m pytest Tests/Dreams/ Tests/UI/test_artifacts_dreams_rows.py Tests/UI/test_artifacts_dreams_modal.py -q` green (plus the current task's new files).

## Verified seam reference (recon, 2026-09-28, dev @ e5ac111967)

Function-anchor references (grep to pin lines; signatures verbatim from recon):

- `LocalWatchlistsService` (`Subscriptions/local_watchlists_service.py`): `async find_source_id_by_url(url) -> int | None`; `async create_source(payload: dict) -> dict` (payload keys `name`, `type` (`"site"→"url"` normalization; url/rss/atom/json_feed/url_list/podcast/sitemap/api), `source`, `check_frequency`, `is_active`; result carries `creation_outcome` "created"/"existing" + the subscription id); `async resolve_or_create_watchlist(name) -> tuple[dict, bool]`; `async add_source_to_watchlist(*, watchlist_id, source_id)` (INSERT OR IGNORE); `async create_alert_rule(*, name, condition_type, condition_value=None, job_id=None, source_id=None, severity="warning") -> dict`; `update_alert_rule(rule_id, **fields)`; `delete_alert_rule(rule_id)`. Alert condition vocabulary: `no_items / error_rate_above / items_below / items_above / run_failed`; severities `info/warning/critical`. Fired alerts → `NotificationDispatchService.dispatch` → durable inbox row (Watchlists Notifications pane displays it); dedupe key `watchlist-alert:{rule_id}:{run_id}`.
- `WatchlistProjection` (`Scheduling/services/watchlist_projection.py`): fabricates `ScheduledTask(id=f"watchlist:{subscription_id}", type="watchlist_job", next_run_at=(last_checked or created_at) + check_frequency)` from subscription rows — the registration seam the Phase 1 plan called an open item. **Creating a subscription row IS the scheduling registration.**
- `ScheduledTasksDB.create_reminder_task(owner_id, title, **kwargs) -> str` (`Scheduling/db/scheduled_tasks_db.py`); columns include `body, schedule_kind ("one_time"/"recurring"), run_at, next_run_at, link_type, link_id`; `ReminderHandler` dispatches via `NotificationDispatchService.dispatch(category="reminder", ...)`.
- `NotificationDispatchService.dispatch(*, app=None, category, title, message, severity="information", source_entity_kind=None, source_entity_id=None, payload=None)`.
- `DreamsProjection` (`Scheduling/services/dreams_projection.py`): `tasks()` currently emits one `dreams:cycle` / `dreams_cycle` task; exposes `DREAMS_TASK_PREFIX` + `parse_dreams_task_id` + `list_jobs(owner_id, now)`; `DreamsCycleHandler(deps_getter)` with `shutdown(timeout)` seam. The queue feeds via the existing `dreams_projection` named parameter — extending `tasks()` output needs NO queue/app changes.
- `Dreams/cycle_service.py`: `CycleDeps` fields `dreams_db, chachanotes_db_getter, media_db_getter, subs_db_getter, pc_service_getter, chat_getter, perform_search, now`; `_feedback_net` (sync, `+1 if kind in ("more","kept","dived","ingested")`, `-1 if kind == "less"`, else 0 — `tracked` currently 0); `run_cycle` stage 0 already calls stale-reclaim + prune; append mode + claims per R11.
- `Dreams/interest_profile.py`: `snapshot(db, *, now_epoch, feedback=None)` returns `{"topics": [...], "goals": [...]}` — goals pass through untouched (no decay, no feedback; `searchable` rides on the profile row). `FEEDBACK_STEP=0.1`, floor 0.05, ceiling 1.0.
- `Dreams/query_synthesis.py`: `async synthesize_queries(chat, *, snapshot, count, exploration_slots)` (payload today is topics+region only); `preview_queries(topics, count)` public wrapper (modal is its only caller).
- `DB/Dreams_DB.py`: `_CURRENT_SCHEMA_VERSION = 1`, inline `_SCHEMA_DDL` tuple, `user_version` pragma gate; profile methods `upsert_profile_entry / list_profile / delete_profile_entry / record_feedback / seen_* / prune_seen / usage_*`. Track tables absent (docstring says they arrive this phase).
- `UI/Screens/artifacts_screen.py`: Dreams group composes after Reports inside `#artifacts-list-pane` (bare Statics, `artifacts-dream-row-*`); refresh trio `_start_dreams_refresh/_refresh_dreams/_apply_dreams` + `_dreams_enabled()`. `artifacts_dreams_modal.py`: bindings `k/d/e/m/l/q` + `i` ingest (http-only gate) + hidden escape; ctor `(story, *, dreams_db_getter, capture_backend_getter, on_changed)`; write shape = direct single-row SQLite + `on_changed()` + dismissible notice.
- Test conventions to copy: `Tests/Subscriptions/test_local_watchlists_service.py` (real `SubscriptionsDB(tmp_path)` + `LocalWatchlistsService` + `ClientNotificationsDB` + `NotificationDispatchService`; alert assertions pin rule_id + one notification row + dedupe key), `Tests/Dreams/test_dreams_scheduler.py` (projection/handler contracts), `Tests/UI/test_artifacts_dreams_{rows,modal}.py` (harness + `_wait_for_dreams` settle helper).

---

### Task 1: Goals enter query synthesis + per-goal searchable toggle + region-labeled preview

**Files:**
- Modify: `tldw_chatbook/Dreams/query_synthesis.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_dreams_modal.py`
- Create: `tldw_chatbook/UI/Screens/artifacts_dreams_goals_modal.py`
- Test: `Tests/Dreams/test_query_synthesis.py` (extend), `Tests/UI/test_artifacts_dreams_goals_modal.py` (new)

**Interfaces:**
- Consumes: `interest_profile.snapshot()` (returns `goals` rows carrying `text`, `searchable`, `source`, `query_angle`); `DreamsDB.upsert_profile_entry(facet, text, *, weight, searchable, source)` / `delete_profile_entry(facet, text)` / `list_profile()`.
- Produces (later tasks + this task's consumers):
  - `query_synthesis.synthesize_queries(chat, *, snapshot, count, exploration_slots)` — unchanged signature, new behavior: the user payload gains `"goals": [g["text"] for g in snapshot["goals"] if g.get("searchable", 1)]` and the system prompt instructs that goal text drives event/deal/social-opportunity query angles while topics drive content angles. **Unsearchable goals are filtered BEFORE payload construction — their text must not appear anywhere in the payload.**
  - `query_synthesis.preview_queries(topics, goals, *, count)` — signature CHANGE (was `(topics, count)`); returns `list[dict]` entries `{"query": str, "goal_derived": bool}` so the preview can label goal-derived lines; the deterministic fallback derives at most one line per searchable goal (`f"{goal} events and tickets"` style) plus the existing topic lines.
  - `DreamsGoalsModal(ModalScreen[None])` — ctor `(dreams_db_getter, on_changed)`; bindings `a` add goal, `x` remove selected, `s` toggle searchable, `q` close (ADR-031; footer advertises exactly these); renders the goals list with `· searchable`/`· private` markers and the region line from `dreams_setting("region")` labeled `Region (used in queries):`.
  - Story modal gains binding `g` → pushes `DreamsGoalsModal`; the preview section renders `preview_queries(...)` rows with `(goal-derived)` suffix where `goal_derived` is true.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Dreams/test_query_synthesis.py — add
import pytest

from tldw_chatbook.Dreams.query_synthesis import preview_queries, synthesize_queries

SNAP_WITH_GOALS = {
    "topics": [{"facet": "topic", "text": "rust tui", "weight": 0.9}],
    "goals": [
        {"facet": "goal", "text": "see Wednesday 13 live", "searchable": 1},
        {"facet": "goal", "text": "private wish", "searchable": 0},
    ],
    "region": "Seattle",
}


@pytest.mark.asyncio
async def test_synthesis_payload_carries_searchable_goals_and_never_private_ones():
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "rust tui news\nconcerts Seattle"}}]}

    await synthesize_queries(chat, snapshot=SNAP_WITH_GOALS, count=3, exploration_slots=1)
    payload = calls[0]["messages_payload"][0]["content"]
    assert "see Wednesday 13 live" in payload
    assert "private wish" not in payload  # searchable=0 never leaves the machine


def test_preview_queries_labels_goal_derived_lines_and_skips_private_goals():
    rows = preview_queries(["rust tui"], SNAP_WITH_GOALS["goals"], count=4)
    queries = [r["query"] for r in rows]
    assert any(r["goal_derived"] for r in rows)
    assert not any("private wish" in q for q in queries)
    assert any("rust tui" in q for q in queries)
```

`Tests/UI/test_artifacts_dreams_goals_modal.py` (new, harness copied from `test_artifacts_dreams_modal.py`'s bare-App pattern; no `bootstrap_profile` needed for bare-App tests, but add it if you mount the full harness): seed a real `DreamsDB` with one goal row (`upsert_profile_entry("goal", "visit Japan", weight=1.0, searchable=1, source="user")`); mount `DreamsGoalsModal`; assert the goal renders with `· searchable`; press `s` → `list_profile()` shows `searchable=0` and the row text flips to `· private`; press `a` with an input of "new goal" → row exists with `source='user'`; press `x` on the selected row → row gone; footer hint string contains exactly the four actions.

- [ ] **Step 2: Run tests to verify they fail** — `../../.venv/bin/python -m pytest Tests/Dreams/test_query_synthesis.py Tests/UI/test_artifacts_dreams_goals_modal.py -v` → FAIL (preview signature, payload goals, modal missing).

- [ ] **Step 3: Implement.** In `query_synthesis.py`: filter searchable goals at the top of `synthesize_queries` and extend the user payload dict; extend `SYSTEM_PROMPT` with the goal-angle instruction sentence; rewrite `preview_queries(topics, goals, *, count)` returning labeled dicts (topic lines `goal_derived=False`, goal lines `f"{goal} events and tickets"` `goal_derived=True`, plus the existing exploration line); update the fallback query builder to accept goals the same way. New `artifacts_dreams_goals_modal.py` mirroring the story modal's structure/DEFAULT_CSS theme-variable pattern (ADR-150: no new tokens), with an `Input` for add and a simple list rendering; goal CRUD via `upsert_profile_entry`/`delete_profile_entry` (weight for goals is nominal `1.0`); `on_changed()` after each mutation. In `artifacts_dreams_modal.py`: add `("g", "goals", "Goals & privacy")` binding + `action_goals` pushing the goals modal; render the preview via the new `preview_queries` shape with `(goal-derived)` labels; update the modal's preview test expectations.

- [ ] **Step 4: Green + gate.** `../../.venv/bin/python -m pytest Tests/Dreams/ Tests/UI/test_artifacts_dreams_rows.py Tests/UI/test_artifacts_dreams_modal.py Tests/UI/test_artifacts_dreams_goals_modal.py -q` — all pass (existing preview/modal tests updated in Step 3).

- [ ] **Step 5: Commit** — `git add` the four files explicitly; `feat(dreams-track): goals drive event/deal query angles with per-goal searchable gate`.

---

### Task 2: DreamsDB schema v2 — track tables + CRUD

**Files:**
- Modify: `tldw_chatbook/DB/Dreams_DB.py`
- Test: `Tests/Dreams/test_dreams_db_track.py` (new)

**Interfaces:**
- Consumes: the existing `_SCHEMA_DDL`/`_initialize_schema` inline-DDL pattern and `_utc_now_iso`.
- Produces (Tasks 3–6 build on these exactly):
  - `create_tracked_item(self, *, origin_story_id=None, mechanism, intent, subscription_id=None, query_template=None, event_date=None, cadence_seconds, created_by_dreams=0) -> int` (CHECKs: mechanism ∈ page/question, intent ∈ event/deal/topic)
  - `get_tracked_item(self, tracked_item_id) -> dict | None`; `list_tracked_items(self, status="active") -> list[dict]` (ordered `created_at DESC`); `find_tracked_by_story(self, origin_story_id) -> dict | None`
  - `set_tracked_status(self, tracked_item_id, status, *, retired_reason=None)` (status ∈ active/paused/retired); `touch_tracked_checked(self, tracked_item_id, now_iso)`
  - `count_active_tracked(self) -> int`
  - `insert_track_run(self, tracked_item_id, *, status, digest_hash, verdict_note="", notified=0) -> int` (status ∈ changed/unchanged/baseline/rebaselined/withheld/error/skipped)
  - `list_recent_track_runs(self, tracked_item_id, limit=5) -> list[dict]` (newest first); `consecutive_track_dispositions(self, tracked_item_id, status) -> int`
  - `_CURRENT_SCHEMA_VERSION = 2` with the two new tables appended to `_SCHEMA_DDL` (additive `CREATE TABLE IF NOT EXISTS`, so v1 files upgrade in place; bump the `user_version` write path).

Schema (verbatim columns per spec §Data model):

```sql
CREATE TABLE IF NOT EXISTS dream_tracked_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin_story_id INTEGER,
    mechanism TEXT NOT NULL CHECK(mechanism IN ('page','question')),
    intent TEXT NOT NULL CHECK(intent IN ('event','deal','topic')),
    subscription_id INTEGER,
    query_template TEXT,
    event_date TEXT,
    cadence_seconds INTEGER NOT NULL,
    quiet_retire_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','paused','retired')),
    retired_reason TEXT,
    created_by_dreams INTEGER NOT NULL DEFAULT 0,
    last_checked TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_tracked_status ON dream_tracked_items(status);
CREATE TABLE IF NOT EXISTS dream_track_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tracked_item_id INTEGER NOT NULL REFERENCES dream_tracked_items(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('changed','unchanged','baseline','rebaselined','withheld','error','skipped')),
    digest_hash TEXT,
    verdict_note TEXT NOT NULL DEFAULT '',
    notified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dream_track_runs_item ON dream_track_runs(tracked_item_id, created_at DESC);
```

- [ ] **Step 1: Write the failing tests** — `test_dreams_db_track.py` with a real `DreamsDB(tmp_path)` fixture: create/get/list roundtrip; CHECK rejections (`mechanism='bogus'` raises `sqlite3.IntegrityError`); `set_tracked_status` + `retired_reason`; `insert_track_run` + `list_recent_track_runs` ordering; `consecutive_track_dispositions` counts a trailing run of the given status (e.g. unchanged, unchanged, changed → consecutive('unchanged') == 0, consecutive('changed') == 1); `count_active_tracked` counts only active; v1→v2 upgrade: create a DB, manually stamp `user_version = 1` via `connection()`, reopen, assert both track tables exist and version reads 2.

- [ ] **Step 2: Verify RED** → ModuleNotFoundError-free but method-missing failures.

- [ ] **Step 3: Implement** — append DDL, bump version, add the eight methods in the file's existing style (parameterized SQL, thread-local connection idiom, `created_at/updated_at` via `_utc_now_iso`).

- [ ] **Step 4: Green + full gate.**

- [ ] **Step 5: Commit** — `feat(dreams-track): DreamsDB schema v2 with tracked-items and track-runs tables + CRUD`.

---

### Task 3: Track-this page mechanism (attach-or-create through LocalWatchlistsService)

**Files:**
- Create: `tldw_chatbook/Dreams/track_service.py`
- Modify: `tldw_chatbook/UI/Screens/artifacts_dreams_modal.py`
- Test: `Tests/Dreams/test_track_service.py` (new), `Tests/UI/test_artifacts_dreams_modal.py` (extend)

**Interfaces:**
- Consumes: Task 2 CRUD; the verified `LocalWatchlistsService` APIs (`find_source_id_by_url`, `create_source`, `resolve_or_create_watchlist`, `add_source_to_watchlist`, `create_alert_rule`); `dreams_setting` (Task 6 adds `tracked_item_cap`/`track_min_check_interval_hours` — until then this task reads them with explicit fallbacks `20`/`12` via `dreams_setting(key, default)`).
- Produces:
  - `class TrackCapReached(RuntimeError)` with `reason_code = "track_cap_reached"`.
  - `async def track_page(subs_service, dreams_db, *, url, title, intent, event_date=None, origin_story_id=None, cadence_seconds=None) -> dict` returning `{"tracked_item_id": int, "subscription_id": int, "outcome": "created" | "attached", "watchlist_id": int}`:
    1. `existing = await subs_service.find_source_id_by_url(url)` — if found, `subscription_id = existing`, `outcome = "attached"`, `created_by_dreams = 0` (the existing subscription is left untouched); else `create_source({"name": f"Dreams: {title[:40]}", "type": "url", "source": url, "check_frequency": cadence, "is_active": True})` → `outcome = "created"`, `created_by_dreams = 1`.
    2. `watchlist, _ = await subs_service.resolve_or_create_watchlist("Dreams Tracked")` then `add_source_to_watchlist(watchlist_id=watchlist["id"], source_id=subscription_id)`.
    3. `await subs_service.create_alert_rule(name=f"Change: {title[:40]}", condition_type="items_above", condition_value={"threshold": 0}, job_id=subscription_id, severity="information")` — any new item emerging from the watched page fires a notification into the shared inbox (Watchlists Notifications pane displays it).
    4. Cap guard FIRST (before any creation): `if await asyncio.to_thread(dreams_db.count_active_tracked) >= cap: raise TrackCapReached(...)`.
    5. `tracked_item_id = await asyncio.to_thread(dreams_db.create_tracked_item, ...)` with `cadence_seconds = max(cadence_seconds or 0, min_interval_hours * 3600)`.
  - `async def untrack(subs_service, dreams_db, tracked_item_id) -> dict` — retire the wrapper (`set_tracked_status(..., "retired", retired_reason="manual")`); if `created_by_dreams` and `subscription_id`: disable the subscription — find the deactivation API by grepping `def .*subscription` in `Subscriptions/local_watchlists_service.py`/`DB/Subscriptions_DB.py` (recon knows `update_alert_rule`/`create_source` exist; if a `set_subscription_active`/`update_subscription(fields={"is_active": False})` exists use it, else `UPDATE subscriptions SET is_active = 0 WHERE id = ?` through `subs_service`'s db seam) — record the chosen seam in the test comment. Attached (not dream-created) subscriptions are NEVER touched.
  - Modal: binding `("t", "track", "Track this")` + `action_track` — synthetic rows excluded (same guard as ingest), `dreams://llm/…` excluded (tracking an llm row has no URL to watch), success records feedback `tracked` + `on_changed()` + notice including the outcome word; `TrackCapReached` → notice, no write.

- [ ] **Step 1: Write the failing tests.** `test_track_service.py` with the `test_local_watchlists_service.py` fixture shape (real `SubscriptionsDB(tmp_path)`, `LocalWatchlistsService`, `ClientNotificationsDB`, `NotificationDispatchService`):
  - created path: fresh URL → subscription row exists with `type='url'` + `check_frequency >= min interval`, watchlist "Dreams Tracked" has the source, alert rule row pinned (`condition_type="items_above"`, `job_id=subscription_id`), tracked item row `mechanism='page'`, outcome "created";
  - attach path: pre-create the subscription for the URL first → outcome "attached", subscription count unchanged, `created_by_dreams=0`;
  - cap: seed `tracked_item_cap` active items → `TrackCapReached`, and assert NO subscription got created (guard-first);
  - untrack: dream-created → wrapper retired AND subscription `is_active=0`; attached → wrapper retired, subscription still active;
  - modal: `t` on an http story → feedback `tracked` row + notice; `t` on llm/synthetic → no writes.

- [ ] **Step 2: Verify RED.** **Step 3: Implement** (`track_service.py` ~120 lines; every service call awaited, DreamsDB work in one `to_thread` hop per stage; loguru context on failures). **Step 4: Green + full gate.** **Step 5: Commit** — `feat(dreams-track): page mechanism - attach-or-create subscription + watchlist membership + change alert`.

---

### Task 4: Question mechanism + dream_track_check scheduling + tracked-feedback flip

**Files:**
- Modify: `tldw_chatbook/Dreams/track_service.py`, `tldw_chatbook/Dreams/cycle_service.py`, `tldw_chatbook/Scheduling/services/dreams_projection.py`, `tldw_chatbook/app.py` (the `_wire_dreams_scheduler_integration` seam only)
- Create: `tldw_chatbook/Scheduling/scheduler/handlers/dream_track_handler.py`
- Test: `Tests/Dreams/test_track_service.py` (extend), `Tests/Dreams/test_cycle_service.py` (feedback flip), `Tests/Dreams/test_dreams_scheduler.py` (extend)

**Interfaces:**
- Consumes: Task 2 CRUD; `CycleDeps` (add ONE field: `dispatch_getter: Callable[[], Any] | None = None` — app wiring supplies `lambda: self.notification_dispatch_service`; existing getters/tests unaffected since it defaults None); `chat`/`perform_search` seams; `NotificationDispatchService.dispatch`.
- Produces:
  - `async def track_question(dreams_db, *, query_template, intent, event_date=None, origin_story_id=None, cadence_seconds=None) -> int` (wrapper row, `mechanism='question'`, same cap guard + cadence floor as `track_page`).
  - `async def run_track_check(deps, tracked_item_id) -> dict` returning `{"status": <disposition>, "notified": bool}`:
    1. Load item (missing → `{"status": "skipped"}` — the item retired between emission and dispatch).
    2. Budget: `usage_get(today)`; if searches or llm_calls exhausted → `insert_track_run(status="skipped", verdict_note="budget")`, touch, return (no spend).
    3. Search: `query = query_template.format(region=...)` (no LLM for synthesis — the template IS the query); `results, _ = await discovery.run_queries(deps.perform_search, engine=dreams_setting("search_engine"), queries=[query], result_count=5)`; `usage_bump(searches=1)`. Empty/error results → `withheld`/`error` run.
    4. Digest: `hashlib.sha256("\n".join(normalize_url(r.url) + r.title for r in results))`. Baseline = digest of the most recent non-error run; no baseline → `baseline` run (notified=0).
    5. Same digest → `unchanged` (notified=0). Else one `chat` call (via `deps.chat_getter`, `asyncio.to_thread`, prompt = previous baseline snippets? — NO: prompt carries ONLY the query + current top snippets + the question "material change vs prior result set? answer JSON {\"changed\": bool, \"note\": str}" — prior digests are hashes, snippets from the PREVIOUS run are not stored; the judge compares semantic freshness signals (dates/prices/sold-out markers) and its verdict is heuristic, labeled as such) → parse `{changed, note}`; changed → `changed` run + `notified=1` + `dispatch(category="dreams_track", title=f"Tracked update: {query_template[:50]}", message=note, severity="information", source_entity_kind="dream_tracked_item", source_entity_id=str(tracked_item_id))`; judge-failure → `error` run. `usage_bump(llm_calls=1)` on any real call.
  - `def rebaseline_track(dreams_db, tracked_item_id)` — inserts a `rebaselined` run carrying the latest non-error digest (next comparison anchors here; re-alerting on the same change stops).
  - Projection: module gains `DREAMS_TRACK_PREFIX = "dream_track"` + `parse_dream_track_task_id(task_id) -> int | None` (single-definition lesson); `tasks(now)` additionally emits, per `status='active'` tracked item, `ScheduledTask(id=f"dream_track:{ti_id}", type="dream_track_check", next_run_at=last_checked + cadence)` with two clamps: never-checked → due now; overdue by more than 48h → `now + cadence` (stale checks skip to the next cadence — spec's bounded catch-up). Paused/retired items emit nothing. Still gated on `dreams_setting("enabled")`.
  - `DreamTrackHandler(deps_getter)` in `dream_track_handler.py` — byte-for-byte the `DreamsCycleHandler` spawn pattern (validate id, `asyncio.create_task` named `dream_track_{id}`, module-level strong-ref set + done-callback discard, `async shutdown(timeout)`, never raises into the loop).
  - app.py: inside `_wire_dreams_scheduler_integration` add `handlers["dream_track_check"] = DreamTrackHandler(deps_getter=...)` reusing the SAME CycleDeps getter plus `dispatch_getter=lambda: self.notification_dispatch_service` — one small hunk, deferred import, `# dreams phase 2` comment.
  - **Feedback flip:** `cycle_service._feedback_net` +1 set gains `"tracked"`; update the `_FEEDBACK_WINDOW_DAYS` comment block; update the existing neutral-tracking test to assert +0.1.

- [ ] **Step 1: Write the failing tests.** question flow with injected fakes on a real DreamsDB (`CycleDeps` with `perform_search` fake returning two URLs, `chat` fake returning `{"choices":[{"message":{"content":"{\"changed\": true, \"note\": \"new dates announced\"}"}}]}`, `dispatch_getter` capturing): first check → `baseline`, notified=0, no dispatch; second check same results → `unchanged`; second check different results → `changed`, notified=1, dispatch captured with `source_entity_id=str(id)`; budget-exhausted → `skipped` + zero bumps; judge-raises → `error`; rebaseline after a change → next identical-results check is `unchanged` (not re-alerted). Projection: emission for active-only; never-checked due now; 72h-overdue clamps to now+cadence; `parse_dream_track_task_id` rejects foreign ids. Handler: parse-fail/deps-None never raise; dispatches `run_track_check` (monkeypatched spy). Feedback: `tracked` now +1. App: import smoke.
- [ ] **Step 2: Verify RED.** **Step 3: Implement.** **Step 4: Green + full gate + import smoke** (`../../.venv/bin/python -c "import tldw_chatbook.app"`). **Step 5: Commit** — `feat(dreams-track): question mechanism with judged change runs, dream_track_check scheduling, tracked feedback positive`.

---

### Task 5: Tracked-updates surfacing (Tracked-first) + reminder promotion

**Files:**
- Modify: `tldw_chatbook/Dreams/dreams_view.py`, `tldw_chatbook/Dreams/track_service.py`, `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Test: `Tests/Dreams/test_dreams_view.py` (extend), `Tests/Dreams/test_track_service.py` (extend), `Tests/UI/test_artifacts_dreams_rows.py` (extend)

**Interfaces:**
- Produces:
  - `dreams_view.list_tracked_updates(dreams_db, *, limit=5) -> list[dict]` — rows shaped `{id: tracked_item_id, label, mechanism, intent, status, last_checked, last_run_status, event_date, synthetic: False}`: question items from their latest run; page items labeled via the tracked row itself (`f"Tracking: {query or 'page'}"` with `last_run_status=None` — page dispositions live in the Subscriptions DB and surface through the Watchlists notifications pane; the row's label carries `(alerts → Watchlists)` so the user knows where page notifications land). Ordered: `changed` runs first, then event_date ascending, then created_at.
  - Artifacts screen: a "Tracked" group composed BEFORE the Dreams rows group inside `#artifacts-list-pane` (same bare-Static idiom, ids `artifacts-dream-track-row-{id}`, `can_focus=True`, empty state `"> Tracked: none"`), rendered from a `list_tracked_updates` read folded into the existing `_refresh_dreams` worker (one extra read in the same thread hop — no new worker/trio; the settle helper from the determinism fix already covers both groups).
  - `async def promote_to_reminder(scheduling_db_getter, tracked_item, *, lead_days=7, owner_id="local") -> str | None` in `track_service.py`: if `event_date` set → `await asyncio.to_thread(scheduling_db.create_reminder_task, owner_id, f"Dreams: {tracked_item['query_template'] or 'tracked event'}", body=f"Tracked Dreams event on {event_date}", schedule_kind="one_time", run_at=iso(event_date - lead_days), next_run_at=same, link_type="dream_tracked_item", link_id=str(id))` returning the reminder id; `track_page`/`track_question` call it after wrapper creation (degrade-never-raise: a reminder failure logs + notes, tracking still succeeds).
  - `list_recent_dreams` gains `"tracked": bool` on story rows whose `id` appears as an `origin_story_id` in active tracked items (single extra parameterized `SELECT origin_story_id FROM dream_tracked_items WHERE status='active'` folded into the existing read hop) so the modal can badge "tracked".

- [ ] **Step 1: Write the failing tests.** View: seeded question item with a `changed` run sorts first; page item label carries the Watchlists pointer; empty → `[]`. Reminder: item with `event_date` → reminder row exists with `run_at == event_date - 7d`, `link_id` pinned; no event_date → None; reminder-db failure → tracking still returned success (degrade test with a raising getter). Rows (UI, with `@pytest.mark.bootstrap_profile`): tracked group renders BEFORE dream rows in the compose order (assert index of first `artifacts-dream-track-row-*` widget < first `artifacts-dream-row-*`); empty state string; story modal badge renders for tracked origin (extend an existing modal test).
- [ ] **Step 2: Verify RED.** **Step 3: Implement** (view ~40 lines; screen hunk mirrors the Dreams group exactly; `track_service` +8 lines; `dreams_view` badge join). **Step 4: Green + full gate** (the determinism settle helper must keep both files green in 2 consecutive batch runs). **Step 5: Commit** — `feat(dreams-track): tracked-updates section first on Artifacts, event reminders, tracked badges`.

---

### Task 6: Lifecycle sweep, caps/floors config, untrack UI, docs

**Files:**
- Modify: `tldw_chatbook/Dreams/track_service.py`, `tldw_chatbook/Dreams/cycle_service.py`, `tldw_chatbook/Dreams/settings.py`, `tldw_chatbook/UI/Screens/artifacts_dreams_modal.py`, `Docs/User_Guide/dreams.md`
- Test: `Tests/Dreams/test_track_service.py` (extend), `Tests/Dreams/test_dreams_settings.py` (extend)

**Interfaces:**
- Produces:
  - `async def sweep_track_lifecycle(dreams_db, *, now) -> list[str]` (notes for degradation): for each active item — `event_date + 1 day < now` → `retired("event_passed")`; `consecutive_track_dispositions(id, "unchanged") >= track_quiet_retire_count` → `retired("quiet")`; `consecutive_track_dispositions(id, "error") >= 3` → `paused("failures")` (mirrors `subscriptions.auto_pause_threshold`). Called at `run_cycle` stage 0 (after stale-reclaim, degrade-never-abort) AND at the top of `run_track_check`.
  - `DREAMS_DEFAULTS` gains `tracked_item_cap: 20`, `track_min_check_interval_hours: 12`, `track_quiet_retire_count: 14` (Task 3's fallbacks now resolve through settings); `dreams.md` key table gains the three rows + a "Tracking" section (track this page / track a question, where page alerts land, retirement rules, reminder behavior) matching the guide's current tone.
  - Modal: binding `("u", "untrack", "Untrack")` + `action_untrack` — `find_tracked_by_story(story["id"])` → `untrack(...)` + notice; no tracked item → gentle notice, no write. Footer advertises `t` and `u` together.
  - Cap/floor single-source: `track_page`/`track_question` read the three keys via `dreams_setting` only (fallback literals removed).

- [ ] **Step 1: Write the failing tests.** Sweep: event-passed retires; 14 consecutive unchanged retires (`quiet`); 3 consecutive errors pauses; mixed states untouched; sweep notes recorded. Settings: three keys present with exact defaults. Modal: `u` on a tracked story retires + disables dream-created subscription (fixture from Task 3); `u` on untracked → notice only. Cycle: stage-0 sweep runs (spy) and a sweep exception does not abort the cycle (degrade test).
- [ ] **Step 2: Verify RED.** **Step 3: Implement.** **Step 4: Green + full gate + the three Dreams UI test files twice (determinism).** **Step 5: Commit** — `feat(dreams-track): lifecycle sweep, track settings keys, untrack action, guide section`.

---

## Self-Review (done during writing)

**Spec coverage:** spec task 9 → Task 1 (goals+searchable+region labeling); task 10 → Tasks 2–3 (schema + page mechanism incl. the registration seam, solved: subscription rows are the registration); task 11 → Task 4 (question mechanism + runs + budget integration); task 12 → Task 5 (tracked-updates first + reminder promotion); task 13 → Task 6 (retirement/pause/caps + settings). Spec §track-loop contracts mapped: baseline-first (Task 4 test), rebaseline (Task 4), withheld honesty (Task 4), attach-not-duplicate + untrack semantics (Task 3), disposition vocabulary (Task 2 CHECK), `tracked` feedback positive (Task 4), bounded catch-up (Task 4 clamps), caps/floors (Tasks 3+6). Person-specific tracking stays excluded (spec permanent non-goal). Watchlists-screen surfacing of dream-created sources and the settings UI remain filed follow-ups (TASK-32903; spec Follow-ups) — deliberately absent.

**Placeholder scan:** the one unverified seam (subscription deactivation API) is a named read-first step with a concrete fallback (`UPDATE ... is_active=0` through the service's db seam) — no TBDs.

**Type consistency:** `create_tracked_item` kwargs identical in Tasks 3/4 call sites; `list_tracked_updates` row shape consumed verbatim by the screen rows in Task 5; `preview_queries(topics, goals, *, count)` signature changed once in Task 1 and its single caller updated there; `DREAMS_TRACK_PREFIX`/`parse_dream_track_task_id` defined once (Task 4) and used by handler + tests; `CycleDeps.dispatch_getter` added once with a default so Phase-1 tests compile untouched.
