# Watchlists Briefings Phase 4 — Scheduled Generation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A watchlist with a briefing cadence gets its briefing written on schedule while the app is open — without a scheduled run and a button press ever falsifying or duplicating each other.

**Architecture:** An in-process claim registry in the services (claims die with the process, so the crash-zombie story is untouched) makes the sweeps claim-aware; a `BriefingProjection` emits due jobs from a new per-watchlist cadence column; the handler fire-and-forgets the generation as its own task so a multi-minute LLM call never stalls the serial scheduler tick. The `briefings` table is the run record — it already carries `generating/complete/empty/failed` and is already rendered.

**Tech Stack:** Python 3.11, Textual, SQLite, pytest (`.venv/bin/python -m pytest`, plain output, never `-q`).

## Global Constraints

- pytest is the ONLY python entry point; never `git stash`/`git checkout --`/`git restore` (Edit reverts only); never any `git worktree` command; regenerate the CSS bundle via `build_css.py` if tcss changes; patch `get_user_data_dir` in storage-touching tests; type-only logging; toasts with interpolation `markup=False`; `--strict-markers`; mutation checks (Edit-revert → RED → restore) per behavioural change.
- No new `persist_event` names. (Survey note: the ADR-029 "six" has already drifted to eight via the dictation stream — record that in close-out as a governance observation, do not fix it here.)
- `Tests/Watchlists/` has the TASK-1345 rotating-victim race under load — re-run any sporadic failure alone before classifying.
- Spec §"Scheduling (phase 4)" is **corrected by this plan** (survey 2026-08-01): `automation_definitions` is a dead table (zero production readers, different DB file, `family` constrained to two non-briefing values) — cadence lives on `watchlists` instead; `local_watchlist_runs` cannot hold a briefing run (`source_id NOT NULL` FK, no `empty` status) — the `briefings` table IS the run record; no "fires while the app is open" copy exists anywhere — Task 4 writes it; `[subscriptions.briefings] morning_digest_time` (`config.py:3616`) is read by nothing — do not reuse it.

## Locked decisions (do not relitigate; DO pin with tests)

1. **In-process claims, not DB claim columns.** Manual generation and the scheduler share one process and one event loop (`app.py:7153` runs `SchedulerLoop.run()` as a coroutine worker). A module-level claim registry fixes every in-process collision; the accepted phase-1 limitation about a *second app instance* stays exactly as it was. No schema, no heartbeat/expiry protocol — a claim cannot outlive the process, so real crash zombies still look exactly like crash zombies.
2. **Sweeps take an explicit `exclude` parameter.** `fail_interrupted_*` are sync and run under `asyncio.to_thread`; the claim sets mutate only on the event loop. Callers snapshot the set and pass it in — no cross-thread reads.
3. **The handler must not stall the tick.** `SchedulerLoop.tick` awaits handlers serially inline (`loop.py:119-158`); checks are quick HTTP fetches, a briefing is a multi-minute LLM call. The briefing handler claims, then spawns the generation as an independent `asyncio.Task` and returns; re-emission of a still-running job is refused by the claim.
4. **Scheduled briefings are opt-in, per watchlist, off by default.** A schedule spends the user's LLM tokens unattended; `briefing_cadence_seconds = NULL` means never.
5. **The `briefings` table is the run record.** It already has richer honest statuses (`generating/complete/empty/failed` + `error='interrupted'`) than `local_watchlist_runs` (`queued/running/completed/failed/cancelled`, no `empty`), and the Artifacts table already renders every row. No second record.

---

### Task 1: In-process generation claims, claim-aware sweeps

**Files:** `Subscriptions/briefing_service.py`, `Subscriptions/briefing_cast.py` (`fail_interrupted_scripts:644`), `Subscriptions/briefing_audio.py` (`fail_interrupted_audio:1247`); `UI/Screens/watchlists_collections_screen.py` (sweep call sites + cast's missing blocking refusal). Tests: extend `Tests/Subscriptions/test_briefing_service.py`, `test_briefing_cast.py`, `test_briefing_audio_pipeline.py`, `Tests/Watchlists/test_watchlists_artifacts_pane.py`.

**Interfaces — Produces:**
```python
# briefing_service.py
class GenerationInFlightError(RuntimeError): ...   # message names the watchlist
def active_briefing_claims() -> frozenset[int]     # snapshot, for sweep excludes
# generate_briefing claims watchlist_id pre-insert (raises GenerationInFlightError if
# already claimed), releases in finally. Analogous: briefing_cast claims briefing_id,
# briefing_audio claims script_id, each with active_*_claims().
def fail_interrupted_briefings(db, watchlist_id=None, *,
                               exclude: Collection[int] = ()) -> int   # skips claimed scopes
```
- [ ] Failing tests first. The load-bearing ones: **a claimed watchlist's `generating` row survives `fail_interrupted_briefings` when passed in `exclude`** (and is swept when not — both directions); a second `generate_briefing` for a claimed watchlist raises `GenerationInFlightError` **before any row insert** (phase-1's no-orphan-row contract); the claim is released on success, on chat failure, AND on a DB error escaping (a stuck claim wedges scheduling forever — assert via `active_briefing_claims()` after each path); same trio for cast/audio. Screen: `_load_briefings`'s sweep now passes the snapshot — a fake "scheduled" claim (claim directly via the service) must survive an Artifacts open (this is survey finding (a) as a test); Generate during a claim → the existing `blocking` refusal toast fires instead of falsifying the row (finding (b)); **Cast gets the blocking refusal it never had** (`watchlists_collections_screen.py:4862-4868` documents its absence — finding (c): a Cast press during a claimed cast must refuse, not run concurrently).
- [ ] Implement. Claim sets are module-level `set[int]`, mutated only in the async orchestrators (event-loop-only ⇒ no lock); `try/finally` release; sweeps gain `exclude` with `()` default so every existing caller is unchanged. Also fix `_default_provider`'s false docstring while in the file (it claims call-time config reads; `config.default_api_endpoint` is assigned once at import — `config.py:5410-5422` — say what is true).
- [ ] Mutations: (i) drop the `exclude` filtering → the survives-sweep test REDs; (ii) release the claim before the chat call instead of in `finally` → the concurrent-refusal test REDs; (iii) remove cast's new blocking refusal → its test REDs. Commit `feat(briefings): in-process generation claims — sweeps spare live runs`.

### Task 2: Per-watchlist cadence in the DB

**Files:** `DB/Subscriptions_DB.py`; test `Tests/Subscriptions/test_briefing_cadence_db.py` (new).
**Interfaces — Produces:**
```python
# watchlists gains: briefing_cadence_seconds INTEGER  (NULL = never; additive ALTER in the
# column-presence block at :687-696)
set_watchlist_briefing_settings(..., briefing_cadence_seconds: object = _UNSET)  # sentinel
# pattern verbatim from default_preset_id (:2334-2405); None clears, _UNSET leaves alone;
# non-positive ints raise ValueError naming the value
def list_briefing_schedules(self) -> List[Dict[str, Any]]
# one row per watchlist with a non-NULL cadence: watchlist_id, name,
# briefing_cadence_seconds, last_completed_at (MAX(created_at) over that watchlist's
# briefings WHERE status IN ('complete','empty') — the SAME allowlist as
# latest_completed_watermark; a failed run must NOT advance the schedule, mirroring
# "failure never advances coverage")
```
- [ ] Failing tests: sentinel trio (set/clear/leave); validation; `list_briefing_schedules` excludes NULL-cadence watchlists (by identity); `last_completed_at` ignores `failed`/`generating` rows (both seeded — the schedule-never-advances-on-failure invariant); reads inside `transaction()`; Google docstrings. Mutations: widen the status allowlist to include `failed` → invariant test REDs; drop the NULL-cadence filter → identity test REDs. Commit `feat(briefings): per-watchlist briefing cadence`.

### Task 3: Projection, handler, wiring

**Files:** `Scheduling/services/briefing_projection.py` (new), `Scheduling/scheduler/handlers/briefing_handler.py` (new), `Scheduling/scheduler/queue.py` (`PriorityQueue.load:35-61`), `app.py` (`:4736-4753` handler dict + construction). Tests: `Tests/Scheduling/test_briefing_projection.py`, `test_briefing_handler.py` (new); extend `Tests/Scheduling/`'s loop tests only if a seam demands it.
**Interfaces — Consumes:** `WatchlistProjection` (`services/watchlist_projection.py:73-93`) as the shape precedent — `list_jobs(owner_id)` emitting `ScheduledTask(id=f"briefing:{watchlist_id}", type="briefing_job", next_run_at=...)`; `SchedulerLoop(db, handlers, ...)` (`loop.py:21-30`) — `tick` dispatches on `task["type"]`, unknown types only warn; Task 2's `list_briefing_schedules`; Task 1's claims.
- [ ] Read first: `watchlist_check_handler.py` (`_WATCHLIST_TASK_PREFIX:19`, `_parse_subscription_id:326-340`, the counter/histogram observability at `:230-234` — mirror it, NO persist_event); `queue.py:35-61` — the two work sources are hardcoded, so **generalize minimally**: a second optional projection parameter threaded exactly like `watchlist_projection` (a projections *list* is a refactor nobody asked for).
- [ ] Failing tests: projection emits one task per scheduled watchlist with `next_run_at = last_completed_at + cadence` (and `now` for never-briefed); no task for NULL cadence; id round-trips through the handler's parser (the two-copies-drift lesson from 2b — one parser, tested); **handler behaviour**: claims-then-spawns and returns before generation completes (assert the tick isn't blocked — a slow fake chat must not delay handler return); a re-emitted job for a claimed watchlist is skipped silently-with-a-log, not queued twice; generation failure inside the spawned task is contained (the task must not become an unhandled-exception app event — assert the failed briefing row instead); `empty` result writes the `empty` row (the spec's "empty rows when nothing is new" — already `_finish_empty`'s behaviour, pin it end-to-end from the handler).
- [ ] `app.py` wiring: `handlers["briefing_job"]`, projection constructed beside the watchlist one, gated on the same `[scheduling] watchlist_checks_enabled`-style config read — add `briefing_schedules_enabled = true` beside it (`config.py:2347-2364`), READ it (no more dead keys on this stream).
- [ ] Mutations: (i) handler awaits the generation inline → the non-blocking test REDs; (ii) projection ignores `last_completed_at` → the next-run test REDs. Commit `feat(briefings): scheduled briefing generation through the scheduler seam`.

### Task 4: Cadence UI + honest copy

**Files:** `UI/Watchlists_Modules/artifacts_pane.py` (`#artifacts-picker-toolbar:775-802`), `UI/Screens/watchlists_collections_screen.py` (mirror the `BriefingModeChanged` chain `:4362-4420`), `Docs/User_Guide/watchlists.md`. Tests: extend `Tests/Watchlists/test_watchlists_artifacts_pane.py`.
- [ ] Failing tests: a third compact `Select` (`Off` default + Daily/12h/Weekly-style options) renders in the picker toolbar only at watchlist scope; changing it writes through `set_watchlist_briefing_settings` off-loop (thread-identity pin, the established pattern) with the stale-write watchlist guard (`:4378`'s shape); the stored cadence is reflected on load (read-path pin, seeded via Task 2's writer); **the scope-note lie is fixed**: `_briefing_scope_label` (`:3223-3237`) says "written on this device, on request" — with a cadence set it must say scheduled-while-open honestly (e.g. "written on this device — scheduled daily while the app is open"), and stay "on request" when cadence is NULL. Both pinned geometry tests stay green.
- [ ] User guide: a short "Scheduled briefings" note — opt-in, per watchlist, **runs only while the app is open** (this is where the spec's promised copy finally gets written), and failures never advance the schedule.
- [ ] Mutations: (i) drop the screen write → persistence test REDs while the Select still shows the value; (ii) make the scope note ignore cadence → the honesty test REDs. Commit `feat(briefings): cadence picker and honest scheduling copy`.

### Task 5: Close-out

- [ ] Sweep `Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/ Tests/UI/ -k watchlist` (baselines: tree-chevron ×2, TASK-1345 rotating victim — isolate before classifying).
- [ ] `task-1540`: check AC #4 — **spec #2 is complete**.
- [ ] Spec: "Phase 4 delivery notes (2026-08-01)" — the four survey corrections (automation_definitions dead / cadence on watchlists; briefings table as run record; the copy written not mirrored; morning_digest_time dead), the claims design and why in-process suffices, the fire-and-forget handler, and the persist_event-drift governance observation (eight names vs the admitted six — flagged for the ADR-029 owner, not fixed here).
- [ ] Cross-worktree ID scan (controller supplies IDs) → file follow-ups for anything parked.
- [ ] Commit `docs(briefings): phase 4 close-out — spec #2 complete`.

## Self-review

**Spec coverage:** job type through the real seam (dict key + projection param — T3); real run records/honest statuses/`empty` rows → Decision 5 + T3's end-to-end empty pin; cadence per watchlist → T2/T4 (mechanism corrected from the dead table, intent preserved); fires-while-open → structural (worker lifecycle) + T4 writes the promised copy. The collision the survey found → T1, first, because everything else builds on sweeps that no longer kill live runs.

**Placeholders:** none; every step names its precedent `file:line`.

**Type consistency:** claims are `int` scopes throughout (watchlist_id/briefing_id/script_id); `exclude: Collection[int]` on all three sweeps; `briefing_cadence_seconds` `int | None` at the DB seam with the `_UNSET` sentinel; `ScheduledTask.type == "briefing_job"` matches the `app.py` dict key and the handler's parser prefix `briefing:`.
