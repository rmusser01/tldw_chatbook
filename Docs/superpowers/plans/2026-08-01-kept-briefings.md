# Kept Briefings Implementation Plan (task-1780)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Briefings/scripts persist into ChaChaNotes (auto for scheduled, Keep for manual), survive watchlist deletion, and can be re-cast later with whatever presets exist then.

**Architecture:** Two new ChaChaNotes tables (schema bump + migration), an additive-idempotent keep service with two callers (Keep button, scheduler handler), a cast-from-text variant with its own claim space, and a scope-independent modal. Spec: `Docs/superpowers/specs/2026-08-01-kept-briefings-design.md` — its Owner decisions and amendments bind every task.

**Tech Stack:** Python 3.11, Textual, SQLite, pytest (`.venv/bin/python -m pytest`, plain output, never `-q`).

## Global Constraints

- pytest is the ONLY python entry point — **no probes, no `python -c`** (this stream logged live-DB incident #4 from one; treat as hard).
- Never `git stash`/`git checkout --`/`git restore` (Edit reverts only); never any `git worktree` command; regenerate the CSS bundle via `build_css.py` on tcss change; patch `get_user_data_dir`/DB paths in every storage-touching test; type-only logging (never body/roster/turn content); toasts `markup=False`; remote-derived text via `rich.text.Text`; `--strict-markers`; mutation checks (Edit-revert → RED → restore) per behavioural change; TASK-1345 rotating-victim flake — isolate before classifying.
- ChaChaNotes schema change: increment `_CURRENT_SCHEMA_VERSION`, add a migration, **re-verify the version number against origin/dev at merge** (five historical collisions).
- All error-ethos conventions from spec #2 phases 1-4: pre-flight refusals raise before any row insert; in-band failures write honest state; parent rows never touched by a child outcome.

---

### Task 1: ChaChaNotes schema + CRUD

**Files:** `tldw_chatbook/DB/ChaChaNotes_DB.py` (+ migration file per its existing convention — READ how the last version bump was done first, including whether migrations live in `DB/migrations/` for this DB); test `Tests/ChaChaNotes/test_kept_briefings_db.py` (new; follow that dir's conventions, or `Tests/DB/` if that is where ChaChaNotes tests live — check).
**Produces (verbatim for later tasks):**
```python
create_kept_briefing(self, *, source_briefing_id, watchlist_name, body_markdown,
    covers_through_item_id=None, covers_from_ts=None, selection_mode=None, model_used=None,
    item_count=0, featured_count=0, overflow_count=0, origin, original_created_at=None) -> int
get_kept_briefing_by_source(self, source_briefing_id) -> Optional[Dict]
get_kept_briefing(self, kept_id) -> Optional[Dict]
list_kept_briefings(self, *, limit=200, offset=0) -> List[Dict]      # kept_at DESC, id DESC
delete_kept_briefing(self, kept_id) -> bool                          # hard delete, cascades scripts
create_kept_script(self, kept_briefing_id, *, source_script_id=None, preset_name,
    roster_snapshot_json, turns_json, model_used=None, original_created_at=None) -> int
list_kept_scripts(self, kept_briefing_id, *, limit=200, offset=0) -> List[Dict]
kept_script_source_ids(self, kept_briefing_id) -> set[int]           # non-NULL sources only
```
- [ ] Schema per the spec verbatim: `source_briefing_id UNIQUE NOT NULL`; `kept_scripts.source_script_id UNIQUE` nullable; `origin CHECK IN ('manual','scheduled')`; FK cascade kept_scripts→kept_briefings with `PRAGMA foreign_keys` verified ON for this DB (test the cascade by observation, not the DDL clause); **no sync columns** (deliberate, comment it).
- [ ] Tests: round-trips; UNIQUE violation on duplicate source ids raises; cascade observed; ordering by identity; pagination LIMIT/OFFSET real (spy params); `kept_script_source_ids` excludes NULLs. Mutations: drop the cascade-enforcement seed → cascade test REDs; drop LIMIT → pagination REDs.
- [ ] Migration + version bump per house convention; a fresh-DB and an upgraded-DB both end at the same schema (test both paths if the house migration tests do — read them).
- [ ] Commit `feat(kept): kept_briefings/kept_scripts tables in ChaChaNotes`.

### Task 2: The keep service

**Files:** create `tldw_chatbook/Subscriptions/briefing_keep.py`; test `Tests/Subscriptions/test_briefing_keep.py`.
**Consumes:** Task 1 CRUD; `subs_db.get_briefing/list_briefing_scripts` (phases 1-2a).
**Produces:** `keep_briefing(subs_db, chacha_db, briefing_id, *, origin) -> dict` returning `{kept_id, created: bool, scripts_added: int}`; `class KeepRefused(RuntimeError)`.
- [ ] Contract per spec: refuses (raises `KeepRefused`, no row) for missing/non-`complete`/empty-body briefings; copies all **complete** scripts; **additive-idempotent** — re-keep adds scripts missing by `source_script_id`, never duplicates (pin with the UNIQUE + `kept_script_source_ids` diff), never overwrites (byte-identical existing rows asserted); all provenance denormalized (watchlist name resolved to TEXT — if the watchlist row is already gone, fall back to `"(deleted watchlist)"`, tested).
- [ ] **Plan-time verification the spec mandates:** determine whether `CharactersRAGDB` is thread-safe under `asyncio.to_thread` (read its connection handling; state the answer in the report). The service itself is sync (callers wrap); document that.
- [ ] Named tests: `test_kept_rows_survive_watchlist_deletion` (AC #3 — delete the watchlist via the real path, re-read kept rows); additive idempotency both directions; empty-body refusal. Mutations: allow empty-body → REDs; break the source_script_id dedup → duplicate test REDs.
- [ ] Commit `feat(kept): additive-idempotent keep service`.

### Task 3: Auto-keep from the scheduler

**Files:** `tldw_chatbook/Scheduling/scheduler/handlers/briefing_handler.py`, `app.py` wiring; extend `Tests/Scheduling/test_briefing_handler.py`.
- [ ] Handler gains optional `chachanotes_db=None`; after a scheduled generation ends `complete` with non-empty body, call `keep_briefing(..., origin='scheduled')` via `asyncio.to_thread` inside the spawned task. Missing handle → skip with debug log. **Any keep failure logs `type(exc).__name__` and never fails the generation** (the briefing row stays `complete`).
- [ ] app.py passes the ChaChaNotes instance if constructed at wiring time, else None (read what app init guarantees — the handler must tolerate None, tested).
- [ ] Tests: complete→kept with origin scheduled; `empty`→NOT kept (auto-keep-skips-empty, named); keep raising → generation still completes and the briefing row untouched (mutation: let the keep exception propagate → REDs); None handle → no crash, no keep.
- [ ] Commit `feat(kept): scheduled briefings auto-keep on completion`.

### Task 4: Cast-from-kept

**Files:** `tldw_chatbook/Subscriptions/briefing_cast.py`; extend `Tests/Subscriptions/test_briefing_cast.py`.
**Produces:** `async generate_script_from_text(chacha_db, kept_briefing_id, *, preset_id, subs_db, chat=chat_api_call, load_character=None) -> dict` (preset resolved from subs_db's `briefing_presets`; `preset_id=None` → app default provider/model, no style notes).
- [ ] Reuses `build_cast_prompt`/`parse_script_turns` verbatim; writes into `kept_scripts` (`source_script_id=NULL`); **its own claim set** (`_ACTIVE_KEPT_CAST_CLAIMS`, keyed by kept_briefing_id — a DIFFERENT id space from live briefing ids, comment why) with `active_kept_cast_claims()` snapshot + `GenerationInFlightError` on collision; pre-flight raises (missing kept row, empty body, missing preset when preset_id given) before any row; failure writes an honest failed state WITHOUT touching the kept briefing row (byte-identical assertion, the named invariant). Hmm — kept_scripts has no status column: on failure, write NO row and surface the error to the caller instead (unlike live casts, there is no observability table here — the modal shows the error as a toast; document this asymmetry in the docstring).
- [ ] Named test: `test_recast_needs_no_subscriptions_rows` (AC #4 — cast from kept after deleting the watchlist AND the original preset; a currently-existing different preset works). Mutations: reuse the live cast claim set → the id-space collision test REDs; drop the kept-row-untouched guard → invariant REDs.
- [ ] Commit `feat(kept): cast a new script from a kept briefing`.

### Task 5: UI — Keep button + Kept briefings modal

**Files:** `UI/Watchlists_Modules/artifacts_pane.py` (Keep button in `#artifacts-toolbar`, `compact=True` — the strip is height:1; a new `Horizontal` costs a pinned-budget row, do not add one), new `UI/Watchlists_Modules/kept_briefings_modal.py` (preset-manager shape: `ModalScreen`, list + detail + hard-delete-with-confirm + preset Select + Cast), screen wiring in `watchlists_collections_screen.py`; extend `Tests/Watchlists/test_watchlists_artifacts_pane.py` + new `Tests/Watchlists/test_kept_briefings_modal.py`.
- [ ] Keep button enabled only when a complete briefing is selected; handler claims a screen-side in-flight flag at dispatch (finally-cleared — the wedge lesson), runs keep via to_thread, toasts kept-with-N-scripts or already-kept+added-count (`markup=False`); mutation: guard claimed in worker body → double-press REDs.
- [ ] Modal: reachable regardless of watchlist scope (button always enabled when the modal's own prerequisites hold — it lists ChaChaNotes content); body via `Markdown(hyperlinks=False)`; every remote-derived string `Text`-wrapped; delete confirms then hard-deletes (cascade observed in a test); Cast uses `generate_script_from_text` off-loop with its own in-flight guard, errors toast honestly (`GenerationInFlightError` gets its specific message — the phase-4 lesson, not the generic DB toast); the new script appears in the modal's detail after cast.
- [ ] Real-input tests (press the actual buttons — the phase-2a input-seam lesson; remember cell-cursor tables never fire `Row*`); both pinned geometry tests stay green; modal geometry gets a real-CSS on-screen assertion + styling mutation (the three-way-vacuity lesson).
- [ ] Commit `feat(kept): Keep action and the kept-briefings modal`.

### Task 6: Close-out

- [ ] Sweep `Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/ Tests/ChaChaNotes|DB/ Tests/UI/ -k watchlist` (baselines: tree-chevron ×2, TASK-1345 rotating victims — isolate first).
- [ ] task-1780: check ACs 1-4 and 6; AC 5 was satisfied by the owner Q&A (note the decisions inline). Verify the ChaChaNotes schema version against origin/dev ONE more time.
- [ ] Spec → Status: implemented; file the sync/export-gap follow-up task (cross-worktree ID scan; uppercase `TASK-` id).
- [ ] Commit `docs(kept): task-1780 close-out`.

## Self-review

**Spec coverage:** decisions 1-3 → T1/T3+T5/T2 scope; amendments: optional-handle/never-fails → T3; empty-skip → T2+T3; additive-idempotent → T2; separate claim space → T4; preset picker → T5; sync-gap follow-up → T6. AC 3 named test T2; AC 4 named test T4; AC 6 → T1+T6 version checks.
**Placeholders:** none; where house conventions are unknown (ChaChaNotes migration shape, test dir), the task mandates reading them first rather than guessing line numbers.
**Type consistency:** `origin` literal set matches CHECK constraint; `source_script_id` nullable everywhere; `keep_briefing` return shape quoted by T3/T5; claim accessor naming mirrors phase 4's.
