# Task 6 — close-out

Final task of the phase 1 delivery. Full test sweep, two backlog tasks filed
(programme tracking + a review-found follow-up), spec status flipped, one
commit.

## Sweep

`Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/ Tests/UI/ -k watchlist`:

```
2 failed, 662 passed, 6335 deselected in 525.80s (0:08:45)
```

Failures, both matching the documented tree-chevron baseline (not ours,
pre-existing, verified against git history predating this branch's commits):

- `Tests/UI/test_destination_visual_parity_correction.py::test_watchlists_tree_chevron_shares_a_row_with_its_watchlist[size0]`
- `Tests/UI/test_destination_visual_parity_correction.py::test_watchlists_tree_chevron_shares_a_row_with_its_watchlist[size1]`

Both fail on the same assertion: the expanded source row's indent no longer
sorts after the watchlist row's chevron column in the painted strip
comparison (`assert 4 > 5`). Unrelated to briefings; not touched by tasks 1-5.

No other failures. The documented focus-race flake (TASK-1345) did not
reproduce this run. The documented `test_chat_shell_bar.py` collection error
also did not reproduce — that file collects cleanly on its own (15 items) and
no collection-error line appeared in the sweep's summary; nothing to action
either way since the brief only says these baselines "may" appear.

`Tests/UI/test_watchlists_inspector.py` unfiltered: 34 passed, including
`test_the_queue_write_runs_off_the_event_loop_thread` (the Task 5 review-round
pattern task-1541 asks a future fix to replicate).

No failures outside the two documented baselines. Nothing BLOCKED.

## Backlog tasks filed

Verified neither filename existed in this worktree's `backlog/tasks/` before
writing (`ls`, not git); highest existing id in this worktree was 1494, so
both new files land clean at the controller-assigned 1540/1541 with no local
collision. Frontmatter/section structure copied from the newest existing file,
`task-1494 - The-readers-full-page-and-previous-snapshot-affordances-were-never-built.md`.

- **`backlog/tasks/task-1540 - Watchlists-briefings-spec-2-programme-tracking.md`**
  — programme tracker for spec #2, pointing at
  `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`. AC #1
  (phase 1: tables, id-watermark selection, `chat_api_call` pipeline,
  Artifacts section, queue affordance, preset-less) checked `[x]`; AC #2-4
  (presets/scripts/audio, exports+feed directory, TASK-1383 scheduling)
  unchecked.

- **`backlog/tasks/task-1541 - Watchlists-screen-item-status-writes-never-leave-the-event-loop.md`**
  — the screen-wide version of the bug Task 5's review found and fixed only
  for the new queue-toggle write. Verified directly against source before
  filing: `WatchlistsBackendController._maybe_await` really is at
  `tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py:29`
  (the dispatch's stated `Widgets/...` path is stale — I corrected it to the
  real path in the task body) and has no `to_thread`; `_update_item_status`
  (`watchlists_collections_screen.py:3923`) routes through it and is invoked
  via bare `run_worker(coroutine, exclusive=True)` from Ingest, Ignore, the
  unread toggle, and the silent mark-read-on-open path; `Subscriptions_DB.py`
  has no `busy_timeout` pragma (grepped, zero hits). AC #1 asks for a
  thread-identity test shaped like
  `test_the_queue_write_runs_off_the_event_loop_thread`; AC #2 forbids adding
  `exclusive=True` cancellation across different items' writes; AC #3 pins
  the existing item-action tests as a regression guard.

## Spec status

`Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` line 4:
`**Status:** proposed` → `**Status:** phase 1 implemented (2026-07-30);
phases 2-4 pending`. Diff confirmed as the sole change to that file (`git
diff --stat`: 1 file, 1 line). Spec #1
(`2026-07-25-watchlists-console-rebuild-design.md`) untouched — not in the
diff at all.

## Commit

`bd883151d6d2ee8b1cf22b0d0c8f2446ff4a0edf` —
`docs(briefings): phase 1 close-out — spec status, tracking task, event-loop follow-up`
(3 files changed: the spec edit + the two new task files).

---

# Whole-branch fix wave

Six per-task reviews passed; the whole-branch review found six more findings across the
generation service, the selection query, the Artifacts screen, and test/doc hygiene. All six
fixed below.

## Fix 1 (Important) — `generate_briefing`'s DB work off the event loop

`generate_briefing` was `async` but every DB call inside it (`insert_briefing`,
`select_briefing_items`, the junction writes, `update_briefing`, `get_briefing`,
`latest_completed_watermark`) ran synchronously on the caller's event loop — the screen dispatches
it from a Textual worker, so a contended sqlite write blocked the whole UI.

Grouped the sync work into four small plain functions (`_start_generation`, `_finish_empty`,
`_finish_success`, `_finish_failure`) and wrapped each in its own `asyncio.to_thread` hop —
one hop per stage, not one per statement. `_invoke_chat` is still awaited directly (unthreaded at
this layer; it already offloads the real network call itself), and the `try/except` still wraps
only `_invoke_chat` — every `to_thread` call for DB work sits outside it, so a DB error from any
of those hops still propagates to the caller uncaught, exactly as before.

**Test-infrastructure fallout, found by running it:** `Tests/Subscriptions/test_briefing_service.py`
used `SubscriptionsDB(":memory:", "test")` throughout. `SubscriptionsDB.conn` is thread-local
(`Subscriptions_DB._initialize_schema`'s own docstring documents the trade-off), so a `:memory:`
connection is private to the thread that opened it — the very first `to_thread` hop reached a
brand-new, unmigrated, empty database on the executor thread ("no such table: briefings"), even
though it was the same `db` object throughout. Converted the file's `_db()` helper to take
`tmp_path` and use a real file (`tmp_path / "subs.db"`), matching the idiom
`test_watchlist_name_and_copy.py`'s `_service(tmp_path)` already uses for the same DB class — a
file-backed connection has no such limitation. All 8 call sites updated; 9th
(`test_interrupted_recovery_only_touches_generating_rows`) switched too for consistency though it
never crosses a thread boundary.

**New test:** `test_the_db_work_runs_off_the_event_loop_thread` — spies on `insert_briefing`,
`update_briefing`, `get_briefing` (the setup hop and the finishing hop), captures
`threading.get_ident()` per call, asserts none match the event-loop thread. Same pattern as
`test_the_queue_write_runs_off_the_event_loop_thread` in `Tests/UI/test_watchlists_inspector.py`.

**Mutation:** reverted the `_start_generation` call from `await asyncio.to_thread(...)` to a
direct synchronous call. RED: `AssertionError: generate_briefing's DB work must run off the
event-loop thread`. Restored; `Tests/Subscriptions/test_briefing_service.py` back to 12 passed.

## Fix 2 (Important) — bound the window query in SQL

`briefing_selection._window_rows` had no `LIMIT`; it materialised the entire coverage window
(every row with `id > watermark`) and the item cap was applied in Python afterward. A watchlist
left unbriefed for a while could have a window backlog far larger than the ~40-item cap,
materialised into full `dict` rows for nothing.

Added `_window_predicate` (the shared WHERE-fragment builder), `_window_count` (exact `COUNT(*)`
over the full window, optionally excluding ids already claimed by the featured side so
`overflow_count` never double-counts an item that is both featured and inside the window), and
`_window_bounds` (`MAX(id)`/`MIN(created_at)` over the full window, unfiltered — `covers_through`
and `covers_from_ts` are properties of everything considered, featured or not). `_window_rows`
gained a required `limit` and now issues `... ORDER BY i.id DESC LIMIT ?`, with the featured-id
exclusion pushed into the SQL `NOT IN` clause too, so the materialised "auto" rows are already
exact — no Python-side overfetch-then-dedup needed. `select_briefing_items`'s combination logic
was rewritten around these three calls; `MODE_CURATED`'s path (a user's queue, not a window
backlog) is untouched — still fully materialised, as the fix scoped it.

All 22 pre-existing tests pass **unmodified** — verified by running them before touching any test
code, then again after.

**New tests:**
- `test_overflow_and_watermark_stay_exact_over_a_backlog_larger_than_the_cap` — seeds `cap + 30`
  window items; asserts `overflow_count` exact (`len(backlog) - cap`), `covers_through_item_id`
  the TRUE max id (not the max kept), and the retained items are the newest `cap` by id.
- `test_the_window_materialisation_is_bounded_not_the_full_backlog` — spies on `_window_rows`,
  seeds featured (queued, below-watermark) items plus a `cap + 30` backlog, asserts every fetch is
  `<= cap + featured_count`, never the full window.

**Mutation:** dropped the `LIMIT ?` from `_window_rows`'s SQL (kept the params list under-fed so
the query stayed syntactically valid). RED: `test_the_window_materialisation_is_bounded_...` —
`assert 37 == 5` (all 37 backlog rows materialised instead of the 5-item cap). Restored;
`Tests/Subscriptions/test_briefing_selection.py` back to 24 passed.

## Fix 3 (Important) — zombie recovery on Artifacts load

`fail_interrupted_briefings` had one production caller, `_sweep_and_guard` (the Generate path).
The spec says a `generating` row not backed by a live worker is failed "on the next Generate
attempt **or Artifacts load**" — the load half was never wired.

Added `WatchlistsCollectionsScreen._zombie_sweep_is_safe()` (the shared predicate: unsafe exactly
when `_briefing_in_flight` is true, because `fail_interrupted_briefings` fails every `generating`
row unconditionally and a live `_generate_briefing` worker owns exactly one such row for the
whole time it runs) and `_fail_interrupted_briefings_if_safe(db, watchlist_id)` (the async
wrapper, `asyncio.to_thread`, off the UI thread). Wired into `_load_briefings` immediately before
the `list_briefings` query, wrapped in its own best-effort `try/except` so a sweep failure cannot
take the `wl-briefings-load` worker's default `exit_on_error=True` down with it. `_sweep_and_guard`
itself needed no change — it always runs at the front of `_generate_briefing`, before that
worker's own row exists, so it never needed this guard in the first place.

**Existing-test fallout:** `test_a_stuck_generating_row_is_refused_then_recovered` seeded its
zombie row *before* opening Artifacts, so the plain load now recovered it before the first press
ever ran, collapsing the test's two-press structure. Moved the zombie's insertion to *after* the
section is already open (post-load), which isolates what the test is actually for — a row that
appears while the screen is already sitting open.

**New tests:**
- `test_a_zombie_generating_row_is_recovered_on_a_plain_artifacts_load` — seeds a zombie, opens
  Artifacts with no button press, asserts the row is `failed`/`interrupted` and its detail pane
  never shows `_GENERATING_COPY`.
- `test_a_live_in_flight_row_is_not_failed_by_a_concurrent_load` — claims `_briefing_in_flight`,
  calls `_load_briefings()` directly, asserts a pre-seeded `generating` row survives untouched.

**Mutation:** removed the `_fail_interrupted_briefings_if_safe` call from `_load_briefings`. RED:
`test_a_zombie_generating_row_is_recovered_on_a_plain_artifacts_load` (row stayed `generating`).
Restored; `Tests/Watchlists/test_watchlists_artifacts_pane.py` back to 16 passed.

## Fix 4 (Minor) — truthful refusal toast

`_briefing_in_flight`'s refusal toast said "A briefing is already being written for this
watchlist" — false whenever the running generation belongs to a different watchlist than the one
on screen (the flag is deliberately screen-global; the `wl-briefing` worker group is
`exclusive=True`, so making it per-watchlist would let a second dispatch cancel a real generation
mid-run).

Added `_briefing_in_flight_watchlist_id`, set alongside `_briefing_in_flight = True` in
`handle_generate_briefing_requested` and cleared alongside it in `_generate_briefing`'s `finally`.
The refusal toast now names the watchlist actually generating (`_watchlist_display_name`) when
that id is set, falling back to watchlist-agnostic copy ("A briefing is already being written.
Nothing else was started.") otherwise — never claiming "this watchlist" when it may not be.

**New test:** `test_the_refusal_toast_names_the_watchlist_actually_generating` — opens Artifacts on
one watchlist, simulates the guard being claimed by a *different* watchlist, presses Generate,
asserts the toast names the running watchlist and never says "for this watchlist".

**Mutation:** reverted the message to the old fixed string. RED: `assert 'Morning AI Brief' in
'A briefing is already being written for this watchlist.'`. Restored;
`Tests/Watchlists/test_watchlists_artifacts_pane.py` back to 16 passed.

## Fix 5 (Minor) — unmarked tests

`Tests/Watchlists/test_watchlists_items_pane.py` had no `pytestmark`, so the whole file — not
just the two new task-5 tests — was invisible to `pytest -m unit`. Added
`pytestmark = pytest.mark.unit` at module level, matching the convention three sibling files in
the same directory already use (`test_watchlist_name_and_copy.py`, `test_region_layout_store.py`,
`test_watchlist_dialogs_escape.py`).

## Fix 6 — deferral notes (docs only)

Added a "Phase 1 delivery notes (2026-07-30)" subsection immediately under the Status line in
`Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`, naming two honest deferrals —
the selection-mode picker (column ships, no writer, `auto`/`curated` unreachable until phase 2)
and citations (plain-text `[item N]` markers only; links-into-reader and pruned-item degradation,
including the named `citation-to-pruned-item-degrades` invariant test, move to phase 2) — both
noted as pending the project owner's confirmation. Mirrored as two unchecked sub-bullets under
AC #2 in `backlog/tasks/task-1540 - Watchlists-briefings-spec-2-programme-tracking.md`. No code
touched.

## Verification

- `Tests/Subscriptions/` — **226 passed**.
- `Tests/Watchlists/` — **218 passed**.
- `Tests/UI/test_watchlists_inspector.py` — **34 passed**.
- Combined single run of all three — **478 passed, 0 failed** (190.14s).
- The two documented tree-chevron baselines
  (`test_watchlists_tree_chevron_shares_a_row_with_its_watchlist[size0]`/`[size1]` in
  `Tests/UI/test_destination_visual_parity_correction.py`) reconfirmed failing in isolation, same
  assertion as task 4's report (`assert 4 > 5`, the source row's indent). Not in any of the three
  directories/files asked for above, so they do not appear in those runs; unrelated to this fix
  wave, pre-existing.

All six mutation checks performed with an editor (add/remove one call or clause), confirmed RED,
then reverted; `git status --short` clean of any mutation artifact between each.

## Files

- `tldw_chatbook/Subscriptions/briefing_service.py` (fix 1)
- `tldw_chatbook/Subscriptions/briefing_selection.py` (fix 2)
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (fixes 3, 4)
- `Tests/Subscriptions/test_briefing_service.py` (fix 1)
- `Tests/Subscriptions/test_briefing_selection.py` (fix 2)
- `Tests/Watchlists/test_watchlists_artifacts_pane.py` (fixes 3, 4)
- `Tests/Watchlists/test_watchlists_items_pane.py` (fix 5)
- `Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md` (fix 6)
- `backlog/tasks/task-1540 - Watchlists-briefings-spec-2-programme-tracking.md` (fix 6)

## Qodo round 1 (PR #1115)

Five findings addressed on `docs/spec-2-watchlists-briefings` from HEAD `81a29fdc9`. Two other
findings (streaming, markdown sanitization) were declined by the controller and left untouched.

### FIX A (bug) — `_load_briefings` blocked the event loop

`db.list_briefings(watchlist_id)` inside `WatchlistsCollectionsScreen._load_briefings`
(`watchlists_collections_screen.py:3117-3119`) ran synchronously inside the `wl-briefings-load`
worker coroutine — `run_worker` only schedules a coroutine back onto the same event loop, it does
not get blocking work off it. Audited the rest of the method for other synchronous DB calls: the
zombie sweep immediately above it already goes through `_fail_interrupted_briefings_if_safe`
(`asyncio.to_thread`), so `list_briefings` was the only offender. Wrapped it the same way:
`rows = await asyncio.to_thread(db.list_briefings, watchlist_id)`. `self.is_mounted` (the guard
this method already uses after both awaits) is unchanged — no new UI mutation was inserted before
it, so its semantics are undisturbed.

**New test:** `test_the_briefings_list_read_runs_off_the_event_loop_thread`
(`Tests/Watchlists/test_watchlists_artifacts_pane.py`), same pattern as
`test_the_queue_write_runs_off_the_event_loop_thread` in `Tests/UI/test_watchlists_inspector.py`
— spies on `db.list_briefings`, records `threading.get_ident()` on every call, asserts none of
them match the event-loop thread's id.

**Mutation:** reverted `await asyncio.to_thread(db.list_briefings, watchlist_id)` to a direct
`db.list_briefings(watchlist_id)` call. RED: `AssertionError` on `read_thread_ids` being empty
(no error above surfaced instead — the point is the assertion path, not a crash). Restored;
pin test green again.

**Full-suite fallout:** the real thread hop this fix adds exposed two existing tests that
asserted post-reload state after only a short *fixed* `pilot.pause(...)`, rather than polling —
`test_a_zombie_generating_row_is_recovered_on_a_plain_artifacts_load` and
`test_moving_the_tree_scope_moves_what_artifacts_is_about`
(`Tests/Watchlists/test_watchlists_artifacts_pane.py`). Both passed in isolation but failed
intermittently in the full 481-test run, where thread-pool contention pushed the
`asyncio.to_thread` dispatch past the fixed wait. Both passed before this branch because the read
they depend on ran synchronously in the same tick. Changed both to poll (10s deadline,
`pilot.pause(0.05)` steps) for the actual expected state instead of trusting a fixed pause,
matching the poll-loop pattern `_press_generate` already uses in the same file. Confirmed green
individually, as a file (17 passed), and in the full combined run below.

### FIX B (bug) — unbounded `NOT IN` placeholders in the window query

`_window_rows`/`_window_count` (`briefing_selection.py`) excluded featured ids with one bound `?`
per id (`i.id NOT IN (?,?,?...)`). In `auto_featured` mode `featured` is the entire
curated/queued set for the watchlist — a heavy user's backlog could bind hundreds of placeholders,
risking SQLite's host-parameter limit and breaking generation outright.

Replaced the id enumeration with the exact predicate `_curated_rows` already uses to define
"is featured": `queued_for_briefing = 1 AND NOT EXISTS (... briefing_items/briefings joined on
status IN ('complete','empty') ...)`. Factored that `NOT EXISTS` fragment out to a shared
module-level literal, `_NOT_COVERED_BY_THIS_WATCHLIST`, reused verbatim by both `_curated_rows`
(selects rows where it's true — er, false, i.e. NOT covered) and the window queries' new
featured-exclusion (`AND NOT (queued_for_briefing = 1 AND <same text>)`), so the two definitions
of "already curated" cannot drift apart. `_window_rows`/`_window_count` now take
`exclude_featured: bool` instead of `exclude_ids: Sequence[int]`; the caller in
`select_briefing_items` passes `mode == MODE_AUTO_FEATURED`, which is exactly the condition under
which `featured_ids` was previously non-empty. Bound parameter count per query is now fixed
(one extra `watchlist_id`) regardless of queue size. All 24 pre-existing pinned tests in
`Tests/Subscriptions/test_briefing_selection.py` pass unmodified — overflow_count,
covers_through_item_id, featured survival under the cap, ordering, and the curated-mode watermark
echo are all untouched by this change (confirmed by re-running the file both before and after).

**New test:** `test_window_query_parameter_count_does_not_scale_with_queue_size` — seeds 60 queued
items, wraps the thread-local `db._local.conn` with `unittest.mock.Mock(wraps=real_conn)` so every
SQL statement `select_briefing_items` issues is visible, and asserts the largest bound-parameter
count across all of them stays under 10 (fixed), never anywhere near the 60-item queue.

**Mutation:** restored a per-id `NOT IN` enumeration (60 placeholders) inside `_window_count`
(the call that always runs regardless of `remaining_cap`, unlike `_window_rows`, which
`select_briefing_items` skips entirely once the featured side alone fills the cap — the first
mutation attempt only touched `_window_rows` and did not RED, exactly because of that skip; moving
it to `_window_count` reproduced the real bug shape). RED: `assert 63 < 10` (a query bound with 63
parameters against a 60-item queue). Restored; all 25 tests in the file green again.

### FIX C — wrapped three new reads in `transaction()`

`get_briefing`, `list_briefings`, `latest_completed_watermark` (`Subscriptions_DB.py:1663-1692`)
executed directly on `self.conn`, bypassing the shared `transaction()` context manager the repo's
compliance rule requires for all DB operations. Wrapped each in `with self.transaction() as conn:`.
Line 911's pre-existing bare read (`get_watchlist_item_counts`) was left untouched, as directed —
out of scope for this PR. No nested-transaction hazard: none of these three methods are called
from inside another open `with db.transaction():` block anywhere in the callers audited
(`briefing_service.py`'s `_start_generation`/`_finish_empty`/`_finish_success` call them
sequentially, never nested).

### FIX D — `update_briefing` identifiers through `sql_validation`

`update_briefing` (`Subscriptions_DB.py:1618-1661`) validated SET-clause column names only against
its local allowlist. Read `sql_validation.py`'s `validate_column_name`/`validate_identifier`
first: `validate_column_name(key, table_name="briefings")` would fail closed unconditionally,
since neither `subscriptions`/`briefings` is a registered `db_type`/table anywhere in that
module's `VALID_TABLES`/`VALID_COLUMNS` maps (`sql_validation.py` only knows the chachanotes/
media/prompts schemas; `SubscriptionsDB` has never used this module before). Registering a new
`briefings` table+column map there was out of scope for a 5-finding review round and would touch
a validation module shared by three unrelated DB classes. Used `validate_identifier(key, "column
name")` instead, per the finding's own fallback guidance — the allowlist stays the primary API
contract (unknown key still raises `ValueError` naming the key), with `validate_identifier` as an
additional belt-and-suspenders check against a future allowlist edit introducing something unsafe
(a reserved keyword, invalid characters).

**New test:** `test_update_briefing_also_enforces_sql_validation_not_just_the_allowlist` — every
real allowlisted column already passes `validate_identifier`, so the only way to prove the second
gate is load-bearing (not decorative dead code shadowed by the allowlist) is to force a divergence:
monkeypatches `Subscriptions_DB.validate_identifier` to always return `False`, then asserts
`update_briefing(b, status="complete")` — `status` being a perfectly valid allowlisted column —
still raises `ValueError`.

**Mutation:** removed the `validate_identifier` call from `update_briefing`. RED: `Failed: DID NOT
RAISE ValueError`. Restored; test green again.

### FIX E — Google-style docstrings

All six new public `Subscriptions_DB.py` methods from this branch (confirmed exhaustively via
`git diff` against the pre-branch commit: `set_item_briefing_queued`, `insert_briefing`,
`update_briefing`, `get_briefing`, `list_briefings`, `latest_completed_watermark`) now carry
Args:/Returns:/Raises: sections as applicable, matching the file's existing convention
(e.g. `_index_unindexed_fts_batch`'s docstring). Existing load-bearing prose — the watermark's
`failed`-exclusion invariant, the queue flag's global/ADR-018 semantics, `update_briefing`'s
allowlist rationale — was kept and folded into the fuller docstrings, not replaced.
`fail_interrupted_briefings` (`briefing_service.py`) already had a complete Google-style docstring
from task 4 and needed no change; it is not a `Subscriptions_DB.py` method.

### Verification

- `Tests/Subscriptions/` + `Tests/Watchlists/` + `Tests/UI/test_watchlists_inspector.py`, one
  combined foreground run: **481 passed, 0 failed** (374.11s / 0:06:14) — the 478-test baseline
  plus the 3 new tests this round added (FIX A, FIX B, FIX D).
- `Tests/Watchlists/test_watchlists_artifacts_pane.py` alone: **17 passed** (41.89s), confirming
  the two timing fixes hold outside full-suite contention too.
- All three documented mutations (FIX A, FIX B, FIX D) confirmed RED on the intended assertion,
  then restored via `Edit` (no `git checkout`/`restore` used); `git status --short` showed only
  the five intended source/test files between each mutation and its restore, never a stray diff.
- FIX C and FIX E were documentation/compliance-only changes (transaction wrapping, docstrings) —
  no new behavior to mutate; covered by the same 481-test run passing.

### Files (this round)

- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (FIX A)
- `tldw_chatbook/Subscriptions/briefing_selection.py` (FIX B)
- `tldw_chatbook/DB/Subscriptions_DB.py` (FIX C, D, E)
- `Tests/Watchlists/test_watchlists_artifacts_pane.py` (FIX A pin test + two timing-robustness
  fixes)
- `Tests/Subscriptions/test_briefing_selection.py` (FIX B and FIX D pin tests)
