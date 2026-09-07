# Task 5 — queue-for-briefing affordance

Spec #2 phase 1 (`Docs/superpowers/specs/2026-07-30-watchlists-briefings-design.md`,
§UI + "The queue flag is global, and never auto-cleared"). Tasks 1-4 built the
tables, selection, generation service and the Artifacts surface. This task
gives the user the OTHER half of curated/auto_featured selection: a way to
put an item in the queue at all.

## What shipped

**`inspector_pane.py`.** `ToggleBriefingQueueRequested(item_id, queued)` —
posted from `on_button_pressed`, following `SaveNoiseSelectorsRequested`'s
exact shape (verb-first message class, one screen `@on` handler). `item_id`
is deliberately the item's raw database row id (`entity["item_id"]`), not the
namespaced `entity["id"]` `SaveNoiseSelectorsRequested.source_id` carries:
`SubscriptionsDB.set_item_briefing_queued` takes the raw int directly, the
flag is local-only (no server form to resolve), so there is nothing to parse
first. `queued` is always the FLIP of the entity's current value — one button
both queues and unqueues, labelled by which it is about to do
("Queue for briefing" / "Unqueue from briefing"). The button renders only for
`deepest.kind == "item"` — the same discriminator Ingest/Ignore already use,
next to them — reusing the whole-branch-review "no silent no-op" idiom for an
unselected/id-less entity.

**`items_pane.py`.** A fifth `DataTable` column, "Queued", rendered as a
single app-controlled glyph (`●`/``) from `item.get("queued_for_briefing")` at
`compose()` time — never item-derived text, so the pre-existing note that
`DataTable` cells markup-parse `str` content cannot bite here.
`update_item_queued_cell(item_id, queued)` mirrors `update_item_status_cell`
exactly: same row-key convention (the entity's own `id`, not the raw DB row
id — a caller holding only the raw id resolves it first, same as
`_repaint_item_status_cell` already has to), same `CellDoesNotExist`
degradation, same "never recompose" contract.

**`watchlists_collections_screen.py`.** `handle_toggle_briefing_queue_requested`
writes through `_briefings_db()` — the SAME accessor Task 4 built for the
Artifacts pane's own local-only writes (`WatchlistBundleService`, not the
scope service) — **synchronously, not a worker**: `set_item_briefing_queued`
is a single indexed `UPDATE ... WHERE id = ?`, the same order of cost as any
other UI-thread state write on this screen, so a worker would only add a
scheduling round trip. On success, `_patch_item_queued_flag` patches every
in-memory dict describing the item (`_loaded_items`, `selected_entity`,
`_selected_content_item`) in place, repaints the Items-table cell via
`update_item_queued_cell`, and — new relative to the noise-selectors
precedent — relabels the mounted Inspector's own button directly
(`button.label = ...`) if it is currently showing this same item, since
Requirement 2 needs the button's own label to flip back without a recompose.
None of this touches a `recompose=True` reactive assignment, so no
full-screen (or even Inspector-level) rebuild happens; on failure, nothing is
patched, nothing is repainted, and an error toast reports it. The log line
names the exception TYPE only (`type(exc).__name__`), never
`logger.opt(exception=True)` — Task 3's diagnose-sink leak, one layer up.

## Tests

`Tests/UI/test_watchlists_inspector.py` (+7), `Tests/Watchlists/test_watchlists_items_pane.py` (+2).

1. `test_pressing_queue_for_briefing_writes_the_flag_and_repaints_the_row` —
   presses the real button on a real seeded item; a fresh `db.get_new_items`
   read confirms the flag flipped; the Items-table cell repaints; the button
   relabels; the SAME `ItemsPane`/`DataTable`/`InspectorPane` instances
   survive (Phase D pattern).
2. `test_pressing_queue_for_briefing_again_unqueues_and_relabels` — the same
   button pressed twice: flag back to `False`, indicator cleared, label back
   to "Queue for briefing" — not a one-way ratchet.
3. `test_the_queue_button_only_renders_for_item_selections` — every OTHER
   selectable kind (`OTHER_ENTITIES`, imported from
   `test_watchlists_item_actions.py` so this cannot be fixed for items by
   breaking one of those fixtures: source/run/rule/notification), plus a real
   item (`REAL_ITEM`) as the positive control.
4. `test_a_failed_queue_write_leaves_the_flag_and_indicator_unchanged` —
   `db.set_item_briefing_queued` raises; flag unchanged (fresh read), cell
   unchanged, button unrelabeled, error toast fired, `screen.is_attached`
   (nothing escaped the handler).
5. `test_the_queued_indicator_renders_from_the_normalized_flag_on_reload` —
   an item pre-queued directly through the DB shows the glyph after a plain
   load, through the real controller — the read path (Task 1) end to end,
   no button press anywhere in the test.

Plus two widget-level tests in `Tests/Watchlists/test_watchlists_items_pane.py`
against a bare `ItemsPane` (no screen): the column renders from the
normalized flag, and `update_item_queued_cell` repaints without recomposing
the pane.

### A test-harness trap this task hit (worth recording)

The Items pane's real load path (`_load_items` → the controller →
`LocalWatchlistsService`) and `_briefings_db()`
(`WatchlistsCollectionsScreen`'s queue-write accessor, via
`WatchlistBundleService`) resolve to the SAME on-disk file in the running
app, but to two DIFFERENT temp files inside `_build_test_app()`:
`get_subscriptions_db_path` is patched only for the duration of
`TldwCli.__init__`, so `WatchlistBundleService`'s EAGER connection (built
inside that init, patch still live) and `LocalWatchlistsService`'s LAZY
per-call factory connection (built later, patch already exited) diverge.
Tests here seed through `local_watchlists_service._db()` (so the item
reaches the real Items pane) and then point `watchlist_bundle_service._db`
at that SAME connection — but only AFTER the screen's initial mount has
settled (`pane.items` populated). Doing the reassignment BEFORE mounting let
the screen's own concurrent startup loads (each constructing a fresh
`SubscriptionsDB` against the same brand-new file) race the one-time schema
migration gate on the shared connection's cached schema view: observed
directly as `OperationalError: no such table: subscription_items` on the
very next write, self-healing on an immediate retry — proof it was a startup
race, not a real absence of the table. Waiting for the settle removes the
race instead of masking it with a retry loop.

## Mutation checks

| # | Mutation | Result |
|---|----------|--------|
| a | handler body replaced with `event.stop(); return` (no-op) | RED ×3: `test_pressing_queue_for_briefing_writes_the_flag_and_repaints_the_row` (`the press must reach SubscriptionsDB.set_item_briefing_queued`), `test_pressing_queue_for_briefing_again_unqueues_and_relabels` (`precondition: first press queued it`), `test_a_failed_queue_write_leaves_the_flag_and_indicator_unchanged` (`the failure must be visible, not a silent no-op`). Discriminator (3) and read-path (5) tests are unaffected, as expected. |
| b | `self._patch_item_queued_flag(...)` call commented out, DB write left running | RED ×2, and layered exactly as specified: test 1 fails at `the row indicator must repaint in place once the write succeeds` — the PRECEDING `_queued_flag(db, item_id) is True` assertion (the DB half) passed. Test 2 fails at `assert '' == '●'` (the indicator), same shape. |
| c | `yield self._queue_briefing_button(deepest.entity or {})` moved unconditionally to the top of the actions block | RED: `test_the_queue_button_only_renders_for_item_selections` fails at `a source selection must not offer the briefing queue toggle`. (Tests 1/2/4 also failed on a duplicate-widget-ID mount error, an artifact of the button now rendering twice for items — not the assertion under test.) |

All three mutations applied with `Edit`, verified, and restored with `Edit`
(never `git checkout --`).

## Test runs

- `Tests/UI/test_watchlists_inspector.py` — **33 passed** (26 → 33, +7 new).
- `Tests/Watchlists/test_watchlists_items_pane.py` — **7 passed** (5 → 7, +2 new).
- `Tests/UI/test_watchlists_inspector.py` + `Tests/Watchlists/test_watchlists_items_pane.py`
  + `Tests/UI/test_watchlists_item_actions.py` + `Tests/UI/test_watchlists_read_status.py`
  + `Tests/Watchlists/test_watchlists_collections_screen.py`
  + `Tests/Watchlists/test_watchlists_artifacts_pane.py` — **90 passed** (no regressions
  in the sibling suites this task's read/write paths touch).
- `Tests/Watchlists/` (full directory) — **215 passed**.

## Sync vs. worker

**Synchronous**, in the `@on` handler itself. `SubscriptionsDB.set_item_briefing_queued`
is one indexed `UPDATE subscription_items SET queued_for_briefing = ? WHERE id = ?`
— the same cost class as any other UI-thread state read/write already done
synchronously on this screen (e.g. `_briefings_db()`/`_can_generate_briefing()`)
— so routing it through `run_worker` would add a scheduling round trip for a
call that is already fast and already off any network/LLM path.

## Files

- `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py`
- `tldw_chatbook/UI/Watchlists_Modules/items_pane.py`
- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- `Tests/UI/test_watchlists_inspector.py`
- `Tests/Watchlists/test_watchlists_items_pane.py`

---

# Fix round 1

One Important, three Minors, all addressed.

## Important — the write no longer runs on the UI thread

`SubscriptionsDB.set_item_briefing_queued` is a transactional
`UPDATE subscription_items ... WHERE id = ?` (`Subscriptions_DB.py:1603`,
`with self.transaction()`), no busy timeout configured beyond SQLite's
default. It ran synchronously inside `handle_toggle_briefing_queue_requested`
-- the same shape Task 4 ruled off the UI thread hours earlier
(`f859f9434`, `_sweep_and_guard`) for the identical reason: this branch's own
docstrings admit a second app instance against the same database file, and a
contended write would block the event loop for up to five seconds before
raising.

The handler now does only what the UI thread is entitled to do -- answer the
no-selection / no-database cases from memory (`event.item_id is None`,
`db is None`, both read-only) -- and dispatches:

```python
self.run_worker(
    self._toggle_briefing_queue(db, event.item_id, event.queued),
    group="wl-queue-toggle",
)
```

`_toggle_briefing_queue` (new) is the worker body, following Task 4's
`handle_generate_briefing_requested` -> `_generate_briefing` shape exactly:
`await asyncio.to_thread(db.set_item_briefing_queued, item_id, queued)`,
then -- write-first, patch-after -- `_patch_item_queued_flag` plus the
targeted cell repaint on success; on an exception, nothing is patched,
nothing is repainted, and an error toast fires, matching the pre-existing
"indicator unchanged on DB raise" contract exactly. `self.is_attached` is
checked before every UI mutation after the `await` (both on the exception
path and the success path), mirroring `_generate_briefing`'s own guard,
since the screen can be popped while the write is in flight. The log line
still names the exception TYPE only (`type(exc).__name__}`), never
`logger.opt(exception=True)`.

**`asyncio.to_thread` is the load-bearing part, not `run_worker` by
itself** -- `run_worker` only schedules a coroutine back onto the same event
loop; `WatchlistsBackendController._maybe_await`
(`tldw_chatbook/UI/Watchlists_Modules/watchlists_backend_controller.py:29`)
has no `to_thread` either, and this handler does not go through the
controller at all (it reaches `SubscriptionsDB` directly via
`_briefings_db()`, same as before). Mutation (a) below pins this seam with a
dedicated test, since every *outcome* test still passes with `to_thread`
mutated away.

### Worker/dedup choice

**No `exclusive=True`, no dedup at all** -- a shared, non-serializing
`group="wl-queue-toggle"` only so these workers are nameable together (e.g.
for a future shutdown-drain), not to collapse them. Reasoning: the write is
a single-row idempotent `UPDATE`, so two overlapping writes to the SAME item
are safe to interleave (last write wins, which is exactly what two rapid
presses on one row mean); two presses on DIFFERENT items must never cancel
each other, and `exclusive=True` cancels whatever else is running under the
same group regardless of which row it touches -- Task 4's own lesson
(`exclusive=True` cancelling an in-flight run manufactures zombie state)
applies here too. A per-item dedup key was considered and rejected as
unnecessary complexity for a write this cheap and this safe to repeat.

## Minor 1 -- the mislabelled docstring

`Tests/UI/test_watchlists_inspector.py`'s
`test_a_failed_queue_write_leaves_the_flag_and_indicator_unchanged` docstring
claimed "the DB write is left to actually run... only the LOG side is
replaced," which was never true -- the test replaces
`db.set_item_briefing_queued` itself with a raising `Mock`. Rewritten to say
exactly that: the write never reaches the real database, and `_queued_flag`
confirms the stored flag stayed exactly as seeded.

## Minor 2 -- the misnamed test

`test_the_queued_indicator_renders_from_the_normalized_flag_on_reload` ->
`test_the_queued_indicator_renders_from_the_normalized_flag_on_load`. The
body never navigates away and back; it seeds a pre-queued item directly
through the DB and opens the screen once. Docstring updated to match ("on a
plain (first) load").

## Minor 3 -- the falsy-id guard

`_patch_item_queued_flag`'s `row_key = row_key or item.get("id")` (two call
sites) would re-assign on a falsy id (`0`/`""`), not just a missing one.
Changed to an explicit `if row_key is None: row_key = ...` guard at both
sites.

## Tests

Moving the write into a worker meant the existing settle loops needed to
wait on the worker's completion, not just the handler returning:

- **Tests 1 and 2** (`writes_the_flag_and_repaints_the_row`,
  `again_unqueues_and_relabels`) already looped on a bounded
  `pilot.pause()` condition, but the condition was the DB flag alone. Since
  the repaint is the LAST thing the worker does (strictly after the awaited
  write), the loop condition was changed to wait on the repainted cell
  instead -- a strictly stronger, still-bounded condition that also removes
  any window where the DB assertion could pass a tick before the cell
  catches up.
- **Test 4** (`a_failed_queue_write_leaves_the_flag_and_indicator_unchanged`)
  used a *fixed* 30-iteration no-op pause loop as its only carrier before
  checking the toast list -- exactly the anti-pattern this task's rules
  warn against. Changed to break early once `toasts` is non-empty, bounded
  at 60 iterations.
- **New test**: `test_the_queue_write_runs_off_the_event_loop_thread` pins
  the `asyncio.to_thread` seam directly (mutation (a) showed no existing
  test reds without it). It captures `threading.get_ident()` inside a spy
  wrapping `db.set_item_briefing_queued` and asserts it differs from the
  test's own (event-loop) thread id.

## Mutation checks (fix round 1)

| # | Mutation | Result |
|---|----------|--------|
| a | `asyncio.to_thread(...)` replaced with a direct synchronous call | RED, **only** the new `test_the_queue_write_runs_off_the_event_loop_thread` (`AssertionError: set_item_briefing_queued must run off the event-loop thread...`). All 5 other queue tests stay GREEN -- the end state (flag set, cell repainted) is identical either way, which is exactly why a dedicated thread-identity test was needed; nothing else can distinguish "off-thread" from "on-thread but still correct." |
| b | `self._patch_item_queued_flag(...)` call removed from the success path (repaint dropped) | RED ×2, layered exactly as before the worker move: `test_pressing_queue_for_briefing_writes_the_flag_and_repaints_the_row` fails at "the row indicator must repaint in place once the write succeeds" (the preceding `_queued_flag(db, item_id) is True` assertion -- the DB half -- passes); `test_pressing_queue_for_briefing_again_unqueues_and_relabels` fails at `assert '' == '●'` on the first press's indicator check, same shape. |
| c | `yield self._queue_briefing_button(entity)` moved unconditionally to the top of `inspector-actions` | RED: `test_the_queue_button_only_renders_for_item_selections` fails at "a source selection must not offer the briefing queue toggle." The other four queue tests (which press the button on an item selection) fail too, but on a `textual.widget.MountError: Tried to insert 2 widgets with the same ID` -- a duplicate-widget artifact of the button now rendering twice for items, not the assertion under test, matching the round-0 report's own note about this exact mutation. |

All three mutations applied with `Edit` and restored with `Edit` (never
`git checkout --`); `git diff --stat` after restoration showed only the two
intentionally-changed files.

## Test runs (fix round 1)

- `Tests/UI/test_watchlists_inspector.py` -- **34 passed** (33 -> 34, +1 new:
  the thread-identity pin).
- `Tests/Watchlists/` (full directory) -- **215 passed** (unchanged from
  round 0; the new test lives in `Tests/UI`, not `Tests/Watchlists`).
- `Tests/UI/test_watchlists_item_actions.py` (the file that actually shares
  fixtures with this task -- `OTHER_ENTITIES`/`REAL_ITEM`; no
  `Tests/Watchlists/test_watchlists_item_actions.py` exists in this repo)
  -- **6 passed**.
- Failure-path test (`a_failed_queue_write_leaves_the_flag_and_indicator_unchanged`)
  re-confirmed post-fix: indicator unchanged, button unrelabeled, error
  toast fired, `screen.is_attached` true (nothing escaped the worker). No
  explicit `exit_on_error` is passed to `run_worker` (matching
  `_generate_briefing`'s own call) -- safe here because the worker body
  wraps its only failure-prone call (`asyncio.to_thread(...)`) in a
  `try/except Exception`, so nothing can escape regardless of the default.

## Files (fix round 1)

- `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- `Tests/UI/test_watchlists_inspector.py`
