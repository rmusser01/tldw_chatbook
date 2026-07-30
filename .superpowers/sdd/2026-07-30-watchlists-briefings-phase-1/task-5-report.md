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
