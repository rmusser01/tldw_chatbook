# Task 2 report — selection: modes, watermark, caps

**Status:** complete. `tldw_chatbook/Subscriptions/briefing_selection.py` (new, read-only);
`Tests/Subscriptions/test_briefing_selection.py` extended with 10 selection tests.

## What shipped

`select_briefing_items(db, watchlist_id, *, mode, item_cap=40, now=None) -> BriefingSelection`,
the interface task 3 consumes verbatim. `BriefingSelection` is a frozen dataclass:
`items` (normalized dicts, featured first then window items, each group newest-first by id),
`featured_ids` (raw item ids of the returned featured items), `overflow_count`,
`covers_through_item_id` (max id **considered**, `None` = do not advance), `covers_from_ts`.

Two SQL reads, both joined through `watchlist_sources`:

- **window** — `i.id > watermark` when `latest_completed_watermark` returns one; otherwise the
  first-briefing fallback, `datetime(i.created_at) >= datetime(now - 7d)` with `now` **injected**.
  `datetime()` on both sides is load-bearing: rows written with an offset-bearing ISO timestamp
  and rows written by `CURRENT_TIMESTAMP` are different *strings* for the same instant, and a raw
  comparison sorts on the `T` versus the space. Ordering is `i.id DESC` in both branches — ids are
  the tiebreaker `created_at`'s one-second resolution cannot provide.
- **curated** — `queued_for_briefing = 1 AND NOT EXISTS (briefing_items ⋈ briefings WHERE
  b.watchlist_id = ?)`. The scoping through `briefings.watchlist_id` is the whole global-flag rule:
  only *this* watchlist's briefings exclude an item.

Modes: `auto` = window only; `curated` = curated only, window never computed, nothing featured
(the brief's "marked featured in the latter" makes featured an `auto_featured` property);
`auto_featured` = union with the queued leg featured and de-duplicated out of the auto leg.

Cap: featured are taken first, the auto leg gets what is left, so the cap squeezes auto first and
only overflows featured when featured alone exceed the cap. `overflow_count` counts both.

Nothing is written — no `briefings` row, no junction row. Selection can be called speculatively
(preview, dry run, test) without changing what it answers next time.

## Two consequences stated rather than left to be discovered

Both are in the module docstring:

1. `covers_through_item_id` includes items the cap dropped. They are not lost — they are counted
   in `overflow_count` and the body states the overflow — so they were reported, and re-selecting
   them would duplicate coverage. Test `test_overflowed_items_still_advance_the_watermark` pins it.
2. In `curated` mode the window is never computed, so the watermark advances to the newest
   *queued* item — a curated briefing can step the coverage line past window items it did not
   include. That follows from the recorded contract ("max id considered") and matches curated
   intent. `latest_completed_watermark` takes a MAX, so it can only move forward, never rewind.

## Tests (10 new, all seeded through the real DB)

`WatchlistBundleService.create` / `add_source`, `db.add_subscription`, `persist_subscription_item`,
`db.set_item_briefing_queued`, and hand-written junction rows (the service owns those writes, so a
test needing prior coverage creates it the way the service will).

- `test_watermark_window_excludes_a_late_added_sources_backlog` — the backlog rows are the
  **newest by timestamp and the oldest by id**, set deliberately against each other.
- `test_failed_briefing_does_not_advance_selection`
- `test_queued_items_bypass_the_window_in_both_modes` — curated + auto_featured yes, auto no.
- `test_curated_excludes_items_this_watchlist_already_covered` — and V still selects it.
- `test_overflow_counts_dropped_items_and_features_survive_the_cap` — cap 3, 2 featured + 5 window,
  exact identities `[queued_new, queued_old, window[-1]]`, `overflow_count == 4`.
- `test_overflowed_items_still_advance_the_watermark` — featured fill the cap, so the max *kept*
  id is strictly below the max *considered* id.
- `test_first_window_is_the_last_seven_days_by_created_at` — 8 days out, 6 days in, injected `now`.
- `test_first_window_orders_same_second_ties_by_id`
- `test_empty_window_returns_none_watermark_and_does_not_advance`
- `test_unknown_mode_is_rejected_by_name`

Runs: `Tests/Subscriptions/test_briefing_selection.py` **18 passed**;
`Tests/Subscriptions/` **208 passed in 57.20s**.

## Mutation checks (each restored by editor afterwards)

| # | Mutation | Observed |
|---|---|---|
| 1 | `i.id > ?` → `datetime(i.created_at) > datetime((SELECT created_at ... WHERE id = ?))` | RED: `assert [5, 3, 2, 1] == [5]` in `test_watermark_window_excludes_a_late_added_sources_backlog` — all three backlog items flooded in. 1 failed, 17 passed. |
| 2 | junction scoping dropped (`b.watchlist_id = ?` → `? IS NOT NULL`) | RED: `assert [2] == [1, 2]` in `test_curated_excludes_items_this_watchlist_already_covered` — W's briefing consumed V's curation. 1 failed, 17 passed. |
| 3 | cap drops featured first (auto sliced first) | RED ×2: `assert [7, 6, 5] == [2, 1, 7]` (overflow test) and `assert [5, 4] == [3, 2]`. 2 failed, 16 passed. |
| 4 | `covers_through_item_id` = max **kept** id | RED: `AssertionError: assert 3 == 5` in `test_overflowed_items_still_advance_the_watermark`. 1 failed, 17 passed. |

Mutation 4 needed the extra test named in the brief's follow-on: with "newest win the cap", the
max kept id equals the max considered id in every ordinary case, so the mutation is invisible
until featured items fill the cap and push the highest-id window items out.
