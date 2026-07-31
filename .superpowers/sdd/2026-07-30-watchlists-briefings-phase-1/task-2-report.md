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

---

# Fix round 1

Three code changes, four new tests. Runs: `test_briefing_selection.py` **22 passed**;
`Tests/Subscriptions/` **212 passed in 84.03s**.

## 1 (Critical) — `curated` no longer advances the watermark

Adjudicated against what I shipped, and the review is right: `covers_through_item_id` is the
*window's* line (spec §Generation-pipeline 3), and curated is defined as selecting "regardless of
the window". A mode that never reads the window must not move it. The reviewer's scenario is the
proof — a month of curated briefings walks the line to 37, then a switch to `auto_featured` loses
36 items with `overflow_count == 0`, nothing in any body, no status, no log. My original reasoning
("it follows from the recorded contract") mistook the contract's letter for its purpose.

`select_briefing_items` now echoes the prior watermark in `MODE_CURATED` (an echo, not `None`, so
a future consumer reading the latest row rather than a `MAX` still sees the right line). The
docstring's consequence section was rewritten to the four honest statements: curated never moves
the window; a mode switch delivers the backlog capped with the overflow note; an already-curated
item may re-appear once through the auto leg (redundant, never lossy); a since-inception-curated
watchlist gets the ordinary 7-day first-briefing rule on switch.

New tests — and the absence of these was exactly why the Critical shipped:
- `test_curated_selection_echoes_the_prior_watermark`
- `test_switching_off_curated_still_delivers_the_accumulated_window` — the scenario end to end,
  three curated briefings *recorded back* the way the service will, then the switch.

## 2 (Important) — junction exclusion now has a status allowlist

`AND b.status IN ('complete', 'empty')`. Only a briefing that reached the user excludes an item
from curation. Writing junction rows before the LLM call is a natural way to implement generation,
so a `failed` briefing plausibly leaves junction rows behind — and those rows were burying the
queued item forever. A positive allowlist rather than `!= 'failed'` so a zombie `generating` row
(crashed worker, TASK-1090's shape) is covered by the same rule.

New test `test_a_failed_briefings_junction_rows_do_not_bury_a_queued_item`: seeds a `failed` AND a
`generating` briefing, each with junction rows for the queued item, asserts it is still curated —
then completes a real briefing and asserts it now IS excluded, so the fix cannot pass by simply
disabling the exclusion.

## 3 (Minor) — `covers_from_ts` compares normalized timestamps

The queries now select `datetime(i.created_at) AS created_at_utc` and `_covers_from` minimizes
over that. No extra query and no `IN (...)` variable-limit hazard on a large window.
`normalize_watchlist_item` builds an explicit dict, so the extra column does not leak into `items`.
New test `test_covers_from_ts_compares_normalized_timestamps` uses a same-day pair where the raw
string order inverts the real order (`'2026-07-25 09:00:00'` sorts below
`'2026-07-25T08:00:00+00:00'` on the space-versus-`T`).

## Mutation checks (restored by editor)

| Mutation | Observed |
|---|---|
| curated returns max-considered again | RED ×2: `AssertionError: assert 3 == 1` (`test_curated_selection_echoes_the_prior_watermark`) and `assert 5 == 1` (`test_switching_off_curated_...`, at `latest_completed_watermark(watchlist) == first` — the curated briefings had walked the line to the queued item). 2 failed, 20 passed. |
| status allowlist dropped (`AND 1 = 1`) | RED: `assert [] == [1]` in `test_a_failed_briefings_junction_rows_do_not_bury_a_queued_item`. 1 failed, 21 passed. |

Round-1's mutation 2 (junction scoping) was re-run against the rewritten `NOT EXISTS` to confirm
the added status predicate did not weaken it: still RED, `assert [2] == [1, 2]`, 1 failed/21 passed.
