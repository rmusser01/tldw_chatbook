---
id: TASK-15776
title: 'Watchlists: collapse _ArticleRow/_DayHeader from a ListItem-wrapping-a-Static into one widget'
status: Done
assignee: ['@claude']
created_date: '2026-08-13 12:31'
labels:
  - perf
  - watchlists
priority: medium
---

## Description

The one structural lever task-15462's profiling investigation actually found
and handed over rather than shipping inline (input-latency burn-down).
Task-15462's cProfile + widget-count sweep of a Watchlists screen push found
the screen's own chrome is a median-cost screen (200 ms, in line with every
other screen); its entire excess is the article feed, which is 224 widgets
because `_load_items` pages at a hard-coded `limit=100` and every
`_ArticleRow`/`_DayHeader` is a `ListItem` wrapping one `Static` (112 rows x
2 widgets each).

Two independent estimates agree the win is real: the sweep's dose-response
slope (112 x ~0.55 ms/widget approx 62 ms approx 15-18% of the push) and a
prototype where the rows render the same `Text` with no child widget
(interleaved runs: paint 414/479 ms -> 335/372 ms). Not shipped inside
task-15462 because it is a structural rewrite of `article_list.py` — a file
task-15460 hardened three tasks earlier in the same programme — requiring an
audit of every `.article-row`/`.article-day-header` and `ListItem > Static`
selector across the CSS bundle, plus rerouting `_repaint_row`'s
`query_one(Static).update()`, plus re-validating in-place filtering, `j`/`k`
cursor skipping, and highlight styling.

A cheaper variant worth considering in the same task, per task-15462's own
note: `_load_items`'s hard-coded `limit=100` is not viewport-proportionate,
so a smaller page (where the viewport allows it) shrinks the same problem
without the structural rewrite.

## Acceptance Criteria

- [x] `_ArticleRow`/`_DayHeader` render as a single self-rendering `ListItem`
      instead of a `ListItem` wrapping a child `Static`, removing ~half the
      feed's mounted widgets
- [x] Every `.article-row`/`.article-day-header`/`ListItem > Static` CSS
      selector affected by the change is audited and updated; visual
      appearance is unchanged
- [x] `_repaint_row`, in-place filtering (task-15460), `j`/`k` cursor
      skipping, and selection/highlight styling all keep their current
      behavior (tests)
- [x] A measured before/after on a 100-item feed shows the predicted
      ~15-18% reduction in screen-push cost, recorded in the task notes

## Implementation Plan

1. Audit consumers at HEAD: CSS bundle sources for `.article-row` /
   `.article-day-header` / `ListItem > Static` selectors; every test and
   screen site that reaches through a row to its inner `Static`
   (`query_one(Static)`, `node.children[0]`, `pane.query(".article-row")`).
2. Born-red pin first: a test asserting each `_ArticleRow`/`_DayHeader` is a
   single childless widget (ListView subtree census == row count). Red at
   HEAD.
3. Before-measures at HEAD with a probe run under the test harness (config
   isolation via conftest): populate time for a 100-item feed (median of
   interleavable repeats), widget census, and a compositor
   `render_strips()` text+style capture of a small deterministic feed
   (including a highlighted row) saved for parity diffing.
4. Collapse: `_DayHeader`/`_ArticleRow` become self-rendering `ListItem`
   subclasses (`render()` returns the row `Text`; classes move onto the
   ListItem; `_repaint_row` routes through a new `update_content()` that
   mirrors `Static.update`'s refresh(layout=True) contract).
5. Re-anchor every consumer from the audit table; re-run the probe for
   after-numbers and byte-identical parity; run the Watchlists blast-radius
   suites + ruff.

## Implementation Notes

The collapse is exactly the task's shape: `_DayHeader` and `_ArticleRow`
are now childless self-rendering `ListItem` subclasses. Each carries its
`Text` in `_content`, returns it from `render()`, and moves the old inner
`Static`'s class (`article-day-header` / `article-row`) onto itself. A
childless `ListItem` declares no `layout`, so Textual's
`Widget.get_content_height` measures the widget's own render -- the same
wrapped-`Text` sizing the inner `Static` produced. `_repaint_row` routes
through a new `_ArticleRow.update_content()` that mirrors `Static.update`
verbatim (swap renderable, `refresh(layout=True)`, which clears the cached
content dimensions). Day-header labels are rendered as literal `Text`
where `Static(label)` used to markup-parse them -- app-derived
`day_bucket` strings, so no visible change, strictly safer.

**CSS audit**: no selector in any bundle source (`tldw_chatbook/css/`)
targets `.article-row`, `.article-day-header`, or a `ListItem > Static`
under `#items-table` -- the only row styling is the generic
`ListView ListItem` padding/hover/`-highlight` rules plus Textual's own
`ListView > ListItem` defaults, all anchored on the ListItem itself and
unaffected. Zero CSS edits needed (verified by grep over `css/**/*.tcss`
and by the byte-identical production-CSS parity capture below).

**Consumer re-anchoring** (every site that reached through a row to its
inner `Static`):
- `article_list.py::_repaint_row`: `row.query_one(Static).update(...)` ->
  `row.update_content(...)`
- `Tests/Watchlists/test_watchlists_article_list.py` (helpers + 12 sites),
  `test_watchlists_pane_filter_in_place.py` (2),
  `test_watchlists_collections_screen.py` (3):
  `node.query_one(Static).renderable` -> `node.render()` (same `Text`,
  spans included; two now-dead local `Static` imports removed)
- `Tests/UI/test_watchlists_items_status_filter.py`,
  `test_watchlists_content_pane.py`, `test_watchlists_inspector.py`:
  `node.children[0].render().plain` -> `node.render().plain`
- `Tests/Watchlists/test_watchlists_pagination.py`:
  `pane.query(".article-row")` now matches the row itself, so
  `.parent.display` -> `.display`
- The screen (`watchlists_collections_screen.py`) only uses the pane API
  and needed no change.

**Born-red pin**:
`test_rows_and_headers_are_single_widgets_with_no_children` (ListView
subtree census == row count, every node childless) -- failed at HEAD
`99ecb5890`, green after.

**Measured before/after** (100-article feed in 12 day buckets = 112 rows,
the audit's shape; median of 21 fresh-app populate runs,
`pane.items = <100 items>` -> idle, Textual pilot at 120x40):
- widgets in the ListView subtree: 224 -> 112 (exactly half)
- populate path: 144.61 ms -> 78.28 ms (-66.3 ms, -45.9% of the pane
  build). The absolute saving matches task-15462's dose-response
  prediction (~62 ms), which as a share of the audit's measured 414-479 ms
  screen push is the predicted ~15-18%; on the pane's own populate path it
  is a far larger share, since that path is mostly widgets.

**Rendered parity**: normalized per-cell (char, style) captures of
`compositor.render_strips()` for a deterministic 3-item feed
(unread+star, reviewed+queued, ingested; two day headers), in both the
plain and the focused+highlighted state, under BOTH the default-CSS
harness and the production-bundle harness
(`ProductionCssArticleListHarness`): all four captures byte-identical
before vs after.

**Tests**: `Tests/Watchlists/` 668 passed;
the nine `Tests/UI/test_watchlists_*` suites 255 passed + 1 failed --
`test_watchlists_select_option_overlays.py::test_a_bordered_compact_select_keeps_its_frame_under_focus_and_hover`,
a Settings-screen Select-frame pin reproduced identically at unmodified
HEAD `99ecb5890` in a clean baseline worktree (pre-existing, unrelated).
`ruff check` clean on all touched files (the 14 `App` F401s in
`test_watchlists_content_pane.py` are pre-existing at HEAD and untouched);
`ruff format --check` drift on `article_list.py` is pre-existing at HEAD
in three untouched hunks -- none of the new code reformats.

**Files**: `tldw_chatbook/UI/Watchlists_Modules/article_list.py` plus the
seven test files listed above.
