---
id: TASK-15776
title: 'Watchlists: collapse _ArticleRow/_DayHeader from a ListItem-wrapping-a-Static into one widget'
status: To Do
assignee: []
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

- [ ] `_ArticleRow`/`_DayHeader` render as a single self-rendering `ListItem`
      instead of a `ListItem` wrapping a child `Static`, removing ~half the
      feed's mounted widgets
- [ ] Every `.article-row`/`.article-day-header`/`ListItem > Static` CSS
      selector affected by the change is audited and updated; visual
      appearance is unchanged
- [ ] `_repaint_row`, in-place filtering (task-15460), `j`/`k` cursor
      skipping, and selection/highlight styling all keep their current
      behavior (tests)
- [ ] A measured before/after on a 100-item feed shows the predicted
      ~15-18% reduction in screen-push cost, recorded in the task notes
