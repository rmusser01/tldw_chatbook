---
id: TASK-23028
title: >-
  The timer census is green while blind to the two largest idle clocks in the app
status: To Do
assignee: []
created_date: '2026-08-27'
labels:
  - testing
  - observability
  - performance
priority: high
---

## Description

`Tests/Architecture/test_timer_path_static_update_inventory.py` exists to find repeating clocks. It
is **green on current dev while missing both of the largest ones**, for two independent reasons:

1. **It matches `set_interval` as an exact callee name.** `UI/Console_Modules/realtime.py:345` now
   spells a **10 Hz** clock `self._set_interval(...)` through a constructor-injected callable, so it
   left the census silently. The root count did **not** move (35 -> 35) because another root arrived
   in the same window and the two cancelled - so nothing looked wrong.
2. **It parses only `tldw_chatbook/**.py`**, and no package file assigns `auto_refresh`. The 15 Hz
   `ProgressBar` clocks (TASK-23022) are armed inside `textual/dom.py` and are structurally
   invisible.

Two further known gaps remain from the prior review: two roots resolve to nothing and nothing
notices, and the call graph cannot cross constructor-injected callables at all - which is how the
whole `UI/Console_Modules` family is wired.

## Acceptance Criteria

- [ ] A clock reached through a renamed or injected callable is either censused or **fails** the census loudly - silence is the defect
- [ ] Framework-armed clocks (`auto_refresh`, indeterminate progress) are covered, or their absence is asserted explicitly rather than implied
- [ ] An unresolvable root fails rather than being skipped
- [ ] The census is verified against the two clocks it currently misses, as regression cases
- [ ] Root-count stability is not treated as evidence of no change - this window had a net-zero count with two real changes underneath

## Evidence

Census run against three trees: 08-22 pin 35 roots / 3 unresolved; 08-24 pin 35 / 2; tip 35 / 2. The
tip diff is exactly two entries that cancel.

Source: `Docs/Design/2026-08-27-holistic-perf-review.md`.
