---
id: TASK-1361
title: Two URL checks in the same second diff against a stale snapshot
status: Done
assignee: []
created_date: '2026-07-29 23:40'
labels:
  - watchlists
  - correctness
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`URLMonitor.check_url` selects the baseline to compare against with
`ORDER BY created_at DESC LIMIT 1` over `url_snapshots`, and `created_at` has one-second resolution.
If two checks for the same source land within the same second, the ordering between the snapshot just
written and the one before it is undefined, so a check can diff against a **stale** baseline.

Found while implementing TASK-1343 — it broke the first draft of a test that performed two checks in
quick succession, which is exactly the shape a retry, a manual "Check now" during a scheduled run, or
a tight test loop produces.

The consequence is a wrong `change_percentage`, a wrong diff, and possibly a spurious item: the
change is measured against the wrong "before".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Baseline selection is deterministic regardless of how many snapshots share a created_at value, for example by tie-breaking on the row id
- [x] #2 A test performs two checks for one source within the same second and asserts the second diffs against the first, not against an older snapshot
- [x] #3 The test fails if the tie-break is removed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an `id DESC` tie-break to the baseline query in `URLMonitor.check_url`
(`monitoring_engine.py:992`). `url_snapshots.id` is `INTEGER PRIMARY KEY AUTOINCREMENT`, so it is
monotonic and breaks the `created_at` tie by true insertion order. Same shape as
`Workspaces/registry_service.py:171`, which already pairs `created_at` with a second key.

**Scope: one live site.** A scan for `url_snapshots` queries with an `ORDER BY` found five. Four are
in `baseline_manager.py`, which nothing imports (TASK-1360) — including a **pruning DELETE**, so if
that module is ever adopted the same ambiguity could delete the wrong snapshots. Left alone
deliberately and recorded in a comment at the fixed site, rather than editing an orphan.

**The test forces the tie rather than racing the clock.** Two snapshots are inserted with an
identical `created_at` and different bodies, stale first, so `id` order and correct-baseline order
agree only if the tie-break is present. It asserts on the **diff**, not the percentage: the diff
names which body was treated as "before", which is the thing that was ambiguous.

Mutation-checked: removing `, id DESC` reddens
`test_two_snapshots_in_one_second_compare_against_the_newer`. `Tests/Subscriptions/` +
`Tests/Scheduling/` 342 passed.
<!-- SECTION:NOTES:END -->
