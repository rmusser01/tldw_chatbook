---
id: TASK-1361
title: Two URL checks in the same second diff against a stale snapshot
status: To Do
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
- [ ] #1 Baseline selection is deterministic regardless of how many snapshots share a created_at value, for example by tie-breaking on the row id
- [ ] #2 A test performs two checks for one source within the same second and asserts the second diffs against the first, not against an older snapshot
- [ ] #3 The test fails if the tie-break is removed
<!-- AC:END -->
