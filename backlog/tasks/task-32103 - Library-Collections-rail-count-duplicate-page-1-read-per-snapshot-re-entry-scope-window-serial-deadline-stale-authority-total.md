---
id: TASK-32103
title: >-
  Library Collections rail count: duplicate page-1 read per snapshot, re-entry
  scope window, serial deadline, stale authority total
status: To Do
assignee: []
created_date: '2026-09-08 22:42'
labels:
  - library
  - collections
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the task-32057 reviews (PR #2525): `get_library_user_content_evidence` already reads `list_page(page=1).total` in the same snapshot pass as the new count prefetch (two HTTP round trips per snapshot in server mode, only one behind the deadline); re-entering a previously scoped Collections canvas (scope persists, page reset by unmount) paints the unfiltered prefetch for one load window although the guide says it no longer flashes; the count read is awaited serially before the gather (worst case 5 s + 5 s) and a count timeout degrades silently; after `deactivate()` without re-adopt the controller-state path still paints the previous authority's `exact_total`. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One page-1 read per snapshot pass feeds both the evidence owner and the rail count
- [ ] #2 The rail never paints an unfiltered total while a scoped page is pending, on every path
- [ ] #3 A count timeout is visible (deadline sentence or Retry), and the previous authority's total is never painted
<!-- AC:END -->
