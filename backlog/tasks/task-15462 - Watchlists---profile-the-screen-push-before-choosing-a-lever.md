---
id: TASK-15462
title: Watchlists: profile the screen push before choosing a lever
status: To Do
assignee: []
created_date: '2026-08-11 12:05'
labels:
  - perf
  - watchlists
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the audit: Watchlists is the heaviest screen never profiled — 0.89 s on fast hardware, no deferral shipped, no widget survey, and `compose` runs `resolve_latest_follow_item()` inline (`watchlists_collections_screen.py:2554`). Per the owner's stability preference this is investigation-first: run the task-2725 method (widget survey + cProfile of one push) BEFORE choosing a lever — the series' headline lesson is that hidden-widget weight predicts nothing when a screen is sync/DB-bound (Schedules' 1.11 s evaporated; Console's deferral measured 4-8% and was reverted). If widget-bound, apply the established defer-past-first-paint recipe (traps banked in tasks 2725/2900/2901); if service/DB-bound, the levers are tasks 15463/15464.

Depends on: 15460/15461 landing first will change the profile — run the profile at whatever order is current and say so. Evidence and method: Docs/Design/2026-08-11-input-latency-audit.md (audit of dev 82b595049; all file:line cites verified there).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A recorded cProfile + widget survey of one Watchlists push names the top costs, committed to the task
- [ ] #2 The chosen lever (deferral, service, or none) is justified against the profile; a wrong-lever conclusion is recorded like task-2902 if applicable
- [ ] #3 If a lever ships: first-paint latency before/after plus the recipe's mechanism and integrity tests
<!-- AC:END -->
