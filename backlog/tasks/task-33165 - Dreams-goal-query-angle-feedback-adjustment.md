---
id: TASK-33165
title: Dreams goal query-angle feedback adjustment
status: To Do
assignee: []
created_date: '2026-09-28 20:23'
labels:
  - dreams
  - phase2
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Spec feedback-loop section: feedback on a goal-derived story adjusts the QUERY ANGLE for that goal, never the goal itself. Phase 2 implemented and pinned the immunity half (goals never decay, never weight-adjusted); the query_angle adjustment mechanism (dream_interest_profile.query_angle column exists, unwritten) was never planned - filed so it is not lost, per the Phase-2 final review recommendation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Feedback on goal-derived stories records a query-angle adjustment consulted by query synthesis,Goals remain immune to weight changes (existing tests stay green)
<!-- AC:END -->
