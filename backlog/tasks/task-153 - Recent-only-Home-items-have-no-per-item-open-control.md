---
id: TASK-153
title: Recent-only Home items have no per-item open control
status: Done
assignee: []
created_date: '2026-07-11 22:01'
updated_date: '2026-07-12 01:08'
labels:
  - follow-up
  - home
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A Home item that appears only in the Recent feed (e.g. a done Library import, a chatbook artifact) has no per-item open control on its canvas: build_home_controls (dashboard_state.py:~400) only emits home-open-details when an approval/active/running/paused/failed COUNT is non-zero, which a recent-only item never bumps. The item is visible but not openable from its own canvas. Pre-existing; H1 done-imports inherited it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A selected recent-only Home item exposes an open/details control on its canvas
- [ ] #2 No crash when the control is invoked for a recent item
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in the correctness batch (branch claude/followups-correctness). See Docs/superpowers/plans/2026-07-11-followups-correctness.md.
<!-- SECTION:NOTES:END -->
