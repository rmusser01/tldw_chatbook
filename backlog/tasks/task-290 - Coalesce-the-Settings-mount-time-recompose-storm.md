---
id: TASK-290
title: Coalesce the Settings mount-time recompose storm
status: To Do
assignee: []
created_date: '2026-07-17 15:17'
labels:
  - ux
  - performance
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Settings on_mount queues two thread workers whose completions each set recompose=True reactives, causing full-screen recomposes at nondeterministic times shortly after mount (briefly blanking the DOM and forcing footer/context re-seeding). task-264 made tests deterministic with a settle helper, but the product-side storm remains. Coalesce the two refreshes into one recompose (or make the reactives non-recompose with targeted updates).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening Settings triggers at most one post-mount recompose
- [ ] #2 No visible DOM blank/flicker between mount and the workers landing
- [ ] #3 Existing settings hub suite passes without the settle helper needing extra waits
<!-- AC:END -->
