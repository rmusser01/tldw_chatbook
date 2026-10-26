---
id: TASK-36571
title: >-
  Console: a run's config projection still spends about 50 ms of UI-loop render
  per send window
status: To Do
assignee: []
created_date: '2026-10-24 03:51'
updated_date: '2026-10-24 06:40'
labels:
  - console
  - performance
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33620.15.4 moved the Console run projection's checked-config operation and most of its database reads off the UI loop, kept the checked display warm through a run, coalesced a run start's explicit syncs into the poll's projection, and kept the send's durable commit quiet. Its AC#4 had asked for the projection's UI-loop time per send window to halve. Measured live and paired on 2026-10-23 (Anthropic claude-haiku-4-5 through the review proxy, 160x45, 30 sends per arm, dev 1381e70735 as the base), it fell from a median of 58.2 to 49.4 ms per window (-15%; warm sends 55.2 to 48.8, first sends 59.6 to 52.7). Half of dev 5b5920b8e2's 65.1 ms, the figure AC#4 started from, is 32.6 ms.

What remains is mostly the render and derivation CPU of the five to seven projections a send window needs. Mean ms per window on that branch: workspace context state 12.9, rail state 8.8, control-state push 8.7, composer actions 8.0, settings summary 6.3, the Inspector build and other unattributed work 5.5, Library activity 4.0, rail visibility 3.0, cost chip 2.4, transcript guidance 1.7. A run start has to paint each of its state changes once, so syncing less often no longer shrinks this; each projection has to cost less.

About 5 ms of it is reads that TASK-33620.15.4 keeps on the loop during a send's start (acceptance to the provider request): the Library activity rows (3.4 ms) and the Inspector's registry and appearance reads. Off the loop, each of those reads opened its own SQLite connection (`run_owned_db_call`: 34 ms wall, 10 ms CPU alone, 20-60 ms CPU in the app) and delayed first sends by ~66 ms, so they only leave the loop once an off-loop read stops costing a connection.

The layout half of the same work is tracked with TASK-33620.15.7 (PR #3211) and its follow-ups TASK-36012 (widgets that request layouts at send start and end without changing size) and TASK-35970 (the collapsed Character section recomposes twice per send). This task is the projection's own derivation and widget-update cost, measured inside the projection.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Measured first, paired against the base: the projection's UI-loop time per send window, by component, live with a real provider
- [ ] #2 The projection's UI-loop time per send window is at most 32.6 ms median (half of dev 5b5920b8e2's 65.1 ms), live, real provider, paired
- [ ] #3 A send reaches the provider no later than on the base build (Enter to the provider request, live, paired and interleaved, at least 30 sends per arm including first sends)
- [ ] #4 What the rails and controls show is never staler than one poll interval behind the base build
- [ ] #5 A deterministic test fails on the base build and pins each cheaper projection component
<!-- AC:END -->
