---
id: TASK-38950
title: >-
  Console: a run tick still rebuilds the Conversations browser on about every
  other tick, because its carry key leaves inputs out
status: To Do
assignee: []
created_date: '2026-11-05 16:20'
labels:
  - console
  - performance
  - ux-review-2026-09-29
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-36571.1 stopped the projection building the workspace context (it reads the label alone) and lets a run tick reuse its predecessor's build once while a carry key matches: the TASK-22201 fingerprint, with the run status read as the run markers and attention facts the build shows. The reuse is capped at once in a row, because the key leaves inputs out (relative ages and date buckets, stars, manual unread marks, activity receipts, appearance, folder availability and its refresh age, persisted-rows and appearance cache ages, archive states, sub-agent and progress counts, collapse preferences, rail height, session character and hand-off fields, the workspace conversation search), and TASK-36571.1's AC#3 allows those one poll interval behind the base build. So the Conversations tray push still rebuilds on about every other tick of a send window, and the build time per window fell by about a third to a half rather than the half TASK-36571.1's AC#2 asked for (see its notes for the paired figures).

Reusing an unchanged build for longer is exact only if every input the build reads is either in the carry key or changes only through a path that rebuilds and pushes outside the run tick. Today nothing enumerates those inputs, and a field-level census of consecutive builds in a send window showed several changing without a fingerprint change (row counts after a persisted-rows refresh, manual unread marks after their loader lands). The owner can instead restate TASK-36571.1's AC#2 at the measured cut.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every input the workspace-context build reads is either part of the carry key or changes only through a path that rebuilds and pushes outside a run tick, and a deterministic inventory test fails when the build starts reading an input that is neither
- [ ] #2 A run tick reuses an unchanged build for as long as its carry key is unchanged, and the run tick's workspace-context build time per send window at least halves against dev, live, real provider, paired, at a stated chat length
- [ ] #3 What the rails show is never staler than one poll interval behind dev for every inventoried input, with a deterministic test per input class
- [ ] #4 A send reaches the provider no later than on dev (Enter to the provider request, live, paired and interleaved, at least 30 sends per arm including first sends)
<!-- AC:END -->
