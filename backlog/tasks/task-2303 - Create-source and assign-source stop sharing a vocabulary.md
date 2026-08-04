---
id: TASK-2303
title: Create-source and assign-source stop sharing a vocabulary
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: high
---

## Description (the why)

UAT: three near-synonym labels coexist for two DIFFERENT operations — the
rail's "Add source" ASSIGNS an existing source to the selected watchlist,
while the header's "Create source" and the pane's "New Source" CREATE one.
Users will click the wrong one confidently. Assignment is also only
discoverable through that ambiguous rail button: the selected source's
Inspector has no assign/move action, and the assignment modal is a bare list
with no instruction line.

UAT findings F1 (high), F18.

## Acceptance Criteria (the what)

- [ ] One verb consistently means "create a new source" and a clearly
      different verb means "put an existing source into a watchlist", across
      rail, header, pane, guidance copy and Inspector.
- [ ] A selected source's Inspector offers the assign/move action.
- [ ] The assignment modal explains what clicking an entry does.
- [ ] First-run guidance references labels that actually exist on screen.
