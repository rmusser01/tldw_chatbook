---
id: TASK-31624
title: >-
  Inspect rail n/p boundary navigation covers the Environment, Tasks and Agents
  sections
status: To Do
assignee: []
created_date: '2026-09-04 23:10'
labels:
  - console
  - inspector
  - a11y
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Environment redesign (TASK-31450) added three sections to the Console
Inspect rail — Environment, Tasks and Agents — and shipped them as
tab-reachable rows only. The rail's own `n`/`p` boundary-focus navigation,
which every pre-existing rail section participates in, skips them, so a
keyboard user moving through the rail section-by-section jumps over the new
content entirely.

Including them was deferred during the arc's final review: the boundary-focus
machinery is shared by every rail section and changing it is high-risk
surgery that would have put the rest of the redesign at risk. The deferral is
recorded as a post-implementation ruling in
`Docs/superpowers/specs/2026-09-04-console-inspector-environment-redesign-design.md`
(§Interactions), which is what this task completes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 With the Inspect rail open, `n`/`p` moves focus into and out of the Environment, Tasks and Agents sections in visual order, alongside the rail's existing sections
- [ ] #2 A section that is currently collapsed or entirely hidden (no rows projected) is skipped rather than trapping focus
- [ ] #3 Boundary navigation over the pre-existing rail sections is unchanged — proven by a test that would fail if their order or reachability moved
- [ ] #4 The spec's §Interactions deferral note is replaced with what actually shipped
<!-- AC:END -->
