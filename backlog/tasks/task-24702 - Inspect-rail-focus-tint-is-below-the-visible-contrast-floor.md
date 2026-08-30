---
id: TASK-24702
title: Inspect rail focus tint is below the visible-contrast floor
status: To Do
assignee:
  - '@claude'
created_date: '2026-08-30 06:18'
updated_date: '2026-08-30 06:24'
labels:
  - console
  - ux
  - inspector
  - a11y
  - critique-2026-08-30
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-24612 gave the rail's two container Tab stops a focus treatment copied from .console-bounded-section-viewport:focus - background $ds-action-focus 12%. Measured live via SGR parse: (31,55,74) on (30,30,30) = 1.35:1, and 1.11:1 on the pinned-card background. WCAG's non-text minimum is 3:1. This is systemic, not local: the convention that was copied is equally invisible. At 80x24, 5 of 12 Tab stops show no focus indicator at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A focused container in the Inspect rail is distinguishable from an unfocused one at 3:1 or better, measured in a running terminal
- [ ] #2 The shared bounded-section focus convention is raised too, not just the two rail containers
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PARTIAL -- deliberately left To Do.

Raised the shared focus tint from '$ds-action-focus 12%' to 45%, in .console-bounded-section-viewport:focus as well as the two containers TASK-24612 added, because every consumer of that convention had the same problem and fixing only the new selectors would have left the rail's sections quieter than its containers.

But 45% does NOT meet the AC. Measured: 12% renders (31,55,74) = 1.35:1 against the rail background and 1.11:1 against the pinned card. Computed against this palette, 45% reaches only ~1.74:1, and a full-opacity accent reaches just 3.77:1 -- so a background TINT cannot clear WCAG's 3:1 non-text floor here until it is ~85-90% opaque, i.e. a solid fill that would fight the text on top of it.

The mechanism is wrong, not the number. The cue wants an outline or edge marker (DESIGN.md's 'outline: heavy $accent'), which changes which cells the container paints and needs its own design pass -- an outline on a scroller overwrites content edge cells. Left open for that decision rather than shipping a number that looks like a fix.

The test added asserts >= 30% as a REGRESSION GUARD and its docstring says explicitly that it is not a contrast guarantee.
<!-- SECTION:NOTES:END -->
