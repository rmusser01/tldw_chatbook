---
id: TASK-38424
title: 'Console: the image Send strip cuts its key word on long model ids'
status: In Progress
assignee: []
created_date: '2026-10-31 03:31'
labels:
  - console
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A staged image the selected model cannot take reads 'Send blocked — <model> can't accept images' in the composer's reason strip (TASK-33621.11). The strip has a width cap, so a long model id pushes the key word out: live it read "Send blocked — claude-haiku-4-5 can't accept imag…". The Send tooltip already carries the full refusal naming the model. Consider shorter strip copy that always fits (for example 'Send blocked — this model can't accept images') and leave the model name to the tooltip.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The image Send strip shows its whole meaning at the narrowest supported composer width with the longest catalog model ids
- [x] #2 The Send tooltip still names the model
- [ ] #3 A mounted test asserts the strip at that width
<!-- AC:END -->

## Implementation Plan

ADR required: no new ADR. ADR path: backlog/decisions/211-ephemeral-provider-failure-presentation.md. Reason: bounded presentation-only copy repair within existing geometry.

Use model-independent strip copy while retaining the full model in the tooltip. Verify mounted ordinary widths and unchanged resize/draft budget.

## Implementation Notes

PR3330 current-base integration shortens the image notice to “this model can't accept images”; full model-specific refusal remains in the tooltip. TASK38424 stays open: existing dynamic width policy can hide/ellipsize below its legibility budget, so AC1 (whole meaning at narrowest width) is not fulfilled by this scoped copy change. Mounted ordinary160/235 exact painted-copy, full-tooltip and resize disclosure evidence passed; no geometry or size budget changed.

Measured current narrow limitation: at110, row107/draft13(content11), strip18(content16), cap18, painted “Send blocked — …”; at80 row77/draft2(content0), cap0, strip hidden. The80 draft region is padding-only; this copy-only repair does not claim narrow editor usability. Existing layout helpers/CSS are compared against actual dev base; no geometry change.
