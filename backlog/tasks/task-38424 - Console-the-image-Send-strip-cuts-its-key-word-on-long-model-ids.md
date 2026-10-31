---
id: TASK-38424
title: 'Console: the image Send strip cuts its key word on long model ids'
status: To Do
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
- [ ] #2 The Send tooltip still names the model
- [ ] #3 A mounted test asserts the strip at that width
<!-- AC:END -->
