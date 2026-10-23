---
id: TASK-36549
title: Frame Console character portraits as a stable radio display
status: To Do
created_date: 2026-10-23 22:56
dependencies:
- TASK-36545
- TASK-36546
labels:
- console
- ui
- accessibility
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add restrained radio-screen identification, truthful voice status and an audio-level meter around the living portrait without shifting the Console sidebar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The existing Character section presents a recessed fixed-geometry portrait frame, character identification, compact voice status and a measured speech-level meter; rows reserve space across state changes.
- [ ] #2 Ready, Preparing voice, Transmitting, Paused and voice-error feedback reflect actual owned activity; silence/underrun empties the meter and unqualified timing shows truthful status without invented movement.
- [ ] #3 Theme colors and authored portrait colors are preserved; any brief transmission effect uses central tokens, adds no sound/continuous static and respects motion preferences.
- [ ] #4 Narrow layouts drop the meter before sacrificing controls, preserve contain fit/rail scroll and retain full-size portrait viewing and keyboard/focus contracts.
- [ ] #5 Only source TCSS is edited and rebuilt; all new visual/motion values use defined design tokens and canonical component patterns.
- [ ] #6 Production-CSS mounted checks prove stable rows, portrait/control bounds, reduced/static motion, truthful status and narrow/tall resize; token and generated-CSS guards pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
