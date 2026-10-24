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
- Docs/superpowers/reviews/2026-10-23-console-radio-portraits-implementation-review.md
updated_date: 2026-10-23 23:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add restrained radio-screen identification, truthful voice status and an audio-level meter around the living portrait without shifting the Console sidebar.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The existing Character section presents a recessed fixed-geometry portrait frame, character identification, compact voice status and a measured speech-level meter; ordinary-layout rows reserve space through rest, speech and timing loss/recovery using visibility hiding, with row removal restricted to compact layout.
- [ ] #2 Ready, Preparing voice, Transmitting, Paused and voice-error feedback reflect actual owned activity; silence/underrun empties the meter and unqualified timing shows truthful status without invented movement.
- [ ] #3 Theme colors and authored portrait colors are preserved; any brief transmission effect uses central tokens, adds no sound/continuous static and respects motion preferences.
- [ ] #4 Narrow layouts drop the meter before sacrificing controls, preserve contain fit/rail scroll and retain full-size portrait viewing and keyboard/focus contracts.
- [ ] #5 Only source TCSS is edited and rebuilt; all new visual/motion values use defined design tokens and canonical component patterns.
- [ ] #6 Production-CSS mounted checks prove stable meter-row height, actual portrait/control bounds and widget identity through timing loss/recovery and rest/transmission; reduced/static motion and narrow/tall resize remain correct, compactness follows layout budget rather than speech state, and token/generated-CSS guards pass.
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
