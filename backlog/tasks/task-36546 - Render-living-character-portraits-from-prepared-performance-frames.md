---
id: TASK-36546
title: Render living character portraits from prepared performance frames
status: To Do
created_date: 2026-10-23 22:56
dependencies:
- TASK-36542
- TASK-36543
- TASK-36545
labels:
- console
- character-chat
- ui
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Animate the selected character expression with audio-reactive mouth poses, independent blinking and subtle idle motion while preserving Console ownership, geometry and motion preferences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prepared sets display the matching mood and source; mouth changes follow owned output snapshots rather than text streaming, with independent blinking and fixed-canvas idle motion.
- [ ] #2 Automatic, manual and valid historical expression precedence is preserved; unprepared, stale or unavailable sets fall back without substituting an unrelated neutral talking face.
- [ ] #3 Same-source Apply/Remove and revision changes refresh the visible portrait after commit; late predecessor preparation or paint cannot restore a removed/replaced set.
- [ ] #4 Dynamic/Static, Reduce motion, animations-off, reactions-off and avatar visibility retain their governing behavior; hide/show, source/profile/session changes and resize fence and release visual work.
- [ ] #5 Frame preparation stays off-thread within existing active/in-flight/shared budgets, paints at most 30 times per second and performs no per-frame decode, SQLite read or rail remount.
- [ ] #6 Mounted tests prove changed pixels, silence closure, mood retention, historical replay restoration, geometry stability, viewer snapshots, bounded resource release and stale-worker rejection.
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
