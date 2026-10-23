---
id: TASK-36547
title: Prepare and review imported portrait performance artwork
status: To Do
created_date: 2026-10-23 22:56
dependencies:
- TASK-36542
- TASK-36546
labels:
- personas
- character-chat
- ui
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-23-console-radio-portraits-implementation.md
- backlog/decisions/242-console-radio-portrait-performance.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users assign ready-made mouth/blink frames to a persisted local character, inspect them at Console size and apply or discard the complete set safely.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Preparation admits only saved local characters and persisted active artwork through exact local profile/service/database and editor/source authority; dirty artwork/bindings require explicit source Save/Apply without saving unrelated text.
- [ ] #2 All six still-image slots share one disposable draft; import/replace/remove status, per-role validation and matching dimensions are visible before Apply.
- [ ] #3 Offline preview cycles all six roles at the Console image-box size, compares against the source and offers idle/blink playback without requiring image generation or TTS.
- [ ] #4 Apply publishes a complete validated set atomically and emits committed revision invalidation; Cancel and source/character/profile changes discard the draft and keep current artwork.
- [ ] #5 Removing an applied performance restores ordinary artwork and invalidates predecessor caches; reopening the editor shows the stored set and its profile-local portability limit.
- [ ] #6 Keyboard-accessible mounted tests cover unsaved artwork, read-only server selection with matching IDs, profile replacement, mismatched/imported frames, cancellation, transaction failure and apply-to-Console handoff.
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
