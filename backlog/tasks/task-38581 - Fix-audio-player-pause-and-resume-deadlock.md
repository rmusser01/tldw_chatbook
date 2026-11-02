---
id: TASK-38581
title: Fix audio player pause and resume deadlock
status: To Do
created_date: 2026-11-02 14:26
labels:
- bug
- audio
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
One Pause click permanently wedges all TTS audio on players without native pause such as afplay on macOS. pause takes a non-reentrant lock then calls get_position and stop which re-acquire it. See bug-sweep-2026-11-02-dev.md section 1.2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 pause and resume return promptly on players without native pause such as afplay
- [ ] #2 Reentrant lock or calls moved outside the lock
- [ ] #3 Regression test pauses a real player without hanging
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
