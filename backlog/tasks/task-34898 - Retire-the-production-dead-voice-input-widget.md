---
id: TASK-34898
title: Retire the production-dead voice input widget
status: To Do
created_date: 2026-10-13 02:04
labels:
- simplification
- dead-code
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Widgets/voice_input_widget.py has zero production importers - its only consumers are four test files (css census plus Tests/Audio). It is the sole reason six dictation-event classes and AudioLevelUpdateEvent are still kept in the Audio_Events catalogs (task-34891 pruned only the truly unreferenced ones). Verify no dynamic references then delete the widget and its tests, then prune the now-orphaned event classes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 voice_input_widget.py and its test-only importers removed,Newly-orphaned Audio_Events classes pruned with a usage census in the notes,Audio and UI census tests pass
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
