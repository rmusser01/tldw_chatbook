---
id: TASK-73.3
title: Expand Settings Console defaults
status: To Do
labels:
- settings
- console
- configuration
- ux
dependencies:
- TASK-73.1
- TASK-73.2
priority: high
parent_task_id: TASK-73
documentation:
- Docs/superpowers/plans/2026-05-29-settings-configuration-hub.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expand Console Behavior into a real global-defaults category for supported Console settings while preserving Console as the owner of active session and run state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Console global defaults load from config and distinguish global defaults from per-session Console overrides.
- [ ] #2 The `streaming` and `enable_streaming` compatibility seam has one documented effective source of truth before new controls are added.
- [ ] #3 Supported defaults can be edited, validated, saved, reverted, and reflected in Console behavior.
- [ ] #4 Invalid numeric or boolean values are blocked with visible recovery copy.
- [ ] #5 Existing large-paste collapse behavior remains intact and covered.
- [ ] #6 Mounted tests verify save/revert and Console reflection for changed defaults.
- [ ] #7 Actual CDP/Textual-web screenshot QA verifies the Console Behavior category and is approved before PR.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Unit, mounted, and integration tests relevant to changed behavior pass
- [ ] #3 Static analysis and diff hygiene checks pass
- [ ] #4 Actual app QA walkthrough completed with screenshots
- [ ] #5 User approval recorded for visible Settings changes
- [ ] #6 Documentation and task notes updated
- [ ] #7 Task status moved to Done after implementation notes are added
<!-- DOD:END -->
