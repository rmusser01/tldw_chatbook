---
id: TASK-34437
title: Remove remaining legacy streaming no-op on ChatMessage and dangling tool-widget
  doc sketch
status: To Do
created_date: 2026-10-08 00:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34433: sibling seam ChatMessage.update_message_chunk (chat_message.py:312-321) is the identical watcher-less zero-caller no-op (comments cite nonexistent handle_streaming_chunk); Docs/Development/Tool-Calling/TOOL-CALLING.md:185 still sketches importing the deleted tool_message_widgets module.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Sibling entry removed with caller-grep evidence,TOOL-CALLING.md sketch updated or annotated historical,Targeted tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 21 review minors + task-21-report.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
