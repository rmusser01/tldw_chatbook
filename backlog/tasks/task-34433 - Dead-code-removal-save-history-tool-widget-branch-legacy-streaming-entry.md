---
id: TASK-34433
title: Dead-code removal save_history tool widget branch legacy streaming entry
status: In Progress
created_date: 2026-10-07 02:43
dependencies:
- TASK-34426
updated_date: 2026-10-07 23:06
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 7 / F18: save_history has zero callers and per-message transactions tool_message_widgets has a swallowed WrongType bug on a dead branch and the legacy streaming entry point on ChatMessageEnhanced is a silent no-op
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Caller greps pasted as evidence before each removal,WrongType branch fixed or module deleted per caller audit,Legacy streaming entry removed with test callers updated,Targeted tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 21 (T21)
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
