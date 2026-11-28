---
id: TASK-40034
title: Drain the readiness read before the forecast test changes policy
status: Done
created_date: 2026-11-09 09:07
labels:
- ci
- console
- test-isolation
priority: high
references:
- Tests/UI/test_console_compaction_forecast.py
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
updated_date: 2026-11-09 21:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UI4 failed the first forecast memory warm after changing the live session budget policy. A controlled run of the original case reproduced the same assertion with a real old-revision readiness read still in flight. The worker correctly refused to install stale config, leaving the first new-revision memory warm without a value.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The forecast fixture drains the old native readiness attempt before the policy change and the first original post-change warm succeeds under the same controlled scheduling.
- [x] #2 The original forecast cases pass with their assertions, ownership checks, and runtime limits intact.
- [x] #3 The correction passes independent review of the published commit and its own hosted checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: Test sequencing only. The existing checked config and session-revision contracts govern the behavior.

1. Preserve the hosted UI4 failure and use the original case with a real readiness attempt issued at the old revision to reproduce it.
2. Drain that attempt using the existing readiness projection before changing the live session policy. Keep the first subsequent warm assertion and forecast checks.
3. Prove the same control succeeds, run the original forecast cases, and require exact review and the published head’s own hosted checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
UI4 failed at the original first memory warm after setting a custom budget. The controlled original case kept a real readiness worker at revision 4 pending when the policy advanced to revision 5. Native publication discarded the stale read, and the first foreground warm returned false. The fixture now drains that existing attempt before changing policy. Under the same control the first foreground warm installed revision 5 and passed. All original assertions remain, including the first warm and no-preview checks. The exact hosted refusal branch was unobserved.

Both affected files passed all 58 original cases at 180 seconds per case. Scoped static comparison found only inherited findings. The rebased candidate also passed all 15 original browser cases and the original mounted journey. Independent exact-commit review and the successor's own hosted checks remain due.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
