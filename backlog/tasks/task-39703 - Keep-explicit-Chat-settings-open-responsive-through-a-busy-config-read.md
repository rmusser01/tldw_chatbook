---
id: TASK-39703
title: Keep explicit Chat settings open responsive through a busy config read
status: In Progress
created_date: 2026-11-09 03:32
labels:
- ci
- console
- settings
- performance
priority: high
references:
- Tests/UI/test_settings_save_reach.py
- Tests/UI/test_console_full_config_preparation.py
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- backlog/decisions/126-complete-local-backup-and-recovery.md
updated_date: 2026-11-09 04:20
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two original UI5 cases lost an explicit Chat settings open after their own real configuration save. Keep the checked action usable while preserving background FULL deferral and native ownership.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Trace the original failing assertions to the exact current-owner/native-admission path under unchanged limits.
- [x] #2 An explicit Settings open completes after a real held config lock retires, while concurrent passive FULL defers before admission; changed owners and cancellation still refuse stale work and physically retire the reader.
- [ ] #3 The original UI5 roster and directly affected ownership controls pass locally, with original software UAT, independent exact-head review and own hosted gates retained.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/126-complete-local-backup-and-recovery.md
Reason: preserve existing checked configuration ownership and Settings source fencing; refine only when one explicit native read may wait after authentic contention.

1. Retain the exact hosted UI5 failures and the natural 831 observation of both original private cases. The original hosted log does not expose the exact return branch. Preserve all surviving assertions and the original 180-second case cap.
2. Add a focused real REBUILD holder control for concurrent explicit Settings and passive FULL. Reproduce the original refusal with real native entry evidence, no fake lock/getter.
3. Keep passive checked admission nonblocking. Bind the native busy outcome to the exact retired attempt and reader only after cleanup; cancellation and generic refusals grant nothing. Only the explicit Settings owner may request one original off-loop blocking native read for that same key and live reader, after its existing current-owner check. Passive FULL avoids awaiting the special unadmitted attempt and keeps original custody after native entry. Keep the original operation/body/source proof and physical retirement, all downstream stale-owner fences, no new timer, timeout, privilege or replay of the open.
4. Verify the two original UI5 private cases, full UI5 roster and relevant readiness/retirement/ownership/Perf controls; rerun the original software UAT after production changes. Retain exact source/normal hooks and independent exact-head review, then publish under a fresh exact lease while auto is off. Own hosted checks are separate. Do not re-open or edit prior completed task records.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
An explicit Chat Settings open now waits for the original checked configuration read only after its own passive read has genuinely refused a busy native lock. It verifies the same live owner, reader and key before continuing. Passive FULL still defers before native admission and waits for a read already admitted. Cancellation and chat changes keep the native lease owned through physical retirement, and the internal busy exception contains no printable configuration. ADR-126 still governs this path; no new ADR or timing limit is needed.

The published UI5 job failed the first original Settings open in two credential-save cases. The same two private cases reproduced under the real REBUILD lock locally; the hosted return branch was not captured. The uninstrumented original UI5 roster passed all 119 cases after the fix, with no skips and the original 180-second case limit. The focused ownership controls, official diagnostic inventory, 61 targeted performance/ownership cases, all 15 original browser cases, and the original three-child mounted journey passed. Nine supplemental unmounted-screen fixture failures were retained separately; two representative failures reproduced with the published runtime, and none is enrolled in the current hosted PR gates.

Exact committed-head review and that head’s own hosted checks are still required. Keep AC3 open and this task In Progress until those receipts are verified.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
