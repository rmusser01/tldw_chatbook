---
id: TASK-32758
title: Restore parent-side diagnostics for failed local STT imports
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 21:47'
updated_date: '2026-09-17 22:00'
labels:
  - bug
  - stt
  - diagnostics
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A reported transcription failure produced no useful application log because native worker logs are intentionally silenced and the parent failure callback does not record the attempt. Preserve enough existing safe context to identify the failed stage and correlate the report.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Accepted local STT failures write one error log containing their job, attempt, generation, last recognized phase and stable failure code.
- [x] #2 Cancellation and stale or duplicate failure callbacks do not write misleading error logs.
- [x] #3 Logs exclude source paths, model paths and native exception text; targeted regressions and diagnostic-inventory checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md. Reason: Restore the parent-side diagnostics already permitted by the existing executor specification, without changing its failure protocol or privacy boundary. 1. Reproduce the missing error log through the real mounted failure callback. 2. Log accepted failures using existing validated identifiers, phase and error code. 3. Verify cancellation/stale callbacks and privacy, review the diagnostic inventory delta, update troubleshooting guidance and publish the follow-up repair.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Accepted non-cancelled local STT failures now leave one application error record with job/attempt IDs, generation, recognized phase and stable error code. Cancellation, stale/duplicate callbacks and shutdown remain quiet; native exception text and paths stay private. Updated the Logs guide, reviewed the single-call diagnostic inventory delta, and recorded the missing-parent-log testing lesson.

Verification: nine mounted callback regressions pass (six logging cases failed before the repair). The related executor/Parakeet run passed 107 cases with two existing profile-ownership fixture failures; both original failing assertions passed under a stable private temporary profile. Independent review found no actionable issues. New tests pass Ruff lint/format; app.py adds no lint diagnostics relative to dev; whitespace checks and all seven preflight guards pass. No full test sweep or live Fedora/model inference was performed.

ADR required: no; follows backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md and its existing executor/privacy boundary. This fixes the confirmed missing-log defect only. The reporter's underlying transcription failure remains unconfirmed, and routing, model loading and inference are unchanged.
<!-- SECTION:NOTES:END -->
