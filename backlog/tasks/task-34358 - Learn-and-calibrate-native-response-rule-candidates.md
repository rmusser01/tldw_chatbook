---
id: TASK-34358
title: Learn and calibrate native response rule candidates
status: Done
assignee:
  - '@codex'
created_date: '2026-10-04 05:54'
updated_date: '2026-10-04 08:04'
labels: []
dependencies:
  - TASK-34357
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 5. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Learning discriminates the recorded and paraphrased violation from acceptable controls within three attempts and 120 seconds.
- [x] #2 Synthetic responses cannot fabricate work success and unrelated action guidance remains inactive.
- [x] #3 Editor tests retain the exact edited candidate and reuse validation only with matching evidence and protocol.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/219-console-learned-response-rules.md. Reason: implements approved calibration and inactive editor candidate service. Follow Task 5 RED-GREEN: bounded drafts and four fixed-evidence controls, hidden host expectations, conservative feedback scope, exact edited-candidate testing and evidence-bound reuse; targeted tests and static verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Bounded native drafting returns inactive candidates; four opaque host-labelled controls share original evidence and use the public evaluator. No prompt replay or executable hook. Exact edited definitions remain unchanged. Feedback-only reuse binds original input, fixture fingerprints and protocol; missing provenance requires retesting. Conservative host action/target checks and original-task wrapper reject additional-action guidance. ADR-219. RED missing builder -> GREEN 21 focused cases; 81 combined model/storage/semantic cases; black/ruff/mypy pass. Scripted classifications qualify orchestration, not actual model quality.
<!-- SECTION:NOTES:END -->
