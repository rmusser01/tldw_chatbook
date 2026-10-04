---
id: TASK-34357
title: Assess response rules using permitted execution evidence
status: Done
assignee:
  - '@codex'
created_date: '2026-10-04 05:54'
updated_date: '2026-10-04 07:56'
labels: []
dependencies:
  - TASK-34356
references:
  - Docs/superpowers/plans/2026-10-03-console-response-rules.md
documentation:
  - backlog/decisions/219-console-learned-response-rules.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Enable the approved Console response-rule behavior for plan task 4. Preserve existing work and make failure states honest.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Frozen completed-response evidence preserves uncertainty and work revisions.
- [x] #2 One strict semantic batch excludes feedback and expected verdict labels.
- [x] #3 Unavailable semantic checks preserve independent deterministic results without a false pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/219-console-learned-response-rules.md. Reason: implements accepted evidence and evaluator service boundary. Follow Task 4: failing incomplete-evidence and semantic-contract tests, permission-preserving source capture, one bounded closed-result batch with hidden calibration labels and no feedback, targeted behavior regression and static verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Freeze only the active task chain and already model-bound result bodies, pairing definitive body-free tool dispatch facts. Closed semantic batches exclude feedback/labels and reject invented evidence, malformed results and overflow. Native context preflight retains transport limits; Save remaps actual durable message versions. ADR-219. Evidence: 81 feature/evidence/storage/resource tests; 52 existing auxiliary gateway cases (one reproduced baseline failure excluded); black/ruff/mypy pass. Current work freshness remains unknown when the execution owner supplies no revision.
<!-- SECTION:NOTES:END -->
