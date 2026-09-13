---
id: TASK-31774
title: Exercise attachment cascade through semantic deletion coordinator
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 19:42'
updated_date: '2026-09-05 19:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore the attachment hard-delete regression at the guarded current-schema deletion boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Raw message deletion remains rejected with existing attachments intact.
- [x] #2 Authorized semantic deletion removes the message and its attachments atomically while complete affected files pass.
- [x] #3 The matching exchange-sidecar cascade test also preserves raw SQL rejection and proves coordinated deletion, with unrelated messages untouched.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fresh two-file baseline reproduced2failures/74passes: raw hard DELETE is correctly rejected for both attachment and exchange sidecar tests. 2. Exercise the existing SemanticRevisionCoordinator hard_delete path in those two cases; first prove raw-SQL rejection and sidecar retention, then exact deletion/cascade and unrelated-row preservation. 3. Run both complete DB files plus semantic guard migration tests, scoped static checks and review. ADR required:no. ADR path:backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md (existing). Reason:test-only alignment to the governed mutation seam; no guard or policy change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two complete-file baseline failures (2failed/74passed) came from raw DELETE bypassing the current semantic guard. Both attachment and exchange cascade tests now first prove raw deletion is rejected with sidecar rows intact, then call the real SemanticRevisionCoordinator hard_delete operation and verify physical message deletion, FK cascade and unrelated message preservation. No SQL guard, authorization capability or runtime code changed. Three complete files including the semantic guard migration matrix:94passed13.44s, XML /private/tmp/tldw-31739-cascade-guard-fixed.xml. Full affected-file Ruff lint, changed-region formatter and diff whitespace checks pass. Self-reviewed against coordinator transaction/savepoint behavior and existing ADR097. New ADR not required; existing backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md governs the tested seam.
<!-- SECTION:NOTES:END -->
