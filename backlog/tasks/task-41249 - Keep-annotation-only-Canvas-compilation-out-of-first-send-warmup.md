---
id: TASK-41249
title: Keep annotation-only Canvas compilation out of first-send warmup
status: Done
created_date: 2026-11-16 20:02
assignee:
- '@codex'
updated_date: 2026-11-16 20:05
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #3537 Perf Guard fails at 95 first-send warmup modules against the unchanged 94-module ratchet after integration of RAG scope evidence. Remove unnecessary runtime import cost while preserving actual first-send and Canvas behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The unchanged first-send warmup census passes at no more than 94 modules with scope evidence still warmed.
- [x] #2 Native Canvas creation, updates, compiler scheduling and first-send imports retain their existing behavior.
- [x] #3 Focused checks and independent review qualify the import-only change.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/097-boot-budget-ratchets.md. Reason: annotation-only import deferral implements the existing ratchet policy; no runtime boundary changes. 1. Reproduce the exact failing census and inspect every PreparedCanvasDocument use and caller. 2. Move its annotation-only import under TYPE_CHECKING. 3. Run the exact census, affected Canvas behavior and first-send contracts, lint and independent review; retain before/after evidence.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Moved PreparedCanvasDocument into TYPE_CHECKING in Canvas/native_authority.py; its three uses are postponed annotations, with no runtime annotation consumers. Runtime construction stays in the existing Canvas service/controller. FIRST_SEND_MODULES, RAG scope-evidence warming, assertions and numeric limits are unchanged. This implements ADR-097 by removing unnecessary import cost; no new ADR required. Refreshed only first_send_warmup.json using scripts/update_boot_budget_snapshots.py --only first-send (no force): exactly scope_evidence added and Canvas.compilation removed relative to the old module snapshot, still 94 modules. Informational LOC values reflect current upstream sources. Red: exact original test failed at 95/94 in 5.28s. Green: exact census, native-authority behavior, compiler scheduling and first-send preimport/cycles: 90 passed in 53.38s; snapshot contracts: 11 passed in 0.61s. Ruff lint and whitespace pass; formatter edit hunks are identical to preexisting HEAD drift, with no new formatting findings. Backlog file/ID and ADR validators pass. Independent requesting-code-review review found no actionable issues. Evidence: /tmp/tldw-ci-3537-warmup-{before,after,snapshot,snapshot-tests}.log. Live normal PR confirmation remains part of PR3537 qualification; no full local suite was run. Existing lessons-testing-evidence.md first-send census lesson already documents the relevant measurement rule.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
