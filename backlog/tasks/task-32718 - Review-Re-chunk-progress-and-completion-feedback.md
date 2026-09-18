---
id: TASK-32718
title: Review Re-chunk progress and completion feedback
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 14:49'
updated_date: '2026-09-17 15:12'
labels:
  - library
  - search-rag
  - verification
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the bounded Library component review so older-engine reporting and Re-chunk remain understandable and keyboard-operable through progress, completion and recovery.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The legacy report and Re-chunk action reflect real available work, and keyboard users can start it in both themes at wide and compact sizes.
- [x] #2 Progress, disabled state and readable completion counts survive mode or source changes without duplicate work; existing backfill exclusion and failure recovery remain correct.
- [x] #3 Targeted tests and private native evidence qualify the supported behavior, with static checks, independent review and updated guide/audit evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: N/A; existing ADR-078 chunking template convergence, ADR-031 and ADR-150 apply. Reason: review and routine feedback repairs within the existing panel, worker and service contracts.
1. Trace the report and Re-chunk worker against TASK-19806/spec section 10, then reproduce keyboard and lifecycle problems using production-styled tests.
2. Repair confirmed display-state or focus defects with existing patterns; preserve policy admission, worker exclusion and derived-data semantics.
3. Run related targeted tests and static checks, private native SQLite journeys, capture inspection and independent review. Update guide/audit/task and commit locally; no full suite, push or merge.
Allocation: all reachable object paths plus 27 worktrees report maximum 32717.
Confirmed by seven failing cases: mode/scope recomposition resets the active button to enabled and drops progress; later recomposition also drops the completed receipt; the h-1 utility clips the semantic-index disclosure at compact width. Cache this panel-owned worker feedback on the panel, rebuild children from it, and let the existing Static auto height wrap the receipt. The cross-panel/backfill admission guard remains authoritative and unchanged.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-chunk now retains panel-owned running state and completion receipts when mode/source changes rebuild its children, and the receipt wraps instead of clipping the re-index disclosure at compact width. Existing policy, shared backfill exclusion and derived-data service semantics are unchanged.

Validation: 75 targeted tests pass, including eleven new progress/receipt/failure-retry cases. Seven initial cases reproduced the defects. Two existing CSS checks were baseline-confirmed as reading the obsolete monolithic sheet; they now use app_css_text with all original assertions. Four final private native journeys execute actual SQLite re-chunking across both themes and sizes, with eight inspected captures. Five source records remain unchanged; four are stamped and one empty source remains skipped. Semantic indexing was disabled and explicitly disclosed. Ten databases pass quick_check; normal exit, PID absence and unchanged default fingerprints are verified.

New files pass Ruff lint/format, changed ranges pass formatting, and modified existing files add no diagnostics. Independent code/test/runner review found no actionable issue; the closeout review caught a missing review artifact, which was added and link-checked. The initial native pass had overlapping transient notifications in compact captures; one confirmation pass waits for natural expiry. No production changes followed the first visual inspection.

Modified panel, targeted tests, Search/RAG guide, audit ledger and testing lesson; evidence is Docs/superpowers/qa/2026-09-17-rag-rechunk-feedback/README.md. ADR required: no; existing backlog/decisions/078-chunking-template-convergence.md, 031-tui-keybinding-and-footer-hint-conventions.md and 150-design-token-system-and-design-language.md apply. No new tokens, CSS, service or storage contracts. Same-panel persistence is qualified; cross-destination continuity and actual semantic reindexing are not. No full suite, push or merge.
<!-- SECTION:NOTES:END -->
