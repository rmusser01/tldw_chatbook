---
id: TASK-34419
title: Bound get_conversation_tree reads to the root page
status: Done
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 08:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 / F5: opening a conversation materializes every message row even when only a 50-root page renders
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Paged API returns identical rows and total_roots vs baseline,Rows fetched bounded to page subtree,Trace callback evidence recorded,Fork path keeps explicit unbounded variant
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 7 (T7)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added get_message_tree_rows_for_conversation_page (additive _page variant per codebase convention): three bounded statements in one transaction — root COUNT, root page, one recursive CTE seeded with the page root ids — replicating the assembly's exact root criterion (parent_message_id IS NULL, cross-conversation-parent NOT a root, documented in docstring) and adding an explicit rowid tiebreaker for determinism. get_conversation_tree consumes (rows, total_roots); Python root slicing deleted; unbounded method and fork path byte-identical (no remaining production callers of the unbounded variant outside tests-as-oracle). Golden matrix: 87 tests (64 rows-level: limits 10/25/50/100 x 8 offsets incl. overflow x both orderings; 20 service-render; 3 trace) against the real unbounded method + verbatim partition/slice replicas on a hostile fixture (15-deep chain, 120 siblings, >50 parent-child timestamp inversions, soft-deletes, images, foreign-parent row) with full dict-equality assertions. Evidence: set_trace_callback — 315-row conversation old 315 rows -> new 300 for page 1 (subtree-heavy), structural bound 2,060 rows -> 51 rows. Files: DB/ChaChaNotes_DB.py, Chat/chat_conversation_service.py, Tests/Chat/test_conversation_tree_bounded_reads.py + 2 extended. Report: .superpowers/sdd/task-7-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
