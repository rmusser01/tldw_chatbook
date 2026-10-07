---
id: TASK-34414
title: Uncorrelate Library conversation content search
status: Done
created_date: 2026-10-07 02:40
updated_date: 2026-10-07 03:43
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 1 / F2: library conversation search uses a correlated EXISTS with leading-wildcard LIKE on messages content scanning the whole corpus per candidate row - the Console seam already removed this exact shape
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Results and hit projections byte-identical to baseline,Correlated EXISTS replaced by uncorrelated IN subquery,Trace evidence shows no correlated EXISTS,Note search LIKE branch decision documented
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 2 (T2)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Replaced the correlated content-LIKE EXISTS branch in search_library_conversations_page with the uncorrelated id IN (SELECT m.conversation_id FROM messages ...) shape proven on the Console seam (_conversation_search_filter). Parameter order and branch index 2 unchanged; COUNT/page/hit projections inherit via the shared branches list. Equivalence proven against a golden baseline captured on unmodified code (title exact/substring, message mid-word substring 'indo', FTS token, keyword, no-match, multi-branch hit projections) — 155 tests green across 3 files. Evidence: EQP changed CORRELATED SCALAR SUBQUERY -> LIST SUBQUERY + bloom filter; 50x50 no-match 8.77ms -> 3.29/5.99ms; the sibling seam's ~70s/150k figure (task-33261/PERF-02, FTS-MATCH variant) cited with correct attribution. Files: DB/ChaChaNotes_DB.py, Tests/ChaChaNotesDB/test_library_conversation_search.py. Report: .superpowers/sdd/task-2-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
