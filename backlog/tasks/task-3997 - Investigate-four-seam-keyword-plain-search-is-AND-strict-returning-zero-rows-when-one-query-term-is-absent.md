---
id: TASK-3997
title: >-
  Investigate: four-seam keyword (plain) search is AND-strict, returning zero
  rows when one query term is absent
status: To Do
assignee: []
created_date: '2026-08-09 05:17'
updated_date: '2026-08-11 18:06'
labels:
  - rag
  - retrieval
  - investigation
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the P1 eval harness (TASK-3894). build_fts_match_query (Library/library_fts_query.py, near L94-101) joins every query term group with AND, so a single term with no match anywhere in the corpus zeroes the entire query result for that seam. On the P1 golden query set this produced 0 rows for 2 of 15 keyword-category queries and only a partial match (1 of 2 relevant documents) for a third, despite the corpus containing documents that satisfy most of the query intent. Whether to move to OR-with-ranking, a configurable AND or OR strictness, or keep AND-strict behavior as an intentional precision-over-recall choice for the plain four-seam mode is a product decision, not a mechanical fix; this task is scoped to investigating and proposing an approach, not implementing one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The current AND-strict behavior effect on the P1 golden query set is documented (which queries return zero or partial results and why) as a baseline for the decision.
- [ ] #2 At least one alternative (for example OR-with-ranking, or configurable strictness) is proposed with its precision-versus-recall tradeoffs for the four-seam keyword path.
- [ ] #3 A product decision on whether and how to change the AND-join behavior is recorded, in this task or a follow-up task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Cross-reference (2026-08-11, TASK-15020/B2): TASK-15400 is the ENGINE-leg half of this same AND-strictness question. This task covers the Library's four-seam path (build_fts_match_query); 15400 covers RAGService._escape_fts5_query, where the effect is far larger (zero rows for 40 of 60 golden queries). 15400's AC#8 overlaps this task's AC#3 — the product decision on AND-join behaviour should be taken once for both paths, or one should explicitly defer to the other. 15400 also carries the measured comparison table and two constraints (the vector-blind fixture kw-plant-maintenance-record, which OR-of-tokens loses; and the load-bearing per-token quoting).
<!-- SECTION:NOTES:END -->
