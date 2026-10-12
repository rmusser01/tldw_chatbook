---
id: TASK-34391
title: Make InMemoryVectorStore add/search sublinear-friendly (matrix + cached norms
  + dict ids)
status: To Do
created_date: 2026-10-11 21:44
labels:
- performance
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
InMemoryVectorStore (default without the chroma extra) does linear self.ids.index membership and pop per add (O(n^2) batch ingest), recomputes np.linalg.norm of both vectors for every pair per query, fully sorts all similarities, and maintains an access-order plain list with O(n) remove. Keep a stacked (n,d) matrix with cached row norms, dict id->index, and argpartition top-k.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Batch ingest of m documents is O(m) structure updates instead of O(m*n),Query similarity uses one matrix product with precomputed norms and returns the same top-k results as before (property or golden test),LRU access-order maintenance is O(1) per touch,Existing vector store tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
