---
id: TASK-38580
title: Fix RAG shared-service search TypeError from query_embedding passthrough
status: In Progress
created_date: 2026-11-02 14:26
labels:
- bug
- rag
priority: high
updated_date: 2026-11-02 14:49
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every search through the shared RAG service raises TypeError because EnhancedRAGServiceV2.search forwards query_embedding to RAGService.search which has no such parameter. Regression from commit 859c76f104 (TASK-34390); unit test doubles masked it. See bug-sweep-2026-11-02-dev.md section 1.1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 RAGService.search accepts and threads query_embedding or V2 stops forwarding it
- [ ] #2 Contract test drives the real EnhancedRAGServiceV2 to RAGService chain
- [ ] #3 Library RAG answer path returns results again
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
