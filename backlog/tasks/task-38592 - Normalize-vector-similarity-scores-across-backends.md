---
id: TASK-38592
title: Normalize vector similarity scores across backends
status: Done
created_date: 2026-11-02 14:27
labels:
- bug
- rag
priority: medium
updated_date: 2026-11-03 02:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
InMemoryVectorStore double-normalizes and inverts l2 scores so score_threshold filters backwards, and cosine score scales differ between the InMemory and Chroma backends so the same threshold behaves differently. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 InMemory l2 scores are not double normalized or inverted
- [x] #2 Cosine score scale matches between InMemory and Chroma backends
- [x] #3 score_threshold filters behave the same on both backends
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Removed the double l2 normalization and unified both stores on raw cosine with elementwise parity and identical-threshold tests. Commit 08b08d4edb.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
