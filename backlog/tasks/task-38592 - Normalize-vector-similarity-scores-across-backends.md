---
id: TASK-38592
title: Normalize vector similarity scores across backends
status: To Do
created_date: 2026-11-02 14:27
labels:
- bug
- rag
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
InMemoryVectorStore double-normalizes and inverts l2 scores so score_threshold filters backwards, and cosine score scales differ between the InMemory and Chroma backends so the same threshold behaves differently. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 InMemory l2 scores are not double normalized or inverted
- [ ] #2 Cosine score scale matches between InMemory and Chroma backends
- [ ] #3 score_threshold filters behave the same on both backends
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
