---
id: TASK-38591
title: Fix RAG swallowed-error handling for cached empties and silent chunk loss
status: In Progress
created_date: 2026-11-02 14:27
labels:
- bug
- rag
priority: medium
updated_date: 2026-11-02 14:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Transient Chroma errors are swallowed to an empty list which is then cached for the TTL hour, and partially failed embedding batches are recorded as fully indexed so the missing chunks are never retried. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Empty results caused by swallowed store errors are not cached for the TTL
- [ ] #2 Partial embedding failure marks items for retry or surfaces an error instead of recording full success
- [ ] #3 Stale fake in test_semantic_honest_states updated so the suite is green
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
