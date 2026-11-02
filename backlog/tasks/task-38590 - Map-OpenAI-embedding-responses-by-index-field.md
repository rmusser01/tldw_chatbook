---
id: TASK-38590
title: Map OpenAI embedding responses by index field
status: To Do
created_date: 2026-11-02 14:27
labels:
- bug
- rag
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Embeddings are zipped by list order rather than the index field, so out-of-order responses from OpenAI-compatible endpoints silently assign the wrong vectors to chunks and the persistent cache then stores wrong vectors under the correct text hash. See bug-sweep-2026-11-02-dev.md section 1.7.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Embeddings are assigned by the index field rather than list order
- [ ] #2 An out-of-order provider response yields the correct text to vector mapping
- [ ] #3 Cache stores correct vectors keyed by text hash
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
