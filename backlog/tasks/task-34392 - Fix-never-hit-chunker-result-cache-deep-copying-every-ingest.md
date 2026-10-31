---
id: TASK-34392
title: Fix never-hit chunker result cache deep-copying every ingest
status: Done
created_date: 2026-10-11 21:44
labels:
- performance
priority: low
updated_date: 2026-10-12 01:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
improved_chunking_process constructs a throwaway Chunker per call; that instance LRU-caches a deepcopy of the full chunk result list under a content hash that can never be hit again (instance dies at end of call). Pure overhead (a full document copy) on every ingest. Either scope the cache process-wide or disable it on this path.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 improved_chunking_process no longer performs a guaranteed-miss cache lookup plus deepcopy of the chunk corpus,Chunking output is unchanged for all chunking methods,Existing chunking tests pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented and verified on perf/34386-34388-hot-path-efficiency; see task file Implementation Notes; PR #3083
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
