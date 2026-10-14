---
id: TASK-34395
title: Fix quadratic session-row staging in the notes git service
status: Done
created_date: 2026-10-12 03:20
labels:
- performance
priority: high
assignee:
- '@zcode'
updated_date: 2026-10-12 04:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Notes/file_notes_git_service.py stage/unstage machinery is O(groups x records x endpoints) per action and O(G^2 x R) for bulk stage (shared ambiguity/closure sets recomputed per group inside the apply loop; tuple membership instead of sets; allowed-path set built as O(index_entries x groups x endpoints) nested scan). With ~200 changed notes and ~1000 index entries a bulk stage is on the order of 10^7 Python operations (seconds), paid on every Refresh/Stage/Commit preflight. Also memoize the per-endpoint stat walk (_map_session_endpoint does 2+ stats per path component per endpoint per inspection).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SessionChangeGroup endpoints membership tests use frozensets or a prebuilt path index,Bulk stage/unstage of G groups computes _ambiguous_group_ids/_effective_mutation_paths/compute_stage_closure once per operation instead of per group,Endpoint parent-dir stat results are memoized within one inspection pass,Behavior identical: same rows staged/unstaged for the same repositories (existing git-service tests pass unchanged),Existing Tests covering the git service pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read file_notes_git_service session-row machinery 2. frozenset endpoints + path index + hoist shared set computation out of per-group loops 3. memoize endpoint stat walks per inspection 4. targeted tests + notes
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented and verified (897 consolidated tests green); see task file Implementation Notes; PR #3083
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
