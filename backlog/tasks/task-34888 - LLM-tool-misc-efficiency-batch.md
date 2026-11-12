---
id: TASK-34888
title: LLM/tool misc efficiency batch
status: Done
created_date: 2026-10-12 03:20
labels:
- performance
priority: low
assignee:
- '@zcode'
updated_date: 2026-10-12 04:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three verified small items: (1) Tools/watchlists_tool_service.py _search_items (~249-280) and _finalize_metadata_page (~1855-1868) re-serialize the whole accumulated payload per row to test the 30KiB byte budget - O(n^2) bytes per tool call (bounded, ~1.5MB dumps worst case); (2) Tools/local_tool_impls.py _relative_target_is_safe (~1042-1063) does two Path.resolve() syscalls per candidate file (workspace root is loop-invariant) and _is_relative_sensitive_path (~1071-1092) rebuilds folded exclusion tuples per candidate; grep regex also compiled twice per call (~965,~997). (save_history variant-closure/resave efficiency lives in task-34396 where it was also filed.)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Watchlists byte budget uses per-item serialization with a running total (or binary search) instead of whole-payload reserialization per row,fs_grep/fs_glob hoist resolved workspace and folded exclusions out of the per-candidate path and only resolve candidates that pass the lexical check,Grep pattern compiled once per call,Existing watchlists/local-tool/chat-persistence tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Incremental watchlists byte budget 2. fs_grep invariant hoisting + single regex compile 3. tests + notes
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
