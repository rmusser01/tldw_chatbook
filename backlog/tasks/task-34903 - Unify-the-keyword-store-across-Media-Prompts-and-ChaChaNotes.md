---
id: TASK-34903
title: Unify the keyword store across Media Prompts and ChaChaNotes
status: Done
created_date: 2026-10-12 17:53
labels:
- simplification
assignee:
- '@zcode'
updated_date: 2026-10-14 20:51
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Media and Prompts ship byte-identical keyword schemas under different table names (the Prompts DDL comment admits the rename) and ChaChaNotes carries a third trigger-driven variant. A keyword store is one table plus one FTS mirror plus one link table per owning entity.

Evidence (verified on dev df4985e): Media: Keywords table plus ~3k LOC region in Client_Media_DB_v2.py (add 3309 soft-delete 5962 rename 6085 merge 6194 usage 6327). Prompts: PromptKeywordsTable (Prompts_DB.py:204-214 - column-for-column identical including prev_version and merge_parent_uuid) plus ~970 LOC. ChaCha: keywords plus keyword_collections and per-entity link tables ~7.2k LOC with trigger-driven FTS. Trap: generic gateways plus the hand-maintained VALID_TABLES allowlist have gone stale three times (TASK-864 19568 19057) - see migrations README.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 KeywordStore + KeywordStoreSpec in DB/keyword_store.py own add/undelete soft-delete get-by-text FTS maintenance and entity-link synchronization parameterized by table/link/fts names (DEVIATION documented: Media rename_keyword merge_keywords fetch_media_for_keywords were verified DEAD - zero callers incl tests - and deleted instead of migrated; usage_stats moved to the store),Media and Prompts keyword blocks converged as thin delegates with public APIs unchanged,ChaCha variant documented as deliberate phase-2 exclusion (different schema - no uuid/prev_version - trigger-driven FTS and three per-entity link tables; not extractable without redesign),VALID_TABLES/VALID_COLUMNS re-synced with live DDL: Keywords gained prev_version+merge_parent_uuid and dropped never-existing deleted_at PromptKeywordsTable column set added LINK_TABLE_COLUMNS key fixed from stale PromptKeywords to live PromptKeywordLinks (its test pin updated from the stale name),Keyword tests green in both DBs (Media+Prompts suites byte-identical to the pre-change 2-failure environmental baseline)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Exact-body map of both keyword regions (schemas CRUD FTS) 2. Design KeywordStore parameterized by table/link/fts names 3. Media + Prompts delegate to it (public APIs unchanged) 4. VALID_TABLES check + keyword tests green both DBs 5. ChaCha variant documented as deliberate exception or phase 2 6. Notes + AC tick
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Phase 1 complete + REVIEW ROUND 1 (subagent in Qodo's place): verdict With-fixes; all findings applied. (1 Important) Media's generic-exception DatabaseError wrap restored on the update/soft-delete delegates (matching add_keyword's preserved shim, with the original message texts). (Minors) Media's caller-less FTS delegate shims dropped; the two remaining log deltas + get_active_keyword_by_text's broadened catch documented in the store header's deviations list. (Recommendation) NEW Tests/DB/test_keyword_store.py: 9 direct spec-seam pins - per-host spec construction exactness, both ConflictError factories' exact original arities (Media 2-positional / Prompts 3-arg with the two site messages), error factories' host types, FTS NUL-scrub index health via MATCH (external-content tables project the raw value - the scrub's observable contract is index health), and the unlink-scope semantics with a mixed active+deleted-parent construction (Media retains the deleted-parent link; Prompts removes both; the empty-branch behavior when ALL parents are deleted skips the unlink entirely - faithful to the originals and now understood). Reviewer's own probe: canonicalized DB state byte-identical old-vs-new modulo timestamps across all six flows. Media+Prompts suites byte-identical to baseline throughout.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
