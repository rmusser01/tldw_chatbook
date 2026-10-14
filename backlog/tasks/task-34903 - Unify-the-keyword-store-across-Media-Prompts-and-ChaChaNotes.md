---
id: TASK-34903
title: Unify the keyword store across Media Prompts and ChaChaNotes
status: Done
created_date: 2026-10-12 17:53
labels:
- simplification
assignee:
- '@zcode'
updated_date: 2026-10-14 20:25
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
Phase 1 complete. DB/keyword_store.py (618 lines): shared KeywordStore+KeywordStoreSpec per ADR-177 (helper not base class) - Media and Prompts delegate add_keyword/_add_keyword_full update_keywords_for_* soft_delete_keyword FTS helpers get_active_keyword_by_text and usage_stats to it; per-host spec callables preserve each host's exact exception construction (incl Media's 2-positional ConflictError - attributes observable) normalization (Prompts whitespace-collapse via instance-method lambdas) and soft-delete unlink scope (Media: active-parent links IN-list; Prompts: all links). Unified: FTS NUL scrub now covers Prompts too (strict improvement); two duplicated log lines consolidated (inventory re-frozen via sanctioned --write: 19 statements in the new owner, Media -38 Prompts -12; checker green). DEAD CODE DELETED per census: Media rename_keyword merge_keywords fetch_media_for_keywords; Prompts _add_keyword_with_retry (all zero callers incl Tests/). sql_validation re-sync: VALID_COLUMNS Keywords (live DDL), +PromptKeywordsTable, LINK_TABLE_COLUMNS PromptKeywords->PromptKeywordLinks (test pin updated from the stale name - the VALID_TABLES allowlist trap caught in the wild). VERIFICATION: Tests/Media_DB + Tests/Prompts_DB byte-identical failure set to baseline (2 known env failures); test_sql_validation 31/31; checker green; app import ok. Tests/DB full run 188 failures ALL attributed: 3-file ID-level diff vs pristine dev identical except test_trusted_alias_copy_positively_retires_native_resources[ordinary] which is order-flaky and fails at pristine in isolation too (subprocess-returncode test); remaining files untouched by this diff. The implementation subagent died after writing the store; host rewiring/validation/inventory completed in-session.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
