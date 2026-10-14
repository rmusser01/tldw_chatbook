---
id: TASK-34903
title: Unify the keyword store across Media Prompts and ChaChaNotes
status: To Do
created_date: 2026-10-12 17:53
labels:
- simplification
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Media and Prompts ship byte-identical keyword schemas under different table names (the Prompts DDL comment admits the rename) and ChaChaNotes carries a third trigger-driven variant. A keyword store is one table plus one FTS mirror plus one link table per owning entity.

Evidence (verified on dev df4985e): Media: Keywords table plus ~3k LOC region in Client_Media_DB_v2.py (add 3309 soft-delete 5962 rename 6085 merge 6194 usage 6327). Prompts: PromptKeywordsTable (Prompts_DB.py:204-214 - column-for-column identical including prev_version and merge_parent_uuid) plus ~970 LOC. ChaCha: keywords plus keyword_collections and per-entity link tables ~7.2k LOC with trigger-driven FTS. Trap: generic gateways plus the hand-maintained VALID_TABLES allowlist have gone stale three times (TASK-864 19568 19057) - see migrations README.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A KeywordStore helper owns add soft-delete get-by-text rename merge usage stats and FTS maintenance parameterized by table and link names,Media and Prompts keyword blocks converge first,ChaChaNotes trigger-driven variant folds in as a second phase or is documented as a deliberate exception,VALID_TABLES allowlist updated without stale entries,Keyword tests pass in all three DBs
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
