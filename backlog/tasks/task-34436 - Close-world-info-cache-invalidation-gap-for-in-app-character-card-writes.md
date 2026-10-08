---
id: TASK-34436
title: Close world-info cache invalidation gap for in-app character-card writes
status: To Do
created_date: 2026-10-08 00:36
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34415 review: in-app card writes (ccp_character_handler update_character, local_character_persona_service:821) bypass WorldBookManager so a changed embedded character_book serves stale lore with no bounded window. Bump on book-bearing card updates or fold a card version into the ADR-221 cache key; also add the counter-atomicity note to ADR-221.

Scope amendment (branch final review): the identical card-edit residual exists in the dictionary bundle cache — `Chat_Dictionary_Lib.py:_resolve_active_dictionaries` keys character-card content by `character_id` only per its own docstring, so card edits bypassing the dictionary write paths serve stale bundles until the next store bump. Fix BOTH caches in this task. While updating ADR-221, also correct its two wording issues: the "zero queries" consequence overclaims for deliberately-uncached bookless conversations (world_info_resolver caches nothing when no books and no character book exist, so those sends keep their per-send book query), and "bounded LRU (8 conversations)" is actually 8 (conversation, character) slots since the cache key includes character_id. Also correct ADR-223 §6's duplicate-dedup overclaim (step 4's "duplicate texts within a batch share one lookup and one embed" — the hash cache collapses the lookups, but the factory still embeds in-batch duplicates on a miss).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Card write that changes embedded book invalidates the processor cache,Card write that changes embedded dictionaries invalidates the bundle cache (_resolve_active_dictionaries residual),ADR-221 updated (residual + atomicity + zero-queries and 8-conversations wording corrections),ADR-223 §6 duplicate-dedup overclaim corrected,Tests cover the ccp and persona-service write paths
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 3 review minors + task-3-report.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
