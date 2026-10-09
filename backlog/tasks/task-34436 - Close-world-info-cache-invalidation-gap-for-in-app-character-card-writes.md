---
id: TASK-34436
title: Close world-info cache invalidation gap for in-app character-card writes
status: Done
assignee:
  - '@codex'
created_date: '2026-10-08 00:36'
updated_date: '2026-10-09 02:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up from TASK-34666 review: in-app card writes (ccp_character_handler update_character, local_character_persona_service:821) bypass WorldBookManager so a changed embedded character_book serves stale lore with no bounded window. Bump on book-bearing card updates or fold a card version into the ADR-221 cache key; also add the counter-atomicity note to ADR-221.

Scope amendment (branch final review): the identical card-edit residual exists in the dictionary bundle cache — `Chat_Dictionary_Lib.py:_resolve_active_dictionaries` keys character-card content by `character_id` only per its own docstring, so card edits bypassing the dictionary write paths serve stale bundles until the next store bump. Fix BOTH caches in this task. While updating ADR-221, also correct its two wording issues: the "zero queries" consequence overclaims for deliberately-uncached bookless conversations (world_info_resolver caches nothing when no books and no character book exist, so those sends keep their per-send book query), and "bounded LRU (8 conversations)" is actually 8 (conversation, character) slots since the cache key includes character_id. Also correct ADR-223 §6's duplicate-dedup overclaim (step 4's "duplicate texts within a batch share one lookup and one embed" — the hash cache collapses the lookups, but the factory still embeds in-batch duplicates on a miss).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Native persona-service and CCP saves changing embedded books refresh warm world-info content.
- [x] #2 Native persona-service and CCP saves changing embedded dictionaries refresh warm dictionary content.
- [x] #3 Managed, nested and borrowed transaction completion cannot publish a reusable stale prompt snapshot.
- [x] #4 ADR-221 states the actual cache bounds, empty-send behavior and invalidation contract.
- [x] #5 ADR-223 accurately describes duplicate embeddings on a cache miss.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add regression coverage for persona-service and CCP card saves affecting both injection caches.
2. Key cache slots by native card version and bypass unversioned embedded content; publish generation changes after transaction completion.
3. Verify managed, nested, borrowed, commit and rollback cases plus unchanged warm-send budgets.
4. Correct ADR-221 cache bounds, uncached empty sends and recursive matching behavior, plus ADR-223 duplicate-miss wording.
ADR required: no
ADR path: backlog/decisions/221-prompt-injection-cold-start-caches.md; backlog/decisions/223-persistent-embedding-content-hash-cache.md
Reason: closes correctness gaps in the existing cache contracts without a new storage or ownership boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both injection caches now include native card versions and bypass unversioned embedded content. Shared generation publication uses the existing managed-transaction observer; borrowed native writes suppress cache reuse until completion and conservatively invalidate afterward. Ordinary conversation writes retain warm-cache budgets. Value-equal recursive world-info behavior is preserved.

Production changes: Character_Chat/world_book_manager.py, world_info_resolver.py, world_info_processor.py and Chat_Dictionary_Lib.py. Regression coverage in Tests/Character_Chat/test_prompt_cache_review_regressions.py exercises the actual persona-service and CCP card-save paths plus managed, nested, borrowed, cross-thread commit/rollback cases. The final cache and Notes qualification ran 40 tests successfully; existing world-info/dictionary cache tests also pass. New test files pass normal Ruff and formatting; modified Python files pass fatal-error lint. Independent database and provider reviewers found no remaining blocker.

ADR check: existing ADR-221 and ADR-223 apply; no new boundary or schema decision. Their descriptions now state the eight-slot cache bound, deliberately uncached empty sends, commit publication, card-version invalidation, value-equal recursion and the actual duplicate-miss behavior.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed both native card-save cache invalidation gaps using card versions and committed store generations. Actual persona-service and CCP writes, transaction completion and recursive matching are covered by regression tests; ADR-221 and ADR-223 now describe the verified behavior.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
