---
id: TASK-34436
title: Close world-info cache invalidation gap for in-app character-card writes
status: Done
assignee:
- '@codex'
created_date: 2026-10-08 00:36
updated_date: 2026-10-09 05:48
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

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Mechanism (both caches, one shape): the card's DB version token is folded into the ADR-221 cache keys — world_info_resolver._cache_key and Chat_Dictionary_Lib._bundle_cache_key both key (conversation_id, character_id, card_version) and validate against the committed store generation. Both in-app write seams (LocalCharacterPersonaService.update_character and ccp_character_handler.update_character) funnel through DB.update_character_card, whose optimistic-locking UPDATE bumps version on every successful save, so any book- or dictionary-bearing card save changes the key and the next resolve refetches. Unversioned embedded content (imported/ad-hoc cards) bypasses the cache rather than freezing edits; the invalidation is unconditional in the safe direction (any card save rebuilds; extra work only). Choice rationale: no new bump code at the write seams — the version token already exists and is transactionally committed with the card row; a payload-contains-book/dict gate would save only a cheap rebuild while risking a missed field.

What each write path writes (evidence): both paths upsert the character_cards row via update_character_card, whose JSON-field list includes extensions — the column that carries both embedded character_book snapshots and chat_dictionaries blocks — so BOTH paths are real writers of BOTH cache inputs; tests cover persona and CCP for both caches.

Tests (Tests/Character_Chat/test_prompt_cache_review_regressions.py, +9): spy form of (a) book-bearing card save -> get_world_books_for_conversation re-fetched then re-warmed, parametrized persona/ccp; (b) dictionary-bearing card save -> load_chat_dictionary re-loaded then re-warmed, persona/ccp; (c) safe direction — a save touching only description rebuilds with byte-identical output (refetch spy pins the rebuild), persona/ccp x world-info/dictionary; (d) equivalence — a card save drives the same cache transition as a direct WorldBookManager write (exactly one refetch, then warm). Red captured by restoring the pre-fix version-less cache keys via a throwaway plugin: 15 card-save tests fail; 26/26 green on the real code.

ADR-221: residual gap paragraph replaced by the native-card-version invalidation contract, zero-queries claim qualified for deliberately-uncached bookless conversations, LRU bound restated as 8 (conversation, character, version) slots, and a counter-atomicity note added (increments are lock-serialized in-process; the lock is invisible to other processes and uncoordinated with SQLite's own writer serialization; a crash between commit and bump loses one increment and self-heals on the next mutation — narrow stale window accepted). ADR-223 6 corrected to 'duplicate texts share one lookup; duplicate misses still reach the factory in their original order' (embed behavior unchanged). Both cache docstrings already reference the version-token invalidation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed both native card-save cache invalidation gaps using card versions and committed store generations. Actual persona-service and CCP writes, transaction completion and recursive matching are covered by regression tests; ADR-221 and ADR-223 now describe the verified behavior.
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->
