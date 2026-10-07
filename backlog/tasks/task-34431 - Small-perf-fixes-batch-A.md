---
id: TASK-34431
title: Small perf fixes batch A
status: Done
created_date: 2026-10-07 02:43
updated_date: 2026-10-07 22:28
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 7 / F19a-c: fork path double-fetches the full conversation CitationTraceBuilder re-encodes all payloads per record and lore listing runs N+1 entry-count queries
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fork performs one full-conversation read,Citation trace performs N encodings for N records with cap intact,Lore counts one GROUP BY query for 10 books
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 19 (T19)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Three fixes: (19a) effective_active_leaf gains keyword-only rows= reuse — selection logic shared verbatim below the guard (structural equivalence, pinned by live/dangling/empty tests), copy_conversation_active_path passes its already-fetched page, fork now performs exactly one source read (spy); (19b) CitationTraceBuilder._ensure_governed_payload_capacity keeps a running _governed_payload_bytes counter — each record canonicalizes only its proposed payloads once (N records -> N encodings, spy on the single encoding seam), counter exactness proven by frozen=True on all three payload models, >cap raise preserves compute-then-raise atomicity, seal-time full recomputation retained as backstop; (19c) WorldBookManager.count_entries_for_books via one parameterized GROUP BY (semantics verified identical to the default get_world_book_entries — entries table has no deleted column, books carry the flag), personas _list_world_books_with_counts makes one call — 10 books, zero per-book queries (spy + equivalence pin). Files: Chat/chat_conversation_service.py, Chat/citation_trace_builder.py, Character_Chat/world_book_manager.py, UI/Screens/personas_screen.py + 4 test files. Report: .superpowers/sdd/task-19-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
