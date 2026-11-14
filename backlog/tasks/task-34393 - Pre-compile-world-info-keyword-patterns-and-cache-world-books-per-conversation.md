---
id: TASK-34393
title: Pre-compile world-info keyword patterns and cache world books per conversation
status: Done
created_date: 2026-10-11 21:44
labels:
- performance
priority: medium
updated_date: 2026-10-12 01:30
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Per character-chat send: _collect_active_world_books re-queries the DB, get_world_books_for_conversation issues one get_world_book_entries query per book (N+1), and world_info_processor._keyword_in_text builds and runs a word-boundary regex per key x per entry x per message over the full scan text. Cache books/entries per conversation with invalidation on attachment change; pre-compile each key's pattern at entry load.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Keyword patterns are compiled once per entry/key, not per message,World-book entries are not re-queried from the DB on every send when nothing changed,Activation/matching behavior identical to the regex-per-call implementation (word boundaries, secondary keys),Existing world-info tests pass
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
