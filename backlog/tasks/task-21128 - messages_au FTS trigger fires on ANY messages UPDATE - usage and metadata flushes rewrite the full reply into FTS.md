---
id: TASK-21128
title: >-
  messages_au FTS trigger fires on ANY messages UPDATE - usage and metadata flushes rewrite the full reply into FTS
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - performance
  - database
  - fts
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `35d4bf3a1`. Evidence, measurements, and file:line cites: `Docs/Design/2026-08-22-holistic-perf-review.md` (finding 21128).

The `messages_au` trigger (recreated, still unconditional, in the v46 migration SQL ~lines
396-405) has no `UPDATE OF content` column list, so usage-only and metadata-only flushes - now
3-4 UPDATEs per chat turn (content finalize, usage flush ChaChaNotes_DB.py:11030-11082,
metadata flush :11098+) - each re-tokenize and rewrite the full assistant reply into
`messages_fts`. WAL+NORMAL, so write amplification rather than fsync storm.

## Acceptance Criteria

- [ ] The trigger fires only on content changes (`AFTER UPDATE OF content ON messages`), preserving the v46 deleted-guards
- [ ] A migration (with version bump per repo policy) ships the trigger change; FTS search results for edited/streamed messages remain correct under existing tests
- [ ] A write-count probe over one streamed turn shows the FTS rewrite count drop from 3-4 to 1
