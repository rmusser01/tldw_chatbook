---
id: TASK-41178
title: Persist article listening and owned audio assets
status: To Do
created_date: 2026-11-15
assignee: []
labels:
  - watchlists
  - tts
priority: high
references:
  - https://github.com/rmusser01/tldw_chatbook/issues/2864
dependencies:
  - TASK-41177
documentation:
  - backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep saved listening recoverable and consistent with source data, deletion and profile backup across crashes and competing processes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Old profiles migrate transactionally and fresh profiles support one durable listening session with independent entry bookmarks and revision-checked atomic transitions.
- [ ] #2 Source-bound private artifacts publish safely and survive referenced retention while missing, corrupt, interrupted or orphaned files have explicit recovery.
- [ ] #3 Deletion, admission and publication races cannot restore deleted derivatives; membership deletion preserves surviving sources and competing destructive writers coordinate or fail busy.
- [ ] #4 Backup and restore support the new schema and assets together, with owner-present and owner-absent, relocation, rollback and DB-only restore evidence.
<!-- AC:END -->
