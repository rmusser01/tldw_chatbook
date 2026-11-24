---
id: TASK-41178
title: Persist article listening and owned audio assets
status: In Progress
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-24T16:50:00Z'
labels:
  - watchlists
  - tts
dependencies:
  - TASK-41177
references:
  - https://github.com/rmusser01/tldw_chatbook/issues/2864
documentation:
  - backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
  - Docs/superpowers/specs/2026-11-15-watchlist-article-listening-design.md
  - Docs/superpowers/plans/2026-11-15-watchlist-article-listening.md
priority: high
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/367-watchlist-article-listening-and-controlled-playback.md
Reason: implement the accepted SubscriptionsDB ownership, private assets, cross-process mutation and backup contracts; amend this ADR before any concrete boundary change.
1. Recheck schema v3, source/deletion paths and current recovery owners at base 806caa420d (dev 9fa2bf1b1d). Freeze the minimal schema, immutable store/asset types, generation and mutation-lease contracts after read-only preflight. Preserve historical recovery catalogs.
2. Add real-SQLite RED/GREEN migration, CAS/checkpoint, source/admission/deletion race and nonblocking process-lease tests. Implement the v3-to-v4 SQL artifact and store with atomic queue/foreground bookmarks and source-bound invalidation; review this foundation before dependent work.
3. Add bounded private WAV publication, validation, same-item reuse, missing/corrupt/interrupted/orphan reconciliation and reference-protected seven-day/512 MiB cleanup. Keep generation/source checks in the publication transaction and retain uncertain filesystem custody.
4. Extend subscriptions.assets and core dependencies, exact schema catalogs/migration/authorizer, detached layout/staging/relocation, source census and runtime quiescence/storage admission together. Preserve existing briefing-only archive identities and owner-absent paths.
5. Run focused migration/store/assets and changed recovery suites serially with normal conftest/profile guards and private test storage; establish any baseline failure on the compared base. Verify owner present/absent, DB-only/assets restore, rollback, relocation, crash-release and competing deletion/publication evidence.
6. Complete independent incremental and whole-task reviews, resolve important findings with covering regressions, update architecture/ADR/plan/task notes, and mark Done only when all four AC pass. Publish a separate draft child PR; no synthesis, transport service or listening UI implementation in this task.
<!-- SECTION:PLAN:END -->
