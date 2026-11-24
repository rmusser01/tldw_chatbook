---
id: TASK-41178
title: Persist article listening and owned audio assets
status: Done
assignee:
  - '@codex'
created_date: '2026-11-15'
updated_date: '2026-11-24T21:37:00Z'
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
- [x] #1 Old profiles migrate transactionally and fresh profiles support one durable listening session with independent entry bookmarks and revision-checked atomic transitions.
- [x] #2 Source-bound private artifacts publish safely and survive referenced retention while missing, corrupt, interrupted or orphaned files have explicit recovery.
- [x] #3 Deletion, admission and publication races cannot restore deleted derivatives; membership deletion preserves surviving sources and competing destructive writers coordinate or fail busy.
- [x] #4 Backup and restore support the new schema and assets together, with owner-present and owner-absent, relocation, rollback and DB-only restore evidence.
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented SubscriptionsDB v4 listening state, source-bound private WAV assets
and integrated subscriptions backup/restore under
[ADR-367](../decisions/367-watchlist-article-listening-and-controlled-playback.md).
Each selector has independent durable queue/foreground bookmarks with revision
and incarnation checks. A stable DB-sibling native lease serializes destructive
work across processes. Source deletion scrubs private state and commits retryable
tombstones; an explicitly bound asset owner performs physical cleanup.

Publication and removal retain the exact qualified Windows NTFS object through
the effect and durability barrier. Full WAV validation precedes reference
attachment. All retained roles protect their generation; uncertain foreign,
unsealed or restored objects are preserved. Missing files retain bookmark
identity until explicit regeneration. Unsupported effect platforms refuse.

The existing subscriptions.assets owner covers both trees. Recovery preserves
historical catalogs, DB-only state and exact local safety snapshots; incoming
images clear foreign cleanup custody. Reviewed selector mappings apply once to
each original partition, including shared DBs and empty overlapping aliases.
Absence retirement is installed-root-only. Rollback proves finite membership
before reading approved identities through the bounded scoped reader. Runtime
maintenance drains physical holders and never starts playback.

Final targeted Windows integration passed 150 cases in 357.45s: full constructor,
read-only, store and asset files, citation controls, four real shared-directory
capture/restore layouts, layer direction and source census. The recovery neighbor
run passed 96 cases, including migration, historical subscriptions catalogs,
detached restore and single-selector rollback. One new fixture enrollment timeout
was corrected and passed in the final integration. Seven broader tests also fail
on parent557df66c73: one Windows casefold assertion identically; six primary v76
fixtures fail earlier on the parent, while repaired frozen-SQL fixtures expose
existing v80 expectations after a successful current-v87 upgrade. Primary recovery
code was unchanged. These selections overlap; no full suite ran. Prior native
file primitives passed 59 cases. Detached restore ran with both earlier layouts
removed and provider/player tripwires active.

Independent A/B/C reviews and the whole-task review's scoped correction are
clear of Critical/Important findings. All eight final findings were addressed;
the reviewer explicitly accepted AC4's documented rollback boundary. The user
recovery guide now describes that refusal before replacement.
All 45 changed Python files pass Ruff, including the two inherited diagnostics
corrected without changing eager read-only initialization. All 13 new Python
files pass whole-file formatting; 53 edited ranges pass formatting. Base-to-head
whitespace checks pass. Committed playback receipts pass their unchanged grader.

Storage effects are Windows-first. Shared-directory capture requires verified
selected participants. Unknown or unselected sibling files refuse. Later rollback
of a newly created multi-selector replacement root remains a conservative
finite-membership/retirement refusal; the test preserves both configs, DB, both
WAVs and control records. Single-selector rollback remains verified. Linux/macOS
qualification, synthesis, transport and listening UI remain separate work.
Issue2864 stays open. Living internals:
[subscriptions architecture](../../Docs/Architecture/subscriptions-watchlists.md).

One nonblocking performance follow-up remains: detached import can copy an
unclaimed WAV from an already privately acquired tree into private staging
before its final dependency check refuses sealing. Existing import limits and
current-profile protection remain enforced; no playback starts.
<!-- SECTION:NOTES:END -->
