---
id: TASK-31242
title: Build character conversation projection and Keyword index
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 02:05'
updated_date: '2026-09-05 20:19'
labels:
  - database
  - search
  - characters
dependencies:
  - TASK-31241
references:
  - >-
    Docs/superpowers/specs/2026-09-03-character-conversation-navigation-design.md
  - >-
    Docs/superpowers/plans/2026-09-03-character-conversation-navigation-implementation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide one authority-safe local read model and selected-branch Keyword corpus for every later Character navigation surface, including deterministic unresolved-link recovery services without exposing new UI.
<!-- SECTION:DESCRIPTION:END -->

## Renumbering provenance

Renumbered from TASK-31234 on 2026-09-04. The final pre-commit worktree sweep
found the older `Auto resume always lands on the cursor item` task created at
01:50; it keeps TASK-31234 under the older-arrival rule. This unshipped task
moves with all plan and dependency references.

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Resolved and unresolved identities are tagged, bounded, data-authority-scoped, and never inferred from names, display text, filesystem paths, or current selection.
- [x] #2 The projection returns bounded recent groups, global Keyword results, exact totals, and stable keyset pages from local conversations only.
- [x] #3 A canonical eligibility projector includes only the selected visible user/assistant branch and fails closed for invalid branch graphs.
- [x] #4 A separate versioned FTS generation is built and maintained without reusing messages_fts or exposing excluded message content.
- [x] #5 Unique legacy authority links backfill deterministically; ambiguous, missing, and deleted links remain typed Unavailable results.
- [x] #6 Library repair candidates are same-authority only and compare-and-set repair invalidates affected derived search state.
- [x] #7 ensure_keyword_index is dormant until an owning UI calls it and performs no startup background work.
- [x] #8 Migration, isolation, paging, eligibility, concurrency, performance, and no-server-canary tests pass on real SQLite fixtures.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reviewed Task2 replay and qualify its isolated PR tree on PR1 head 5539118b; preserve delivery 74993748e and all source/safety refs.
2. Carry audited Task2-owned later fixes and opt-in SQLite owner cleanup; exclude UI, activation, Meaning, settings, and unrelated shared-file changes.
3. Reproduce complete descending (last_modified, created_at, conversation_id) paging on real SQLite before correcting cursor, predicates, browse ordering, and next cursors; cover ties, limit-one traversal, mutation refresh, and unchanged-row continuity.
4. Verify exact identity/migration/projection/selected-branch/ownership tests and affected migration/derived-index guards; run scoped Ruff/format, inherited diagnostic comparison, whitespace and reference checks. Rebase onto merged PR1 dev only after tests stop.
ADR required: no
ADR path: backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md
Reason: faithful existing authority/storage/pagination contract.
Final schema version: 66; current origin/dev 7aa048790 is schema 65, preserving shipped migrations.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prepared the independently reviewable Task2 projection/Keyword slice from PR1 5539118b. Replayed the four reviewed Task2 commits and carried later owning fixes for strict identity decoding, exact-character Keyword filtering, unavailable paging, fenced repair/preview reads and opt-in SQLite resource evidence. Corrected cursor, row metadata, SQL keyset predicates and resolved/unavailable recent ordering to descending (last_modified, created_at, conversation_id), preserving Keyword relevance ordering and all authority/revision fences. Five real-SQLite ordering cases failed before the correction and pass afterward. Existing source timestamps required no additional migration: schema remains unshipped v66 after shipped v65.
Packaging qualification also registered the dedicated DDL module with the schema guard, allowlisted its five actual tables, updated the current-version migration pin, and recorded no-statistics plans for the five existing indexes. A focused RED/GREEN guard fix recognizes standalone index names beyond idx_/uq_ while still excluding negative assertions. No threshold, inventory exclusion or pre-convention exemption was added.
Final exact identity/migration/projection/selected-branch/ownership plus affected schema/SQL-validation/index-guard gate: 124 passed, 1 environment RequestsDependencyWarning in 26.14s. Opt-in cleanup: 65 owners, 40 handles drained, 0 registered remaining; process FD delta +13 is subset evidence, not whole-process resource clearance. New-file Ruff/format passed; legacy Ruff diagnostic counts and message multisets unchanged (ChaChaNotes 588, sql_validation 6, schema checker 1, index checker 1). Schema/index/task-ID, CSS bundle, profile-owned-path and diagnostic inventories passed. Complete PR-range whitespace passed.
ADR required: no; backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md governs. Lesson added for the observed guard source/name assumptions. Task remains In Progress pending independent review, controller-owned PR publication and final integration onto merged PR1/current dev; no full suite, native app or Meaning/UI work was performed.
<!-- SECTION:NOTES:END -->
