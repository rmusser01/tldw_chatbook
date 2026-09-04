---
id: TASK-31242
title: Build character conversation projection and Keyword index
status: Done
assignee: []
created_date: '2026-09-04 02:05'
updated_date: '2026-09-04 04:36'
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
1. Confirm schema v65 remains the highest shipped allocation and implement one guarded v65→v66 migration with fresh-schema parity.
2. Define and RED/GREEN-test immutable authority-scoped identity, projection, paging, repair, generation, and Keyword status contracts.
3. RED/GREEN-test the one-transaction selected-branch eligibility projector and fail-closed branch validation.
4. RED/GREEN-test and implement the separate external-content FTS schema, dormant generation build, authority-safe projection queries, legacy link handling, and CAS repair.
5. Run only Task 2 targeted pytest, Ruff, git diff --check, migration parity/census, self-review, task/ADR hygiene, and commit the scoped slice.

ADR required: no
ADR path: backlog/decisions/116-character-conversation-navigation-and-local-semantic-search.md
Reason: This task directly implements ADR-116’s already-approved identity, storage, projection, repair, and Keyword boundaries.
Final schema version: 66 (v65→v66; origin/dev remains v65 at implementation start).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fix Rounds 1–2 resolve all independent-review findings. Recent projection is section-first and SQL-bounded, force-includes zero-chat/old current groups, and reserves unavailable capacity. Keyword construction streams LIMIT 128 ID batches, validates branches linearly, durably applies dirty-only replacement/removal with reconciliation/restart recovery/cleanup, and exposes typed revision-aware status. Search selection and projector validation share one snapshot with a final revision retry fence. Round 2 adds a maintained validated-eligibility digest fence used by both FTS COUNT and candidate LIMIT/OFFSET, so all pages and exact totals describe the same validated set even when a corrupt candidate lies before a nonzero offset or beyond the first 50 results; projector uncertainty still fails closed. Exact targeted pytest: 43 passed; scoped Ruff new files/changed surface and E4/E7/E9/F: passed; git diff --check: passed. Ruling: the reviewer-approved scoped Ruff gate is valid; 588 all-rules ChaChaNotes_DB.py diagnostics are pre-existing and outside Task 2 changed hunks. Schema remains unshipped v66 with fresh/upgrade parity; no shipped migration was rewritten. ADR required: no; ADR-116 governs. No UI files changed and no generalized lesson was required.
<!-- SECTION:NOTES:END -->
