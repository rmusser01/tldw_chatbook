---
id: TASK-31242
title: Build character conversation projection and Keyword index
status: To Do
assignee: []
created_date: '2026-09-04 02:05'
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

## Renumbering provenance

Renumbered from TASK-31234 on 2026-09-04. The final pre-commit worktree sweep
found the older `Auto resume always lands on the cursor item` task created at
01:50; it keeps TASK-31234 under the older-arrival rule. This unshipped task
moves with all plan and dependency references.

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Provide one authority-safe local read model and selected-branch Keyword corpus for every later Character navigation surface, including deterministic unresolved-link recovery services without exposing new UI.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Resolved and unresolved identities are tagged, bounded, data-authority-scoped, and never inferred from names, display text, filesystem paths, or current selection.
- [ ] #2 The projection returns bounded recent groups, global Keyword results, exact totals, and stable keyset pages from local conversations only.
- [ ] #3 A canonical eligibility projector includes only the selected visible user/assistant branch and fails closed for invalid branch graphs.
- [ ] #4 A separate versioned FTS generation is built and maintained without reusing messages_fts or exposing excluded message content.
- [ ] #5 Unique legacy authority links backfill deterministically; ambiguous, missing, and deleted links remain typed Unavailable results.
- [ ] #6 Library repair candidates are same-authority only and compare-and-set repair invalidates affected derived search state.
- [ ] #7 ensure_keyword_index is dormant until an owning UI calls it and performs no startup background work.
- [ ] #8 Migration, isolation, paging, eligibility, concurrency, performance, and no-server-canary tests pass on real SQLite fixtures.
<!-- AC:END -->
