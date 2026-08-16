---
id: TASK-16482
title: Align Library Prompt browsing to 20-item pages
status: In Progress
assignee: []
created_date: '2026-08-15 02:46'
updated_date: '2026-08-16 00:49'
labels:
  - library
  - pagination
  - prompts
dependencies:
  - TASK-16481
references:
  - >-
    Docs/superpowers/specs/2026-08-14-library-top-level-source-pagination-design.md
  - backlog/decisions/067-library-top-level-pagination-contracts.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make all Prompts reachable through the approved 20-item Library pager while preserving Prompt-specific browse, debounce, source authority, mutation history, and versioned cross-page selection behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Prompt browse uses 20-item pages with exact applied ranges, totals, Previous, Next, loading, and recoverable error presentation.
- [ ] #2 Search, sort, and Prompt collection scopes apply to the complete source before paging and successful scope changes start on page 1.
- [ ] #3 The existing version-captured Prompt selection basket remains cross-page; paging or scope changes neither clear nor implicitly add entries.
- [ ] #4 Prompt normalized current_page, page alias, per_page, exact total, cardinality, and stable identities are validated; malformed envelopes fail closed.
- [ ] #5 An out-of-range Prompt request applies its single coherent clamped response without a redundant second service call.
- [ ] #6 Prompt focus, navigation restoration, stale-generation, unmount, and dedicated-request isolation behaviors match the approved design.
- [ ] #7 Automated service/state, mounted Textual, geometry, mutation, and isolated live verification pass with no regression to Prompt history or source behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/067-library-top-level-pagination-contracts.md
Reason: direct implementation of the accepted source-owned Prompt pagination contract; no storage, runtime, or ownership boundary changes.

Detailed plan: Docs/superpowers/plans/2026-08-15-task-16482-library-prompt-pagination.md

1. Pin the Library Prompt scope default to 20 and verify its explicit DB/service propagation without changing generic API defaults, coherent transactions, clamp, filter, or stable order.
2. Preserve and validate normalized current_page/page/per_page, exact totals/cardinality, and stable unique local identities.
3. Extend the existing Prompt browse controller with requested/applied last-good recovery state and the shared pure pager display.
4. Wire restore, debounce, focus, generations, unmount, broad-snapshot isolation, and the immutable cross-page selection basket through mounted tests.
5. Render the Prompt-specific pager at both supported geometries with retained rows, disabled reasons, and Retry.
6. Preserve delete/undo/history/version behavior and add committed-but-stale recovery.
7. Run inverse mutations, owner/live/privacy/static gates, reviews, docs, and Backlog closeout.
<!-- SECTION:PLAN:END -->
