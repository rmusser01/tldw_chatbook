---
id: TASK-16482
title: Align Library Prompt browsing to 20-item pages
status: To Do
assignee: []
created_date: '2026-08-15 02:46'
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
- [ ] #2 Search, sort, type, and Prompt collection scopes apply to the complete source before paging and successful scope changes start on page 1.
- [ ] #3 The existing version-captured Prompt selection basket remains cross-page; paging or scope changes neither clear nor implicitly add entries.
- [ ] #4 Prompt normalized current_page, page alias, per_page, exact total, cardinality, and stable identities are validated; malformed envelopes fail closed.
- [ ] #5 An out-of-range Prompt request applies its single coherent clamped response without a redundant second service call.
- [ ] #6 Prompt focus, navigation restoration, stale-generation, unmount, and dedicated-request isolation behaviors match the approved design.
- [ ] #7 Automated service/state, mounted Textual, geometry, mutation, and isolated live verification pass with no regression to Prompt history or source behavior.
<!-- AC:END -->
