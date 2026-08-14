---
id: TASK-16221
title: >-
  Make Watchlists Read list and Content independently scrollable with explicit
  pagination
status: Done
assignee: []
created_date: '2026-08-14 03:23'
updated_date: '2026-08-14 08:29'
labels:
  - watchlists
  - ui
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-13-watchlists-nested-scroll-pagination-design.md
  - Docs/superpowers/plans/2026-08-13-watchlists-nested-scroll-pagination.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make long Watchlists reading sessions usable by giving the centre column, Read list, and Content reader distinct scroll ownership while preserving the fixed rails and existing reader interactions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Watchlists centre column scrolls vertically while the left Watchlists rail and right Inspector remain fixed.
- [x] #2 On Read, the item-list region sizes from 10 to 50 terminal rows and its ListView scrolls internally after reaching the cap.
- [x] #3 On Read, the Content region sizes from 20 to 50 terminal rows and its article body scrolls internally after reaching the cap.
- [x] #4 The Read list exposes explicit Previous, Page N, and Next controls with 50 visible items per backend page and correct boundary/loading states.
- [x] #5 Page state survives scoped region replacement, resets for agreed context changes, corpus-search results remain authoritative, and failed explicit page loads preserve the current page and selected Content.
- [x] #6 Existing collapse, solo/restore, mark-read, filtering, and in-page reader navigation behavior remains functional.
- [x] #7 Focused automated geometry, pagination, state, and error tests pass, and isolated live TUI QA verifies nested scrolling at representative terminal sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add presentation-only pager controls, authoritative-search state, and non-selecting first-row focus to ArticleListPane with widget tests.
2. Add screen-owned 50-item page state, 51-row lookahead queries, transactional Previous/Next handlers, and focused pagination tests.
3. Enforce context resets, stale-result guards, page/query provenance for the open-item pin, authoritative corpus-search results, and empty-page fallback.
4. Convert only the workbench centre to VerticalScroll, add the Read-mode class, and bound the Read list to 10–50 rows with an internal ListView scroll.
5. Wrap only the rendered Content body in VerticalScroll and bound the Read Content region to 20–50 rows while keeping actions/footer fixed.
6. Regenerate CSS, run focused/broad/static/live verification, self-review, document evidence, and close TASK-16221 only when all DoD gates pass.

Detailed plan: Docs/superpowers/plans/2026-08-13-watchlists-nested-scroll-pagination.md
ADR required: no
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: contained UI/pagination refinement within the existing reader-first boundaries; no schema, dependency, service contract, or new application structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a reader-first nested-scroll layout for Watchlists. The centre is now the outer vertical document while both rails remain fixed; Read ITEMS grows from 10 to 50 rows with a 40-row inner ListView, explicit Previous/Page N/Next controls, and transactional 50-item backend paging; Content grows from 20 to 50 rows with a 45-row inner body, fixed actions/footer, compact keyboard-reachable horizontal action overflow, and identity-preserving solo/restore. Pagination now handles context resets, authoritative corpus search, stale/cancelled presentation rollback, same-page selection provenance, and empty-page fallback without exposing uncommitted rows.

Verification: final pagination 33 passed; exact pagination/context/content regression set 131 passed; focused nine-file integration run 463 passed; broad Watchlists run 1032 passed before two narrow test-fixture corrections, followed by 14/14 highlight and 8/8 final authority rollback checks. Production-CSS live QA seeded 101 items and an 80-paragraph article at 180x50 and 120x36, verifying 50/50/1 pages, fixed rails, independent centre/list/body scrolling, rendered pager/actions/article end/footer, body-only search, and solo/restore. Ruff, CSS build/bundle sync, git diff checks, Impeccable layout checks, spec review, and final independent code review passed. Known broad-suite failures were classified as pre-existing midnight-sensitive date fixtures, sandbox socket PermissionErrors, an obsolete source-audit expectation, and unrelated non-Watchlists visual-parity failures.

ADR required: no. Existing reader-first boundary: backlog/decisions/042-watchlists-reader-first-ia.md. Added an evidence-backed production-hierarchy/stylesheet harness lesson to backlog/docs/lessons-testing-evidence.md.
<!-- SECTION:NOTES:END -->
