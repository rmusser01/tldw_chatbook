---
id: TASK-22451
title: Add contextual aggregate feed selection to Watchlists Navigation
status: In Progress
assignee: []
created_date: '2026-08-26 04:30'
updated_date: '2026-08-27 07:04'
labels:
  - watchlists
  - reader
dependencies:
  - TASK-22450
references:
  - >-
    Docs/superpowers/specs/2026-08-25-watchlists-read-aggregate-feed-selection-design.md
  - backlog/decisions/042-watchlists-reader-first-ia.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose individual feeds beneath All Sources, Unassigned, and All Unread throughout the shared Watchlists Navigation rail, preserving each parent context and consuming the stable Feed Items snapshot foundation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All Sources, Unassigned, and All Unread expand independently and show complete, stably sorted contextual feed children without N+1 queries.
- [ ] #2 Selecting a child produces the correct contextual source scope, breadcrumb, item predicate, paging identity, and exact active occurrence.
- [ ] #3 All Unread shows only feeds with unread items, forces the effective Unread filter without overwriting the parked manual filter, and safely pins a selected zero-count feed.
- [ ] #4 Pending, failed, superseded, deleted, and membership-changing scopes reconcile without relabeling stale rows or losing focus and expansion state.
- [ ] #5 Aggregate branches remain expandable throughout Watchlists; server-backed management tabs disable local feed selection and park the local scope honestly.
- [ ] #6 Affected Watchlists tests, modified-file Ruff, and branch diff checks pass.
<!-- AC:END -->

## Implementation Plan

1. Repair the two inherited Reader recovery tests that still mock the retired `list_items()` path, then establish a clean focused baseline against the typed `list_reader_items_page()` contract.
2. Extend `TreeScope` with contextual source-parent authority and give aggregate roots and watchlists independent screen-owned expansion state.
3. Make the stable tree snapshot own complete All Sources and Unassigned rows plus existing bulk counts off the event loop, publishing only the latest generation and retaining branch-local stale/error state without reusing capped management data or introducing N+1 queries.
4. Carry contextual source authority through breadcrumbs, exact Reader empty/failure copy, Reader predicates, paging keys, atomic pending/commit behavior, and exact active occurrence styling.
5. Drive query, paging, empty-state, and status-control behavior from one effective Unread decision while parking the manual filter; keep local management activation atomic and server management feed selection honestly disabled.
6. Reconcile invalid committed/pending, deleted, membership-changing, read-write-failing, and zero-unread scopes while preserving the open item, view positions, focus, and expansion.
7. Run only affected Watchlists tests, modified-file Ruff, the Impeccable UI detector, and branch diff checks; review the diff and complete task documentation.

**ADR required:** no

**ADR path:** `backlog/decisions/042-watchlists-reader-first-ia.md`

**Reason:** ADR-042 already owns the Reader-first navigation, stable snapshot, and atomic scope-commit boundaries. This task extends contextual occurrences inside those boundaries without changing storage, runtime, or service ownership.

Detailed execution plan: `Docs/superpowers/plans/2026-08-27-watchlists-contextual-aggregate-feed-selection.md`
