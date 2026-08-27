---
id: TASK-22451
title: Add contextual aggregate feed selection to Watchlists Navigation
status: To Do
assignee: []
created_date: '2026-08-26 04:30'
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
