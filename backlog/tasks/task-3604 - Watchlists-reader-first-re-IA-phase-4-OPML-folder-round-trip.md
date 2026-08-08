---
id: TASK-3604
title: 'Watchlists reader-first re-IA, phase 4: OPML folder round-trip'
status: In Progress
assignee: []
created_date: '2026-08-08 22:40'
updated_date: '2026-08-08 22:40'
labels:
  - watchlists
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement phase 4 of the reader-first design (Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md, ADR-042/043): map OPML folders to watchlists on import (innermost folder wins, case-insensitive reuse, top-level feeds stay Unassigned, additive only) and nest watchlists as folders on export, so the structure round-trips losslessly. The spec's polish tasks 2308/2310/2312/2313 are already Done. Plan: Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-4-opml-round-trip.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 OPML parse preserves folder structure (innermost folder wins, feed-with-children stays a feed),Import creates/reuses watchlists by folder name case-insensitively, assigns member sources, leaves top-level feeds Unassigned, and returns an honest summary,Export nests one folder per watchlist with member feeds, deterministic order, hostile names escaped,A fresh-DB import of an exported document reproduces the exact watchlist structure (round-trip pin),Folderless OPML behaves exactly as before,Tests/Subscriptions and Tests/Watchlists green, ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-4-opml-round-trip.md

ADR required: yes (created)
ADR path: backlog/decisions/043-opml-watchlist-folder-mapping.md
Reason: the folder-to-watchlist mapping is an interchange/conflict policy (naming, nesting, reuse, additive-only) that ADR-042 does not cover.
<!-- SECTION:PLAN:END -->
