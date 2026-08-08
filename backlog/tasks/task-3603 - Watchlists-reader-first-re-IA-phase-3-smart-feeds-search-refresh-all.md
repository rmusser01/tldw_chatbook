---
id: TASK-3603
title: 'Watchlists reader-first re-IA, phase 3: smart feeds, search, refresh-all'
status: In Progress
assignee: []
created_date: '2026-08-08 15:44'
updated_date: '2026-08-08 15:46'
labels:
  - watchlists
  - ux
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement phase 3 of the reader-first design (Docs/superpowers/specs/2026-08-05-watchlists-reader-first-design.md, ADR-042): All Unread + Today rail nodes beside Starred, / corpus-wide search via subscription_items_fts with LIKE fallback, r refresh-all with guardrails + aggregated notification, and an N-new-items pill. Plan: Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-3-smart-feeds-search-refresh-all.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rail shows All Unread and Today smart feeds with correct badges that refresh through the existing counts path,/ focuses the items search and results span the whole corpus via FTS5, with LIKE fallback when FTS is unavailable and no FTS-syntax errors on hostile input,r checks every active non-paused source exactly once per press, toasts one aggregated summary with the unread delta, and shows an N-new-items pill that never yanks the list mid-triage,Help text advertises / and r (decision 031),Tests/Watchlists and Tests/Subscriptions green, ruff clean
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-08-08-watchlists-reader-first-phase-3-smart-feeds-search-refresh-all.md

ADR required: no (already exists)
ADR path: backlog/decisions/042-watchlists-reader-first-ia.md
Reason: ADR-042 covers the re-IA; phase 3 is a direct implementation of it, same ruling as phase 2.
<!-- SECTION:PLAN:END -->
