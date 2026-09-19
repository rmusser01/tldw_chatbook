---
id: TASK-32870
title: Add Library Reports reader with independent kept copies
status: To Do
assignee: []
created_date: '2026-09-19 21:57'
labels:
  - library
  - artifacts
dependencies:
  - TASK-32869
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make all local report copies discoverable through the existing Library reader layout, with an explicit Kept filter and durable access after Watchlist deletion. Keep generation, retention, and playback with their current owners. Governed by ADR-172 and Docs/superpowers/specs/2026-09-19-library-artifacts-design.md; execution is stage 1 of Docs/superpowers/plans/2026-09-19-library-artifacts.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reports default to All reports, Kept shows durable copies independently of Subscriptions, and imported source-ID collisions never hide or mislabel content.
- [ ] #2 Every matching report is reachable through bounded pages; detail loads and actions are fenced to the current selected identity.
- [ ] #3 Library geometry, field-first Escape, focus restoration, local scope, artifact-only onboarding, and truthful error recovery work with production CSS.
- [ ] #4 Existing Keep, report export, Watchlists handoff, scripts, audio, and demo recovery capabilities remain accessible and targeted checks pass.
<!-- AC:END -->
