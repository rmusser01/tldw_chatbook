---
id: TASK-32870
title: Add Library Reports reader with independent kept copies
status: To Do
assignee: []
created_date: '2026-09-19 21:57'
updated_date: '2026-09-20 06:08'
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
- [ ] #5 A Keep conflict is refused inside the service before any parent or script mutation, including a concurrent-create conflict; compatible re-keeps preserve saved origin and still add complete scripts.
- [ ] #6 Each page and exact-item locator reads coherent metadata and rows from an explicit read snapshot; coordinated WAL writes cannot split a response and borrowed transactions retain their owner.
- [ ] #7 Enter focuses the reader, reader arrows preserve list selection, and Back or unconsumed Escape restores the selected list row before arrows navigate it again.
- [ ] #8 Library suspend stops owned debounce timers and stale presentation effects while compatible data reads may settle without leaving Loading stranded on resume.
<!-- AC:END -->
