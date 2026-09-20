---
id: TASK-32870
title: Add Library Reports reader with independent kept copies
status: Done
assignee:
  - '@codex'
created_date: '2026-09-19 21:57'
updated_date: '2026-09-20 07:18'
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
- [x] #1 Reports default to All reports, Kept shows durable copies independently of Subscriptions, and imported source-ID collisions never hide or mislabel content.
- [x] #2 Every matching report is reachable through bounded pages; detail loads and actions are fenced to the current selected identity.
- [x] #3 Library geometry, field-first Escape, focus restoration, local scope, artifact-only onboarding, and truthful error recovery work with production CSS.
- [x] #4 Existing Keep, report export, Watchlists handoff, scripts, audio, and demo recovery capabilities remain accessible and targeted checks pass.
- [x] #5 A Keep conflict is refused inside the service before any parent or script mutation, including a concurrent-create conflict; compatible re-keeps preserve saved origin and still add complete scripts.
- [x] #6 Each page and exact-item locator reads coherent metadata and rows from an explicit read snapshot; coordinated WAL writes cannot split a response and borrowed transactions retain their owner.
- [x] #7 Enter focuses the reader, reader arrows preserve list selection, and Back or unconsumed Escape restores the selected list row before arrows navigate it again.
- [x] #8 Library suspend stops owned debounce timers and stale presentation effects while compatible data reads may settle without leaving Loading stranded on resume.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes. ADR path: backlog/decisions/172-library-artifacts-browse-and-navigation.md. Reason: Implements the accepted Library browse and navigation contract. 1. Add coherent bounded report metadata reads, namespaced identity and exact target location. 2. Enforce service-owned Keep conflict checks before all writes. 3. Mount Reports and Kept in the shared Library reader with lifecycle and focus guards. 4. Verify targeted SQLite, Library, production CSS and live interaction evidence; record results.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Reports and Kept in Library’s shared adaptive reader with bounded snapshot pages, exact identities, durable export/scripts/audio/watchlist actions, and service-owned Keep compatibility guards. Real tests cover imported collisions, rollback, borrowed transactions, source deletion, delayed replies, query and focus, and stale-action Retry. Final reader regressions passed, including native Enter/Down/Escape. Behavior lives in focused catalog/controller/widget modules with thin screen wiring. ADR: backlog/decisions/172-library-artifacts-browse-and-navigation.md. Evidence and explicit baseline limits: Docs/superpowers/plans/2026-09-19-library-artifacts-verification.md. User guide updated. No full suite requested; changed/new focused tests pass, added-line Ruff diagnostics are zero, generated CSS checks pass, and native private-profile TldwCli was verified. Existing recovery-initialization and repository-wide screen-size/workflows-style failures remain documented; their limits were not raised.
<!-- SECTION:NOTES:END -->
