---
id: TASK-19022
title: Add Library starter rail and lifecycle-aware landing
status: In Progress
assignee: []
created_date: '2026-08-20 20:53'
labels:
  - library
  - ux
  - onboarding
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-08-20-library-lifecycle-progressive-disclosure-design.md
  - >-
    Docs/superpowers/plans/2026-08-20-library-starter-rail-landing.md
  - backlog/decisions/076-library-lifecycle-progressive-disclosure.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give new empty profiles a compact production-path Get started experience while preserving the full Library for existing users and power users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A new empty profile sees Import, New note, and Explore all tools without a false empty claim while evidence is unresolved.
- [ ] #2 Any authoritative eligible user content permanently graduates the profile; bundled, sample, trash-only, inaccessible, and failed-import records do not.
- [ ] #3 Explore all tools persists independently of section collapse, and deep links or command-palette routes remain reachable.
- [ ] #4 Legacy profiles without a lifecycle preference default to the expanded Library, while corrupt preferences fail safely to expanded.
- [ ] #5 Starter and transition states are keyboard complete, announced in text, focus-safe, and usable at 100x30 and 170x48.
- [ ] #6 Only modified/touched component and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

1. Add the pure lifecycle/evidence contracts and commit ADR-076 with backward-compatible preference coercion.
2. Add source-owned provenance-aware tri-state evidence seams for Notes, Media, Conversations, Prompts, Skills, and Collections without returning records or private content.
3. Make LibraryScreen own one generation-fenced evidence aggregation, serialized lifecycle persistence, restart restoration, truthful loading/failure status, and unmount authority.
4. Render the compact Starter rail with production Import/New note routes, remembered Explore disclosure, and deep-link bypass.
5. Render the lifecycle-aware landing with truthful unresolved/recovery copy, settled graduation, semantic focus preservation, and both supported geometries.
6. Run only touched/direct-owner tests and static checks, complete production-hierarchy mounted UAT at 100x30 and 170x48, obtain independent reviews, update docs, and close through Backlog CLI. Isolated-profile live UAT remains owned by Wave 1 closeout.

Detailed TDD steps, exact files, commands, inverses, and commit boundaries are in `Docs/superpowers/plans/2026-08-20-library-starter-rail-landing.md`.

ADR required: yes
ADR path: `backlog/decisions/076-library-lifecycle-progressive-disclosure.md`
Reason: The task adds a long-lived profile-local UX lifecycle, persistence contract, source-evidence boundary, and navigation-disclosure policy. ADR-067 remains authoritative for source paging and data ownership.
