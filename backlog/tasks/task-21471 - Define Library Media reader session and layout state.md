---
id: TASK-21471
title: Define Library Media reader session and layout state
status: Done
assignee:
  - '@codex'
created_date: '2026-08-23'
updated_date: '2026-08-23'
labels:
  - library
  - media
  - state
  - testing
dependencies: []
priority: high
documentation:
  - Docs/superpowers/plans/2026-08-23-library-media-netnewswire-reader.md
  - Docs/superpowers/specs/2026-08-23-library-media-netnewswire-reader-design.md
  - backlog/decisions/084-library-media-reader-ia.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Define the pure dependency-root state contract for the Library Media reader so responsive pane geometry and asynchronous detail loading can be verified independently before the redesigned UI is mounted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Manual pane preferences remain immutable while responsive effective state collapses Library before Items and preserves two reachable five-column grips.
- [x] #2 Fixed widths use declared targets, opt-in custom widths clamp to declared bounds, and returning space restores target widths without resize thrash.
- [x] #3 Explicit pane opens receive temporary priority, collapse the other pane first, and may use only the requested pane minimum while Reader remains permanent.
- [x] #4 Reader session transitions separate selected and loaded backend-qualified identities, preserve mode, and reject stale success and failure completions by generation and canonical id.
- [x] #5 Focused automated tests, import compilation, inverse mutation checks, and whitespace validation pass for the pure state module.
<!-- AC:END -->

## Implementation Plan

1. Add RED table-driven tests for preference normalization, responsive geometry, explicit-open priority, hysteresis, and idempotence.
2. Implement the smallest frozen dataclasses and pure normalization/layout functions needed to satisfy the approved contract.
3. Add RED tests for selected-versus-loaded transitions, pending copy, immediate generations, canonical backend identity, stale response fencing, mode persistence, and external detail sessions.
4. Implement immutable session transitions without importing UI, service, configuration, or record owners.
5. Run focused GREEN verification, restore and record collapse-order and stale-generation inverse checks, self-review the owned diff, and close the task.

ADR required: yes

ADR path: `backlog/decisions/084-library-media-reader-ia.md`

Reason: implements ADR-084's preferred/responsive/effective state contract.

Design and execution references:

- `Docs/superpowers/plans/2026-08-23-library-media-netnewswire-reader.md`
- `Docs/superpowers/specs/2026-08-23-library-media-netnewswire-reader-design.md`
- `backlog/decisions/084-library-media-reader-ia.md`

## Implementation Notes

Implemented the dependency-root Library Media reader state contract with frozen dataclasses and pure functions only. Preference normalization keeps manual choices separate from effective responsive collapse, fixed/custom widths use the approved targets and bounds, both grips remain accounted for, explicit-open priority survives narrow resize resolutions, and hysteresis restores exact targets without interpolation.

Added an immutable selected-versus-loaded session reducer with separate backing IDs, settle delay/immediate generations, truthful pending copy, persistent Reader mode, local/server canonical identity isolation, and success/failure fencing on both generation and requested canonical ID. Direct construction rejects inconsistent settled identities and no database rows are stored.

TDD evidence included the initial missing-module RED, missing-resolver RED, missing-session RED, and review follow-up RED for persistent priority and settled identity consistency. The focused suite passes 28 tests. Reversing Library-first collapse made the 120-column table case fail; removing generation matching made the stale success/failure test fail; both mutations were restored. Ruff check/format, focused compile/import, and `git diff --check` pass. Existing pytest dependency and temporary-directory cleanup warnings remained environmental after exit code 0.

An independent code review identified and prompted fixes for inherited explicit-open priority and cross-slot identity consistency. Its below-60 explicit-minimum observation is outside the approved 60-column verification floor; the implementation preserves the declared pane minimum and non-raising truncation behavior rather than inventing a conflicting sub-floor geometry contract.

No UI, service, configuration, spec, ADR, or plan files changed. No new general lesson was added because the task did not surface a reusable repository trap beyond the existing Backlog and testing guidance. This five-digit task record was maintained directly, as required by `backlog/docs/lessons-backlog-hygiene.md`, to avoid the documented Backlog CLI misaddressing bug.

ADR required: yes

ADR path: `backlog/decisions/084-library-media-reader-ia.md`

Reason: implements ADR-084's preferred/responsive/effective state contract.
