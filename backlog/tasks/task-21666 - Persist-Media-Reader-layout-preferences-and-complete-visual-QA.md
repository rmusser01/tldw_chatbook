---
id: TASK-21666
title: Persist Media Reader layout preferences and complete visual QA
status: Done
assignee: []
created_date: '2026-08-24 16:35'
updated_date: '2026-08-24 17:07'
labels:
  - library
  - media
  - settings
  - tui
dependencies: []
references:
  - Docs/superpowers/plans/2026-08-23-library-media-netnewswire-reader.md
  - backlog/decisions/084-library-media-reader-ia.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Persist manual Library Media pane and optional width preferences through canonical Appearance settings, keep responsive state session-only, and complete integrated documentation and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Canonical Appearance settings load, validate, reset, save, and revert both preferred pane states plus fixed/custom width mode and bounded custom widths.
- [x] #2 Saving layout preferences deep-merges existing Library configuration and refreshes mounted/new Library Media shells without media reads or a whole-screen recompose.
- [x] #3 Manual grip changes persist only preferred pane state; responsive collapse remains session-only and never writes configuration.
- [x] #4 Malformed configuration normalizes safely and cannot overflow or crash the Media shell.
- [x] #5 Source CSS, generated bundles, user documentation, and focused regression suites are current and green.
- [x] #6 Production-shaped visual verification covers wide, medium, compact, and narrow Media Reader states with no horizontal overflow or hidden essential actions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing Appearance model tests for load, strict validation, bounds, reset values, and deep-merge persistence.
2. Extend the existing Appearance defaults model and canonical Settings controls without creating another settings surface.
3. Load normalized preferences in Library, persist manual grip changes off-thread, and add one app-owned refresh generation for mounted screens.
4. Add end-to-end settings/manual-versus-responsive persistence tests and required mutation inverse.
5. Finish source CSS and documentation, regenerate/check bundles, and run the complete focused regression matrix.
6. Run production-shaped multi-size visual verification, focused review, task hygiene, and final commit.

ADR required: no new ADR
ADR path: backlog/decisions/084-library-media-reader-ia.md
Reason: ADR-084 already defines persisted manual preferences, responsive-session ownership, fixed defaults, and optional custom widths; this task implements that accepted boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented canonical Appearance controls for remembered Library/Items pane state, fixed-default versus custom widths, bounded validation, reset/revert, and deep-merged Library saves. Library shells normalize these preferences at construction, refresh mounted geometry in place after Settings saves, and persist only manual grip changes off-thread. Per-pane serialization prevents rapid toggles from writing out of order; failed writes restore the prior preference and warn. Responsive collapse remains derived session state and the mutation inverse proved the no-write regression test fails if resize persistence is introduced.

Production visual QA used the exact `TldwCli.CSS_PATH` stack and a scratch `TLDW_CONFIG_PATH` at 160×50, 120×35, 100×30, and 80×24. It exposed and fixed two compact defects: Textual's default Button minimum clipped More/Info at a 50-column Reader, and Items retained loading copy after detail settlement. Final captures and geometry are in `Docs/superpowers/qa/library-media-reader-2026-08/`; all Reader actions, both five-column grips, settled row copy, and Reader containment are verified.

Documentation now covers the three roles, manual versus responsive collapse, search scopes, Reader modes/actions, local-only Items, server compatibility detail, authoritative complete text, eligible image preview/fallback, delete/Undo/Trash, keyboard behavior, and Appearance settings. Source CSS and generated bundles are synchronized.

Verification: 714 focused reader/settings/config tests passed; after the final concurrency review change, 50 affected shell/settings/config/CSS-integrity tests passed. Ruff, compileall, CSS bundle reproducibility, and `git diff --check` passed. Expected environment warnings were limited to the existing requests dependency-version warning and Python 3.12 `audioop` deprecation. Required review covered stale responses, responsive persistence leakage, hidden controls/focus, resize reads, server claims, complete-text authority, CSS drift, and Watchlists coupling. ADR-084 remains the applicable decision; no new ADR was required. Added the incident-backed config/package import-cycle recurrence to `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
