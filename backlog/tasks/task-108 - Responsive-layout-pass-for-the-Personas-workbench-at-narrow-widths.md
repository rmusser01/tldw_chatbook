---
id: TASK-108
title: Responsive layout pass for the Personas workbench at narrow widths
status: Done
assignee: []
created_date: '2026-06-12 20:16'
updated_date: '2026-06-26 17:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At ~900px the inspector readiness text wraps mid-phrase and the work-area middle column collapses; do a responsive review (min-widths, text wrapping, possibly collapsing a pane).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Work area stays usable at 80-col equivalent,Inspector text wraps cleanly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-read the Personas workbench, inspector, and existing UI tests on latest dev.
2. Write a failing regression test for an 80-column Personas workbench so the center work area remains usable and the inspector readiness copy is cleanly wrapped/compact.
3. Implement the smallest responsive layout change in the existing Personas screen and inspector styles, keeping all three panes reachable and avoiding the separate collapsible-rails scope.
4. Run focused Personas UI tests plus diff checks, then update acceptance criteria and implementation notes.

ADR required: no
ADR path: N/A
Reason: This is a contained Textual UI responsive-layout fix inside the existing Personas destination; it does not change storage, schema, sync policy, provider/runtime boundaries, service contracts, security, dependencies, or long-lived application structure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a screen-owned compact workbench class for narrow Personas layouts and compact pane minimums in both the fallback CSS and bundled TCSS. The 80-column regression now runs with the generated app stylesheet loaded, so it verifies the real pane chrome and rule ordering. Inspector readiness copy now uses shorter `Console blocked:` / `Console ready` labels and explicit wrapping to avoid mid-phrase dash wrapping. Regenerated `tldw_cli_modular.tcss`.

ADR required: no
ADR path: N/A
Reason: UI-only responsive behavior within the existing Personas screen; no storage, sync, provider/runtime boundary, service contract, security, dependency, or long-lived application-structure decision changed.
<!-- SECTION:NOTES:END -->
