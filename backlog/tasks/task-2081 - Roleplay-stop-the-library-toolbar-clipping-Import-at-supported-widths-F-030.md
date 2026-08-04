---
id: TASK-2081
title: 'Roleplay: stop the library toolbar clipping Import at supported widths (F-030)'
status: Done
assignee: []
created_date: '2026-08-03 17:24'
updated_date: '2026-08-04 07:16'
labels:
  - ux-review
  - roleplay
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 100x30 only New and Sort survive; Import, Duplicate, Tag are clipped while empty-state copy still says 'use New or Import'. Compact threshold is 90 so it never engages at 100. Evidence: roleplay-100x30.png, personas_screen.py:368. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All toolbar actions are reachable at 100x30 (wrap or overflow menu with New pinned),Rendered-layout regression test at 100x30 and 80x24
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing rendered-layout tests (styled app at 100x30 + 80x24) asserting all library toolbar buttons fit inside the pane. 2. Slim toolbar/filterbar buttons (min-width:0) in PersonasLibraryPane DEFAULT_CSS. 3. Pane-local responsive stacking: toggle a class from on_resize when the pane is too narrow for one row; CSS switches the two bars to vertical layout so every action wraps into view. Global 90-col compact threshold left unchanged (pane-local wrap makes it orthogonal; documented in notes). 4. Run persona UI suites + ruff. ADR required: no - UI layout/copy fix, no schema/boundary/contract change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrap fix in PersonasLibraryPane: toolbar/filterbar buttons now size to their labels (min-width:0 instead of Textual's 16), and the pane toggles a personas-library-stacked-controls class from on_resize when its content is narrower than the single-row width (label-derived measurement, so no oscillation); CSS then switches both bars to vertical layout so New/Import/Duplicate/Sort/Tag each wrap onto their own full-width row. Reached via set_mode/set_sort_label/set_tag_label resyncs too. The global PERSONAS_COMPACT_WORKBENCH_MAX_WIDTH=90 was reconsidered and deliberately left unchanged: pane-local wrapping fixes reachability at every width, while engaging the compact split at 91-100 cols would only shrink the center card below its designed 40-col minimum. Files: tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py, Tests/UI/test_personas_library_toolbar_layout.py (new rendered-layout regression tests at 100x30 + 80x24 + 170x50 counter-case). Verified: 3 new tests pass; Tests/UI/test_personas_library_pane.py + paging (27) pass; test_personas_workbench.py + test_personas_library_scale.py (302) pass; ruff clean. ADR: not required (UI layout fix, no schema/boundary change).
<!-- SECTION:NOTES:END -->
