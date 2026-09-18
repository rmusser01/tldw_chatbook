---
id: TASK-32765
title: Show the complete compact Notes next-step status
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 01:36'
updated_date: '2026-09-18 01:48'
labels:
  - design-system
  - ui
  - library
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The PR visual review shows the compact Notes introductory status losing the final word of its next action at 80 columns. Keep the complete guidance readable without hiding nearby controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The Notes introductory status paints completely at 80x24 in dark and light themes.
- [x] #2 The existing below-64 authority wording and visible Notes actions survive resizing through compact and wide layouts.
- [x] #3 Targeted production-style tests and inspected private native captures qualify the repair and refresh the pre-merge visual review.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: existing backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md
Reason: local text-height repair within the existing Notes layout and token contracts.

1. Reproduce the missing final word using actual production styles and compositor paint.
2. Allow the list authority text to take its natural height while preserving the separate work-pane layout and existing wording.
3. Verify both themes, compact/wide resize, adjacent action paint, generated styles and scoped static checks.
4. Inspect private native captures, record lifecycle evidence, update the visual review and save the repair to draft PR #2704. Merge remains subject to explicit user approval.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The compact list authority now uses natural height and explicitly overrides the inherited two-row maximum with the parent-height limit; the separate work pane retains two rows. This restores the missing final word at 80x24 without changing tokens or wording. Updated the source stylesheet and regenerated Library screen sheet.

Four new production-CSS compositor tests cover empty/populated Notes, both themes, resize through 60/64/80/100/120/170 columns, complete New/Add from files labels and filter focus. All 49 targeted cases, seven preflight checks, new-file Ruff check/format and whitespace checks pass. Independent review found no actionable issue and separately passed populated/dark. Eight final native Console/Notes captures were inspected; the final private process returned and exited zero with ten healthy DBs, no app errors, released lock and unchanged defaults. An earlier runner focus failure/forced shutdown is recorded as unqualified.

ADR required: no; existing ADR-150/161 apply. Guide, completion ledger, conflict report and visual gallery now link the refreshed before/after evidence in Docs/superpowers/qa/2026-09-17-notes-authority-layout/README.md. No full suite; note edits, sync and provider execution were not requalified. The optional authority prefix can persist on a below-64 resize as before; the paint regression accepts either noun presentation while requiring the full action. PR #2704 remains draft pending explicit merge approval.
<!-- SECTION:NOTES:END -->
