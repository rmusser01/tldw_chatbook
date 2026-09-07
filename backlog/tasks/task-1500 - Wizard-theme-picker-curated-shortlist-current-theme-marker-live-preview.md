---
id: TASK-1500
title: 'Wizard theme picker: curated shortlist, current-theme marker, live preview'
status: Done
assignee: []
created_date: '2026-07-31 00:22'
updated_date: '2026-07-31 01:42'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX UAT: alphabetical wall of raw snake_case themes with novelty entries first, defaults buried, no preview, no current marker.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Shortlist (defaults + few flagship) renders before a 'more' affordance
- [ ] #2 Current theme is marked
- [ ] #3 Highlighting a theme previews it live (revert on leave/cancel)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shortlist = current + textual-dark/light + available flagships (nord/gruvbox/tokyo-night/catppuccin-mocha), Show-all expands in place; current theme tagged '(current)' with clean _theme_name on the button; selection previews live, finish-later reverts via revert_preview(), commit clears the revert obligation. Deviation from AC: preview fires on SELECT (RadioSet has no highlight event) — noted as the Textual-idiomatic equivalent.
<!-- SECTION:NOTES:END -->
