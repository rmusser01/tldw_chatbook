---
id: TASK-32221
title: 'Library Export summary pluralisation: ''1 notes'''
status: Done
assignee: []
created_date: '2026-09-10 14:54'
updated_date: '2026-09-10 19:02'
labels:
  - library
  - export
  - copy
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The export scope summary reads '1 notes'. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 18.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Counts pluralise correctly across media/conversations/notes/prompts
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. _count_phrase helper in library_export_scope; media renders as 'N media items'
2. Apply to the everything summary and the four per-kind summaries
3. Parametrised pins in Tests/Library/test_library_export_scope.py
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One `_count_phrase` helper in `library_export_scope.py` pluralises every
noun. "media" is already plural, so the widest scope renders it as "1
media item" / "N media items" while conversations/notes/prompts take the
bare noun; the per-kind summaries ("Notes · N items") and the explicit
selection line ("Selected notes · 1 item") take the same helper, which is
where the live "1 notes" was reachable. Only the Prompts branch had
counted correctly before.

Pins updated for the new strings: `Tests/Library/test_library_export_scope.py`
(2 existing + 5 new parametrised cases),
`Tests/Library/test_library_export_state.py`, `Tests/UI/test_library_shell.py`
(3 export scope-line assertions). Live-verified: "Selected notes · 1 item"
with exactly one note selected, and "Everything: 12 media items · 6
conversations · 7 notes · 5 prompts".
<!-- SECTION:NOTES:END -->
