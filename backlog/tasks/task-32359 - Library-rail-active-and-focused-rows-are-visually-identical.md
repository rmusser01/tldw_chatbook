---
id: TASK-32359
title: 'Library rail: active and focused rows are visually identical'
status: Done
assignee: []
created_date: '2026-09-11 06:18'
updated_date: '2026-09-11 07:45'
labels:
  - library
  - accessibility
  - critique-10
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Both the active destination and the focused row render bold + underline with backgrounds three RGB units apart (B D9, caps 16/09 .ansi). Focus elsewhere is shown by shape; the rail is the exception. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The focused rail row is distinguishable from the active row by a glyph or shape, not colour alone
- [x] #2 Pinned with a painted assertion
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Painted-frame failing test: focused rail row vs active rail row.
2. CSS-only: focus gets the house thick left bar; active keeps the background.
3. Rebuild the bundle.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
CSS-only, as scoped: `.library-rail-row` had no `:focus` rule at all and fell back to the Button default, so the focused row and the active destination both rendered bold + underline over backgrounds three RGB units apart. `.library-rail-row:focus` now takes the house focus shape -- `border-left: thick $ds-action-focus` with `padding: 0 1 0 0` so the label neither shifts nor clips, plus `outline: none` to suppress the generic `*:focus{outline:solid}` fallback. `-selected` keeps the background treatment, unchanged. `library_rail.py` untouched.

Pinned with the painted-frame idiom from `Tests/UI/test_library_row_focus_cue_t31983.py` at 235x52 and 100x30: focus the Media row while Conversations is active, assert the block glyph paints on the focused row and not on the active one, then swap and assert the reverse. Live-confirmed at 235x52 on the seeded profile (F6 + Tab into the rail: the focused Media row paints the bar, the active Conversations row keeps only its marker).

Files: tldw_chatbook/css/components/_agentic_terminal.tcss (+ regenerated bundle), Tests/UI/test_library_crit10_layout.py, Docs/User_Guide/library.md.
<!-- SECTION:NOTES:END -->
