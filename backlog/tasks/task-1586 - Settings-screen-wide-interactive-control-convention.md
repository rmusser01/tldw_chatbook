---
id: task-1586
title: 'Settings: screen-wide interactive-control convention (design)'
status: To Do
assignee: []
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - design
dependencies:
  - task-1582
priority: medium
---

## Description (the why)

Split from task-1582. The critique rescore's underlying P2 ask — one
consistent visual convention distinguishing interactive controls from
prose across all Settings categories (bracketed toggles, bordered or
otherwise visually-distinct inputs, a visible focus ring on center-pane
fields) — is a design project, not a hygiene fix. Discovered constraint:
`.settings-compact-input` deliberately uses `border: none` at `height: 1`
because a Textual border consumes rows; bordering every input triples its
height and reflows every dense form on the screen. Any convention must
either accept the taller forms, use background/color tokens instead of
borders, or introduce a marker-glyph idiom (e.g. `▸` prompts, bracketed
toggles) that costs no rows.

## Acceptance Criteria (the what)

- [ ] A documented convention distinguishes editable controls from prose
      at rest (not only on focus) without breaking dense-form layouts
- [ ] A visible focus indicator exists on center-pane fields under the
      real CSS bundle
- [ ] The convention is applied consistently across Settings categories
      and captured in the design docs
