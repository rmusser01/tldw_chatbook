---
id: TASK-1586
title: 'Settings: screen-wide interactive-control convention (design)'
status: Done
assignee:
  - '@claude'
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

- [x] A documented convention distinguishes editable controls from prose
      at rest (not only on focus) without breaking dense-form layouts
- [x] A visible focus indicator exists on center-pane fields under the
      real CSS bundle
- [x] The convention is applied consistently across Settings categories
      and captured in the design docs

## Implementation Plan (the how)

1. Survey the token palette and the constraint space (borders cost rows;
   left borders cost a column).
2. RED tests: token repoint, source-CSS edge rules, computed-style check
   under the real bundle.
3. Mint `$ds-control-edge`; repoint `$ds-input-focus-bg`; edge rules on
   `.settings-compact-input`; bundle rebuild; live verify; document in
   DESIGN.md.

## Implementation Notes

The convention is **edge-marked fields**: a one-column `border-left`
(solid `$ds-control-edge` = `$surface-lighten-2`) marks every editable
field at rest — left borders cost a column, never a row, so the one-row
dense forms keep their height. The edge's presence is the carrier
(structural marker), color is reinforcement. On focus it flips to
`thick $ds-action-focus` with the background swapping to `$ds-focus-bg`
and bold text.

Root-cause find along the way: `$ds-input-focus-bg` still aliased
`$ds-surface-raised` (= `$surface`) — the exact nullified-focus failure
task-345's own comment documents and fixed for `$ds-focus-bg` but not
this token. Settings fields' focus background swap was a no-op, which is
precisely the critique's "no visible focus indicator on center-pane
fields". Repointed to `$ds-focus-bg`.

Consistency comes free: every Settings form (screen forms plus the theme
editor, splash viewer, and image-gen panel widgets) uses the
`.settings-compact-input` class, so the two CSS rules cover all
categories. Toggles/switches already carry text-state words and disabled
actions carry "— no changes" annotations from tasks 1582/1585 — the
convention document (DESIGN.md "Dense-form control convention") captures
all of it as one contract.

Live-verified in tmux: rest edge (`Threshold │ 50`, `Provider │ …`),
focus flip (thick $primary edge + #51677e bg + bold, captured with
colors), rest-return on blur. Files: `_variables.tcss`,
`_agentic_terminal.tcss` (+ rebuilt bundle), `DESIGN.md`,
`Tests/UI/test_settings_configuration_hub.py`.
