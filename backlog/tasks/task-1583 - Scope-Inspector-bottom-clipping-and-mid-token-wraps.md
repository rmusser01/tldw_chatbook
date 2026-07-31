---
id: task-1583
title: 'Scope Inspector: bottom clipping and mid-token wraps'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - rescore-p2
dependencies: []
priority: medium
---

## Description (the why)

Critique rescore P2: in 8 of 20 evidence captures the Scope Inspector's
last visible line is cut mid-sentence ("Saves apply to your local",
"…Nothing is sent to", "Recovery: use each prompt's") — reassurance copy
that reads worse truncated than absent. The 34-char column also breaks
tokens mid-word ("crede/ntial_source", "config.tom/l"). A scrollbar exists
but the default viewport reliably clips the standing local-scope note.

## Acceptance Criteria (the what)

- [x] The inspector's default viewport does not cut the standing
      local-scope note mid-sentence on common category/terminal sizes
      (shorten the copy, reflow, or reserve a fold indicator row)
- [x] Config paths and TOML key names wrap at token boundaries or are
      ellipsized, not split mid-word
- [x] Overflow remains reachable (scrollbar/fold indicator preserved)

## Implementation Plan (the how)

1. RED tests: note lives outside the scrollable body; fold helper breaks
   at separators only; short values pass through.
2. Move the note into the pinned header; add `_fold_long_tokens` applied
   in `_detail_row`.

## Implementation Notes

Chose "reflow" for the note: it moved (text unchanged, task-181 copy
preserved) from the scrollable body's last row into the pinned inspector
header, where it is always fully visible — the truncation "Nothing is sent
to" inverted the reassurance's meaning. Mid-token wraps: module-level
`_fold_long_tokens(text, limit=26)` splits whitespace tokens longer than
the narrow column that contain `.`/`/` separators into continuation-
indented lines broken only after separators; `_detail_row` applies it to
string values. The scroll body keeps its 1-cell scrollbar. TDD RED-first;
existing note-presence tests stay green because visible-text checks span
header and body. Files: `tldw_chatbook/UI/Screens/settings_screen.py`,
`Tests/UI/test_settings_configuration_hub.py`.
