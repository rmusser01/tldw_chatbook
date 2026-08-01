---
id: TASK-1623
title: 'Scope Inspector: fold indicator + slash-preferred token folding'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r3-p2
dependencies: []
priority: medium
---

## Description (the why)

Critique round 3 P2: 8 of 26 rest-state captures ended the inspector mid-sentence with nothing signalling more content; the token folder also broke '~/.config/tldw_cli/config.toml' as 'config.' / 'toml' — a mid-filename break at the extension dot.

## Acceptance Criteria (the what)

- [x] A reserved bottom row ('▼ more — scroll the inspector') shows exactly while the body overflows
- [x] Path folding prefers slash boundaries; extension dots never split a filename

## Implementation Notes

`_update_inspector_overflow_hint` (mount/resize/category-switch via call_after_refresh, guarded queries, real laid-out sizes); `_fold_long_tokens` two-phase split (slash chunks first, dot-split only for over-long slash-free chunks). Styled-harness test pins hint==overflow on Privacy at 40 rows; live-verified shown on Privacy, hidden on Artifacts.
