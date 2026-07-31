---
id: task-1620
title: 'Settings: persistence badge names the save model on every category'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r3-p1
dependencies: []
priority: high
---

## Description (the why)

Critique round 3 P1: five save models coexist (draft/s+r, autosave, immediate, editor-owned, per-item); the footer keys honestly come and go with the model but nothing NAMED it, so users trained on Save/Revert got silent contract changes and Workspaces acted irreversibly with no warning.

## Acceptance Criteria (the what)

- [x] Every category's State bar leads with a persistence badge in the same position
- [x] The four own-persistence categories and Workspaces render the State banner at all (previously skipped)
- [x] Workspaces' banner names the reversal path for each immediate action
- [x] The dirty 'Unsaved changes' banner keeps priority over the badge

## Implementation Notes

Badge helper `_persistence_badge` + banner refactor (`_category_state_scope_text`); banners added to Theme/Splash/Internal Prompts/Image Gen/Workspaces branches. Every-category coverage pinned by test. Live-verified. Implementation preceded the new tests for this batch (critique findings served as RED evidence); disclosed per TDD policy.
