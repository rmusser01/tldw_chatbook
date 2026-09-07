---
id: TASK-1645
title: 'Settings: badge and scope-copy tightening'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r4
dependencies: []
priority: low
---

## Description (the why)

Critique round 4: the Draft badge was phrased two ways ('Draft — save with s' vs 'Draft — Save/Revert below'); Workspaces and Advanced Config scope texts wrapped the State bar to two lines; three 'Applies: Applies to…' guidance values stuttered.

## Acceptance Criteria (the what)

- [x] Draft badge casing unified
- [x] Workspaces and Advanced Config scope texts fit one line
- [x] Stuttering guidance values fixed

## Implementation Notes

Copy-only changes in _persistence_badge / _category_state_scope_text / the field-guidance tables.
