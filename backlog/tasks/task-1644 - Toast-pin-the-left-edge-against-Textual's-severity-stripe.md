---
id: TASK-1644
title: 'Toast: pin the left edge against Textual's severity stripe'
status: Done
assignee:
  - '@claude'
created_date: '2026-07-31'
labels:
  - settings
  - ux
  - critique-r4
dependencies: []
priority: medium
---

## Description (the why)

Critique round 4: an orange LEFT stripe survived over the round-3 full-border restyle — Textual's Toast DEFAULT_CSS sets 'border-left: outer <color>' per severity, which outranked the border shorthand.

## Acceptance Criteria (the what)

- [x] Every severity variant pins its own round border-left
- [x] The base Toast rule pins its left edge too

## Implementation Notes

CSS-only in _base.tcss; live-verified the toast renders a full round border in the accent tint with no outer stripe.
