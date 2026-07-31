---
id: task-1643
title: 'RAG: annotate the inert built-in Delete'
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

Critique round 4 P2: a full-saturation red Delete sat active-looking directly under 'Built-in profile — read-only', drawing the eye first and contradicting the banner.

## Acceptance Criteria (the what)

- [x] Built-in profiles render 'Delete — built-in', disabled
- [x] User profiles keep the plain enabled Delete

## Implementation Notes

Label + disabled flag keyed on info['read_only'], matching the convention that inert destructive actions carry their reason in text (DESIGN.md).
