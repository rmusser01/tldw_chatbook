---
id: TASK-2242
title: 'MCP: kill-switch label states its blast radius (R2)'
status: Done
assignee: []
created_date: '2026-08-04 16:18'
updated_date: '2026-08-04 19:46'
labels:
  - ux-review
  - mcp
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
'block tool calls in chat: no' is lowercase/telegraphic and its scope (also built-in tools) lives only in a tooltip. Post-fix re-review P2. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Label is plain and title-cased with scope stated persistently,Tests updated
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no — label/copy change on an existing toggle.

1. mcp_permissions_mode.py _kill_switch_label(): title-cased plain label 'Block all tool calls in chat: On/Off' (keeps the toggle Button mechanic and trailing affordance glyph).
2. compose(): persistent one-line dim scope hint under the button ('Also disables built-in tools …') so the blast radius no longer lives only in the tooltip/docstring; DEFAULT_CSS rule mirroring #mcp-perm-legend.
3. Tests: update the two label assertions; add a hint-content test.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Label now 'Block all tool calls in chat: On/Off ▸' (title-cased, plain); persistent dim hint states built-in-tools blast radius; toggle mechanic unchanged; tests updated.
<!-- SECTION:NOTES:END -->
