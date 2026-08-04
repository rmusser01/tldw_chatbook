---
id: TASK-2239
title: >-
  MCP: give off/opt-in its own display state (no alarm glyph, no 'Needs setup')
  (R2)
status: To Do
assignee: []
created_date: '2026-08-04 16:18'
labels:
  - ux-review
  - mcp
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fresh-install status self-contradicts: banner 'off' with ready glyph, table row 'Needs setup', callout 'turned off — Enable'. Post-fix re-review P1. See Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Off/opt-in has its own muted display state in STATE_LABELS/badge path,Table row no longer reads 'Needs setup' for the off built-in,Off summary is not prefixed with the ready/alarm glyph,Tests updated
<!-- AC:END -->
