---
id: TASK-82
title: >-
  Library screen UX fixes (slim header, de-dup headings, remove duplicate nav,
  TCSS styles, actionable empty states)
status: In Progress
assignee:
  - '@claude'
created_date: '2026-06-10 03:15'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implements the Library portion of Docs/superpowers/specs/2026-06-09-notes-workbench-design.md. ADR: none required (implements existing design-system contract).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Status row no longer duplicates purpose line
- [ ] #2 Inspector heading rendered once
- [ ] #3 Duplicate source-browser buttons removed with mode chips as the single path
- [ ] #4 No inline styles.width/height assignments remain in library_screen.py compose
- [ ] #5 Empty states expose a primary action
- [ ] #6 Library/destination UI test suites pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move pane/chip sizing to _agentic_terminal.tcss and rebuild modular css
2. Slim status row, de-dup inspector heading, drop next-action row
3. Remove library-open-search/collections buttons and handlers
4. Actionable empty states
5. Update pinned tests and run suites
<!-- SECTION:PLAN:END -->
