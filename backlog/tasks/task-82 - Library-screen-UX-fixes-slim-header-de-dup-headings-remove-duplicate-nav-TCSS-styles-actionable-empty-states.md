---
id: TASK-82
title: >-
  Library screen UX fixes (slim header, de-dup headings, remove duplicate nav,
  TCSS styles, actionable empty states)
status: Done
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
- [x] #1 Status row no longer duplicates purpose line (superseded: PR #503's dynamic per-mode status row already removed the duplication)
- [x] #2 Inspector heading rendered once (superseded: PR #503 renamed nested headings to mode-specific titles, e.g. "Hub inspector")
- [x] #3 Duplicate source-browser buttons removed with mode chips as the single path (superseded: PR #503 made the Library Modules buttons load-bearing hub navigation with active-state sync; they stay)
- [x] #4 No inline styles.width/height assignments remain in library_screen.py compose
- [x] #5 Empty states expose a primary action (superseded: PR #503's hub empty-state copy is deliberate and test-pinned)
- [x] #6 Library/destination UI test suites pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Move pane/chip sizing to _agentic_terminal.tcss and rebuild modular css
2. Slim status row, de-dup inspector heading, drop next-action row
3. Remove library-open-search/collections buttons and handlers
4. Actionable empty states
5. Update pinned tests and run suites
<!-- SECTION:PLAN:END -->

## Implementation Notes

Scope was reconciled with PR #503 (Library content hub / Search-RAG evidence workflow),
which landed on dev after this task was planned and superseded the status-row, heading,
duplicate-button, and empty-state items with its own deliberate, test-pinned design.

Delivered:
- Single-row mode strip: removed all inline `styles.*` from the mode bar/label/chips and panes,
  plus the `LIBRARY_MODE_*`/`LIBRARY_SOURCE_*_WIDTH`/`LIBRARY_FRAME_*` constants; sizing lives in
  `$ds-library-mode-*` tokens (now 1) and TCSS pane rules (browser keeps #503's fixed 31 cols).
- `Button.library-mode-chip` selector out-specifies Textual's `Button.-style-default` tall borders;
  active chip reads via background + bold underline (border would eat the 1-row chip). Restored the
  `bold underline` non-obscuring focus signal that #503 had regressed to plain `bold`.
- Deleted `_frame_library_region` (hardcoded `#6f7782` border) in favor of `$ds-grid-line` TCSS;
  added `LibraryScreen.DEFAULT_CSS` baseline geometry for stylesheet-less harness tests.
- Updated the two stale chip-contract tests (master-shell design system, non-obscuring focus) that
  were already failing on dev; both now pass. Zero new failures vs the dev baseline.
- ADR: none required — implements the existing design-system contract.
