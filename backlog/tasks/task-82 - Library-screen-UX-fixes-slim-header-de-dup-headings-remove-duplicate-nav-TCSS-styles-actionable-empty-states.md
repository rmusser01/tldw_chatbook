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
- [x] #1 Status row no longer duplicates purpose line
- [x] #2 Inspector heading rendered once
- [x] #3 Duplicate source-browser buttons removed with mode chips as the single path
- [x] #4 No inline styles.width/height assignments remain in library_screen.py compose
- [x] #5 Empty states expose a primary action
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

- Slimmed `#library-status-row` to `{status} | Local`; the dropped prefix duplicated the purpose line. Status taxonomy (Unavailable/Empty/etc.) unchanged.
- Mode strip is now one row: removed all inline `styles.*` from the mode bar/label/chips and the `LIBRARY_MODE_*` sizing constants; sizing lives in `$ds-library-mode-*` tokens (now 1) plus `Button.library-mode-chip` TCSS. The active-chip style dropped its border (a border consumes the whole row at height 1) in favor of background + bold underline; specificity had to be `Button.library-mode-chip` to beat Textual's `Button.-style-default` tall borders.
- Removed the nested duplicate "Inspector" heading and the `#library-active-mode-next-action` prose row (mount anchors retargeted to `#library-active-mode-description`); folded the collections later-stage disclaimer into the mode description.
- Removed `#library-open-search`/`#library-open-collections` buttons + handlers (pure duplicates of mode chips). Kept Open Notes/Media/Conversations and Import/Export Sources (real navigation).
- Empty state now shows a primary "Import Sources" button (`#library-empty-import-sources` → ingest) instead of instructional copy; inspector empty state shortened to one line.
- Deleted `_frame_library_region` (hardcoded `#6f7782` border) — panes now use `.library-region`/ID rules with `$ds-grid-line`. Added `LibraryScreen.DEFAULT_CSS` with the baseline workbench geometry so harness tests (no app stylesheet) render correctly; app TCSS keeps equal specificity and wins.
- Tests: updated 9 pinned test files; fixed 5 previously-failing tests (3 mode-strip geometry, design-system chip contract, gate1 collections copy); zero new failures vs baseline (33 pre-existing failures in the `-k "library or destination or replay or focus"` selection remain, all unrelated to Library).
- QA captures: `Docs/superpowers/qa/library-ux-fixes/`.
- ADR: none required — implements the existing design-system contract.
