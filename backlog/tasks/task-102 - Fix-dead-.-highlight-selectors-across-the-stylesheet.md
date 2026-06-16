---
id: TASK-102
title: Fix dead .--highlight selectors across the stylesheet
status: Done
assignee: []
created_date: '2026-06-11 20:28'
updated_date: '2026-06-16 19:52'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Textual sets -highlight (single dash) on highlighted ListItems; several bundle rules use .--highlight and never match. Sweep and fix non-personas occurrences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No dead --highlight selectors remain in non-personas TCSS source files,Bundle regenerated after fixes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a stylesheet selector bugfix that preserves existing UI architecture.

1. Add a QA regression that fails when non-personas TCSS source files or the generated bundle contain dead .--highlight selectors.
2. Run the focused regression and confirm it fails on the current dev state.
3. Replace dead .--highlight selectors with Textual's single-dash .-highlight selector in the affected TCSS source files.
4. Regenerate the modular TCSS bundle.
5. Rerun focused verification and git diff checks, then update the backlog task notes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added a QA regression that rejects dead `.--highlight` selectors in non-personas TCSS source files and in the generated bundle. Updated the affected shared list, embeddings, Chatbooks, and config search selectors to Textual's `.-highlight` form, then regenerated `tldw_chatbook/css/tldw_cli_modular.tcss`.

Verification:
- `python -m pytest -q Tests/QA/test_textual_highlight_selectors.py --tb=short`
- `git diff --check`
- `rg -n "\\.--highlight" tldw_chatbook/css --glob '*.tcss'`
<!-- SECTION:NOTES:END -->
