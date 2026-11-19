---
id: TASK-36528
title: File Notes 'Save failed' paints at 3.89:1 in textual-dark, and the Offline state elides to one character at 40 columns
status: Done
assignee:
- '@claude'
created_date: 2026-10-24 00:26
updated_date: 2026-11-03 21:22
labels:
- follow-up
- library
- notes
- accessibility
- tests
priority: medium
references:
- tldw_chatbook/Widgets/Library/library_file_notes_workspace.py
- Tests/UI/test_library_file_notes_workspace.py
dependencies:
- TASK-36383
- TASK-36513
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by TASK-36383 (PR #3232, 2026-10-23). `Tests/UI/test_library_file_notes_workspace.py` has 10 tests that fail on dev outside that task. The file runs on no PR lane. Re-run on dev 5b5920b8e2, all 10 failed. Six are covered elsewhere:
- the five `test_notes_authority_*` nodes and `test_wide_files_task_return_restores_database_browse_receipt` by TASK-32185 AC#4;
- `test_file_notes_linked_root_leads_with_friendly_folder_identity[60x20]` by TASK-36513.

This task is two of the other three; the third, the owned-replica failure, is filed on its own:
- **A product contrast defect.** `test_high_stakes_file_notes_states_are_legible_in_shipped_themes[(120, 40)]` fails with "textual-dark: 'Save failed' paints at 3.89:1, below 4.5:1". The error state is styled by `#file-notes-save-status.-error { color: $text; background: $error 14%; text-style: bold; }` (workspace.py:1106-1110). The test was already failing on 2026-09-11 (TASK-32185's notes).
- **The Offline state word at 40x20.** The `[(40, 20)]` case fails on `assert 'Offline' in '.'`: at 40 columns, the root status is elided to one character. TASK-36513 traced the same squeeze at 60x20 to the always-displayed 27-cell "Review recovered pairing…" button. Whether that button also explains 40x20 is not confirmed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every high-stakes File Notes save and root state paints at 4.5:1 or better in textual-dark, textual-light and high_contrast_yellow_black
- [x] #2 Both cases of test_high_stakes_file_notes_states_are_legible_in_shipped_themes pass on dev
<!-- AC:END -->

## Implementation Plan

ADR required: no
ADR path: N/A
Reason: One app-tier colour pair and its pin; no architectural change.

1. Measured on eba3ad14b5: the app tier's `.file-notes-git-commit-error, #file-notes-save-status.-error` rule (`_agentic_terminal.tcss`) paints `$ds-status-error-readable` (= `$text-error`) on `$error-darken-3`, overriding the widget's own `$text` on `$error 14%`. "Save failed" reads 3.89:1 in textual-dark, 1.19:1 in textual-light and 2.63:1 in high_contrast_yellow_black (81 of 91 available themes under 4.5:1). Both colours derive from `$error`, so they converge on light themes.
2. The 40x20 "Offline" squeeze had TASK-36513's cause (the always-shown pairing button); that fix already cleared it.
3. Repaint the shared rule as `$text` on `$error 14%`, the recipe Conflict and Offline already pass with (measured: 88 of 91 themes at 4.5:1 or better, the three named themes 12.98, 10.90 and 14.58). The Git commit error shares the rule, so it gets the same legible pair.
4. Rebuild the bundle with `build_css.py`, update the app-tier pin in `test_css_build_integrity.py`, run both cases of the legibility test (red first), the CSS integrity tests and preflight.

## Implementation Notes

**Contrast cause.** The app tier's `.file-notes-git-commit-error, #file-notes-save-status.-error` rule in `css/components/_agentic_terminal.tcss` (added 2026-08-20 by 2f1a2de651, nine days after 55e2fcb79b wrote this legibility test) painted `$ds-status-error-readable` (= `$text-error`) on `$error-darken-3`. It overrode the widget's own `$text` on `$error 14%`. Both colours derive from `$error`, so they converge: measured on eba3ad14b5, "Save failed" painted 3.89:1 in textual-dark, 1.19:1 in textual-light and 2.63:1 in high_contrast_yellow_black (81 of the 91 themes the harness registers fall under 4.5:1). The test stopped at the first theme, so only 3.89:1 was ever reported.

**Fix.** The shared rule now paints `$text` on `$error 14%`, the pair Conflict and Offline already pass with. The same probe measured 12.98:1 (textual-dark), 10.90:1 (textual-light) and 14.58:1 (high_contrast_yellow_black), with 88 of 91 themes at 4.5:1 or better. The exceptions are ansi-dark and ansi-light (1.92:1, ANSI palettes) and litestep (3.54:1); `$text-error` on `$error-muted` was also measured and was worse (19 themes under 4.5:1). The Git commit error shares the rule, so it gets the same pair; its `$text-error` on `$error-darken-3` had the same convergence. Bundle rebuilt with `build_css.py`; the app-tier pin in `test_css_build_integrity.py::test_file_notes_error_ink_and_disabled_opacity_are_app_tier` now names the new tokens.

**40x20 Offline.** This was the same cause as TASK-36513: the always-displayed 27-cell pairing button left the root status 1 cell at 40 columns. With 1d6a55b5a5 the button is hidden unless a pairing awaits review, and the part-wise fit keeps "Offline · Warning" whole in the 18 cells left.

**Evidence.** On 1d6a55b5a5 both cases of `test_high_stakes_file_notes_states_are_legible_in_shipped_themes` were red: "textual-dark: 'Save failed' paints at 3.89:1, below 4.5:1" (before 1d6a55b5a5 the 40x20 case stopped earlier, at `assert 'Offline' in '.'`). After the fix both pass, with `test_css_build_integrity.py` and `test_component_pattern_governance.py`: 51 passed at load ~32. Preflight reports the bundle in sync.

**Files.** `tldw_chatbook/css/components/_agentic_terminal.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (regenerated), `Tests/UI/test_css_build_integrity.py`.
