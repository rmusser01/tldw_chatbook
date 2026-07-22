---
id: TASK-430
title: >-
  File picker interaction model - visible cursor, uniform open behavior, layered
  Esc
status: Done
assignee: []
created_date: '2026-07-21 09:38'
updated_date: '2026-07-22 07:05'
labels:
  - widgets
  - ux
  - file-picker
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from the RP/character-card UX review (Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md). Observed live while importing a card (EnhancedFileOpen): the selection cursor is bold+underline with a near-identical background and is effectively invisible; single-click on a directory navigates immediately while files require select+Enter (clicking ".." just to focus the list teleports you up a level); entering a full FILE path in the Ctrl+L bar then Enter/Go only navigates to the parent listing instead of opening the file; Esc pressed inside the Recent overlay dismisses the entire picker. An expert driver needed ~10 interactions to open one file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selected row is clearly distinguishable at a glance in the default theme
- [x] #2 Directories and files share one predictable activation model (single-click selects for both; a consistent action opens/descends)
- [x] #3 Confirming a full file path in the path bar opens/returns that file
- [x] #4 Esc dismisses only the topmost overlay (Recent/Bookmarks), not the whole picker
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
File-picker interaction model overhauled — all changes are OVERRIDES in EnhancedFileDialog/SearchableDirectoryNavigation (enhanced_file_picker.py); NO edits to vendored Third_Party/textual_fspicker/.
AC#1 visible cursor: the generated bundle's global OptionList highlight rule paints $surface (≈dialog bg, invisible) and overrides widget DEFAULT_CSS by Textual origin priority; added an id-scoped #file-list-pane .option-list--option-highlighted rule using $ds-focus-bg/$ds-focus-fg in the SOURCE css/components/_lists.tcss (bundle regenerated). Added a narrow exact-string exemption to test_non_obscuring_focus_contract.py's neutral-choice scan + a compensating positive test.
AC#2 uniform activation: single-click SELECTS (no navigate), Enter/double-click/Go-button OPENS (descend dir / return file). CRUX: Textual dispatches convention handlers across the WHOLE MRO, so the vendored _on_option_list_option_selected (navigate-on-select) needed _SUPPRESSED_BASE_HANDLERS suppression on the LIST widget; added action_open_highlighted + Enter rebind + on_click chain>=2 double-click + OpenFile message + broadened Go-button (_confirm_single) to open a highlighted dir OR file.
AC#3 path bar: overrode the suppressed _on_path_input_submit so a typed existing FILE returns/confirms (via _should_return/_confirm_single) instead of cd-to-parent; dir/nonexistent branches preserved.
AC#4 layered Esc: action_smart_dismiss closes topmost overlay (path bar→search→recent→bookmarks) before dismissing.
Verification: 350-test cross-consumer regression across 10 picker+consumer suites (no consumer regressed); real-driver tests (pilot.press/Button.press/pilot.click(times=2)); whole-branch opus review; LIVE-VERIFIED (visible cursor rgb 81,103,126, single-click-selects/Enter-opens dir+file end-to-end, layered Esc).
<!-- SECTION:NOTES:END -->
