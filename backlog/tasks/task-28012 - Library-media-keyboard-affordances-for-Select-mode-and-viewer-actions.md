---
id: TASK-28012
title: Library media - keyboard affordances for Select mode and viewer actions
status: Done
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 06:58'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Originally covered both the old viewer's five-button action row (now obsolete - the Reader has a More overflow menu) and the list's Select mode. Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). The Select-mode half stands: Space on a focused row toggles nothing in either normal or select mode, no key is advertised for entering Select mode or toggling rows, and the counter stays "0 selected". Related but distinct rendering defect (clipped-invisible Select/Trash/bulk buttons) is tracked separately - see the toolbar-clipping task filed from the same run. Scope here: a keyboard path to enter Select mode, toggle rows (Space), and reach the bulk actions, advertised in the footer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Row selection can be entered ('s') and toggled (Space on the focused row) from the keyboard
- [x] #2 The select keys are advertised in the media-list footer (s: select; in select mode space + done selecting)
- [x] #3 Existing mouse paths (Select/Done button, row click) are unchanged
- [x] #4 Viewer action-row accelerator keys split to task-28027 (scoped out here)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Delivered keyboard access to bulk SELECTION on the media list (the concrete live gaps: Space was inert, no key entered select mode, nothing advertised). Discovery: Enter on a focused row ALREADY toggles selection (Textual Button binds Enter) - so only Space and the enter-select-mode key + footer were missing. Added: Binding 's' -> action_library_media_toggle_select_mode (reuses the extracted _toggle_library_media_select_mode seam the Select/Done button now shares); Binding 'space' -> action_library_media_toggle_row_selection (toggles the FOCUSED media row, mirroring handle_library_media_row's select branch). Both gated in check_action (s: media list view, not confirming/in-flight; space: select mode + focused media row). Footer: new LIBRARY_MEDIA_LIST_SHORTCUTS (adds 's: select') and LIBRARY_MEDIA_SELECT_SHORTCUTS (space toggle + s done) constants, wired into the footer seam; updated the two existing side-by-side footer tests that pinned the media list to LIBRARY_LIST_SHORTCUTS. A focused Input consumes printable s/space first, so they still type in the filter/search. Viewer action-row accelerators (old AC#2) split to task-28027. Tests: 3 fake unit tests + 1 integration (s enters, space toggles, footer). Files: UI/Screens/library_screen.py, Tests/UI/test_library_multiselect_media.py, Tests/UI/test_library_media_reader_flow.py, Tests/UI/test_library_media_side_by_side.py.
<!-- SECTION:NOTES:END -->
