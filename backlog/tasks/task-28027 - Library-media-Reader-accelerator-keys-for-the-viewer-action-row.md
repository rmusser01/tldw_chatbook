---
id: TASK-28027
title: Library media Reader - accelerator keys for the viewer action row
status: Done
assignee: []
created_date: '2026-09-02 06:57'
updated_date: '2026-09-02 16:04'
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
Split from task-28012 (which delivered select-mode keyboard access on the LIST). The Reader's action row (Find / Read later / Use in Console / More, with Edit metadata / Open original / Open manager / Move to trash under More) has no accelerator keys - every action is a Tab-walk (Alex persona red flag from the 2026-09-01 critique). Give the common Reader actions bound keys, advertised in the footer or F1 help, without stealing the printable keys the search/filter inputs need. Note existing Reader keys already taken: / (focus search), F6 (pane), ] / [ (next/prev item, task-28005), enter (next match when searching, task-28011).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The common Reader actions (at least Find, Read later, Use in Console, Move to trash) have bound keys
- [x] #2 The keys are advertised in the viewer footer or F1 help and gated to the Reader
- [x] #3 A focused search/filter input still receives those printable keys
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reader action-row accelerator keys (l=Read later, c=Use in Console, t=Move to trash->confirm). Bindings show=False, gated in check_action to a plain media Reader (view=='viewer', no edit/confirm/analysis-edit sub-state); l/t are local-only (mirroring the buttons, hidden for external/server detail), c works for server items too. Actions reuse the button seams: extracted _start_library_media_read_later_toggle (shared by button + key), _open_selected_media_handoff for c, and the same confirm-arm for t. A focused search/filter Input consumes the printable key first (Textual routing, same as the existing r/s bindings), so they still type there. Advertised via F1 help (action_show_workbench_help auto-includes gated bindings) rather than the already-crowded reader footer (AC allows footer OR F1). Tests: check_action gating (plain/substate/external/list), 3 action unit tests, F1-lists-l/c/t, and t-key end-to-end arms the delete confirm. Files: UI/Screens/library_screen.py, Tests/UI/test_library_media_reader_flow.py, Tests/UI/test_screen_navigation.py.
<!-- SECTION:NOTES:END -->
