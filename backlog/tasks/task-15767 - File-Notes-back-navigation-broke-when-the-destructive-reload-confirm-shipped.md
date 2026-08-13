---
id: TASK-15767
title: File Notes back-navigation broke when the destructive-reload confirm shipped
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - notes
  - filesystem
  - regression
priority: high
---

## Description

`Tests/UI/test_screen_navigation.py::
test_action_library_notes_files_back_returns_to_database` is red on dev.
Confirmed live: `AttributeError: 'WorkspaceProbe' object has no attribute
'cancel_reload_confirmation'`.

Controller bisect (per the reviewing session's finding): first-bad commit is
`062c3ee30` ("fix: confirm destructive File Notes reload"), task-15503's PR
which added the retained inline confirmation before a conflict/error reload
can replace the File Notes draft (see task-15503's Implementation Notes —
the new opener, "Discard draft and reload", and its Cancel/Confirm/Escape
flow). That PR's own regression matrix passed 17 tests at the time, so this
is a drift introduced afterward or missed by that matrix's scope, not a
defect in the confirmation feature itself.

Two things need reconciling:
1. `test_action_library_notes_files_back_returns_to_database`'s own test
   double (`WorkspaceProbe`, defined multiple times in
   `test_screen_navigation.py`) needs a `cancel_reload_confirmation` method to
   match whatever the production back-navigation path now calls on it.
2. The production back-navigation path itself needs auditing against
   task-15503's confirmation state: if a user presses "back" while the new
   confirmation dialog is open (or in a conflict/error state that the
   confirmation now gates), the current behavior needs to be intentional, not
   an unhandled attribute lookup.

## Acceptance Criteria

- [ ] `test_action_library_notes_files_back_returns_to_database` passes on
      dev without weakening what it originally asserted (back navigation from
      File Notes returns to the database view)
- [ ] Every `WorkspaceProbe` double in `test_screen_navigation.py` that the
      back-navigation path can reach implements whatever contract production
      now expects, verified against the real
      `library_file_notes_workspace.py` shape from task-15503 (not a stub
      that merely silences the error)
- [ ] Pressing "back" while File Notes is mid-confirmation (or in the
      conflict/error state task-15503 added) has explicit, tested behavior —
      not a crash
- [ ] `Tests/UI/test_library_file_notes_workspace.py` (task-15503's own
      regression matrix) stays green
