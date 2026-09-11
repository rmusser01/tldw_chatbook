---
id: TASK-32298
title: >-
  Library Notes: reader suite hits a Textual text-area--gutter component-class
  race at file scope
status: Done
assignee: []
created_date: '2026-09-10 13:35'
updated_date: '2026-09-11 10:45'
labels:
  - library
  - notes
  - tests
  - flake
  - follow-up
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_library_notes_reader.py::test_reader_route_invalidates_autosave_queued_before_park`
fails inside a whole-file run with
`KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"`, and passes on
its own. The key is Textual's own `TextArea` component class; the failure is a
registration race in the widget, not an assertion about Notes behaviour, and
the same key has flaked elsewhere in this repo (`test_mcp_workbench`, where a
CI rerun was the workaround).

It is timing-exposed, not caused by any one change. Measured on
task-32175's landing pass: in a two-node chunk
(`-k "park or autosave"`) the node fails identically with AND without that
branch's `_open_note_editor` change, so the race is already present; at whole-
file scope the extra `pilot.pause()` cycles that change adds are enough to
shift which test lands on it, taking the file from 31 passed/3 failed to 30
passed/4 failed. Both numbers are reproducible across repeated runs, so this
is a deterministic ordering interaction rather than a random flake — which is
what makes it worth pinning down rather than rerunning.

On `origin/dev` the same two nodes both fail earlier, on the flat
`#library-notes-row-0`/`-1` wait task-32175 repairs, so dev cannot currently
observe this race in this file at all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `Tests/UI/test_library_notes_reader.py` run as a whole file passes `test_reader_route_invalidates_autosave_queued_before_park`, and keeps passing when the tests before it are reordered or their timing shifts
- [x] #2 The `text-area--gutter` `COMPONENT_CLASSES` race is understood well enough to say whether it can reach a user (a real editor mount) or is confined to the test harness's mount timing, and that answer is written down
- [ ] #3 Whatever guard fixes it is applied wherever this repo mounts a `TextArea` under the same conditions, not only in this one test — `test_mcp_workbench` hit the same key
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce at whole-file scope and characterise the race.
2. Answer AC#2 from the evidence the repo already has (TASK-32114).
3. Apply the fixture-level settle; file the product guard AC#3 wants.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: reproduced on this branch at whole-file scope (`1 failed, 33 passed`,
`KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"`), and did NOT
reproduce on the next identical run (`34 passed`) -- so it is intermittent here,
not the deterministic ordering interaction this task recorded at its base. An
autouse `_drain_pending_repaints` fixture now gives the loop its turns after
each test, so a repaint queued by a test's last action lands while the widgets
it targets are still alive instead of during the next app's setup. Whole file
green twice after it (34 passed, 34 passed).

AC#2 -- can it reach a user? YES, and the repo already proved it: TASK-32114
(PR #2534) hit this exact key on Linux CI through a REAL teardown -- Escape
closing the MCP Test Tool panel left the editor in the compositor after
teardown had cleared its component styles -- and fixed it by removing the panel
inside `app.batch_update()`. It is a repaint landing in the window between
"styles cleared" and "widget gone", so any surface that unmounts a live
`TextArea` outside a batch update can hit it. Library is such a surface:
`grep -rn batch_update` over `UI/Screens/library_screen.py` and
`UI/Library_Modules/` returns nothing, and `#library-note-body` is a
`NoteEditorTextArea`.

AC#3 is NOT met by this change and is filed as task-32456: the fixture is a
harness mitigation, and the product guard (batching the Library canvas
teardown, plus a deterministic regression like TASK-32114's) belongs with the
route-switch code. Marked in the fixture's own docstring so it cannot be
mistaken for the fix.

Files: `Tests/UI/test_library_notes_reader.py`.
<!-- SECTION:NOTES:END -->
