---
id: TASK-32298
title: >-
  Library Notes: reader suite hits a Textual text-area--gutter component-class
  race at file scope
status: In Progress
assignee: []
created_date: '2026-09-10 13:35'
updated_date: '2026-09-11 19:10'
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
- [ ] #1 `Tests/UI/test_library_notes_reader.py` run as a whole file passes `test_reader_route_invalidates_autosave_queued_before_park`, and keeps passing when the tests before it are reordered or their timing shifts
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
AC#1 is NOT met. It was ticked on a frequency argument and is now unticked
with the measurement that contradicts it.

Evidence, whole-file scope, `Tests/UI/test_library_notes_reader.py`:

| branch state | runs | result |
|---|---|---|
| f0fa91cdd6, `_drain_pending_repaints` present | 4 (review) + 1 (fix round) | `1 failed, 33 passed` every time |
| `_drain_pending_repaints` deleted | 2 | `1 failed, 33 passed` both times |

Always the same node
(`test_reader_route_invalidates_autosave_queued_before_park`), always
`KeyError: "No 'text-area--gutter' key in COMPONENT_CLASSES"`, always raised
INSIDE the test body at the Notes -> Media route switch
(`test_library_notes_reader.py:1212` -> `_wait_for_selector` ->
`pilot.pause(0.02)` -> `Screen._on_timer_update` -> `_render_chops` ->
`TextArea.render_lines`). The fixture drained after `yield`, i.e. after the
body, so it could never reach that window: it is inert, measured in both
directions above, and has been deleted rather than left reading as a fix.

Mechanism, traced into Textual 8: `App._prune` clears the widget's
`_component_styles` when the node's message-loop task exits
(`widget.py:4534`) but only SCHEDULES the `parent.refresh(layout=True)` that
drops it from the compositor map (`AwaitRemove`'s `post_mount`, run via
`call_next`). A repaint landing in that window composites a `TextArea` whose
styles are gone. That is why only `app.batch_update()` closes it --
`_on_timer_update` returns immediately while `app._batch_count` is set.

Two test-side isolations were tried at the switch and both failed, so they are
not shipped:

1. Drain the prune with `asyncio.sleep(0)` turns, then `screen.refresh(layout=True)`
   before the wait: still `1 failed, 33 passed` on the first of two runs, same
   node, same KeyError.
2. Hold the whole teardown inside `pilot.app.batch_update()`: the KeyError did
   not recur, but the shifted timing broke a sibling node instead
   (`test_reader_route_parks_dirty_note_selection_and_preview_without_saving`,
   `assert screen._notes_state.autosave_timer is not None` after returning to
   Notes) -- `1 failed, 33 passed` again. Emulating the missing product guard
   from the test would also have hidden the defect rather than fixed it.

AC#2 stands (answered below). AC#3 is NOT met and is not met by anything on
this branch: the product guard belongs to task-32456
(`app.batch_update()` around the Library canvas teardown, plus the
deterministic regression TASK-32114 has for the MCP panel). This task stays
In Progress behind it, and the one remaining red in the file is filed there.

AC#2 -- can it reach a user? YES, and the repo already proved it: TASK-32114
(PR #2534) hit this exact key on Linux CI through a REAL teardown -- Escape
closing the MCP Test Tool panel left the editor in the compositor after
teardown had cleared its component styles -- and fixed it by removing the
panel inside `app.batch_update()`. It is a repaint landing in the window
between "styles cleared" and "widget gone", so any surface that unmounts a
live `TextArea` outside a batch update can hit it. Library is such a surface:
`grep -rn batch_update` over `UI/Screens/library_screen.py` and
`UI/Library_Modules/` returns nothing, and `#library-note-body` is a
`NoteEditorTextArea`.

Files: `Tests/UI/test_library_notes_reader.py` (the inert fixture removed).
<!-- SECTION:NOTES:END -->
