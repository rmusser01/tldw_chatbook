---
id: TASK-32298
title: >-
  Library Notes: reader suite hits a Textual text-area--gutter component-class
  race at file scope
status: To Do
assignee: []
created_date: '2026-09-10 13:35'
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
- [ ] #2 The `text-area--gutter` `COMPONENT_CLASSES` race is understood well enough to say whether it can reach a user (a real editor mount) or is confined to the test harness's mount timing, and that answer is written down
- [ ] #3 Whatever guard fixes it is applied wherever this repo mounts a `TextArea` under the same conditions, not only in this one test — `test_mcp_workbench` hit the same key
<!-- AC:END -->
