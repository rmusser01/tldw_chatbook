---
id: TASK-21232
title: >-
  Dev red - test_library_canvas_scoped_sync harness is missing
  _library_prompt_browse_controller
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - bug
  - test-health
  - dev-red
  - library
dependencies: []
priority: high
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; found pre-existing
during TASK-21101's implementation and re-confirmed at close-out.

`Tests/UI/test_library_canvas_scoped_sync.py` builds its screen double from
`types.SimpleNamespace` and never sets `_library_prompt_browse_controller`, which the
production code under test reads. Re-run on dev `b2b1e2e0d`: **4 failed, 5 passed**, with
`AttributeError: 'types.SimpleNamespace' object has no attribute
'_library_prompt_browse_controller'` raised from the freshness check.

The failure is in the harness, not the subject. Those four tests are not measuring the
canvas-scoped sync behaviour they were written for, so the Library canvas seam — the seam
TASK-21116 is converting more sites onto, and the seam TASK-21242 must repair — is
under-covered while looking covered. That is the worst state for a guard to be in while
another task is actively changing its subject.

## Acceptance Criteria

- [ ] `Tests/UI/test_library_canvas_scoped_sync.py` passes in full on dev
- [ ] Each of the four repaired tests exercises the canvas-scoped sync assertion it was written for, shown by failing when its subject behaviour is mutated — not merely by no longer raising AttributeError
- [ ] A missing screen attribute the subject reads fails with a message naming the attribute, or the double is derived from the real screen's attribute surface so it cannot drift again
