---
id: TASK-21237
title: >-
  Drag-selection hardening residue from TASK-21114
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - console
  - performance
  - test-coverage
dependencies: []
priority: low
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; the three Minors
from the TASK-21114 review (PR #2007, `898cd8852`). Filed as one task because all three sit in
the same handler that change rewrote.

TASK-21114 cut transcript drag-selection from **151 body re-wraps per drag to 1** and made the
mouse-move handler **32× to 213× faster**, with equivalence proven twice independently. Three
items were ledgered rather than fixed:

1. Entering keyboard selection **mid-mouse-drag** leaves a ghost highlight. The pre-change
   code cleared it on every mouse-move; removing that per-move sweep is what made the handler
   fast, and removing it is what took the clear away.
2. The arm-time sweep is not pinned by its own test. The scenario the existing test covers is
   menu dismissal, which happens to exercise the same path — so the sweep can be deleted
   without a red.
3. The early-return layers in the handler are pairwise redundant and none is individually
   pinned, so any one of them can be deleted with the suite still green.

## Acceptance Criteria

- [ ] Entering keyboard selection during an active mouse drag leaves no stale highlight on any row
- [ ] A test covers the arm-time sweep in a menu-less scenario and fails when the sweep is removed
- [ ] Each early-return layer is individually pinned by a test that fails when only that layer is deleted, or the redundant layers are removed
- [ ] TASK-21114's per-drag wrap count (1) and handler timings do not regress
