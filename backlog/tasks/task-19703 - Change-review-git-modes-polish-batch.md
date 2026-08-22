---
id: TASK-19703
title: 'Change review git modes: polish batch from whole-branch review'
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - console
  - change-review
dependencies:
  - TASK-16801
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Three non-blocking items the TASK-16801 whole-branch review raised, batched so
they do not each cost a separate branch. None changes behaviour a user relies
on today; all three are places where a failure would be reported less honestly
than the rest of the feature manages.

1. The detection worker's final `call_from_thread` is unguarded, while every
   other landing in the screen routes through the shared helper that catches
   only Textual's teardown `RuntimeError`. During app teardown this one can
   raise a logged worker failure instead of exiting quietly.
2. Both git modals' `_submit` handlers wrap their body in a bare `except`, so a
   genuine bug inside submit makes the confirm button silently do nothing —
   the user presses it and no feedback appears at all.
3. The merge/rebase/cherry-pick in-progress refusal is enforced by the engine's
   guard step, so the user only learns about it after filling in and submitting
   the commit modal. The design spec places that check before the modal opens,
   and the User Guide's wording implies the earlier placement.

Item 3 was a deliberate implementation choice (it avoids a private import and a
time-of-check/time-of-use window), so resolving it may mean amending the spec
and the Guide rather than moving the check — either is acceptable, but the
three artefacts should agree.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The detection worker's landing uses the same teardown-only guard as every other landing, so a real bug there surfaces as a traceback rather than a debug line
- [ ] #2 A bug raised inside either git modal's submit reports itself to the user instead of leaving the confirm button silently inert; a test proves it
- [ ] #3 The in-progress refusal's placement is consistent across the code, the design spec and the User Guide — whichever placement is chosen
<!-- AC:END -->
