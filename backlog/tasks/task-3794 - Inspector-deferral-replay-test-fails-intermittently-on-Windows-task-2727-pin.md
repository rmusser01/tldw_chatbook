---
id: TASK-3794
title: Inspector deferral-replay test fails intermittently on Windows (task-2727 pin)
status: To Do
created_date: 2026-08-08 07:40
labels:
- tests
- ui
- personas
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/UI/test_personas_inspector_pane.py::test_state_pushed_before_children_mount_defers_then_replays` (the deferral pin from task-2727) fails deterministically on the owner's Windows box while the machine is under load, yet passed once on pristine code. Failure signature: the readiness label stays at the compose-time guidance text 'Pick a character or persona to start chatting.' instead of containing 'task-2727-probe', meaning the `call_after_refresh(self._apply_action_state)` replay (`personas_inspector_pane.py` on_mount, ~line 433) ran before all children mounted and raised mid-way, leaving the compose-time guidance text.

Stash matrix recorded during task-3793: both-pristine PASSED once; pane-pristine + test-modified FAILED; pane-modified + test-pristine FAILED; full-changes FAILED 5/5. Follow-up rerun on a quiet machine (nothing else running, full 4-suite pass taking 109s): FAILED again — 6/6 with the task-3793 changes applied. Each file independently triggers it, which points to a timing-sensitive interaction between the `call_after_refresh` replay and child mounting rather than a logic regression from the task-3793 avatar changes (CSS padding removal + avatar mount sizing cannot touch the deferral path; both my-pane+pristine-test and pristine-pane+my-test fail, so no single change is responsible). Not proven either way — the pristine pass is a single sample and may itself have been lucky.

Filed per owner instruction during task-3793 close-out; does not block the avatar-rendering PR.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Failure is reproducible in isolation on a quiet machine, or proven load-dependent
- [ ] #2 Root cause identified (deferral replay vs child mount ordering)
- [ ] #3 Fix landed and the test passes reliably across repeated runs
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->

<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
