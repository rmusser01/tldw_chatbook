---
id: TASK-19047
title: >-
  stts_profile_library view-switch dismissal test still flakes under CPU load
  (outside 16842's family fix)
status: To Do
assignee: []
created_date: '2026-08-20 08:40'
labels:
  - test-health
  - flake
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16842 fixed a five-test flake family in
`Tests/UI/test_stts_profile_library.py` (settle on the asserted state, bound
by wall clock — see its 2026-08-16 entry in
`backlog/docs/lessons-testing-evidence.md`). Its reviewer then reproduced a
sixth, pre-existing failure OUTSIDE 16842's diff:
`test_switching_stts_view_dismisses_owned_profile_modal_and_worker` fails
under CPU-burner load at BOTH the wave base `cef56efaf` and head.

Verified still present at dev `1bf7f234e` (:2840): the test uses the file's
`_wait_until` idiom for the modal/unmount/settings-pane/worker-finish waits,
but still contains unsettled one-shot samples in the same shapes 16842's
lessons entry catalogues (e.g. the `voice_profile_action` worker census and
its `not is_finished` assert taken immediately after the modal wait, and a
one-shot `pilot.click` before it). No backlog task covers this test (grepped
backlog/tasks at dev). Likely another settle-on-asserted-state candidate; the
diagnosis must come from a load reproduction, not from reading the test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The failure is reproduced under load first (the 16842 lessons entry's CPU-burner recipe) and the exact failing assertion identified — no speculative patch
- [ ] #2 The fix follows the established 16842 idiom: wall-clock-bounded waits polling the asserted condition itself; no new fixed pauses or attempt-count waits
- [ ] #3 Post-fix evidence meets 16842's bar: repeated full-file runs green including runs under the same load, plus standalone runs of this test
<!-- AC:END -->
