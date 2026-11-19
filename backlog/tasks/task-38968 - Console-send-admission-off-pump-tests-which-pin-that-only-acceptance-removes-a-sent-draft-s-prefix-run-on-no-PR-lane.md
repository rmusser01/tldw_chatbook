---
id: TASK-38968
title: "Console send-admission off-pump tests, which pin that only acceptance removes a sent draft's prefix, run on no PR lane"
status: To Do
created_date: 2026-11-06 01:21
labels:
- follow-up
- tests
- console
- ci
priority: low
references:
- Tests/UI/test_console_send_admission_off_pump.py
- scripts/ui_pr_gate_census.txt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down. Checked on dev 29592ecc9b and on the census as it will be after the round-8 branches merge, 2026-11-05.

`Tests/UI/test_console_send_admission_off_pump.py` is not in `scripts/ui_pr_gate_census.txt`, and no workflow targets it. That includes `test_an_enter_deferred_behind_a_send_sends_its_own_capture_only` (`:479`), whose comment at `:507-508` reads "Only that acceptance owns removing the captured prefix from the composer." The only tasks that name the file, TASK-33620.15 and TASK-33622.9, are Done.

Runs through the scratch-profile wrapper:
- Whole file at `-n 4`: 3 of 3 green, 24 of 24 tests each time. Wall time 39-44 s at load 13-18.
- Serial (`-p no:xdist --timeout=180`): 115 s. The pinned test's call took 4.8 s.

So the whole file costs more than the 60 s per file that the lane docstrings allow. Adding named selectors fits that budget better than adding the whole file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The file, or named selectors that include the :479 pin, runs on a PR lane, with the added lane time measured and recorded (whole file on dev: 115 s serial, 39-44 s at -n 4)
- [ ] #2 The pinned test fails under a temporary source edit that removes the captured prefix from the composer before durable acceptance
- [ ] #3 The selected tests pass 10 of 10 runs in the lane's own invocation
<!-- AC:END -->
