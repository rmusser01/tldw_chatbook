---
id: TASK-38968
title: "Console send-admission off-pump tests, which pin that only acceptance removes a sent draft's prefix, run on no PR lane"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:21
updated_date: 2026-11-07 07:28
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
- [x] #1 The file, or named selectors that include the :479 pin, runs on a PR lane, with the added lane time measured and recorded (whole file on dev: 115 s serial, 39-44 s at -n 4)
- [x] #2 The pinned test fails under a temporary source edit that removes the captured prefix from the composer before durable acceptance
- [x] #3 The selected tests pass 10 of 10 runs in the lane's own invocation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Lane the :479 pin as one UI census tail node; the whole file is 115 s serial.
2. Negative control: a temporary edit that commits the captured draft at the press, which removes the prefix before durable acceptance.
3. Run the pin 10x in the lane's own invocation (serial, `--timeout=180`) and record the times and load.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Laned the :479 pin as one node; no test or product change. Base 8b762b52fa, 2026-11-07.

**Lane (AC#1).** `Tests/UI/test_console_send_admission_off_pump.py::test_an_enter_deferred_behind_a_send_sends_its_own_capture_only` is appended at the UI census tail as one node entry. The whole file is 115 s serial on dev (filing), over the per-entry budget, so only the pin is laned. Lane-step time, in the lane's own invocation (serial, `-p no:xdist --timeout=180 --durations=0`), 10 runs at load 61-89: pytest session 24.4-39.1 s (call 15.4-25.6 s), process wall 35-55 s. The machine was far busier than a CI runner.

**Negative control (AC#2).** A temporary edit in `send_acknowledgement.request_visible_send` committed the captured draft straight after `capture_draft_for_send()`, at the press and so before durable acceptance. The pin failed (`timed out waiting for …<lambda>`, the wait for the composer to still read `"yz"`) in 25 s at load 50.

**10 of 10 (AC#3).** The 10 lane-invocation runs above all passed.

**Files.** `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
<!-- SECTION:NOTES:END -->
