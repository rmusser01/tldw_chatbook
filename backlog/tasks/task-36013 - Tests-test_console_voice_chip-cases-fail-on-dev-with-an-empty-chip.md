---
id: TASK-36013
title: 'Tests: nine test_console_voice_chip cases fail on dev with an empty chip'
status: To Do
assignee: []
created_date: '2026-10-21 07:46'
labels:
  - tests
  - console
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while checking TASK-33620.15.7 (2026-10-21). Under the plain command, `Tests/UI/test_console_voice_chip.py` fails at setup with `RecoveryRequired: raw_source_selection_changed` on this machine. With every case marked `bootstrap_profile`, 9 of its 15 cases fail, the same 9 on dev 906497d9b4 and on that branch. Seven assert copy inside the chip and find it painted empty (the chip carries the `w-0` class), for example "assert '◉' in ''" and "assert 'No microphone access.' in ''"; one reads `chip.styles.width.value` and gets None; one fails a bare `assert False`. Among them is `test_idle_collapses_a_chip_that_was_showing`. The suite therefore cannot show whether the voice chip still paints its states, which is the coverage the chip's idle-path change in TASK-33620.15.7 would have relied on.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 It is established whether the chip really paints empty in these states at the suite's terminal width (a product bug) or the cases are stale (a test bug)
- [ ] #2 The voice chip suite passes under the plain command on dev
- [ ] #3 Each repaired case fails when the chip's copy is broken
<!-- AC:END -->
