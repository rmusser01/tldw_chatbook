---
id: TASK-35662
title: >-
  Readiness projection mode-bar test fails on dev because its stub screen has
  no rate reading
status: To Do
assignee: []
created_date: '2026-10-19 18:05'
labels:
  - console
  - tests
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`test_actual_mode_bar_defers_cold_config_and_uses_owned_presentation` in Tests/UI/test_console_readiness_config_projection.py fails on dev ce4aa4a231 (macOS, plain command) with `AttributeError: 'types.SimpleNamespace' object has no attribute '_console_rate_reading'`. The other 26 tests in the file pass.

The tokens-per-second readout (c5ceca1247, TASK-35100) made the mode bar read the screen's `_console_rate_reading` while a run is active. The test drives the real mode-bar method on a SimpleNamespace screen, which has no such attribute. The file is not in scripts/ui_pr_gate_census.txt, so no required lane runs it, and the red went unnoticed.

Found under TASK-35660 while running the callers of the shared provider-selection helper; that branch fails the same way, and its changes do not reach this test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test passes on dev under the plain command, and still fails if the mode bar stops deferring a cold config read
- [ ] #2 Whether other tests drive the mode bar on a stub screen without the rate reading is checked, and any found are fixed the same way
<!-- AC:END -->
