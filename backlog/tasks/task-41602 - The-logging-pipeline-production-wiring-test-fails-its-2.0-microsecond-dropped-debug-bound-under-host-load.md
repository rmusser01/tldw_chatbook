---
id: TASK-41602
title: "The logging pipeline's production-wiring test fails its 2.0 microsecond dropped-debug bound under host load"
status: To Do
assignee: []
created_date: 2026-11-17 19:40
labels:
- follow-up
- tests
- flaky
- logging
priority: low
references:
- Tests/test_logging_pipeline_single_pass.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during a round-15 covering run (2026-11-17). `Tests/test_logging_pipeline_single_pass.py::test_production_wiring_drops_debug_before_evaluating_lazy_arguments` measured 2.39 microseconds at load about 50-80 and failed. It passed 3 of 3 alone, at base and at head.

**The check** (read at 212f015a91):
- The child script (`:437-463`) times 20,000 dropped `logger.debug` calls with `time.perf_counter()`.
- The test asserts that the mean per call is under 2.0 microseconds (`:505`).
- The comment at `:504` records about 0.15 microseconds with the level gate and about 7-8 without it.
- The bound is wall-clock time on a shared host, so it measures the host's load as well as the gate.

The file runs in `test.yml`'s full suite and in `nightly-deep.yml`, not on a PR lane. TASK-33262 added the test and is Done. No open task covers this failure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The test still fails when the level gate is removed (the 7-8 microsecond path), shown red under a temporary source edit
- [ ] #2 It passes 10 of 10 at `-n 4` with host load above 60, and the notes record the load
- [ ] #3 The notes state the speed assertion and the measurement and load it rests on
<!-- AC:END -->
