---
id: TASK-21245
title: >-
  Flake - test_overflow_focus_order_and_recovery_stay_within_context_section
  fails in isolation
status: To Do
assignee: []
created_date: '2026-08-23'
labels:
  - test-health
  - flake
  - console
  - needs-owner
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; found during
TASK-21117 and confirmed still flaky at close-out.

`Tests/UI/test_console_rail_reconciliation.py:1946`
`test_overflow_focus_order_and_recovery_stay_within_context_section` is a pre-existing flake in
a left-rail `_RailHarness` that mounts no Inspector. It goes green in large runs and
misbehaves in isolation — the direction that hides it from the suite and surfaces it during
targeted verification, i.e. exactly when a task needs a trustworthy signal.

Measurements:

- TASK-21117's interleaved A/B: base 3P/1F, branch 3P/1F — symmetric, so not caused by that
  branch.
- That implementer's larger sample: base 7P/3F, branch 10P/0F.
- Close-out re-probe on dev `b2b1e2e0d`, **15 isolated runs: 13 passed, 2 failed**. Still
  flaky, unchanged in character.

## Acceptance Criteria

- [ ] The source of the nondeterminism is identified, not suppressed with a retry, a sleep, or a skip marker
- [ ] The test passes 30 consecutive isolated runs on dev
- [ ] It still fails when the focus-order behaviour it names is mutated
- [ ] If the flake proves to be a real product race rather than a harness artefact, that defect is filed separately with its own evidence
