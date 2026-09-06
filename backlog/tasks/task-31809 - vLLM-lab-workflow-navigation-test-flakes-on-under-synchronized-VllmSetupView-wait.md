---
id: TASK-31809
title: >-
  vLLM lab workflow navigation test flakes on under-synchronized VllmSetupView
  wait
status: To Do
assignee: []
created_date: '2026-09-06 00:45'
labels:
  - tests
  - flake
dependencies: []
priority: medium
---

## Description (the why)

`Tests/UI/test_vllm_lab_workflow.py::test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff`
flakes with `textual.css.query.NoMatches: No nodes match 'VllmSetupView' on
LLMScreen()` in its SETUP phase, before the behavior under test. Measured
during task-31808 on the unfixed dev baseline: 2 of 5 runs failed at this
site (the other 3 reached the then-red handoff assertion), and post-fix the
same flake reproduced at a similar rate, so it is pre-existing and
machine-load-sensitive, unrelated to the warm-handoff fix.

Mechanism: the wait loop at the first LLM visit polls
`first_screen._vllm_profiles_loaded and list(first_screen.query(VllmSetupView))`
for up to 30 bare `pilot.pause()` iterations but then asserts only
`_vllm_profiles_loaded`; five more bare pauses later it calls
`query_one(VllmSetupView)`, which raises when the view has not mounted yet.
The loop's exit condition and its post-condition assert are not the same
predicate.

## Acceptance Criteria (the what)

- [ ] The setup wait loop's budget and exit predicate cover the condition the
      test then asserts (view mounted, not just profiles loaded).
- [ ] 10 consecutive local runs of the test pass without a NoMatches setup
      failure.
