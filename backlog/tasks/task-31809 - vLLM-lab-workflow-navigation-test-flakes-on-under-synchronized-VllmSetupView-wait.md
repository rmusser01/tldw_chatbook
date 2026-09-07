---
id: TASK-31809
title: >-
  vLLM lab workflow navigation test flakes on under-synchronized VllmSetupView
  wait
status: Done
assignee: []
created_date: '2026-09-06 00:45'
updated_date: '2026-09-06 15:43'
labels:
  - tests
  - flake
dependencies: []
priority: medium
---

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the NoMatches flake in a loop (baseline: ~3/8 fail).
2. Instrument the setup wait loops to find the mechanism.
3. Fix the under-synchronized VllmSetupView waits and the block-load failsafe.
4. Confirm 10+ consecutive runs pass with no NoMatches setup failure.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: three test-internal synchronization gaps, all machine-load-sensitive, in `test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff`.

1. **First-visit wait.** The loop broke on `_vllm_profiles_loaded AND query(VllmSetupView)` but asserted only `_vllm_profiles_loaded`, then called `query_one(VllmSetupView)` five bare pauses later -- a run whose short budget expired with profiles loaded but the view not yet mounted passed the assert and then raised NoMatches. The deeper cause: the VllmSetupView pane (`#llm-view-vllm`) mounts lazily as part of the window's compose, and setting `active_view="vllm"` before that pane exists makes `watch_active_view` hit a QueryError that is only logged, so the deferred mount worker is never started and the view NEVER appears (instrumented: `prof@0 view@-1`). Fix: wait for `#llm-view-vllm` before switching `active_view`; then wait for BOTH profiles-loaded AND the view, and assert BOTH before querying.

2. **Fresh-visit wait.** Same lazy-pane race on the second LLM visit -- added the same pane-wait before `active_view`, and assert the view is present before `query_one`.

3. **Fresh-visit blocked load.** `_NavigationRepository.load()`'s `release_second_load.wait(5)` is a failsafe against a never-released hang, but under concurrent test load the pause-loops between `block_next_load.set()` and the pre-hydration assertions ran ~5.6s (measured), so the 5s failsafe fired early, unblocked the load, flipped `_vllm_profiles_loaded` True, and rendered "Ready at ..." where the test asserts "Setup incomplete". Fix: raise the failsafe to 60s (never fires during a slow-but-successful run; still bounds a genuine hang).

No product assertions were weakened -- only the test's own synchronization scaffolding was hardened. Validation: 12/12 consecutive passes on a shared machine at load ~8-9 (baseline was ~3/8 failing).

Files: `Tests/UI/test_vllm_lab_workflow.py`.
<!-- SECTION:NOTES:END -->

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

- [x] The setup wait loop's budget and exit predicate cover the condition the
      test then asserts (view mounted, not just profiles loaded).
- [x] 10 consecutive local runs of the test pass without a NoMatches setup
      failure.
