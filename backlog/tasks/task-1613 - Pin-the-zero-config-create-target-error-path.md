---
id: TASK-1613
title: Pin the zero-config create-target error path
status: Done
assignee: []
created_date: '2026-07-31 15:10'
updated_date: '2026-08-01 02:29'
labels:
  - evals
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Task-1482 review coverage gap. Clicking the "+ New target" mini-form's button (id #evals-bench-create-target -- task-1611 T2 renamed its rendered label from the older "Create target from configured llama.cpp server" and made it always-rendered, not only in the zero-llama_cpp-models state; the button's id and its zero-config gate are unchanged from task-1482) with no llama_cpp URL configured notifies "No llama.cpp server is configured; set one in Settings first." -- verified correct by a reviewer's live probe, but no test pins the copy or the no-row-created outcome.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test drives the zero-config click and asserts the exact toast and that no eval_models row is created
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify the current exact toast string and control id for the zero-config create-target path (evals_screen.py's _on_bench_create_target_requested, gated on sample_bench.configured_llama_cpp_url(app_config) is None) against the task's claimed copy/button label.
2. Fix the task description: the button id is #evals-bench-create-target, unchanged since task-1482, but task-1611 T2 renamed its rendered LABEL from "Create target from configured llama.cpp server" to "+ New target" (and made it always-rendered, not zero-models-gated) -- the task file's copy is stale.
3. Add a new test using the (unconfigured) evals_app fixture + bench_with_zero_llama_models: click #evals-bench-create-target, assert the exact toast text and severity, and assert no eval_models row was created.
4. Red-before-green: temporarily neutralize the gate to confirm the test fails, then restore and confirm it passes; run the full bench editor test file.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Verified the current toast copy ("No llama.cpp server is configured; set one in Settings first.", severity="error") is unchanged since the reviewer's original probe -- evals_screen.py's _on_bench_create_target_requested, gated on sample_bench.configured_llama_cpp_url(app_config) is None, before ever reaching db.create_model. Confirmed the control is #evals-bench-create-target and, per the task's own instruction, fixed the task description: task-1611 T2 renamed the button's rendered LABEL from "Create target from configured llama.cpp server" to "+ New target" (and made it always-rendered rather than zero-models-gated); the id and zero-config gate are unchanged from task-1482.
Added test_create_target_with_no_llama_cpp_server_configured_notifies_and_creates_nothing in Tests/UI/test_evals_bench_editor.py, using the (unconfigured) evals_app fixture + bench_with_zero_llama_models: clicks #evals-bench-create-target and asserts the exact toast text/severity plus evals_db.list_models(provider="llama_cpp") == [] (no row created) and no target row staged. Red-before-green: temporarily neutralized the gate (if False and ...) -- test failed on the empty-notifications assert AND the log showed a model row being created, confirming both assertions bite; restored and re-ran green.
Full Tests/UI/test_evals_bench_editor.py: 60 passed (59 + this new test).
<!-- SECTION:NOTES:END -->
