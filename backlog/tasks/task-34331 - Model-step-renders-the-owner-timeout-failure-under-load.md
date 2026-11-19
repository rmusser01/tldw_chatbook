---
id: TASK-34331
title: 'Model step renders the owner-timeout failure under load'
status: To Do
assignee: []
created_date: '2026-10-03 16:30'
labels:
  - first-run-wizard
  - model-step
  - tests
  - ux-review-2026-10-02
dependencies:
  - TASK-34100.1
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When the user leaves Provider before its model discovery settles, the Model step waits for that discovery, up to the discovery timeout, and then should show "Couldn't reach the server (timeout)…" with Retry and manual entry. Under load it sometimes never does.

Evidence from TASK-34100.1's reviews:
- `Tests/Wizards/test_first_run_setup_wizard.py::test_mounted_model_owner_timeout_fences_late_result_and_keeps_manual_retry` fails with "Model owner timeout did not render bounded failure" even with a 10 s wait on the condition. It failed 2 of 15 runs on base 1d8fe87659, 2 of 8 concurrent runs at load ~40 on the branch, and 1 of 3 serial runs at load ~60.
- Live, on base and on the branch: leaving Provider before a slow or unroutable llama.cpp discovery settles leaves "Checking the selected provider…" on screen indefinitely, and Model shows "None (recommended)".

Both point at the hand-off between `ProviderStep`'s selected discovery and `ModelStep`'s wait on it (`_outcome_from_selected_discovery`, `_selected_discovery_done`, the owner-timeout branch in `ModelStep`'s load). The fix may be in the product or in the test; the first step is to find which.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The root cause of the missing timeout failure is identified and recorded, with evidence that separates a product race from a test artefact
- [ ] #2 Leaving Provider before a slow discovery settles always ends on Model with either the discovered models or the timeout failure with Retry and manual entry, never an indefinite "Checking…"
- [ ] #3 The owner-timeout test passes 20 of 20 concurrent runs under load, and a test that fails on the pre-fix code covers the cause
- [ ] #4 Verified live against a real llama.cpp server made slow or unroutable, with no mock server
<!-- AC:END -->
