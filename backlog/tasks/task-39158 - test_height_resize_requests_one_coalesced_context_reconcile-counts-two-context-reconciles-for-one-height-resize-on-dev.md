---
id: TASK-39158
title: "test_height_resize_requests_one_coalesced_context_reconcile counts two context reconciles for one height resize on dev"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- console
priority: low
references:
- Tests/UI/test_console_resize_reflow.py
- tldw_chatbook/UI/Screens/chat_screen.py
dependencies:
- TASK-39132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (TASK-39132's covering runs, round-12 branch fu/r12-cprod). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08.

- `Tests/UI/test_console_resize_reflow.py::test_height_resize_requests_one_coalesced_context_reconcile` is a plain test, so locally it stops at `RecoveryRequired`.
- Under a temporary `bootstrap_profile` marker it fails `assert 2 == 1`, on both dev and the stack. The marker came from a scratch plugin; runs were through run.sh at `-n 4` and load ~22-27.
- TASK-39132's notes record the same under a temporary `@private_profile_test`, with and without that task's change, 3 runs each.
- The file has one UI census entry (`test_model_summary_sync_invalidates_mounted_context_allocation`). This node runs on no lane.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The second reconcile is explained, and fixed or re-pinned: a product regression (one height resize should coalesce to one reconcile) is fixed with this test red before, and drift is re-pinned with the commit that changed the count named
- [ ] #2 The node runs plain through the suite's existing profile seams, on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->
