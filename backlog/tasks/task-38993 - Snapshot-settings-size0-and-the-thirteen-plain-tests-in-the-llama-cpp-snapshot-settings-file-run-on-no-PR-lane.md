---
id: TASK-38993
title: "Snapshot settings size0 and the thirteen plain tests in the llama.cpp snapshot settings file run on no PR lane"
status: To Do
created_date: 2026-11-06 01:42
labels:
- follow-up
- tests
- settings
- ci
priority: low
references:
- Tests/UI/test_llamacpp_snapshot_settings.py
- scripts/ui_pr_gate_census.txt
- Tests/private_profile.py
dependencies:
- TASK-38111
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05 (load 5-8).

`Tests/UI/test_llamacpp_snapshot_settings.py` has 15 nodes. The census lists only one of them: `test_f9_snapshot_controls_paint_and_keyboard_save[size1]` (`scripts/ui_pr_gate_census.txt:427`). No workflow targets the file.

**Not on a lane:**
- `test_f9_snapshot_controls_paint_and_keyboard_save[size0]` (80x24). It is a private-profile test like `[size1]`, and it passes locally (7.2 s call, 8.3 s alone).
- The 13 plain tests, which carry no private-profile marker:
  - `test_malformed_f9_preferences_leave_provider_surface_and_revert_recovery[bad0-2]`;
  - `test_f9_snapshot_save_rejects_noninteger_values_without_coercion`;
  - `test_success_canonicalizes_unchanged_f9_draft_and_continues_once` (6 cases);
  - `test_f9_save_preserves_new_edits_while_persistence_waits`;
  - `test_f9_snapshot_draft_uses_category_save_offthread_and_reverts`;
  - `test_f9_stale_snapshot_draft_requires_revert_before_save`.

**Locally.** A whole-file run at `-n 4` gives 13 failed and 2 passed in 12.2 s. Every failure is `RecoveryRequired: raw_source_selection_changed`, the TASK-38111 class. So the plain tests can join a lane only once TASK-38111 is fixed, or once each of them carries a private-profile marker.

TASK-36530 changed the waits of three of these tests and verified them under a temporary private-profile marker. No lane has run them since.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 test_f9_snapshot_controls_paint_and_keyboard_save[size0] runs on a PR lane, with its measured lane time recorded (8.3 s alone on dev)
- [ ] #2 The 13 plain tests pass locally without a temporary marker, either because TASK-38111 is fixed or because each carries a private-profile marker
- [ ] #3 The whole file runs on a PR lane, with its measured whole-file time at -n 4 recorded, and each laned test still fails under a temporary source edit that removes the behaviour it pins (one negative control per test function)
<!-- AC:END -->
