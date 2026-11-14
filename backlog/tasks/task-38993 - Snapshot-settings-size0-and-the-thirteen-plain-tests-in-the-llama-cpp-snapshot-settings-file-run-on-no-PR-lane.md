---
id: TASK-38993
title: "Snapshot settings [size0] and the thirteen plain tests in the llama.cpp snapshot settings file run on no PR lane"
status: Done
assignee: ['@claude']
created_date: 2026-11-06 01:42
updated_date: 2026-11-08 00:45
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
- [x] #1 test_f9_snapshot_controls_paint_and_keyboard_save[size0] runs on a PR lane, with its measured lane time recorded (8.3 s alone on dev)
- [x] #2 The 13 plain tests pass locally without a temporary marker, either because TASK-38111 is fixed or because each carries a private-profile marker
- [x] #3 The whole file runs on a PR lane, with its measured whole-file time at -n 4 recorded, and each laned test still fails under a temporary source edit that removes the behaviour it pins (one negative control per test function)
<!-- AC:END -->

## Implementation Plan

1. Give each of the six plain test functions `@private_profile_test` (each runs on its own fresh profile, as `[size1]` does), so they pass without a temporary marker.
2. Lane `[size0]` and the plain tests as UI census tail entries of about 30 s or less, split by node where needed; measure with the UI step's command.
3. One negative control per laned test function: a temporary source edit that removes the pinned behaviour turns it red.

## Implementation Notes

**Approach.** Each of the six plain test functions in `Tests/UI/test_llamacpp_snapshot_settings.py` now carries `@private_profile_test` (and takes `request`), as `[size1]` did, so it runs on its own fresh profile in a child process. All 15 nodes are on the UI census: 14 tail entries (the three malformed-preferences cases and the six canonicalize cases one per case, `[size0]`, and one per other test) beside the existing `[size1]` entry. The canonicalize cases got named ids (`leading-zero`, `plus-sign`, `padded`): the census takes no whitespace in an entry, and the old `[False- 27 ]` id has two spaces.

Laning exposed two pause-based waits, now waits on state:
- `test_success_canonicalizes_unchanged_f9_draft_and_continues_once`: 4 of 6 cases failed (`'027' == '27'`) after `pilot.pause(0.2)`: the save thread's write landed but the worker had not yet canonicalized the field. It now waits until `_snapshot_preferences_saving` clears, through the file's existing `_until` helper.
- `test_malformed_f9_preferences_leave_provider_surface_and_revert_recovery`: 3 of 3 failed serially at load 50-155 with `NoMatches: '#settings-revert-category'`: the guided Save/Revert pair had not mounted after one pause. It now waits for the button to exist.

**Red evidence (AC#2).** Before the markers, the whole file at `-n 4`: 13 failed (all `RecoveryRequired: raw_source_selection_changed`), 2 passed. After: 15 passed in 4 of 5 runs at `-n 4` (83-118 s whole file, load 45-110); the fifth, at load about 110, had one failure that two reruns did not reproduce (not identified).

**Lane times (AC#1, AC#3).** Serially with the UI step's command (`--timeout=180 --durations=0`), at load 50-155: `[size0]` 38 s; the per-entry costs 21-38 s, except the canonicalize cases about 27 s each. Whole file at `-n 4`: 83-118 s.

**Negative controls (AC#3), one per test function**, each a temporary edit of `tldw_chatbook/UI/Screens/settings_screen.py`, run through run.sh at `-n 4`, then reverted (source unchanged in the commit):
- malformed preferences: Revert leaves the keep field enabled when the load failed: 3 failed;
- non-integer values: coerce with `int(float(...))` instead of `validate_bounded_integer`: 1 failed;
- canonicalize: drop the field's canonical rewrite: 6 failed;
- preserves new edits: canonicalize even when the draft changed (`if True:`): 1 failed;
- off-thread: call the save on the event loop instead of `asyncio.to_thread`: 1 failed;
- stale draft: save with `expected=None`: 1 failed;
- `[size0]` keyboard Save: the category Save skips the snapshot draft: 1 failed.

**Files.** `Tests/UI/test_llamacpp_snapshot_settings.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py` (floor). The census commit is shared with TASK-39144, which re-dealt the census after these entries joined.

**Review r12l-rev1 (2026-11-08).** `test_f9_stale_snapshot_draft_requires_revert_before_save` was still vacuous: it waited `pilot.pause(0.2)` and checked for "Revert", which the draft's own copy shows before any Save. It now waits for the save worker to settle and asserts the refusal's "changed elsewhere" copy. With the conflict branch's copy removed from `settings_screen.py` (temporary edit, reverted), the new body failed and the old body passed; green without the edit.
