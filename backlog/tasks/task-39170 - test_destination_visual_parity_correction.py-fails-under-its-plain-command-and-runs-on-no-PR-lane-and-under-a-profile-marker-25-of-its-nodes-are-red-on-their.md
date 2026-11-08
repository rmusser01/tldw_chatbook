---
id: TASK-39170
title: "test_destination_visual_parity_correction.py fails under its plain command and runs on no PR lane, and under a profile marker 25 of its nodes are red on their own assertions on dev"
status: To Do
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- test-infra
- ui
priority: low
references:
- Tests/UI/test_destination_visual_parity_correction.py
- Tests/UI/app_factory.py
- Tests/conftest.py
dependencies:
- TASK-2560
- TASK-2960
- TASK-38111
- TASK-39134
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (settings stream, TASK-39134's notes, round-12 branch fu/r12-settings). Checked on dev b590a15afd, 2026-11-08 (run.sh, `-n 4`, load ~21-25).

**The file.** It has 70 test functions. Three are `@private_profile_test`, none carries `bootstrap_profile`, and no PR lane or census entry runs it.

**Plain.**
- 113 nodes fail with `RecoveryRequired: raw_source_selection_changed`, through `_build_test_app`.
- 2 fail on their own: `test_library_source_snapshot_times_out_to_stable_error` and `test_library_source_snapshot_timeout_handles_blocking_async_services` (`'types.SimpleNamespace' object has no attribute 'app_config'`).

**Under a temporary `bootstrap_profile` marker** (scratch plugin), 25 nodes are red on their own assertions:
- 6: a mode or category strip 18 rows tall (`#artifacts-mode-strip`, `#acp-mode-strip`, `#settings-category-strip`: "is too tall: Region(x=0, y=5, width=140, height=18)").
- 4: `KeyError: 'mcp'` and `KeyError: 'skills'` in the compact-size and tab-order matrices.
- 7: Watchlists. `NoMatches '#watchlists-detail-pane'`; `NoMatches '#sources-type-select'` (2); the selected row not painted in reverse video (2); "Timed out waiting for ChatScreen" (2).
- 4: copy pins. `'Last pull: —'` (Schedules) and `'Run unavailable'` (Workflows), plus the operational loading-state matrix for each.
- 1: `test_core_default_empty_or_blocked_states_keep_workbench_geometry[chat-…]` (`assert 30 < 0`).
- 1: `test_settings_boundary_note_is_fully_visible_at_rest[size0]` (140x42), "boundary note Region(x=106, y=37, width=31, height=3) outside the inspector window Region(x=106, y=19, width=31, height=18)". It is red on the round-12 stack too.
- 2: the two snapshot-timeout nodes above.

**Round 12 and open tasks.**
- TASK-39134 (round 12) rewrote one pin in this file on purpose: `test_settings_inspector_floor_yields_to_the_detail_pane_when_narrow`, 140 → 150 columns.
- TASK-39134 also saw five Settings nodes red on both trees: the two strips, the 140x42 boundary note, and `KeyError: 'mcp'` / `'skills'`.
- Open tasks already cover some of the file's nodes: TASK-2560 (Schedules nodes and a missing media DB) and TASK-2960 (MCP forced loading).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The file's nodes run under its plain command without `RecoveryRequired`, through the suite's existing profile seams; no test binds config to the real profile, and RecoveryRequired admission is unchanged
- [ ] #2 Each own-assertion red is classified as test drift or a product defect, and fixed or retired with the commit that moved the behaviour named; a product defect (such as the boundary note outside the inspector at 140x42) gets a test that is red before its fix
- [ ] #3 Overlap with TASK-2560 and TASK-2960 is reconciled in the notes
- [ ] #4 The file runs on a PR lane (never the fast PR contract), split as its timing needs, with three timings recorded
<!-- AC:END -->
