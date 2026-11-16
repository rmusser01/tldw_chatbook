---
id: TASK-41136
title: "test_settings_custom_endpoints.py and test_console_provider_gateway.py run on no PR lane"
status: Done
created_date: 2026-11-15 02:20
updated_date: 2026-11-16 23:07
assignee:
- '@claude'
labels:
- follow-up
- tests
- lanes
- providers
priority: low
references:
- Tests/UI/test_settings_custom_endpoints.py
- Tests/Chat/test_console_provider_gateway.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 endpoint stream fu/r14-endpoint 6cb8e34461 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** The gateway is where Console resolves providers and endpoints, and the custom-endpoint Settings file pins how users add and fix endpoints; TASK-40300 and TASK-40302 changed both behaviours. Neither file is guarded on PRs.

**Evidence.** Neither file is named in `.github/workflows/*.yml` or `scripts/ui_pr_gate_census.txt`, at the stack top or on the endpoint branch. TASK-40300's notes record three gateway reds owned by TASK-39151.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both files' green nodes run on a PR lane (the gateway in a non-UI job, the Settings file in the UI census), with measured times
- [x] #2 Each red node is listed with its owner
<!-- AC:END -->

## Implementation Plan

1. Run both files to find their red nodes and each red node's owner.
2. Gateway file: a whole-file target in an existing non-UI job, its red nodes deselected by name with the owner named; measure it serially with that job's flags.
3. Settings file: its green nodes at the UI census tail with measured times, `MINIMUM_FILES` set to the entry count.
4. Run the census checker, Tests/CI and the job's whole step.

## Implementation Notes

- **Gateway file, non-UI lane.** `Tests/Chat/test_console_provider_gateway.py` joins `pr-fast-lane-arch` ("Run the architecture-review tests") as a whole-file target, last, because its module marker is `bootstrap_profile` (that step runs sandboxed targets first, TASK-32873). `ARCH_REVIEW_TARGETS` in `Tests/CI/test_ci_queue_pressure_contract.py` gains the file. Time: 494 passed, 3 deselected in 243.5 s serially (`-p no:xdist --timeout=180`, load ~56). The job's comment block records it.
- **Red nodes and owners.**
  - Gateway, deselected in the step with `--deselect=` (so `_pytest_targets` reads no node as a target), owner **TASK-41053** (To Do; TASK-39151 and TASK-40300 recorded them red at base): `test_all_qwencloud_kwargs_paths_forward_pinned_mode_and_base` and `test_console_send_leaves_base_url_unpinned_in_default_case[deepseek]` (handler kwargs carry `api_base_url='https://api.deepseek.com'`), `TestGatewayExchangeCapture::test_openai_stop_closes_transport_through_gateway_boundary` (stream content never recorded within 5 s).
  - Settings file, not in the census, **no owner task yet (proposed follow-up)**: `test_delete_after_detach_removes_entry`, `test_detach_preserves_entry_url_for_blank_session_base_url`, `test_convert_custom_slot_creates_entry_and_keeps_slot`, `test_edit_mutation_clearing_env_var_deletes_reference_on_disk`, `test_settings_screen_duplicate_button_opens_template_modal`, `test_settings_screen_duplicate_seeds_from_disk_when_app_snapshot_stale`, `test_settings_screen_duplicate_reports_missing_slug_instead_of_blank`, `test_settings_screen_duplicate_falls_back_to_view_config_when_snapshot_read_only`, `test_settings_screen_edit_flow_clears_env_var_reference`. All nine raise `RecoveryRequired: raw_source_selection_changed` (`raw_participants.py:307`) plainly, under a stamped `bootstrap_profile`, and under a temporary `@private_profile_test` (reverted), so this is not the local profile trap: `_activate_temp_config` sets `TLDW_CONFIG_PATH` to a tmp file mid-test and force-reloads, a source switch the config-participant admission refuses. TASK-39169 recorded the same nine red at base ae9f9e8d21. The likely repair is TASK-40300's: write the entry into the test's own selected config instead of switching sources.
- **Settings file, UI census.** Its four green nodes are appended at the tail (`MINIMUM_FILES` 2145 -> 2149): `test_custom_endpoint_rows_render_safe_display` 5.4 s (first-test session setup), `test_delete_guard_lists_referencing_sessions` 0.1 s, `test_edit_mutation_rejects_malformed_env_var_reference` 0.1 s, `test_edit_mutation_keeps_valid_env_var_without_delete` 0.3 s; 4 passed in 9.0 s serially with the UI lane's flags (`-p no:xdist --timeout=180 --durations=0`), load ~49.
- **Runs.** `scripts/check_ui_pr_gate_census.py`: OK, 2149 entries, floor 2149. `Tests/CI` at `-n 4`: 1640 passed, 1 skipped, 1 failed, the failure being TASK-41135's perf-guard trigger pin, fixed in that task's commit (12 passed after). The step's bootstrap-profile tail, from `Tests/DB/test_private_sqlite.py` through the gateway with its deselects, serially as the workflow lists it (load ~90, 1590 s): `5 failed, 966 passed, 2 skipped, 3 deselected`; every gateway node passed. The five are not this change: three inventory censuses hit the 180 s timeout at that load, one restore-timing assert read `1.11 < 1.0`, and `test_every_backup_enabled_owner_has_a_behavioral_operation` (extra `recovery.personal_context.repository`) fails alone at the base sources too. The full step was not timed locally: at load ~100 each AST census test takes 256-265 s (`test_participants_public_names.py` alone 548.8 s), so local wall time says nothing about the job's 20-minute cap; the gateway adds 4.1 min serially at load ~56, against the step's 5-8 min plus 2.5 min recorded at load 32-52.
- Files: `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`, `scripts/ui_pr_gate_census.txt`, `scripts/check_ui_pr_gate_census.py`.
