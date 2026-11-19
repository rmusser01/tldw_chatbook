---
id: TASK-36161
title: >-
  Tests: three out-of-lane test files are red on origin/dev (size ratchet, async
  MCP snapshot, local control service)
status: To Do
assignee: []
created_date: '2026-10-22 18:16'
updated_date: '2026-11-03 06:05'
labels:
  - tests
  - ci
  - dev-red
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while rebasing TASK-33620.15.5's branch onto origin/dev 7700c63a68 (2026-10-22). Three test files that no PR-required lane runs fail on a detached origin/dev worktree, with the same test ids on the branch, so they are dev's own reds, seen only after merge by nightly-deep's full `pytest ./Tests/` run (test.yml runs on main push or dispatch only) (see lessons-testing-evidence: 'A regression test no PR lane runs went red within days'). Tests/Architecture/test_module_size_ratchet.py: 8 rows over budget (mcp_workbench.py 6954/6760, change_review_screen.py 4978/4967, llm_screen.py 5185/5180, console_transcript.py 8652/8353, app_command_providers.py 1133/1123, app_ingest_queue.py 4999/4985, app_lifecycle.py 2338/2134, tldw_api/client.py 16795/16687); CI's derived-artifacts job runs only the app.py, app_startup.py, app_served.py and three Settings rows. Tests/Chat/test_console_async_mcp_snapshot.py: 26 failures, mostly AttributeError on private names the code no longer has (ConsolePromptQueueUIController._launch_chain_async, ConsoleChatController._compose_local_provider_async) plus off-loop assertions. Tests/MCP/test_local_control_service.py: 3 failures, AttributeError: 'FakeLocalStore' object has no attribute 'path'. Reproduce: run each file alone from a detached origin/dev worktree (`.venv/bin/python -m pytest <file> -q`); the reds are deterministic (line counts, missing attributes), not timing.

**Addendum (2026-11-19, TASK-41633): nodes still red on dev d0486978dd in the async MCP snapshot and snapshot source-contract files (missing private APIs, preparation-source refusal, console_snapshot_owner_changed).**
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_async_local_composer_reads_switch_off_loop`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_queued_capture_rejects_replaced_controller_before_queue_or_draft_commit`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_screen_wiring_prepares_native_mcp_sources_off_ui_loop[normal]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_screen_wiring_prepares_native_mcp_sources_off_ui_loop[queued]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_accepts_only_mirroring_of_unchanged_composer[]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_accepts_only_mirroring_of_unchanged_composer[previous draft]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_accepts_only_mirroring_of_unchanged_composer[unchanged input]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[app_config]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[attachment]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[builder]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[cancel]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[draft]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[evidence]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[prefill]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[session_controller]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_refuses_drift_without_turn_or_draft_loss[stash]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_still_refuses_real_draft_drift[edit_away_and_back]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_still_refuses_real_draft_drift[edit]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_still_refuses_real_draft_drift[same_text_scope]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_preparation_still_refuses_real_draft_drift[store_only]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_user_entry_issues_archive_proof_without_changing_captured_configuration[normal]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_actual_wired_user_entry_issues_archive_proof_without_changing_captured_configuration[queued]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_async_maximum_rejects_equal_callable_in_retained_binding_slot[inventory]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_async_maximum_rejects_equal_callable_in_retained_binding_slot[permission_marker]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_async_maximum_rejects_replaced_inventory_callback_during_native_read`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_async_snapshot_joins_sources_and_composes_on_actual_loop[True]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_local_async_composer_retains_native_custody_and_refuses_changed_owner[app_config]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_local_async_composer_retains_native_custody_and_refuses_changed_owner[cancel]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_local_async_composer_retains_native_custody_and_refuses_changed_owner[path]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_local_async_composer_retains_native_custody_and_refuses_changed_owner[session]`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_mcp_projected_names_match_original_provider_after_deny_and_collisions`
- `Tests/Chat/test_console_async_mcp_snapshot.py::test_queue_capture_reads_native_mcp_sources_off_caller_loop`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_custom_permission_switch_keeps_original_deny_and_caller_contract[instance-catalog]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_empty_mcp_live_clears_previous_counts`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_empty_mcp_maximum_keeps_ordinary_bounds[custom-empty-set]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_empty_mcp_maximum_keeps_ordinary_bounds[nonempty]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_empty_mcp_maximum_keeps_ordinary_bounds[unset]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_empty_mcp_maximum_preserves_plugin_route[plugin_maximum0]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_empty_mcp_maximum_preserves_plugin_route[plugin_maximum1]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_empty_stock_mcp_maximum_skips_catalog_work[dispatch]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_empty_stock_mcp_maximum_skips_catalog_work[preview]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_exact_native_custom_catalog_keeps_late_inventory_receiver`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_source_dependency_retarget_during_native_body_is_refused[log_descriptor]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_source_dependency_retarget_during_native_body_is_refused[permission_descriptor]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_source_dependency_retarget_during_native_body_is_refused[permission_switch_descriptor]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_source_dependency_retarget_during_native_body_is_refused[permission_switch]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_source_dependency_retarget_during_native_body_is_refused[store_load]`
- `Tests/MCP/test_console_snapshot_source_contracts.py::test_unrelated_custom_catalog_descriptor_is_not_read_by_preparation[local]`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each named test id passes on origin/dev, or the test is retired with a recorded reason
- [x] #2 Size-ratchet rows are re-pinned at measured values with dated owner-decision comments, or the files shrink back under them
- [ ] #3 Each of the three files is on a PR-required lane, or the reason it is not is recorded
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-10-22, on origin/dev e2cffbed60: a fourth file, same shape. Tests/MCP/test_local_server_tools.py::test_deny_state_handler_returns_tool_result fails on a detached origin/dev worktree and on TASK-33620.15.5's rebased branch with the same id (ToolResult.approval_decision 'denied' != None: the handler now records the denial the test's expected value omits). derived-artifacts.yml runs three other Tests/MCP files but not this one, so only nightly-deep sees it. The other three files' reds are unchanged on e2cffbed60 (size ratchet: the same 8 rows).
2026-11-03: AC#2 is met by TASK-38640. #3351 re-pinned every over-budget module-size row at its measured size with a dated waiver comment, and the module- and screen-size ratchet files pass on dev 18482c1cf7 (205 passed). AC#1 and AC#3 still cover the other files.
<!-- SECTION:NOTES:END -->
