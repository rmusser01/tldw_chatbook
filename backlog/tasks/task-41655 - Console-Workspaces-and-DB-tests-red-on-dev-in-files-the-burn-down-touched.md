---
id: TASK-41655
title: "Console, Workspaces and DB tests red on dev in files the burn-down touched"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- tests
priority: medium
references:
- Tests/ChaChaNotesDB/test_trigger_census.py
- Tests/Chat/test_console_semantic_mutation_inventory.py
- Tests/Chat/test_console_ssh_custody.py
- Tests/Workspaces/test_native_confirmed_undo.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** The landing of #3581..#3593 compared every failing node in the stack's 50 test files against dev 45b1949ca3. These non-UI nodes fail on dev itself, not because of the stack:
- `Tests/ChaChaNotesDB/test_trigger_census.py`: `test_no_unexpected_triggers` and `test_trigger_shapes_match`, each for `chain_migrated_from_v4` and `fresh_bootstrap`
- `Tests/Chat/test_console_semantic_mutation_inventory.py`: `test_all_dynamic_sql_reviews_match_derived_production_targets`, `test_live_sql_mutation_sites_are_classified`, `test_repository_scan_results_are_cached_and_immutable`
- `Tests/Chat/test_console_ssh_custody.py`: 4 nodes (`test_caller_snapshot_cannot_prove_mcp_footprint_is_bounded`, `test_changed_original_provider_callback_refuses_after_enqueue`, `test_original_local_virtual_source_and_roots_are_in_initial_maximum`, `test_stock_local_declarations_keep_private_effects_bounded`)
- `Tests/Chat/test_console_ordinary_tree.py`: `test_actual_received_turn_with_native_killed_mcp_is_bounded[True]`, `test_skill_tool_child_joins_the_original_ordinary_claim`
- `Tests/Chat/test_console_reference_custody.py`: `test_unadmitted_reference_never_calls_custom_parser[activity]`, `test_zero_roots_has_no_native_operation_or_git_launch`
- `Tests/Chat/test_console_configuration_preparation.py`: `test_ready_local_skill_capture_runs_original_body_on_worker_and_matches_snapshot`
- `Tests/Chat/test_console_trace_provenance.py`: `test_explicit_prepared_request_constructor_rejects_missing_on_provenance`
- `Tests/Performance/test_console_skill_trust_stock_owned.py`: `test_stock_console_skill_setup_has_original_owned_lifetime[send_button]`, `[send_enter]`
- `Tests/Workspaces/test_native_confirmed_undo.py`: `test_finish_return_retains_original_worker_db_and_scratch`, `test_historical_smudge_helper_retains_writer_after_ui_unmount[cancel]`, `[timeout]`
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed node passes, or is filed as its own task with its cause
- [ ] #2 Nodes that turn out to be load flakes are recorded as such, with a passing isolated run
<!-- AC:END -->
