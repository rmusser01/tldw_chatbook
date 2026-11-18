---
id: TASK-41655
title: "Console, Workspaces and DB tests red on dev in files the burn-down touched"
status: In Progress
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
- [x] #2 Nodes that turn out to be load flakes are recorded as such, with a passing isolated run
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (test drift repairs only; no product behaviour changes).

1. Re-run every listed node on dev d0486978dd (after #3627) and record which are already green.
2. Re-run each red node alone to separate deterministic reds from load flakes.
3. For each deterministic red, find the commit that made it red (probe older dev merges in a detached worktree) and decide: test drift (retarget, naming the commit), or a product/design cause (report with node ids; do not restore any code-identity check, ADR-356 rule 4).
4. Fix the test drifts; run the covering files, `Tests/Architecture` serially and preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Re-checked on dev d0486978dd (2026-11-18, load 20-100 on 18 CPUs). Per node:

**Already green on d0486978dd (#3627):** the three `test_console_semantic_mutation_inventory.py` nodes. `test_repository_scan_results_are_cached_and_immutable` is a load-timeout flake: its whole-repo AST scan took 286 s alone at load 75-84 against the 300 s default timeout (passed alone; timed out at `-n 4` and once alone at load 80-90).

**Fixed here (test drift):**
- `test_trigger_census.py`, both nodes x both params: v85->v86 (7970fbcada) added five `console_prompt_mixin_*` triggers and rewrote `conversations_ad`, and v84->v85 (dd8d60c9e8) rewrote `conversations_sync_update`, without pinning them. Each new digest was computed from the declaring migration's SQL text and matches the live one. Base red 4/4, head 19/19.
- `test_console_trace_provenance.py::test_explicit_prepared_request_constructor_rejects_missing_on_provenance`: 7970fbcada added the derived `cache_boundary` field (`init=False`); the test fed every field back to the constructor and died with a `TypeError`. It now passes constructor fields only and reaches the missing-provenance refusal.
- `test_console_configuration_preparation.py::test_ready_local_skill_capture_runs_original_body_on_worker_and_matches_snapshot`: since 11d71e5db5 (#3480), synchronous capture under a managed runtime reads only closed MCP metadata (`MCP/console_snapshot.py::_mcp_metadata_is_closed`: no `policy_enforcer`, stock manifest loaded). `runtime_case`'s loop projection installs both a policy enforcer and a custom manifest provider, so the legacy snapshot is pending with an empty ceiling. The test now asserts the open metadata and the deferral. It no longer asserts session ephemerality, because the deferral does not depend on it. The owned-vs-legacy parity over every field is unchanged.
  - **Dropped property, for the owner to see.** The test used to assert "legacy synchronous capture still reads MCP; stock temporary sessions capture an empty ceiling because they never compose MCP". #3480 (11d71e5db5) removed that behaviour, and #3480 was merged without owner approval. This repair records the new behaviour; it does not endorse it. Under #3480 an open-metadata capture is pending, and a pending capture makes the ordinary footprint opaque (`console_ordinary_tree.py` `_capture_maximum`), even for a temporary session.
  - **New node for the non-empty case:** `test_closed_mcp_metadata_gives_owned_and_legacy_capture_one_ceiling`. With closed metadata (stock manifest, no enforcer), both captures produce the same non-empty ceiling and agree on every field except the skill nonce and `mcp_definition_complete`. Only the owned capture can prove its definition set complete; the synchronous `_capture_mcp_definition_selection` always reports incomplete. Mutation check: dropping one definition from the owned capture's maximum fails the node, on `mcp_definition_maximum`/`mcp_tool_maximum`. The whole file passes at `-n 4` (24 passed).

**Load flakes (AC #2):** `Tests/Workspaces/test_native_confirmed_undo.py`, the 3 listed nodes. They fail on an 8 s `_wait_for` for a worker thread. `test_finish_return_retains_original_worker_db_and_scratch` passed in the base `-n 4` run and alone at load 84 (20 s), and failed alone 3 times at load 85-100 ("timed out waiting for actual finish entry"). `test_historical_smudge_helper_retains_writer_after_ui_unmount[cancel]` and `[timeout]` passed alone together (load ~90), and failed alone in the next run. Under load, a different 4-6 nodes of this 33-node file fail in each run.

**Not fixed, reported to the coordinator (AC #1 open until they are filed):**
- `test_console_ssh_custody.py` (4 nodes), `test_console_ordinary_tree.py` (2), `test_console_reference_custody.py` (2): deterministic, red alone. They were red when #3480 merged (e10ce63fe8) and at its branch tip. They came with #3480 and were never green on dev. Causes:
  - ssh `stock_local[False-False]` / `virtual_source[False]`: the MCP capture in the rig is never witnessed, so `_capture_maximum` marks the footprint opaque at its first clause. Submitting without the caller snapshot does not help. TASK-41060 already owns `stock_local`.
  - ssh `caller_snapshot[True]` raises "Original MCP capture source is unavailable".
  - ssh `changed_original_provider_callback[handler]` returns "ordinary local handler source changed".
  - The ordinary-tree nodes are TASK-41028's (and TASK-41312's notes).
  - reference `activity`: a held opaque claim now refuses the whole Send ("Workspace activity is busy").
  - reference `zero_roots`: ordinary admission always calls `RuntimeCapacity.begin_execution`, which the test forbids.
  - The two reference nodes have since been rewritten under the owner decision below.
- `test_console_skill_trust_stock_owned.py[send_enter]`, `[send_button]`: "Cold Send did not reach original configuration SQL" since they landed (#3323, 4a38b3d2db), on every dev merge probed. With a 60 s SQL budget they hit the 10 s control deadline, so this is not timing. TASK-34563.35 (In Progress) owns this behaviour. These two nodes do not pin a code-identity refusal, so TASK-41319's D2 retarget does not apply.

**Evidence.** Base `-n 4` run of all 10 listed files: 103 failed, 411 passed (83 of the failures are TASK-41653's sweep). Head: trigger census plus historical bootstrap 102 passed serially; trace provenance, configuration preparation and semantic inventory at `-n 4`: 203 passed, plus 1 load timeout that passed alone in 286 s. `./scripts/preflight.sh` passed. `Tests/Architecture` serially on head: 3 failed, 1476 passed. The 3 failures are `test_persistent_diagnostic_inventory.py::test_reviewed_diagnostic_changes_are_metadata_only` and two `test_timer_path_static_update_inventory.py` nodes, and the same 3 fail on base d0486978dd.

**Root cause of the census drift.** The checklist in `tldw_chatbook/DB/migrations/README.md` named `VALID_TABLES` and the index census only. Its new steps 7 and 8 cover `expected_table_columns.py` and the trigger census, including how to compute a trigger digest from the migration's SQL.

**Reference custody: owner decision (2026-11-18, option A, "Keep refusing the Send").** Both #3480 nodes were rewritten to pin the current behaviour, and each keeps the safety property it was after:
- `test_unadmitted_reference_never_calls_custom_parser[activity]`: while another workspace activity holds the worktree, the Send is refused with "Workspace activity is busy; retry after it finishes." The refusal comes before any reference parsing (the custom parser is never called), execution owner, tool operation, provider call or persisted row. The `[no-roots]` param keeps its old assertions; the tool-operation check now applies to it too.
- `test_zero_roots_has_no_native_operation_or_git_launch`: every ordinary turn takes its one execution owner, the design TASK-40715 built on, so the test now allows exactly one owner instead of forbidding all of them. Both references are refused ("no allowed workspace roots are configured"), with no tool operation reserved and no git launch. The sandbox is not created, and capacity drains.
- Mutation checks (product restored after each):
  - Calling the parser inside ordinary admission, before the refusal, fails both `unadmitted` params ("failed admission still invoked the effectful parser").
  - Letting `_ReferenceRead.prepare` reserve an operation with zero roots fails `zero_roots` ("a reference reserved a native operation without roots").
- `Tests/Chat/test_console_reference_custody.py` at `-n 4`: 39 passed.

**Filing (coordinator).** These unowned nodes stay open, so the task stays In Progress:
- ssh custody: `test_caller_snapshot_cannot_prove_mcp_footprint_is_bounded[True-ssh_console0]`, `test_changed_original_provider_callback_refuses_after_enqueue[handler]`, `test_original_local_virtual_source_and_roots_are_in_initial_maximum[False-ssh_console0]`.

**Files.** `Tests/ChaChaNotesDB/test_trigger_census.py`, `Tests/Chat/test_console_trace_provenance.py`, `Tests/Chat/test_console_configuration_preparation.py`, `Tests/Chat/test_console_reference_custody.py`, `tldw_chatbook/DB/migrations/README.md`.
<!-- SECTION:NOTES:END -->
