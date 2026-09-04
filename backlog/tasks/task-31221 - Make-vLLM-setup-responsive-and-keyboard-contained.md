---
id: TASK-31221
title: Make vLLM setup responsive and keyboard-contained
status: Done
assignee:
  - codex
created_date: '2026-09-03 22:34'
labels:
  - vllm
  - lab
  - accessibility
  - responsive
dependencies:
  - TASK-31214
  - TASK-31215
  - TASK-31217
  - TASK-31219
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the complete vLLM setup, activity, profile, and Console-handoff workflow remains visible and operable at supported terminal sizes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 80x24, 100x30, and 120x40 every visible focusable descendant remains within its owning pane.
- [x] #2 Labels, inputs, and actions stack at compact widths without truncating the recovery or primary-action meaning.
- [x] #3 Tab traversal stays within the active provider pane and lifecycle transitions move focus to the newly relevant action.
- [x] #4 Provider navigation uses one documented key meaning that does not conflict with the Lab footer.
- [x] #5 Production-stylesheet compositor and keyboard tests cover first-run, loading, ready, failure, current-versus-next, and handoff states.
<!-- AC:END -->

## Implementation Plan

1. Add production-stylesheet compositor coverage for the exact 8-state × 3-size matrix below, including every visible focusable's containment in its owning pane and complete state-specific Tab walks.
2. Remove provider-child bracket/digit bindings and derive `vllm-wide`, `vllm-medium`, or `vllm-compact` from the mounted vLLM body width.
3. Re-compose setup groups at medium/compact widths, collapse the catalog through the existing Lab rail store at compact width, preserve the reopen control, and keep readiness plus the next action in the first viewport with a conditional fold cue.
4. Make only explicit lifecycle transitions focus their phase action (`Start vLLM`, `Stop`, `Use in Console`, or recovery); prove timer/background refresh preserves focus.
5. Edit `tldw_chatbook/css/features/_lab.tcss`, regenerate `tldw_chatbook/css/tldw_cli_modular.tcss`, and run the two focused verification matrices plus exact static checks.
6. Drive the production Lab surface at every supported size under disposable HOME/XDG/config/data/cache roots, fingerprint real state before/after, and record whether the host can honestly qualify a real vLLM/model flow.
7. Map every approved spec Goal, State model, Validation, Responsive, and Testing bullet to a concrete test node before closing the task.

Exact geometry matrix:

| State | 80×24 | 100×30 | 120×40 |
|---|---|---|---|
| `setup_incomplete` | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk |
| `preflight_ready` | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk |
| `launching` | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk |
| `loading` | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk |
| `ready` | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk |
| `failed` | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk |
| `dirty_restart` | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk |
| `profile_management` | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk | production CSS containment + complete Tab walk |

ADR required: no

ADR path: `backlog/decisions/115-vllm-lab-console-readiness-and-profiles.md`

Reason: TASK-31221 directly implements ADR-115's already accepted responsive composition, keyboard ownership, focus, and live-evidence contract; it introduces no new runtime, persistence, security, or cross-module boundary.

## Implementation Notes

Implemented a measured, production-stylesheet vLLM compositor with three body-width
classes, stable semantic focus styling, lifecycle-specific top actions, compact
stacking, a painted fold cue, persisted rail collapse/reopen behavior, and literal
Tab containment. Provider-child bracket/digit bindings were removed so the Lab
frame remains the sole owner of its documented mode-focus keys. Passive projection
preserves focus; explicit state changes land on Start, Stop, Use in Console, or the
recovery action. `_lab.tcss` remains the source and the generated modular bundle was
rebuilt deterministically.

Fix Round 1 gates profile deletion behind the repository's existing confirmation
dialog pattern. Cancel and Escape preserve the exact profile document, confirmation
deletes the captured selected/revision claim once, and a changed selection or
revision fails closed. The real dialog is keyboard-contained and fully inside its
modal pane at all three supported terminal sizes.

Fix Round 2 makes each deletion-dialog presentation terminally one-shot. Rapid
Confirm/Confirm, Confirm/Cancel, and Cancel/Confirm queues settle one callback and
one captured claim, disable the terminal controls immediately, close only the
dialog, and leave the owning Models screen current.

ADR required: no. ADR-115 already owns the responsive, focus, runtime, persistence,
privacy, and Console-handoff boundaries; this task adds no new architectural choice.

### Approved-spec evidence map

Every bullet below names its implementing test node (parameterized cases inherit the
same node name).

**Goals**

1. Prerequisites/recovery before Start — `test_initial_vllm_setup_is_guided_and_blocks_start`, `test_preflight_issue_settles_owner_view_and_recovery`.
2. Local launch and existing-server modes — `test_source_specific_controls_and_mode_drafts_are_preserved`, `test_existing_server_accepts_namespace_model_id`.
3. Separate process/API/model/Console/default states — `test_ready_requires_health_and_exact_models_identity`, `test_lifecycle_projection_enables_stop_only_while_runtime_is_active`, `test_handoff_buttons_enable_only_for_current_verified_target`, `test_vllm_handoff_stages_only_current_target_and_uses_normal_navigation`.
4. No-retype Lab-to-Console handoff — `test_vllm_handoff_intents_are_secret_free_exact_and_strict`, `test_vllm_handoff_stages_only_current_target_and_uses_normal_navigation`.
5. Device-local reusable profiles and confirmed deletion — `test_default_path_is_device_local_active_profile_data`, `test_profile_round_trip_has_exact_v1_schema_and_excludes_launch_only_fields`, `test_profile_repository_io_is_threaded_and_selected_profile_restores`, `test_profile_delete_cancel_or_escape_preserves_exact_document`, `test_confirmed_profile_delete_executes_selected_claim_once_and_recreates_default`, `test_profile_delete_confirmation_rejects_stale_selection_claim`, `test_profile_delete_queued_terminal_actions_settle_once`.
6. Immutable current versus editable next restart — `test_launch_snapshot_is_immutable_exact_and_changed_labels_are_allowlisted`, `test_current_server_is_separate_from_modified_next_restart_without_path_leak`.
7. Bounded sanitized activity/recovery — `test_owner_keeps_only_current_operation_bounded_allowlisted_activity`, `test_owner_snapshot_excludes_launch_privacy_canaries`, `test_mounted_activity_renders_ready_and_expands_bounded_failure`.
8. Focus-correct at all three sizes — `test_every_visible_focusable_is_inside_its_owner`, `test_complete_tab_walk_stays_in_active_vllm_provider`, `test_background_projection_preserves_focus_but_explicit_transition_moves_it`.

**State model**

1. Not configured — `test_initial_vllm_setup_is_guided_and_blocks_start`, geometry `setup_incomplete`.
2. Checking — `test_owned_settlement_requires_token_bound_to_the_launch_claim` proves the current owner remains Checking until a claim-bound settlement.
3. Ready to start — `test_preflight_blocker_is_adjacent_and_start_enables_only_for_current_success`, geometry `preflight_ready`.
4. Launching — geometry `launching`, `test_explicit_vllm_state_transition_focuses_phase_action[launching-vllm-stop]`.
5. Loading model — `test_ready_requires_health_and_exact_models_identity`, geometry `loading`.
6. API ready — `test_ready_requires_health_and_exact_models_identity`, `test_handoff_buttons_enable_only_for_current_verified_target`, geometry `ready`.
7. Console connected/adopted — `test_vllm_handoff_stages_only_current_target_and_uses_normal_navigation`; the existing session acceptance contract, not a second vLLM runtime state, owns navigation.
8. Stopping — `test_stop_request_settles_the_owned_server_without_opening_a_picker`, `test_failed_stop_keeps_recovery_state`.
9. Needs attention — `test_preflight_issue_settles_owner_view_and_recovery`, `test_probe_timeout_is_bounded_and_sanitized`, geometry `failed`.
10. Semantic target edits invalidate connection evidence — `test_semantic_fingerprint_changes_for_every_launch_field_except_profile_name`, `test_mounted_draft_edit_fences_old_readiness_generation`, `test_invalidate_advances_generation_and_clears_ready_target`.
11. Profile-name-only edits do not change the launch fingerprint, while launch fields do — `test_semantic_fingerprint_changes_for_every_launch_field_except_profile_name`, `test_live_claim_retry_keeps_exact_launch_after_non_network_draft_edit`.

**Validation and recovery**

1. `python_unavailable` — `test_command_rejects_stale_or_failed_preflight`, `test_preflight_issue_settles_owner_view_and_recovery`.
2. Missing or mismatched vLLM CLI — `test_bare_python_requires_sibling_vllm_not_path_lookup`, `test_explicit_environment_rejects_unrelated_global_vllm_cli`, `test_preflight_rejects_a_non_executable_vllm_cli`.
3. Required/invalid model — `test_preflight_accepts_hugging_face_repository_ids`, `test_preflight_validates_selected_local_model_directory`, `test_preflight_reports_missing_local_model_directory`.
4. Busy/invalid port — `test_preflight_rejects_out_of_bounds_structured_values`; the production live drive additionally settled to the bounded `port_unavailable` recovery classification without exposing socket details.
5. Network exposure and loopback normalization — `test_defaults_are_real_and_safe_values`, `test_wildcard_binds_use_loopback_client_urls`, `test_current_server_is_separate_from_modified_next_restart_without_path_leak`.
6. Managed/secret argument conflict — `test_raw_arguments_cannot_override_managed_or_secret_flags`, `test_profile_round_trip_has_exact_v1_schema_and_excludes_launch_only_fields`.
7. Early process exit — `test_process_exit_prevents_any_http_probe`, `test_process_exit_during_probe_prevents_ready_publication`.
8. Health timeout — `test_probe_timeout_is_bounded_and_sanitized`.
9. Expected model missing — `test_healthy_api_without_exact_model_is_not_ready`, `test_existing_server_rejects_path_like_or_noncanonical_model_ids`.
10. Credential required/configured credential — `test_auth_required_never_echoes_response_or_credential`, `test_configured_authorization_is_used_without_entering_result`.
11. Profile unavailable/repair and deletion confirmation — `test_load_rejects_invalid_profile_field_types_and_values`, `test_profile_accepts_nonexistent_safe_local_directory_for_repair`, `test_corrupt_document_fails_closed_without_overwrite`, `test_profile_delete_cancel_or_escape_preserves_exact_document`, `test_profile_delete_confirmation_rejects_stale_selection_claim`.
12. Profile-store unavailable/future-safe recovery — `test_future_version_is_preserved_byte_for_byte`, `test_atomic_write_failure_preserves_old_bytes`, `test_fresh_save_ownership_failure_precedes_all_filesystem_mutation`.
13. No exception/HTTP/path/model/process leakage — `test_preflight_rejects_oversize_or_unclassified_probe_output`, `test_owner_snapshot_excludes_launch_privacy_canaries`, `test_vllm_handoff_intents_are_secret_free_exact_and_strict`, `test_current_server_is_separate_from_modified_next_restart_without_path_leak`.
14. Allowlisted bounded Activity only — `test_owner_keeps_only_current_operation_bounded_allowlisted_activity`, `test_mounted_activity_renders_ready_and_expands_bounded_failure`, `test_oversized_models_response_is_rejected_without_retention`.

**Responsive and keyboard contract**

1. Wide body composition, horizontal-capable setup, and above-fold readiness/action — `test_every_visible_focusable_is_inside_its_owner[state-size2]` across all eight states (`size2` is 120x40); the spec permits rather than requires both rails to remain open.
2. Medium Inspector collapse, stacked overflowing groups, and conditional fold cue — `test_every_visible_focusable_is_inside_its_owner[state-size1]` across all eight states (`size1` is 100x30).
3. Compact persisted catalog collapse plus painted standard reopen action — `test_every_visible_focusable_is_inside_its_owner[state-size0]` across all eight states (`size0` is 80x24).
4. Compact complete rows/full-width Browse/owned focus geometry — `test_every_visible_focusable_is_inside_its_owner[profile_management-size0]`, including local Browse; `test_profile_delete_confirmation_is_contained_and_keyboard_cancelable` proves the real deletion dialog's controls remain inside both direct owner and modal viewport at 80x24, 100x30, and 120x40.
5. Compact readiness plus current next action and fold cue in the first paint — `test_every_visible_focusable_is_inside_its_owner[state-size0]` asserts the state action and `more below` are rendered.
6. Displayed/enabled controls only, no hidden provider bodies, cyclic active-pane Tab order — `test_complete_tab_walk_stays_in_active_vllm_provider` for the exact eight-state by three-size matrix.
7. Explicit transition focus and passive refresh preservation — `test_explicit_vllm_state_transition_focuses_phase_action`, `test_background_projection_preserves_focus_but_explicit_transition_moves_it`.
8. Provider selection remains Arrow/Enter and Escape remains the existing Lab convention — `Tests/UI/test_lab_frame_mode_keys.py` and `Tests/UI/test_lab_frame.py` in the compatibility matrix.
9. Brackets retain Lab mode-focus ownership; provider digits/brackets are absent — `test_provider_child_has_no_bracket_or_digit_bindings`, `Tests/UI/test_lab_frame_mode_keys.py`.
10. Profile-deletion Confirm, Cancel, and Escape are terminally one-shot — `test_profile_delete_queued_terminal_actions_settle_once`, `test_profile_delete_cancel_or_escape_preserves_exact_document`.

**Testing and evidence**

1. Pure source/path/field projection — `test_preflight_accepts_hugging_face_repository_ids`, `test_preflight_validates_selected_local_model_directory`, `test_preflight_reports_missing_local_model_directory`, `test_preflight_rejects_out_of_bounds_structured_values`.
2. Public CLI plus managed/secret rejection — `test_local_command_uses_public_cli_and_one_served_alias`, `test_raw_arguments_cannot_override_managed_or_secret_flags`.
3. Wildcard IPv4/IPv6 normalization — `test_wildcard_binds_use_loopback_client_urls`.
4. Fingerprint/generation invalidation — `test_semantic_fingerprint_changes_for_every_launch_field_except_profile_name`, `test_older_generation_cannot_replace_newer_owner_state`, `test_mounted_recomposition_and_detach_invalidate_readiness_generation`.
5. Served-model/path-ID admissibility — `test_chatbook_owned_ready_result_requires_exact_served_alias`, `test_existing_server_rejects_path_like_or_noncanonical_model_ids`, `test_existing_server_accepts_namespace_model_id`.
6. Product-state projection — `test_lifecycle_sync_projects_vllm_without_legacy_button_queries`, `test_lifecycle_projection_enables_stop_only_while_runtime_is_active`, `test_explicit_vllm_state_transition_focuses_phase_action`.
7. Profile limits/round trip/atomic/corrupt/future/confirmed-deletion cases — `test_repository_caps_profiles_at_32`, `test_profile_round_trip_has_exact_v1_schema_and_excludes_launch_only_fields`, `test_atomic_write_failure_preserves_old_bytes`, `test_corrupt_document_fails_closed_without_overwrite`, `test_future_version_is_preserved_byte_for_byte`, `test_profile_delete_cancel_or_escape_preserves_exact_document`, `test_confirmed_profile_delete_executes_selected_claim_once_and_recreates_default`, `test_profile_delete_confirmation_rejects_stale_selection_claim`, `test_profile_delete_queued_terminal_actions_settle_once`.
8. Loopback health/models ready/missing/auth/timeout/cancel/stale contracts — `test_ready_requires_health_and_exact_models_identity`, `test_healthy_api_without_exact_model_is_not_ready`, `test_auth_required_never_echoes_response_or_credential`, `test_probe_timeout_is_bounded_and_sanitized`, `test_cancellation_prevents_any_http_probe`, `test_older_generation_cannot_replace_newer_owner_state`.
9. Exact launch/stop/restart process ownership — `test_stop_request_settles_the_owned_server_without_opening_a_picker`, `test_restart_proves_old_process_dead_and_released_before_new_generation`, `test_restart_termination_failure_keeps_old_snapshot_and_never_reserves`.
10. Mounted session adoption without config writes — `test_vllm_handoff_stages_only_current_target_and_uses_normal_navigation`.
11. Durable-default delegation preserving unrelated settings — `test_existing_chat_action_routes_ignore_later_new_chat_default`; this exact compatibility node is a documented pre-existing baseline failure at Task 5 base `0643c2713a`, original feature base `127cc898ab`, and fetched `origin/dev` `d6eb7fe1c2`, not introduced by TASK-31221.
12. Recomposition/profile restore/obsolete-worker invalidation — `test_profile_repository_io_is_threaded_and_selected_profile_restores`, `test_mounted_recomposition_and_detach_invalidate_readiness_generation`.
13. Production stylesheet geometry states — `test_every_visible_focusable_is_inside_its_owner` covers `setup_incomplete`, `preflight_ready`, `launching`, `loading`, `ready`, `failed`, `dirty_restart`, and `profile_management` at 80x24, 100x30, and 120x40, with every visible enabled focusable fully inside both its direct owner and the active vLLM viewport; `test_profile_delete_confirmation_is_contained_and_keyboard_cancelable` applies the same rule to the real modal.
14. Complete Tab walk/hidden providers/lifecycle landing — `test_complete_tab_walk_stays_in_active_vllm_provider`, `test_explicit_vllm_state_transition_focuses_phase_action`, `test_background_projection_preserves_focus_but_explicit_transition_moves_it`.
15. Live evidence — production `TldwCli` was driven under disposable HOME/XDG/config/data/cache roots at all three sizes. The host has cached `Qwen/Qwen2.5-0.5B-Instruct`, but neither a `vllm` executable nor importable `vllm` package; therefore no real server/model was launched and loopback tests are contract verification, not real-vLLM qualification. Real config and data aggregate fingerprints were unchanged before/after.

### Verification and trade-offs

The final primary focused matrix passed 216 tests. The compatibility matrix passed
271 tests and retained two unrelated baseline failures. Both exact failures
reproduce in detached worktrees at Task 5 base `0643c2713a`, original feature base
`127cc898ab`, and the real `origin/dev` fetched at `2026-09-04T06:46:11Z`
(`d6eb7fe1c24188ead22359b6bc8d0713de2829fa`), proving they are not TASK-31221
regressions. No unrelated Console test-harness/database repair was included. Fix
Round 1 changed no stylesheet source, so the generated bundle was intentionally not
rebuilt. The Impeccable detector returned no findings.

Modified files: `vllm_setup_view.py`, `LLM_Management_Window.py`, `llm_screen.py`,
`_lab.tcss`, generated `tldw_cli_modular.tcss`,
`Tests/UI/test_vllm_lab_workflow.py`, and new
`Tests/UI/test_vllm_lab_geometry.py`.

Fix Round 1 modified only `llm_screen.py`, `test_vllm_lab_workflow.py`,
`test_vllm_lab_geometry.py`, and this task ledger.

Fix Round 2 modified only `llm_screen.py`, `test_vllm_lab_workflow.py`, and this
task ledger. The shared `ConfirmationDialog` and its public semantics remain
unchanged; the one-shot dismissal guard is private to vLLM profile deletion.

No lesson entry was added: this task produced no new generalizable incident beyond
the production-CSS/layout evidence rules already recorded in
`backlog/docs/lessons-testing-evidence.md` and the isolated-state rules in
`backlog/docs/lessons-live-verification.md`.
