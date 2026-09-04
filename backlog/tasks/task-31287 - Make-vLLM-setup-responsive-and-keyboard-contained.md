---
id: TASK-31287
title: Make vLLM setup responsive and keyboard-contained
status: Done
assignee:
  - codex
created_date: '2026-09-03 22:34'
updated_date: '2026-09-04 18:11'
labels:
  - vllm
  - lab
  - accessibility
  - responsive
dependencies:
  - TASK-31283
  - TASK-31284
  - TASK-31285
  - TASK-31286
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
- [x] #6 While vLLM is active, the outer Lab header, status, and visible Inspector project selected profile, ownership, safe verified endpoint/model, persistence scope, Current versus Next state, and the reachable next action; projection refreshes without focus theft and respects compact/medium Inspector collapse.
- [x] #7 The production-stylesheet evidence matrix covers actual Checking, existing-server discovery and selection, Console/default presentations where applicable, mounted edit -> check -> restart, and presentation recomposition with assertion-level mappings for every claimed outcome.
- [x] #8 The contextual Inspector describes Console use truthfully before and after handoff/return as session-only with defaults unchanged, without claiming an adoption state the Lab does not own.
- [x] #9 The 11-state responsive/Tab matrix treats the existing-server profile selector as disabled and unreachable while keeping Local-mode selection reachable, and fresh-screen hydration preserves truthful Use/Reverify/recovery actions without stealing focus.
- [x] #10 During initial hydration the mounted Lab hides Use, Console, and Reverify actions derived only from unreconciled app state, keeps exact owned Stop reachable, and projects persistent adjacent profile-store recovery on load failure without focus theft.
- [x] #11 The production-mounted delayed-hydration flow renders only non-verified readiness, checklist, and activity copy before reconciliation—even after lifecycle refresh—then restores the exact verified projection without focus theft after hydration.
- [x] #12 During delayed hydration all visible vLLM draft, mode, check, and profile mutation controls are disabled and absent from actionable Tab flow, while exact owned Stop and non-destructive disclosures remain operable.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
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

ADR path: `backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md`

Reason: TASK-31287 directly implements ADR-117's already accepted responsive composition, keyboard ownership, focus, and live-evidence contract; it introduces no new runtime, persistence, security, or cross-module boundary.

Task 6 Fix Round 2:
1. Reproduce the primary-suite FD-growth warning, bisect/group the owning test files, and inventory descriptor types/owners before changing code.
2. Fix a branch-owned production leak if the evidence identifies one; otherwise add only evidence-backed harness cleanup at the smallest test owner.
3. Re-run the qualified primary matrix with the FD gate below its hard threshold, then record the measurements and exact owner evidence before restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This is verification-resource hygiene within ADR-117's existing test/evidence contract, not a runtime architecture change.

Final UX fix round:
3. Expand the RED production-stylesheet matrix with actual Checking, external discovery/selection, Console/default presentations, contextual outer-Lab projection, and complete state-specific focus/containment assertions at 80x24, 100x30, and 120x40.
4. Add RED mounted tests for focus-preserving contextual refresh, collapsed Inspector behavior, safe canonical Current/Next copy, and the real active-runtime edit -> Check draft -> Restart flow using a controlled process owner.
5. Implement the smallest vLLM view/outer-Lab projection and `_lab.tcss` source changes; regenerate the modular bundle and keep the first viewport's next action reachable without forbidden bindings.
6. Run the Impeccable detector once after UI completion, then execute sequential focused GREEN nodes, the qualified primary and incumbent compatibility matrices, geometry/Tab matrices, CSS sync twice, statics, inventories, live isolated qualification, and diff review.
7. Replace every stale evidence-map claim with an assertion-level mapping to the exact new node/case, record honest live limitations, check the new ACs only when verified, and restore Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This round directly completes ADR-117's already accepted contextual Lab, responsive, focus, and evidence obligations; it adds no new application structure or ownership boundary.

UX Fix Round 2/5:
8. Add a RED mounted handoff/return regression for state-agnostic truthful Inspector persistence copy.
9. Replace only the vLLM Inspector's unsupported adoption-state claim; keep verified target, current/next, focus, and responsive projection unchanged.
10. Run focused GREEN plus the complete primary/geometry/compatibility/CSS/static/inventory/diff qualification before checking the new AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This is copy correction within ADR-117's existing session/default handoff boundary, not a persistence or ownership change.

UX Fix Round 3/5:
11. Extend the RED geometry/Tab matrix for existing-server selector containment and add mounted fresh-screen no-focus-theft assertions.
12. Keep the existing adaptive composition and focus policy; change only enabled/reachable actions derived from mode and exact readiness evidence.
13. Run the 71-case geometry gate, workflow and full primary suites, compatibility, generated CSS, statics, inventories, privacy, and diff checks before checking the AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already fixes the responsive containment, focus, and truthful readiness requirements; no new UI architecture is introduced.

UX Fix Round 4/5:
14. Add RED mounted assertions for the delayed-hydration action/focus surface and persistent adjacent profile-store failure recovery.
15. Change only the view/controller projection needed to mask unreconciled READY state while preserving independently truthful runtime Stop.
16. Re-run workflow, geometry, primary, compatibility, CSS, static, inventory, privacy, and diff gates before checking the new AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This directly enforces ADR-117's existing truthful-action and focus contract during initial profile hydration; it introduces no new UI or handoff architecture.

UX Fix Round 5/5:
17. Expand the existing production-shaped delayed-hydration RED test to assert readiness label, all four checklist rows, activity summary/details, Use/default visibility, staging, Stop, and focus across initial mount and explicit lifecycle refresh.
18. Change only the child projection fence needed to keep every READY-derived surface truthful before reconciliation; preserve current copy, layout, keyboard order, and exact post-hydration presentation.
19. Re-run workflow, the 71-case geometry/Tab matrix, five-file primary, compatibility, CSS, privacy, inventories, statics, scope, and diff gates before checking the AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This completes ADR-117's existing truthful loading-state projection and adds no new visual or application structure.
Final closure:
20. Extend the production delayed-hydration assertions to every mutation control and exercise real click/press/edit behavior without changing focus or exact owner evidence.
21. Preserve the established adaptive layout, disclosure controls, and independently truthful Stop while adding only the pending-hydration interaction fence.
22. Re-run workflow, the 71-case geometry/Tab matrix, full qualification, CSS determinism, and direct static gates before restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This is an interaction-state correction within ADR-117's existing responsive and hydration model; no new UI structure, copy system, or keyboard grammar is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
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

Fix Round 3 makes the first terminal outcome own the deletion dialog's public
`result` as well as its callback, captured claim, repository effect, and single
stack pop. Later queued Confirm, Cancel, Escape, or backdrop actions cannot rewrite
that result. The guard remains private to vLLM profile deletion.

ADR required: no. ADR-117 already owns the responsive, focus, runtime, persistence,
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
7. Bounded sanitized activity/recovery — `test_owner_keeps_only_current_operation_bounded_allowlisted_activity`, `test_owner_snapshot_excludes_launch_privacy_canaries`, `test_mounted_activity_renders_ready_and_expands_bounded_failure`, `test_probe_deadline_reports_overall_thirty_second_elapsed_bucket`, `test_vllm_failure_details_never_cross_logs_notifications_or_global_state`.
8. Focus-correct at all three sizes — `test_every_visible_focusable_is_inside_its_owner`, `test_complete_tab_walk_stays_in_active_vllm_provider`, `test_background_projection_preserves_focus_but_explicit_transition_moves_it`.

**State model**

1. Not configured — `test_initial_vllm_setup_is_guided_and_blocks_start`, geometry `setup_incomplete`.
2. Checking — geometry `checking`, `test_checking_exposes_generation_bound_cancel_action`, and `test_mounted_cancel_check_only_cancels_current_generation` prove the visible current-generation Cancel action and owner settlement boundary.
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

1. Wide body composition, horizontal-capable setup, and above-fold readiness/action — `test_every_visible_focusable_is_inside_its_owner[state-size2]` across all eleven states (`size2` is 120x40); the spec permits rather than requires both rails to remain open.
2. Medium Inspector collapse, stacked overflowing groups, and conditional fold cue — `test_every_visible_focusable_is_inside_its_owner[state-size1]` across all eleven states (`size1` is 100x30).
3. Compact persisted catalog collapse plus painted standard reopen action — `test_every_visible_focusable_is_inside_its_owner[state-size0]` across all eleven states (`size0` is 80x24).
4. Compact complete rows/full-width Browse/owned focus geometry — `test_every_visible_focusable_is_inside_its_owner[profile_management-size0]`, including local Browse; `test_profile_delete_confirmation_is_contained_and_keyboard_cancelable` proves the real deletion dialog's controls remain inside both direct owner and modal viewport at 80x24, 100x30, and 120x40.
5. Compact readiness plus current next action and fold cue in the first paint — `test_every_visible_focusable_is_inside_its_owner[state-size0]` asserts the state action and `more below` are rendered.
6. Displayed/enabled controls only, no hidden provider bodies, cyclic active-pane Tab order — `test_complete_tab_walk_stays_in_active_vllm_provider` for the exact eleven-state by three-size matrix, including Check/Cancel, external selector, Use in Console, and Make default presentations where applicable.
7. Explicit transition focus and passive refresh preservation — `test_explicit_vllm_state_transition_focuses_phase_action`, `test_background_projection_preserves_focus_but_explicit_transition_moves_it`.
8. Provider selection remains Arrow/Enter and Escape remains the existing Lab convention — `Tests/UI/test_lab_frame_mode_keys.py` and `Tests/UI/test_lab_frame.py` in the compatibility matrix.
9. Brackets retain Lab mode-focus ownership; provider digits/brackets are absent — `test_provider_child_has_no_bracket_or_digit_bindings`, `Tests/UI/test_lab_frame_mode_keys.py`.
10. Profile-deletion Confirm, Cancel, Escape, and backdrop outcomes are terminally one-shot, and the first terminal action owns the public dialog result — `test_profile_delete_queued_terminal_actions_settle_once`, `test_profile_delete_cancel_or_escape_preserves_exact_document`.

**Testing and evidence**

1. Pure source/path/field projection — `test_preflight_accepts_hugging_face_repository_ids`, `test_preflight_validates_selected_local_model_directory`, `test_preflight_reports_missing_local_model_directory`, `test_preflight_rejects_out_of_bounds_structured_values`.
2. Public CLI plus managed/secret rejection — `test_local_command_uses_public_cli_and_one_served_alias`, `test_raw_arguments_cannot_override_managed_or_secret_flags`.
3. Wildcard IPv4/IPv6 availability and normalization — `test_ipv6_wildcard_availability_checks_the_requested_bind`, `test_wildcard_binds_use_loopback_client_urls`.
4. Fingerprint/generation invalidation — `test_semantic_fingerprint_changes_for_every_launch_field_except_profile_name`, `test_older_generation_cannot_replace_newer_owner_state`, `test_mounted_recomposition_and_detach_invalidate_readiness_generation`.
5. Served-model/path-ID admissibility — `test_chatbook_owned_ready_result_requires_exact_served_alias`, `test_existing_server_rejects_path_like_or_noncanonical_model_ids`, `test_existing_server_accepts_namespace_model_id`.
6. Product-state projection — `test_lifecycle_sync_projects_vllm_without_legacy_button_queries`, `test_lifecycle_projection_enables_stop_only_while_runtime_is_active`, `test_explicit_vllm_state_transition_focuses_phase_action`.
7. Profile limits/round trip/atomic/corrupt/future/confirmed-deletion cases — `test_repository_caps_profiles_at_32`, `test_profile_round_trip_has_exact_v1_schema_and_excludes_launch_only_fields`, `test_atomic_write_failure_preserves_old_bytes`, `test_corrupt_document_fails_closed_without_overwrite`, `test_future_version_is_preserved_byte_for_byte`, `test_profile_delete_cancel_or_escape_preserves_exact_document`, `test_confirmed_profile_delete_executes_selected_claim_once_and_recreates_default`, `test_profile_delete_confirmation_rejects_stale_selection_claim`, `test_profile_delete_queued_terminal_actions_settle_once`.
8. Loopback health/models ready/missing/auth/timeout/cancel/stale contracts — `test_ready_requires_health_and_exact_models_identity`, `test_healthy_api_without_exact_model_is_not_ready`, `test_auth_required_never_echoes_response_or_credential`, `test_probe_timeout_is_bounded_and_sanitized`, `test_cancellation_prevents_any_http_probe`, `test_older_generation_cannot_replace_newer_owner_state`.
9. Exact launch/stop/restart process ownership — `test_stop_request_settles_the_owned_server_without_opening_a_picker`, `test_restart_proves_old_process_dead_and_released_before_new_generation`, `test_restart_termination_failure_keeps_old_snapshot_and_never_reserves`.
10. Mounted session adoption without config writes — `test_vllm_handoff_stages_only_current_target_and_uses_normal_navigation`.
11. Durable-default delegation preserving unrelated settings — `test_existing_chat_action_routes_ignore_later_new_chat_default`; this exact compatibility node is a documented pre-existing baseline failure at Task 5 base `0643c2713a`, original feature base `127cc898ab`, and fetched `origin/dev` `d6eb7fe1c2`, not introduced by TASK-31287.
12. Recomposition/profile restore/obsolete-worker invalidation — `test_profile_repository_io_is_threaded_and_selected_profile_restores`, `test_mounted_recomposition_and_detach_invalidate_readiness_generation`.
13. Production stylesheet geometry states — `test_every_visible_focusable_is_inside_its_owner` covers `setup_incomplete`, `checking`, `preflight_ready`, `launching`, `loading`, `ready`, `failed`, `dirty_restart`, `profile_management`, `existing_discovery`, and `existing_ready` at 80x24, 100x30, and 120x40, with per-state outcome copy and first-action assertions plus every visible enabled focusable fully inside both its direct owner and the active vLLM viewport; `test_profile_delete_confirmation_is_contained_and_keyboard_cancelable` applies the same rule to the real modal.
14. Complete Tab walk/hidden providers/lifecycle landing — `test_complete_tab_walk_stays_in_active_vllm_provider`, `test_explicit_vllm_state_transition_focuses_phase_action`, `test_background_projection_preserves_focus_but_explicit_transition_moves_it`.
15. Live evidence — production `TldwCli` was driven under disposable HOME/XDG/config/data/cache roots at all three sizes. The host has cached `Qwen/Qwen2.5-0.5B-Instruct`, but neither a `vllm` executable nor importable `vllm` package; therefore no real server/model was launched and loopback tests are contract verification, not real-vLLM qualification. Real config and data aggregate fingerprints were unchanged before/after.

### Verification and trade-offs

The final primary focused matrix passed 217 tests. The compatibility matrix passed
271 tests and retained two unrelated baseline failures. Both exact failures
reproduce in detached worktrees at Task 5 base `0643c2713a`, original feature base
`127cc898ab`, and the real `origin/dev` fetched at `2026-09-04T06:46:11Z`
(`d6eb7fe1c24188ead22359b6bc8d0713de2829fa`), proving they are not TASK-31287
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

Fix Round 3 modified only `llm_screen.py`, `test_vllm_lab_workflow.py`, and this
task ledger. The vLLM-private guard now makes the first terminal result immutable;
the shared `ConfirmationDialog` and every other call site remain untouched.

Task 6 integration fix round closes four verified review findings through the
smallest owning seams: field-adjacent preflight recovery (including opening a
collapsed Advanced field), requested-address IPv6 wildcard availability,
30-second overall readiness-window Activity, and real notifier/logger/global-state
privacy canaries. It also reconciles the feature-owned Console handoff assertion
with the current upstream structured readiness summary. Exact RED/GREEN and final
matrix evidence is recorded in the Task 6 report.

The original responsive work produced no new lesson beyond the existing
production-CSS/layout and isolated-state rules. Task 6 Fix Round 2 did expose a
generalizable worker-thread database teardown trap; its measured incident and
registry-drain rule are now recorded in
`backlog/docs/lessons-testing-evidence.md`.

## Renumbering provenance

This task previously held id TASK-31221. During the branch integration sweep,
current `origin/dev` already shipped `task-31221 -
Media-type-chooser-options-are-invisible-zero-height-OptionList.md` at add commit
`f9577ba8a913b09c523b643193dbbf1eb777a3af`. The unmerged vLLM task therefore
moved to collision-free TASK-31287 as the last member of a monotonic vLLM task
block, carrying every dependency and documentation
reference with it. The vLLM record was originally added by
`ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0`.

Task 6 Fix Round 2 scientifically isolated the primary FD warning. The original
220-test primary ended at 249 descriptors from 12 (`+237`, limit 200). Core
setup/connection/profile tests grew only `12 -> 16`; the UI
workflow/geometry group grew `12 -> 351`, workflow alone `12 -> 115`, and a
representative geometry factory group `12 -> 18`. Live `lsof` plus GC owner
inspection found one non-accumulating KQUEUE/event-loop socket pair, but one
worker-thread `_QuiescentSQLiteConnection` per test-created app/profile. The
test factory let the real on-mount FTS backfill create that connection while
fixture teardown closed only the main thread's handle. This is harness
ownership: production creates one long-lived app/profile rather than dozens.

The smallest cleanup extends the existing config-singleton teardown to enter
`ChaChaNotesDB.quiesce_connections`, which drains and closes every registered
same-file worker handle; simpler DB doubles retain their old `close` fallback.
Two mounted cases went from 2 live SQLite/9 regular descriptors after final
teardown to 0 SQLite/3 regular descriptors. Four repeated geometry mounts held
stable at `12 -> 15` total rather than linear growth. The final qualified
primary with `TLDW_TEST_FD_GROWTH_LIMIT=200` passed `273` tests in `263.28s`
with no FD warning. No production lifecycle change or new ADR was warranted;
ADR-117 already owns the test/evidence contract.

The final UX fix round makes the outer Lab frame contextual while vLLM is
active. Its header, status chips, and wide Inspector project selected profile,
runtime ownership, the safe canonical verified endpoint/model, persistence
scope, Current versus Next, and the reachable action; generic destination rows
are hidden for this provider. Selection, lifecycle, draft, and passive
recomposition refresh that projection without focus theft, while compact and
medium layouts keep the Inspector collapsed. The production-stylesheet matrix
now asserts eleven real states at 80x24, 100x30, and 120x40: setup incomplete,
checking, preflight ready, launching, loading, ready, failed, dirty restart,
profile management, external discovery, and external ready. Each cell asserts
state copy, first reachable action, selector null/exact state where relevant,
containment, and the complete active-pane Tab order.
`test_outer_lab_chrome_tracks_verified_vllm_context_without_focus_theft` maps
contextual chrome; `test_mounted_external_selection_starts_fresh_exact_probe`
and `test_mounted_external_changed_list_clears_and_fences_stale_selection` map
external selection; `test_mounted_edit_check_and_restart_uses_exact_live_claim`
maps edit/check/restart; and
`test_mounted_recomposition_preserves_exact_readiness_but_detach_invalidates`
maps presentation recomposition. The matrix's repeated factory apps explicitly
close their Library Collections, Workspace, Subscriptions, and Evaluations
database owners after each mount; the previously offending 120x40 slice moved
from `+42` descriptors to no warning under a tightened `20`-descriptor limit
without reducing any state/size assertions. No new ADR or generalized lesson:
ADR-117 already owns the contextual Lab/evidence requirements, and the existing
testing lesson already records repeated-app SQLite-owner cleanup.

Final qualification: `308` primary tests passed in `428.25s` with the full
eleven-state by three-size geometry/Tab matrix and no descriptor-growth warning.
Production CSS build/sync/staleness passed `39` tests; the bundle was generated
twice from `_lab.tcss` with identical SHA-256
`8dd093edc0a8a6ce6281c42f39eb7c450b59146dea7b9e9e28bc6dfa903b32ae`.
The broader compatibility run passed `358` with three unrelated failures, all
reproduced on exact base `d3d6a031379d5ffbd6545b4463e798c2ed83dd84`
(Settings origin helper absent, existing-chat terminal persistence baseline,
and Research-to-Library destination mapping). The static/inventory matrix
passed `136`, skipped `1`, and retained five exact-base failures in untouched
persistent-diagnostic/timer ownership; direct profile-path and diagnostic
inventory checks pass with no drift. A broad CSS class-coverage probe likewise
retains only existing Console/Library missing-style debt in untouched sources;
the generated bundle diff contains only the Lab additions. Impeccable's
post-edit detector reported no findings. Format, critical Ruff, `py_compile`,
privacy/scope review, and `git diff --check` pass. The host has no installed or
importable vLLM, so live server qualification remains unavailable and no
unrelated service or model was downloaded or started.

UX Fix Round 2/5 makes the Inspector persistence statement state-agnostic:
`Console use is session-only; defaults unchanged`. Mounted assertions cover the
same copy before and after staging the verified target for Console use. The
final production-stylesheet primary passed `325` tests, including all `71`
eleven-state/three-size geometry and complete Tab-order cases. Compatibility
passed `350/352`; both failures and every source path in their traces are
unchanged from exact base `d3d6a03`. CSS build/sync/staleness passed `39`; two
fresh builds reproduced SHA-256
`8dd093edc0a8a6ce6281c42f39eb7c450b59146dea7b9e9e28bc6dfa903b32ae`
with no generated diff. Direct profile and diagnostic inventories passed.
The broader static replay passed `134`, skipped `1`, and retained seven
exact-base failures: the previously recorded five persistent-diagnostic/timer
findings plus two untouched worker-inventory findings. Critical Ruff,
`compileall`, scoped format, privacy/scope review, Impeccable's post-edit
detector, and `git diff --check` pass. The host remains without an installed or
importable vLLM, so no live-vLLM claim, download, or unrelated service start was
made. ADR-117 remains the governing UX/evidence record; no new ADR or lesson was
needed.

UX Fix Round 3/5 removes the disabled Existing-server profile selector from the
complete active-pane Tab sequence in every responsive state/size cell while
retaining the Local selector. Fresh-screen navigation assertions cover exact
READY/Use/Stop projection, mismatch recovery, and focus preservation without
using the same-screen recomposition seam. The assertion-level owners are
`test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff`,
`test_fresh_screen_mismatched_profile_invalidates_ready_target_safely`,
`test_existing_mode_forged_profile_events_preserve_repository`,
`test_every_visible_focusable_is_inside_its_owner`, and
`test_complete_tab_walk_stays_in_active_vllm_provider`. Workflow passed `60`; all `71`
production-stylesheet geometry/Tab cases passed. The final complete normal-FD
five-file primary passed `329` in `418.16s` after the last test-harness
tightening. Compatibility remains `350/352`, with the same two untouched
Console/Settings failures recorded in Round 2. CSS build/sync/staleness passed
`39`; two builds reproduced SHA-256
`8dd093edc0a8a6ce6281c42f39eb7c450b59146dea7b9e9e28bc6dfa903b32ae`
without generated drift. The documented profile/persistent-diagnostic command
reported `79 passed, 1 skipped, 2 failed`; both failures are the already recorded
untouched persistent-diagnostic owners, while diagnostic path privacy passed
`98`. The prior Round 2 `134/1/7` broad-static aggregate is provenance-only: its
exact invocation was not recorded and was not guessed or presented as replayed.
Critical Ruff, scoped format, `compileall`, and the one valid post-edit Impeccable
scan pass. ADR-117 remains sufficient; no new ADR or generalized lesson arose.

UX Fix Round 4/5 adds the pending-profile and failed-store presentations to the
real mounted workflow without changing the existing responsive composition or
keyboard grammar. Until hydration reconciles, the child view and contextual
outer Lab mask inherited READY/verified target copy, keep Use, Make default, and
Reverify out of the action surface, disable profile controls, and retain Stop
for an exact live owned process. A lifecycle projection cannot bypass that
fence. Profile-load failure persists adjacent repair/reload copy and an outer
`Profiles need repair` status while preserving the user's current focus. The
production-shaped assertions live in
`test_navigation_to_fresh_models_screen_preserves_exact_ready_handoff` and
`test_fresh_screen_profile_load_failure_invalidates_ready_with_recovery`; the
unchanged eleven-state/three-size matrix remains the assertion owner for
containment and complete Tab order.

Round 4 verification passed workflow `65`, geometry/Tab `71`, and the full
five-file primary `334` in `411.92s` with the normal FD limit. Compatibility
remains `350/352`, matching the two untouched Console/Settings baseline nodes.
CSS build/sync/staleness passed `39`; two source builds reproduced SHA-256
`8dd093edc0a8a6ce6281c42f39eb7c450b59146dea7b9e9e28bc6dfa903b32ae`
without generated drift. Profile and diagnostic inventories, critical Ruff,
scoped format, `compileall`, the `7`-case privacy selection, forbidden-scope
review, and `git diff --check` passed. No stylesheet source changed in this
round. The exact prior Round 2 broad-static aggregate remains provenance-only
and was not guessed. The host has neither an installed executable nor importable
vLLM module, so loopback remains contract evidence rather than live-server
qualification. ADR-117 remains sufficient; no new ADR or lesson was required.
UX Fix Round 5/5 extends the real delayed-hydration navigation regression to
assert every visible truth surface before and after explicit lifecycle refresh:
readiness remains Setup incomplete, no checklist row is verified, Activity stays
non-ready, Use/default remain absent, staging refuses, Stop remains independently
truthful, and focus is preserved. Exact repository reconciliation restores the
verified endpoint/model, Model and Network checks, ready Activity, and both
handoff actions. The geometry projection now declares its reconciled fixture
state rather than depending on an unsafe child default; all eleven states and
three supported sizes retain production-CSS containment and Tab order.

Round 5 evidence is workflow `70`, geometry/Tab `71`, and complete primary `339`
passing under the normal FD limit. Compatibility remains `350/352` with only the
same untouched Console/Settings baseline nodes. CSS tests passed `39`; two builds
reproduced SHA-256
`8dd093edc0a8a6ce6281c42f39eb7c450b59146dea7b9e9e28bc6dfa903b32ae`
and bundle sync passed. Privacy `7`, critical Ruff, scoped format, `compileall`,
both inventories, scope review, and diff checks passed. No stylesheet source or
generated bundle changed, and no live-vLLM claim was possible on this host.
ADR-117 remains sufficient; no new ADR or generalized lesson arose.

Final closure adds the missing interaction fence without changing layout or
keyboard grammar. While hydration is pending, every visible draft, mode,
profile, Check, and launch mutation control is disabled and therefore absent
from the actionable Tab sequence; non-destructive Collapsible disclosures are
unchanged, and exact owned Stop remains enabled and focusable. The production
delayed-load regression attempts mouse click, programmatic press, direct field
edits, and forged screen messages, then proves exact readiness returns only
after reconciliation. Its pre-fix state exposed `19` mutation controls; the
focused final set passed `5`.

Final qualification passed workflow `74`, all `71` eleven-state/three-size
production-stylesheet geometry and complete Tab-order cases, and the five-file
primary `343` in `446.64s`. Compatibility retains the same two untouched
baseline failures (`350/352`). Privacy `7`, CSS `39`, deterministic double build
and sync, critical Ruff, scoped format, `compileall`, both inventories, scope
review, and `git diff --check` passed. The CSS source and generated bundles are
unchanged. ADR-117 remains sufficient; no new ADR or lesson arose.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

A second merge-time sweep found that `origin/dev` had advanced to
`1a1b5c19e0bb3243effb1ae9671158b6670ad6da` and now canonically claimed the
intermediate TASK-31263 and TASK-31264 IDs for unrelated theme follow-up work.
The complete vLLM sequence therefore moved together from TASK-31263..31268 to
the next contiguous block proven free across every fetched non-vLLM ref,
TASK-31282..31287. This responsive-completion task maps TASK-31268 ->
TASK-31287; ADR-117 remained collision-free.
