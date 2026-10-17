"""Contracts for the bounded fast PR lane and comprehensive CI cadence."""

from __future__ import annotations

import copy
import shlex
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_ROOT = PROJECT_ROOT / ".github" / "workflows"
PULL_REQUEST_TYPES = ["opened", "synchronize", "reopened", "ready_for_review"]
PUSH_ONLY_CANCELLATION = (
    "${{ github.event_name == 'push' && github.ref != 'refs/heads/main' }}"
)
LANES = (
    "github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && "
    "(inputs.pr != '' || github.ref != 'refs/heads/dev'))"
)
FAST_LANE_TARGETS = (
    "Tests/CI",
    "Tests/test_smoke.py",
    "Tests/Canvas/test_gateway.py::test_navigation_completion_preserves_independently_current_selection",
    "Tests/MCP/test_approval_timeout_policy.py",
    "Tests/MCP/test_control_plane_bridge.py",
    "Tests/MCP/test_live_server_request_wiring.py",
    "Tests/Agents/test_execution_capacity.py",
    "Tests/Agents/test_fleet_messages.py",
    "Tests/Agents/test_session_todo_store.py",
    "Tests/DB/test_agent_orchestration_dev_migration.py",
    "Tests/DB/test_agent_run_budget_accounting.py",
    "Tests/DB/test_automatic_work_budget.py",
    "Tests/DB/test_automatic_runtime_owner.py",
    "Tests/DB/test_automatic_wake_attempts.py",
    "Tests/Model_Artifacts/test_operation_leases.py",
    "Tests/Model_Artifacts/test_operation_leases_process.py",
    "Tests/UI/test_mcp_workbench.py",
    "Tests/UI/test_mcp_tools_mode.py",
    "Tests/Widgets/test_detach_safe_text_area.py",
    "Tests/Architecture/test_console_controllers_define_their_self_attributes.py",
    # TASK-34000.1: a Screen that flushes on navigation must answer the quit
    # walk; quit prompts are awaited only through the choke point; the Notes
    # autosave max-wait arithmetic. Static AST scans (one cached tree walk
    # each) and pure unit tests: a few seconds for the lane in total.
    "Tests/Architecture/test_flush_screens_have_quit_hooks.py",
    "Tests/Architecture/test_quit_flow_prompt_choke_point.py",
    "Tests/Library/test_library_note_autosave_max_wait.py",
    # TASK-34000 wave 1a final review (I2): the fast pins for the export seam
    # (atomic write, no-clobber, symlink write-through, the Report and
    # Collections replace checks) and for every sync status publication
    # carrying its time. About 6 s together; nothing else gated them.
    "Tests/Library/test_library_file_export.py",
    "Tests/Architecture/test_notes_sync_snapshot_construction.py",
    # TASK-34000.25: the ``[<title>](media://<uuid>)`` source line a note
    # takes from a Media item is pure and refuses anything but a uuid
    # (under a second).
    "Tests/Library/test_library_media_source_link.py",
    # TASK-34000.25 fix round 2: the compose-time gate lookup reads Textual's
    # ``_pending_children``; a Textual change fails here by name (one bare
    # ``run_test``, about a second).
    "Tests/Library/test_library_notes_canvas_pending_lookup.py",
    # TASK-34000.48: every file-profile comparison and binding commit goes
    # through the newline rule (AST scan of the three sync modules, seconds).
    "Tests/Architecture/test_notes_sync_binding_profile_commits.py",
    # TASK-34000.49: binding.note_version is never a precondition against
    # the live note (AST scan of the executor, under a second).
    "Tests/Architecture/test_notes_sync_version_proxy_sites.py",
    # TASK-34789: the pinned sync root's containment and claimed-inode rules.
    # The file ran in no pull-request lane, so its real-inode test, which only
    # reaches the recycled-number branch on a filesystem that recycles, had
    # never run on Linux before a merge. Plain filesystem unit tests, seconds.
    "Tests/Notes/test_sync_containment.py",
    # TASK-34100.5: the first-reply and handoff unit guards -- the direct
    # runtime's unavailable tools never reach the agent catalog, plain arrival
    # words, the missing-revision trace diagnostic, the provider's own reason
    # on a failed first reply, and the cold local first-token window. Pure
    # unit tests, a few seconds together.
    "Tests/Agents/test_mcp_unavailable_builtin_tools.py",
    "Tests/Chat/test_console_arrival_vocabulary.py",
    "Tests/Chat/test_console_trace_missing_revision_diagnostic.py",
    "Tests/Chat/test_first_reply_failure_copy.py",
    "Tests/Chat/test_first_token_window.py",
    "Tests/Architecture/test_surface_swap_guard.py",
    "Tests/Library/test_ingest_analysis_load_settings.py",
    "Tests/test_config_load_settings_table_guard.py",
    "Tests/Architecture/test_study_handler_service_keywords.py",
    # TASK-25907.23 / 34800: bounded privacy, closed-native and Canvas parity.
    "Tests/Personal_Context/test_context_service.py::test_device_only_additions_cannot_change_context_selection_or_explanation",
    "Tests/Personal_Context/test_context_service.py::test_quarantine_hint_cannot_change_permitted_packing",
    "Tests/Personal_Context/test_context_service.py::test_explanation_uses_the_snapshot_pass_and_excludes_ineligible_records",
    "Tests/Personal_Context/test_context_service.py::test_explanation_expiry_invalidates_without_manifest_revision_change",
    "Tests/Personal_Context/test_context_service.py::test_explanation_rejects_record_version_reuse_of_snapshot_cache_key",
    "Tests/Personal_Context/test_context_service.py::test_explanation_rejects_replacement_profile_with_reused_revisions",
    "Tests/Agents/test_profile_tool_provider.py::test_device_only_reads_hide_content_and_preserve_manual_owner_access",
    "Tests/Personal_Context/test_native_profile_barrier.py::test_v2_manifest_blocks_readable_v1_before_content_authentication",
    "Tests/Personal_Context/test_native_profile_barrier.py::test_compiled_inventory_covers_every_protected_native_call_site",
    "Tests/Personal_Context/test_native_profile_barrier.py::test_reused_operation_rechecks_manifest_after_external_commit",
    "Tests/Personal_Context/test_native_profile_barrier_routes.py::test_real_v1_write_routes_then_current_v2_deny_without_side_effects[record]",
    "Tests/Personal_Context/test_native_profile_barrier_routes.py::test_real_v1_write_routes_then_current_v2_deny_without_side_effects[proposal]",
    "Tests/Personal_Context/test_native_profile_barrier_routes.py::test_exports_deny_before_destination_write",
    "packages/tldw_profile_core/tests/test_v2_aggregates.py::test_duplicate_aware_and_compiled_json_paths",
    "packages/tldw_profile_core/tests/test_v2_aggregates.py::test_every_serializer_revalidates_unsafe_root_state",
    "packages/tldw_profile_core/tests/test_v2_meaning.py::test_fixed_projection_and_json_round_trip",
    "packages/tldw_profile_core/tests/test_v2_meaning.py::test_fixed_digest_and_scope_cannot_transplant_meaning",
    "packages/tldw_profile_core/tests/test_evidence_binding.py::test_complete_identity_changes_when_one_bound_value_changes",
    "packages/tldw_profile_core/tests/test_evidence_binding.py::test_digest_and_instance_admission_revalidate_unsafe_copies",
    "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[goal-canvas]",
    "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[native-tight]",
    "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[fenced-tight]",
    "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[unreadable]",
    "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_inspector_reserves_the_dispatched_request_budget[canvas-denied]",
    "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_changes_during_project_preview_discard_snapshot",
    "Tests/Chat/test_console_canvas_profile_preview.py::test_canvas_changes_discard_late_profile_preview",
    "Tests/Chat/test_console_canvas_profile_preview.py::test_preview_runtime_roster_retires_only_its_worker_connection",
    # TASK-34406.7: pin the same six App/startup/served size controls.
    "Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app.py]",
    "Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/app.py]",
    "Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app_startup.py]",
    "Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/app_startup.py]",
    "Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/app_served.py]",
    "Tests/Architecture/test_module_size_ratchet.py::test_budget_is_not_left_slack[tldw_chatbook/app_served.py]",
    # TASK-33007 (review of #3057, ruling C11): the size rows of the three
    # Settings modules Phase 7 pins ran in no PR lane, so a PR could grow them
    # unseen. Three node ids (under a second), not the file: it holds rows of
    # modules this phase does not touch that are red (11 at 37f5e55b5b).
    "Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Screens/settings_screen.py]",
    "Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Settings_Modules/providers_models_card.py]",
    "Tests/Architecture/test_module_size_ratchet.py::test_module_does_not_grow_past_its_budget[tldw_chatbook/UI/Settings_Modules/settings_field_rows.py]",
    # Ruling C16 (2026-10-12): the review's fix wave added these two non-UI
    # files and no lane ran them. R3 #15's Source-word resolver (pure, under a
    # second) and R3 #1, #2, #5-#7's model picker keys (17 bare mounts, ~9 s).
    "Tests/Chat/test_console_value_layers_skipped_model_defaults.py",
    "Tests/Widgets/test_model_search_picker_list_keys.py",
    # Ruling C16 for the capture fix wave (2026-10-13): six tests it added to
    # files no lane runs (node ids; each under a second, 3 of 3 serial alone).
    # Capture round 2 finding 1: Make default checks the default it writes.
    "Tests/Chat/test_console_settings_defaults.py::test_make_default_gate_validates_the_temperature_the_write_saves",
    "Tests/Chat/test_console_settings_defaults.py::test_make_default_gate_refuses_a_cleared_value_that_leaves_an_invalid_one",
    # Capture finding 11: Temperature's range is the provider's own.
    "Tests/Chat/test_sampling_params.py::test_temperature_range_is_the_providers_own",
    # Capture findings 9 and 20, round 1 finding 4: the model picker's lists.
    "Tests/Widgets/test_model_search_picker.py::test_a_saved_model_the_endpoint_serves_is_listed_as_served_now",
    "Tests/Widgets/test_model_search_picker.py::test_saved_only_list_says_catalog_refresh_is_off_when_it_is",
    "Tests/Widgets/test_model_search_picker.py::test_an_unreadable_catalog_setting_is_logged_by_type",
    # Review of #3057, finding 2: the unsaved-work check knows every Settings key.
    "Tests/Backup_Recovery/test_unsaved_editors.py::test_retained_settings_check_knows_every_key_settings_save_state_writes",
    # Its siblings: the check sees a pipeline voice edit, and the key pin
    # reads or refuses every way save_state could write a key.
    "Tests/Backup_Recovery/test_unsaved_editors.py::test_retained_speech_pipeline_voice_draft_is_compared",
    "Tests/Backup_Recovery/test_unsaved_editors.py::test_save_state_key_pin_reads_or_refuses_every_way_a_key_is_written",
    # TASK-33008 (Phase 8 review): the phase's non-UI tests ran in no PR lane.
    # The return route's navigation and handoff stores (whole files, ~2 s
    # each), the status row's cost chip (whole file, ~40 s under load), and
    # the wizard state module's trust-chain nodes. Each 3 of 3 serial. The
    # wizard file's own nodes are bootstrap_profile (its module pytestmark),
    # so they run in ADMISSION_TARGETS (review round 2).
    "Tests/State/test_conversation_settings_navigation.py",
    "Tests/State/test_pending_handoff_store.py",
    "Tests/Chat/test_console_status_chips_cost.py",
    "Tests/Wizards/test_first_run_setup_state.py::TestProviderTrustChain::test_a_cleared_key_is_a_keyless_check_that_lifts_the_key_mark",
    "Tests/Wizards/test_first_run_setup_state.py::TestProviderTrustChain::test_the_provider_notice_and_line_both_offer_to_continue",
    "Tests/Wizards/test_first_run_setup_state.py::TestProviderTrustChain::test_a_subscription_table_spelled_otherwise_reads_as_readiness_reads_it",
    "Tests/Wizards/test_first_run_setup_state.py::TestProviderTrustChain::test_provider_tells_a_listing_that_cannot_list_as_model_does",
    # Review round 5: a 403 or a 404 to Test reads as the listing reads it.
    "Tests/Wizards/test_first_run_setup_state.py::TestProviderTrustChain::test_a_test_that_could_not_list_reads_as_the_listing_reads_it",
    # Owner decision D2 (2026-10-17): Model's keyless-403 hint, 3 of 3 serial.
    "Tests/Wizards/test_first_run_setup_state.py::TestProviderTrustChain::test_the_model_step_hints_at_a_key_only_after_a_keyless_403",
    # TASK-33001.8 (Phase 9): chat_defaults.model belongs only to its own
    # provider -- the resolver (whole file, pure, under a second) and the real
    # Ctrl+T session-default path fed by both surviving mirror producers.
    "Tests/Provider/test_provider_model_resolution.py",
    "Tests/Chat/test_console_session_settings.py::test_ctrl_t_on_a_control_provider_without_a_model_never_borrows_another",
    # TASK-33005.11: the one readiness vocabulary, a key check verifying only
    # the models it listed included (whole file, pure, about a second).
    "Tests/Chat/test_readiness_words.py",
    # TASK-33005.7: the empty transcript's ready line leads with the readiness
    # word (one pure node, under a second).
    "Tests/Chat/test_console_onboarding_state.py::test_the_ready_line_leads_with_the_readiness_word_every_surface_shows",
)

# PR2921: source-bound additions leave the original fast/admission argv intact.
DREAMS_BOUNDARY_TARGETS = (
    "Tests/Dreams/test_dispatch_currentness.py",
    "Tests/Dreams/test_cycle_service.py::test_mixed_profile_cycle_keeps_topic_and_goal_stories_without_changing_budgets",
    "Tests/Dreams/test_cycle_service.py::test_no_goal_mark_or_list_marker_ever_reaches_the_search",
    "Tests/Dreams/test_cycle_service.py::test_cycle_and_check_cannot_both_spend_the_last_search",
    "Tests/Guardian/test_live_discovery_consent.py",
    "Tests/Guardian/test_captured_dispatch_ownership.py",
    "Tests/Guardian/test_daily_native_lifecycle.py",
    "Tests/Guardian/test_crisis_provenance.py",
    "Tests/Guardian/test_crisis_resources.py",
    "Tests/Backup_Recovery/test_guardian_owner.py",
    "Tests/Backup_Recovery/test_dreams_owner.py",
    "Tests/Backup_Recovery/test_guardian_activation.py",
    "Tests/Backup_Recovery/test_dreams_activation.py",
    "Tests/Backup_Recovery/test_restored_source_first_fetch.py",
    "Tests/Backup_Recovery/test_collections_provenance_owner.py",
    "Tests/Library/test_library_artifacts_dreams.py",
    "Tests/Watchlists/test_runtime_name_literals.py",
    "Tests/Notifications/test_notification_toast_markup.py",
)
#: TASK-34929: the Dreams tracking tests, gated whole in their own job.
DREAMS_TRACKING_TARGET = "Tests/Dreams/test_track_service.py"
DREAMS_ADMISSION_TARGETS = (
    "Tests/Dreams/test_boot_dispatch_admission.py",
    "Tests/Library/test_capture_url_provenance.py::test_actual_capture_first_and_retry_apply_egress_at_every_hop",
    "Tests/Subscriptions/test_local_watchlists_service.py",
    "Tests/Watchlists/test_watchlists_bulk_source_authoring.py",
    "Tests/UI/test_settings_dreams_guardian_native_lifecycle.py",
)

#: The admission-sensitive invocation, exactly, in order (review of #3106,
#: W1). Only the sandboxed step had an exact pin, so dropping a file here or
#: shortening its ``--timeout=300`` passed every contract. These are
#: ``bootstrap_profile`` suites (TASK-32873); they run in their own job,
#: ``pr-fast-lane-admission`` (TASK-34916).
ADMISSION_TARGETS = (
    "Tests/Chat/test_console_run_log_connection_ownership.py",
    "Tests/Chat/test_console_chat_create_connection_ownership.py",
    "Tests/DB/test_chachanotes_connection_quiescence.py",
    "Tests/DB/test_operation_owned_connection_retirement.py",
    "Tests/UI/test_console_runtime_ownership.py",
    "Tests/UI/test_console_partial_resize.py",
    "Tests/Chat/test_console_viewless_hooks.py",
    "Tests/Agents/test_install_skill_runtime_tool.py",
    "Tests/Chat/test_console_chat_create_integration.py",
    "Tests/Notes/test_notes_sync_tail_edit.py",
    "Tests/Notes/test_notes_sync_delete_restore_signal.py",
    "Tests/Notes/test_notes_sync_crlf_single_line.py",
    "Tests/Notes/test_notes_sync_source_moved_settle.py",
    "Tests/Notes/test_notes_sync_watcher_after_heal.py",
    "Tests/Notes/test_notes_sync_version_only_move.py",
    "Tests/Notes/test_notes_sync_resave_window.py",
    "Tests/Notes/test_notes_sync_attention_fence.py",
    "Tests/Agents/test_first_request_window_honest_plan.py",
    "Tests/Chat/test_console_capacity_refusal_copy.py",
    # Roleplay frame B1 (TASK-33910.2): the mounted Roleplay frame suites.
    "Tests/UI/test_roleplay_frame_harness.py",
    "Tests/UI/test_roleplay_header.py",
    "Tests/UI/test_roleplay_hostile_names.py",
    "Tests/UI/test_roleplay_journeys.py",
    "Tests/UI/test_app_factory_cleanup.py",
    # TASK-33008 (Phase 8 review): the footer drops a key the screen lists as
    # unavailable. The footer reads guarded config while it composes, so the
    # node keeps the bootstrap profile and runs here.
    "Tests/UI/test_app_footer_shortcut_context.py::test_a_global_key_the_screen_lists_as_unavailable_is_left_out",
    # TASK-33008 review round 2: the wizard's Phase 8 nodes (moved here from
    # the sandboxed step: the wizard file's pytestmark is bootstrap_profile,
    # TASK-32873), the refused-key line, and the card-fit and toast nodes,
    # which read guarded config through the app import or _build_test_app.
    "Tests/Wizards/test_first_run_setup_wizard.py::test_model_status_is_plain_text_that_space_and_enter_cannot_pick",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_model_render_in_flight_at_quit_mounts_nothing",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_401_to_a_request_that_sent_no_key_asks_for_a_key",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_401_to_a_test_that_sent_no_key_asks_for_a_key",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_401_to_a_test_reads_as_a_key_problem_after_next",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_claude_subscription_is_never_told_to_add_an_api_key",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_check_that_asks_for_a_key_brings_authentication_into_view",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_listing_that_sent_no_key_and_got_through_makes_the_key_optional",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_the_pinned_notice_keeps_clear_of_the_frame",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_refused_key_is_told_as_checked_and_the_section_not_optional",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_listing_that_refused_a_sent_key_makes_the_key_required",
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_403_to_a_listing_of_a_sent_key_does_not_reject_it",
    # Owner decision D2 (2026-10-17): a keyless 403 hints at a key on both
    # steps, by the listing and by Test (bootstrap_profile file), 3 of 3 serial.
    "Tests/Wizards/test_first_run_setup_wizard.py::test_a_keyless_403_reads_as_unavailable_with_a_key_hint_on_both_steps",
    "Tests/UI/test_console_setup_card_fit.py::test_a_long_provider_name_never_wraps_the_readiness_word",
    "Tests/UI/test_console_setup_card_fit.py::test_the_name_budget_constant_matches_the_card_rule",
    "Tests/UI/test_console_toast_placement.py::test_a_dialog_over_the_console_draws_each_notice_once",
)

#: The lasting-sync real-stack files (a real database, a real ``.md``, the
#: production runtime). They are ``bootstrap_profile``, so they run in the
#: admission-sensitive invocation, never in the sandboxed one above. Pinned as
#: a subset: that step also holds other teams' files.
NOTES_SYNC_REAL_STACK_TARGETS = (
    "Tests/Notes/test_notes_sync_tail_edit.py",
    "Tests/Notes/test_notes_sync_delete_restore_signal.py",
    "Tests/Notes/test_notes_sync_crlf_single_line.py",
    "Tests/Notes/test_notes_sync_source_moved_settle.py",
    "Tests/Notes/test_notes_sync_watcher_after_heal.py",
    "Tests/Notes/test_notes_sync_version_only_move.py",
    "Tests/Notes/test_notes_sync_resave_window.py",
    "Tests/Notes/test_notes_sync_attention_fence.py",
)
HEAVY_JOB_KEYS = {
    "core-tests",
    "artifact-lease-spike",
    "artifact-lease-shape",
    "artifact-lease-gate",
    "ui-tests",
    "textual-minimum",
    "all-tests",
    "test-summary",
    "backup-platform-windows",
}
STANDALONE_WORKFLOWS = (
    "derived-artifacts.yml",
    "perf-guard.yml",
)


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOW_ROOT / name).read_text(encoding="utf-8"))


def _triggers(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def _named_step(job: dict, name: str) -> dict:
    return next(step for step in job["steps"] if step.get("name") == name)


def _pytest_targets(run: str) -> tuple[str, ...]:
    tokens = shlex.split(run.replace("\\\n", " "), comments=True)
    return tuple(
        token
        for token in tokens[1:]
        if token == "Tests"
        or token.startswith(("Tests/", "packages/tldw_profile_core/tests/"))
    )


def _assert_required_aggregation(workflow: dict) -> None:
    fast = workflow["jobs"]["pr-fast-lane"]
    required = workflow["jobs"]["derived-artifacts"]
    assert not fast.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in fast["steps"])
    assert required["name"] == "Derived artifacts reproduce from their sources"
    # TASK-32908: ui-fast-lane joined the aggregation. Both lanes are ordinary
    # jobs; `derived-artifacts` remains the only branch-protection context, so
    # each lane needs its own verdict step below or a red lane would leave the
    # required check green.
    # TASK-33621.27: the Console P0 regression gate is the third lane.
    # TASK-34916: the PR Fast Lane's admission-sensitive half is its own job.
    # TASK-34929: the Dreams tracking tests are the fifth lane.
    # TASK-34986: PR2921's Dreams/Guardian steps left console-p0-gate.
    assert required.get("needs") == [
        "pr-fast-lane",
        "pr-fast-lane-admission",
        "ui-fast-lane",
        "console-p0-gate",
        "dreams-boundary",
        "dreams-tracking",
    ]
    assert required["if"] == "${{ always() }}"
    assert not required.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in required["steps"])

    # TASK-33621.27 review round 4: `!cancelled() &&` drops the implicit
    # `success()`, so a red lane does not skip the next lane's verdict and
    # every red lane is reported in one pass (run 38072845848 showed the UI
    # and P0 verdicts skipped behind the PR lane's).
    for lane, step_name in (
        ("pr-fast-lane", "Require successful PR fast lane"),
        ("pr-fast-lane-admission", "Require successful PR fast lane (admission-sensitive)"),
        ("dreams-boundary", "Require successful Dreams and Guardian boundary contracts"),
        ("dreams-tracking", "Require successful Dreams tracking tests"),
    ):
        verdict = _named_step(required, step_name)
        assert not verdict.get("continue-on-error", False)
        assert verdict["if"] == (
            f"${{{{ !cancelled() && ({LANES}) && needs.{lane}.result != 'success' }}}}"
        )
        assert f"needs.{lane}.result" in verdict["run"]
        assert "exit 1" in verdict["run"]
    for job_name in ("pr-fast-lane-admission", "dreams-boundary", "dreams-tracking"):
        job = workflow["jobs"][job_name]
        assert not job.get("continue-on-error", False)
        assert all(not step.get("continue-on-error", False) for step in job["steps"])

    ui_verdict = _named_step(required, "Require successful UI fast lane")
    assert not ui_verdict.get("continue-on-error", False)
    assert ui_verdict["if"] == (
        f"${{{{ !cancelled() && ({LANES}) && needs.ui-fast-lane.result != 'success' }}}}"
    )
    assert "needs.ui-fast-lane.result" in ui_verdict["run"]
    assert "exit 1" in ui_verdict["run"]

    # The UI lane is bounded the same way the fast lane is: serial jobs,
    # minimal install, its own timeout. TASK-32908 put it in its own job
    # precisely so it cannot eat pr-fast-lane's 30-minute budget; TASK-34353
    # split it into contiguous shards so the census fits that timeout. A
    # matrix job's `needs.<job>.result` is success only when every shard is.
    p0_verdict = _named_step(required, "Require successful Console P0 regression gate")
    assert not p0_verdict.get("continue-on-error", False)
    assert p0_verdict["if"] == (
        f"${{{{ !cancelled() && ({LANES}) && needs.console-p0-gate.result != 'success' }}}}"
    )
    assert "needs.console-p0-gate.result" in p0_verdict["run"]
    assert "exit 1" in p0_verdict["run"]
    p0 = workflow["jobs"]["console-p0-gate"]
    assert not p0.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in p0["steps"])

    ui = workflow["jobs"]["ui-fast-lane"]
    assert ui["runs-on"] == "ubuntu-latest"
    assert list(ui["strategy"]["matrix"]) == ["shard"]
    assert ui["timeout-minutes"] <= 20
    assert not ui.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in ui["steps"])
    ui_commands = "\n".join(str(step.get("run", "")) for step in ui["steps"])
    assert ui_commands.count("pip install") == 1


#: Each exact-pinned PR Fast Lane step: (job, step, targets, test limit).
EXACT_STEPS = {
    "sandboxed": ("pr-fast-lane", "Run fast PR contract", FAST_LANE_TARGETS, 180),
    "admission-sensitive": (
        "pr-fast-lane-admission",
        "Run admission-sensitive suites",
        ADMISSION_TARGETS,
        300,
    ),
}


def _assert_exact_step(workflow: dict, which: str) -> None:
    job, name, targets, limit = EXACT_STEPS[which]
    run = _named_step(workflow["jobs"][job], name)["run"]
    assert tuple(shlex.split(run.replace("\\\n", " "), comments=True)) == (
        "pytest",
        *targets,
        f"--timeout={limit}",
        "--tb=short",
    )


def _assert_fast_lane_invocation(workflow: dict) -> None:
    _assert_exact_step(workflow, "sandboxed")


def _assert_admission_invocation(workflow: dict) -> None:
    _assert_exact_step(workflow, "admission-sensitive")


def test_heavy_tests_run_only_on_main_push_or_manual_dispatch() -> None:
    """Keep comprehensive test fan-out off pull-request and schedule events."""
    workflow = _workflow("test.yml")
    triggers = _triggers(workflow)

    assert set(triggers) == {"push", "workflow_dispatch"}
    assert triggers["push"]["branches"] == ["main"]
    assert set(workflow["jobs"]) == HEAVY_JOB_KEYS
    assert workflow["jobs"]["backup-platform-windows"]["if"] == (
        "github.event_name == 'workflow_dispatch' && inputs.backup_platform_only == true"
    )
    assert workflow["permissions"] == {"contents": "read"}
    assert "createComment" not in (WORKFLOW_ROOT / "test.yml").read_text()


def test_dedicated_nightly_owns_exact_schedule_and_full_tree_matrix() -> None:
    """Pin dedicated nightly ownership, cadence, matrix, and tested commit."""
    workflow = _workflow("nightly-deep.yml")
    triggers = _triggers(workflow)

    assert set(triggers) == {"schedule", "workflow_dispatch"}
    assert triggers["schedule"] == [{"cron": "30 8 * * *"}]
    assert set(workflow["jobs"]) == {"resolve-dev-sha", "nightly-deep"}

    resolver = workflow["jobs"]["resolve-dev-sha"]
    assert resolver["outputs"] == {"sha": "${{ steps.resolve.outputs.sha }}"}
    resolver_checkout = next(
        step for step in resolver["steps"] if step.get("uses") == "actions/checkout@v4"
    )
    assert resolver_checkout["with"] == {"ref": "dev"}
    resolve = _named_step(resolver, "Resolve one dev commit for every matrix leg")
    assert resolve["id"] == "resolve"
    assert "git rev-parse HEAD" in resolve["run"]
    assert "GITHUB_OUTPUT" in resolve["run"]

    nightly = workflow["jobs"]["nightly-deep"]
    assert nightly["needs"] == ["resolve-dev-sha"]
    assert nightly["strategy"]["matrix"]["include"] == [
        {"os": "ubuntu-latest", "python-version": "3.12", "io-encoding": "utf-8"},
        {"os": "ubuntu-latest", "python-version": "3.13", "io-encoding": "utf-8"},
        {"os": "macos-latest", "python-version": "3.12", "io-encoding": "utf-8"},
        {"os": "windows-latest", "python-version": "3.12", "io-encoding": "cp1252"},
    ]
    checkout = next(
        step for step in nightly["steps"] if step.get("uses") == "actions/checkout@v4"
    )
    assert checkout["with"] == {
        "ref": "${{ needs.resolve-dev-sha.outputs.sha }}",
        "fetch-depth": 0,
    }
    record = _named_step(nightly, "Record tested dev commit")
    assert "needs.resolve-dev-sha.outputs.sha" in record["run"]
    assert "GITHUB_STEP_SUMMARY" in record["run"]
    run = _named_step(
        nightly, "Run deep suite (serial, thorough, slow tiers, cache-off)"
    )
    assert "pytest ./Tests/" in run["run"]
    assert "--run-slow" in run["run"]
    assert "-n auto" not in run["run"]


@pytest.mark.parametrize(
    ("job_name", "display_name", "test_step"),
    [
        ("pr-fast-lane", "PR Fast Lane", "Run fast PR contract"),
        (
            "pr-fast-lane-admission",
            "PR Fast Lane (admission-sensitive)",
            "Run admission-sensitive suites",
        ),
    ],
)
def test_fast_lane_is_one_serial_minimal_python_312_job(
    job_name: str, display_name: str, test_step: str
) -> None:
    """Keep each half of the fast lane serial, bounded, and minimally provisioned.

    Args:
        job_name: The workflow job key.
        display_name: Its check name.
        test_step: Its one pytest step.
    """
    fast = _workflow("derived-artifacts.yml")["jobs"][job_name]

    assert fast["name"] == display_name
    assert fast["if"] == LANES
    assert fast["runs-on"] == "ubuntu-latest"
    # Roleplay frame B1 (TASK-33910.2): owner decision 2026-10-09, "Raise
    # timeout to 35 (Rec.)" -- 30 -> 35 to fit B1's mounted Roleplay suites in
    # the admission-sensitive step (worst case measured 28m20s). TASK-34916
    # split the two pytest invocations into two jobs instead of raising it
    # again: over 90 PR runs the halves replay at max 20.8 and 20.1 min.
    assert fast["timeout-minutes"] == 35
    assert "strategy" not in fast
    # checkout, setup-python, install, one pytest step. TASK-32873 keeps the
    # admission-sensitive suites in their own pytest invocation (their
    # keep_bootstrap_profile enrollment poisons sandboxed suites sharing a
    # process); TASK-34916 moved that invocation to its own job.
    assert len(fast["steps"]) == 4
    assert fast["steps"][-1]["name"] == test_step
    assert "if" not in fast["steps"][-1]

    setup = next(
        step for step in fast["steps"] if step.get("uses") == "actions/setup-python@v5"
    )
    assert setup["with"]["python-version"] == "3.12"

    install = _named_step(fast, "Install fast-lane dependencies")["run"]
    assert shlex.split(install) == [
        "python",
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "pytest",
        "pytest-asyncio",
        "pytest-timeout",
        "packaging",
    ]
    assert "requirements-test.txt" not in install
    assert ".[" not in install
    all_commands = "\n".join(str(step.get("run", "")) for step in fast["steps"])
    assert all_commands.count("pip install") == 1


def test_console_p0_gate_is_one_serial_minimal_python_312_job() -> None:
    """TASK-33621.27: the P0 job is bounded and provisioned like the fast lane.

    It left pr-fast-lane because that job reached 33m41s of its 35-min cap
    with the P0 block in it (run 38064483856). Its own timeout is sized from
    that run's figures in the workflow comment; a much larger one would let
    it hide a hang behind a long wait, a smaller one would fail on a slow
    runner.
    """
    p0 = _workflow("derived-artifacts.yml")["jobs"]["console-p0-gate"]

    assert p0["name"] == "Console P0 regression gate"
    assert p0["if"] == LANES
    assert p0["runs-on"] == "ubuntu-latest"
    assert p0["timeout-minutes"] == 20
    # checkout, setup-python, install, sandboxed, admission-sensitive.
    # TASK-34986 moved PR2921's two Dreams/Guardian steps to dreams-boundary
    # (with them this job measured max 18.7 min of 20 over 76 PR runs).
    assert len(p0["steps"]) == 5
    assert [step.get("name") for step in p0["steps"][-2:]] == [
        "Run the sandboxed Console P0 regression tests",
        "Run the admission-sensitive Console P0 regression tests",
    ]
    assert "strategy" not in p0
    setup = next(
        step for step in p0["steps"] if step.get("uses") == "actions/setup-python@v5"
    )
    assert setup["with"]["python-version"] == "3.12"
    install = _named_step(p0, "Install fast-lane dependencies")["run"]
    assert shlex.split(install) == [
        "python",
        "-m",
        "pip",
        "install",
        "-e",
        ".",
        "pytest",
        "pytest-asyncio",
        "pytest-timeout",
        "packaging",
    ]
    commands = "\n".join(str(step.get("run", "")) for step in p0["steps"])
    assert commands.count("pip install") == 1
    assert "-n auto" not in commands and "--dist" not in commands
    sandboxed = _named_step(p0, "Run the sandboxed Console P0 regression tests")
    admission = _named_step(
        p0, "Run the admission-sensitive Console P0 regression tests"
    )
    # One red step must not hide the other; a cancelled run stops (round 4).
    assert admission["if"] == "${{ !cancelled() }}"
    for step, timeout in ((sandboxed, "--timeout=180"), (admission, "--timeout=300")):
        tokens = shlex.split(step["run"].replace("\\\n", " "), comments=True)
        assert tokens[0] == "pytest"
        assert tokens[-2:] == [timeout, "--tb=short"]
        assert _pytest_targets(step["run"]), step["name"]
    # P0 targets live here, not in pr-fast-lane (whose budget they broke).
    fast_targets = set(
        _pytest_targets(
            _named_step(
                _workflow("derived-artifacts.yml")["jobs"]["pr-fast-lane-admission"],
                "Run admission-sensitive suites",
            )["run"]
        )
    )
    assert not fast_targets & set(_pytest_targets(admission["run"]))


def test_dreams_tracking_lane_is_one_serial_minimal_python_312_job() -> None:
    """TASK-34929: the Dreams tracking tests run, whole, on every pull request.

    Its own job, so no existing lane moves toward its cap (6m50s on its first
    run, 38588725796); not path-filtered, so a change outside Dreams (egress,
    Scheduling, Notifications) that breaks the file is caught too.
    """
    workflow = _workflow("derived-artifacts.yml")
    job = workflow["jobs"]["dreams-tracking"]

    assert job["name"] == "Dreams tracking tests"
    assert job["if"] == LANES
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 20
    assert "strategy" not in job
    assert len(job["steps"]) == 4
    setup = next(
        step for step in job["steps"] if step.get("uses") == "actions/setup-python@v5"
    )
    assert setup["with"]["python-version"] == "3.12"
    install = _named_step(job, "Install fast-lane dependencies")["run"]
    assert shlex.split(install) == shlex.split(
        _named_step(workflow["jobs"]["pr-fast-lane"], "Install fast-lane dependencies")["run"]
    )
    step = _named_step(job, "Run the Dreams tracking tests")
    assert "if" not in step
    assert tuple(shlex.split(step["run"].replace("\\\n", " "), comments=True)) == (
        "pytest",
        DREAMS_TRACKING_TARGET,
        "--timeout=180",
        "--tb=short",
    )
    # It runs once: no other PR-lane step lists the file or one of its tests.
    others = [
        f"{key}: {candidate['name']}"
        for key, other in workflow["jobs"].items()
        if key != "dreams-tracking"
        for candidate in other.get("steps", ())
        if any(
            target.split("::", 1)[0] == DREAMS_TRACKING_TARGET
            for target in _pytest_targets(str(candidate.get("run", "")))
        )
    ]
    assert others == []


def test_fast_lane_target_set_is_exact_and_non_overlapping() -> None:
    """Require the approved exact pytest targets without nested selections."""
    workflow = _workflow("derived-artifacts.yml")
    fast = workflow["jobs"]["pr-fast-lane"]
    run = _named_step(fast, "Run fast PR contract")["run"]
    targets = _pytest_targets(run)

    _assert_fast_lane_invocation(workflow)
    assert targets == FAST_LANE_TARGETS
    for index, target in enumerate(targets):
        target_path = Path(target)
        for other in targets[index + 1 :]:
            other_path = Path(other)
            assert target_path not in other_path.parents
            assert other_path not in target_path.parents


def test_admission_target_set_is_exact() -> None:
    """Review of #3106, W1: the admission-sensitive step is pinned like the
    sandboxed one -- every target, their order, and the 300 s test limit."""
    _assert_admission_invocation(_workflow("derived-artifacts.yml"))
    assert len(set(ADMISSION_TARGETS)) == len(ADMISSION_TARGETS)
    assert set(NOTES_SYNC_REAL_STACK_TARGETS) <= set(ADMISSION_TARGETS)


def test_admission_sensitive_step_gates_the_notes_sync_real_stack_files() -> None:
    """Keep the "no silent winner" and no-hold pins on every pull request.

    TASK-34000 wave 1a final review (I2): on a pull request only these lists
    and the UI census run; the rest of ``Tests/Notes`` runs on pushes to
    ``main``. A later PR could otherwise break the released pass, the Recovery
    negative controls or the re-save wait and merge green.
    """
    workflow = _workflow("derived-artifacts.yml")
    admission = workflow["jobs"]["pr-fast-lane-admission"]
    targets = _pytest_targets(
        _named_step(admission, "Run admission-sensitive suites")["run"]
    )

    missing = [name for name in NOTES_SYNC_REAL_STACK_TARGETS if name not in targets]
    assert not missing, f"not gated on pull requests: {missing}"
    assert len(set(targets)) == len(targets)
    for target in targets:
        # TASK-33621.27: a target may be one test's node id (`file::test`).
        path = PROJECT_ROOT / target.split("::", 1)[0]
        assert path.exists(), f"gated target is gone: {target}"
    # Their enrollment poisons sandboxed suites sharing a process (TASK-32873).
    assert not set(targets) & set(FAST_LANE_TARGETS)


def test_required_context_fails_closed_and_keeps_artifact_checks_install_free() -> None:
    """Fail the stable gate closed while preserving install-free diagnostics."""
    workflow = _workflow("derived-artifacts.yml")
    _assert_required_aggregation(workflow)

    required = workflow["jobs"]["derived-artifacts"]
    # Everything after the LAST lane verdict is a derived-artifact checker.
    # TASK-32908 added a second verdict step (the UI lane); anchoring on the
    # first one would have classified it as a checker and demanded
    # `!cancelled()` on a step that must stay conditional on its lane.
    verdict_index = max(
        index
        for index, step in enumerate(required["steps"])
        if str(step.get("name", "")).startswith("Require successful ")
    )
    checker_steps = required["steps"][verdict_index + 1 :]
    assert checker_steps
    assert all(step.get("if") == "${{ !cancelled() }}" for step in checker_steps)
    assert "pip install" not in "\n".join(str(step) for step in required["steps"])


def test_required_aggregation_contract_rejects_missing_prerequisite() -> None:
    """Reject a required gate that no longer depends on the fast lane."""
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    mutated["jobs"]["derived-artifacts"].pop("needs", None)

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


def test_required_aggregation_contract_rejects_partial_failure_check() -> None:
    """Reject aggregation that handles failure but accepts other bad results."""
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    verdict = _named_step(
        mutated["jobs"]["derived-artifacts"], "Require successful PR fast lane"
    )
    verdict["if"] = verdict["if"].replace("!= 'success'", "== 'failure'")

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


@pytest.mark.parametrize(
    ("job_name", "step_name"),
    [
        ("pr-fast-lane", None),
        ("pr-fast-lane-admission", None),
        ("pr-fast-lane-admission", "Run admission-sensitive suites"),
        ("derived-artifacts", "Require successful PR fast lane (admission-sensitive)"),
        ("dreams-tracking", None),
        ("dreams-tracking", "Run the Dreams tracking tests"),
        ("derived-artifacts", "Require successful Dreams tracking tests"),
        ("console-p0-gate", None),
        ("console-p0-gate", "Run the admission-sensitive Console P0 regression tests"),
        ("dreams-boundary", None),
        ("dreams-boundary", "Run Dreams admission-sensitive contracts"),
        ("derived-artifacts", "Require successful Dreams and Guardian boundary contracts"),
        ("derived-artifacts", None),
        ("derived-artifacts", "Require successful PR fast lane"),
        ("derived-artifacts", "Require successful Console P0 regression gate"),
        ("derived-artifacts", "Generated stylesheets reproduce from their sources"),
    ],
)
def test_required_aggregation_contract_rejects_continue_on_error(
    job_name: str, step_name: str | None
) -> None:
    """Reject error-tolerant jobs or steps in the required gate.

    Args:
        job_name: Workflow job to mutate.
        step_name: Optional named step within the job to mutate.
    """
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    job = mutated["jobs"][job_name]
    target = job if step_name is None else _named_step(job, step_name)
    target["continue-on-error"] = True

    with pytest.raises(AssertionError):
        _assert_required_aggregation(mutated)


@pytest.mark.parametrize(
    "flag",
    [
        "--collect-only",
        "-k smoke",
        "--ignore=Tests/CI",
        "--deselect=Tests/test_smoke.py",
    ],
)
@pytest.mark.parametrize("which", list(EXACT_STEPS))
def test_fast_lane_contract_rejects_selection_suppressing_flags(flag: str, which: str) -> None:
    """Reject pytest flags that can turn the exact lane into a subset.

    Args:
        flag: Selection-suppressing argument appended to the pytest command.
        which: The exact-pinned step to mutate.
    """
    job, name, _, _ = EXACT_STEPS[which]
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    run_step = _named_step(mutated["jobs"][job], name)
    run_step["run"] += f" {flag}"

    with pytest.raises(AssertionError):
        _assert_exact_step(mutated, which)


@pytest.mark.parametrize("which", list(EXACT_STEPS))
def test_fast_lane_contract_rejects_a_shorter_test_limit(which: str) -> None:
    """Review of #3106, W1: `--timeout=300` -> `--timeout=30` must fail too.

    Args:
        which: The exact-pinned step to mutate.
    """
    job, name, _, limit = EXACT_STEPS[which]
    mutated = copy.deepcopy(_workflow("derived-artifacts.yml"))
    run_step = _named_step(mutated["jobs"][job], name)
    run_step["run"] = run_step["run"].replace(f"--timeout={limit}", f"--timeout={limit // 10}")

    with pytest.raises(AssertionError):
        _assert_exact_step(mutated, which)


def test_focused_guards_keep_dev_pr_and_dev_main_push_coverage() -> None:
    """Retain focused guards on dev PRs and pushes to dev and main."""
    for workflow_name in STANDALONE_WORKFLOWS:
        triggers = _triggers(_workflow(workflow_name))

        assert triggers["pull_request"]["branches"] == ["dev"]
        assert triggers["pull_request"]["types"] == PULL_REQUEST_TYPES
        assert triggers["push"]["branches"] == ["dev", "main"]


def test_pull_request_workflows_are_never_cancelled_in_progress() -> None:
    """Prevent base-branch churn from cancelling pull-request gate runs."""
    for workflow_name in STANDALONE_WORKFLOWS:
        concurrency = _workflow(workflow_name)["concurrency"]
        assert concurrency["cancel-in-progress"] == PUSH_ONLY_CANCELLATION

    heavy = _workflow("test.yml")
    assert "pull_request" not in _triggers(heavy)
    assert heavy["concurrency"]["cancel-in-progress"] == PUSH_ONLY_CANCELLATION


def test_bundle_and_backlog_checks_run_on_push_events() -> None:
    """The bundle and backlog-id checks still run on dev/main pushes.

    With css-bundle-guard/backlog-guard deleted, the required workflow is the
    only place these checks run -- so they must run on dev/main pushes too, not
    only inside the pull-request-only fast lanes.
    """
    workflow = _workflow("derived-artifacts.yml")
    assert {"dev", "main"} <= set(_triggers(workflow)["push"]["branches"])
    steps = workflow["jobs"]["derived-artifacts"]["steps"]
    for script in (
        "tldw_chatbook/css/check_bundle_sync.py",
        "scripts/check_backlog_task_ids.py",
    ):
        matching = [step for step in steps if script in str(step.get("run", ""))]
        assert matching, f"{script} is not run by the required job"
        for step in matching:
            assert "pull_request" not in str(step.get("if", "")), (
                f"{script} must not be pull-request-only"
            )
    assert not (PROJECT_ROOT / ".github/workflows/css-bundle-guard.yml").exists()
    assert not (PROJECT_ROOT / ".github/workflows/backlog-guard.yml").exists()


@pytest.mark.parametrize("which", list(EXACT_STEPS))
@pytest.mark.parametrize("comment_only", [False, True])
def test_exact_targets_and_invocation_reject_missing_executable_target(comment_only, which):
    job, name, targets, _ = EXACT_STEPS[which]
    workflow = copy.deepcopy(_workflow("derived-artifacts.yml"))
    step = _named_step(workflow["jobs"][job], name)
    # The last target, and (review of #3106, W1) one from the middle.
    for target in (targets[-1], targets[len(targets) // 2]):
        step["run"] = step["run"].replace(target + " \\\n", "")
        assert target not in step["run"]
        if comment_only:
            step["run"] += "\n# " + target + "\n"
        assert target not in _pytest_targets(step["run"])
    with pytest.raises(AssertionError):
        _assert_exact_step(workflow, which)


def test_target_parser_preserves_quoted_hash_and_continuation():
    target = "Tests/example.py::test_value[hash#literal]"
    run = (
        "pytest \\\n  '"
        + target
        + "' \\\n  --timeout=180 --tb=short\n# Tests/comment_only.py\n"
    )
    assert _pytest_targets(run) == (target,)


def test_exact_invocation_accepts_comments_without_changing_executable_argv():
    workflow = copy.deepcopy(_workflow("derived-artifacts.yml"))
    step = _named_step(workflow["jobs"]["pr-fast-lane"], "Run fast PR contract")
    step["run"] = "# Required tests follow.\n" + step["run"] + "\n# End of command.\n"
    assert _pytest_targets(step["run"]) == FAST_LANE_TARGETS
    _assert_fast_lane_invocation(workflow)


def _assert_dreams_required_steps(workflow: dict) -> None:
    # TASK-34986: the two PR2921 steps run in their own job, moved verbatim
    # from console-p0-gate (with them it measured max 18.7 min of 20).
    job = workflow["jobs"]["dreams-boundary"]
    assert job["name"] == "Dreams and Guardian boundary contracts"
    assert job["if"] == LANES
    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 20
    assert "strategy" not in job
    names = (
        "Run Dreams and Guardian boundary contracts",
        "Run Dreams admission-sensitive contracts",
    )
    # checkout, setup-python, install, then the two steps in this order.
    assert len(job["steps"]) == 5
    assert tuple(step.get("name") for step in job["steps"][-2:]) == names
    setup = next(
        step for step in job["steps"] if step.get("uses") == "actions/setup-python@v5"
    )
    assert setup["with"]["python-version"] == "3.12"
    assert shlex.split(_named_step(job, "Install fast-lane dependencies")["run"]) == shlex.split(
        _named_step(workflow["jobs"]["pr-fast-lane"], "Install fast-lane dependencies")["run"]
    )
    for name in names:
        owners = [key for key, candidate in workflow["jobs"].items()
                  for step in candidate.get("steps", ()) if step.get("name") == name]
        assert owners == ["dreams-boundary"]
    assert not job.get("continue-on-error", False)
    for name, targets, limit in (
        ("Run Dreams and Guardian boundary contracts", DREAMS_BOUNDARY_TARGETS, 180),
        ("Run Dreams admission-sensitive contracts", DREAMS_ADMISSION_TARGETS, 300),
    ):
        step = _named_step(job, name)
        assert step["if"] == "${{ !cancelled() }}"
        assert not step.get("continue-on-error", False)
        tokens = tuple(shlex.split(step["run"].replace("\\\n", " "), comments=True))
        assert tokens == ("pytest", *targets, f"--timeout={limit}", "--tb=short")
        assert len(set(targets)) == len(targets)
        for target in targets:
            assert (PROJECT_ROOT / target.split("::", 1)[0]).is_file()
    assert not set(DREAMS_BOUNDARY_TARGETS) & set(DREAMS_ADMISSION_TARGETS)


def test_dreams_critical_boundaries_are_executable_required_lane_members() -> None:
    _assert_dreams_required_steps(_workflow("derived-artifacts.yml"))


@pytest.mark.parametrize("name,targets", [
    ("Run Dreams and Guardian boundary contracts", DREAMS_BOUNDARY_TARGETS),
    ("Run Dreams admission-sensitive contracts", DREAMS_ADMISSION_TARGETS),
])
def test_dreams_membership_cannot_be_satisfied_by_comments(name, targets) -> None:
    workflow = _workflow("derived-artifacts.yml")
    step = _named_step(workflow["jobs"]["dreams-boundary"], name)
    step["run"] = step["run"].replace(targets[0] + " \\\n", "")
    step["run"] += "\n# " + targets[0]
    with pytest.raises(AssertionError):
        _assert_dreams_required_steps(workflow)


@pytest.mark.parametrize("name", [
    "Run Dreams and Guardian boundary contracts",
    "Run Dreams admission-sensitive contracts",
])
@pytest.mark.parametrize("mutation", [
    "missing", "order", "duplicate", "wrong-job", "moved-back", "suppression",
    "conditional", "continue-on-error", "job-budget", "test-budget",
])
def test_dreams_required_placement_rejects_drift(name, mutation):
    workflow = _workflow("derived-artifacts.yml")
    job = workflow["jobs"]["dreams-boundary"]
    step = _named_step(job, name)
    if mutation == "missing":
        job["steps"].remove(step)
    elif mutation == "order":
        job["steps"][-2:] = reversed(job["steps"][-2:])
    elif mutation == "duplicate":
        workflow["jobs"]["pr-fast-lane"]["steps"].append(copy.deepcopy(step))
    elif mutation == "wrong-job":
        job["steps"].remove(step)
        workflow["jobs"]["pr-fast-lane"]["steps"].append(step)
    elif mutation == "moved-back":  # TASK-34986: back into console-p0-gate
        job["steps"].remove(step)
        workflow["jobs"]["console-p0-gate"]["steps"].append(step)
    elif mutation == "suppression":
        step["run"] += " --collect-only"
    elif mutation == "conditional":
        step["if"] = "success()"
    elif mutation == "continue-on-error":
        step["continue-on-error"] = True
    elif mutation == "job-budget":
        job["timeout-minutes"] += 1
    else:
        step["run"] = step["run"].replace("--timeout=", "--timeout=1")
    with pytest.raises(AssertionError):
        _assert_dreams_required_steps(workflow)
