"""TASK-34807: bounded response-rule contracts in the required existing P0 job."""

import shlex
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/derived-artifacts.yml"
SELECTION = {
    "sandboxed": [
        "Tests/Chat/test_response_rules_preparation_custody.py",
        "Tests/DB/test_response_rules_migration.py::test_historical_migrates_using_installed_sql_without_losing_history[80]",
        "Tests/DB/test_response_rules_migration.py::test_rule_lookup_indexes_without_statistics",
        "Tests/Backup_Recovery/test_response_rules_restore.py::test_every_accepted_v80_catalog_migrates_to_its_exact_current_variant",
        "Tests/Backup_Recovery/test_response_rules_restore.py::test_restricted_import_preparation_disables_only_qualified_selected_payload",
        "Tests/Chat/test_console_voice_pair_persistence.py::test_voice_promotion_response_rule_residue_refuses_without_mutation",
        "Tests/DB/test_private_sqlite_inventory.py::test_response_rule_import_dispatch_is_exact_and_validated",
    ],
    "admission-sensitive": [
        "Tests/Chat/test_response_rules_trace_admission.py",
        "Tests/Chat/test_response_rules_goal_intersection.py",
        "Tests/Chat/test_response_rules_runtime.py::test_first_send_checks_stored_global_rule_without_prior_manager_use",
        "Tests/Chat/test_response_rules_runtime.py::test_cancelled_rule_db_call_keeps_real_handle_until_own_cleanup",
        "Tests/Chat/test_response_rules_runtime.py::test_real_completed_tool_work_is_retained_during_native_repair",
        "Tests/Chat/test_response_rules_external_hook.py",
        "Tests/Chat/test_response_rules_write_contention.py",
        "Tests/UI/test_response_rules_modal.py::test_saved_edit_stays_selected_and_pinned_for_next_manager_action",
        "Tests/Chat/test_deepseek_explicit_mode.py",
        "Tests/UI/test_response_rules_modal.py::test_manager_detached_subtree_callbacks_do_not_read_widgets",
        "Tests/UI/test_response_rules_modal.py::test_manager_superseded_reload_cannot_replace_current_scope_rows",
        "Tests/UI/test_response_rules_modal.py::test_manager_reload_preserves_edits_made_while_db_is_pending",
        "Tests/Chat/test_response_rules_builder.py::test_new_learning_timestamps_are_canonical_and_old_offsets_still_parse",
    ],
}
INHERITED = {
    "Run the sandboxed Console P0 regression tests": [
        "pytest",
        "Tests/Console/test_console_markdown_export.py",
        "Tests/Chat/test_console_trace_row_sources.py",
        "Tests/Chat/test_console_trace_rewritten_tail.py",
        "Tests/Architecture/test_wizard_lifecycle_guards.py",
        "Tests/DB/test_chachanotes_v74_auxiliary_failure_reason.py",
        "Tests/UI/test_console_composer_run_controls.py::test_disabled_reason_names_the_queue_state_not_send_or_setup",
        "Tests/UI/test_console_composer_run_controls.py::test_queue_state_labels_match_the_prompt_queue_presentation",
        "Tests/UI/test_console_composer_run_controls.py::test_palette_lists_stop_and_the_composer_menu_actions_class_safe",
        "Tests/UI/test_app_keep_alive_dead_screen.py::test_a_dead_content_screen_over_only_the_placeholder_takes_the_loud_exit",
        "Tests/UI/test_app_keep_alive_dead_screen.py::test_a_recovery_that_raises_is_logged_before_the_loud_exit",
        "Tests/UI/test_app_keep_alive_dead_screen.py::test_retiring_a_dead_screen_resumes_a_worker_awaiting_a_screen_above_it",
        "Tests/UI/test_app_keep_alive_dead_screen.py::test_pump_loop_ended_matches_what_textual_does_to_the_pump",
        "Tests/UI/test_app_keep_alive_dead_screen.py::test_a_retired_screen_is_torn_down_however_its_loop_ended",
        "Tests/UI/test_app_keep_alive_dead_screen.py::test_a_dead_screen_whose_unmount_raises_is_still_dropped",
        "Tests/UI/test_app_keep_alive_dead_screen.py::test_a_screen_handler_error_leaves_only_live_screens_on_the_stack",
        "Tests/UI/test_console_project_instruction_choose_folder.py::test_choose_folder_opens_picker_and_applies_the_chosen_binding[enable]",
        "Tests/UI/test_console_conversation_action_menu.py::test_unwritable_save_path_shows_an_error_and_keeps_the_app_running",
        "Tests/UI/test_console_conversation_action_menu.py::test_save_prompt_closes_without_writing_and_restores_focus[cancel]",
        "--timeout=180",
        "--tb=short",
    ],
    "Run the admission-sensitive Console P0 regression tests": [
        "pytest",
        "Tests/Agents/test_provider_tool_schema_conformance.py",
        "Tests/Chat/test_console_tool_definition_rejection.py",
        "Tests/Agents/test_local_tool_provider.py::test_todo_update_delete_rule_is_enforced_by_the_handler_not_the_schema",
        "Tests/Chat/test_console_capture_send_triggers.py",
        "Tests/Chat/test_console_trace_rewritten_history.py::test_captured_sends_after_edit_and_resend_keep_trace_capture[agent-last]",
        "Tests/Chat/test_console_trace_rewritten_history.py::test_captured_sends_after_edit_and_resend_keep_trace_capture[plain-last]",
        "Tests/Chat/test_console_trace_rewritten_history.py::test_captured_sends_after_compact_now_keep_trace_capture[warm-agent]",
        "Tests/UI/test_console_blocked_send_recovery.py::test_each_trigger_chat_sends_and_renders_the_reply[system-prompt-before-first-send]",
        "Tests/UI/test_console_blocked_send_recovery.py::test_provenance_failure_recovery_actions_work[retry]",
        "Tests/Chat/test_console_compaction_failure.py",
        "Tests/Chat/test_console_compaction_live_session.py::test_live_automatic_compaction_commits_memory_and_the_send_replies",
        "Tests/Chat/test_console_compaction_live_session.py::test_live_compact_now_succeeds_after_real_durable_sends",
        "Tests/Chat/test_console_compaction_live_session.py::test_live_failed_compaction_records_reason_and_discloses_spend",
        "Tests/Chat/test_console_compaction_live_session.py::test_live_failed_compact_now_says_nothing_changed",
        "Tests/Chat/test_console_compaction_live_session.py::test_live_failure_is_not_rebilled_until_the_policy_changes",
        "Tests/UI/test_console_row_menu_action_sweep.py::test_the_sweep_covers_every_declared_action_constant",
        "Tests/UI/test_console_row_menu_action_sweep.py::test_conversation_row_action_never_ends_the_app",
        "Tests/Wizards/test_first_run_provider_catalog.py::test_wizard_provider_list_is_the_settings_picker_set",
        "Tests/Wizards/test_first_run_provider_catalog.py::test_arrowing_through_every_listed_row_keeps_the_step_live",
        "Tests/Wizards/test_first_run_provider_catalog.py::test_a_raising_step_handler_keeps_the_step_and_the_keyboard",
        "Tests/Wizards/test_first_run_provider_catalog.py::test_next_after_a_step_error_never_exits_the_app",
        "Tests/UI/test_console_hook_review_send_freeze.py::test_the_app_pump_answers_ctrl_q_while_the_enter_review_is_open",
        "Tests/UI/test_console_hook_review_send_freeze.py::test_a_declined_send_review_settles_and_keeps_the_draft",
        "Tests/UI/test_console_hook_review_send_freeze.py::test_allow_all_resumes_the_captured_send_exactly_once",
        "Tests/Chat/test_console_trace_graph_gc.py::test_stale_mark_never_sweeps_after_concurrent_reachability_change",
        "Tests/Chat/test_console_trace_admission_gc.py::test_collection_between_admission_and_reservation_preserves_exact_revision",
        "Tests/Chat/test_console_trace_admission_gc.py::test_canonical_ancestry_cannot_retain_payload_through_another_owners_policy",
        "Tests/Chat/test_console_retry_json_wire.py::test_json_schema_retry_fence_precedes_actual_http[unchanged-groq]",
        "Tests/Chat/test_console_retry_json_wire.py::test_json_schema_retry_fence_precedes_actual_http[changed-type-groq]",
        "Tests/Chat/test_console_retry_json_identity.py::test_failed_retry_rejects_python_equal_json_changes[true-to-one-response_format-deepseek]",
        "Tests/Chat/test_console_retry_json_identity.py::test_failed_retry_rejects_python_equal_json_changes[unchanged-response_format-deepseek]",
        "Tests/Chat/test_console_native_commit.py::test_owned_native_commit_drains_original_callback_and_connection_after_global_cancel",
        "Tests/Chat/test_console_shared_tool_cache.py::test_stock_cold_adopts_once_and_warm_reuses_without_payload_decode",
        "Tests/Chat/test_console_context_presentation_current.py::test_context_display_retains_exact_native_handle_through_cancel[True]",
        "Tests/UI/test_console_hook_refresh_lifetime.py::test_cancelled_visit_drains_actual_native_custody[True]",
        "Tests/UI/test_console_received_intent_feedback.py::test_actual_send_receives_intent_before_original_configuration_read",
        "Tests/UI/test_console_browser_current_native_lifetime.py::test_current_host_shutdown_drains_original_browser_callback[all_created-False]",
        "Tests/UI/test_console_preparing_current_sources.py::test_preparing_changed_source_after_original_tab_await_replays_full",
        "Tests/UI/test_console_final_poll_completion.py::test_idle_final_poll_waits_for_original_full_refresh",
        "Tests/Performance/test_collections_extraction_shutdown.py::test_collections_shutdown_retains_real_interrupted_settlement[all_created]",
        "--timeout=300",
        "--tb=short",
    ],
    "Run Dreams and Guardian boundary contracts": [
        "pytest",
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
        "--timeout=180",
        "--tb=short",
    ],
    "Run Dreams admission-sensitive contracts": [
        "pytest",
        "Tests/Dreams/test_boot_dispatch_admission.py",
        "Tests/Library/test_capture_url_provenance.py::test_actual_capture_first_and_retry_apply_egress_at_every_hop",
        "Tests/Subscriptions/test_local_watchlists_service.py",
        "Tests/Watchlists/test_watchlists_bulk_source_authoring.py",
        "Tests/UI/test_settings_dreams_guardian_native_lifecycle.py",
        "--timeout=300",
        "--tb=short",
    ],
}


#: Where each inherited pytest step runs. TASK-34986 moved the two Dreams
#: steps out of console-p0-gate, verbatim, into dreams-boundary.
STEP_JOBS = {
    "Run the sandboxed Console P0 regression tests": "console-p0-gate",
    "Run the admission-sensitive Console P0 regression tests": "console-p0-gate",
    "Run Dreams and Guardian boundary contracts": "dreams-boundary",
    "Run Dreams admission-sensitive contracts": "dreams-boundary",
}


def validate(document):
    jobs = document["jobs"]
    commands = {}
    for job_name in sorted(set(STEP_JOBS.values())):
        job = jobs[job_name]
        assert job["timeout-minutes"] == 20 and len(job["steps"]) == 5
        assert (
            job["if"]
            == "github.event_name == 'pull_request' || (github.event_name == 'workflow_dispatch' && (inputs.pr != '' || github.ref != 'refs/heads/dev'))"
        )
        assert not job.get("continue-on-error", False)
        assert job_name in jobs["derived-artifacts"]["needs"]
        for s in job["steps"]:
            if s.get("run", "").lstrip().startswith("pytest"):
                assert STEP_JOBS.get(s["name"]) == job_name, s["name"]
                commands[s["name"]] = s
    assert set(commands) == set(INHERITED)
    for name, inherited in INHERITED.items():
        step = commands[name]
        assert not step.get("continue-on-error", False)
        assert step.get("if") == (
            None if "sandboxed Console P0" in name else "${{ !cancelled() }}"
        )
        tokens = shlex.split(step["run"].replace("\\\n", " "), comments=True)
        label = (
            "sandboxed"
            if "sandboxed Console P0" in name
            else "admission-sensitive"
            if "admission-sensitive Console P0" in name
            else None
        )
        added = SELECTION[label] if label else []
        for target in added:
            assert tokens.count(target) == 1, target
        assert [token for token in tokens if token not in added] == inherited
        assert not {
            "-k",
            "-m",
            "--ignore",
            "--deselect",
            "--collect-only",
        }.intersection(tokens)


def test_rule_critical_targets_execute_without_altering_inherited_processes():
    validate(yaml.safe_load(WORKFLOW.read_text()))


@pytest.mark.parametrize(
    "label,target",
    [(label, target) for label, targets in SELECTION.items() for target in targets],
)
@pytest.mark.parametrize("damage", ["remove", "comment"])
def test_missing_or_comment_only_rule_target_is_rejected(label, target, damage):
    document = yaml.safe_load(WORKFLOW.read_text())
    step = next(
        s
        for s in document["jobs"]["console-p0-gate"]["steps"]
        if s.get("name") == "Run the " + label + " Console P0 regression tests"
    )
    lines = step["run"].splitlines()
    step["run"] = "\n".join(
        ("# " + line if damage == "comment" else "") if target in line else line
        for line in lines
    )
    with pytest.raises(AssertionError):
        validate(document)


@pytest.mark.parametrize(
    "damage",
    [
        "filter",
        "deselect",
        "ignore",
        "collect",
        "continue",
        "condition",
        "cap",
        "aggregate",
    ],
)
def test_suppressed_or_weakened_critical_process_is_rejected(damage):
    document = yaml.safe_load(WORKFLOW.read_text())
    job = document["jobs"]["console-p0-gate"]
    step = next(
        s
        for s in job["steps"]
        if s.get("name") == "Run the admission-sensitive Console P0 regression tests"
    )
    if damage in {"filter", "deselect", "ignore", "collect"}:
        step["run"] += {
            "filter": " -k never",
            "deselect": " --deselect=" + SELECTION["admission-sensitive"][0],
            "ignore": " --ignore=Tests/Chat",
            "collect": " --collect-only",
        }[damage]
    elif damage == "continue":
        step["continue-on-error"] = True
    elif damage == "condition":
        step["if"] = "false"
    elif damage == "cap":
        job["timeout-minutes"] = 25
    else:
        document["jobs"]["derived-artifacts"]["needs"].remove("console-p0-gate")
    with pytest.raises(AssertionError):
        validate(document)


UI_PHASE_SELECTION = (
    "Tests/UI/test_console_response_rules.py::test_mapped_learning_phase_and_stop_follow_live_helper",
    "Tests/UI/test_console_response_rules.py::test_rule_phase_events_project_only_the_mounted_current_session",
)


def _validate_ui_phase_membership(census_path):
    from scripts.check_ui_pr_gate_census import read_census, shard

    entries = read_census(census_path)
    # The UI lane's own shard count (TASK-34917 took it from four to six).
    total = len(yaml.safe_load(WORKFLOW.read_text())["jobs"]["ui-fast-lane"]["strategy"]["matrix"]["shard"])
    shards = [shard(entries, index, total) for index in range(total)]
    for target in UI_PHASE_SELECTION:
        assert sum(part.count(target) for part in shards) == 1


def test_native_phase_controls_are_executable_in_required_ui_census():
    _validate_ui_phase_membership(ROOT / "scripts/ui_pr_gate_census.txt")


@pytest.mark.parametrize("target", UI_PHASE_SELECTION)
@pytest.mark.parametrize("damage", ["remove", "comment"])
def test_missing_or_comment_only_native_phase_control_is_rejected(
    tmp_path, target, damage
):
    original = (ROOT / "scripts/ui_pr_gate_census.txt").read_text()
    damaged = original.replace(target, "# " + target if damage == "comment" else "")
    path = tmp_path / "census.txt"
    path.write_text(damaged)
    with pytest.raises(AssertionError):
        _validate_ui_phase_membership(path)
