# Console private-delegate cleanup proposal (TASK-31717 continuation)

> Proposal only: root must approve the exact scope and allocate atomic execution tasks before production edits.

**Goal:** Remove obsolete private forwarding layers without extending runtime ownership boundaries; move only two pure policy helpers into their existing consumers.

**Architecture:** Keep all public methods, Textual actions/events, state descriptors, and the historically retained review-notes STAY boundary. Screen callers use existing controllers directly. Cross-controller wiring stays named and late-bound. Never replace screen state or behavior with a generic proxy/presenter.

**Tech stack:** Python AST census, Textual 8, existing controller assembly, pytest and Ruff.

ADR required: no. ADR path: N/A. Direct implementation of existing DESIGN.md §7 and the screen-decomposition design; no new ownership boundary or policy. The two pure helpers implement existing command/settings policy in the owner that already consumes it.

## Exact eligible private delegates

Each listed method has no decorator and no statement other than its controller call after its docstring. `return`/`await` shape and passed arguments were inspected. All 22 statement-only delegates target functions annotated `None` with no value-returning branch, so removing the wrapper does not expose a formerly discarded return value.

The retained exclusions are `_ensure_console_agent_bridge` (documented screen-level injection seam) and `_console_active_session_is_ephemeral` (the transcript probes the screen by this exact name). All `action_*`, event entry points, public prompt/provider interfaces and decorated methods remain.

### `_settings_durability` — 2 methods, 9 body lines

- `_commit_console_settings_submission_live` (7 lines)
- `_dispatch_console_settings_submission` (2 lines)

### `_settings_navigation` — 2 methods, 22 body lines

- `_open_console_settings` (20 lines)
- `_consume_pending_conversation_settings_return` (2 lines)

### `_context_cost` — 9 methods, 67 body lines

- `_console_inspector_next_send_factories` (11 lines)
- `_console_next_send_token_estimate` (6 lines)
- `_active_console_settings_context_estimate` (4 lines)
- `_console_settings_context_estimate_for_session` (9 lines)
- `_active_console_context_control_state` (10 lines)
- `_console_context_control_state_for_session` (14 lines)
- `_build_console_settings_summary_state` (2 lines)
- `_build_console_cost_state` (2 lines)
- `_build_console_inspector_cost_data` (9 lines)

### `_row_actions` — 2 methods, 4 body lines

- `_console_conversation_state` (2 lines)
- `_save_console_conversation_markdown` (2 lines)

### `_provider_selection` — 6 methods, 29 body lines

- `_provider_readiness_app_config` (2 lines)
- `_providers_models_for_console_settings` (9 lines)
- `_build_console_provider_selection` (4 lines)
- `_build_console_provider_selection_for_settings` (6 lines)
- `_active_console_provider_model_display` (4 lines)
- `_active_console_settings_readiness` (4 lines)

### `_message` — 12 methods, 87 body lines

- `_recent_console_image_messages` (5 lines)
- `_console_messages_from_conversation_tree` (9 lines)
- `_rehydrate_console_message_image` (9 lines)
- `_rehydrate_console_message_attachments` (9 lines)
- `_rehydrate_console_message_generation_metadata` (13 lines)
- `_console_citation_message_body` (4 lines)
- `_append_native_console_system_message` (9 lines)
- `_console_save_as_destinations` (4 lines)
- `_save_console_message_image` (4 lines)
- `_save_console_message_as_note` (4 lines)
- `_open_console_message_edit_modal` (8 lines)
- `_select_console_message_variant` (9 lines)

### `_prompts` — 6 methods, 20 body lines

- `_ensure_console_prompt_history` (3 lines)
- `_open_console_prompts_modal` (5 lines)
- `_console_command_insert_prompt` (3 lines)
- `_consume_pending_console_prompt_insert` (3 lines)
- `_console_command_apply_system` (3 lines)
- `_open_console_system_prompt_editor` (3 lines)

### `_dictation` — 4 methods, 40 body lines

- `_sync_console_dictation_availability` (11 lines)
- `_request_console_dictation_stop` (10 lines)
- `_request_console_dictation_cancel` (9 lines)
- `_request_console_dictation_start` (10 lines)

### `_hands_free` — 1 methods, 8 body lines

- `_enter_console_hands_free_loop` (8 lines)

### `_retrieval` — 1 methods, 3 body lines

- `_open_console_library_search` (3 lines)

### `_session` — 1 methods, 11 body lines

- `_current_console_conversation_id` (11 lines)

### `_submission` — 9 methods, 24 body lines

- `_submit_console_native_draft` (4 lines)
- `_on_console_submission_accepted` (2 lines)
- `_console_pending_image_attachment` (2 lines)
- `_console_attachment_blocked_reason` (2 lines)
- `_console_send_blocked_reason` (2 lines)
- `_send_console_message_from_visible_action` (2 lines)
- `_dispatch_console_draft_send` (4 lines)
- `_restore_console_send_stash` (2 lines)
- `_recover_stuck_console_send_stash` (4 lines)

### `_commands` — 5 methods, 18 body lines

- `_dispatch_console_command` (2 lines)
- `_insert_prompt_text_into_composer` (2 lines)
- `_console_command_rewind` (2 lines)
- `_apply_console_rewind_choice` (10 lines)
- `_clear_console_composer_draft` (2 lines)

### `_image` — 1 methods, 3 body lines

- `_console_command_generate_image` (3 lines)

### `_video` — 2 methods, 6 body lines

- `_console_command_generate_video` (3 lines)
- `_console_command_stream_video` (3 lines)

### `_skill` — 1 methods, 2 body lines

- `_console_command_skills` (2 lines)

## Two pure policy moves

- `_console_rewind_summary_disabled_reason` (27 body lines plus its decorator) becomes a static method on `ConsoleCommandsController`. Its only runtime consumer is that same controller; remove the now-redundant constructor callback and matching wiring keyword. Preserve run/tip/durable-unit checks exactly. Move `complete_durable_units` import only if the screen has no remaining use; use type-only controller annotation imports where appropriate.
- `_console_settings_initial_draft` (24 body lines plus its decorator) becomes a static method on `ConsoleSettingsNavigationController`. Its consumers are the settings owner and one screen restore flow at the old line3653. Remove the redundant constructor callback/wiring keyword and retarget that screen call. The three `ConsoleSettingsDraftState`/field-draft/provenance imports are used only here and move with it. Three direct test references migrate without changing draft assertions.

## Binding phases and patch seams

- Screen `__init__` contains **zero** references to the 64 eligible delegates: no new construction-order owner reads should be introduced there.
- `console_view_hooks` contains two relevant references. `_on_console_submission_accepted` must remain an explicitly late-bound synchronous lambda so building hooks before `_submission` exists remains valid, like TASK31773's appliers. `_ensure_console_prompt_history` is already reached only when the local `prompts` owner is available; retain that guard and existing invocation phase.
- All callback dependencies in `wiring.py` resolve their controller at invocation time, not construction. Retarget lambda bodies, not callback acquisition timing. Audit each direct callable passed to Textual scheduling separately: retain a lambda where the original facade delayed an owner lookup.
- The one non-comment external production screen-name probe is the retained ephemeral-session method. Other same-named references outside Console modules are comments or independent methods on Settings/Home/Library; do not rewrite them.
- Test patching is receiver-specific: migrate a patch of an eligible screen facade to its actual controller method and update bare-shell wiring explicitly. Do not change similarly named fake dependency-protocol keys, another screen's method, or an already-correct controller patch. Preserve asserted call arguments, await behavior, cancellation and error injection phases.
- Keep the agent-bridge screen injection seam and its tests unchanged. New late-binding regressions cover hook construction without the owner, replacement-owner invocation, and a representative post-construction patched callback.

## Measured size estimate (read-only simulation)

Baseline after TASK31908: **17266 lines /571 direct methods**, ceiling **16873/559** (393/12 deficit). The64 eligible bodies total353 lines (not counting separators); deleting them and retargeting screen references in memory yields16913 raw lines while retaining surrounding blank lines. Formatting both baseline and candidate for comparison gives17265→16894; this is not a claim that blank-line pruning closes the deficit. The two pure policy bodies add51 lines plus decorators, before their screen imports/callbacks are retired. Final actual counts, including longer direct-call wrapping and deferred hooks, must pass the unchanged caps; stop and report if this exact scope does not suffice.

## Implementation sequence after approval

1. Allocate independent atomic task records under root coordination; record baseline tests and owner/caller inventory before edits.
2. Move the two pure policy helpers with AST body equivalence and focused policy/rewind/settings tests.
3. Remove exact private delegates by existing-owner cohorts. Retarget production screen callers and named wiring lambdas; preserve framework/public/excluded seams and early hook binding.
4. Retarget only affected test receivers and explicitly initialize bare shells. Do not reduce expectations or widen catches.
5. Re-run normalized AST/argument fidelity, phase regressions, relevant whole files below, static checks and exact line/method census. Root reconciles shared diagnostic/Architecture evidence after source is final.
6. Obtain root diff review per cohort and commit scoped files only. No caps, allowlists, mixins, broad proxies, or unrelated STAY-boundary moves.

## Verification matrix

- **Pure policy:** complete model-apply-chips, UI/Chat session-settings, rewind restore/e2e, and command-composer files.
- **Wiring and phases:** controller-wiring, internals decomposition, runtime-ownership, screen reuse, bare-shell Architecture guard, plus direct callback replacement assertions. Source census must prove no eligible facade remains and required exclusions remain real.
- **Lifecycle and dispatch:** complete native chat flow, generation actions/video/image, composer, draft/stash, run-state, dictation/hands-free, prompt/skill, and affected branching/edit/terminal integration files.
- **Ownership safeguards:** existing controller-boundary and wave6 inventories, worker-group check, screen-size ratchets. Root owns shared inventories; no cap changes.
- **Exact reference census:** the following83 Python test/support files mention at least one eligible name. This is a conservative verification/caller list, not permission to edit every file: comments, independent owner references and already-correct protocol fakes may require no change. Whole executable test files in this list form the affected sweep, with framework helpers covered through their consumers. Record any baseline failures distinctly.

- `Tests/Architecture/test_console_realtime_controller_boundary.py`
- `Tests/Architecture/test_console_wave6_closeout_inventory.py`
- `Tests/Architecture/test_console_wave6_inventory.py`
- `Tests/Chat/test_console_attachment_riders.py`
- `Tests/Chat/test_console_chat_controller.py`
- `Tests/Chat/test_console_conversation_hydration.py`
- `Tests/Chat/test_console_generate_video.py`
- `Tests/Chat/test_console_generation_actions.py`
- `Tests/Chat/test_console_generation_card.py`
- `Tests/Chat/test_console_generation_store.py`
- `Tests/Chat/test_console_h3_image_edit.py`
- `Tests/Chat/test_console_remote_images.py`
- `Tests/Chat/test_console_run_state_per_session.py`
- `Tests/Chat/test_console_session_settings.py`
- `Tests/Chat/test_console_sibling_nav.py`
- `Tests/Chat/test_console_turn_execution_context.py`
- `Tests/Chat/test_console_user_sibling_nav.py`
- `Tests/Chat/test_console_video_capacity.py`
- `Tests/Chat/test_console_video_controller.py`
- `Tests/Chat/test_console_video_message.py`
- `Tests/ProductionApp/test_chat_composition_retirement.py`
- `Tests/UI/app_factory.py`
- `Tests/UI/test_chat_screen_console_inspector_loader.py`
- `Tests/UI/test_chat_screen_worker_groups.py`
- `Tests/UI/test_console_ask_user_typed_answers.py`
- `Tests/UI/test_console_citation_sources.py`
- `Tests/UI/test_console_command_composer.py`
- `Tests/UI/test_console_composer_cursor.py`
- `Tests/UI/test_console_composer_history.py`
- `Tests/UI/test_console_composer_menu.py`
- `Tests/UI/test_console_composer_undo.py`
- `Tests/UI/test_console_controller_wiring.py`
- `Tests/UI/test_console_conversation_action_menu.py`
- `Tests/UI/test_console_conversation_persistence.py`
- `Tests/UI/test_console_cost_chip_estimate_cache.py`
- `Tests/UI/test_console_cost_chip_screen.py`
- `Tests/UI/test_console_dictation.py`
- `Tests/UI/test_console_dictation_streaming.py`
- `Tests/UI/test_console_draft_sync_equality_gate.py`
- `Tests/UI/test_console_edit_resend_wiring.py`
- `Tests/UI/test_console_fewer_permission_prompts_command.py`
- `Tests/UI/test_console_fleet_wake_ui_freshness.py`
- `Tests/UI/test_console_hands_free_wiring.py`
- `Tests/UI/test_console_internals_decomposition.py`
- `Tests/UI/test_console_launch_wake.py`
- `Tests/UI/test_console_library_search_modal.py`
- `Tests/UI/test_console_live_work_handoffs.py`
- `Tests/UI/test_console_message_controller.py`
- `Tests/UI/test_console_model_apply_chips.py`
- `Tests/UI/test_console_native_chat_flow.py`
- `Tests/UI/test_console_parallel_runs.py`
- `Tests/UI/test_console_pending_attachment_stash.py`
- `Tests/UI/test_console_prompts_controller.py`
- `Tests/UI/test_console_provider_apply_defaults_flow.py`
- `Tests/UI/test_console_rag_settings_modal.py`
- `Tests/UI/test_console_rail_reconciliation.py`
- `Tests/UI/test_console_raw_cli_send.py`
- `Tests/UI/test_console_regenerate_feedback.py`
- `Tests/UI/test_console_research_command.py`
- `Tests/UI/test_console_resize_reflow.py`
- `Tests/UI/test_console_resume_active_path.py`
- `Tests/UI/test_console_rewind_restore.py`
- `Tests/UI/test_console_roleplay_resume_navigation.py`
- `Tests/UI/test_console_send_draft_snapshot.py`
- `Tests/UI/test_console_session_controller.py`
- `Tests/UI/test_console_session_settings.py`
- `Tests/UI/test_console_settings_title_laziness.py`
- `Tests/UI/test_console_skill_commands.py`
- `Tests/UI/test_console_skill_controller.py`
- `Tests/UI/test_console_staged_evidence_strip.py`
- `Tests/UI/test_console_system_prompt.py`
- `Tests/UI/test_console_tick_gating.py`
- `Tests/UI/test_console_turn_activity_line.py`
- `Tests/UI/test_console_workbench_contract.py`
- `Tests/UI/test_console_workspace_files_integration.py`
- `Tests/UI/test_uat_first_time_character_chat.py`
- `Tests/integration/test_console_agent_marker_anchoring_e2e.py`
- `Tests/integration/test_console_branching_e2e.py`
- `Tests/integration/test_console_edit_resend_e2e.py`
- `Tests/integration/test_console_rewind_e2e.py`
- `Tests/integration/test_console_terminal_lifetime.py`
- `Tests/integration/test_console_thinking_end_to_end.py`
- `Tests/test_application_state_ownership.py`
