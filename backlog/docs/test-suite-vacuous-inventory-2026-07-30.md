# Vacuous-test inventory — 2026-07-30 (task-1464, category 4)

Machine-generated candidates, per the owner ruling: **inventory only, no bulk
action** — clean up per-file when those files are touched. Read the caveats
before acting on any row.

## Caveats (why this list is candidates, not verdicts)

- The scanner counts a test as assertion-free when its own body has no
  `assert`, no `pytest.raises`, no unittest `self.assert*`, and no mock
  `assert_*` calls. **Assertions living in helper functions or fixtures are
  invisible to it** — e.g. contract suites that delegate to shared checkers
  will false-positive here. Verify per file before deleting anything.
- Mount-smoke tests (build a widget/screen, no explicit assert) still catch
  compose/crash regressions; assertion-free is not the same as valueless.
- Mock-only tests pin call graphs across seams: cheap, occasionally useful,
  prone to rot. The ruling keeps them.

## Assertion-free candidates: 249 tests in 118 files

| Count | File |
|---|---|
| 21 | `Tests/UI/test_non_obscuring_focus_contract.py` |
| 12 | `Tests/UI/test_destination_visual_parity_correction.py` |
| 11 | `Tests/TTS/test_profile_schema.py` |
| 9 | `Tests/Image_Generation/test_live_backends.py` |
| 7 | `Tests/Scheduling/test_watchlist_check_handler.py` |
| 7 | `Tests/tldw_api/test_notes_workspace_client.py` |
| 6 | `Tests/UI/test_console_mcp_approval.py` |
| 6 | `Tests/tldw_api/test_chat_conversation_client.py` |
| 5 | `Tests/test_enhanced_rag.py` |
| 5 | `Tests/UI/test_mcp_audit_mode.py` |
| 5 | `Tests/tldw_api/test_mcp_unified_client.py` |
| 4 | `Tests/UI/test_destination_shells.py` |
| 4 | `Tests/UI/test_file_picker_action_tooltips.py` |
| 4 | `Tests/UI/test_personas_workbench.py` |
| 4 | `Tests/Scheduling/test_scheduler_loop.py` |
| 4 | `Tests/RAG/test_rag_ui_integration.py` |
| 3 | `Tests/UI/test_mcp_workbench.py` |
| 3 | `Tests/UI/test_chatbook_action_recovery_tooltips.py` |
| 3 | `Tests/UI/test_settings_configuration_hub.py` |
| 3 | `Tests/UI/test_console_native_chat_flow.py` |
| 3 | `Tests/UI/test_disabled_action_recovery_tooltips.py` |
| 3 | `Tests/Evals/test_eval_runner.py` |
| 3 | `Tests/Chat/test_message_feedback.py` |
| 3 | `Tests/Prompts_DB/test_prompts_db_legacy.py` |
| 3 | `Tests/Character_Chat/test_world_info_regex.py` |
| 3 | `Tests/DB/test_private_sqlite.py` |
| 2 | `Tests/UI/test_bulk_selection_tooltips.py` |
| 2 | `Tests/UI/test_voice_blend_dialog.py` |
| 2 | `Tests/UI/test_screen_navigation.py` |
| 2 | `Tests/UI/test_schedules_workbench.py` |
| 2 | `Tests/Evals/test_eval_orchestrator.py` |
| 2 | `Tests/Chat/test_console_speech_snapshots.py` |
| 2 | `Tests/CI/test_textual_runtime_contract.py` |
| 2 | `Tests/Agents/test_run_log_writer.py` |
| 2 | `Tests/Utils/test_git_url_validation.py` |
| 2 | `Tests/Utils/test_config_encryption.py` |
| 2 | `Tests/MCP/test_control_plane_bridge.py` |
| 2 | `Tests/Local_Ingestion/test_dictation_window_provider_ids.py` |
| 2 | `Tests/Diarization/test_diarization_integration.py` |
| 1 | `Tests/test_minimal_world_book.py` |
| 1 | `Tests/test_utilities.py` |
| 1 | `Tests/test_smoke.py` |
| 1 | `Tests/test_fts5_pattern.py` |
| 1 | `Tests/Evaluations_Interop/test_evaluation_scope_service.py` |
| 1 | `Tests/Evaluations_Interop/test_server_evaluations_service.py` |
| 1 | `Tests/UI/test_evals_empty_states.py` |
| 1 | `Tests/UI/test_persona_profile_widgets.py` |
| 1 | `Tests/UI/test_library_file_notes_git.py` |
| 1 | `Tests/UI/test_file_picker_filters_callable.py` |
| 1 | `Tests/UI/test_latest_dev_core_app_usability_smoke.py` |
| 1 | `Tests/UI/test_tools_settings_window.py` |
| 1 | `Tests/UI/test_console_context_modal.py` |
| 1 | `Tests/UI/test_notes_unsaved_indicator_removed.py` |
| 1 | `Tests/UI/test_skill_script_confirm_card.py` |
| 1 | `Tests/UI/test_product_maturity_phase1_first_run.py` |
| 1 | `Tests/UI/test_product_maturity_phase1_navigation_smoke.py` |
| 1 | `Tests/UI/test_study_flashcards_screen.py` |
| 1 | `Tests/UI/test_tag_action_recovery_tooltips.py` |
| 1 | `Tests/UI/test_console_character_avatar.py` |
| 1 | `Tests/UI/test_workbench_pane_focus.py` |
| 1 | `Tests/UI/test_personas_expression_generate.py` |
| 1 | `Tests/UI/test_library_shell.py` |
| 1 | `Tests/UI/test_personas_lore.py` |
| 1 | `Tests/Media_DB/test_sync_client_integration.py` |
| 1 | `Tests/Media_DB/test_media_db_v2.py` |
| 1 | `Tests/Model_Artifacts/test_service.py` |
| 1 | `Tests/Evals/test_evals_db.py` |
| 1 | `Tests/Evals/test_eval_integration.py` |
| 1 | `Tests/Evals/word_bench/test_storage.py` |
| 1 | `Tests/Chat/test_citation_artifact_ownership.py` |
| 1 | `Tests/Chat/test_citation_trace_repository.py` |
| 1 | `Tests/Scheduling/test_schema.py` |
| 1 | `Tests/Scheduling/test_watchlist_scheduling_end_to_end.py` |
| 1 | `Tests/Transcription/test_faster_whisper_transcription.py` |
| 1 | `Tests/Transcription/test_mlx_whisper_transcription.py` |
| 1 | `Tests/Transcription/test_mlx_parakeet_transcription.py` |
| 1 | `Tests/Web_Scraping/test_security.py` |
| 1 | `Tests/Image_Generation/test_http_client.py` |
| 1 | `Tests/integration/test_file_extraction_integration.py` |
| 1 | `Tests/integration/test_file_operations_with_validation.py` |
| 1 | `Tests/Agents/test_run_log_search.py` |
| 1 | `Tests/Agents/test_builtin_tool_gate.py` |
| 1 | `Tests/Library/test_library_local_rag_search_service.py` |
| 1 | `Tests/Library/test_library_ingest_jobs.py` |
| 1 | `Tests/Utils/test_startup_polish_regressions.py` |
| 1 | `Tests/MCP/test_control_plane_permissions.py` |
| 1 | `Tests/MCP/test_unified_context_store.py` |
| 1 | `Tests/MCP/test_control_plane_lifecycle.py` |
| 1 | `Tests/RAG/test_scope_store_filtering.py` |
| 1 | `Tests/RAG/simplified/test_collection_fingerprint.py` |
| 1 | `Tests/RAG/simplified/test_vector_store_errors.py` |
| 1 | `Tests/RAG/simplified/test_collection_indexes.py` |
| 1 | `Tests/Chatbooks/test_chatbook_unit.py` |
| 1 | `Tests/Notes/test_notes_integration.py` |
| 1 | `Tests/Notes/test_sync_engine.py` |
| 1 | `Tests/Notes/test_notes_api_integration.py` |
| 1 | `Tests/Notes/test_library_notes_sync_integration.py` |
| 1 | `Tests/Character_Chat/test_world_book_manager.py` |
| 1 | `Tests/TTS/test_higgs_integration.py` |
| 1 | `Tests/TTS/test_tts_request_admission.py` |
| 1 | `Tests/TTS/test_profile_service.py` |
| 1 | `Tests/TTS/test_audio_cpp_contract.py` |
| 1 | `Tests/TTS/test_tts_improvements.py` |
| 1 | `Tests/DB/test_subscriptions_db_watchlists.py` |
| 1 | `Tests/DB/test_pagination.py` |
| 1 | `Tests/DB/test_subscriptions_db.py` |
| 1 | `Tests/DB/test_sql_debug_logging.py` |
| 1 | `Tests/DB/test_core_sqlite_owner_privacy.py` |
| 1 | `Tests/Auth_Account/test_server_auth_account_service.py` |
| 1 | `Tests/Skills/test_skill_trust_service.py` |
| 1 | `Tests/Skills/test_verify_content_binary.py` |
| 1 | `Tests/Performance/test_app_startup_performance.py` |
| 1 | `Tests/LLM_Provider_Catalog/test_llm_provider_catalog_scope_service.py` |
| 1 | `Tests/Audio_Services/test_server_audio_services_service.py` |
| 1 | `Tests/Audio_Services/test_audio_services_scope_service.py` |
| 1 | `Tests/LLM_Management/test_mlx_lm.py` |
| 1 | `Tests/Diarization/test_diarization_service.py` |
| 1 | `Tests/RAG_Search/test_embeddings_performance.py` |

## Mock-callgraph-only: 119 tests in 47 files

| Count | File |
|---|---|
| 14 | `Tests/Scheduling/test_watchlist_check_handler.py` |
| 14 | `Tests/tldw_api/test_mcp_unified_client.py` |
| 9 | `Tests/LLM_Management/test_mlx_lm.py` |
| 8 | `Tests/UI/test_library_screen.py` |
| 6 | `Tests/Scheduling/test_reminder_handler.py` |
| 5 | `Tests/Transcription/test_mlx_whisper_transcription.py` |
| 4 | `Tests/UI/test_code_repo_copy_paste_window.py` |
| 4 | `Tests/UI/test_study_screen.py` |
| 4 | `Tests/UI/test_personas_workbench.py` |
| 3 | `Tests/UI/test_home_screen.py` |
| 3 | `Tests/Scheduling/test_scheduling_service.py` |
| 2 | `Tests/UI/test_chat_search_enhanced.py` |
| 2 | `Tests/UI/test_media_window_v2_parity.py` |
| 2 | `Tests/Evals/test_eval_orchestrator.py` |
| 2 | `Tests/Scheduling/test_scheduler_loop.py` |
| 2 | `Tests/Watchlists/test_watchlists_collections_screen.py` |
| 2 | `Tests/Subscriptions/test_notification_dispatch_service.py` |
| 2 | `Tests/Audio/test_dictation_service.py` |
| 2 | `Tests/Audio/test_voice_input_widget.py` |
| 2 | `Tests/TTS/test_tts_app_ownership.py` |
| 1 | `Tests/UI/test_command_palette_providers.py` |
| 1 | `Tests/UI/test_console_mcp_approval.py` |
| 1 | `Tests/UI/test_product_maturity_phase3_knowledge_entry.py` |
| 1 | `Tests/UI/test_stts_playground_audio_cpp.py` |
| 1 | `Tests/UI/test_product_maturity_phase3_library_study_context.py` |
| 1 | `Tests/UI/test_console_context_modal.py` |
| 1 | `Tests/UI/test_console_skill_commands.py` |
| 1 | `Tests/UI/test_skill_script_confirm_card.py` |
| 1 | `Tests/UI/test_command_palette_basic.py` |
| 1 | `Tests/UI/test_search_rag_window.py` |
| 1 | `Tests/UI/test_console_skill_install_confirm.py` |
| 1 | `Tests/UI/test_personas_preview_restore.py` |
| 1 | `Tests/UI/test_library_shell.py` |
| 1 | `Tests/Evals/test_specialized_runners.py` |
| 1 | `Tests/Scheduling/test_sync_engine.py` |
| 1 | `Tests/Transcription/test_mlx_parakeet_transcription.py` |
| 1 | `Tests/Web_Scraping/test_security.py` |
| 1 | `Tests/integration/test_code_repo_integration.py` |
| 1 | `Tests/Event_Handlers/test_note_ingest_events.py` |
| 1 | `Tests/RAG/simplified/test_rag_service_basic.py` |
| 1 | `Tests/Audio/test_recording_service.py` |
| 1 | `Tests/Audio/test_console_dictation.py` |
| 1 | `Tests/Notes/test_notes_library_unit.py` |
| 1 | `Tests/TTS/test_tts_improvements.py` |
| 1 | `Tests/Wizards/test_first_run_setup_wizard.py` |
| 1 | `Tests/Widgets/test_chat_message_enhanced.py` |
| 1 | `Tests/RAG_Search/test_embeddings_unit.py` |
