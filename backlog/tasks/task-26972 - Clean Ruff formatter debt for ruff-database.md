---
id: TASK-26972
title: Clean Ruff formatter debt for ruff-database
status: To Do
assignee: []
created_date: '2026-08-31 18:31'
updated_date: '2026-08-31 18:31'
labels:
  - maintenance
  - formatting
  - quality
dependencies:
  - TASK-26000
references:
  - Docs/superpowers/specs/2026-08-30-task-26000-ruff-formatter-debt-design.md
  - Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json
priority: medium
---

<!-- TASK-26000-BATCH: ruff-database -->
<!-- TASK-26000-PATHS-SHA256: e2a925fcc5561e5a4cc0c3a313ea66f2fb4e1f14bb29e2e432dadd771568031b -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-database` Ruff formatter batch at the owner boundary recorded as: Database modules, migrations, and direct database tests.. The focused test surface recorded by TASK-26000 is `["Tests/ChaChaNotesDB", "Tests/DB"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/ChaChaNotesDB/test_chachanotes_db.py",
  "Tests/ChaChaNotesDB/test_character_persona_runtime_parity.py",
  "Tests/ChaChaNotesDB/test_console_dispatch_checkpoint_repository.py",
  "Tests/ChaChaNotesDB/test_console_library_policy_repository.py",
  "Tests/ChaChaNotesDB/test_historical_bootstrap.py",
  "Tests/ChaChaNotesDB/test_migration_atomicity.py",
  "Tests/DB/test_agent_run_steps_child_table.py",
  "Tests/DB/test_agent_runs_db.py",
  "Tests/DB/test_chachanotes_agent_lessons_seed_migration.py",
  "Tests/DB/test_chachanotes_bare_open_self_migration.py",
  "Tests/DB/test_chachanotes_character_authority_migration.py",
  "Tests/DB/test_chachanotes_citation_provenance_migration.py",
  "Tests/DB/test_chachanotes_console_context_memory_migration.py",
  "Tests/DB/test_chachanotes_console_library_migration_seed_openers.py",
  "Tests/DB/test_chachanotes_console_library_policy_migration.py",
  "Tests/DB/test_chachanotes_console_project_context_migration.py",
  "Tests/DB/test_chachanotes_default_assistant_enrichment_migration.py",
  "Tests/DB/test_chachanotes_fts_backfill_pacing.py",
  "Tests/DB/test_chachanotes_kept_briefings.py",
  "Tests/DB/test_chachanotes_message_metadata_migration.py",
  "Tests/DB/test_chachanotes_message_usage_migration.py",
  "Tests/DB/test_chachanotes_note_folders_migration.py",
  "Tests/DB/test_chachanotes_note_organization_receipts_migration.py",
  "Tests/DB/test_chachanotes_notes_organization_migration.py",
  "Tests/DB/test_chachanotes_sync_conflict_preservation_migration.py",
  "Tests/DB/test_chachanotes_sync_log_retention.py",
  "Tests/DB/test_chachanotes_sync_log_retention_migration.py",
  "Tests/DB/test_chachanotes_trajectory_metadata_migration.py",
  "Tests/DB/test_chachanotes_v47_messages_fts_backfill.py",
  "Tests/DB/test_chachanotes_v50_console_policy_tombstone_cleanup.py",
  "Tests/DB/test_chachanotes_v53_safe_capture_trim.py",
  "Tests/DB/test_chachanotes_v54_before_first_cursor.py",
  "Tests/DB/test_chachanotes_v55_console_memory_selection_migration.py",
  "Tests/DB/test_chachanotes_v56_console_trace_query_plans.py",
  "Tests/DB/test_chachanotes_world_book_priority_migration.py",
  "Tests/DB/test_chachanotes_world_book_regex_migration.py",
  "Tests/DB/test_character_cards_paging.py",
  "Tests/DB/test_character_conversation_seek_pagination.py",
  "Tests/DB/test_check_index_plan_pins.py",
  "Tests/DB/test_client_media_debug_logging.py",
  "Tests/DB/test_client_media_pagination.py",
  "Tests/DB/test_feature_store_lazy_open.py",
  "Tests/DB/test_fts5_quoting_search_seams.py",
  "Tests/DB/test_held_connections.py",
  "Tests/DB/test_media_db_schema_v6.py",
  "Tests/DB/test_media_db_schema_v7.py",
  "Tests/DB/test_media_db_schema_v8.py",
  "Tests/DB/test_media_db_schema_v9.py",
  "Tests/DB/test_pragma_settings.py",
  "Tests/DB/test_schema_table_allowlist_guard.py",
  "Tests/DB/test_search_conversations_fts.py",
  "Tests/DB/test_sql_validation.py",
  "Tests/DB/test_subscriptions_db.py",
  "Tests/DB/test_subscriptions_db_agent_read_only.py",
  "Tests/DB/test_subscriptions_db_briefing_provenance_migration.py",
  "Tests/DB/test_subscriptions_db_site_configs.py",
  "Tests/DB/test_subscriptions_db_watchlists.py",
  "Tests/DB/test_subscriptions_db_watchlists_agent_search.py",
  "Tests/DB/test_subscriptions_db_watchlists_reader_snapshot.py",
  "Tests/DB/test_workspace_db.py",
  "Tests/Media_DB/test_media_db_properties.py",
  "Tests/Media_DB/test_media_db_v2.py",
  "Tests/Prompts_DB/test_prompts_db_properties.py",
  "Tests/Prompts_DB/test_prompts_db_pytest.py",
  "tldw_chatbook/DB/AgentRuns_DB.py",
  "tldw_chatbook/DB/Client_Media_DB_v2.py",
  "tldw_chatbook/DB/Evals_DB.py",
  "tldw_chatbook/DB/Library_Collections_DB.py",
  "tldw_chatbook/DB/RAG_Indexing_DB.py",
  "tldw_chatbook/DB/Subscriptions_DB.py",
  "tldw_chatbook/DB/Workspace_DB.py",
  "tldw_chatbook/DB/chachanotes_fts_backfill.py"
]
```

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
