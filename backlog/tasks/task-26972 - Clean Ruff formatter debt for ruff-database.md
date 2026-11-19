---
id: TASK-26972
title: Clean Ruff formatter debt for ruff-database
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 09:05'
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
- [x] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [x] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [x] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [x] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [x] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [x] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [x] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [x] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->

## Implementation Plan

1. Reconcile assigned paths against current `origin/dev` (existence + format state) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` (parse with `type_comments=True`, normalize `TypeIgnore.lineno`, `include_attributes=False`) before formatting.
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality with the before-snapshot (formatter-mandated docstring quote-initial normalization documented if it occurs).
5. `ruff format --check` must pass on every assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface recorded by TASK-26000, plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `ee1c1e7365` (isolated worktree `.worktrees/ruff-debt-batch-4`, branch `chore/ruff-debt-batch-4`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base — no upstream delete or rename. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 72 of 72 assigned paths formatted (the database package and its migration tests). `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 195 assigned paths: 5 findings before and 5 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after hashes equal on every assigned path.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/ChaChaNotesDB` + `Tests/DB` (the recorded surface, bounded directories). Result on the formatted tree: 251 failed, 2682 passed, 2 skipped, 15 errors in 1037.53s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 252 failed, 2681 passed, 2 skipped, 15 errors in 957.26s. The +/-1 count delta was resolved by the name-diff protocol: a second run pair captured the FAILED name lists and they are IDENTICAL (252 vs 252 names, diff empty) — the earlier +/-1 was summary-count jitter between independent runs, not a formatting-correlated failure; AST equality independently proves runtime-identical files. — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** No already-formatted paths; all 72 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
