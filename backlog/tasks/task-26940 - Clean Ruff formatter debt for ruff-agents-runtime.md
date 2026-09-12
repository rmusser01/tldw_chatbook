---
id: TASK-26940
title: Clean Ruff formatter debt for ruff-agents-runtime
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

<!-- TASK-26000-BATCH: ruff-agents-runtime -->
<!-- TASK-26000-PATHS-SHA256: f0fa7deb2bd65a653992b210e0def013e4e58c0062c5f5db7be4015223bd0e42 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-agents-runtime` Ruff formatter batch at the owner boundary recorded as: Agent runtime, catalog, fleet, and directly corresponding agent tests.. The focused test surface recorded by TASK-26000 is `["Tests/Agents"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Agents/conftest.py",
  "Tests/Agents/test_agent_lesson_promotion_end_to_end.py",
  "Tests/Agents/test_agent_lessons_runtime_guidance.py",
  "Tests/Agents/test_agent_loop_load_dedupe.py",
  "Tests/Agents/test_agent_models.py",
  "Tests/Agents/test_agent_runs_db_connection_reuse.py",
  "Tests/Agents/test_agent_runs_wake_ledger.py",
  "Tests/Agents/test_agent_runtime.py",
  "Tests/Agents/test_agent_runtime_preparation.py",
  "Tests/Agents/test_agent_runtime_review_hook.py",
  "Tests/Agents/test_agent_service.py",
  "Tests/Agents/test_agent_service_on_step.py",
  "Tests/Agents/test_agent_service_review_state_scope.py",
  "Tests/Agents/test_agent_step_incremental_persistence.py",
  "Tests/Agents/test_build_spawn_schema.py",
  "Tests/Agents/test_builtin_file_tools.py",
  "Tests/Agents/test_builtin_gate_real_tool_coverage.py",
  "Tests/Agents/test_fleet_continuation.py",
  "Tests/Agents/test_fleet_runtime.py",
  "Tests/Agents/test_fleet_send_to_agent.py",
  "Tests/Agents/test_fleet_steering_mailbox.py",
  "Tests/Agents/test_fleet_stop_semantics.py",
  "Tests/Agents/test_install_skill_runtime_tool.py",
  "Tests/Agents/test_library_tool_provider.py",
  "Tests/Agents/test_local_tools_integration.py",
  "Tests/Agents/test_mcp_provider_profile.py",
  "Tests/Agents/test_mcp_refusal_provenance.py",
  "Tests/Agents/test_persona_policy.py",
  "Tests/Agents/test_provider_continuation_runtime.py",
  "Tests/Agents/test_raw_shell_integration.py",
  "Tests/Agents/test_raw_shell_tool_provider.py",
  "Tests/Agents/test_run_log_cross_run_search.py",
  "Tests/Agents/test_run_log_eviction.py",
  "Tests/Agents/test_run_log_on_record.py",
  "Tests/Agents/test_run_log_prompt_integration.py",
  "Tests/Agents/test_run_log_resolve_existing.py",
  "Tests/Agents/test_run_log_search.py",
  "Tests/Agents/test_run_log_service_wiring.py",
  "Tests/Agents/test_run_log_stats_slice_runtime_tools.py",
  "Tests/Agents/test_run_log_survivor_lifetime.py",
  "Tests/Agents/test_run_log_workspace_isolation.py",
  "Tests/Agents/test_run_log_writer.py",
  "Tests/Agents/test_run_skill_script_runtime_tool.py",
  "Tests/Agents/test_run_tool_policy.py",
  "Tests/Agents/test_search_run_log_runtime_tool.py",
  "Tests/Agents/test_skill_tool_spawn.py",
  "Tests/Agents/test_tool_catalog.py",
  "Tests/Agents/test_tool_catalog_owner_cache.py",
  "Tests/Agents/test_trace_agent_lineage.py",
  "Tests/Agents/test_trace_approval_capture.py",
  "tldw_chatbook/Agents/agent_lesson_promotion.py",
  "tldw_chatbook/Agents/agent_models.py",
  "tldw_chatbook/Agents/agent_runtime.py",
  "tldw_chatbook/Agents/agent_service.py",
  "tldw_chatbook/Agents/fleet_coordinator.py",
  "tldw_chatbook/Agents/human_input_wait.py",
  "tldw_chatbook/Agents/library_rag_tool_provider.py",
  "tldw_chatbook/Agents/library_tool_provider.py",
  "tldw_chatbook/Agents/local_tool_provider.py",
  "tldw_chatbook/Agents/mcp_tool_provider.py",
  "tldw_chatbook/Agents/persona_policy.py",
  "tldw_chatbook/Agents/project_instruction_runtime.py",
  "tldw_chatbook/Agents/raw_shell_tool_provider.py",
  "tldw_chatbook/Agents/run_log.py",
  "tldw_chatbook/Agents/run_log_format.py",
  "tldw_chatbook/Agents/run_log_search.py",
  "tldw_chatbook/Agents/tool_catalog.py"
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
