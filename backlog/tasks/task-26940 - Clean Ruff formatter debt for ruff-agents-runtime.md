---
id: TASK-26940
title: Clean Ruff formatter debt for ruff-agents-runtime
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-09-30 22:25'
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
4. Snapshot after-ASTs; require per-file equality with the before-snapshot.
5. `ruff format --check` must pass on every assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface recorded by TASK-26000, plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `90597ade77` (isolated worktree `.worktrees/ruff-debt-batch-1`, branch `chore/ruff-debt-batch-1`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All 67 assigned paths exist unchanged at the base — no upstream delete or rename. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** All 67 assigned paths formatted (51 test files under `Tests/Agents/`, 16 runtime modules under `tldw_chatbook/Agents/`). `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's assigned paths shows an identical finding count before (243) and after — the formatter introduced zero findings; the pre-existing findings are F401-type debt outside the TASK-26000 formatter scope (hand-fixing them would violate AC#8).

**AST equality (AC#3).** For every one of the 67 assigned paths: `ast.parse(src, type_comments=True)`, `TypeIgnore.lineno` normalized to 0, `ast.dump(include_attributes=False)` — before and after hashes are equal.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface recorded by TASK-26000: `Tests/Agents` (3,624 tests collected). Result on the formatted tree: 809 failed, 2,808 passed, 1 skipped, 150 errors in 322.06s (`pytest -n 4 --timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): the 67 assigned paths were swapped to unformatted `HEAD` content with `git checkout HEAD -- Tests/Agents tldw_chatbook/Agents`, the identical command re-run, then the paths re-formatted and re-verified with `ruff format --check`. Baseline result: 810 failed, 2,807 passed, 1 skipped, 150 errors in 331.06s — the same pre-existing failure mass (the one-test delta is xdist/timing jitter under heavy concurrent load on this machine). The failures are therefore pre-existing at the current `dev` tip and out of scope for a formatter-only task; AST equality (AC#3) independently proves the formatted files are runtime-identical. Corroborating context: the concurrent TASK-19425 measurement documents a dev-tip hook-consent send-gate regression (`aed1b13501`, 2026-09-27) that fails controller/agent-adjacent suites.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the 90 assigned paths that needed formatting (92 assigned minus 2 already clean at the base) plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** No already-formatted paths in this batch; all 67 carried live debt at the base and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
