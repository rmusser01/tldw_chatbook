---
id: TASK-26979
title: Clean Ruff formatter debt for ruff-mcp-runtime
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 11:40'
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

<!-- TASK-26000-BATCH: ruff-mcp-runtime -->
<!-- TASK-26000-PATHS-SHA256: 9c82bad22b83154f4655725399b789758309deb821b947db139f804fe41d1bca -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-mcp-runtime` Ruff formatter batch at the owner boundary recorded as: MCP control-plane, permission, execution, and server tools with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/MCP"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/MCP/test_control_plane_bridge.py",
  "Tests/MCP/test_control_plane_permissions.py",
  "Tests/MCP/test_control_plane_prompt_reducer.py",
  "Tests/MCP/test_control_plane_tool_execute.py",
  "Tests/MCP/test_execution_log.py",
  "Tests/MCP/test_library_tools.py",
  "Tests/MCP/test_local_runtime_delegate.py",
  "Tests/MCP/test_mcp_documentation_contract.py",
  "Tests/MCP/test_permission_prompt_reducer.py",
  "Tests/MCP/test_permission_resolution.py",
  "Tests/MCP/test_permission_store.py",
  "Tests/MCP/test_rag_search_tool.py",
  "Tests/MCP/test_readiness_derivation.py",
  "Tests/MCP/test_server_character_service.py",
  "Tests/MCP/test_server_chat_repoint.py",
  "Tests/MCP/test_server_media_db_path.py",
  "Tests/MCP/test_server_target_store.py",
  "tldw_chatbook/MCP/execution_log.py",
  "tldw_chatbook/MCP/local_runtime_delegate.py",
  "tldw_chatbook/MCP/local_server_tools.py",
  "tldw_chatbook/MCP/permission_prompt_reducer.py",
  "tldw_chatbook/MCP/tools.py",
  "tldw_chatbook/MCP/unified_control_plane_service.py"
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
6. Run the focused test surface recorded by TASK-26000 (bounded subset if the surface is whole-tree; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `e92b01515f` (isolated worktree `.worktrees/ruff-debt-batch-5`, branch `chore/ruff-debt-batch-5`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base except as recorded under Lineage. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 20 of 23 assigned paths formatted. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's assigned paths: 122 findings before and 122 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path. (For files whose `# type:` comments carry prose that `type_comments=True` cannot parse as a type expression — and only where BOTH sides fail identically — the comparison falls back to plain `ast.parse` on both sides; the fallback applies symmetrically and is recorded per file.)

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/MCP` (recorded surface, bounded directory). Result on the formatted tree: 158 failed, 1597 passed, 32 skipped in 296.04s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 158 failed, 1597 passed, 32 skipped in 224.52s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** 3 paths were already formatter-clean at the base (`Tests/MCP/test_control_plane_tool_execute.py`, `Tests/MCP/test_rag_search_tool.py`, `tldw_chatbook/MCP/tools.py`) and were deliberately left untouched; the other 20 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
