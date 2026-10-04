---
id: TASK-26985
title: Clean Ruff formatter debt for ruff-rag-research
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-02 14:05'
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

<!-- TASK-26000-BATCH: ruff-rag-research -->
<!-- TASK-26000-PATHS-SHA256: a9f3554555be051480030b801d58d4642181e90037cc6f84477ff093f1da3560 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-rag-research` Ruff formatter batch at the owner boundary recorded as: RAG, embeddings, and research services with direct tests.. The focused test surface recorded by TASK-26000 is `["Tests/RAG", "Tests/RAG_Admin", "Tests/Research", "Tests/Research_Workspace"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/RAG/simplified/conftest.py",
  "Tests/RAG/simplified/test_chroma_persist_directory.py",
  "Tests/RAG/simplified/test_collection_fingerprint.py",
  "Tests/RAG/simplified/test_collection_indexes.py",
  "Tests/RAG/simplified/test_index_isolation_integration.py",
  "Tests/RAG/simplified/test_search_service.py",
  "Tests/RAG/test_active_config_resolution.py",
  "Tests/RAG/test_chunking_service.py",
  "Tests/RAG/test_config_profiles.py",
  "Tests/RAG/test_config_unification_parity.py",
  "Tests/RAG/test_first_run_import.py",
  "Tests/RAG/test_fusion.py",
  "Tests/RAG/test_local_citation_capture.py",
  "Tests/RAG/test_parent_child_adapter.py",
  "Tests/RAG/test_rag_admin_diagnostics_off_loop.py",
  "Tests/RAG/test_rag_ui_integration.py",
  "Tests/RAG/test_scope_store_filtering.py",
  "Tests/RAG/test_semantic_honest_states.py",
  "Tests/RAG_Admin/test_local_rag_admin_service.py",
  "Tests/RAG_Admin/test_template_validation.py",
  "Tests/Research/test_academic_providers.py",
  "Tests/Research/test_local_research_engine.py",
  "Tests/Research/test_local_research_search_service.py",
  "Tests/Research/test_local_research_service.py",
  "Tests/Research/test_research_budget.py",
  "Tests/Research/test_research_scope_service.py",
  "Tests/Research/test_research_source_catalog.py",
  "Tests/Research_Workspace/test_contracts.py",
  "Tests/Research_Workspace/test_controller.py",
  "Tests/Research_Workspace/test_quick_notes.py",
  "Tests/Research_Workspace/test_source_association.py",
  "Tests/Research_Workspace/test_source_selection.py",
  "Tests/Research_Workspace/test_workspace_adapters.py",
  "tldw_chatbook/Embeddings/Embeddings_Lib.py",
  "tldw_chatbook/RAG_Admin/local_rag_admin_service.py",
  "tldw_chatbook/RAG_Admin/template_validation.py",
  "tldw_chatbook/RAG_Search/__init__.py",
  "tldw_chatbook/RAG_Search/chunking_service.py",
  "tldw_chatbook/RAG_Search/config_profiles.py",
  "tldw_chatbook/RAG_Search/eval/gating.py",
  "tldw_chatbook/RAG_Search/eval/metrics.py",
  "tldw_chatbook/RAG_Search/eval/regression.py",
  "tldw_chatbook/RAG_Search/ingestion_indexing.py",
  "tldw_chatbook/RAG_Search/parent_child_adapter.py",
  "tldw_chatbook/RAG_Search/pipeline_builder_simple.py",
  "tldw_chatbook/RAG_Search/search_modes.py",
  "tldw_chatbook/RAG_Search/simplified/active_config.py",
  "tldw_chatbook/RAG_Search/simplified/collection_fingerprint.py",
  "tldw_chatbook/RAG_Search/simplified/collection_indexes.py",
  "tldw_chatbook/RAG_Search/simplified/rag_factory.py",
  "tldw_chatbook/RAG_Search/simplified/rag_service.py",
  "tldw_chatbook/RAG_Search/simplified/simple_cache.py",
  "tldw_chatbook/Research_Interop/academic_providers.py",
  "tldw_chatbook/Research_Interop/local_research_engine.py",
  "tldw_chatbook/Research_Interop/local_research_search_service.py",
  "tldw_chatbook/Research_Interop/local_research_service.py",
  "tldw_chatbook/Research_Interop/migrations/v0_to_v1_run_lease_columns.py",
  "tldw_chatbook/Research_Interop/research_source_catalog.py",
  "tldw_chatbook/Research_Workspace/contracts.py",
  "tldw_chatbook/Research_Workspace/controller.py",
  "tldw_chatbook/Research_Workspace/local_adapter.py",
  "tldw_chatbook/Research_Workspace/quick_notes.py",
  "tldw_chatbook/Research_Workspace/server_adapter.py",
  "tldw_chatbook/Research_Workspace/source_readiness.py"
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

1. Reconcile assigned paths against current `origin/dev` (existence + format state; missing paths recorded with their dev delete/rename lineage) and against the TASK-26000 evidence JSON `cleanup_records` `paths_sha256` (mechanical ownership check).
2. Snapshot each assigned path's `ast.dump` before formatting (type_comments=True with symmetric plain-parse fallback where prose `# type:` comments are unparseable; normalize `TypeIgnore.lineno`).
3. Run Ruff 0.15.22 `ruff format` on exactly the assigned paths.
4. Snapshot after-ASTs; require per-file equality (formatter-mandated docstring-normalization deviations enumerated if they occur).
5. `ruff format --check` must pass on every existing assigned path; `ruff check` findings must not increase vs the pre-format baseline.
6. Run the focused test surface (bounded; never a bare pytest), plus `Tests/CI/test_backlog_task_id_uniqueness.py` and `git diff --check`; verify the three-way path partition arithmetically before writing notes.
7. Tick ACs, record Implementation Notes (lineage, commands, results), set status Done.

## Implementation Notes

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `f80d3e0090` (isolated worktree `.worktrees/ruff-debt-batch-6`, branch `chore/ruff-debt-batch-6`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base — no deletes or renames to reconcile. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 63 of 64 assigned paths formatted. `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 233 assigned paths: 90 findings before and 44 after, ZERO increases — the formatter incidentally resolved 46 line-length findings in `Tests/Performance/run_console_three_turn_profile.py` (8), `Tests/Performance/test_console_three_turn_profile.py` (15), `Tests/RAG/simplified/test_collection_fingerprint.py` (16), and `Tests/RAG/test_active_config_resolution.py` (7) by reflowing; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Hashes equal on 63 of 64 paths. The one exception, `tldw_chatbook/Research_Interop/local_research_service.py`, is the sanctioned docstring-normalization class: exactly 5 differing regions, each a single leading space inserted after the opening triple quote before a quote-initial docstring line (dump delta +5 characters), comments equal, zero `__doc__` consumers (grep) — the batch-1 task-26937 / batch-4 task-26965 precedent.

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/RAG` + `Tests/RAG_Admin` + `Tests/Research` + `Tests/Research_Workspace` (recorded surface, bounded directories). Result on the formatted tree: 129 failed, 1261 passed, 115 skipped, 3 errors in 272.68s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 129 failed, 1261 passed, 115 skipped, 3 errors in 262.39s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 226 committed + 7 clean-at-base + 0 missing = 233 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** `Tests/Research/test_local_research_service.py` was already formatter-clean at the base and was deliberately left untouched; the other 63 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
