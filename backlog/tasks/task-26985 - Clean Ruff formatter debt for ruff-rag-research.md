---
id: TASK-26985
title: Clean Ruff formatter debt for ruff-rag-research
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
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
