---
id: TASK-26986
title: Clean Ruff formatter debt for ruff-rag-search-tests
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

<!-- TASK-26000-BATCH: ruff-rag-search-tests -->
<!-- TASK-26000-PATHS-SHA256: 8755502d11bd7b10887b323064c454a9156e5476460c032c36188a753703341c -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-rag-search-tests` Ruff formatter batch at the owner boundary recorded as: Legacy RAG_Search query, fusion, reranker, and privacy tests.. The focused test surface recorded by TASK-26000 is `["Tests/RAG_Search"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/RAG_Search/test_cross_encoder_reranker.py",
  "Tests/RAG_Search/test_db_connection_pool_thread_safety.py",
  "Tests/RAG_Search/test_fts5_match_construction.py",
  "Tests/RAG_Search/test_fts5_query_escaping.py",
  "Tests/RAG_Search/test_fusion_config_knobs.py",
  "Tests/RAG_Search/test_fusion_rescue_pin.py",
  "Tests/RAG_Search/test_hybrid_allowlist_pushdown.py",
  "Tests/RAG_Search/test_hybrid_doc_fusion.py",
  "Tests/RAG_Search/test_hybrid_fusion_metadata.py",
  "Tests/RAG_Search/test_keyword_leg_chacha.py",
  "Tests/RAG_Search/test_keyword_leg_db_resolution.py",
  "Tests/RAG_Search/test_keyword_leg_prompts.py",
  "Tests/RAG_Search/test_keyword_leg_pushdown.py",
  "Tests/RAG_Search/test_keyword_leg_tiered_merge.py",
  "Tests/RAG_Search/test_rag_diagnostic_privacy.py",
  "Tests/RAG_Search/test_reranker_construction.py"
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
