---
id: TASK-26973
title: Clean Ruff formatter debt for ruff-evals
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

<!-- TASK-26000-BATCH: ruff-evals -->
<!-- TASK-26000-PATHS-SHA256: b1d8e221fc30b111c73f98c2bb2cb7def2ea6928786ff49fb16204604f4cb988 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-evals` Ruff formatter batch at the owner boundary recorded as: Evaluation runners, harnesses, and direct evaluation tests.. The focused test surface recorded by TASK-26000 is `["Tests/Evals", "Tests/RAG_Eval"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Evals/character_probe/test_bench_storage.py",
  "Tests/Evals/character_probe/test_conversation_storage.py",
  "Tests/Evals/character_probe/test_engine_end_to_end.py",
  "Tests/Evals/character_probe/test_probe_storage.py",
  "Tests/Evals/character_probe/test_prompt.py",
  "Tests/Evals/character_probe/test_runner.py",
  "Tests/Evals/character_probe/test_tags.py",
  "Tests/Evals/test_eval_execution_contracts.py",
  "Tests/Evals/test_eval_orchestrator.py",
  "Tests/Evals/test_eval_orchestrator_db_path.py",
  "Tests/Evals/test_evals_db.py",
  "Tests/Evals/test_evals_db_v3_to_v4_migration.py",
  "Tests/Evals/test_integration.py",
  "Tests/Evals/test_research_report_scorer.py",
  "Tests/Evals/word_bench/conftest.py",
  "Tests/Evals/word_bench/test_analysis.py",
  "Tests/Evals/word_bench/test_capture_client.py",
  "Tests/Evals/word_bench/test_engine_end_to_end.py",
  "Tests/Evals/word_bench/test_models.py",
  "Tests/Evals/word_bench/test_normalizer.py",
  "Tests/Evals/word_bench/test_run_existing_bench.py",
  "Tests/Evals/word_bench/test_runner.py",
  "Tests/Evals/word_bench/test_storage.py",
  "Tests/Evals/word_bench/test_storage_authoring.py",
  "Tests/RAG_Eval/conftest.py",
  "Tests/RAG_Eval/harness/baseline_io.py",
  "Tests/RAG_Eval/harness/canonicalize.py",
  "Tests/RAG_Eval/harness/cross_encoder_probe.py",
  "Tests/RAG_Eval/harness/environment.py",
  "Tests/RAG_Eval/harness/fixture_probe.py",
  "Tests/RAG_Eval/harness/fusion_sweep.py",
  "Tests/RAG_Eval/harness/goldenset.py",
  "Tests/RAG_Eval/harness/ingest.py",
  "Tests/RAG_Eval/harness/prf_probe.py",
  "Tests/RAG_Eval/harness/runner.py",
  "Tests/RAG_Eval/test_baseline_io.py",
  "Tests/RAG_Eval/test_canonicalize.py",
  "Tests/RAG_Eval/test_cross_encoder_probe.py",
  "Tests/RAG_Eval/test_cross_encoder_probe_run.py",
  "Tests/RAG_Eval/test_environment_cache_dir.py",
  "Tests/RAG_Eval/test_fixture_authoring_probe.py",
  "Tests/RAG_Eval/test_fixture_probe.py",
  "Tests/RAG_Eval/test_fusion_decision_rule.py",
  "Tests/RAG_Eval/test_fusion_sweep.py",
  "Tests/RAG_Eval/test_goldenset_integrity.py",
  "Tests/RAG_Eval/test_granularity_census.py",
  "Tests/RAG_Eval/test_harness_run.py",
  "Tests/RAG_Eval/test_harness_scoped.py",
  "Tests/RAG_Eval/test_harness_smoke.py",
  "Tests/RAG_Eval/test_hyde_probe.py",
  "Tests/RAG_Eval/test_metrics.py",
  "Tests/RAG_Eval/test_prf_probe.py",
  "Tests/RAG_Eval/test_prf_probe_run.py",
  "Tests/RAG_Eval/test_regression_gating.py",
  "Tests/RAG_Eval/test_runner_error_paths.py",
  "tldw_chatbook/Evals/ab_testing.py",
  "tldw_chatbook/Evals/character_probe/cards.py",
  "tldw_chatbook/Evals/character_probe/storage.py",
  "tldw_chatbook/Evals/eval_orchestrator.py",
  "tldw_chatbook/Evals/eval_runner.py",
  "tldw_chatbook/Evals/specialized_runners.py",
  "tldw_chatbook/Evals/word_bench/analysis.py",
  "tldw_chatbook/Evals/word_bench/capture_client.py",
  "tldw_chatbook/Evals/word_bench/normalizer.py",
  "tldw_chatbook/Evals/word_bench/runner.py",
  "tldw_chatbook/Evals/word_bench/storage.py"
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
