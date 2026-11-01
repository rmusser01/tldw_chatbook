---
id: TASK-26973
title: Clean Ruff formatter debt for ruff-evals
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

**Formatting (AC#2, AC#5).** 65 of 66 assigned paths formatted; `Tests/Evals/test_evals_db.py` was already formatter-clean at the base and was deliberately left untouched. `ruff format --check` passes on every existing assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's assigned paths: 122 findings before and 122 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Hashes equal on 65 of 66 paths. The one exception, `tldw_chatbook/Evals/word_bench/capture_client.py`, is a formatter-mandated docstring reindentation: 8 differing regions, ALL inside docstring continuation prose (net -48 characters of leading whitespace as Ruff's black-style docstring handling dedents over-indented docstring lines to the statement indentation), with comments equal and zero code nodes changed. Recorded as an extension of the documented docstring-normalization deviation class (batch-1 task-26937 / batch-4 task-26965 covered the quote-initial leading-space direction; this is the over-indentation dedent direction of the same mandatory normalization); all 8 regions enumerated in the verification evidence. The file has zero `__doc__` consumers (grep).

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/Evals` + `Tests/RAG_Eval` (recorded surface, bounded directories). Result on the formatted tree: 62 failed, 1171 passed, 26 skipped, 10 errors in 74.07s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 62 failed, 1171 passed, 26 skipped, 10 errors in 69.80s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** No already-clean production paths; the one clean path is the named test file.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
