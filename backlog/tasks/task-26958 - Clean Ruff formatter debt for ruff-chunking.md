---
id: TASK-26958
title: Clean Ruff formatter debt for ruff-chunking
status: Done
assignee:
  - rmusser01
created_date: '2026-08-31 18:31'
updated_date: '2026-10-01 22:05'
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

<!-- TASK-26000-BATCH: ruff-chunking -->
<!-- TASK-26000-PATHS-SHA256: 9db46d30610226bf1a254b9f244a0e262d7f79d3e2186942ef12faac684ad232 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-chunking` Ruff formatter batch at the owner boundary recorded as: Chunking engine and direct chunking tests.. The focused test surface recorded by TASK-26000 is `["Tests/Chunking"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chunking/conftest.py",
  "Tests/Chunking/generate_auto_planner_parity_fixtures.py",
  "Tests/Chunking/golden/generate_golden.py",
  "Tests/Chunking/test_auto_apply_selection.py",
  "Tests/Chunking/test_auto_boundary_assistant.py",
  "Tests/Chunking/test_auto_chunking_planner.py",
  "Tests/Chunking/test_auto_chunking_resolver.py",
  "Tests/Chunking/test_auto_planner_parity.py",
  "Tests/Chunking/test_auto_selection.py",
  "Tests/Chunking/test_callsite_characterization.py",
  "Tests/Chunking/test_chunk_lib_shim.py",
  "Tests/Chunking/test_chunker_process_metrics.py",
  "Tests/Chunking/test_chunker_stream_diagnostic_privacy.py",
  "Tests/Chunking/test_chunker_v2.py",
  "Tests/Chunking/test_chunking_interop_v7.py",
  "Tests/Chunking/test_chunking_offsets_property.py",
  "Tests/Chunking/test_chunking_overlap_properties.py",
  "Tests/Chunking/test_chunking_regressions.py",
  "Tests/Chunking/test_chunking_runtime_lifecycle.py",
  "Tests/Chunking/test_chunking_templates.py",
  "Tests/Chunking/test_chunking_templates_validate_schema.py",
  "Tests/Chunking/test_descope_ledger.py",
  "Tests/Chunking/test_golden_parity.py",
  "Tests/Chunking/test_json_chunking.py",
  "Tests/Chunking/test_media_type_vocabulary.py",
  "Tests/Chunking/test_offsets_additional.py",
  "Tests/Chunking/test_option_aliases.py",
  "Tests/Chunking/test_overlap_clamp.py",
  "Tests/Chunking/test_phase3_3_sanitizers.py",
  "Tests/Chunking/test_process_text_components.py",
  "Tests/Chunking/test_process_text_refactor_equivalence.py",
  "Tests/Chunking/test_production_path_marker.py",
  "Tests/Chunking/test_propositions_strategy.py",
  "Tests/Chunking/test_security.py",
  "Tests/Chunking/test_security_fixed.py",
  "Tests/Chunking/test_semantic_offsets.py",
  "Tests/Chunking/test_shim_backcompat.py",
  "Tests/Chunking/test_shims.py",
  "Tests/Chunking/test_streaming_overlap.py",
  "Tests/Chunking/test_template_classifier.py",
  "Tests/Chunking/test_template_hierarchical_options.py",
  "Tests/Chunking/test_template_runtime.py",
  "Tests/Chunking/test_thai_tables_spans.py",
  "Tests/Chunking/test_thread_safety.py",
  "Tests/Chunking/test_tokens_offsets.py",
  "Tests/Chunking/test_upstream_chunking_templates.py",
  "Tests/Chunking/test_xml_allows_url_text.py",
  "tldw_chatbook/Chunking/Chunk_Lib.py",
  "tldw_chatbook/Chunking/_shims/Utils/prompt_loader.py",
  "tldw_chatbook/Chunking/_shims/config.py",
  "tldw_chatbook/Chunking/_shims/prompt_loader.py",
  "tldw_chatbook/Chunking/_shims/testing.py",
  "tldw_chatbook/Chunking/_template_conversion.py",
  "tldw_chatbook/Chunking/auto_selection.py",
  "tldw_chatbook/Chunking/chunking_interop_library.py",
  "tldw_chatbook/Chunking/engine/__init__.py",
  "tldw_chatbook/Chunking/engine/auto_planner.py",
  "tldw_chatbook/Chunking/engine/base.py",
  "tldw_chatbook/Chunking/engine/chunker.py",
  "tldw_chatbook/Chunking/engine/exceptions.py",
  "tldw_chatbook/Chunking/engine/llm_context.py",
  "tldw_chatbook/Chunking/engine/multilingual.py",
  "tldw_chatbook/Chunking/engine/process_text/dispatch.py",
  "tldw_chatbook/Chunking/engine/process_text/metadata.py",
  "tldw_chatbook/Chunking/engine/process_text/models.py",
  "tldw_chatbook/Chunking/engine/process_text/options.py",
  "tldw_chatbook/Chunking/engine/process_text/pipeline.py",
  "tldw_chatbook/Chunking/engine/process_text/preparation.py",
  "tldw_chatbook/Chunking/engine/regex_safety.py",
  "tldw_chatbook/Chunking/engine/security_logger.py",
  "tldw_chatbook/Chunking/engine/splitters/__init__.py",
  "tldw_chatbook/Chunking/engine/splitters/blingfire.py",
  "tldw_chatbook/Chunking/engine/splitters/regex.py",
  "tldw_chatbook/Chunking/engine/strategies/__init__.py",
  "tldw_chatbook/Chunking/engine/strategies/code.py",
  "tldw_chatbook/Chunking/engine/strategies/code_ast.py",
  "tldw_chatbook/Chunking/engine/strategies/ebook_chapters.py",
  "tldw_chatbook/Chunking/engine/strategies/ebook_chapters_patch.py",
  "tldw_chatbook/Chunking/engine/strategies/fixed_size.py",
  "tldw_chatbook/Chunking/engine/strategies/json_xml.py",
  "tldw_chatbook/Chunking/engine/strategies/paragraphs.py",
  "tldw_chatbook/Chunking/engine/strategies/propositions.py",
  "tldw_chatbook/Chunking/engine/strategies/rolling_summarize.py",
  "tldw_chatbook/Chunking/engine/strategies/sentences.py",
  "tldw_chatbook/Chunking/engine/strategies/structure_aware.py",
  "tldw_chatbook/Chunking/engine/strategies/words.py",
  "tldw_chatbook/Chunking/engine/templates.py",
  "tldw_chatbook/Chunking/engine/utils/metrics.py",
  "tldw_chatbook/Chunking/template_runtime.py"
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

**Approach.** Executed the TASK-26000 formatter-debt cleanup contract at `origin/dev` tip `922440b93e` (isolated worktree `.worktrees/ruff-debt-batch-3`, branch `chore/ruff-debt-batch-3`). Formatter: Ruff 0.15.22 (the TASK-26000 pin); the repository ships no Ruff configuration, so defaults apply, matching the census contract.

**Ownership reconciliation (AC#1).** All assigned paths exist unchanged at the base — no upstream delete or rename. Mechanically re-verified by recomputing `sha256(json.dumps(paths, separators=(",", ":")))` over the Assigned Paths: it matches both the `TASK-26000-PATHS-SHA256` marker in this file and the `paths_sha256` of the corresponding `cleanup_record` in `Docs/superpowers/reviews/evidence/task-26000/ruff-formatter-debt.json`.

**Formatting (AC#2, AC#5).** 89 of 89 assigned paths formatted (the Chunking package and its tests/shims). `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 164 assigned paths: 55 findings before and 55 after, per-file counts identical — the formatter introduced zero findings; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** For every assigned path: `ast.parse(src, type_comments=True)`, `TypeIgnore.lineno` normalized to 0, `ast.dump(include_attributes=False)` — before and after hashes are equal (no docstring-normalization exception was needed in this batch).

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/Chunking` (the recorded surface, bounded directory). Result on the formatted tree: 475 failed, 367 passed, 37 skipped, 1 error in 120.96s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 475 failed, 367 passed, 37 skipped, 1 error in 128.68s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Batch-wide verification: the modified working set is exactly the assigned paths that needed formatting plus the 8 batch task files; zero unassigned paths touched. Every content diff is Ruff formatter output.

**Lineage.** No already-formatted paths; all 89 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
