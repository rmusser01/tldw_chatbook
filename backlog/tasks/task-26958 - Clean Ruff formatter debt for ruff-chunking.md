---
id: TASK-26958
title: Clean Ruff formatter debt for ruff-chunking
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
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
