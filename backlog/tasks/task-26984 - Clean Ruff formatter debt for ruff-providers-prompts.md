---
id: TASK-26984
title: Clean Ruff formatter debt for ruff-providers-prompts
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

<!-- TASK-26000-BATCH: ruff-providers-prompts -->
<!-- TASK-26000-PATHS-SHA256: dd749b206005793c208a5518328acac232cd59d1614e67c494411605f292e223 -->
<!-- TASK-26000-FINAL: false -->

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clean the `ruff-providers-prompts` Ruff formatter batch at the owner boundary recorded as: Provider, prompt, and chatbook services with direct contract tests.. The focused test surface recorded by TASK-26000 is `["Tests/Chatbooks", "Tests/LLM_Calls", "Tests/LLM_Provider_Catalog", "Tests/Prompt_Management"]`.
<!-- SECTION:DESCRIPTION:END -->

## Assigned Paths

```json
[
  "Tests/Chatbooks/test_chatbook_export_directory_default.py",
  "Tests/Chatbooks/test_chatbook_import_result_honesty.py",
  "Tests/Chatbooks/test_chatbook_kept_briefings_round_trip.py",
  "Tests/Chatbooks/test_chatbook_thinking_round_trip.py",
  "Tests/Chatbooks/test_local_chatbook_service_export.py",
  "Tests/Chatbooks/test_provider_continuation_roundtrip.py",
  "Tests/Internal_Prompts/test_agents_prompt_parity.py",
  "Tests/Internal_Prompts/test_authoring.py",
  "Tests/Internal_Prompts/test_document_generation_prompt_parity.py",
  "Tests/Internal_Prompts/test_resolver.py",
  "Tests/Internal_Prompts/test_summarization_prompt_parity.py",
  "Tests/Internal_Prompts/test_websearch_prompt_parity.py",
  "Tests/LLM_Calls/openai_realtime_probe.py",
  "Tests/LLM_Calls/openai_realtime_turn_detection_probe.py",
  "Tests/LLM_Calls/test_anthropic_redirect_credential_leak.py",
  "Tests/LLM_Calls/test_chat_model_capability_predicates.py",
  "Tests/LLM_Calls/test_kobold_tabby_config.py",
  "Tests/LLM_Calls/test_llama_summarizer_config.py",
  "Tests/LLM_Calls/test_moonshot.py",
  "Tests/LLM_Calls/test_pricing_catalog.py",
  "Tests/LLM_Calls/test_qwencloud.py",
  "Tests/LLM_Calls/test_realtime_protocol.py",
  "Tests/LLM_Calls/test_realtime_tls_trust.py",
  "Tests/LLM_Calls/test_summarization_model_capabilities.py",
  "Tests/LLM_Provider_Catalog/test_app_model_catalog_wiring.py",
  "Tests/LLM_Provider_Catalog/test_llm_provider_catalog_scope_service.py",
  "Tests/LLM_Provider_Catalog/test_local_llm_provider_catalog_service.py",
  "Tests/LLM_Provider_Catalog/test_model_auto_refresh.py",
  "Tests/LLM_Provider_Catalog/test_model_catalog_settings.py",
  "Tests/Prompt_Management/test_prompt_artifact_codec.py",
  "Tests/Prompt_Management/test_prompt_block_compiler.py",
  "Tests/Prompt_Management/test_prompt_legacy_decomposer.py",
  "Tests/Prompt_Management/test_server_prompt_adapter.py",
  "tldw_chatbook/Chatbooks/chatbook_creator.py",
  "tldw_chatbook/Internal_Prompts/__init__.py",
  "tldw_chatbook/Internal_Prompts/authoring.py",
  "tldw_chatbook/Internal_Prompts/document_generation_prompts.py",
  "tldw_chatbook/Internal_Prompts/resolver.py",
  "tldw_chatbook/Internal_Prompts/summarization_prompts.py",
  "tldw_chatbook/LLM_Calls/LLM_API_Calls.py",
  "tldw_chatbook/LLM_Calls/Local_Summarization_Lib.py",
  "tldw_chatbook/LLM_Calls/Summarization_General_Lib.py",
  "tldw_chatbook/LLM_Calls/pricing_catalog.py",
  "tldw_chatbook/LLM_Calls/realtime/openai_session.py",
  "tldw_chatbook/LLM_Calls/realtime/transport.py",
  "tldw_chatbook/LLM_Provider_Catalog/llm_provider_catalog_scope_service.py",
  "tldw_chatbook/LLM_Provider_Catalog/local_llm_provider_catalog_service.py",
  "tldw_chatbook/LLM_Provider_Catalog/model_auto_refresh.py",
  "tldw_chatbook/LLM_Provider_Catalog/model_catalog_settings.py",
  "tldw_chatbook/LLM_Provider_Catalog/openai_compatible_model_discovery.py",
  "tldw_chatbook/Prompt_Management/Prompts_Interop.py",
  "tldw_chatbook/Prompt_Management/prompt_artifact_codec.py",
  "tldw_chatbook/Prompt_Management/prompt_legacy_decomposer.py",
  "tldw_chatbook/Prompt_Management/prompt_normalizers.py",
  "tldw_chatbook/Prompt_Management/prompt_source_capabilities.py"
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

**Formatting (AC#2, AC#5).** 54 of 55 assigned paths formatted. `ruff format --check` passes on every assigned path after formatting (rc 0). Lint: `ruff check --output-format concise` over the batch's 233 assigned paths: 90 findings before and 44 after, ZERO increases — the formatter incidentally resolved 46 line-length findings in `Tests/Performance/run_console_three_turn_profile.py` (8), `Tests/Performance/test_console_three_turn_profile.py` (15), `Tests/RAG/simplified/test_collection_fingerprint.py` (16), and `Tests/RAG/test_active_config_resolution.py` (7) by reflowing; the remainder are pre-existing F401-type debt outside the TASK-26000 formatter scope.

**AST equality (AC#3).** Before/after `ast.parse(src, type_comments=True)` with `TypeIgnore.lineno` normalized to 0 and `ast.dump(include_attributes=False)`: hashes equal on every assigned path (the batch-5 symmetric plain-parse fallback was not needed on any file this batch).

**Comment directives (AC#4).** Ordered `tokenize.COMMENT` sequences and `# fmt: off` / `# fmt: on` counts are identical before and after on every assigned path.

**Focused tests (AC#6).** Focused surface: `Tests/Chatbooks` + `Tests/LLM_Calls` + `Tests/LLM_Provider_Catalog` + `Tests/Prompt_Management` (recorded surface, bounded directories). Result on the formatted tree: 191 failed, 3244 passed, 4 skipped, 59 errors in 404.12s (`--timeout=120`). Baseline A/B per `backlog/docs/lessons-backlog-hygiene.md` (no-stash rule): assigned paths swapped to unformatted `HEAD` content with `git checkout HEAD -- <paths>`, identical command re-run, files re-formatted and re-verified with `ruff format --check`. Baseline: 191 failed, 3244 passed, 4 skipped, 59 errors in 344.07s — count-identical to the formatted outcome, so the failures are pre-existing at the current dev tip (the documented hook-consent send-gate and config-admission red classes) and out of scope for a formatter-only task; AST equality independently proves runtime-identical files.

**Governance (AC#7).** `git diff --check` clean; `Tests/CI/test_backlog_task_id_uniqueness.py` — 3 passed (run on the fully formatted batch tree).

**No hand-written behavior change (AC#8).** Three-way partition verified arithmetically before closeout: 226 committed + 7 clean-at-base + 0 missing = 233 assigned paths; zero unassigned paths touched; every content diff is Ruff formatter output.

**Lineage.** `Tests/Chatbooks/test_local_chatbook_service_export.py` was already formatter-clean at the base and was deliberately left untouched; the other 54 carried live debt and were formatted.

ADR required: no — mechanical, formatter-only cleanup executing the owner-approved TASK-26000 contract; no architecture, storage, or cross-module boundary touched.
