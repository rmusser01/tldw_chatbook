---
id: TASK-26984
title: Clean Ruff formatter debt for ruff-providers-prompts
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
- [ ] After rebasing onto current `origin/dev`, reproduce and reconcile every TASK-26000 assigned path; if upstream deleted, renamed, modified, or already formatted it, record that lineage and amend ownership mechanically without silently dropping it or absorbing an unassigned path. <!-- TASK-26000-CONTRACT: rebase-reconcile --><!-- TASK-26000-CONTRACT: drift-reconciliation -->
- [ ] Run Ruff 0.15.22 formatting on only the assigned paths, with no unassigned Python path changed. <!-- TASK-26000-CONTRACT: assigned-paths-only -->
- [ ] Before and after formatting, parse each assigned file on Python 3.12.11 with `ast.parse(..., type_comments=True)`, normalize only `TypeIgnore.lineno`, and require equal `ast.dump(..., include_attributes=False)`. <!-- TASK-26000-CONTRACT: ast-type-comments -->
- [ ] Preserve ordered comment-token text; anchor inline `# noqa`, `# type: ignore`, and single-target Ruff directives to the same deepest AST-node path and significant-token position, preserve standalone file directives between the same adjacent statement paths, and require each `# fmt: off` / `# fmt: on` range to enclose the same ordered AST-node interval. <!-- TASK-26000-CONTRACT: comment-directives -->
- [ ] Ruff lint and `ruff format --check` pass on every touched Python path. <!-- TASK-26000-CONTRACT: ruff-checks -->
- [ ] Implementation Notes record the focused-test rationale and every exact test command/result. <!-- TASK-26000-CONTRACT: focused-tests -->
- [ ] `git diff --check` and `Tests/CI/test_backlog_task_id_uniqueness.py` pass. <!-- TASK-26000-CONTRACT: governance -->
- [ ] The diff contains no hand-written production behavior change. <!-- TASK-26000-CONTRACT: no-handwritten-behavior -->
<!-- AC:END -->
