---
id: TASK-869
title: 'Refresh default models for DeepSeek, Anthropic, and OpenAI'
status: Done
assignee:
  - '@codex'
created_date: '2026-07-27 03:58'
updated_date: '2026-07-27 13:37'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace stale provider defaults with current vendor-supported balanced general-purpose models while preserving supported alternatives and keeping provider request payloads compatible with the selected model families.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh installations default DeepSeek to deepseek-v4-flash, Anthropic to claude-sonnet-5, and OpenAI to gpt-5.6-terra
- [x] #2 Provider catalogs retain supported alternatives and exclude retired DeepSeek aliases from active defaults
- [x] #3 OpenAI GPT-5.6 default requests use a compatible token and reasoning contract without regressing explicit Responses API reasoning flows
- [x] #4 Anthropic Claude Sonnet 5 requests omit unsupported sampling parameters and honor supported adaptive-thinking settings
- [x] #5 The selected OpenAI and Anthropic defaults are recognized by capability metadata
- [x] #6 Focused configuration, provider-payload, and capability tests pass
- [x] #7 ADR-020 and the approved design are linked; no new ADR is required
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implementation plan: Docs/superpowers/plans/2026-07-26-provider-default-model-refresh.md
Design: Docs/superpowers/specs/2026-07-26-provider-default-model-refresh-design.md
ADR required: no
ADR path: backlog/decisions/020-automatic-model-catalog-refresh.md
Reason: bundled defaults and request shaping stay within existing provider boundaries; ADR-020 already governs catalog discovery and persistence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented provider default refresh. New bundled/catalog and fresh-install defaults: OpenAI gpt-5.6-terra, Anthropic claude-sonnet-5, DeepSeek deepseek-v4-flash; supported alternatives remain and existing user config precedence is preserved (no migration). GPT-5.6 uses Chat Completions with max_completion_tokens and top-level reasoning_effort=none for ordinary/tool requests, while explicit reasoning, summary, or verbosity continues through Responses with max_output_tokens. Sonnet 5 omits unsupported sampling fields, uses output_config effort where supported, and keeps adaptive thinking/output_config handling compatible. DeepSeek changes only the fallback model; its /chat/completions endpoint and payload contract are unchanged. Direct vision capability mappings now cover the OpenAI and Anthropic defaults without broad family patterns.

Changed production/test/doc files: tldw_chatbook/config.py; tldw_chatbook/LLM_Calls/LLM_API_Calls.py; tldw_chatbook/model_capabilities.py; Tests/test_config_model_catalog_defaults.py; Tests/Chat/test_chat_functions.py; Tests/test_model_capabilities.py; Docs/superpowers/plans/2026-07-26-provider-default-model-refresh.md; backlog/tasks/task-869 - Refresh-default-models-for-DeepSeek-Anthropic-and-OpenAI.md.

ADR required: no. Existing ADR: backlog/decisions/020-automatic-model-catalog-refresh.md (ADR-020). Approved design: Docs/superpowers/specs/2026-07-26-provider-default-model-refresh-design.md. Implementation plan: Docs/superpowers/plans/2026-07-26-provider-default-model-refresh.md.

Verification scope (2026-07-26): the approved focused suite, .venv/bin/python -m pytest Tests/test_config_model_catalog_defaults.py Tests/Chat/test_chat_functions.py Tests/test_model_capabilities.py -q, passed 89 tests. Fatal Ruff rules (E9,F63,F7,F82) passed on all six touched files. Ruff format passed for the five formatter-clean files; config.py remains excluded because of its pre-existing formatting baseline. compileall and git diff --check passed. Broad default Ruff is baseline-failing with 452 existing findings. A broader serial suite attempt was manually interrupted after 448 passed, 1 skipped, and 0 observed failures at 3% in 140.83s because its projected runtime was impractical; xdist was also attempted but is unsuitable because this repository tests share state. This was not a full-suite pass.

Approved scoped exception: TASK-869 remains Done because its acceptance criteria and approved implementation plan require the focused 89-test suite, not a full-suite pass; the broader-suite limitation is documented here transparently.

Post-rebase verification (2026-07-27): rebased the isolated TASK-869 series onto latest origin/dev and preserved newer Anthropic prompt-caching plus context-window metadata behavior. The focused suite passes 97 tests; Ruff fatal rules, compileall, and git diff --check origin/dev pass. A whole-file Ruff format pass is not claimed after the rebase because current dev contributes unrelated formatting findings in Tests/Chat/test_chat_functions.py and tldw_chatbook/model_capabilities.py; those files were not mechanically reformatted out of scope. Added a regression assertion and corrected the Sonnet 5 sampling-omission warning so it no longer incorrectly says thinking is enabled. Renumbered this task from TASK-519 to TASK-869 after latest dev introduced an unrelated TASK-519, preventing a duplicate-ID backlog guard failure.
<!-- SECTION:NOTES:END -->
