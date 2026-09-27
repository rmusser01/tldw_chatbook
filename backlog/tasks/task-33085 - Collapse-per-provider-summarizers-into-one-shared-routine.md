---
id: TASK-33085
title: Collapse per-provider summarizers into one shared routine
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, providers]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nineteen near-identical summarize_with_* functions across LLM_Calls/Summarization_General_Lib.py and LLM_Calls/Local_Summarization_Lib.py (roughly 4k LOC) each repeat the same sequence per provider: key-from-param-or-config, text extraction, default temperature, headers, payload assembly, retry, and SSE parsing — dispatched by a ~200-line elif chain. Each new provider currently costs another ~250-line function. One shared OpenAI-shaped request routine plus a per-provider config row (config key, default model, endpoint quirks) removes an entire class of per-provider edits; providers with genuinely unique flows (Anthropic, Google wire format, huggingface quirks) keep adapter hooks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single shared routine executes the common request lifecycle for all supported providers.
- [ ] #2 Each provider is expressed as a config row covering config key, default model, and endpoint quirks.
- [ ] #3 The elif dispatch chain is replaced by a table lookup.
- [ ] #4 Existing summarization behavior is preserved per provider, verified by targeted tests plus golden-output checks.
- [ ] #5 The net LOC change is recorded in Implementation Notes.
<!-- AC:END -->
