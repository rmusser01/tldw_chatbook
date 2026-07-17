---
id: TASK-263
title: Native tool-calls for normalizing providers (Anthropic/Google/Cohere)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-16 21:37'
updated_date: '2026-07-17 02:55'
labels:
  - agents
  - providers
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-243 wired native provider tool-calls for providers whose handlers return the raw OpenAI-compatible response dict (openai, groq, openrouter, mistral, deepseek, moonshot, custom-openai x2). Anthropic, Google, and Cohere were excluded from NATIVE_TOOLS_PROVIDERS because their handlers in LLM_API_Calls.py normalize responses and silently DROP provider-native tool-use blocks (e.g. chat_with_anthropic extracts only type==text content parts and never builds message.tool_calls, even though it maps finish_reason tool_use -> tool_calls). Until their normalizers are completed, these providers pay the fence-protocol overhead despite advertising native function-calling.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 chat_with_anthropic builds OpenAI-shape message.tool_calls from tool_use content blocks (non-streaming), and its streaming path reassembles input_json_delta fragments into the same shape,OpenAI-shape role=tool history messages are converted to the provider-native result format (Anthropic tool_result content blocks) before dispatch,Each converted provider is added to NATIVE_TOOLS_PROVIDERS only after an end-to-end native round-trip against the real API,Fence fallback remains intact for these providers until conversion lands (no partial states where finish_reason says tool_calls but message carries none)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan at Docs/superpowers/plans/2026-07-17-anthropic-native-tool-calls.md — Anthropic-only scope (user decision 2026-07-17; google/cohere -> follow-ups): (1) request-side conversion in chat_with_anthropic (OpenAI tools -> input_schema w/ Anthropic-shape passthrough; assistant.tool_calls -> tool_use blocks; role=tool -> coalesced tool_result user turn); (2) non-streaming tool_use -> message.tool_calls; (3) streaming content_block_start/input_json_delta -> OpenAI delta.tool_calls fragments (gateway accumulator shape, cross-layer pinned); (4) coordinator live gate w/ real key (user-provided) THEN set flip + service test + QA evidence. Executed via SDD.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Anthropic-only scope (user decision; google/cohere -> tasks 266/267). All conversion contained in chat_with_anthropic: (1) request side — _anthropic_tools_payload converts OpenAI function-shape to {name, description, input_schema} with historical Anthropic-shape passthrough; assistant.tool_calls -> tool_use content blocks (text block first; parse_native_tool_calls-equivalent junk guards after a review Important caught empty-name/empty-content 400 vectors); role=tool -> tool_result blocks coalesced into ONE user turn; (2) non-streaming — tool_use blocks -> OpenAI message.tool_calls (key absent for text-only, byte-compat pinned); (3) streaming — content_block_start/input_json_delta wired to the previously-dead placeholder, emitting OpenAI delta.tool_calls fragments keyed by 0-based tool position (gateway _ToolCallAccumulator contract, cross-layer pinned by a test importing the real gateway pieces). Flip gated per AC#3: live gate against the real API (claude-haiku-4-5, streaming ON) ran BEFORE the flip via in-harness set override — case A single round-trip PASS incl. tool_result turn acceptance; case B PASS with a genuine PARALLEL two-tool batch in one turn (also live-covers the interleaved-two-block streaming path). Evidence: Docs/superpowers/qa/anthropic-native-2026-07/. Key handled via git-excluded local file, never logged/committed. Suites: Agents 136, anthropic-native file 14, cross-suite green (one pre-existing http-client-swap contention flake, standalone 4/4 green, gateway untouched by branch).
<!-- SECTION:NOTES:END -->
