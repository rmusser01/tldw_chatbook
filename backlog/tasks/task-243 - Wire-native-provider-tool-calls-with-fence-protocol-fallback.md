---
id: TASK-243
title: Wire native provider tool-calls with fence-protocol fallback
status: Done
assignee:
  - '@claude'
created_date: '2026-07-16 16:00'
updated_date: '2026-07-17 00:03'
labels:
  - agents
  - console
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The vertical-slice design spec called for native tool-calls where the provider advertises them, with the fence-first text protocol as a fallback for tool-incapable models (Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md line 119). Only the fallback was ever implemented: agent_service.py's _make_call_model never sets tools=/tool_choice= and ModelTurn.tool_calls is always empty, so every provider — including ones that already support native function-calling end-to-end in LLM_API_Calls.py (OpenAI, Anthropic, Cohere, OpenRouter, Mistral, Gemini, HF, DeepSeek, Moonshot) — pays the fence-protocol's per-turn prompt overhead and one-tool-call-per-reply limit today. run_agent_loop already iterates multiple ToolCall entries per turn (agent_runtime.py line 276), so native multi-call batching is unlocked as soon as ModelTurn.tool_calls is populated from a real response.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A per-model/provider capability check selects native tool-calls when supported and falls back to the fence protocol otherwise (llama_cpp and other local backends without tools= support in PROVIDER_PARAM_MAP keep working via fallback)
- [x] #2 ModelTurn.tool_calls is populated from a real native tool-call response for at least one native-capable provider end-to-end (user decision 2026-07-16: custom-openai-api against real llama.cpp accepted — identical wire shape/code path as the OpenAI-compatible cloud providers; no cloud credential in environment. Evidence: Docs/superpowers/qa/native-tool-calls-2026-07/)
- [x] #3 A native multi-tool-call reply is dispatched as multiple ToolCall entries in one run_agent_loop turn without engine changes
- [x] #4 Existing fence-protocol tests and the Console agent-reply integration tests still pass unchanged for tool-incapable models
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan at Docs/superpowers/plans/2026-07-16-native-tool-calls.md — 6 tasks: (1) Agents/native_tools.py capability set + OpenAI converters + PROVIDER_PARAM_MAP groq/deepseek tools passthrough; (2) engine native history (ModelTurn.assistant_message echo + role=tool results keyed on call_id, fence path byte-identical); (3) AgentService native branch (tools=, protocol suppression, parse, AgentConfig.native_tools, spawn propagation); (4) gateway tools= passthrough + delta.tool_calls accumulation -> ProviderToolCalls sentinel (only when tools requested); (5) bridge adapter capture + execution_key-first endpoint + [console] native_tool_calls kill-switch; (6) live gate on a real cloud provider + fence regression on llama.cpp + follow-up task for Anthropic-family normalization. Executed via SDD.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented via SDD (5 tasks + coordinator-run live gate), plan at Docs/superpowers/plans/2026-07-16-native-tool-calls.md, final whole-branch review APPROVE. Native mode: NATIVE_TOOLS_PROVIDERS (handler-verified raw-OpenAI-dict providers: openai/groq/openrouter/mistral/deepseek/moonshot/custom-openai x2) selects tools= + fence-protocol suppression in AgentService._make_call_model; ModelTurn.assistant_message echoed verbatim + role=tool results keyed on call_id (fence path byte-identical, key-set pinned); gateway accumulates delta.tool_calls fragments into a ProviderToolCalls sentinel (only when tools= requested — plain sends byte-identical, regression-pinned); bridge adapter captures the sentinel (never hits transcript) with Finding-A leaked-prose reset parity; [console] native_tool_calls kill-switch default ON; groq/deepseek PROVIDER_PARAM_MAP gaps fixed; PRE-EXISTING crasher fixed (custom-openai-api + local-llm died on every call: provider_name=dict.capitalize()). AC#2: real native round-trip PASSED end-to-end through the Console reply engine against real llama.cpp via custom-openai-api (Docs/superpowers/qa/native-tool-calls-2026-07/) — literal 'cloud provider' wording awaiting user decision (no cloud credential available in environment). Follow-up task-246 filed (Anthropic/Google/Cohere normalizers drop tool_use). Deferred minors recorded in .superpowers/sdd/progress.md (m1-m6, all triaged defer by final review).
<!-- SECTION:NOTES:END -->
