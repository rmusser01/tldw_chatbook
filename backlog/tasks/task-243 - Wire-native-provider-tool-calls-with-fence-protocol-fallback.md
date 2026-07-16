---
id: TASK-243
title: Wire native provider tool-calls with fence-protocol fallback
status: To Do
assignee: []
created_date: '2026-07-16 16:00'
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
- [ ] #1 A per-model/provider capability check selects native tool-calls when supported and falls back to the fence protocol otherwise (llama_cpp and other local backends without tools= support in PROVIDER_PARAM_MAP keep working via fallback)
- [ ] #2 ModelTurn.tool_calls is populated from a real native tool-call response for at least one cloud provider end-to-end
- [ ] #3 A native multi-tool-call reply is dispatched as multiple ToolCall entries in one run_agent_loop turn without engine changes
- [ ] #4 Existing fence-protocol tests and the Console agent-reply integration tests still pass unchanged for tool-incapable models
<!-- AC:END -->
