---
id: TASK-33351
title: >-
  Add gateway and host presets from the Hermes and oh-my-pi comparison (Vercel,
  ZenMux, Kilo, SiliconFlow, Baseten, GMI, Ollama Cloud, Upstage, Arcee,
  Qianfan, Nous, Venice, Meta)
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-28 23:30'
labels:
  - providers
  - engine
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
NousResearch/hermes-agent (about 40 providers) and can1357/oh-my-pi (about 83) support many OpenAI Chat Completions providers that tldw_chatbook does not. The ones that need no new engine capability -- a fixed host, a plain Bearer key, Chat Completions -- can each be a strict ADR-179 engine preset derived from the provider's public documentation. This adds thirteen: the gateways Vercel AI Gateway, ZenMux and Kilo; the hosts SiliconFlow, Baseten, GMI Cloud and Ollama Cloud; and the model makers Upstage, Arcee AI, Baidu Qianfan, Nous Research, Venice and Meta's new Model API (which replaced the retired Llama API and does support Chat Completions).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All thirteen providers are selectable and dispatch through the hosted engine with no per-provider module
- [x] #2 Every base URL, env var, allowance and reasoning setting is traceable to a cited public doc page or a recorded unauthenticated probe
- [x] #3 Documented-but-array reasoning fields never break replies: gateways that document an exclude option are asked to leave reasoning out; any array that still arrives fails closed
- [x] #4 Providers with no documented models route ship a seeded list and refuse discovery
- [x] #5 Every pre-existing engine preset keeps its request contract
- [x] #6 Settings/Console user guides and README list the new providers, env vars and caveats
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory Hermes (origin/main) and oh-my-pi providers; keep the ones that fit the engine unchanged (Chat Completions, fixed host, Bearer).
2. Research each provider's public docs in parallel and probe each host unauthenticated.
3. Add records, dispatch/parity-list wiring, config seeds/tables, and the Kilo discovery path.
4. Pin every documented value with doc-shaped bodies and negative controls; mutation-check.
5. Docs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Thirteen strict engine presets, data only (records + the existing hand lists + config tables). Chosen from the Hermes/oh-my-pi inventories as the providers needing no new engine capability; subscription logins (Codex, Copilot, Cursor, ...), Responses- or Anthropic-only APIs, cloud signing (Bedrock, Vertex) and xAI (ADR-179) were left out, as were Azure/CoreWeave/Cloudflare/OpenCode (small engine work each).

Notable decisions: Vercel and ZenMux document reasoning as a `reasoning_details` array the strict value rule rejects, so both send the documented `reasoning: {exclude: true}`. Kilo's documented mid-stream failure (top-level `error` + `finish_reason: "error"`) is allowed at the top level and mapped to a provider error via finish_provider_errors -- no new engine field needed. SiliconFlow's `eos` finish is a normal end. Meta reads `META_API_KEY`, not its SDKs' generic `MODEL_API_KEY` (configurable). Upstage and Qianfan document no models route and ship seeded. Nous's docs are bot-walled (facts via NousResearch/hermes-agent#47950): fully strict, tools off. Reasoning is private and replayed for hosts returning `reasoning_content` (SiliconFlow, Baseten, Arcee, Qianfan, Venice), matching Nebius/Novita.

Probes (unauthenticated, 2026-09-28): public OpenAI-shaped model lists at Vercel (390), Kilo (395), Ollama Cloud (17), Nous (421), Venice (126), ZenMux (204); Bearer header read by Vercel, GMI, Arcee; auth-gated routes at SiliconFlow, Upstage, Meta.

Verification: Tests/LLM_Calls/test_gateway_host_presets.py (123 tests); four mutations (Kilo error finish, Vercel exclude, Kilo discovery path, SiliconFlow eos) each turn tests red. Registry parity, catalog, readiness, engine, doc-derived/inference-cloud preset and UI-ready census suites pass. No live calls (no keys).
Qodo round (2026-09-29): Ollama Cloud documents `n`, `user` and `tool_choice` as unsupported but inherited the default payload flags. `tool_choice` is now an ordinary payload flag (on by default; the engine checks it before normalizing), and Ollama Cloud's flags leave out all three, so a caller-supplied value fails closed locally instead of being sent. Tools are still sent. No in-app path passes these fields today (`n` is not in the engine param map; Console passes neither `user` nor `tool_choice`), so the change only guards the direct `chat()` API. Both mutations (drop the engine gate; drop the record override) turn the new tests red. The Settings guide's "All except MiniMax are discovery-first" now also names Upstage and Qianfan.
<!-- SECTION:NOTES:END -->
