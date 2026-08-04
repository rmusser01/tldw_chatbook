---
id: TASK-2122
title: >-
  Nine of eleven providers report no usage on the streaming path
status: To Do
assignee: []
created_date: '2026-08-03 19:20'
labels:
  - cost-ticker
  - llm-calls
  - correctness
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The cost ticker's core promise is "real usage from the API, never an estimate." On the
streaming path that holds for **OpenAI and Anthropic only**. An audit of every
`stream_generator` in `LLM_Calls/LLM_API_Calls.py`, bounded by indentation so the
non-streaming path in the same function is not miscounted, found that **nine of eleven
providers emit no usage chunk at all when streaming**.

| Provider | Streaming generator | Usage emitted |
|---|---|---|
| openai | 758-829 | yes (`stream_options`) |
| anthropic | 1548-1741 | yes (PR1 work) |
| cohere | 2363-2555 | **no** |
| deepseek | 2950-2966 | **no** |
| google | 3417-3541 | **no** |
| groq | 3969-3992 | **no** |
| huggingface | 4373-4416 | **no** |
| mistral | 4740-4750 | **no** |
| openrouter | 4993-5003 | **no** |
| moonshot | 5363-5399 | **no** |
| zai | 5691-5711 | **no** |

Every one of these providers **does** handle usage on its non-streaming path, so the
gap is invisible in any non-streaming test and looks like working code on inspection.

There are two distinct root causes, needing two different fixes:

**1. OpenAI-compatible passthroughs** (deepseek, groq, huggingface, mistral,
openrouter, moonshot, zai). These generators are pure SSE relays — `for line in
response.iter_lines(): yield line + "\n\n"` — so they would forward a usage chunk
faithfully if one ever arrived. It never does: `stream_options: {"include_usage": True}`
is set at **exactly one site**, line 649, inside `chat_with_openai`. These providers are
simply never asked for usage. The fix is to request it, reusing the 400-degrade retry
already proven on the OpenAI path (line 768) since not every compatible endpoint accepts
the field.

**2. Native-protocol translators** (cohere, google). These parse provider-shaped SSE and
synthesize OpenAI-shaped chunks, so no payload flag can help — the translator has to read
the usage and emit a chunk. Both drop data the provider is already sending:
- Cohere's `message-end` branch (line 2485) reads `delta.finish_reason` and ignores
  `usage`, even though the generator's own comment above it documents `message-end` as
  carrying `(delta.finish_reason, usage)`. Cohere's non-streaming path already knows the
  shape: `usage.billed_units.{input_tokens,output_tokens}` (line 2631).
- Google's generator never reads `usageMetadata`, which Gemini sends on the final
  streaming chunk. Its non-streaming path maps it at line 3637.

**How exposed is this today?** Measured, not assumed — a probe over the shipped config
template resolving `build_default_console_session_settings` per provider:

- `[chat_defaults]` ships **no** `streaming` key, so the `True` fallback in
  `console_session_settings.py:436` does not apply; resolution falls through
  `default_sources = (model_profile, saved_defaults, chat_defaults, provider_settings)`
  to the per-provider template value, which is `streaming = false`.
- So for anthropic/openai/google/cohere/groq/deepseek/openrouter/moonshot the Console is
  **non-streaming out of the box** and usage capture works today.
- **Mistral is the exception and is broken by default**: its `[api_settings.mistral]`
  block is the only one with no `streaming` key, so it falls through to the `True`
  default. Mistral streams out of the box and therefore records no usage at all. That
  template inconsistency is worth fixing on its own.
- For the other eight, enabling streaming is a **one-keystroke, first-class action** —
  the Alt+M quick popover and the Settings screen's `chat_defaults.streaming` field, and
  a `chat_defaults` value outranks the per-provider template. The moment a user turns
  streaming on, usage capture silently stops: cost falls back to estimation and the cache
  chip loses the ground truth it derives warm/cold from — the two things the ticker
  exists to provide. Nothing warns them.

Missed because PR1's verification targeted Anthropic and PR3's live verification ran
against Anthropic only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `stream_options: {"include_usage": True}` is requested for the OpenAI-compatible providers, with the existing 400-degrade retry so endpoints that reject the field still stream
- [ ] #2 Cohere's streaming `message-end` handler reads `usage.billed_units` and emits a usage chunk matching the non-streaming path's bucket mapping
- [ ] #3 Google's streaming generator reads `usageMetadata` from the final chunk and emits a usage chunk matching its non-streaming mapping
- [ ] #4 A test per provider drives a recorded SSE stream through the generator and asserts a usage chunk reaches the gateway with correct disjoint buckets
- [ ] #5 A single guard test enumerates the streaming generators and fails if one emits no usage, so a newly added provider cannot silently regress
- [ ] #6 Verified live against at least one OpenAI-compatible provider and one native translator that a streamed Console turn persists real `usage_json`
- [ ] #7 `[api_settings.mistral]` gets an explicit `streaming` key so it stops differing from every other provider block by omission
<!-- AC:END -->
