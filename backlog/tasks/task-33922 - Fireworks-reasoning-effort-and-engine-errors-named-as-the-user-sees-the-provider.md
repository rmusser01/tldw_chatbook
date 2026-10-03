---
id: TASK-33922
title: >-
  Fireworks reasoning effort, and engine errors named as the user sees the
  provider
status: Done
assignee:
  - '@Robert'
created_date: '2026-10-03 01:38'
updated_date: '2026-10-03 02:17'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two follow-ups from the provider-engine burn-down. Fireworks documents reasoning_effort (none, low, medium, high, xhigh; no minimal) but its preset refused every level, so Console offered nothing to turn thinking off. And the hosted transport named providers by their key in HTTP error copy ('nvidia authentication failed'), which Console shows the user, while a bare 404 gave no hint that the model or the key's access is the usual cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fireworks sends a documented reasoning effort level, refuses undocumented ones locally, and every level Settings or Console offers is one it sends
- [x] #2 Engine HTTP error copy names each preset by its catalog display name, while the error's provider identity stays the key
- [x] #3 A 404 tells the user to check the model name and the key's access, in the copy Console shows
- [x] #4 Legacy adapters' error copy is unchanged apart from the 404 hint
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fireworks: record reasoning_effort on, with a documented reasoning_effort_values set (no minimal); engine refuses other levels; Console draft rebase carries the level; Settings offers only Fireworks' levels.
2. Transport: optional display_name on HostedHTTPTransportConfig used only in message text; engine passes record.display_name; legacy adapters unchanged.
3. 404 copy names the likely cause.
4. Tests + mutations; Settings/Console guide updates.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fireworks: docs.fireworks.ai/api-reference/post-chatcompletions documents reasoning_effort none/low/medium/high/xhigh (plus max/adaptive, which Console does not offer) and no minimal. New ProviderRecord.reasoning_effort_values (None = forward any level, the custom-hosted behaviour) holds the documented set for FIREWORKS. The engine refuses any other level locally; reasoning_effort_values_sent filters Settings' list to it; _capability_generation_fields carries reasoning_effort for engine presets that send it, so the draft rebase keeps the level. Console support stays "unknown" because model support varies (some always reason). Not live-verified (TASK-33640). The thinking + reasoning_effort conflict cannot arise: no preset sends thinking (pinned since TASK-33503).
Error copy: hosted_chat named providers by their key in every HTTP error Console shows ("provider returned HTTP 401 (nvidia authentication failed...)"). HostedHTTPTransportConfig.display_name (optional) now labels the copy, while the error's provider identity stays the key; the engine passes record.display_name (pinned equal to the catalog name by TASK-33002.14's parity test). A 404 now reads "<name> could not find that model or endpoint (status 404). Check the model name and that your key can use it.": Fireworks, SambaNova, Nous and GMI answer an unknown model, or a bad key with a model check first, with 404 (no-key probes, TASK-33640). Legacy adapters keep key-named copy apart from that 404 hint. Mutations: engine name dropped, label ignored, level check off, Settings filter off, rebase field off -> each turns tests red.
Qodo round (#2970): Console offers every level on every provider and the draft rebase carries a level across a provider switch, so a Fireworks allowlist without "minimal" let a turn fail locally. FIREWORKS.reasoning_effort_map now sends Console's "minimal" as "low" (its lightest documented level), and reasoning_effort_values_sent applies the map, so Settings offers all six. Mapping the value was chosen over filtering the Console modal and rebase, which would have touched two size-ratcheted modules and the Select Changed-echo path. Console replaced the transport's copy with its own ("Provider error from nvidia: ..."), so the display name and 404 hint never reached the user: safe_provider_error_copy now names the provider by provider_display_name and calls a 404 "model or endpoint not found", and the model-recovery copy covers 404 as well as 400. Seven gateway pins of "Provider error from openai/anthropic" were rewritten on purpose to the display names. New tests: engine -> real owned_json_post (fake session) -> Console copy for 401 and 404; the Console modal carries "minimal" to Fireworks and the request sends "low"; chat_api_call dispatch sends the chosen level.
Not done, closed with reasons: NVIDIA qwen3-235b "thinking" kwarg (NVIDIA's public listing names no Qwen model, 2026-09-30); Cerebras strict tools (optional per Cerebras' API reference, and most of Chatbook's tool schemas would fail strict validation -- TASK-33500).
<!-- SECTION:NOTES:END -->
