---
id: TASK-33502
title: 'NVIDIA NIM: let users turn Qwen thinking off'
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-29 19:24'
updated_date: '2026-09-29 20:02'
labels:
  - providers
  - engine
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Qwen3.5 models on NVIDIA NIM think on every turn by default; the only documented switch is `chat_template_kwargs.enable_thinking`. Chatbook sends nothing, so users pay reasoning latency and tokens with no way to turn thinking off.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Choosing reasoning effort 'none' for an NVIDIA model that documents the thinking toggle sends the documented thinking-off field
- [x] #2 Other models and other effort levels send exactly what they send today, and unsupported levels are still refused locally
- [x] #3 Console offers the thinking-off choice only where it will actually be sent
- [x] #4 Tests pin the payload for matching, non-matching and default cases
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a ProviderRecord field mapping model globs to the chat_template_kwargs key that toggles thinking; set it on NVIDIA for qwen/qwen3.5-* (enable_thinking).
2. Engine: for a matching model, reasoning_effort none sends {key: false}, any other level {key: true}; non-matching models keep the local refusal.
3. Console support answers supported only for matching models.
4. Tests for matching/non-matching/default payloads and the support answer; Settings guide note.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
NVIDIA NIM's Qwen3.5 schema (build.nvidia.com/qwen/qwen3.5-397b-a17b, read 2026-09-29) has no reasoning_effort; thinking is on by default and switched only by chat_template_kwargs.enable_thinking. New ProviderRecord field thinking_toggle_models (model glob -> kwarg) plus a stdlib helper thinking_toggle_key (provider_registry.py), set only on NVIDIA for qwen/qwen3.5-*. The engine sends {key: effort != "none"} for a matching model (NIM has on/off, no levels, so any non-none level means on), keeps the local refusal for other models, and sends nothing when no effort is chosen. Console: the shared helper answers "supported" for matching models, and _capability_generation_fields adds reasoning_effort for them so the draft rebase carries the choice. Because every level maps to a sent value, no per-provider option filtering was needed in the modal. Settings/Console guides updated. Mutations (engine branch, rebase field) turn tests red. Not done: qwen3-235b-a22b's different kwarg ("thinking") -- its page 404s; add a glob if it is live.
<!-- SECTION:NOTES:END -->
