---
id: TASK-83
title: Add provider-aware generation settings and thinking controls
status: Done
labels:
- settings
- providers
- console
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Settings and Console fully support configurable generation defaults for Console-backed providers, including common sampler defaults and provider-specific OpenAI/Anthropic reasoning or thinking controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Providers & Models exposes editable per-model defaults for the common sampler fields needed by Console provider calls: temperature, top_p, min_p, top_k, max_tokens, seed, presence_penalty, and frequency_penalty.
- [x] #2 Console Behavior exposes matching global fallback defaults where appropriate, and per-model defaults continue to take precedence over global defaults.
- [x] #3 Console effective session settings resolve sampler and thinking controls from per-model defaults, chat_defaults, provider settings, and hard fallbacks consistently.
- [x] #4 ConsoleProviderGateway forwards supported sampler fields and provider-specific thinking/reasoning controls into chat_api_call without sending unsupported blank values.
- [x] #5 OpenAI-compatible reasoning settings and Anthropic thinking settings are stored provider-specifically, validated, and mapped to provider request payloads where supported.
- [x] #6 Unsupported provider reasoning controls are visibly unavailable rather than silently pretending to work.
- [x] #7 Focused unit/mounted tests cover Settings rendering/saving, Console default resolution, gateway forwarding, and provider payload shaping.
- [x] #8 Actual CDP/Textual-web screenshots verify the changed Settings UI before PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/006-provider-aware-generation-settings.md
Reason: This slice changes provider/runtime boundaries, persisted generation defaults, Console effective settings, and provider request-shape mapping.

1. Add failing pure tests for ConsoleSessionSettings default resolution, validation, summary, and gateway forwarding of sampler plus thinking controls.
2. Add failing mounted Settings tests proving Providers & Models and Console Behavior expose/save the expanded sampler defaults and provider-aware reasoning/thinking controls.
3. Add failing provider adapter tests for OpenAI reasoning payload fields and Anthropic thinking payload fields, including blank-value omission.
4. Extend the Console generation settings contract, selection model, default resolution, validation, and gateway forwarding with minimal new fields.
5. Extend Settings field lists, normalizers, save/revert order, copy, and rendering so global defaults and per-model overrides stay consistent.
6. Extend chat_api_call/provider adapters to accept and map supported thinking/reasoning controls without sending unsupported blank values.
7. Run focused verification, capture actual Textual-web/CDP screenshots for the changed Settings UI, update task notes/AC, and prepare a small PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added provider-aware sampler defaults to Settings Providers & Models and Console Behavior, including temperature, top_p, min_p, top_k, max_tokens, seed, presence_penalty, frequency_penalty, streaming, OpenAI reasoning fields, and Anthropic thinking fields.
- Extended Console session settings, provider selection, and ConsoleProviderGateway forwarding so per-model defaults override global chat_defaults and blank/unsupported values are omitted before provider calls.
- Extended OpenAI request shaping to use Responses API payloads when reasoning/verbosity controls are present, including Responses streaming normalization back to chat-style SSE; extended Anthropic request shaping for thinking effort/budget payloads.
- Added provider-gated Settings UI behavior so OpenAI exposes reasoning/verbosity while Anthropic thinking controls are disabled, and Anthropic exposes thinking controls while OpenAI reasoning controls are disabled.
- Added ADR-006 to document the Settings/Console/provider-adapter boundary for persisted generation defaults and request-shape translation.
- Verified with focused tests: `python -m pytest -q Tests/UI/test_console_session_settings.py Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_chat_functions.py::TestChatApiCall Tests/Chat/test_chat_functions.py::TestProviderRequestPayloads Tests/UI/test_settings_configuration_hub.py --tb=short` passed with 317 tests.
- Verified compile and diff hygiene with `python -m py_compile ...` and `git diff --check`.
- Captured actual Textual-web/CDP screenshots under `Docs/superpowers/qa/product-maturity/screen-qa/settings/generation-controls-2026-06-08/`; user approved the updated provider-gated controls screenshot.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Settings now behaves as a real generation-configuration hub for Console provider use: global defaults, per-model overrides, gateway forwarding, and provider payload shaping are covered by regression tests and actual CDP screenshot approval.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
