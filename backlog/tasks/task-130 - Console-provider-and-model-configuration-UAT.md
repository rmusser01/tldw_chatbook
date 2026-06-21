---
id: TASK-130
title: Console provider and model configuration UAT
status: Done
assignee:
- '@codex'
created_date: 2026-06-21 00:36
labels:
- console
- providers
- settings
- uat
dependencies:
- TASK-128
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verify and harden provider, model, endpoint, and model-setting configuration from Settings through Console runtime use for providers already supported by chat_api_call().
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 User can select or manually enter a provider supported by chat_api_call() without selected text being clipped or obscured.
- [x] #2 User can select or manually enter a model and see the effective provider/model reflected consistently in Settings, Console rail, and send readiness.
- [x] #3 Endpoint and credential-source fields are visible, editable where allowed, and preserved according to the existing Settings ownership contract.
- [x] #4 Model settings, including sampling and provider-specific controls, are visible and editable without hiding focused input text.
- [x] #5 Providers without native streaming support can complete as a single assistant message instead of being treated as broken.
- [x] #6 Focused regression coverage verifies provider selection, model entry, endpoint preservation, model setting persistence, and non-streaming fallback.
- [x] #7 Rendered CDP/Textual-web evidence is captured before approval and PR completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: TASK-130 hardens existing Settings-to-Console provider/model and generation-settings behavior under ADR-002 and ADR-006. It should not introduce a new provider registry, settings ownership rule, runtime boundary, storage contract, or provider-adapter service contract.

1. Inspect existing Console provider/model/session settings seams and Settings provider/model controls without changing chat lifecycle, workspace ownership, or message action behavior.
2. Add focused failing regression/UAT coverage for provider selection/manual provider entry, model entry and visible effective provider/model state, endpoint and credential-source preservation, model setting persistence/focus-visible inputs, and non-streaming fallback for chat_api_call()-supported providers.
3. Implement the smallest provider/model/settings readiness changes needed to satisfy the new failing tests while preserving ADR-002 provider identity normalization and ADR-006 effective-settings precedence.
4. Run targeted Console/provider/settings tests and any focused UI regression tests touched by the changes.
5. Capture actual CDP/Textual-web screenshot evidence for provider/model and model-setting focused states if rendered UI changes are made, storing it under Docs/superpowers/qa/console-uat-parallelization/ using the TASK-130 naming convention.
6. Add concise implementation notes and evidence paths to TASK-130, but do not mark the task Done until main-thread visual approval is obtained.
<!-- SECTION:PLAN:END -->

## Parallel Ownership

Owns Settings-to-Console provider/model configuration, readiness signals, model settings, endpoint/credential field behavior, and streaming fallback verification. Avoid chat tab lifecycle, message actions, and workspace conversation ownership except through stable public seams.

ADR required: no, unless implementation introduces new provider configuration ownership or runtime boundary rules.
ADR path: N/A unless implementation planning identifies a contract change.
Reason: Existing provider and Settings contracts should be exercised and hardened, not redesigned.

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
PR slice: exposed the Console settings modal's missing generation controls for seed, presence/frequency penalties, OpenAI-style reasoning/summary/verbosity, and Anthropic-style thinking/budget fields. The modal body now uses an explicit scrollable container so overflow controls remain reachable in Textual-web and by keyboard traversal.

Added regression coverage for the scrollable modal body and provider-specific generation control persistence. Captured approved CDP/Textual-web evidence at `Docs/superpowers/qa/console-uat-parallelization/TASK-130-provider-model-modal-controls-cdp.png`.

Final closeout: exposed the safe credential source in the Console settings summary so users can distinguish env-backed credentials from config-backed credentials without exposing secret values. Added regression coverage for `Credential: env OPENAI_API_KEY` and `Credential: config api_settings.<provider>.api_key`, then verified the broader provider/model path with the focused Console settings/provider/model suite plus generic non-streaming fallback coverage.

Approved rendered evidence:

- `Docs/superpowers/qa/console-uat-parallelization/TASK-130-provider-model-modal-controls-cdp.png`
- `Docs/superpowers/qa/console-uat-parallelization/task-130-provider-credential-source-cdp-2026-06-21.png`

Verification:

- `python -m pytest -q Tests/UI/test_console_session_settings.py -k "provider or model or endpoint or credential or generation or summary" --tb=short`
- `python -m pytest -q Tests/Chat/test_console_provider_gateway.py::test_stream_chat_generic_non_streaming_yields_completion_once Tests/UI/test_console_native_chat_flow.py::test_console_native_generic_provider_send_renders_completed_message --tb=short`

ADR required: no
ADR path: N/A
Reason: The final slice only changes safe display copy and regression evidence for existing provider/settings contracts; it does not introduce new provider ownership, runtime boundary, persistence, or service-contract decisions.
<!-- SECTION:NOTES:END -->
