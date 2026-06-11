---
id: TASK-95
title: Console honors api_base_url endpoint overrides
status: Done
labels:
- console
- settings
- provider-config
- regression
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix Console provider endpoint resolution so llama.cpp and compatible local providers use the same endpoint precedence as Settings and readiness checks. This prevents merged default api_url values from overriding a user-configured api_base_url endpoint during actual Console send flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Console session defaults honor api_base_url for llama_cpp/local_llamacpp endpoints.
- [x] #2 Console readiness honors api_base_url when merged provider defaults also contain api_url.
- [x] #3 Console active provider selection uses the same endpoint precedence as readiness and Settings.
- [x] #4 Regression tests cover endpoint precedence for defaults, readiness, and active Console selection.
- [x] #5 CDP verification captures Console using the configured localhost llama.cpp endpoint and rendering a sent message response.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: N/A
Reason: This is a bugfix aligning existing endpoint alias precedence across Console, readiness, and Settings; it does not introduce a new storage, provider boundary, service contract, or long-lived architectural decision.

1. Reproduce the mismatch with a config that contains both merged default api_url and user api_base_url for llama_cpp.
2. Add failing regression coverage for default session settings, readiness, and active Console provider selection.
3. Align endpoint alias precedence so api_base_url wins over default api_url consistently.
4. Verify the focused provider/session tests and run diff whitespace checks.
5. Use CDP against textual-web to verify Console displays the configured localhost endpoint and renders a sent response.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Aligned Console endpoint resolution so llama.cpp-compatible providers prefer explicit `api_base_url` over merged default `api_url` across session defaults, readiness checks, and active send selection. The fix centralizes the active Console selection path on the shared endpoint helper and adds regressions for default settings, readiness, and `ChatScreen` provider selection.

Verification:
- `python -m pytest -q Tests/Chat/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py --tb=short` -> 89 passed.
- `python -m pytest -q Tests/Chat/test_console_provider_gateway.py Tests/Chat/test_console_session_settings.py Tests/UI/test_console_native_chat_flow.py --tb=short` -> 137 passed.
- `git diff --check` -> passed.
- CDP screenshot evidence: `/private/tmp/console-current-uat-start.jpg`, `/private/tmp/console-current-uat-typed-viewport.jpg`, `/private/tmp/console-current-uat-response.jpg`.

Direct endpoint probe confirmed the localhost service at `127.0.0.1:9099` returns the same UAT stub payload shown in Console, so the UI is reaching the configured endpoint and rendering the service response.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console now honors `api_base_url` endpoint overrides consistently when merged provider defaults also include `api_url`, preventing the active Console send path from falling back to the built-in llama.cpp default endpoint.

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
