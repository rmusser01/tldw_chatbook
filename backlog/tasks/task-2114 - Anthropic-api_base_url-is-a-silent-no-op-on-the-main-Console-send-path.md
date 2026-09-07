---
id: TASK-2114
title: Anthropic api_base_url is a silent no-op on the main Console send path
status: Done
assignee:
  - '@claude'
created_date: '2026-08-03 14:20'
updated_date: '2026-08-03 21:31'
labels:
  - llm-calls
  - anthropic
  - config
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`[api_settings.anthropic].api_base_url` is honored only on the auxiliary/one-shot
completion path, not on the main Console chat send. A user who sets it to redirect
traffic through a proxy, gateway, or self-hosted relay sees the setting silently
ignored for the calls that matter most — no error, no warning, no visible difference,
while every primary chat request still goes to the default Anthropic endpoint.

Found during the real-provider live verification of the cost-ticker program
(2026-08-03), while configuring a scratch profile against the live Anthropic API.
Pre-existing; not introduced by the cost-ticker PRs.

The failure mode is silence, which is what makes it worth fixing: a proxy that is
configured but bypassed looks identical to one that is working, until someone
inspects egress.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A configured `[api_settings.anthropic].api_base_url` is used for the main Console chat send (streaming and non-streaming), not just the auxiliary completion path
- [x] #2 With no `api_base_url` configured, the request URL is byte-identical to today's default endpoint (no behavior change for the common case)
- [x] #3 A test asserts the posted URL honors a configured base URL on the primary send path, and a second test pins the unconfigured default
- [x] #4 Any other provider adapter sharing this split (auxiliary honors, primary ignores) is identified in the implementation notes, or explicitly confirmed as Anthropic-only
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Trace the primary Console send path (ChatScreen -> console_chat_controller -> ConsoleProviderGateway.stream_chat -> _stream_generic_chat -> _chat_api_kwargs -> chat_api_call -> chat_with_anthropic) and compare it against the auxiliary/one-shot path (_auxiliary_chat_api_kwargs) to confirm exactly where api_base_url gets dropped.
2. Confirm chat_with_anthropic's own api_base_url parameter (and its URL-building fallback chain) already works correctly in isolation -- the bug is purely that the gateway never threads resolution.base_url into the primary kwargs dict.
3. Fix _chat_api_kwargs to forward resolution.base_url as api_base_url, scoped to Anthropic only (execution_key == "anthropic"), preserving byte-identical default behavior when unconfigured.
4. Add tests: unit-level _chat_api_kwargs coverage (configured + non-Anthropic no-op), gateway-level resolve_for_send+stream_chat kwargs-forwarding coverage, and true end-to-end posted-URL coverage (real chat_api_call + stubbed requests.Session) for both the configured and unconfigured cases.
5. Survey the other cloud provider adapters for the same auxiliary-honors/primary-ignores split and report findings in Implementation Notes (AC#4) without fixing them.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: traced the primary Console send path (ChatScreen -> console_chat_controller -> `ConsoleProviderGateway.stream_chat` -> `_stream_generic_chat` -> `_chat_api_kwargs` -> `chat_api_call` -> `chat_with_anthropic`) against the auxiliary/one-shot path (`_auxiliary_chat_api_kwargs`). Both ultimately reach the same `chat_api_call(**kwargs)` -> handler dispatch, and `chat_api_call` forwards `api_base_url` unconditionally to ANY provider handler whenever the top-level parameter is not `None` (`Chat_Functions.py`, outside `PROVIDER_PARAM_MAP` entirely). `_auxiliary_chat_api_kwargs` always includes `"api_base_url": resolution.base_url or None`; `_chat_api_kwargs` (the primary path, shared by every non-llama.cpp provider) never included the key at all -- so a configured `[api_settings.anthropic].api_base_url` never reached `chat_with_anthropic` on a real Console send, silently falling back to the built-in `https://api.anthropic.com/v1`.

`resolution.base_url` (set by `resolve_for_send` via `effective_provider_endpoint`) already resolves to the SAME built-in default when nothing is configured, so simply forwarding it is byte-identical in the unconfigured case (AC#2) and picks up a real override when one is configured (AC#1).

Fix: `_chat_api_kwargs` (`tldw_chatbook/Chat/console_provider_gateway.py`) now adds `kwargs["api_base_url"] = resolution.base_url or None` when `resolution.execution_key == "anthropic"`, scoped narrowly per this task's instructions rather than fixing every provider as a side effect.

Tests added (`Tests/Chat/test_console_provider_gateway.py`): two unit-level `_chat_api_kwargs` tests (configured value forwarded for Anthropic; key absent for a non-Anthropic resolution, proving the scoping), and two full end-to-end tests that drive the REAL `chat_api_call` (no `chat_api_call_fn` stand-in) with a stubbed `requests.Session`, asserting the actual posted URL both honors a configured base URL and stays byte-identical to the default when unconfigured (`test_console_send_honors_configured_anthropic_base_url`, `test_console_send_default_anthropic_url_unchanged_when_unconfigured`).

AC#4 (other providers, dispatched to a research pass, read-only, no code changes):
- Every cloud adapter with a `_BUILTIN_PROVIDER_ENDPOINTS` entry (openai, cohere, deepseek, google, groq, huggingface, mistral/mistralai, openrouter, moonshot, zai) already accepts and honors its own `api_base_url` parameter with the identical Anthropic-style fallback chain, and `chat_api_call`'s forwarding is provider-agnostic (no `PROVIDER_PARAM_MAP` entry needed). Since `_chat_api_kwargs` is the SHARED primary-path builder for every non-llama.cpp provider, ALL of these have the structurally identical gap Anthropic had -- confirmed no per-provider bypass exists other than llama.cpp's direct `stream_llamacpp_chat`/`complete_llamacpp_chat` path (which takes `base_url` as a first-class parameter and never goes through `_chat_api_kwargs`/`chat_api_call` at all).
- Nuance: for google, huggingface, moonshot, and mistral/mistralai, the adapter's OWN config-fallback reads a config key/section disconnected from Console's canonical `[api_settings.<provider>]` (`google_api` vs `[api_settings.google]`; legacy top-level `huggingface_api`/`moonshot_api`; `chat_with_mistral` always reads `api_settings["mistral"]` while the shipped default section is `[api_settings.mistralai]`) -- these are user-visible with the same severity as Anthropic's bug. For openai, cohere, deepseek, groq, openrouter, and zai, the adapter's fallback DOES read the same canonical section Console resolves from, and Console's "unsaved endpoint" gate (`provider_uses_endpoint` + `generic_endpoint_differs`) currently blocks sending when the session selection diverges from the saved config -- which masks the primary-path gap for those six today, though the code path is still inconsistent and worth a follow-up.
- Not fixed here, per this task's explicit scope (Anthropic only) -- recommend a follow-up task to either generalize the `_chat_api_kwargs` fix to all providers, or fix the disconnected-config-key adapters (google/huggingface/moonshot/mistral) as the higher-severity subset.

Files touched: tldw_chatbook/Chat/console_provider_gateway.py; Tests/Chat/test_console_provider_gateway.py; backlog/tasks/task-2114.
<!-- SECTION:NOTES:END -->
