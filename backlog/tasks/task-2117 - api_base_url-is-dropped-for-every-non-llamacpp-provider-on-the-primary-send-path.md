---
id: TASK-2117
title: >-
  api_base_url is dropped for every non-llamacpp provider on the primary send path
status: To Do
assignee: []
created_date: '2026-08-03 15:10'
labels:
  - llm-calls
  - config
  - console
priority: high
dependencies:
  - TASK-2114
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-2114 fixed the Anthropic case. Investigating it revealed the cause is not
Anthropic-specific: `ConsoleProviderGateway._chat_api_kwargs` — the shared kwargs
builder for the primary Console send path — structurally drops `api_base_url` for
**every** provider except llama.cpp.

Confirmed affected: openai, cohere, deepseek, google, groq, huggingface,
mistral/mistralai, openrouter, moonshot, zai. (`llama_cpp` / `local_llamacpp` are
exempt — they take a separate direct-base_url code path.)

Severity splits in two:

- **Unmasked, live defects — google, huggingface, moonshot, mistral/mistralai.** For
  these the config key is disconnected from Console's canonical
  `[api_settings.<provider>]` section and nothing else stops the send, so a configured
  base URL is silently ignored while requests go to the default endpoint. Same failure
  shape as TASK-2114: no error, no warning, indistinguishable from a working proxy.
- **Currently masked — openai, cohere, deepseek, groq, openrouter, zai.** Console's
  "unsaved endpoint" send-gate happens to block these before the drop matters. That is
  a coincidence of an unrelated guard, not a fix; if the gate is ever relaxed or
  bypassed these become live defects too.

Fixing this centrally in `_chat_api_kwargs` is preferable to ten per-provider patches,
but each provider adapter's parameter name for the base URL must be verified rather
than assumed — the adapters are not uniform (see `PROVIDER_PARAM_MAP`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A configured `[api_settings.<provider>].api_base_url` reaches the primary Console send path for the four unmasked providers (google, huggingface, moonshot, mistral/mistralai)
- [ ] #2 The remaining affected providers (openai, cohere, deepseek, groq, openrouter, zai) either carry the fix too, or their masking gate is documented as the deliberate reason they are excluded
- [ ] #3 With no `api_base_url` configured, every provider's request URL is unchanged from today (no regression for the default case)
- [ ] #4 Each adapter's actual base-URL parameter name is verified against its signature/PROVIDER_PARAM_MAP rather than assumed uniform
- [ ] #5 Tests cover at least one unmasked provider end-to-end (configured base URL reaches the posted request) plus the unconfigured default
- [ ] #6 llama_cpp / local_llamacpp behavior is confirmed unchanged by the fix
<!-- AC:END -->
