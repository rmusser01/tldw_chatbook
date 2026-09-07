---
id: TASK-2121
title: >-
  Console prompt caching is unavailable through OpenRouter (gate, session_id, pricing)
status: To Do
assignee: []
created_date: '2026-08-03 19:05'
labels:
  - cost-ticker
  - console
  - openrouter
priority: medium
dependencies:
  - TASK-2120
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR2 gave the Console a per-turn `cache_control` breakpoint so conversation history
becomes a reusable cache prefix. `ConsoleProviderGateway.resolve_for_send` sets
`prompt_caching` only when `identity.execution_key == "anthropic"`, so a user running
Claude **through OpenRouter** gets none of it — despite OpenRouter explicitly accepting
`cache_control` and translating between provider formats (Anthropic-style
`cache_control` ⇄ OpenAI-style `prompt_cache_breakpoint`). The capability is there; our
gate excludes it by provider name.

Three separate gaps, worth doing together:

1. **The per-turn breakpoint is gated out.** The condition keys on the execution key
   rather than on whether the *routed model* supports explicit caching. Anthropic and
   Gemini via OpenRouter both want breakpoints; OpenAI/DeepSeek/Groq/Moonshot via
   OpenRouter cache automatically and want none.

2. **No `session_id` — sticky routing is fragile.** OpenRouter keeps a conversation on
   one upstream provider to keep its cache warm, but by default identifies the
   conversation by *hashing the opening messages*. It exposes `session_id` (body or
   `x-session-id` header, ≤256 chars) for explicit control, and with a `session_id`
   stickiness engages on any successful request rather than only after a cache hit is
   observed. The Console has a stable per-session id already; not sending it leaves
   routing to a heuristic and re-routes cold whenever the sticky provider is
   unavailable.
   **Cost-ticker consequence:** a provider fallback breaks the cache with an
   *unchanged payload*, so the fingerprint break-detector cannot see it and will not
   alert. The chip still reports honestly after the fact (warm/cold comes from the last
   send's real usage), but the user gets no warning and no cause.

3. **No pricing entries for OpenRouter**, so every OpenRouter model resolves to `None`
   and the chip falls back to tokens-only. Note OpenRouter pricing is per-routed-model,
   so this likely needs the routed model reported in the response rather than a static
   table.

Verified against OpenRouter's prompt-caching guide (provider sticky routing section) on
2026-08-03. Live end-to-end confirmation was not possible: the repo-root OpenRouter key
is rejected by OpenRouter itself (`401 User not found`) — a working key is needed to
validate any fix.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The per-turn `cache_control` breakpoint is emitted for explicit-caching models routed via OpenRouter (Anthropic, Gemini), and NOT for automatic-caching ones
- [ ] #2 The gate keys on routed-model caching capability rather than a hardcoded execution-key equality check
- [ ] #3 The Console sends a stable `session_id` (or `x-session-id`) for OpenRouter requests, derived from the existing per-session id
- [ ] #4 Cache reads/writes returned by OpenRouter land in the correct disjoint buckets (depends on TASK-2120)
- [ ] #5 Pricing resolves for OpenRouter responses using the routed model reported in the response, or the tokens-only fallback is documented as intended
- [ ] #6 Validated end-to-end against a working OpenRouter key: a second same-conversation send reports a cache read
<!-- AC:END -->
