---
id: TASK-26015
title: OpenAI and Codex prompt cache key
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 18:33'
labels:
  - console
  - context
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Caching is Anthropic-only. Verified on origin/dev: a named grep for prompt_cache_key, cachedContent and implicit_cach across tldw_chatbook returns zero, so OpenAI-family requests get whatever implicit caching the provider infers from the raw prefix and nothing stabilizes it across turns. Hermes derives a content-addressed cache key with a rotation-stable scope. Chatbook already normalizes cached-token usage on the read side (Chat/provider_usage.py:235,285-286 handles OpenAI cached_tokens), so the accounting half exists and only the request half is missing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 OpenAI-family requests carry a stable cache key derived from the conversation's stable prefix
- [x] #2 The key is stable across turns within a conversation and changes when the stable prefix genuinely changes
- [x] #3 The key does not leak conversation content - it is a digest, not a summary, asserted by a test
- [x] #4 Providers that do not accept the parameter are unaffected; no request is malformed by its presence
- [x] #5 Cached-token savings appear in the existing usage accounting so the benefit is measurable rather than assumed
- [x] #6 Disabled by config reproduces today's behavior exactly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: digest stability/prefix-sensitivity/no-content-leak; flows into payload when enabled; absent by default\n2. _openai_prompt_cache_key (sha256 of {system, tools}, 'tldw-'+32hex) + _openai_cache_key_enabled gate\n3. payload['prompt_cache_key'] under the gate; config sample
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
_openai_prompt_cache_key(system_message, tools) = 'tldw-' + sha256({system, tools} canonical JSON)[:32]: digests only the STABLE prefix (system + tool list), so it is identical across a conversation's turns (AC#2, the turn messages are excluded) and changes when the prefix changes (pinned both ways); a SHA-256 digest never the text (AC#3, pinned: system words absent, <=64 chars). Added to the OpenAI chat/responses payload as prompt_cache_key under [caching] openai_cache_key (default false = today's payload exactly, AC#6 pinned; fail-safe OFF on read error). NOT conversation-scoped by design — two conversations sharing a system prompt + tools SHOULD share a cache node (rotation-stable scope). AC#4: OpenAI ignores an unrecognized field and the value is a plain ASCII string, so no request is malformed (applies to both the chat-completions and responses branches — payload is shared). AC#5: cached-token savings already surface via provider_usage's prompt_tokens_details.cached_tokens read path (unchanged) — the request half was the only gap, now closed. Only chat_with_openai (the OpenAI-family bridge) sets it; other providers untouched. 3 new tests; caching suite 14 passed.
<!-- SECTION:NOTES:END -->
