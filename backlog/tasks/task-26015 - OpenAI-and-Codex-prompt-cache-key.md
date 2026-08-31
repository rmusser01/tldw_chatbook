---
id: TASK-26015
title: OpenAI and Codex prompt cache key
status: To Do
assignee: []
created_date: '2026-08-31 15:45'
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
- [ ] #1 OpenAI-family requests carry a stable cache key derived from the conversation's stable prefix
- [ ] #2 The key is stable across turns within a conversation and changes when the stable prefix genuinely changes
- [ ] #3 The key does not leak conversation content - it is a digest, not a summary, asserted by a test
- [ ] #4 Providers that do not accept the parameter are unaffected; no request is malformed by its presence
- [ ] #5 Cached-token savings appear in the existing usage accounting so the benefit is measurable rather than assumed
- [ ] #6 Disabled by config reproduces today's behavior exactly
<!-- AC:END -->
