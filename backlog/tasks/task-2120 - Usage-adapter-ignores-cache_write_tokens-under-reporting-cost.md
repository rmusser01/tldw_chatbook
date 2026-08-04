---
id: TASK-2120
title: >-
  Usage adapter ignores cache_write_tokens, under-reporting cost on cache writes
status: To Do
assignee: []
created_date: '2026-08-03 19:05'
labels:
  - cost-ticker
  - llm-calls
  - correctness
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ProviderUsage.from_provider_payload` maps cache writes only on the Anthropic-native
branch (`cache_creation_input_tokens`). Both OpenAI-shaped branches read
`prompt_tokens_details.cached_tokens` / `input_tokens_details.cached_tokens` into
`cache_read` and leave `cache_write` at zero — there is no write field in either path.

Per OpenRouter's prompt-caching docs, its usage object reports **`cache_write_tokens`**
alongside `cached_tokens` inside `prompt_tokens_details`, and OpenAI's own explicit
caching (GPT-5.6+) also bills cache writes at **1.25x** input. Because our adapter
computes `uncached_input = prompt_tokens - cached_tokens`, any write tokens fall into
the `uncached_input` bucket and are priced at the **1x input rate instead of 1.25x**.

Effect: the cost ticker under-reports spend on exactly the turns that cost the most —
the cache-write turns. It is a silent undercount, not a visible error, which is the
worst shape for a number users are meant to trust. It also makes the chip's cache
accounting inconsistent across providers: identical caching behavior is fully priced on
Anthropic and partially priced everywhere else.

Also unused: OpenRouter's `cache_discount` field, which reports realized savings
directly and could corroborate our computed figure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 Both OpenAI-shaped adapter branches read a cache-write field (`cache_write_tokens`, plus any provider-specific spelling verified against real payloads) into `cache_write`
- [ ] #2 `uncached_input` excludes write tokens so the four buckets stay disjoint and do not double-count
- [ ] #3 Unit tests cover a payload carrying cached + write + prompt tokens together, asserting exact bucket values
- [ ] #4 Pricing applies the cache-write rate to those tokens, verified against a seeded model with a non-null `cache_write_per_mtok`
- [ ] #5 Providers that report no write field are unaffected (regression pinned)
<!-- AC:END -->
