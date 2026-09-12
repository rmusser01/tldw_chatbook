---
id: TASK-26014
title: 'Anthropic prompt cache: 1-hour TTL tier'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 18:31'
labels:
  - console
  - context
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every cache breakpoint uses the 5-minute default, so a user returning to a conversation after a coffee break re-pays the full prefix. Verified on origin/dev: all three cache_control sites emit a bare {"type": "ephemeral"} marker - LLM_Calls/LLM_API_Calls.py:1466, :1511, :1544 - and a grep for "ttl" or 1h in that file returns zero. Anthropic supports a 1-hour tier. Chatbook already has the surrounding safety: a capability gate, a [caching] kill switch, and a degrade-retry that strips breakpoints on a 400 (:1597-1615), so the risk of adding a tier is bounded.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Cache breakpoints can emit a 1-hour TTL where the provider and model support it
- [x] #2 The tier is configurable, and the default is stated explicitly with its cost reasoning in the task notes
- [x] #3 A model or route that does not support the longer tier silently falls back to the 5-minute marker rather than erroring
- [x] #4 The existing degrade-retry still strips all cache_control on a 400 mentioning it, unchanged
- [x] #5 Cache read and creation token accounting continues to report correctly with the longer tier - verified against Chat/provider_usage.py:235,285-286
- [x] #6 The [caching] kill switch disables the longer tier along with everything else
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: marker helper (5m default / 1h configured+supported / junk+unsupported fall back), 1h flows into payload + beta header, 5m adds no header\n2. _cache_control_marker(model) reads [caching] cache_ttl, fail-safe to 5m; _anthropic_supports_1h_ttl gate\n3. Replace the 3 literal markers; add anthropic-beta extended-cache-ttl header iff a 1h marker is present (_contains_extended_ttl)\n4. Config sample
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
All three cache_control sites now emit _cache_control_marker(current_model) instead of a literal {type:ephemeral}: reads [caching] cache_ttl ('5m' default = bare marker, byte-identical to before, pinned; '1h' = {type:ephemeral, ttl:'1h'} only when _anthropic_supports_1h_ttl; unsupported model / unrecognized value / config-read failure all fall back to 5m — AC#3). The anthropic-beta: extended-cache-ttl-2025-04-11 header is added iff a 1h marker actually reached the payload (_contains_extended_ttl walk) — never on the 5m default (pinned both directions). DEFAULT + COST (AC#2): default 5m; a 1h cache WRITE bills ~2x input vs 5m's 1.25x, so 1h wins only when returns routinely exceed the 5m window (coffee-break case) — conservative default keeps today's economics. AC#4: degrade-retry is untouched — _without_cache_control strips the whole cache_control dict (ttl included) recursively, and the 400-retry still fires on any cache_control mention (existing tests green). AC#5: provider_usage reads response fields cache_creation_input_tokens/cache_read_input_tokens, which Anthropic reports identically across TTL tiers (the per-TTL split lives in a sub-object; the totals this code reads are unchanged) — accounting is TTL-agnostic by construction, no code change. AC#6: _cache_control_marker only ever runs under caching_active (= _anthropic_supports_caching AND _anthropic_caching_enabled), so the [caching] kill switch disables the 1h tier with everything else. 5 new tests; caching+prefix+mocked-API suites 41 passed. Note: the beta header is belt-and-suspenders — the 1h tier graduated to GA, but sending the opt-in header is harmless and safe against older API versions.
<!-- SECTION:NOTES:END -->
