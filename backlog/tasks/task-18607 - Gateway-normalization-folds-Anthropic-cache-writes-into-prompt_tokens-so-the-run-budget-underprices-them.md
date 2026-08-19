---
id: TASK-18607
title: >-
  Gateway normalization folds Anthropic cache writes into prompt_tokens, so
  the run budget underprices them
status: To Do
assignee: []
created_date: '2026-08-19 03:30'
labels:
  - agents
  - console
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found reviewing PR #1824 (TASK-18603, cache-aware budget weighting).
`_budget_weighted_tokens` weights a cache WRITE at its real 1.25x input
rate when the usage arrives in Anthropic's native shape -- but the Console's
streaming gateway normalizes split Anthropic usage into the OpenAI shape
first, and that normalization folds `cache_creation_input_tokens` into
`prompt_tokens` while reporting only `cached_tokens` separately. Through
that path a cache write is indistinguishable from uncached input and is
weighted at 1.0x.

For a budget this is the permissive direction, not the conservative one: a
write-heavy run consumes less budget than it truly costs and can overshoot
the user's spend ceiling by roughly 25% of its write portion. It is also
inconsistent -- the same run is accounted differently depending on which
envelope shape reaches the service.

The Console send path (the production path for agent runs) always goes
through the gateway, so the correctly-weighted native path is currently
defensive-only.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The gateway's split-usage normalization preserves the cache-write bucket (or an equivalent marker) so `_budget_weighted_tokens` can price writes at their published rate on the Console send path.
- [ ] #2 A streamed Anthropic turn with `cache_creation_input_tokens > 0` consumes budget at the 1.25x write rate, matching the native-shape accounting.
- [ ] #3 Consumers of the normalized OpenAI-shaped usage that cannot see a write bucket (cost ticker, persistence) keep their current totals -- the raw `prompt_tokens` sum may not change under whatever representation is chosen.
- [ ] #4 The two shapes (native Anthropic envelope vs gateway-normalized) produce identical weighted totals for identical provider numbers.
<!-- AC:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Not fixed in PR #1824 deliberately: changing what the gateway emits is a
wire-shape change with multiple readers (cost ticker, usage persistence,
`tagged_visual_memory_message`-style integrity checks are NOT involved but
the cost ticker and DB rows are), and it deserved its own task rather than
riding a budget PR. The PR documents the gap at
`agent_service._budget_weighted_tokens` and in
`test_anthropic_split_usage_reaches_agent_budget_with_cache_buckets`.
<!-- SECTION:NOTES:END -->
