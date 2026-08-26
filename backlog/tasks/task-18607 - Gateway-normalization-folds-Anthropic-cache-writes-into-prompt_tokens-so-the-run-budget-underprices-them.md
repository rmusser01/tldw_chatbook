---
id: TASK-18607
title: >-
  Gateway normalization folds Anthropic cache writes into prompt_tokens, so
  the run budget underprices them
status: Done
assignee:
  - '@Robert'
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
- [x] #1 The gateway's split-usage normalization preserves the cache-write bucket (or an equivalent marker) so `_budget_weighted_tokens` can price writes at their published rate on the Console send path.
- [x] #2 A streamed Anthropic turn with `cache_creation_input_tokens > 0` consumes budget at the 1.25x write rate, matching the native-shape accounting.
- [x] #3 Consumers of the normalized OpenAI-shaped usage that cannot see a write bucket (cost ticker, persistence) keep their current totals -- the raw `prompt_tokens` sum may not change under whatever representation is chosen.
- [x] #4 The two shapes (native Anthropic envelope vs gateway-normalized) produce identical weighted totals for identical provider numbers.
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: unit test — chat-completions-shaped usage with a
   `prompt_tokens_details.cache_creation_tokens` key parses into
   `ProviderUsage.cache_write` (uncached excludes it); parity test — a native
   Anthropic split payload run through `_openai_usage_from_provider_call`
   then re-parsed yields identical buckets to parsing the native payload
   directly, with top-level `prompt_tokens` unchanged (AC#3/#4); flip the
   pinning test so the gateway shape budgets writes at 1.25x (AC#2).
2. GREEN: emit `cache_creation_tokens` in the bridge's normalized
   `prompt_tokens_details`; read it in `from_provider_payload`'s
   chat-completions branch; add the key to the strict-count validator.
3. Update the two known-gap comment blocks (agent_service docstring, bridge
   test comment) that document the fold.
4. Run touched suites; PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Two-hunk fix, TDD (all three tests watched failing first):

- `console_agent_bridge._openai_usage_from_provider_call` now also emits the
  write bucket as `prompt_tokens_details.cache_creation_tokens` (an
  extension key of ours, mirroring the `cached_tokens` emit pattern);
  top-level `prompt_tokens`/`total_tokens` are byte-identical to before, so
  flat-sum readers (cost ticker, persistence) are untouched (AC#3).
- `ProviderUsage.from_provider_payload`'s chat-completions branch reads the
  key into `cache_write` and subtracts it from `uncached_input`, so
  `_budget_weighted_tokens` prices writes at their published rate (1.25x on
  Anthropic) with NO change to the budget code itself (AC#1/#2).
- `cache_creation_tokens` added to `_BUDGET_USAGE_DETAIL_COUNT_KEYS` so the
  strict validator vets the key we now emit on round-trips.
- Tests: unit (details key -> cache_write bucket), round-trip parity
  (native vs normalized -> identical ProviderUsage, AC#4), and the
  end-to-end pinning test's 111-write case flipped 5,075 -> 5,103
  (ceil of 111 x 1.25 on claude-sonnet-4-6 catalog rates). Stale known-gap
  comment blocks in agent_service and the bridge test updated.
- 565 tests green across provider_usage / console_agent_bridge /
  console_agent_run_budget / console_provider_gateway / agent_service.

No second fold site exists: the gateway's stream signals store RAW provider
payloads and every consumer normalizes via `ProviderUsage`, which handles
both envelope shapes.
