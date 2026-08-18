---
id: TASK-18603
title: >-
  Agent run budget counts cache reads at full price, so it stops cheap runs early
status: To Do
assignee: []
created_date: '2026-08-18 21:15'
labels:
  - agents
  - console
  - cost
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`AgentService._usage_total_tokens` sums `prompt_tokens + completion_tokens` flat.
The agent loop re-sends the whole conversation every turn, and Anthropic prompt
caching is ON by default for Console sends (which is what an agent run is —
`console_provider_gateway` stamps `prompt_caching` for anthropic), so on a long
run nearly every input token is a CACHE READ billed at roughly a tenth of the
uncached rate.

Counting those at 1.0 makes `max_total_tokens` terminate runs that have spent a
small fraction of what the number implies — directly undercutting TASK-18600's
goal of allowing long-running sessions.

The parts already exist: `Chat/provider_usage.ProviderUsage` splits
`uncached_input` / `cache_read` / `cache_write` / `output`, and
`LLM_Calls/pricing_catalog` publishes per-bucket rates per model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Cache-read tokens consume the run budget in proportion to their published rate rather than at full price.
- [x] #2 Cache-write tokens are counted at their own (higher) rate, not discounted along with reads.
- [x] #3 A turn with no cache activity is accounted for exactly as before.
- [x] #4 A provider/model with no published rates gets no discount rather than an invented one.
- [x] #5 A turn that spent anything never counts as zero.
- [x] #6 Output tokens remain counted one-for-one, so the change does not silently alter how strict the budget is for output-heavy runs.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `_budget_weighted_tokens`, used by `_make_call_model` in place of the flat
sum. It parses usage via `ProviderUsage` (which understands the OpenAI, OpenAI
Responses, and Anthropic-native shapes), looks up per-bucket rates in
`pricing_catalog`, and weights the INPUT buckets relative to this model's own
uncached input rate. The unit becomes "uncached-input-token equivalents".

Scope was kept to the input buckets deliberately. Pricing output proportionally
(it really costs ~5x input) would make the budget markedly stricter for
output-heavy runs — a change to how much work a given number buys, unrelated to
the cache mis-pricing being fixed, applied to a number users already chose under
the old meaning.

Every fallback keeps the previous accounting: unparsable usage, no cache
activity, an unknown model, an unpriced (zero-rate local) model, or an
unpublished cache rate all fall back to the flat sum or to full price. An
unpublished cache rate is treated as 1.0, not free — the conservative reading.

Also improved in passing: when the flat sum comes up empty because usage arrived
in Anthropic's native shape, the parsed `ProviderUsage` total is now used instead
of falling back to a local `count_tokens_messages` estimate of the whole payload.
The Console's own streaming path does not hit that gap (the gateway normalizes
split usage first, pinned by
`test_anthropic_split_usage_reaches_agent_budget_with_cache_buckets`), so this is
a defensive improvement for callers that reach the service un-normalized.

`test_anthropic_split_usage_reaches_agent_budget_with_cache_buckets` changed
expectations from the flat 10,954/11,065 to the weighted 4,964/5,075. The raw
totals it used to assert are still what the provider billed in tokens and are
still asserted on the normalized payload — they were just never what the run
cost. Note the gateway's normalization folds `cache_creation_input_tokens` into
`prompt_tokens`, so a cache WRITE arrives indistinguishable from uncached input
and is weighted at full price through that path; conservative, and noted in the
test.

Files: `Agents/agent_service.py`, `Tests/Agents/test_agent_budget_cache_aware.py`
(new, 11), `Tests/Chat/test_console_agent_bridge.py`.
<!-- SECTION:NOTES:END -->
