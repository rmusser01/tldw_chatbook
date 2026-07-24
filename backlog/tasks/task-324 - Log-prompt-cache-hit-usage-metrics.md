---
id: TASK-324
title: Log prompt-cache hit usage metrics across providers
status: Done
assignee: []
created_date: '2026-07-20 18:45'
labels: [llm, caching, observability]
dependencies: [task-323]
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
No code inspects cache-hit usage fields — a grep for `cached_tokens`, `cache_read_input_tokens`, and `cache_creation_input_tokens` returns nothing. There is therefore no way to confirm whether the byte-stable prefix already earns OpenAI automatic-prefix cache hits, and no baseline to measure the value of enabling Anthropic caching (task-323).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Response `usage` cache fields are read and logged for OpenAI (`prompt_tokens_details.cached_tokens`) and Anthropic (`cache_read_input_tokens` / `cache_creation_input_tokens`)
- [x] #2 The metrics are emitted alongside the existing token `log_histogram` metrics
- [x] #3 Absent/missing usage fields degrade gracefully (no crash)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Logged provider-reported prompt-cache usage fields for OpenAI and Anthropic (`LLM_API_Calls.py`). Built with task-323; measures/verifies its cache hits (task-323 AC#4).

Approach: both providers already had a non-streaming `usage` `log_histogram` block (OpenAI: prompt/completion/total; Anthropic: input/output/total). This **appends** the missing cache metrics to those existing blocks (no duplication):
- OpenAI `chat_with_openai`: `openai_api_cached_tokens` from `(usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)` — the `or {}` handles the field being absent OR `null`.
- Anthropic `chat_with_anthropic`: `anthropic_api_cache_read_input_tokens` and `anthropic_api_cache_creation_input_tokens` from `usage.get(..., 0)`.
All reads are graceful (existing `usage = response_data.get("usage", {})` / `if usage:` guard + `.get(..., 0)`), so absent/malformed usage never crashes (AC#3). Scoped to the non-streaming path (where `usage` is on the response, same as the pre-existing token metrics); streaming cache-usage (final SSE event) is a documented follow-up.

Testing (`Tests/Chat/test_cache_usage_metrics.py`, spying `log_histogram`): OpenAI `cached_tokens` present→exact value / absent→0; Anthropic cache_read/creation present→exact / absent→0, and the pre-existing input/output histograms still fire.

Executed via SDD (opus whole-branch review Ready-to-merge 0 Crit/0 Imp). Follow-up (non-gap per review): an explicit `prompt_tokens_details: None` test case.

Files: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`, `Tests/Chat/test_cache_usage_metrics.py`.
<!-- SECTION:NOTES:END -->
