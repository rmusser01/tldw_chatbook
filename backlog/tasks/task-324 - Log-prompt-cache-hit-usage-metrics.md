---
id: TASK-324
title: Log prompt-cache hit usage metrics across providers
status: To Do
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
- [ ] #1 Response `usage` cache fields are read and logged for OpenAI (`prompt_tokens_details.cached_tokens`) and Anthropic (`cache_read_input_tokens` / `cache_creation_input_tokens`)
- [ ] #2 The metrics are emitted alongside the existing token `log_histogram` metrics
- [ ] #3 Absent/missing usage fields degrade gracefully (no crash)
<!-- AC:END -->
