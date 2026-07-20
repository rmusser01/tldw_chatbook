---
id: TASK-321
title: Replace word/char token estimation with a real tokenizer estimate
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
labels: [llm, tech-debt]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Token counting is only accurate for OpenAI (tiktoken). For every other provider, `count_tokens_messages` falls back to whitespace word counts via `len(role.split())` / `len(content.split())` (`token_counter.py:183,188`), and `approximate_token_count` (`Chat_Functions.py:68`) is also a whitespace word count. Word counts under-estimate real tokens by ~1.3-1.5x for English and far more for code/CJK, so any budget or trim built on these numbers under-counts and still overflows the model window.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Non-OpenAI token estimates no longer use whitespace word counts (`.split()`)
- [ ] #2 A consistent estimator (provider tokenizer where available, else a chars-based estimate with documented headroom) is used everywhere token counts feed a budget or a UI limit
- [ ] #3 Tests assert the estimate is at least a conservative floor for representative code and CJK samples
<!-- AC:END -->
