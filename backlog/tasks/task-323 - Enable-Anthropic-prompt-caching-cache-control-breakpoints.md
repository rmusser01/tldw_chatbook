---
id: TASK-323
title: Enable Anthropic prompt caching via cache_control breakpoints
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
labels: [llm, caching, performance]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Anthropic prompt caching is never activated. `chat_with_anthropic` sends the system prompt as a plain string (`LLM_API_Calls.py:906`) and tools as a plain list (`LLM_API_Calls.py:924`), with no `cache_control` breakpoints anywhere in the LLM path. The request prefix is already byte-stable (audit confirmed no cache-busting: no dynamic date injection, RAG/dictionary substitution only touch the final user message, deterministic tool ordering), so this is a pure missed cost/latency win that applies to ordinary chats today, not just agent runs. Prior analysis in `Docs/superpowers/reviews/2026-07-17-provider-prompt-caching-note.md` (task-245) reached the same conclusion.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `chat_with_anthropic` emits the system prompt as structured text block(s) with a `cache_control: {"type":"ephemeral"}` breakpoint on the largest stable prefix
- [ ] #2 The tools array optionally carries a breakpoint (<=4 breakpoints total), gated to Claude models that support caching
- [ ] #3 Behavior is unchanged for non-Anthropic providers and for models without cache support
- [ ] #4 A cache read is observed on the second identical request (verified against the usage metrics from task-324)
<!-- AC:END -->
