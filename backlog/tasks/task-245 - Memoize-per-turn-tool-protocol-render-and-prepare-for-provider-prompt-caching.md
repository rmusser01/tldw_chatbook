---
id: TASK-235
title: Memoize per-turn tool-protocol render and prepare for provider prompt-caching
status: To Do
assignee: []
created_date: '2026-07-16 16:00'
labels:
  - agents
  - console
  - performance
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
_make_call_model re-renders render_tool_protocol(runtime_schemas + active_schemas) from scratch on every provider call within a run, even though the active set is unchanged on most turns (measured: a post-load 16-tool catalog's protocol text is ~1,329 tokens, resent unchanged on every one of up to 16 turns — ~21k tokens of byte-identical text over one run). CPU cost of the render itself is negligible (measured ~81 microseconds), so this is not a CPU fix; it is the precondition for later provider-side prompt caching (e.g. Anthropic cache_control breakpoints), which no provider call in LLM_API_Calls.py currently sets.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The rendered protocol string is cached/reused across consecutive turns within one run when the active schema set (by name) has not changed since the last render
- [ ] #2 A cache invalidates correctly the moment load_tools admits a new schema into the active set
- [ ] #3 A short design note records what would be needed to additionally mark this stable prefix for provider-side prompt caching, as a distinct follow-up rather than in-scope work here
<!-- AC:END -->
