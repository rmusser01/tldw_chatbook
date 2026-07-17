---
id: TASK-245
title: Memoize per-turn tool-protocol render and prepare for provider prompt-caching
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-16 16:00'
updated_date: '2026-07-17 01:42'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan at Docs/superpowers/plans/2026-07-17-protocol-render-memoization.md — 1 SDD task: per-run closure memo in _make_call_model keyed by schema-name tuple (fence branch only; native untouched; per-run scoping by construction), invalidates when load_tools grows the active set; + Docs/superpowers/reviews/2026-07-17-provider-prompt-caching-note.md for AC#3 (Anthropic cache_control needs structured system blocks — moot until task-263; OpenAI prefix caching needs only the byte-stability this delivers; native tools= symmetry as follow-up).
<!-- SECTION:PLAN:END -->
