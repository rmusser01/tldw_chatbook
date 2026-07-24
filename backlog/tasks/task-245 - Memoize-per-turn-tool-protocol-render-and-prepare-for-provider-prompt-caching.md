---
id: TASK-245
title: Memoize per-turn tool-protocol render and prepare for provider prompt-caching
status: Done
assignee:
  - '@claude'
created_date: '2026-07-16 16:00'
updated_date: '2026-07-17 01:51'
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
- [x] #1 The rendered protocol string is cached/reused across consecutive turns within one run when the active schema set (by name) has not changed since the last render
- [x] #2 A cache invalidates correctly the moment load_tools admits a new schema into the active set
- [x] #3 A short design note records what would be needed to additionally mark this stable prefix for provider-side prompt caching, as a distinct follow-up rather than in-scope work here
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Plan at Docs/superpowers/plans/2026-07-17-protocol-render-memoization.md — 1 SDD task: per-run closure memo in _make_call_model keyed by schema-name tuple (fence branch only; native untouched; per-run scoping by construction), invalidates when load_tools grows the active set; + Docs/superpowers/reviews/2026-07-17-provider-prompt-caching-note.md for AC#3 (Anthropic cache_control needs structured system blocks — moot until task-263; OpenAI prefix caching needs only the byte-stability this delivers; native tools= symmetry as follow-up).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Per-run closure memo in AgentService._make_call_model (fence branch only; native untouched): the protocol renders once per active-set change, keyed by the schema-name tuple over runtime+active schemas — safe because ToolSchema is frozen and the loop's active list only ever grows monotonically via load_tools with both across-round and in-batch name dedupe (no in-place replacement, no reordering), so a name tuple uniquely identifies content within a run. Invalidation is exactly the load_tools-admits moment (AC#2 test: pre-load system content lacks the tool, post-load contains it, render count == 2). Byte-identity structural: the memo returns the same string render_tool_protocol would produce (AC#1 test also pins payload byte-stability across turns). AC#3: Docs/superpowers/reviews/2026-07-17-provider-prompt-caching-note.md — Anthropic cache_control needs structured system blocks (moot until task-263), OpenAI implicit prefix caching needs only the byte-stability + append-only history this run already has, native tools= memo symmetry as follow-up, measure cached_tokens before investing. Reviewer RED-verified both tests against the pre-implementation tree. Modified: agent_service.py, test_agent_service.py, + note doc.
<!-- SECTION:NOTES:END -->
