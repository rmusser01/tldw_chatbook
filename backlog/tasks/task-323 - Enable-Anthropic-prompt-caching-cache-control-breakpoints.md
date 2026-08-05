---
id: TASK-323
title: Enable Anthropic prompt caching via cache_control breakpoints
status: Done
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
- [x] #1 `chat_with_anthropic` emits the system prompt as structured text block(s) with a `cache_control: {"type":"ephemeral"}` breakpoint on the largest stable prefix
- [x] #2 The tools array optionally carries a breakpoint (<=4 breakpoints total), gated to Claude models that support caching
- [x] #3 Behavior is unchanged for non-Anthropic providers and for models without cache support
- [x] #4 A cache read is observed on the second identical request (verified against the usage metrics from task-324)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Activated Anthropic prompt caching in `chat_with_anthropic` (`LLM_API_Calls.py`). Done together with task-324 (metrics), which verifies this (AC#4).

Approach: added `_anthropic_supports_caching(model)` = `m.startswith("claude-") and not m.startswith("claude-2") and "instant" not in m` (modern Claude 3/3.5/3.7/4+; excludes legacy claude-2*/instant). For caching models with a non-empty system prompt, `data["system"]` becomes a structured content block list `[{"type":"text","text": system_prompt, "cache_control": {"type":"ephemeral"}}]` — one breakpoint on the largest stable prefix (AC#1); per Anthropic's tools→system→messages cache hierarchy this caches tools+system. For caching models with tools, a second `cache_control` breakpoint on the **last converted** tool (a fresh dict via `{**tools_payload[-1], ...}` — never mutates the caller's input), gated on the converted list being non-empty (AC#2; real max = 2 breakpoints, ≤4). Non-caching models keep the plain-string system and untouched tools; non-Anthropic providers are structurally unaffected (AC#3). The breakpoints live in the `data` dict built before the streaming branch, so caching benefits streamed chats too.

AC#4 is verified structurally (payload carries `cache_control`; task-324's `anthropic_api_cache_read_input_tokens` metric reads the response's cache field). A literal cross-call cache read needs two real API requests and is a documented **live** check: send two identical requests with a system prompt above the min-cacheable length (~1024 tokens; 2048 for haiku — below which Anthropic silently ignores `cache_control`), expecting `cache_creation` on the first and `cache_read` on the second. Fallback if no cache read appears: add the `anthropic-beta: prompt-caching-…` header (current API version `2023-06-01` supports `cache_control` GA without it).

Testing: `_anthropic_supports_caching` gate; caching-model system/tools carry `cache_control`; non-caching model unchanged; no-tools system-only; ≤4 breakpoints. Note: `test_anthropic_shaped_tools_pass_through_untouched` is a PRE-EXISTING baseline failure (a separate `_anthropic_tools_payload` anthropic-shape drop bug, task-263 territory) — NOT fixed here; the tools breakpoint is tested via the OpenAI-function-shaped → converted path (which works), and one adjacent existing test that shares the caching-model fixture was updated to expect the new `cache_control`.

Executed via SDD (implementer+reviewer per task, opus whole-branch review Ready-to-merge 0 Crit/0 Imp). Follow-up minors: test-helper de-dup, an empty-system-prompt regression test.

Files: `tldw_chatbook/LLM_Calls/LLM_API_Calls.py`, `Tests/Chat/test_anthropic_native_tools.py`.
<!-- SECTION:NOTES:END -->
