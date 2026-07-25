---
id: TASK-622
title: >-
  LLM context composition 400s on llama.cpp when conversation ends with an assistant turn
status: To Do
assignee: []
created_date: '2026-07-25 10:15'
updated_date: '2026-07-25 10:15'
labels:
  - image-generation
  - bug
  - uat
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Live UAT 2026-07-25: no-prompt `/generate-image` on a llama.cpp session silently used the shallow keyword extractor instead of the task-559 LLM-composed prompt. Reproduced outside the app: `compose_llm_context_prompt` passes the conversation turns as the payload, and when the LAST turn is an assistant message (the normal case — user asks, assistant answers, user runs `/generate-image`), `chat_with_llama` treats the trailing assistant message as a response PREFILL; llama.cpp rejects it with `400: "Assistant response prefill is incompatible with enable_thinking."` The graceful fallback masks the failure by design (debug log only), so the feature quietly degrades to the keyword prompt on the app's most common local provider.

Repro: `compose_llm_context_prompt([("user", ...), ("assistant", ...)], LLMContextOptions(api_endpoint="llama_cpp", model=..., ...))` against a live llama.cpp server → 400 → returns None.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The composition payload never ends on an assistant turn: append the compose instruction as a final USER-role message (provider-agnostic fix — better prompt construction and avoids every prefill-detecting handler), or an equivalent approach that works across providers.
- [ ] With a live llama.cpp session whose last turn is an assistant message, no-prompt `/generate-image` produces a genuinely LLM-composed prompt (integration-style test with a faked chat_call asserting the payload shape; the trailing-role invariant pinned).
- [ ] Fallback behavior on real failures is unchanged (still silent, still keyword extractor).
- [ ] Existing task-559 unit-3 tests stay green.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:NOTES:END -->
