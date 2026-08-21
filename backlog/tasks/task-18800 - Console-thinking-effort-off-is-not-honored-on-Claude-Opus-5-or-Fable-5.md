---
id: TASK-18800
title: Console thinking-effort 'off' is not honored on Claude Opus 5 or Fable 5
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-18 23:52'
updated_date: '2026-08-21 04:35'
labels:
  - llm
  - console
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Setting the Console thinking effort to 'off' does not turn thinking off for Claude Opus 5, Fable 5 or Mythos 5. On those models thinking is ON by default when the 'thinking' parameter is omitted, and the Anthropic request builder only emits an explicit thinking={"type": "disabled"} config for the Claude Sonnet 5 family (_anthropic_is_sonnet_5 in LLM_Calls/LLM_API_Calls.py). For every other model the 'off' branch returns no thinking config at all, which on Sonnet 4.5 and earlier genuinely means no thinking but on Opus 5 / Fable 5 silently leaves adaptive thinking running and billed.

Found while fixing TASK-18414 and deliberately left out of scope there: that task's acceptance criteria cover only the two request-validity gates (sampling parameters and a fixed thinking budget), and its fix makes the 'off' path send a valid request either way. This is a third, separate capability -- 'does this model think by default, so that turning it off requires an explicit disabled config' -- and it is not uniform across the family: an explicit thinking={"type": "disabled"} is accepted on Opus 5 only at effort 'high' or lower and is rejected outright with a 400 on Fable 5 and Mythos 5, which require the parameter to be omitted instead. Naively widening the existing Sonnet 5 branch would therefore introduce a new 400 on Fable 5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting thinking effort 'off' on Claude Opus 5 produces a request that actually disables thinking
- [ ] #2 Selecting thinking effort 'off' on Claude Fable 5 or Mythos 5 does not produce an HTTP 400
- [ ] #3 The by-default-thinking decision is a capability predicate alongside the two added in TASK-18414, not a new name check in the request builder
- [ ] #4 Models where omitting the thinking parameter already means no thinking (Opus 4.8 and earlier, Sonnet 4.6 and earlier, Haiku) are unchanged, pinned by a regression test
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Standalone curl probes pinning the capability surface (opus-5 disabled accepted; fable-5 disabled 400; disabled-observable via content block types + thinking_tokens)\n2. Commit probe verdicts (session-limit insurance)\n3. Add thinks-by-default / rejects-disabled-thinking capability predicates in model_capabilities.py (outside config tables, family-matched)\n4. Rewire _anthropic_thinking_config off branch to consult them; keep _anthropic_is_sonnet_5 for the Sonnet 5 effort shape only\n5. Red-first pins, mutation tests with Edit-based restores\n6. Live verification at the chat_api_call seam + origin/dev control\n7. Hygiene: ACs, notes, Done, report, PR
<!-- SECTION:PLAN:END -->
