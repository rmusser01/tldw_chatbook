---
id: TASK-25902
title: 'Agent loop: cross-provider fallback chain'
status: To Do
assignee: []
created_date: '2026-08-31 15:08'
updated_date: '2026-08-31 22:52'
labels:
  - agents
  - reliability
dependencies:
  - TASK-25901
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
When a provider is exhausted or down, chatbook has no way to continue on another. Verified on origin/dev: the two fallback_ hits in the codebase are config-key fallbacks for model-name lookup (Chat/console_session_settings.py:725-727) and a use-Console-default in personas, neither of which is error-driven switching; a 429 retries the same key and then raises ChatRateLimitError (Chat/Chat_Functions.py:1180-1184). Builds on the retry classification from task-25901 - fallback is what happens when retry is exhausted or the error is credit/quota terminal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ordered fallback provider list is configurable and consulted when the primary provider exhausts retries or returns a credit/quota-terminal error
- [ ] #2 Switching providers mid-run is visible in the transcript and the run log, naming the failing provider, the chosen provider, and the reason - never silent
- [ ] #3 A fallback candidate the existing per-provider readiness check reports as unconfigured is skipped without an attempt, and the skip is visible
- [ ] #4 Accumulated history is projected into the target provider's protocol at the switch (ADR-110): native tool_calls/role:"tool" pairs and fence-prefixed text round-trip without losing or reordering any turn
- [ ] #5 An unpaired tool call (result never arrived) projects with an explicit no-result marker rather than being dropped
- [ ] #6 Assistant text containing a look-alike fence tag is left as text, not parsed into a tool call
- [ ] #7 A projection that cannot be performed faithfully refuses the fallback and raises the original provider's error, rather than sending a degraded history
- [ ] #8 Prompt-cache state (protocol text, provider cache breakpoints) is rebuilt for the target provider rather than carried over
- [ ] #9 Fallback does not rebuild LoopDeps - the wall-budget origin TASK-25913's tool clamp reads from is unchanged by a provider switch
- [ ] #10 With no fallback chain configured, behaviour is byte-identical to today and no projection code runs
- [ ] #11 Round-trip property tests (native->fence->native) assert semantic equivalence, driven from provider_supports_native_tools' own provider list rather than a hand-copied one
- [ ] #12 Verified against a real second provider before the task is closed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: YES - ADR-110 (backlog/decisions/110-cross-provider-fallback-and-history-projection.md), Status: Proposed.

BLOCKED on owner acceptance of ADR-110. Do not implement before then.

Why an ADR: the original acceptance criteria said "request shaping is re-resolved for the fallback provider", which understated the problem. The accumulated message HISTORY is provider-shaped - `_append_tool_result` (agent_runtime.py:836-848) writes native `role:"tool"` messages paired by tool_call_id for openai/anthropic/google/cohere, and fence-prefixed user messages for every other provider. Switching mid-run without projecting the history produces a confused model rather than a loud failure.

Sequence once accepted:
1. `project_history_for_protocol(messages, native)` as a pure function, with round-trip property tests driven from `provider_supports_native_tools`' own list.
2. Fallback selection: readiness-gated chain walk, consulted only after retry (TASK-25901) is exhausted or on a credit/quota-terminal class.
3. Switch seam inside the model-call path only - never by rebuilding LoopDeps (TASK-25913's review established why).
4. Trace step naming from-provider, to-provider, and reason.
5. Live verification against a real second provider.
<!-- SECTION:PLAN:END -->
