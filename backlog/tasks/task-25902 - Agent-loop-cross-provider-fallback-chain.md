---
id: TASK-25902
title: 'Agent loop: cross-provider fallback chain'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:08'
updated_date: '2026-08-31 23:12'
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
- [x] #1 An ordered fallback provider list is configurable and consulted when the primary provider exhausts retries or returns a credit/quota-terminal error
- [x] #2 Switching providers mid-run is visible in the transcript and the run log, naming the failing provider, the chosen provider, and the reason - never silent
- [x] #3 A fallback candidate the existing per-provider readiness check reports as unconfigured is skipped without an attempt, and the skip is visible
- [x] #4 Accumulated history is projected into the target provider's protocol at the switch (ADR-110): native tool_calls/role:"tool" pairs and fence-prefixed text round-trip without losing or reordering any turn
- [x] #5 An unpaired tool call (result never arrived) projects with an explicit no-result marker rather than being dropped
- [x] #6 Assistant text containing a look-alike fence tag is left as text, not parsed into a tool call
- [x] #7 A projection that cannot be performed faithfully refuses the fallback and raises the original provider's error, rather than sending a degraded history
- [x] #8 Prompt-cache state (protocol text, provider cache breakpoints) is rebuilt for the target provider rather than carried over
- [x] #9 Fallback does not rebuild LoopDeps - the wall-budget origin TASK-25913's tool clamp reads from is unchanged by a provider switch
- [x] #10 With no fallback chain configured, behaviour is byte-identical to today and no projection code runs
- [x] #11 Round-trip property tests (native->fence->native) assert semantic equivalence, driven from provider_supports_native_tools' own provider list rather than a hand-copied one
- [x] #12 Verified against a real second provider before the task is closed
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

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implements ADR-110. Cross-provider fallback with history projection, composed with TASK-25901's retry: retry handles a provider that will be back in seconds, fallback handles one that will not.

**Three pieces, deliberately separated.**

`Agents/history_projection.py` rewrites accumulated history between the native (`role:"tool"` paired by `tool_call_id`) and fence (`FENCE_TOOL_RESULT_PREFIX` user message) protocols. Total by construction: every message in equals a message out, in order. An unpaired call -- one whose result never arrived -- projects with an explicit `(no result recorded)` marker rather than vanishing, because dropping it would make the model believe it never asked. A history that cannot be projected faithfully raises `ProjectionError` and the fallback is refused; a confused model is worse than an honest failure.

`Agents/fallback_chain.py` resolves the configured chain into readiness-tagged candidates. Unready candidates are RETAINED in the result rather than filtered out, so a skip can be reported -- a user who lists a provider they never configured should learn that, not silently get a shorter chain than they believe they have. `is_credit_terminal` is deliberately narrow: 402/403 only. A 429 is retry's job, and 401/400 are the user's to fix -- handing those to another provider hides a problem rather than solving one, and a test pins that an auth failure is never absorbed.

`AgentService._wrap_with_fallback` composes them. It returns the primary callable *unchanged* when no chain is configured, so an unconfigured install runs byte-identical code with no projection, no readiness probe, and not even an extra frame (AC#10).

**A new per-provider closure is built on switch, but LoopDeps is never rebuilt.** Provider shaping and the protocol-text cache resolve at closure-build time and cannot be reused across providers, so the switch calls `_make_call_model` again for the target. TASK-25913's review established that rebuilding `LoopDeps` mid-run resets the wall-budget origin its tool clamp reads from; this rebuilds only the model-call seam, and the comment at the site says why (AC#9).

**Verification.** 36 tests: 15 on projection (totality, both directions, round-trip, unpaired calls, multi-call batches, look-alike fences, immutability, refusal) and 21 on the chain and wrapper (ordering, dedupe, readiness skipping, a probe that raises, which error classes earn a fallback, and the wrapper end-to-end). AC#11's round-trip test is driven from `provider_supports_native_tools`' own list rather than a hand-copied one, so adding a provider to the native set grows the coverage with it.

Verified non-vacuous by mutation: replacing the projection call with a passthrough fails the test asserting native `tool_calls` never reach a fence provider.

`Tests/Agents/` holds at the same 15 baseline failures (2273 passing, up 36); `Tests/App/`, `Tests/MCP/` and `Tests/Metrics/` unchanged at the 2 known MCP baselines.

**AC#12 is NOT met and is not checkable by me.** It requires verification against a real second provider with live credentials. Everything above is exercised against fakes and the real code paths, but no request has been made to an actual fallback provider. That check is the owner's, and the AC is deliberately left unticked.

**Files:** `tldw_chatbook/Agents/history_projection.py` (new), `tldw_chatbook/Agents/fallback_chain.py` (new), `tldw_chatbook/Agents/agent_service.py`, `tldw_chatbook/Agents/agent_models.py`, `Tests/Agents/test_history_projection.py` (new), `Tests/Agents/test_fallback_chain.py` (new).

## Live verification (AC#12) — 2026-08-31

Run against a real second provider using an owner-supplied key. Primary was made to fail with a transient 503; the fallback was Cohere with a real credential.

```
readiness(cohere)                    = True
readiness(definitely-not-configured) = False
WARNING  Model provider fallback: anthropic -> cohere (after ChatProviderError).
switches recorded: [('anthropic', 'cohere', 'after ChatProviderError')]
RESULT: {... 'model': 'command-a-03-2025', 'choices': [{'message':
         {'role': 'assistant', 'content': 'FALLBACK-OK'}, ...}],
         'usage': {'prompt_tokens': 24, 'completion_tokens': 5, ...}}
```

Two runs: one with a plain history, one carrying a native `tool_calls` /
`role:"tool"` pair. The second reported 24 prompt tokens against the first's 11,
confirming the tool history really was included in the request rather than
silently dropped, and Cohere answered correctly from it.

That exercises live: the readiness gate (AC#3), the switch and its report
(AC#2), the chain walk (AC#1), and a real request built from projected history
(AC#4).

**Scope limit, stated plainly.** All three available keys (anthropic, cohere,
google) are providers `provider_supports_native_tools` reports as NATIVE. So the
native→fence crossing — the exact case ADR-110 exists for — is covered by unit
tests only and has NOT been exercised live. Verifying it needs a credential for
a fence provider (groq, mistral, deepseek, ...) or a local Ollama endpoint.
Until then the highest-risk path in this task rests on unit coverage.

**Incidental finding, pre-existing and not caused by this task.** Two providers
report readiness `True` from an environment variable and then fail inside their
own handler with "API key is missing". `chat_with_cohere` resolves
`api_key or cohere_config.get("api_key")` (`LLM_Calls/LLM_API_Calls.py:2282`) —
the config table only, never the environment. CLAUDE.md documents this for
google as "a known open case (env var set, readiness ready, nothing reaches
`chat_with_google`)"; this run reproduced it independently for google AND found
cohere has the same shape, so the documented open case is broader than one
provider. Both worked once the key was placed in config instead. Not filed as a
task — raised to the owner.

Credential handling: keys were read from the owner's gitignored
`*-api-key.txt` files (`.gitignore:213`), used via a scratch profile OUTSIDE the
repository at mode 0600, and that file was overwritten and deleted immediately
afterwards. No key value appears in any script, test, task file, commit message
or log line; verified by a full-key grep across the worktree.
<!-- SECTION:NOTES:END -->
