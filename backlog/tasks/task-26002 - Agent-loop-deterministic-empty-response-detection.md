---
id: TASK-26002
title: 'Agent loop: deterministic empty-response detection'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:43'
updated_date: '2026-09-01 00:24'
labels:
  - agents
  - reliability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
An empty model turn is treated as a successful finish. Verified on origin/dev: Agents/agent_runtime.py:1427 returns RUN_DONE when a turn yields no tool calls, with no check that the turn produced any text or tokens; a named grep for empty_response, empty completion and blank response across Agents/ and Chat/console_agent_bridge.py returns zero. A provider returning empty output therefore looks to the user like the agent decided it was finished. Hermes treats two consecutive zero-output-token completions from the same model, provider and finish_reason as deterministic and stops retrying rather than burning budget.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A turn producing neither visible text nor tool calls is not reported as a successful completion
- [x] #2 Two consecutive empty responses from the same provider and model stop the run with an honest message instead of retrying indefinitely
- [x] #3 A single empty response is retried, composing with task-25901's retry policy, rather than immediately failing
- [x] #4 The terminal message names the provider and model so the user can act on it
- [x] #5 Tests cover: one empty then success, two consecutive empties, and empty-text-with-tool-calls which is legitimate and must not trip
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Adds a check at an existing terminal branch; no new seam.

1. Detect at the `if not calls:` return -- that is the single place an empty turn was being reported as a finished answer.
2. Count CONSECUTIVE empties, reset by any turn that produces text or a tool call, so two blips separated by real content are not treated as a deterministic fault.
3. A tool call with no text resets the counter: that is the ordinary shape of a model deciding to call a tool, and treating it as empty would break every tool-using run.
4. Name the provider and model in the terminal message; add `AgentConfig.provider` for it, defaulted empty so the pure loop never requires it.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
An empty model turn is no longer reported as a finished answer.

**The defect.** `run_agent_loop` returned `RUN_DONE` for any turn with no tool calls, without checking it produced anything. A provider returning empty output was therefore indistinguishable, to the user, from the agent deciding it was done — the run just ended with nothing.

**One empty is a blip; two are a verdict.** A single empty turn is retried. Two consecutive empties from the same provider and model mean the fault is deterministic — the request or the model is wrong — and asking a third time spends money to reach the same place. `EMPTY_TURN_LIMIT = 2`.

**"Consecutive" is the load-bearing word.** The counter resets on any turn that produces text or a tool call, so two empties separated by real work are two blips rather than a fault. A tool call with no text explicitly resets it: that is the ordinary shape of a model deciding to call a tool, and counting it as empty would have broken every tool-using run. That case has its own test (AC#5).

**AC#4 needed a new field.** The loop knew the model but not the provider, so `AgentConfig.provider` was added — defaulted empty, because the pure loop must not require it and an unset value simply reads as "unknown provider". At the Console seam it is wired to the existing `api_endpoint` local rather than re-deriving it: that is the key the request is actually sent under, and it already carries the `execution_key -> provider -> "agent"` fallback chain.

**Verification.** 10 tests: one-empty-then-answer, two consecutive empties, whitespace-only variants, the terminal message naming provider and model, a tool call with no text not tripping, the counter resetting on intervening content, and a normal answer unaffected. Verified non-vacuous by mutation: disabling the emptiness check fails 8 of the 10.

One test was wrong on the first pass and is worth recording: it used a text-only turn as the "intervening content" that resets the counter. A turn with text and no calls IS a finished answer and ends the run, so it could never sit in the middle of a sequence. Only a tool call can. The test now says so in its docstring, since the mistake is easy to repeat.

`Tests/Agents/` holds at the same 15 baseline failures (2283 passing, up 10); `Tests/App/`, `Tests/MCP/` and `Tests/Metrics/` unchanged at the 2 known MCP baselines.

**Files:** `tldw_chatbook/Agents/agent_runtime.py`, `tldw_chatbook/Agents/agent_models.py`, `tldw_chatbook/Chat/console_agent_bridge.py`, `Tests/Agents/test_empty_response.py` (new).
<!-- SECTION:NOTES:END -->
