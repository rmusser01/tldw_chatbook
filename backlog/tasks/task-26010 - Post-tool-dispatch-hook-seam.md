---
id: TASK-26010
title: Post-tool-dispatch hook seam
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:44'
updated_date: '2026-09-01 06:24'
labels:
  - agents
  - tools
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
There is a pre-dispatch hook and nothing after. Verified on origin/dev: Agents/agent_runtime.py:1510 offers review_tool_calls(calls, run_id) and :1610 before_tool_dispatch, both consumed by Chat/console_chat_controller.py:1483,1661,2006 - but nothing observes a completed call. The one module that looked like a post-tool surface, Tools/file_operation_hooks.py, is dead: Tests/Tools/test_system_a_is_retired.py:73,80 pins that install_claude_code_hooks has no callers. Several later wants (usage telemetry, incident capture, verification policies) all need this seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A hook fires after a tool call completes, receiving the call, its outcome and its timing
- [x] #2 The hook is observational: raising inside it cannot fail the tool call or the run
- [x] #3 It fires for successful, failed, denied and timed-out calls, with the outcome distinguishable
- [x] #4 It fires for tool calls made by sub-agents as well as the primary run, with the owning run identifiable
- [x] #5 With no hook registered there is no measurable overhead and behavior is unchanged
- [x] #6 The dead Tools/file_operation_hooks.py is either removed or explicitly documented as retired, so it is not mistaken for this seam
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. An optional observational callback on an existing per-run closure; no new storage or protocol.

1. Home the hook on the service's per-run invoke closure -- it already carries run_id (so sub-agents attribute to their own run) and wraps the timeout path (so the observed duration covers the whole bounded call).
2. Wrap ONLY when a hook is registered; otherwise return the original closure untouched, so an unconfigured install pays nothing.
3. Review-hook denials never dispatch, but their refusal IS their completion: observe them through a wrapper on the review callable with a distinct "review_denied" outcome, so a denial-heavy run is not invisible to an observer watching dispatches.
4. Strictly observational: a raising hook is logged at debug and costs only its own observation.
5. AC#6 by documentation: the dead file_operation_hooks.py gets a RETIRED header pointing at this seam; deletion is a separate cleanup with its own import audit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `AgentService(post_tool_dispatch=...)`: an observational hook receiving `(call, result, duration_seconds, run_id)` after every tool call completes.

**Placement.** The service's per-run invoke closure, for two structural reasons: it already carries `run_id`, so sub-agents fire the hook with their own id and attribution needs no plumbing (AC#4); and it wraps `_call_with_timeout`, so the observed duration covers the whole bounded call including a timeout, not just the provider's happy path. Outcomes are distinguishable through `ToolResult` itself -- ok/error for failures, `outcome="blocked"` for gate denials, `TOOL_OUTCOME_TIMEOUT` for timeouts -- and a timed-out call's duration is asserted to cover the bound (AC#1/#3).

**Review denials are completions too.** A call the review hook refuses never dispatches, so an observer watching only dispatches would see a denial-heavy run as silent. `_wrap_review_with_observation` reports each non-proceed verdict with a synthetic result carrying `outcome="review_denied"` and duration 0.0 -- distinct from a gate denial, because nothing ran. Proceed verdicts are not observed; the dispatch itself will report.

**Zero cost unconfigured (AC#5).** With no hook, `_make_invoke_tool` returns the original closure untouched and the review callable passes through unchanged -- not a wrapper with an early-out, no wrapper at all. Verified by mutation: forcing the unwrapped branch fails 5 of the 9 tests.

**AC#6.** `Tools/file_operation_hooks.py` now opens with a RETIRED header naming its zero-caller status (pinned by `test_system_a_is_retired.py`, which still passes) and pointing at this seam. Kept on disk because deletion is a separate cleanup with its own import audit; nothing there should be extended.

**A test-infrastructure finding worth recording.** After adding this task's test file, the `Tests/Agents/` full-suite failure count dropped from the long-standing 15 to 7 -- with ZERO new failure names (verified by diffing sorted names, as every task in this lane has). The 8 that flipped pass in isolation at the review base too: they are ordering/pollution-sensitive members of the "baseline 15", and new test files shifted collection order under `-p no:randomly`. The stable baseline is therefore at most 7; the other 8 are flaky-by-ordering, which the repo's baseline bookkeeping should not treat as fixed OR as regressions when they flip.

**Files:** `tldw_chatbook/Agents/agent_service.py`, `tldw_chatbook/Tools/file_operation_hooks.py` (retired header), `Tests/Agents/test_post_tool_dispatch_hook.py` (new).

## Review round (2026-08-31)

**I-5:** the hook did not fire for SKILL-tool calls — the outer invoke wrapper's skill branch returns before the observed registry closure, so the "fires after every tool call completes" claim was quietly false for exactly the calls users script. All three skill-branch exits (not-permitted, sub-agent budget, and the run itself) now report through the same `_fire_post_tool_dispatch`, with their own timing. Scope stated precisely: spawn/wait/send_to_agent runtime tools remain unobserved — they were never registry dispatches, and observing them is a different seam.

The reviewer confirmed the verdict-keying in `_wrap_review_with_observation` matches `_effective_review_verdict`'s real precedence (the wrapper receives correlation-stamped calls), and that `invoke_tool_at_step` routes through the outer wrapper without a signature break.
<!-- SECTION:NOTES:END -->
