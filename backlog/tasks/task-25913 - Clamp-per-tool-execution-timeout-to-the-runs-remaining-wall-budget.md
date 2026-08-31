---
id: TASK-25913
title: Clamp per-tool execution timeout to the run's remaining wall budget
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:10'
updated_date: '2026-08-31 17:55'
labels:
  - agents
  - defect
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A single tool call can run past the run's wall-clock budget. Verified on origin/dev: max_wall_seconds is checked only at the top of the loop (Agents/agent_runtime.py:1175), while _call_with_timeout(fn, seconds, ...) at Agents/agent_service.py:1522 takes an absolute bound with no reference to time already spent. The engine default is 300s (Agents/agent_models.py:400) but Console raises it to 3600s (Chat/console_agent_bridge.py:408), so a hung tool can hold a run roughly an hour past a budget the user set. Found while verifying an area agent's claim during the 2026-08-31 parity pass; this is a defect, not a parity gap - hermes is not the reason to fix it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A tool call's effective timeout is the lesser of the configured per-tool timeout and the run's remaining wall budget
- [x] #2 A call cut short by the remaining-budget clamp is reported distinctly from one that hit the per-tool ceiling, so the cause is legible
- [x] #3 Human-approval waits continue to pause the deadline as they do today (ADR-067 refcounted marks) and are not counted against the clamp
- [x] #4 A run with no wall budget configured behaves exactly as today
- [x] #5 A test asserts a long tool call cannot push a run past max_wall_seconds
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Tightens an existing bound; no new seam or dependency.

1. Extract the arithmetic as a pure helper so the interesting cases (no budget, exhausted budget, which side wins) are testable without an AgentService.
2. Resolve the clamp ONCE at dispatch, not continuously -- an approval wait that happens afterwards must not shrink a call already running, which is what ADR-067's refcounted marks protect.
3. Return a small positive floor when the budget is already spent: the dispatch site treats a falsy timeout as "run unbounded", so zero would produce the opposite of the intended behaviour.
4. Give the clamped stop its own message; "timed out" and "the run is nearly over" are different facts for whoever reads the transcript.
5. Capture the run start where the dispatch closure is built -- marginally earlier than the loop's own `started`, which makes the clamp conservative rather than permissive.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
A tool call is now bounded by the lesser of its own ceiling and the run's remaining wall budget.

**The defect.** `max_wall_seconds` was checked only between loop iterations, while `_call_with_timeout` took an absolute per-call bound. The engine default is 300s but Console raises it to 3600s, so one hung tool could hold a run roughly an hour past a budget the user set -- the loop simply never got back to its own check.

**Resolved once, at dispatch.** `_effective_tool_timeout` computes the bound when the call is dispatched, not continuously. That is deliberate: a human approval wait occurring afterwards must not shrink a call that is already running, which is exactly what ADR-067's refcounted marks protect inside `_call_with_timeout`. A test drives a call whose work outlasts its clamped bound while `pauses_deadline` is active and asserts it still succeeds.

**Zero would have inverted the fix.** The dispatch site reads `if timeout and timeout > 0`, treating a falsy value as "run unbounded". An already-exhausted budget therefore returns a small positive floor rather than 0 -- returning zero would have made the worst case unbounded, the precise opposite of the intent.

**Distinct reporting (AC#2).** A clamped stop reads "tool call stopped after Ns: <tool> (the run's wall-clock budget was about to expire)" rather than "timed out". Those are different facts: one says the tool is slow, the other says the run is nearly over, and they lead a reader to different actions.

**Run start** is captured where the dispatch closure is built, immediately before the loop records its own `started`. Being marginally early makes the remaining-budget calculation conservative, which is the safe direction for a bound.

**Test scope, stated honestly.** Eight tests: the helper's cases (either side winning, no budget configured, exhausted budget, the AC#5 end-to-end property), distinct reporting, and the ADR-067 pause interaction. The three-line wiring into `_make_invoke_tool` is pinned by source assertion rather than a live `AgentService`, using the same approach `Tests/Agents/test_mcp_refusal_provenance.py` already uses in this repo -- constructing an AgentService here would need substantially more scaffolding than the code under test, and the failure mode that matters (helper exists, nothing calls it) is what the pin catches.

**Verification.** `Tests/Agents/` shows the same 15 baseline failures before and after, verified by diffing sorted failure names (2205 passing, up 8). The budget-configuration consumers in `Tests/Chat/test_console_agent_run_budget.py` and `Tests/UI/test_settings_agent_run_budget.py` pass unchanged.

**Files:** `tldw_chatbook/Agents/agent_service.py`, `Tests/Agents/test_tool_timeout_wall_clamp.py` (new).
<!-- SECTION:NOTES:END -->
