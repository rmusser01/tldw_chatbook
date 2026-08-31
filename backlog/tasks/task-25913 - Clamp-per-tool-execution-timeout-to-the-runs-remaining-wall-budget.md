---
id: TASK-25913
title: Clamp per-tool execution timeout to the run's remaining wall budget
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:10'
updated_date: '2026-08-31 19:15'
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

## Review round — a false claim in these notes, and a worse-than-the-bug fix

**The 0.05s floor was wrong, and the reasoning in the plan was wrong with it.** Returning a tiny positive bound for an exhausted budget still *starts* the tool on a daemon thread and abandons it 50 ms later. `_call_with_timeout`'s own docstring warns that an abandoned worker "may still complete and act for real" after a timeout is reported — so for a tool that writes files or spends money, dispatch-and-abandon is worse than the overrun this clamp exists to prevent. The guard at `if timeout and timeout > 0` was real, but the answer was a third branch, not a magic float. `_effective_tool_timeout` now returns `None` for "do not dispatch" and the call site refuses without running anything.

**A claim in these notes was false.** They stated: "A test drives a call whose work outlasts its clamped bound while `pauses_deadline` is active and asserts it still succeeds." It did not. The work completed in about a millisecond, comfortably inside the bound, and the test passed with `pauses_deadline` returning False — it exercised nothing about ADR-067. The work now genuinely outlasts the bound, and a companion test with the pause disabled asserts the same call IS stopped; that pair is what stops the first one going vacuous again.

**AC#5 was asserted by a tautology.** The old test re-derived `remaining = budget - elapsed` from hand-picked numbers and asserted the arithmetic it had just performed. It now asserts the property that matters across several dispatch points: whatever bound comes back, running for it cannot end after the budget does.

**Nothing proved the clamp reached a real call.** The source-text pin passed even if the computed value were discarded. Replaced with an integration test that builds a real `AgentService` with an injected clock, drives the real `_make_invoke_tool` closure, and asserts the bound `_call_with_timeout` actually received (60.0 from the remaining budget, not the 3600 ceiling). Verified non-vacuous by mutation: discarding the clamped value makes it fail, restoring it makes it pass. The source-pin test was deleted as redundant.

**Carried forward for the next lane:** `run_started` is a second, independent clock reading taken in `_make_invoke_tool`. Any future path that rebuilds `LoopDeps` mid-run — a retry that reconstructs deps, a fallback provider swap, both upcoming in this lane — would silently reset the clamp's origin and make the bound permissive again, with no test to catch it. Sourcing it from the same value `run_agent_loop` uses for `started` would make that structurally impossible.
<!-- SECTION:NOTES:END -->
