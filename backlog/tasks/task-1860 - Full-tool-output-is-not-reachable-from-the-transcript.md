---
id: TASK-1860
title: 'Tool trace: the full output of a call is not reachable from the transcript'
status: To Do
assignee: []
created_date: '2026-08-01 20:45'
labels:
  - console
  - ux
  - agents
dependencies:
  - TASK-1842
  - TASK-1843
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Carved out of TASK-1842 AC #3, which that task's PR did not deliver. 1842 fixed the *destruction* of tool markers (the next message erased them); it did not add a route to the **full** result.

What the transcript shows today is a truncated preview: `format_agent_step_marker` renders `⚙ {tool} → {preview}`, capped by `_console_tool_result_display_cap()`. The cap is user-adjustable (Settings, or `TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS`), but that is a global preference set before the fact — not a way to see *this* call's output *now*. There is no expand affordance, and nothing marks a marker as truncated, so a user cannot tell whether they are looking at the whole result or the first N characters of it.

This matters most in the case the original report described: a call that fails or is denied. The user's question is "what did it actually return / how far did it get", and the answer is currently unavailable in-context.

The data exists — steps are persisted to AgentRunsDB and re-derived on resume (`inject_resume_agent_markers`) — so this is a disclosure route, not a retention problem.

Note that TASK-1843 **removed** the Console Inspector's "Review tool call" action because it was permanently dead (it advertised a trace it could never open). That removal was correct, but it means there is now no surface at all offering this. Any fix here should be a live route, not a re-added stub.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The full result of a tool call is reachable from the transcript by keyboard, without changing a setting first
- [ ] #2 A truncated marker is visibly marked as truncated, so the preview is never mistaken for the whole result
- [ ] #3 A failed, denied, or timed-out call exposes whatever output it did produce, not only its failure line
- [ ] #4 Multiple calls in one turn are each independently reachable
- [ ] #5 The route works for a resumed session, whose markers are re-derived from AgentRunsDB rather than produced live
- [ ] #6 A test drives the mounted widget and asserts the full (untruncated) text is reachable -- not a helper returning a string
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC #6 is written that way deliberately. Two defects in the TASK-1842..1846 PR were helpers that worked while the screen stayed wrong, each with a passing test that called the helper directly with a shape production never builds. Assert on the mounted widget.

`RunBudget.max_tool_result_chars` governs what the MODEL saw and is a different number from the display cap — do not conflate them when deciding what "full" means. If the model itself only ever received a truncated result, showing the user more than the model got is still correct (it answers "what did the tool return"), but the distinction should be legible rather than silently blurred.
<!-- SECTION:NOTES:END -->
