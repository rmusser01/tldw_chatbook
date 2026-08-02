---
id: TASK-1843
title: 'Console reports two contradictory tool counts, and Review tool call is permanently dead'
status: To Do
assignee: []
created_date: '2026-08-01 19:30'
labels:
  - console
  - ux
  - honest-states
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two surfaces in the same panel disagree, and one action can never fire.

`console_tool_count` is read in five places and **assigned nowhere in production** -- `getattr(self.app_instance, "console_tool_count", 0)` always returns 0. The codebase documents this at `console_display_state.py:385`.

The approvals chip was later fixed to use `effective_tool_count = tool_count + (mcp_tool_count or 0)`, but the Inspector row at `:677` still uses the never-populated hook. Result: the Inspector reads "Tools: 0 ready" beside a chip showing a real count.

The same dead counter gates "Review tool call" (`:718`), so the action is permanently disabled and permanently claims "No tool calls are ready for review." Its handler (`chat_screen.py:17433-17443`) is a `notify()` stub that would do nothing even if enabled.

PRODUCT.md: "advanced capabilities must be honest -- unavailable, WIP, dry-run, and blocked states must be explicit." A control that is permanently disabled while claiming a reason is the opposite.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 No Console surface reports a tool count that another Console surface contradicts
- [ ] #2 The count is derived once at a shared source, not fixed independently at each call site
- [ ] #3 Review tool call either opens a real trace or is removed, with no surface still advertising it
- [ ] #4 A test asserts the chip label and the Inspector row derive from the same value
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fix at the shared source in `from_values`, not at both call sites. This is the same bug shape already fixed once on the chip and missed on the row -- fixing it per-site is how it recurs a third time.

Removing the action is the recommended path; a real tool-trace viewer is TASK-1842's territory and should not block this cleanup.
<!-- SECTION:NOTES:END -->
