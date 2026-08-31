---
id: TASK-26010
title: Post-tool-dispatch hook seam
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
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
- [ ] #1 A hook fires after a tool call completes, receiving the call, its outcome and its timing
- [ ] #2 The hook is observational: raising inside it cannot fail the tool call or the run
- [ ] #3 It fires for successful, failed, denied and timed-out calls, with the outcome distinguishable
- [ ] #4 It fires for tool calls made by sub-agents as well as the primary run, with the owning run identifiable
- [ ] #5 With no hook registered there is no measurable overhead and behavior is unchanged
- [ ] #6 The dead Tools/file_operation_hooks.py is either removed or explicitly documented as retired, so it is not mistaken for this seam
<!-- AC:END -->
