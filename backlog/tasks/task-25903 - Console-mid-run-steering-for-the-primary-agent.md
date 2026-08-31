---
id: TASK-25903
title: 'Console: mid-run steering for the primary agent'
status: To Do
assignee: []
created_date: '2026-08-31 15:08'
updated_date: '2026-08-31 15:11'
labels:
  - console
  - agents
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A user who wants to correct a running agent must stop it, losing every completed tool result, then retype. Verified on origin/dev: the steering machinery already exists and is wired for fleet children only - Agents/agent_runtime.py:1196-1230 drains a mailbox before each model call and never splits a tool_calls/tool pair, format_steering_message and the MAX_STEERING_CHARS cap are in place, and a user-facing steering bar exists (UI/Console_Modules/agent.py:1445) - but drain_mailbox is None for a primary run by explicit design (Agents/agent_service.py:3486), so typed text goes to the prompt queue for the next turn instead (Chat/console_prompt_queue.py:60). This is the core TUI interaction and the smallest of the top-ranked gaps because the protocol-coherent drain point is already proven in production for children.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Text submitted while the primary agent is running can be delivered to the current run instead of queued, at the user's choice
- [ ] #2 Steered text is drained at the same protocol-coherent point children use - before a model call, after budget and cancel checks - and never splits a tool_calls/tool message pair
- [ ] #3 The steered message is visible in the transcript as user-authored, distinct from the original prompt
- [ ] #4 The existing queue path remains available and unchanged for users who prefer it; the default is stated explicitly in the task notes
- [ ] #5 Steering a run that has already finished or been cancelled is refused honestly rather than silently dropped
- [ ] #6 The same character cap and sanitization applied to child steering applies here - verified by tests
<!-- AC:END -->
