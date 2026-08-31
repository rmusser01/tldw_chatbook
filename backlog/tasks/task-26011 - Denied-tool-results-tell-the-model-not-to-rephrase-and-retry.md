---
id: TASK-26011
title: 'Denied tool results: tell the model not to rephrase and retry'
status: To Do
assignee: []
created_date: '2026-08-31 15:44'
labels:
  - agents
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A denied tool call invites an immediate near-identical retry. Verified on origin/dev: the refusal copy is fixed text - Agents/mcp_tool_provider.py:87 USER_DENY_REFUSAL and Agents/builtin_tool_gate.py:359 - and states the denial without instructing against retrying by another route, so the model commonly rephrases and re-asks, burning turns and approval prompts. Hermes states explicitly that the model must not retry, must not rephrase, and must not pursue the same outcome by a different path, and that silence is not consent. Complements task-18920 (deny with a reason) and task-18929 (denial circuit breaker) without depending on either.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Denied-call results instruct the model not to retry the same call, not to rephrase it, and not to pursue the same outcome by another route
- [ ] #2 The instruction is distinct from the user's own reason text where one is supplied, so user words are never confused with system policy
- [ ] #3 Timeout, unresolved, kill-switch and Off refusals keep their existing distinct copy - only the user-denial path changes
- [ ] #4 Copy is asserted by a test so it cannot silently drift
- [ ] #5 No permission-model change: this is result text only
<!-- AC:END -->
