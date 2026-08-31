---
id: TASK-26002
title: 'Agent loop: deterministic empty-response detection'
status: To Do
assignee: []
created_date: '2026-08-31 15:43'
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
- [ ] #1 A turn producing neither visible text nor tool calls is not reported as a successful completion
- [ ] #2 Two consecutive empty responses from the same provider and model stop the run with an honest message instead of retrying indefinitely
- [ ] #3 A single empty response is retried, composing with task-25901's retry policy, rather than immediately failing
- [ ] #4 The terminal message names the provider and model so the user can act on it
- [ ] #5 Tests cover: one empty then success, two consecutive empties, and empty-text-with-tool-calls which is legitimate and must not trip
<!-- AC:END -->
