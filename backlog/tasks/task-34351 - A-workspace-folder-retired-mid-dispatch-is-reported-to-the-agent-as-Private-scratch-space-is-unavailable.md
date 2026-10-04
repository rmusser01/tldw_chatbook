---
id: TASK-34351
title: >-
  A workspace folder retired mid-dispatch is reported to the agent as 'Private scratch space is unavailable'
status: To Do
assignee: []
created_date: '2026-10-03 18:37'
labels:
  - workspace
  - agents
  - follow-up-33940
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-33940.2 stopped worker failures on an admitted workspace folder from being reported as "Private scratch space is unavailable; the tool was not run." One other branch still sends that copy when the problem is a workspace folder, not scratch.

In `Agents/local_tool_provider.py`, when a path tool's run authority names an alias whose specs have been removed (a run retired the workspace root between resolving the authority and looking up its specs; see the TASK-28238 comment at that branch), the provider returns `LOCAL_AUTHORITY_UNAVAILABLE_REFUSAL`, the scratch copy. Refusing is correct there. The wording is not: it sends the model, and the user reading the activity row, to the wrong place, which is the confusion TASK-33940 started from.

This belongs with the Console copy audit in TASK-33621, which tracks failure copy that states false facts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When a path tool is refused because its workspace folder was retired while the call was being dispatched, the result text says the workspace folder is no longer available to this run and does not mention scratch space
- [ ] #2 The scratch-unavailable copy is still returned when the private scratch space itself is unavailable
- [ ] #3 The new copy is registered with the Console activity presentation so the activity row shows it as blocked
- [ ] #4 A test reproduces the retire race against the real provider, fails on dev `01a2020981`, and passes with the fix
<!-- AC:END -->
