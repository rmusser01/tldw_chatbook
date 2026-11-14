---
id: TASK-35985
title: >-
  Console: a send in a named workspace with a bound folder and Change Review
  enabled stalls at 'Response accepted; waiting for dispatch'
status: To Do
assignee: []
created_date: '2026-10-21 05:14'
labels:
  - console
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found live on 2026-10-20 while verifying TASK-33620.15.1.3. Setup: isolated profiles, Anthropic claude-haiku-4-5, 235x52. Create a workspace with a git folder bound, either from the Console's Workspaces > New dialog or from Settings > Workspaces > Create workspace…. Either way it gets its default Agent, "Workspace 1 Agent". Enable Change Review for that folder in Settings > Workspaces, then send "Say hi in three words."

The turn never dispatches:
- The transcript shows "Response accepted; waiting for dispatch." with Retry response and Discard.
- The composer reads "Send blocked — resolve response recovery first".
- The left rail's workspace row shows "Blocked · Workspace…".
- The send-stage trace records controller_submit entered, then blocked.

It reproduced in all four rounds: on origin/dev 2d071d6d71 (once from each dialog) and twice on the TASK-33620.15.1.3 branch. So it predates that change. Nobody has investigated the cause yet. It is not known which of three things triggers it: the workspace Agent, the bound folder or the Change Review consent. A variant with no Agent failed during setup and was not retried. Users hit this on the main path for working in a project folder, and nothing on screen says why or what to do.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The trigger is identified: which of the workspace Agent, the bound folder and the Change Review consent blocks the send, and the reason the controller records for it
- [ ] #2 A first send in a named workspace with a bound git folder and Change Review enabled reaches the provider and renders the reply (live, real provider)
- [ ] #3 Where a send must still be held in that state, the transcript and composer say why and what to do, instead of 'waiting for dispatch'
- [ ] #4 A deterministic test fails on the base build and pins the fixed behaviour
<!-- AC:END -->
