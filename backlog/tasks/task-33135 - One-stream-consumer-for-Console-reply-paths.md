---
id: TASK-33135
title: One stream consumer for Console reply paths
status: To Do
assignee: []
created_date: '2026-09-27 19:46'
labels: [refactor, console]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console runs two complete reply pipelines: _run_direct_provider_reply in the controller and the agent path through console_agent_bridge.run_reply. Both duplicate request preparation, the streaming loop, thinking projection, terminal-state handling, and usage attachment, and the duplication is actively drifting (the same quadratic-fix comment was pasted into both files on a recent branch; usage attachment is invoked eight times to cover both pipelines' outcomes). The gateway also builds chat kwargs through two twins that have already diverged. A shared stream consumer (prepare, loop, thinking projection, terminal state, usage) used by both paths removes copy-drift from the hottest path in the app. The direct path cannot be deleted outright (prefill turns and character mode deliberately bypass the agent loop), but it can become a thin wrapper over the shared consumer.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both reply paths consume one shared stream-consumer component.
- [ ] #2 Thinking projection is implemented exactly once.
- [ ] #3 The _chat_api_kwargs twins are unified to build from the prepared request.
- [ ] #4 Prefill and character-mode behavior is unchanged, covered by tests.
- [ ] #5 Cancel, stop, and usage outcomes are covered by tests for both paths.
<!-- AC:END -->

## Renumbering provenance

Filed 2026-09-27 as task-33090. origin/dev minted its own task-33090 before this branch merged, so per the landed-keeps-id rule this task moved to task-33135; every inbound reference moved with it.
