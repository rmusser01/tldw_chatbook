---
id: TASK-15665
title: >-
  A live sub-agent's provider client is never closed when the app tears down
  mid-flight
status: Done
assignee:
  - '@codex'
created_date: '2026-08-11 21:30'
updated_date: '2026-09-08 03:48'
labels:
  - console
  - agents
  - resources
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR 3a-1 Task 6b (audit F5) made `ConsoleProviderGateway.aclose()` SKIP running loops and RETAIN their entries, because the previous behaviour closed a live child's connection pool on the child's own loop mid-request. Retaining is the right trade, but nothing closes those clients early: if the app tears down while children are in flight, one pool per live child is left to its finalizer instead of being closed deliberately. Bounded by `max_live_subagents` per conversation, so small, but it is an accepted leak rather than a handled shutdown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A teardown with live children closes each child's pool once that child's own loop stops, deliberately rather than by finalizer
- [x] #2 No pool belonging to a still-running child is closed mid-request (the existing F5 regression still passes)
- [x] #3 A test asserts no retained client entry outlives its owning lifeline
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow the TASK-15665 section of Docs/superpowers/plans/2026-09-07-fleet-lifecycle-and-identity-repairs.md. Reproduce first, implement the bounded repair, and run its targeted regressions before marking Done.

ADR required: no additional ADR
ADR path: backlog/decisions/130-model-call-lifeline-client-teardown.md
Reason: Direct implementation of the accepted resource/identity contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Primary and child model-call lifelines now close their own gateway HTTP clients before their event loops stop. ConsoleProviderGateway.aclose_current_loop removes only the caller loop's entry and leaves injected clients untouched. The driver drains cancelled pending work and async generators before pool closure; it owns final loop close even when cleanup exceeds the existing bounded join. Repeated shutdown, failed thread start, cleanup failure, and delayed cleanup are covered. App-level aclose continues to spare running children.

The original real-bridge regression failed because a successful surviving child left its pool open after closing its loop. Self-review then caught a race in an initial stop/restart cleanup shape; a new red regression recorded an open pool at the first driver stop. Cleanup now stays on the running loop through closure, preventing app teardown from classifying that pool as idle. The testing lesson records the actual incident.

Files: tldw_chatbook/Chat/console_agent_bridge.py, tldw_chatbook/Chat/console_provider_gateway.py, Tests/Chat/test_fleet_client_lifecycle.py. Reviewed diagnostic inventory update: one constant, payload-free cleanup-failure warning and its aggregate count; inventory verification passes (522 owners, 1215 TASK-492 calls, 7155 TASK-494 calls, 7 sink files).

ADR required: yes; implemented backlog/decisions/130-model-call-lifeline-client-teardown.md. It records the resource boundary, thread-owned cleanup, bounded wait, and limits under permanently blocked work or process exit.

Verification: 710 targeted tests passed across disjoint groups: 471 gateway/bridge/lifeline tests (469 passed initially; two localhost-server fixtures needed sandbox permission and both passed on the permitted rerun), 195 identity/continuation/coordinator/runtime tests, and 44 teardown/close-session/runtime-lifetime/fanout/stop tests. New test modules pass Ruff lint/format. The four edited production modules add no Ruff diagnostics compared with this pass's starting working tree. Scoped git diff whitespace checks pass. No full suite or live-provider run. Self-review completed.

Docs: backlog/docs/agent-orchestration-review-2026-09-07.md, Docs/superpowers/plans/2026-09-07-fleet-lifecycle-and-identity-repairs.md, and Docs/User_Guide/console/agent-runs-and-tools.md. Existing unrelated working-tree changes and the earlier mailbox/wake repairs were preserved. Changes are uncommitted.
<!-- SECTION:NOTES:END -->
