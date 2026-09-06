---
id: TASK-31556
title: Console run-store test doubles lack exchange recovery
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 01:25'
updated_date: '2026-09-05 01:27'
labels:
  - console
  - tests
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore current Console bridge and resume-wiring coverage after AgentRunsBridge began requiring the sibling run store's message-exchange recovery seam.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both stale run-store test doubles implement a discriminating `get_message_exchanges` seam.
- [x] #2 The exact two CI regressions pass.
- [x] #3 The focused owning module and static checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce both CI failures and trace AgentRunsBridge construction against the production AgentRunsDB protocol.
2. Add the smallest exchange-recovery behavior to each local fake without weakening production validation.
3. Run the exact regressions, focused owning modules where practical, Ruff, and diff checks.

ADR required: no
ADR path: N/A
Reason: this repairs test doubles against an existing Console persistence seam and changes no runtime architecture.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added an empty, discriminating `get_message_exchanges(message_id)` seam to the deliberately minimal ChaChaNotes doubles used by bridge construction. Runtime code remains strict and unchanged.
- Evidence: the two exact CI regressions plus the in-memory/durable transition test pass 3/3; the complete agent-controller ownership module passes 7/7.
- ADR required: no; this is test-double maintenance for an established persistence contract.
<!-- SECTION:NOTES:END -->
