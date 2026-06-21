---
id: TASK-128
title: Console UAT harness and acceptance matrix
status: In Progress
assignee: []
created_date: '2026-06-21 00:35'
updated_date: '2026-06-21 00:49'
labels:
  - console
  - uat
  - qa
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a reusable Console UAT harness so all Console workstreams verify against the same CDP setup, fixtures, screenshots, and acceptance checklist.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A Console UAT acceptance matrix exists and tracks Harness, Chat Lifecycle, Provider + Model Configuration, Message Actions, Workspace + Resume, and final integration statuses.
- [ ] #2 A repeatable CDP/Textual-web launch procedure exists with stable ports, isolated config paths, fixture data expectations, and screenshot naming conventions.
- [ ] #3 The harness records actual rendered screenshots or recordings only; generated mockups, SVGs, and code-layout screenshots are explicitly disallowed for approval.
- [ ] #4 The harness defines the required visual checkpoint states: not started, in progress, blocked, needs screenshot, approved, and merged.
- [ ] #5 The harness includes a focused regression/UAT command list that later workstreams can reuse without rediscovering setup.
- [ ] #6 The harness PR does not implement Console feature behavior beyond documentation, fixtures, or test harness plumbing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define the Console UAT acceptance matrix and workstream ownership boundaries.
2. Restore a durable Console CDP evidence protocol because the previously referenced runbook is absent on current dev.
3. Keep the harness PR documentation- and fixture-only; do not implement Console behavior.
4. Verify the generated task files parse through Backlog.md and git diff checks pass.
5. Commit the coordination branch so downstream workstreams can branch after merge.

ADR required: no.
ADR path: N/A.
Reason: This task defines verification workflow and coordination artifacts, not product architecture or data/runtime contracts.
<!-- SECTION:PLAN:END -->
