---
id: TASK-128
title: Console UAT harness and acceptance matrix
status: Done
assignee:
  - '@codex'
created_date: '2026-06-21 00:35'
updated_date: '2026-06-21 21:36'
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
- [x] #1 A Console UAT acceptance matrix exists and tracks Harness, Chat Lifecycle, Provider + Model Configuration, Message Actions, Workspace + Resume, and final integration statuses.
- [x] #2 A repeatable CDP/Textual-web launch procedure exists with stable ports, isolated config paths, fixture data expectations, and screenshot naming conventions.
- [x] #3 The harness records actual rendered screenshots or recordings only; generated mockups, SVGs, and code-layout screenshots are explicitly disallowed for approval.
- [x] #4 The harness defines the required visual checkpoint states: not started, in progress, blocked, needs screenshot, approved, and merged.
- [x] #5 The harness includes a focused regression/UAT command list that later workstreams can reuse without rediscovering setup.
- [x] #6 The harness PR does not implement Console feature behavior beyond documentation, fixtures, or test harness plumbing.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed the Console UAT harness after the parallel workstreams merged into `dev`. The acceptance matrix now records the completed Chat Lifecycle, Provider + Model Configuration, Message Actions, and Workspace + Resume streams, and adds `TASK-132` as the explicit final integration replay task. The CDP evidence protocol is the reusable harness: it defines actual rendered Textual-web/CDP evidence requirements, prohibited evidence, isolated launch requirements, screenshot naming, minimum Console states, and final integration replay expectations.

This closeout is documentation/task hygiene only. It intentionally does not implement or modify Console runtime behavior.

Verification:

- `python -m pytest -q Tests/UI/test_product_maturity_phase1_harness.py::test_backlog_task_frontmatter_ids_are_unique --tb=short`
- `git diff --check`

ADR required: no
ADR path: N/A
Reason: This task defines QA coordination and evidence workflow only; it does not change storage, sync, provider/runtime boundaries, service contracts, security policy, or long-lived product architecture.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
