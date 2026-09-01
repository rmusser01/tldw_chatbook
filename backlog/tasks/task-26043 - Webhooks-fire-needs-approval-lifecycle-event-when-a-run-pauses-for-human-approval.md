---
id: TASK-26043
title: >-
  Webhooks: fire needs-approval lifecycle event when a run pauses for human
  approval
status: To Do
assignee: []
created_date: '2026-09-01 23:47'
labels:
  - interop
  - ops
dependencies:
  - TASK-26031
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-26031 shipped signed run-lifecycle webhooks and wired completed+failed at the AgentService terminal choke point. The third 'at minimum' event, needs-approval, fires when a run pauses for a human approval card -- which happens inside the app-context approval bridge (ConsoleChatController.request_mcp_approvals / the loop-thread review bridge), not at a clean persisted run-state seam, and review_tool_calls runs even when tools auto-approve so it cannot be hooked directly. Wire needs-approval at the point a card is actually raised, using the existing run_webhooks.schedule_run_webhook + the [webhooks] config, deduped so it fires once per pause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A run pausing for human approval POSTs a needs-approval webhook once per pause
- [ ] #2 It does not fire when tools auto-approve (no card shown)
- [ ] #3 Reuses run_webhooks.schedule_run_webhook and the [webhooks] config gate
<!-- AC:END -->
