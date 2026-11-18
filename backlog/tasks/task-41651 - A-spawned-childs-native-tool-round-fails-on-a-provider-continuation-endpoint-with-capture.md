---
id: TASK-41651
title: "A spawned child's native tool round fails on a provider-continuation endpoint, with capture on or off"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- agents
- trace
priority: medium
references:
- tldw_chatbook/Agents/agent_service.py
dependencies:
- TASK-41310
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** While verifying TASK-41310, a spawned child that made a native tool call against an endpoint that uses provider continuations failed before any trace was written. It fails the same way with capture off, so the trace work is not the cause.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A spawned child completes a native tool round on a provider-continuation endpoint
- [ ] #2 A test reproduces the failure on dev and passes after the fix, with capture on and off
<!-- AC:END -->
