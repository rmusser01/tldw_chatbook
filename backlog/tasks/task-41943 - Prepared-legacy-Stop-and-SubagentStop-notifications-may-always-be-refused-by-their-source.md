---
id: TASK-41943
title: "Prepared legacy Stop and SubagentStop notifications may always be refused by their source check"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- agents
- hooks
- bug-suspected
priority: medium
references:
- tldw_chatbook/Agents/legacy_hook_ownership.py
- tldw_chatbook/Agents/activation.py
dependencies:
- TASK-41639
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** Read during TASK-41639 (not reproduced): a prepared legacy Stop/SubagentStop notification is captured under `worker_guard(bridge)` but checked on the notify worker against a fresh observation with no service. The `db.agent_runs` path differs between the two, so the sets may differ (not just their order) and every such notification may be refused.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A test drives a real legacy Stop (and SubagentStop) notification through capture and the notify-worker check and shows whether it is refused
- [ ] #2 If it is refused, capture and check observe the same service and a genuinely changed source is still refused (red first)
<!-- AC:END -->
