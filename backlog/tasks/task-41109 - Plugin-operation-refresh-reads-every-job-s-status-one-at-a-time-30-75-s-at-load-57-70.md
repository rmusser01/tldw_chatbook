---
id: TASK-41109
title: "Plugin operation refresh reads every job's status one at a time, 30-75 s at load 57-70"
status: To Do
created_date: 2026-11-15 02:20
labels:
- follow-up
- plugins
- performance
priority: low
references:
- tldw_chatbook/UI/Plugins_Modules/operations.py
- tldw_chatbook/Plugins/service.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** Refresh runs inside the awaited button handler, so the Plugins screen does not settle until it finishes. TASK-40311's notes measured the refresh reading 29 statuses one at a time: 30.8 s and more at load about 70, 73-75 s at load about 57.

**Evidence.** `Plugins_Modules/operations.py:210-231` (`refresh`): after the page read, `for identity, job in tuple(self.jobs.items()): state = await self.service.operation_status(identity)`, for every job known to the view, not just the page. Each call goes through the service (`service.py:445`).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One refresh reads all statuses it needs in a single service call, or only for the page shown
- [ ] #2 Refresh with 29 jobs is measured before and after at the same load and is at least 5x faster
- [ ] #3 The plugin review tests that wait on refresh stay green
<!-- AC:END -->
