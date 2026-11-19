---
id: TASK-41950
title: "A retained Change Review initialization job's opaque claim refuses every later workspace activity until restart"
status: To Do
assignee: []
created_date: '2026-11-19 07:30'
labels:
- follow-up
- workspaces
- needs-decision
priority: medium
references:
- tldw_chatbook/Workspaces/change_review_consent.py
- tldw_chatbook/Workspaces/worktree_activity.py
dependencies:
- TASK-41635
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41630..41655 follow-up burn-down (2026-11-18/19). Checked on dev d0486978dd unless stated; paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** When a Change Review init job runs under its own owner and its git cleanup is retained (unproven), it keeps an opaque activity claim, and an opaque claim conflicts with every other. That likely refuses later Console sends ("Workspace activity is busy") as well as later inits until restart. Releasing the tool slot frees only the slot. TASK-40716 decided claims stay held until restart; this asks whether that should hold for an init job's claim.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded (ADR-134 or ADR-285)
- [ ] #2 A test pins the chosen behaviour for a later Send while an init job's cleanup is retained
<!-- AC:END -->
