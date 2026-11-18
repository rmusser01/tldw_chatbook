---
id: TASK-41645
title: "Narrow which workflow controls an unproven return refuses (option C)"
status: In Progress
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- workflows
- console
priority: medium
references:
- tldw_chatbook/Chat/console_workflow_runs.py
- backlog/decisions
dependencies:
- TASK-41320
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-41320 made an unproven control return or metadata read refuse further controls with `workflow_return_unproven`, and the Console says that a restart releases held chats. On 2026-11-17 the owner chose restart-only and asked for the narrowing to be filed. Under option C, only controls that act on the unproven resource refuse; the rest keep working without a restart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 ADR-285 records which controls an unproven return refuses and why
- [ ] #2 Controls outside that set work on a run with an unproven return
- [ ] #3 Tests cover one refused and one allowed control
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, a dated ADR-285 amendment recording which controls an unproven return refuses and why.

1. Analysis first: list, per unproven resource (a control return `control_unknown`, which is per run; an unproven metadata read, whose marker is coordinator-wide), which controls refuse today and which the resource itself requires to refuse. If the set turns out to be a product trade-off rather than determined by the resource, stop and report options.
2. Under the read marker, refuse before any effect only the controls whose completion needs a proven workflow metadata read: reconcile (already), and approve, start and revise, whose new driver can only release its chat through a proven terminal publication. Stop, Pause, Resume, quality acceptance and result removal keep working and report "applied" (TASK-41647).
3. Test: under the marker, Approve is refused with `workflow_return_unproven` and changes nothing; Stop on the same workflow still applies.
4. ADR-285 amendment; covering files on base and head.
<!-- SECTION:PLAN:END -->
