---
id: TASK-31585
title: Classify remaining diagnostic CI failure clusters
status: Done
assignee: []
created_date: '2026-09-05 05:03'
updated_date: '2026-09-05 05:24'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replay the highest-frequency failures from the superseded diagnostic workflow on the rebased current dev head.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Highest-frequency remaining diagnostic modules are replayed on current dev
- [x] #2 Reproducing failures are grouped into atomic repair tasks
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Replay the highest-frequency unverified modules from diagnostic run 33939188536 on the rebased current dev head. 2. Separate stale old-head failures from reproducible current failures. 3. Open atomic repair tasks for any current failures and record clean modules. ADR required: no. ADR path: N/A. Reason: this is read-only test classification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Replayed six high-frequency diagnostic modules on the rebased current dev head:
1,113 tests passed and 134 failed. Failures were isolated to MCP documentation (39),
Library shell (35), screen navigation (28), Console workbench (13), Console side chat
(13), and Console modal dismissal (6). Created TASK-31586 through TASK-31591 as
separate repair units. ADR required: no; this task made no runtime changes.
<!-- SECTION:NOTES:END -->
