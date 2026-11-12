---
id: TASK-34876
title: 'CE-007: late-session tab navigation and Ctrl+P focus quirk'
status: Done
assignee: []
created_date: '2026-09-13 16:36'
updated_date: '2026-09-20 05:05'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Post-#2668 UAT re-run (same QA folder, captures 30b+): late in a long Console session, tab navigation order and Ctrl+P palette focus behaved inconsistently once. Single occurrence, S4; repro unknown.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Reproduced or cleared with a focused keyboard-navigation pass
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed in PR #2736 (commit 580c1c7033): nav-bar optimistic active state resynced after completed navigation. The Ctrl+P/composer half never reproduced (70/70 clean, documented in the UAT minors report). Merged to dev as 030fd9935d0.
<!-- SECTION:NOTES:END -->
