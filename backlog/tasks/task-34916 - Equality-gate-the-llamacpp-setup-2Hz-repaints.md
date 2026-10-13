---
id: TASK-34916
title: Equality-gate-the-llamacpp-setup-2Hz-repaints
status: To Do
assignee: []
created_date: '2026-10-13 14:45'
labels:
  - performance
  - ui-perf
  - wave-1-timers
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
llamacpp_setup_view arms set_interval(0.5, refresh_state) (llamacpp_setup_view.py:198) which re-renders launch state, query_ones the profile Select plus 5 buttons, snapshots owner state, and updates Statics every tick with no equality guard — unlike LabScreen.refresh_lab_status which guards its writes. Pure repaint churn at 2Hz while the Lab > llama.cpp view is mounted.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Widget writes happen only when computed state differs from last rendered state,query_one and snapshot lookups skipped or cached when state is unchanged
<!-- AC:END -->
