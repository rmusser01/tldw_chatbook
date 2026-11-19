---
id: TASK-41519
title: "The macOS plugin qualification job runs 20 to 29 minutes of its 30 minute cap, and the per-child caps are not tightened"
status: To Do
assignee: []
created_date: 2026-11-17 05:27
labels:
- follow-up
- ci
- plugins
priority: medium
references:
- .github/workflows/derived-artifacts.yml
- Tests/Plugins
dependencies:
- TASK-41104
- TASK-41217
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by plgb (round 15, TASK-41104 and TASK-41217). The macOS plugin-qualification job takes 20.0 to 28.6 of its 30 minutes, so a CPU-busy hang could hit the job cap before any per-child cap. TASK-41104's per-child caps should be tightened once the PR's qualification run records per-child CI times.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Per-child CI times from a qualification run are recorded in the notes
- [ ] #2 Per-child caps are set from them, and a hang fails on its child cap before the job cap
- [ ] #3 The job's typical time has at least 5 minutes of margin
<!-- AC:END -->
