---
id: TASK-2312
title: Stable tab strip and status line across Watchlists sections
status: To Do
assignee: []
created_date: '2026-08-04'
labels:
  - watchlists
  - ux
  - uat-2026-08-04
dependencies: []
priority: medium
---

## Description (the why)

UAT: the section tab strip changes position between tabs — outside the
content boxes on Overview/Sources, INSIDE the bordered Feeds region on
Items/Runs — so the navigation control visibly jumps as you use it. The
snapshot status line likewise wanders (top header line on Sources, buried
under the feed list on Items). The centre header also shows Sources-flavored
content ("No sources yet" + create CTAs) while the Overview tab is active.

UAT findings F2, F22, F23.

## Acceptance Criteria (the what)

- [ ] The tab strip occupies the same visual position on every section.
- [ ] The snapshot/status line has one consistent home.
- [ ] Header content matches the active section (Overview header does not
      advertise Sources actions), or is explicitly section-agnostic in a way
      that reads as global status.
- [ ] Existing region-gating and layout tests stay green.
