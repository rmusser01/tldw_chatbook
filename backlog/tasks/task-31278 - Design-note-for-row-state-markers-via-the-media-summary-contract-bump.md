---
id: TASK-31278
title: Design note for row state markers via the media summary-contract bump
status: In Progress
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-04 15:09'
labels:
  - library
  - media-ux
  - design
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 close, user ruling: reviewed and analysis markers on list rows need the 5-key media summary contract bumped (task-28008 analysis presence, task-28009 read markers), which was deferred twice as invasive. The user wants the proposal as a design note first, then its own implementation PR. The note must cover the contract change, the projection for has_analysis and reviewed state, the source of truth for reviewed (set-local done marks vs a per-item marker), rendering inside a 38-col row, and the migration plan for every producer and test fake of the summary shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A design note exists at backlog/docs/design-library-row-state-markers.md covering contract, projection, reviewed source-of-truth, 38-col rendering and the producer/fake migration plan
- [ ] #2 The note lists the decisions that need the user's ruling with a recommendation for each
- [ ] #3 User approval is recorded in the note before any implementation task is started
- [x] #4 Implementation tasks reference 28008 and 28009 rather than duplicating them
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design note drafted at backlog/docs/design-library-row-state-markers.md (options A/B/C, recommendation A: seven-key contract with has_analysis projected in SQL and reviewed decorated from the active set; four decisions listed for the user). Awaiting user approval (AC#2) before any implementation of 28008/28009.
<!-- SECTION:NOTES:END -->
