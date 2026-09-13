---
id: TASK-31278
title: Design note for row state markers via the media summary-contract bump
status: Done
assignee: []
created_date: '2026-09-04 13:54'
updated_date: '2026-09-05 06:16'
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
- [x] #2 The note lists the decisions that need the user's ruling with a recommendation for each
- [x] #3 User approval is recorded in the note before any implementation task is started
- [x] #4 Implementation tasks reference 28008 and 28009 rather than duplicating them
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Design note approved by the user on 2026-09-04 at the critique #5 close: Option A (seven-key media summary contract with has_analysis + reviewed), reviewed sourced from the active review set only in v1, glyphs ✓/· in the leading state slot with the word analysed on the secondary line plus the matched keyword for keyword hits, one PR inside fix wave 5 (tasks 28008/28009). See backlog/docs/design-library-row-state-markers.md §5.
<!-- SECTION:NOTES:END -->
