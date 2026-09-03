---
id: TASK-28245
title: >-
  Review sets - Phase 2b: resume the active set's cursor item on entering the
  media Reader
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-03 01:47'
updated_date: '2026-09-03 05:59'
labels:
  - library
  - media-ux
dependencies:
  - TASK-28241
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Split from task-28241 AC#4. The walker (AC1-3) resumes the SET STATE automatically (the active flag + cursor persist across restarts, so ]/[ walk from the saved cursor). What remains is the convenience of AUTO-LOADING the cursor item into the Reader when the user opens the media area with a set active, rather than requiring one keypress. This is startup/entry-timing sensitive (the browse list must settle first, and cold-start yanks the initial tab), so it is split out for careful live verification. Design: backlog/docs/design-library-review-sets.md AC4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening the media Reader with an active set loads the set's cursor item automatically (no keypress needed)
- [ ] #2 Per-item scroll resume (ReadingProgress) still restores within the loaded item
- [ ] #3 Does not fire during cold-start tab switching or fight the initial-tab navigation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. _maybe_auto_resume_review_set: worker (group library_review_set) resolving the active set's cursor off-loop; opens the Reader at it ONCE per set (screen-session guard), only if still on the media list and the screen is current - TDD via fakes
2. Hooks: end of the rail-select seam (media row) + on_mount's media-list branch
3. AC#2: loading goes through _open_library_media_viewer = same path as a row press, ReadingProgress untouched
4. AC#3: worker's final still-on-media-list + is_current gates make a cold-start yank abort; live tmux verify incl. restart
<!-- SECTION:PLAN:END -->
