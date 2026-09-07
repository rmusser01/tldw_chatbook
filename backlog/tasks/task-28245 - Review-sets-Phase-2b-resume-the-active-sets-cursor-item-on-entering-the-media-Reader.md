---
id: TASK-28245
title: >-
  Review sets - Phase 2b: resume the active set's cursor item on entering the
  media Reader
status: Done
assignee:
  - '@claude'
created_date: '2026-09-03 01:47'
updated_date: '2026-09-03 07:38'
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
- [x] #1 Opening the media Reader with an active set loads the set's cursor item automatically (no keypress needed)
- [x] #2 Per-item scroll resume (ReadingProgress) still restores within the loaded item
- [x] #3 Does not fire during cold-start tab switching or fight the initial-tab navigation
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. _maybe_auto_resume_review_set: worker (group library_review_set) resolving the active set's cursor off-loop; opens the Reader at it ONCE per set (screen-session guard), only if still on the media list and the screen is current - TDD via fakes
2. Hooks: end of the rail-select seam (media row) + on_mount's media-list branch
3. AC#2: loading goes through _open_library_media_viewer = same path as a row press, ReadingProgress untouched
4. AC#3: worker's final still-on-media-list + is_current gates make a cold-start yank abort; live tmux verify incl. restart
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2342 (dev e364bf662). _auto_resume_review_set_worker resolves the active set's live cursor off-loop and opens via _open_library_media_viewer (same path as a row press, so ReadingProgress restore untouched - AC#2); once per set per screen session (guard burned only when the open happens). ONE hook only: the rail-select seam's media branch - the mount-leg kick was added then deliberately REMOVED because its boot-time timing made the worker's lazy imports race the _ui_ready module census (flaky Perf Guard 977>972); the rail gesture cannot race boot, which also settles AC#3 structurally. Auto-resume runs in its own exclusive worker group (library_review_set_resume) after Qodo caught the shared group cancelling in-flight set creation. Live-verified: restart -> click Media -> auto-loads cursor item at '2 of 3 · 1 reviewed'; Escape -> away -> re-entry shows the list.
<!-- SECTION:NOTES:END -->

## Superseded (task-31234)

The once-per-set gate was REMOVED by task-31234 (critique #3 P1, user ruling at the close): re-entry re-armed the banner over an off-set document — the frame restored, the item not. Auto-resume now opens the cursor item on EVERY entry gesture; "Escape + re-entry shows the list" no longer holds (re-entry lands in the Reader at the saved place). AC#3's cold-start yank guard is unchanged.
