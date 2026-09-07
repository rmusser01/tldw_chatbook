---
id: TASK-31954
title: Library media - the reader progress restore is scheduled twice per mode change
status: Done
assignee: []
created_date: '2026-09-07 08:26'
updated_date: '2026-09-07 20:03'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
H2 review: handle_library_media_reader_mode schedules _restore_library_media_loaded_progress through the viewer seam and again through _sync_library_media_viewer_state's own call_after_refresh, unconditionally re-armed by nulling _library_media_progress_restored_id. Harmless today because scroll_to is idempotent and pre-existing, but two owners for one restore is how a future scroll change becomes a visible jump.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One owner schedules the progress restore for a reader mode change
- [x] #2 A pin counts the restore invocations for a single mode change
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Count the restore invocations per mode change (pin). 2. One owner: the mode handler claims `_library_media_progress_restored_id` instead of nulling it, so the tail's arm-once guard stays quiet.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
`handle_library_media_reader_mode` claims the restored id rather than nulling it; the sync tail's arm-once guard then does not re-schedule, so the handler's seam call is the single scheduler. Pin: one restore per Read → Analysis → Read. Finding (pre-existing, rider): the restore does not visibly land on a mode change on the merge-base either — `scroll_to` races the rendered Markdown body's parse — so the two owners were two ineffective schedules, not one working plus a spare.
<!-- SECTION:NOTES:END -->
