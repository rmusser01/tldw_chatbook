---
id: TASK-34943
title: A config file edited outside the app is not picked up while the app runs
status: To Do
created_date: 2026-10-12 05:26
labels:
- config
- correctness
priority: medium
updated_date: 2026-10-14 02:41
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The app decides whether its cached settings are still current by looking at the config file's identity (which file it is, its type, mode, owner and link count) and never at its size or modification time. An edit that keeps the same file, which is what many editors do, is therefore never noticed: the old provider, model or key stays in effect until a restart or an in-app settings save. A second, separate check that does compare modification time and size is defeated by its own throttle: the first of two reads inside one throttle window uses up the window and the second answers from the cache. Both were verified on a scratch profile while investigating flaky tests for TASK-34789 (an in-place edit was served stale on four reads over five seconds; four reads 1.2 seconds apart after an edit each returned the old value). A third route was inferred and not run: on a filesystem that hands freed inode numbers back, two replace-by-rename saves in a row can leave the file with its old identity. The existing hot-reload tests patch the throttle to zero or to an hour, so they see none of this.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A config file edited in place outside the app is read again by the next settings read after the throttle window
- [ ] #2 A config file replaced twice by rename, ending with the identity it had before, is read again
- [ ] #3 The external-edit check answers the same for every read inside one throttle window
- [ ] #4 A test runs each case at the production throttle value and fails before the fix
- [ ] #5 Settings reads that find nothing changed do no more file I/O than today
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-10-14 (review of PR #3101): this overlaps TASK-34700, which covers the throttle half (the warm-cache check consumes the 1 s window, so the bootstrap's own check inside it serves the stale cache; pinned by the strict xfail Tests/test_config_hot_reload.py::test_external_edit_is_picked_up_at_the_shipped_throttle). This task adds the identity-only cache check as a second cause. PR #3101 measured the effect on [web_security] allowed_hosts: an in-place save and a rename save were both still refused on five checks 1.2 s apart; clearing the caches (what a restart does) admitted the host. Fix the two together.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
