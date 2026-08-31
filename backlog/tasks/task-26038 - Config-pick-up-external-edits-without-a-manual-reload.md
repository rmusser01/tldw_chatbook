---
id: TASK-26038
title: 'Config: pick up external edits without a manual reload'
status: To Do
assignee: []
created_date: '2026-08-31 15:47'
labels:
  - ops
  - config
  - ux
dependencies:
  - TASK-26036
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Editing config.toml outside the app has no effect until the user finds the reload button. Verified on origin/dev: the config cache is keyed on path only (config.py:5101-5107) and requires force_reload=True to re-read; the only user-facing path is a Reload config action in Settings (UI/Screens/settings_screen.py:9037), and a named grep for watchdog, Observer( and hot reload returns no config hits. A user who edits the file in their editor sees stale behavior and reasonably concludes their edit did not work. Hermes keys its cache on file mtime and size so the next read picks up an external edit.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An external edit to the config file is picked up on the next read without a manual reload action
- [ ] #2 Detection is by cheap file metadata comparison, not a filesystem watcher or a polling thread
- [ ] #3 A file being written concurrently does not cause a partial parse - a failed parse falls back per the last-known-good behavior
- [ ] #4 The existing in-process config generation protocol still holds: readers that validated their view are not silently invalidated mid-operation
- [ ] #5 The manual reload action continues to work
- [ ] #6 No measurable overhead is added to the config read hot path - measured and recorded
<!-- AC:END -->
