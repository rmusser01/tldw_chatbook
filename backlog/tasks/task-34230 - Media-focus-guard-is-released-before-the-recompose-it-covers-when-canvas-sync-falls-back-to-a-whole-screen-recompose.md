---
id: TASK-34230
title: >-
  Media focus guard is released before the recompose it covers when canvas sync
  falls back to a whole-screen recompose
status: To Do
assignee: []
created_date: '2026-10-04 00:07'
labels:
  - library
  - focus
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Library canvas sync helper returns False both when it drops the caller's follow-up (a suppressed or refused sync) and when its whole-screen fallback queues that follow-up. The Media list path (TASK-32171) and the media-trash path both treat every False as "the follow-up will not run" and release the restore-focus guard at once. On the fallback path the guard is therefore dropped just before the recompose whose automatic focus fallback it exists to mask, so the pending list-entry focus can be disarmed or an unintended row selected.

Found in the PR #2993 review. Narrow: it needs the media canvas to be unmounted when the sync runs. Not fixed there because TASK-32171's release traded a guard that stuck forever for this early release, which is the better failure, and the helper has several return paths that need telling apart.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 When the canvas sync takes the whole-screen fallback with an entry focus pending, the restore-focus guard stays armed until the queued follow-up has run
- [ ] #2 When the sync is suppressed or refused, the guard is still released immediately, and TASK-32171's stuck-guard regression test still passes
- [ ] #3 The media-trash path behaves the same way as the Media list path
- [ ] #4 A test tells the two False outcomes apart and fails if they are conflated again
<!-- AC:END -->
