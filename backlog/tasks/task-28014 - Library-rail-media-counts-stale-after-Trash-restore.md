---
id: TASK-28014
title: Library rail - media counts stale after Trash restore
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). Confirmed and worse than filed: deleting updates rail and canvas counts together, but after RESTORE from Trash the media canvas enters a degraded state - header shows bare "Media" with "Media changed; retry to load a current page", the restored item is missing from the list, and the pager shows "List may be out of date / Page boundary is unknown" with a manual Retry button. After clicking Retry the canvas shows Media (3) but the rail STILL says Media (2) - rail and canvas disagree even post-retry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Rail media count and Details tally match the canvas immediately after restore and other Trash mutations
- [ ] #2 A pinning test covers the restore-count path
<!-- AC:END -->
