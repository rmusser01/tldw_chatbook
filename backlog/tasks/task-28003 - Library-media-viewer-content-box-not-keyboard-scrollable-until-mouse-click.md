---
id: TASK-28003
title: Library media viewer - content box not keyboard-scrollable until mouse click
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - bug
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Re-verified 2026-09-02 live on dev tip (worktree media-ux-fixes @ b7e89b6de, tmux scratch-profile run). Still broken: freshly loading an item then pressing Down x3 and PageDown scrolls nothing; after one mouse click inside the content box (inner focus border appears) the same keys scroll fine. Constraint discovered live: when the LIST has focus, Down moves the selection and auto-loads the adjacent item (desired behavior owned by the list) - so the fix must place focus INSIDE the Reader content at load/open time (or provide an advertised key such as F6 pane cycling that actually reaches the content) without stealing the list's arrow keys.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening a media item allows scrolling the content by keyboard immediately, no mouse required
- [ ] #2 The keyboard path is real via focus placement (and advertised if a key is involved)
- [ ] #3 A regression test covers initial-focus scrollability
<!-- AC:END -->
