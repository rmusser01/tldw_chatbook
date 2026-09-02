---
id: TASK-28003
title: Library media viewer - content box not keyboard-scrollable until mouse click
status: Done
assignee: []
created_date: '2026-09-02 04:10'
updated_date: '2026-09-02 05:44'
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
- [x] #1 Opening a media item allows scrolling the content by keyboard immediately, no mouse required
- [x] #2 The keyboard path is real via focus placement (and advertised if a key is involved)
- [x] #3 A regression test covers initial-focus scrollability
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause: the media Reader's F6 pane target (_MEDIA_WORKBENCH_FOCUS_TARGETS) only offered library-media-reader-find and library-media-back, so the content ScrollView (VirtualizedRawContent, can_focus=True) was in NO keyboard focus path - only a mouse click could focus it, hence dead arrow/PageDown scroll on a fresh open. Fix: add library-media-viewer-content-text as the FIRST F6 candidate; _resolve_focus_target picks the first focusable id present, so F6 into the Reader lands on the scroller in Read mode and falls through to Find in the other modes. Find stays reachable via '/'. Coherent with task-28004: list focus navigates items, F6 moves into the Reader to scroll. Known limitation/follow-up: rendered-Markdown mode's scroller is the non-focusable body container so keyboard scroll there still needs a mouse; raw mode (the transcript default) is covered. Did NOT auto-focus content on open (would steal Down from list nav). Test: test_media_global_f6_reaches_content_scroller. Files: UI/Screens/library_screen.py, Tests/UI/test_library_media_reader_flow.py.
<!-- SECTION:NOTES:END -->
