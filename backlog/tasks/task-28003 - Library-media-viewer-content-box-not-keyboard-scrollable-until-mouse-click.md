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
Live-reproduced: opening an item then pressing Down/PageDown (and Tab plus arrows) does not scroll the content box; after one mouse click inside the box, PageDown scrolls fine. Keyboard-only users cannot read past the first visible lines of a transcript. Root cause untraced - initial focus does not land on (or route keys to) the scrollable content container.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening a media item allows scrolling the content by keyboard immediately, no mouse required
- [ ] #2 The keyboard path is real via focus placement (and advertised if a key is involved)
- [ ] #3 A regression test covers initial-focus scrollability
<!-- AC:END -->
