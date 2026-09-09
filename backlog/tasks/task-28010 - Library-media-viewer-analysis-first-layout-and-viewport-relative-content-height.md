---
id: TASK-28010
title: >-
  Library media viewer - analysis-first layout and viewport-relative content
  height
status: To Do
assignee: []
created_date: '2026-09-02 04:10'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Compose order is metadata, Content (toggle, search, body), Analysis, Highlights, then a five-button action row (library_media_viewer.py:117-243), and the content body is hard-capped at max-height 18 (_agentic_terminal.tcss:2758-2768) even on a 52-row terminal. Analysis-first readers scroll past the transcript apparatus on every item; transcript readers get an 18-line porthole regardless of screen size. Put Analysis above Content (or make Content collapsible) and size the content box relative to the viewport.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The analysis is readable without scrolling past the content block
- [ ] #2 Content height scales with terminal height instead of a fixed 18 rows
- [ ] #3 The full transcript remains reachable and scrollable
<!-- AC:END -->
