---
id: TASK-31576
title: Library Reader - transcripts with hashtag lines should not default to Rendered
status: To Do
assignee: []
created_date: '2026-09-05 03:23'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 4 PR C made video and audio transcripts default to the Rendered (Markdown) mode. A transcript whose lines start with # (hashtags, not headings) now renders as a wall of headings. A two-signal sniff (heading followed by a blank line, or a low heading density) should pick Plain for those.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A transcript with hashtag lines opens in Plain
- [ ] #2 A Markdown-shaped transcript opens Rendered
- [ ] #3 Tests cover both shapes
<!-- AC:END -->
