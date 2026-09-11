---
id: TASK-32347
title: >-
  Library Media rows: the age suffix '· 10m' reads as a duration on audio and
  video
status: To Do
assignee: []
created_date: '2026-09-11 06:14'
labels:
  - library
  - media
  - ux
  - critique-10
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every media row ends '· 10m' (audio · 10m, video · 10m, pdf · 10m). The value is the item's age (the same row read 11m and 14m later) but on audio/video rows it reads as length (A caps 36/39/47). The secondary-line format is pinned by test_media_secondary_fallback_when_no_type_no_age, so this is a pinned design decision the critique disagrees with. Evidence: critique #10 snapshot (.impeccable/critique/2026-09-11T06-13-11Z__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 1f3184655b, captures under the session scratchpad crit10/A and crit10/B).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Age is labelled ('added 10m ago') or replaced by type-aware metadata (duration for audio/video, pages for pdf/ebook, words for article/document) with the date on Info
- [ ] #2 The pin is updated to the new format, not loosened
<!-- AC:END -->
