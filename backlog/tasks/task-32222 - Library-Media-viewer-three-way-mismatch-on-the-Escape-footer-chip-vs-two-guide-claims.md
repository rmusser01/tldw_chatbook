---
id: TASK-32222
title: >-
  Library Media viewer: three-way mismatch on the Escape footer chip vs two
  guide claims
status: In Progress
assignee: []
created_date: '2026-09-10 14:55'
updated_date: '2026-09-10 16:55'
labels:
  - library
  - media
  - docs
  - critique-9
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The viewer's Escape chip reads 'esc focus Library' while the guide claims two different targets in two places. Evidence: critique #9 snapshot (.impeccable/critique/2026-09-10T*__tldw-chatbook-ui-screens-library-screen-py.md; dual-agent live review at dev 02374bf66a on fresh + seeded profiles, 235x52/100x30/60x24). Register row 19.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 One chip text, one guide sentence, matching the real target
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Read _library_media_escape_label per viewer state; compare against the pins and the live chip.
2. Fix the two contradicting guide sentences to one sentence quoting the chip.
3. Pin the chip text against the guide string in a test.
<!-- SECTION:PLAN:END -->
