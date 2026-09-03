---
id: TASK-28244
title: 'Review sets - Phase 5 (optional): read-later source and 28009 bridge'
status: To Do
assignee: []
created_date: '2026-09-02 22:29'
labels:
  - library
  - media-ux
dependencies:
  - TASK-28240
  - TASK-28242
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Optional: build a review set from the read-later queue, and bridge done-marks to a future global read marker (design: backlog/docs/design-library-review-sets.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A 'Review read-later' entry builds a set from list_read_it_later_media_ids() (already ordered saved_at DESC)
- [ ] #2 If/when task-28009 adds a global media read marker, completing an item in a set optionally flips it; until then done-marks stay set-local
<!-- AC:END -->
