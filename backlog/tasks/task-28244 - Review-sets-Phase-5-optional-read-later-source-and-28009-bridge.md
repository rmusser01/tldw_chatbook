---
id: TASK-28244
title: 'Review sets - Phase 5 (optional): read-later source and 28009 bridge'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-02 22:29'
updated_date: '2026-09-03 05:03'
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

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. PICKER_READ_LATER constant + 'Review read-later' action in the picker dialog footer - TDD
2. Worker branch: ids via list_read_it_later_media_ids (ordered saved_at DESC), titles via search_media id_allowlist, client-side reorder to the id order, cap via build_pinned_items, create origin read_later + land - TDD
3. AC#2 contingent on 28009 (not built): done-marks stay set-local - no code
4. Docs stamp + live spot-check
<!-- SECTION:PLAN:END -->
