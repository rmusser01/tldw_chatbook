---
id: TASK-28244
title: 'Review sets - Phase 5 (optional): read-later source and 28009 bridge'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-02 22:29'
updated_date: '2026-09-03 06:37'
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
- [x] #1 A 'Review read-later' entry builds a set from list_read_it_later_media_ids() (already ordered saved_at DESC)
- [ ] #2 If/when task-28009 adds a global media read marker, completing an item in a set optionally flips it; until then done-marks stay set-local
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. PICKER_READ_LATER constant + 'Review read-later' action in the picker dialog footer - TDD
2. Worker branch: ids via list_read_it_later_media_ids (ordered saved_at DESC), titles via search_media id_allowlist, client-side reorder to the id order, cap via build_pinned_items, create origin read_later + land - TDD
3. AC#2 contingent on 28009 (not built): done-marks stay set-local - no code
4. Docs stamp + live spot-check
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2340 (dev b8179f059). 'Review read-later' action in the set picker builds a set from list_read_it_later_media_ids (saved_at DESC, bounded by new limit param at REVIEW_SET_CAP+1), titles via the bounded id_allowlist query reordered client-side to the saved order, shared create+land path. AC#2 intentionally unchecked: contingent on task-28009's global read marker, which does not exist - done-marks stay set-local per design.
<!-- SECTION:NOTES:END -->
