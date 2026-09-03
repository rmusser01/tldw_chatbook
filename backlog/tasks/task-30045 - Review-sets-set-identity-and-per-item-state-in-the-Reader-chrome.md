---
id: TASK-30045
title: Review sets - set identity and per-item state in the Reader chrome
status: Done
assignee:
  - '@claude'
created_date: '2026-09-03 13:06'
updated_date: '2026-09-03 16:46'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique 2026-09-03 P2 + user ruling on Q2: the review set is a workflow object and deserves a real runtime surface, not only a footer string. Today the set's name never appears while walking and the current item's reviewed state is displayed nowhere (m's only feedback is the aggregate counter changing).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 While a set is active the Reader shows the set's name and progress in its chrome
- [x] #2 The current item's reviewed state is visible at a glance and updates when m toggles it
- [x] #3 The chrome disappears when no set is active
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Screen helper _active_review_set_banner: 'Reviewing: <name> - X of M · N reviewed · ✓ reviewed / · not yet reviewed' over one live snapshot; None when inactive - TDD via walker fakes\n2. LibraryMediaViewer gains review_banner param; composes a markup-safe one-line header when set (id library-media-review-banner); screen threads it from _build_library_media_reader\n3. m/walk already sync the viewer, so the banner updates in place - live verify
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2351 (dev 72474d391). Reader banner 'Reviewing: <name> - X of M · N reviewed · ✓ reviewed': _active_review_set_banner (one live snapshot; per-item state only for LIVE items after Qodo caught tombstones claiming state; fails closed w/ logged warning) + LibraryMediaViewer.review_banner (markup-safe Static; absent when empty). Key finding: the in-place _sync_library_media_viewer_state seam services m/walk keystrokes, so the banner had to join its compare-and-assign inputs - without that it went stale/absent (found live). Live-verified: appears on create, m flips state in place, ] advances the ordinal, disappears with no active set.
<!-- SECTION:NOTES:END -->
