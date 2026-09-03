---
id: TASK-30045
title: Review sets - set identity and per-item state in the Reader chrome
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-03 13:06'
updated_date: '2026-09-03 14:03'
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
- [ ] #1 While a set is active the Reader shows the set's name and progress in its chrome
- [ ] #2 The current item's reviewed state is visible at a glance and updates when m toggles it
- [ ] #3 The chrome disappears when no set is active
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Screen helper _active_review_set_banner: 'Reviewing: <name> - X of M · N reviewed · ✓ reviewed / · not yet reviewed' over one live snapshot; None when inactive - TDD via walker fakes\n2. LibraryMediaViewer gains review_banner param; composes a markup-safe one-line header when set (id library-media-review-banner); screen threads it from _build_library_media_reader\n3. m/walk already sync the viewer, so the banner updates in place - live verify
<!-- SECTION:PLAN:END -->
