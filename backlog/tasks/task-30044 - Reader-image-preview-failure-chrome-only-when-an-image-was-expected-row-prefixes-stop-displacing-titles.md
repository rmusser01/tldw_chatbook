---
id: TASK-30044
title: >-
  Reader - image-preview failure chrome only when an image was expected; row
  prefixes stop displacing titles
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-03 13:06'
updated_date: '2026-09-03 13:59'
labels:
  - library
  - media-ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique 2026-09-03 P2: every plain document's Read tab leads with 'Image preview unavailable - showing complete stored text' plus a persistent Retry preview button above the content (the unavailable reason stamps the status even with no image type hint), and the wide-mode row prefix 'Loaded in Reader' consumes ~28 of ~35 label cells leaving titles as 'Quart'/'SQLit'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A plain document with no image type hint shows no preview-unavailable status and no Retry preview button
- [ ] #2 An item whose type indicates an image keeps the status and retry path
- [ ] #3 List rows keep their titles legible while carrying loading/loaded state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. image_preview_eligibility callers: stamp the unavailable status ONLY when an image was expected (type hint / image media_type) - TDD\n2. Row prefixes: wide-mode 'Loaded in Reader'/'Selected · loading preview' 28-cell prefixes become the compact short grammar so titles survive - TDD\n3. Live verify: plain document shows no failure chrome; loaded row keeps its title
<!-- SECTION:PLAN:END -->
