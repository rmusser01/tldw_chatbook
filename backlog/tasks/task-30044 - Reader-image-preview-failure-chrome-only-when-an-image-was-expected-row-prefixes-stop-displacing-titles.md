---
id: TASK-30044
title: >-
  Reader - image-preview failure chrome only when an image was expected; row
  prefixes stop displacing titles
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
Critique 2026-09-03 P2: every plain document's Read tab leads with 'Image preview unavailable - showing complete stored text' plus a persistent Retry preview button above the content (the unavailable reason stamps the status even with no image type hint), and the wide-mode row prefix 'Loaded in Reader' consumes ~28 of ~35 label cells leaving titles as 'Quart'/'SQLit'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A plain document with no image type hint shows no preview-unavailable status and no Retry preview button
- [x] #2 An item whose type indicates an image keeps the status and retry path
- [x] #3 List rows keep their titles legible while carrying loading/loaded state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. image_preview_eligibility callers: stamp the unavailable status ONLY when an image was expected (type hint / image media_type) - TDD\n2. Row prefixes: wide-mode 'Loaded in Reader'/'Selected · loading preview' 28-cell prefixes become the compact short grammar so titles survive - TDD\n3. Live verify: plain document shows no failure chrome; loaded row keeps its title
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2351 (dev 72474d391). image_preview_expected (MIME hint or media_type/type == image) gates the unavailable status: plain documents show no failure chrome or Retry button (AC1); image-typed items keep the status/retry path (AC2, truth-table pinned); an item that STOPS being image-expected sheds any previously stamped status (Qodo round). Row prefixes: both densities use 'Loaded · '/'Loading · ' so titles survive (AC3). Live-verified.
<!-- SECTION:NOTES:END -->
