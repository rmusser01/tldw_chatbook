---
id: TASK-21665
title: Add capability-gated local Media image preview
status: Done
assignee: []
created_date: '2026-08-24 16:17'
updated_date: '2026-08-24 16:33'
labels:
  - library
  - media
  - tui
dependencies: []
references:
  - Docs/superpowers/plans/2026-08-23-library-media-netnewswire-reader.md
  - backlog/decisions/084-library-media-reader-ia.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Render eligible local PNG, JPEG, and WebP originals above the complete stored Media text when terminal image capability is available, with honest text-preserving fallbacks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Local PNG/JPEG/WebP originals are eligible only after the existing local original-file check; remote/server and other formats never download.
- [x] #2 Capability-off, decode, and render failures preserve the complete stored text and expose honest item-local fallback or Retry copy.
- [x] #3 Preview loading is off-loop and generation-fenced so a late item A result cannot mount over loaded item B.
- [x] #4 Hide/show state is per item for the current screen session and does not reload detail.
- [x] #5 Mounted preview appears above byte-for-byte unchanged complete text and never changes item load success.
- [x] #6 Focused preview, viewer, and CSS integrity regression suites pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red pure eligibility and fallback tests for local-only supported images.
2. Implement a narrow preview helper using existing file, image-fit, graphics, and mosaic seams.
3. Add generation-fenced off-loop preview loading and per-item session visibility state without replacing stored text.
4. Add mounted order, retry, hide/show, stale-generation, and external-detail tests.
5. Add source CSS, regenerate bundles, run focused suites and required mutation inverses.
6. Self-review, document implementation, and close the task.

ADR required: no new ADR
ADR path: backlog/decisions/084-library-media-reader-ia.md
Reason: ADR-084 already fixes the local-only eligible formats, capability fallback, and complete-text authority; this task implements that accepted boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a lazy local-original eligibility/decode/render helper for PNG, JPEG, and WebP, reusing the existing graphics-or-mosaic terminal image path without adding a dependency or remote fetch path.
- Added an ephemeral per-screen preview cache, per-item hide/show state, item-local retry/fallback copy, and generation-fenced off-loop file/decode work. Preview failures never fail the Media item or replace its complete stored text.
- Rendered previews above the unchanged Reader text and added bounded source CSS, then regenerated the committed modular stylesheet.
- Added pure and mounted coverage for capability-off behavior, exact text preservation, remote/unsupported no-download gates, decode/render failure, retry, per-item visibility, and late A-after-B completion.
- Verification: 216 focused Media Reader regression tests passed; final preview/viewer-state run passed 80 tests; Ruff, compileall, CSS regeneration, and `git diff --check` passed. All three required mutation inverses failed at their intended assertions and were restored.
- ADR: existing [ADR-084](../decisions/084-library-media-reader-ia.md) applies; no new ADR was required. No new generalizable lesson was identified beyond the existing async projection and production-CSS lessons.
<!-- SECTION:NOTES:END -->
