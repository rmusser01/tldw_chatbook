---
id: TASK-36551
title: Compose exact paired portrait frames from bounded expression regions
status: To Do
created_date: 2026-10-24 14:24
dependencies:
- TASK-36550
labels:
- console
- characters
- portrait-compositing
references:
- backlog/decisions/242-console-radio-portrait-performance.md
documentation:
- Docs/superpowers/specs/2026-10-23-console-radio-portraits-design.md
- Docs/superpowers/plans/2026-10-24-portrait-compositing-implementation.md
- backlog/decisions/243-source-bound-portrait-compositing.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve portrait lighting and exact eye/mouth pairing by deriving the six playback poses from a single prepared base and reusable expression edits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Validated source-pixel rectangles identify image-left eye, image-right eye and mouth on the exact prepared base; malformed, out-of-bounds, overlapping and undersized regions are rejected before composition.
- [ ] #2 One base with open eyes and closed mouth plus eyelid/small-mouth/open-mouth edits produces exactly the six existing roles; base alpha and every pixel outside the masks remain unchanged, mouths match across blink pairs, and eyes match across mouth states.
- [ ] #3 Canonical prepared-base identity hashes domain-tagged decoded RGBA pixels and dimensions, independently of PNG encoding; equivalent encodings validate the same recipe, while changed dimensions or any RGBA byte including transparent-pixel RGB changes the identity.
- [ ] #4 Authoritative composition-snapshot replacement replaces all frames and components without merging; a fresh rest-only/zero-component snapshot cannot inherit a complete old set, and composed can_apply/frames publication requires all three current components and six current roles.
- [ ] #5 Component retries replace their snapshot atomically, preserve unaffected encoded frames exactly and never repeatedly feather finished patches; failed validation retains previous frames, components, provenance and readiness together.
- [ ] #6 Composition and hashing reject incompatible, animated, malformed or over-budget inputs, use existing shared decode reservations off the UI thread and release images, pixel buffers and reservations on success, cancellation and failure; discard releases candidate refs and image bytes stay out of routine repr/logs.
- [ ] #7 Existing manual six-frame imports use their separate readiness rule and remain usable without regions; pixel, encoding-equivalence, fresh-reset, failed-retry, transparent/non-square/edge-region and failure-cleanup controls verify the new contract.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
