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
- [ ] #3 Composition rejects incompatible dimensions and animated/malformed/over-budget results, uses the existing shared decode reservations off the UI thread, and releases all buffers/reservations on success, cancellation and failure.
- [ ] #4 Component replacement updates its dependent frame group atomically in the disposable draft, preserves unaffected frames exactly, and never repeatedly feathers an already composited patch.
- [ ] #5 Existing manual six-frame imports remain usable without regions; pixel-oracle, transparent/non-square/edge-region, repeated-retry, partial-candidate and failure-cleanup tests cover the new composition contract.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
