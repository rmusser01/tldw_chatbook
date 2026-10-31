---
id: TASK-36551
title: Compose exact paired portrait frames from bounded expression regions
status: Done
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
assignee:
- '@codex'
updated_date: 2026-10-24 15:55
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Preserve portrait lighting and exact eye/mouth pairing by deriving the six playback poses from a single prepared base and reusable expression edits.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated source-pixel rectangles identify image-left eye, image-right eye and mouth on the exact prepared base; malformed, out-of-bounds, overlapping and undersized regions are rejected before composition.
- [x] #2 One base with open eyes and closed mouth plus eyelid/small-mouth/open-mouth edits produces exactly the six existing roles; base alpha and every pixel outside the masks remain unchanged, mouths match across blink pairs, and eyes match across mouth states.
- [x] #3 Canonical prepared-base identity hashes domain-tagged decoded RGBA pixels and dimensions, independently of PNG encoding; equivalent encodings validate the same recipe, while changed dimensions or any RGBA byte including transparent-pixel RGB changes the identity.
- [x] #4 Authoritative composition-snapshot replacement replaces all frames and components without merging; a fresh rest-only/zero-component snapshot cannot inherit a complete old set, and composed can_apply/frames publication requires all three current components and six current roles.
- [x] #5 Component retries replace their snapshot atomically, preserve unaffected encoded frames exactly and never repeatedly feather finished patches; failed validation retains previous frames, components, provenance and readiness together.
- [x] #6 Composition and hashing reject incompatible, animated, malformed or over-budget inputs, use existing shared decode reservations off the UI thread and release images, pixel buffers and reservations on success, cancellation and failure; discard releases candidate refs and image bytes stay out of routine repr/logs.
- [x] #7 Existing manual six-frame imports use their separate readiness rule and remain usable without regions; pixel, encoding-equivalence, fresh-reset, failed-retry, transparent/non-square/edge-region and failure-cleanup controls verify the new contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/243-source-bound-portrait-compositing.md
Reason: Implement the approved source-bound composition and atomic candidate contracts; no new architectural decision.

1. Read the task, revised spec/Unit 1 and existing normalization, shared-budget and authoring callers.
2. Write and run failing canonical identity, exact pixel-pair, strict region/input and snapshot reset/readiness tests.
3. Implement bounded Pillow composition/recovery and atomic immutable composition snapshots, preserving manual import behavior.
4. Verify transparent/non-square/edge regions, repeated finished-plane reuse, malformed/animated/over-budget inputs, cleanup and failed-snapshot retention.
5. Run affected core/authoring and existing caller regressions, Ruff/format and diff checks; update architecture/task evidence, self-review and commit before TASK-36552.
<!-- SECTION:PLAN:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented bounded portrait_compositing core and immutable draft candidate replacement under ADR-243. Canonical tagged RGBA identity ignores PNG encoding, strict rectangles/masks preserve base alpha/outside pixels, finished-plane assembly/recovery enforce exact pairs without repeated feathering. Atomic composed snapshots replace old slots; manual merges and composed readiness remain distinct; failed edits retain lineage and discard releases bytes.

Red-to-green evidence: 18 missing-contract failures, then two reproduced invalid-input/empty-edit edge failures. Final core/authoring tests: 27 passed; affected legacy generation/motion/mounted Personas caller gate: 58 passed before additional verification-only controls. Full 1024x1024, foreign-budget refusal, set failure, success/OSError/cancellation image closure and unaffected encoded retry frames pass. Ruff lint/format and diff checks pass; targeted tests only.

Files: new Character_Chat/portrait_compositing.py, updated portrait_performance_authoring.py, new compositor tests and extended authoring tests; Architecture/character-chat.md, implementation-plan checkpoint and atomic review. Self-review: Docs/superpowers/reviews/2026-10-24-task-36551-review.md. Existing ADR-243 applies; no new decision/dependency/provider/DB/TTS/UI behavior. Region storage and generation integration remain later tasks.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
