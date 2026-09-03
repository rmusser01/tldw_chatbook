---
id: TASK-30042
title: >-
  Review sets - no silent failures: every gesture responds, storage health
  surfaces
status: Done
assignee:
  - '@claude'
created_date: '2026-09-03 13:05'
updated_date: '2026-09-03 14:31'
labels:
  - library
  - media-ux
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique 2026-09-03 P1 + explicit user ruling: silent failure is never acceptable. Today service-is-None paths return silently (picker press, create, walk), auto-resume swallows all exceptions, and a wedged collections DB is indistinguishable from the feature not existing - against the product's own no-hidden-recovery-states commitment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Pressing Sets always responds: with unavailable storage it says so instead of doing nothing
- [x] #2 Create/walk/toggle/exit/picker/auto-resume surface an error notice on any failure instead of returning silently
- [x] #3 No review-set code path swallows an exception without user-visible feedback
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Explicit gestures (Sets press, Review these/selected create) notify 'Review-set storage is unavailable.' when service is None - TDD\n2. walk/toggle/exit wrapped: exceptions notify instead of propagating/silent - TDD\n3. Auto-resume + liveness failures notify once per session (guarded) - TDD\n4. Live spot-check
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shipped in PR #2346 (dev 7b7d742bc). Sets press / create with no storage notify _REVIEW_SET_STORAGE_UNAVAILABLE (error); walk/toggle/exit wrap storage errors into notices (walk consumes the key); auto-resume notifies on error and warns once per session on missing storage; every wrapper logs the traceback (diagnostic inventory reviewed + regenerated). Only the ordinary no-active-set case stays quiet.
<!-- SECTION:NOTES:END -->
