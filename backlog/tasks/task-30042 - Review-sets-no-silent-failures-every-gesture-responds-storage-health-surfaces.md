---
id: TASK-30042
title: >-
  Review sets - no silent failures: every gesture responds, storage health
  surfaces
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-03 13:05'
updated_date: '2026-09-03 13:07'
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
- [ ] #1 Pressing Sets always responds: with unavailable storage it says so instead of doing nothing
- [ ] #2 Create/walk/toggle/exit/picker/auto-resume surface an error notice on any failure instead of returning silently
- [ ] #3 No review-set code path swallows an exception without user-visible feedback
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Explicit gestures (Sets press, Review these/selected create) notify 'Review-set storage is unavailable.' when service is None - TDD\n2. walk/toggle/exit wrapped: exceptions notify instead of propagating/silent - TDD\n3. Auto-resume + liveness failures notify once per session (guarded) - TDD\n4. Live spot-check
<!-- SECTION:PLAN:END -->
