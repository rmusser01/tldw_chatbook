---
id: TASK-32023
title: Animate Console character expressions with Dynamic and Static settings
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 03:03'
updated_date: '2026-09-10 15:23'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Allow characters to show animated expressions while letting users choose static expression playback and respecting motion accessibility preferences.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Dynamic displays animated expression frames in a mounted Console avatar while Static changes expressions using frame zero.
- [x] #2 F9 save cancel restart and global motion preferences preserve manual reaction precedence and update the same selected asset.
- [x] #3 Preparation and playback are bounded off-thread and reject stale results across authority session geometry and lifecycle changes.
- [x] #4 Targeted tests and mounted rendering evidence cover timing disposal fallback settings and unchanged exported bytes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Follow Docs/superpowers/plans/2026-09-07-character-expression-playback.md. ADR required: yes. ADR path: backlog/decisions/144-character-expression-playback.md. Reason: motion preference and avatar lifecycle. Implement settings and policy, bounded decoding and timeline, then mounted Console integration with targeted product evidence.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Dynamic/Static Console character expressions through the canonical F9 Appearance settings, preserving manual reactions, global motion preferences and immutable source/export bytes. Added bounded off-thread GIF/WebP/APNG preparation, a disposable mounted avatar, fixed-layout graphics updates, overlay-aware clocks, neutral/text fallback and fail-soft rendering/geometry teardown. No database migration or new dependency.

Followed ADR-144 (backlog/decisions/144-character-expression-playback.md). Plan and verification: Docs/superpowers/plans/2026-09-07-character-expression-playback.md and Docs/superpowers/reviews/2026-09-07-character-expression-playback-verification.md. Logical implementation steps are committed together as a complete feature. TASK-32013 was renumbered to TASK-32023 after a concurrent main-checkout collision.

Validation: final lifecycle/geometry/Console run 19 passed; F9 Appearance run 11 passed; pure decoder/model coverage passed, including eight decode/release memory-probe cycles. Native Actor Pack bytes are identical before/after Static; mounted captures prove red/blue/red frames. New code passes Ruff; modified existing files introduce no new Ruff diagnostics, changed ranges formatted and diff-check clean. The broader run exposed mounting and teardown races, which were fixed and verified in the final focused run. Existing focused-input width failure reproduces on unchanged runtime and is documented; color-only geometry requires a color-capable harness. No full-suite claim or live Kitty/Sixel protocol claim. Self-review completed and the overlay-verification incident recorded in lessons-live-verification.md.

2026-09-10 integration onto dev 16c72b5b1e: prior validation above is historical source-branch evidence; combined-code targeted qualification is in progress.

2026-09-10 integration complete on dev 16c72b5b1e, preserving source history. Playback ADR renumbered to 144 after cached-ref collision scan. Combined-code evidence and exact targeted commands: Docs/superpowers/reviews/2026-09-10-buddy-feature-integration-verification.md. F9 Appearance 14 passed; mounted selection 262 passed with 11 Petdex constant-import failures fixed and reverified in the final 206-pass run. Prior notes remain historical source-branch results; no full-suite claim.
<!-- SECTION:NOTES:END -->
