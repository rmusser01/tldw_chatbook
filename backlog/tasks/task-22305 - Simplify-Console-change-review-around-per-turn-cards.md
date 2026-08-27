---
id: TASK-22305
title: Simplify Console change review around per-turn cards
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 06:41'
updated_date: '2026-08-26 20:31'
labels:
  - console
  - change-review
  - ux
  - performance
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Console file-change review useful without the redundant cross-turn Inspector aggregation: keep per-turn changed-file cards and the full Review screen, add a direct guarded Undo All action to each turn card, and retire the rail-only polling/cache surface.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each changed-file turn card keeps its file list, expandable diffs, notes, and Review action and adds a visible **Undo All** button.
- [x] #2 **Undo All** confirms before mutation and names files changed since the turn before allowing overwrite.
- [x] #3 **Undo All** runs preflight and revert I/O off the Textual UI thread, prevents duplicate dispatch, and reports per-file failures honestly.
- [x] #4 **Undo All** supports ordinary single-window multi-root turns but refuses same-root multi-window turns without touching disk and opens that turn in Review.
- [x] #5 Active-run refusal and provider/tracking failures touch no files and produce honest user-facing copy.
- [x] #6 The Inspector no longer mounts or computes a cross-turn Changed files section, and its worker, cache, config, documentation, and dead supporting code are removed.
- [x] #7 Live and resumed change markers still render the same per-turn card, and the full Review screen remains reachable by the card button and keyboard action.
- [x] #8 Production-shaped automated tests and live visual verification cover card actions, refusal, confirmation, success, and the absence of the Inspector section.
- [x] #9 Review-screen preflight and revert file I/O run off the Textual UI thread.
- [x] #10 Undo and Review refuse mutations when any live Console session targets an affected workspace root, even when that session is not currently viewed.
- [x] #11 A later multi-root failure preserves and reports earlier successful restores and explicitly identifies every unprocessed path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red-first card tests for Undo All visibility, loading, busy, success, and preserved Review/file rows.
2. Add red-first screen orchestration tests for confirmation, off-thread I/O, active-run/provider failures, multi-root support, and duplicate-root Review refusal.
3. Implement the card event/state and screen-owned guarded Undo All flow by reusing the existing provider, modal, and revert engine.
4. Remove the Inspector Changed Files composition, screen worker/cache/guard/invalidation paths, and related constructor contracts.
5. Delete rail-only widget, projection/provider aggregation, store memo, tests, config docs, and regenerate CSS/inventories.
6. Run focused suites, the 417-node Console change-review baseline, Ruff/CSS/inventory gates, production-shaped visual/live verification, and self-review.
7. Add red-first regressions for Review-screen thread use, background-session root conflicts, and multi-root partial completion.
8. Route live execution-context roots through the existing provider/revert seams and preserve per-path outcomes across later-root failures.
9. Re-run focused review/controller tests and repository gates, then resolve the Qodo threads.

ADR required: yes
ADR path: backlog/decisions/089-console-per-turn-change-review-ownership.md
Reason: this changes long-lived Console ownership and removes an Inspector information-architecture surface in favor of turn-owned review.

Detailed plan: Docs/superpowers/plans/2026-08-25-console-turn-change-review-simplification.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented turn-owned changed-file review with a visible guarded Undo All action, confirmation-time and mutation-time safety checks, unambiguous multi-root labels, same-root multi-window refusal into Review, honest partial/provider failure handling, and preserved historical cards. Removed the Inspector cross-turn Changed files widget, worker/cache/config/store memo/projection and regenerated CSS plus diagnostic inventory. Rebased onto origin/dev 3daa56bf4f with ADR-089 and preserved the newer ADR-091 inventory. Addressed all three Qodo findings: Review preflight/revert hashing now runs off the Textual thread; immutable dispatch roots provide a cross-session root-aware mutation guard for card undo, Review revert, and commit; and a later multi-root exception retains earlier successes while naming every unprocessed path. Updated the shared Console test fixture to current durable-store and typed-destination contracts. Verification: final 401-test Console/change-review baseline passed before the review round; focused Qodo safety set passed 54/54; complete Change Review screen/commit suites passed 80/80; persistent diagnostic inventory suite passed 65/65; Ruff lint, CSS reproduction, inventory verification, dead-symbol search, and git diff checks passed. Production-CSS compositor checks verified wide and narrow card action reachability, cancellation, real disk restoration, and Undone history preservation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console file-change review is now turn-owned: users keep per-turn file lists, diffs, notes, and Review, gain safe Undo All on every turn card, and no longer pay for the redundant Inspector aggregation.
<!-- SECTION:FINAL_SUMMARY:END -->
