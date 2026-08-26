---
id: TASK-22305
title: Simplify Console change review around per-turn cards
status: Done
assignee:
  - '@codex'
created_date: '2026-08-26 06:41'
updated_date: '2026-08-26 18:04'
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
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add red-first card tests for Undo All visibility, loading, busy, success, and preserved Review/file rows.
2. Add red-first screen orchestration tests for confirmation, off-thread I/O, active-run/provider failures, multi-root support, and duplicate-root Review refusal.
3. Implement the card event/state and screen-owned guarded Undo All flow by reusing the existing provider, modal, and revert engine.
4. Remove the Inspector Changed Files composition, screen worker/cache/guard/invalidation paths, and related constructor contracts.
5. Delete rail-only widget, projection/provider aggregation, store memo, tests, config docs, and regenerate CSS/inventories.
6. Run focused suites, the 417-node Console change-review baseline, Ruff/CSS/inventory gates, production-shaped visual/live verification, and self-review.

ADR required: yes
ADR path: backlog/decisions/089-console-per-turn-change-review-ownership.md
Reason: this changes long-lived Console ownership and removes an Inspector information-architecture surface in favor of turn-owned review.

Detailed plan: Docs/superpowers/plans/2026-08-25-console-turn-change-review-simplification.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented turn-owned changed-file review with a visible guarded Undo All action, confirmation-time and mutation-time safety checks, unambiguous multi-root labels, same-root multi-window refusal into Review, honest partial/provider failure handling, and preserved historical cards. Removed the Inspector cross-turn Changed files widget, worker/cache/config/store memo/projection and regenerated CSS plus diagnostic inventory. Integrated on origin/dev ad16ed8158 with ADR-089. Verification: final 401-test Console/change-review baseline passed; latest-dev overlap suite passed 360/361 before its stale opener contract was repaired, then focused opener/Undo passed 12/12; final persistent diagnostic inventory suite passed 65/65; Ruff lint, formatter checks, CSS reproduction, inventory verification, dead-symbol search, and git diff checks passed. Production-CSS compositor checks verified wide and narrow card action reachability, cancellation, real disk restoration, and Undone history preservation.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Console file-change review is now turn-owned: users keep per-turn file lists, diffs, notes, and Review, gain safe Undo All on every turn card, and no longer pay for the redundant Inspector aggregation.
<!-- SECTION:FINAL_SUMMARY:END -->
