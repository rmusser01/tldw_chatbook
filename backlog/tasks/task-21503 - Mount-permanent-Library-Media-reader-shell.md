---
id: TASK-21503
title: Mount permanent Library Media reader shell
status: Done
assignee:
  - '@codex'
created_date: '2026-08-24 05:14'
updated_date: '2026-08-24 06:55'
labels:
  - library
  - media
  - ui
  - testing
dependencies:
  - TASK-21471
references:
  - Docs/superpowers/plans/2026-08-23-library-media-netnewswire-reader.md
  - Docs/superpowers/specs/2026-08-23-library-media-netnewswire-reader-design.md
  - backlog/decisions/084-library-media-reader-ia.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mount the permanent three-role Library Media reader shell so Library navigation, Media Items, and Reader remain usable together while responsive and manual collapse preserve Reader priority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Media mounts Library, Items, and permanent Reader together while non-Media Library routes retain their existing shell.
- [x] #2 Library and Items each have one fixed five-column full-height grip with truthful expanded/collapsed action copy, pointer activation, keyboard activation, tooltip, and accessibility label.
- [x] #3 Shell-width resolution uses the Task 1 state contract, updates geometry in place, keeps Reader permanent and contained through the 60-column floor, and never invokes Media services or replaces list identity during resize.
- [x] #4 Normal Media row activation keeps Items mounted, begins the existing immediate exclusive detail load, and settles matching detail into the permanent Reader without regressing multi-select, Trash, Back, or viewer actions.
- [x] #5 Production-shaped focused tests, compositor geometry evidence, CSS bundle gates, inverse mutations, compilation, whitespace checks, and self-review pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add production-shaped RED tests for the Media shell roles, truthful five-column grips, accessibility, activation, containment, resize call ledger, and non-Media inverse.
2. Implement the smallest Library-local shell and grip widgets with in-place layout synchronization.
3. Convert only the Media compose and viewer-sync paths to retain Items and permanent Reader while preserving current immediate detail workers and existing actions.
4. Resolve settled shell widths through the Task 1 session/preferences contract, handle manual toggles and focus transfer without persistence or service calls, and add only Media-local source CSS.
5. Regenerate CSS outputs, run focused GREEN and inverse verification, self-review, document evidence, and close the task.

ADR required: yes

ADR path: backlog/decisions/084-library-media-reader-ia.md

Reason: directly implements ADR-084’s accepted permanent Reader and preferred-versus-responsive pane layout contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Mounted a Media-only three-role shell that retains the Library rail, Items canvas, and Reader while two fixed five-column grips apply Task 1 layout decisions in place. Other Library destinations keep the incumbent shell.
- Preserved the existing exclusive detail worker and Media actions while replacing Media list/viewer child swaps with selected-versus-loaded session settlement. Added current-generation same-route reconciliation so a superseded entry worker cannot leave the permanent Reader stale.
- Added Media-local source CSS and regenerated the committed bundle. The focused aggregate passed with 104 tests (88 deselected); the dedicated shell suite passed 11/11; Ruff lint, compilation, bundle reproduction, and `git diff --check` passed. Ruff format check reports existing whole-file drift in six legacy files, so unrelated mechanical formatting was intentionally excluded.
- Inverse evidence: forcing the Reader's actual inline minimum to 54 columns failed the direct 60-column contract (54 rendered versus 50 expected); replacing in-place resize sync with `refresh(recompose=True)` failed the permanent shell/Items identity contract. The plan's literal CSS `min-width: 44` mutation was not a valid inverse because 44 fits the 50-column allocation and inline geometry is the owning cascade.
- Spec review passed with no gaps. The bounded independent quality reviewer exceeded its one-minute cap and was stopped; direct diff review found no blocker. ADR-084 remains the governing decision.
<!-- SECTION:NOTES:END -->
