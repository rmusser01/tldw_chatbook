---
id: TASK-19024
title: Simplify Library Prompt editing
status: Done
assignee: []
created_date: '2026-08-21 07:09'
updated_date: '2026-08-21 08:18'
labels:
  - library
  - ux
  - prompts
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give first-time users a concise Prompt editor while preserving exact structured Prompt data, safety states, and efficient lifecycle actions for returning users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Eligible Prompts default to a concise Basic view without changing their stored block representation.
- [x] #2 Advanced remains available, while incompatible or safety-sensitive Prompts force an explained Advanced view without overwriting the remembered preference.
- [x] #3 Basic edits preserve block identities, ordering, metadata, version history, and ordinary save/conflict behavior.
- [x] #4 New, clean, dirty, conflict, and mutation states expose only lifecycle-valid actions with guarded recovery.
- [x] #5 Mode and action disclosure preserve draft content, native focus, undo, and scroll across supported terminal sizes.
- [x] #6 Only touched-component and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A. This task implements the Prompt-specific disclosure and
lifecycle composition already accepted by ADR-076; it does not change Prompt
storage, service, safety, or versioning ownership.

1. Add pure Basic eligibility and preference coercion over the existing
   `PromptEditorState` and immutable block working copy.
2. Keep Basic and Advanced regions mounted over one draft and switch them with
   targeted display updates so draft content, focus, undo, and scroll survive.
3. Persist one Prompt-only profile preference while deriving temporary forced
   Advanced presentation for incompatibility, conversion, conflict, or unsafe
   update states.
4. Replace overlapping global actions with a lifecycle-valid action strip and
   an inline More actions disclosure that routes to existing handlers.
5. Prove exact representation round-trip, safety overrides, mutation/error
   behavior, and production geometry with touched/direct-owner tests only.
6. Update Prompt documentation, record inverses/static evidence, review, and
   close through Backlog CLI.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a profile-remembered Basic/Advanced Prompt presentation over the
  existing immutable block working copy. Basic edits the exact singleton
  System/User blocks; Advanced retains structured blocks, metadata,
  Collections, and retained history. Recipes, multi-block or compatibility
  artifacts, conflicts, and unsafe updates force an explained Advanced view
  without replacing the remembered preference.
- Kept both editor regions mounted and switched only their display state.
  Lifecycle actions now expose only the valid new, clean, dirty, conflict, or
  busy operations; secondary actions use one inline More actions disclosure.
  Existing save, conflict, conversion, membership, history, export, Console,
  delete, and undo handlers remain the owners.
- Updated the ASCII-only Prompt guide and added one bounded Advanced-extras
  CSS rule so retained-history actions remain visible in compact terminals.
- TDD evidence: the final touched-owner selector passed **159 tests** with
  **404 deselected**. Required one-at-a-time inverses each failed the intended
  node and were restored: multi-block Basic admission, mode-switch recompose,
  forced-mode preference overwrite, Delete on a new draft, and flattened
  Basic block replacement.
- Static evidence: Ruff lint passed all seven changed Python owners/tests;
  the three previously conforming Python files pass Ruff format. Four large
  legacy owners fail Ruff format identically at the task base and were not
  bulk-reformatted. CSS source/bundle parity and `git diff --check` passed.
  Impeccable review covered focus, disabled/error states, copy, and the exact
  100x30/170x48 layouts; the compact retained-history clipping it exposed is
  fixed and regression-tested.
- ADR required: no. ADR-076 already owns this Prompt disclosure and lifecycle
  structure; no storage, service, safety, or cross-source boundary changed.
- Per user direction, repository-wide pytest was not run; only modified Prompt
  component and direct-owner tests are claimed.
<!-- SECTION:NOTES:END -->
