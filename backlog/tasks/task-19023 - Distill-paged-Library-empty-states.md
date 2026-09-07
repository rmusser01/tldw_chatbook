---
id: TASK-19023
title: Distill paged Library empty states
status: Done
assignee: []
created_date: '2026-08-21 06:30'
updated_date: '2026-08-21 07:05'
labels:
  - library
  - ux
  - empty-state
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace zero-result paging and inactive selection mechanics in Media, Conversations, and Prompts with one truthful, source-specific recovery path while preserving filtered-zero and retained-error behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Fresh unfiltered zero-result Media, Conversations, and Prompts surfaces omit pager boundary mechanics and inactive selection copy.
- [x] #2 Each empty surface presents the approved production recovery action or actions appropriate to its source.
- [x] #3 Filtered-zero states preserve the active filter and offer clear reset recovery without claiming the source is empty.
- [x] #4 Loading errors and retained prior pages keep their existing truthful Retry and paging authority.
- [x] #5 Keyboard focus, accessibility, and compositor visibility pass at 100x30 and 170x48.
- [x] #6 Only touched-component and direct-owner tests are run; no repository-wide pytest claim is made.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A. This task applies ADR-067's source-owned pager authority and
ADR-076's lifecycle composition grammar without changing storage, service, or
cross-module ownership contracts.

Reason: the change is a source-specific presentation refinement over existing
exact totals, scopes, recovery actions, and route handlers.

1. Pin the four truthful presentation classes for each source: fresh
   unfiltered empty, fresh filtered empty, uninitialized/error, and retained
   stale/error.
2. In the Media, Conversations, and Prompts canvases, omit zero-boundary pager
   and inactive selection mechanics only for an authoritative fresh empty
   result. Keep current loading/error/stale rendering unchanged.
3. Add source-owned recovery actions per fresh empty state: Import media,
   start a Conversation in Console, or create/import a Prompt. For filtered
   empty states, preserve the active scope and expose one explicit reset action.
4. Route those controls through existing LibraryScreen navigation and request
   seams; do not add a generic empty-state widget, controller, or router.
5. Add focused pure/mounted regressions for copy, omitted controls, actual
   action dispatch, scope reset, retained recovery, semantic focus, and
   production-CSS compositor containment at 100x30 and 170x48.
6. Run only the modified component and direct-owner tests, Ruff on changed
   Python files, CSS parity only if CSS changes, and `git diff --check`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added source-owned fresh-empty branches for Media, Conversations, and
  Prompts. They retain the authoritative `(0)` title while removing inactive
  pager, export, and selection mechanics.
- Preserved filtered scope and literal no-match copy. Reset actions clear only
  the relevant type, query, or collection and return to page one through the
  existing source request seams.
- Reused existing destinations for Import media, Console conversation start,
  blank Prompt creation, and Prompt import. Prompt creation uses the existing
  primary-action treatment while Import remains secondary.
- Kept delete receipts, mutation state, loading, retained refresh failure,
  pager status, and Retry on their existing authoritative paths. In particular,
  a retained exact-zero page cannot mask a newer loading/error state.
- Focused verification: 67 selected canvas/direct-owner cases passed; 35
  selected LibraryScreen paging/filter/recovery/focus/geometry cases passed;
  the exact distilled contract subset passed 27 cases. The mounted production
  hierarchy uses `TldwCli.CSS_PATH` and proves source plus filtered recovery,
  compositor containment, literal scope visibility, semantic Tab reachability,
  and action dispatch at 100x30 and 170x48.
- Four one-at-a-time inverses failed as required and were immediately restored:
  continuing normal pager composition from a fresh zero; classifying a Media
  type-filtered zero as source-empty; removing retained status/Retry guards;
  and bypassing the existing Import destination seam.
- Ruff check passed on all eight changed Python files; `git diff --check`
  passed. All eight files have pre-existing whole-file Ruff formatter drift at
  the task base, so no unrelated bulk formatting was introduced. Impeccable's
  post-edit detector returned no findings. No CSS changed, so no bundle rebuild
  or parity claim was needed.
- ADR check: no new ADR. The implementation applies ADR-067 and ADR-076 without
  changing storage, service, controller, or route ownership.
- Per user direction, repository-wide pytest was not run; only modified/touched
  Library component and direct-owner gates are claimed.
<!-- SECTION:NOTES:END -->
