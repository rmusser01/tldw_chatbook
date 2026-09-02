---
id: TASK-28226
title: Harden Console row-menu dismissal from workspace markers
status: Done
assignee:
  - '@Codex'
created_date: '2026-09-02 05:41'
updated_date: '2026-09-02 06:25'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ensure the workspace and chat action popups opened from the right-edge `@` and
`*` markers always honor the established Escape and click-outside dismissal
contract, including priority hands-free key routing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Root `@` and `*` menus close on Escape before hands-free exits.
- [x] #2 Submenu Escape returns to the root before a second Escape closes it.
- [x] #3 Clicking outside either popup dismisses it without dispatching an action or restoring opener focus.
- [x] #4 Real marker-pointer paths have regression coverage.
- [x] #5 Targeted tests and lint pass.
- [x] #6 Escape focus restoration leaves the two-row workspace tree and its markers visible.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a mounted Console regression proving an open row action menu claims the
   priority hands-free Escape before the loop exits.
2. Make the priority hands-free Escape binding yield while either row menu is
   live so the existing menu and screen fallback handlers retain ownership.
3. Extend the real Workspaces-tree marker tests to cover `@` and `*` dismissal
   through outside click and root/submenu Escape without changing focus or
   dispatch semantics.
4. Run the two row-menu suites, seam-adjacent hands-free tests, Ruff, and diff
   checks; record implementation evidence and complete the acceptance criteria.
5. Reproduce the live-UAT focus-outline clipping in a two-row tree, add a
   computed-style regression, and keep the existing cursor cue while removing
   the overpainting outline.

ADR required: no
ADR path: `backlog/decisions/068-console-text-selection-and-annotations.md`
Reason: this is a direct regression fix under the existing Console overlay
dismissal contract and changes no storage, ownership, security, or interface
boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Gated the priority hands-free Escape binding while a live workspace or
  conversation row menu is mounted, allowing the existing submenu-back,
  root-close, and stranded-focus handlers to run before hands-free exits.
- Reused the menus' weak registries and ignored nodes already scheduled for
  pruning, avoiding both a Console DOM walk and a transient swallowed Escape
  immediately after dismissal.
- Added mounted production-style tests that click the real `@` and `*` tree
  markers, cover menu-focused and stranded-focus Escape ordering, and prove
  outside clicks neither dispatch actions nor steal focus from the click.
- Real-terminal UAT used the full app, isolated scratch profile, actual `@` and
  `*` tree markers, and terminal mouse events. It confirmed root Escape close,
  submenu-to-root then close on two Escapes, and outside-click dismissal while
  leaving the composer focused. The inspector remained in the Console context.
- UAT exposed the global four-sided focus outline overpainting both rows of a
  compact workspace tree after Escape restored focus. Added a production-style
  two-row rendering regression and disabled that outline while retaining the
  tree's existing focus background and cursor cues; regenerated the canonical
  Console CSS bundle.
- Verified the two row-menu suites plus Console popup etiquette (42 tests),
  Ruff on the touched Python files, the CSS generator, and `git diff --check`.
  The only warning is the repository environment's pre-existing Requests
  dependency warning. A broader 173-test check produced 171 passes and two
  failures proven unchanged on `origin/dev`: a stale inspector-handle CSS
  expectation and an integer-CSS helper that rejects existing `!important`
  declarations.
- ADR required: no. The implementation follows
  `backlog/decisions/068-console-text-selection-and-annotations.md`; no new
  storage, ownership, security, or interface decision was introduced.
- No new lesson was added: the existing live-verification guidance already
  covers checking focused edge rows in the real terminal, and this incident is
  now captured by the rendering regression and task record.
<!-- SECTION:NOTES:END -->
