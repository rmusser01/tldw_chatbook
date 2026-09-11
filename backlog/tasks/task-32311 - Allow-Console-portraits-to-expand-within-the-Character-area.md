---
id: TASK-32311
title: Allow Console portraits to expand within the Character area
status: Done
assignee:
  - '@codex'
created_date: '2026-09-11 02:15'
updated_date: '2026-09-11 03:36'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Selected character portraits stop growing before filling the available image area. Scale the complete portrait up or down without distortion while keeping Character controls visible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Small and large portraits use the available image box with preserved proportions and no cropping
- [x] #2 Graphics and mosaic rendering agree and resize without clipping or hiding controls
- [x] #3 Targeted avatar tests and static checks pass
- [x] #4 Character body grows with a taller terminal and shrinks back on resize, reserving measured control rows
- [x] #5 After portrait and control updates settle, the left rail remains stationary without repeated fitting or scrolling
- [x] #6 Saved-conversation background refresh uses a 10-second cache interval while explicit invalidation remains immediate
- [x] #7 Mounted cache-refresh coverage proves stable rows through worker completion and rail reconciliation recovers from a missing Character header
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing decision)
ADR path: backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md
Reason: user requested viewport-responsive Character section sizing, replacing its fixed maximum while preserving existing ownership.

1. Reproduce source-size cap with failing tests.
2. Remove source-pixel caps from Character contain geometry.
3. Amend ADR-083 and make the Character body ceiling follow the measured outer viewport, minus its header, with the existing 35 rows as a compact fallback. Use that same budget for portrait fitting after subtracting controls.
4. Add grow/shrink mounted regression and update fixed-ceiling assertions. Run targeted avatar and rail tests and static checks; document behavior.
5. Investigate live rail motion. Keep the last matching persisted conversation rows visible during TTL refresh, verify query/invalidation boundaries, and confirm stability in the actual textual-serve app. Routine cache-display fix: no additional ADR required.
6. Raise the saved-conversation cache TTL to 10 seconds as requested; retain explicit invalidation and verify existing cache-expiry tests. No additional ADR required for this polling-interval adjustment.
7. Address Qodo review: add a mounted cache-expiry/worker-completion regression, guard Character header lookup during partial recomposition with a recovery test, and correct stale interval documentation. Routine fixes within ADR-083; no new ADR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on codex/console-portrait-fit from freshly fetched origin/dev ed6fd5db0a.

Character portraits now enlarge or shrink with an aspect-preserving contain fit. The Character body follows the measured Context viewport minus its header (35-row compact fallback), and the image uses only rows left after controls. Changes to measured control geometry start a fresh fit epoch so asynchronous search expansion cannot leave an oversized stale portrait. Graphics and mosaic continue to share the same box and off-loop rendering.

Amended existing ADR-083: backlog/decisions/083-console-edge-rails-and-workspace-tree-ownership.md. Updated the Console user guide, portrait regression tests, and the rail tests to the existing production section order. No dependencies or storage changes.

Validation: initial new small-portrait tests failed at 8x4 vs expected 40x20 cells; the resize regression failed at unchanged 24-row height. With the fix, all 92 avatar/off-loop tests passed, and the ready-Console resize regression passed separately. Visual captures at 180x72 and 180x144 show the complete bordered test image at 11x44 and 25x100 cells, preserving circular markers. Ruff lint/format, diff whitespace, and Backlog ID checks pass.

The broader rail run exposed stale order expectations (corrected to the shipped Character placement) and the unrelated test_bounded_rail_shell_regions_are_compositor_contained[160x45] Inspector Sources hint hit-test failure. That same hit-test failure and old order expectations were reproduced on an untouched archive of ed6fd5db0a. Final rail verification: 49 passed / 2 compositor cases deselected; the larger compositor case passed in the combined run, and the smaller case is the reproduced baseline failure. Tests used NO_COLOR unset with TERM=xterm-256color and COLORTERM=truecolor so the existing color-paint assertion could exercise color output. Full repository suite was not run.

Live verification follow-up: ran the actual application through its built-in textual-serve path using an isolated copy of the local profile and selected the existing Samira conversation. The user reported repeated left-rail movement. Tracing proved the persisted conversation cache returned no rows at its two-second TTL boundary, making Conversations alternate between 53 and 13 rows while portrait geometry stayed constant. The sync accessor now retains the last matching query/selection result while its existing worker refreshes; explicit invalidation and different keys still clear the display. No additional ADR is required for this routine display bug fix.

Added three cache-expiry/query/selection regressions. Red: same-key expiry returned an empty list. Green: 131 targeted controller, tray guard, and tick tests passed; five unrelated failures reproduced on untouched dev ed6fd5db0a (constructor dependency documentation and four saved-chat opener cases). Changed-line Ruff formatting, Ruff lint, and diff checks pass. Two actual served browser captures 22.9 seconds apart have identical pixels across the full left rail, with Samira's portrait visible. Evidence: /private/tmp/rail-stable-first.png and /private/tmp/rail-stable-second.png. Removed temporary diagnostics. Added the populated-profile/cache-expiry verification lesson.

Requested refresh-rate follow-up: raised the persisted conversation cache TTL from 2 to 10 seconds. The screen now re-exports the controller constant instead of retaining a duplicate 2-second value. Explicit invalidation, streaming, portrait geometry, and the separate appearance cache retain their behavior. Seven targeted cache/expiry/invalidation tests pass; Ruff lint, changed-line formatting, and diff checks pass. The running browser session will use the new interval after its next app restart.

Qodo review follow-up: added a mounted ready-Console integration regression covering cache expiry, stable row identity/visible bounds/rail geometry while a real refresh worker is blocked, and publication of changed data after completion. Removing the stale-row fallback reproduced disappearing rows. Guarded the Character header lookup during prepare alongside the bounded-section lookup; the new partial-recomposition regression reproduced NoMatches before the fix and verifies flag recovery plus subsequent resizing. Removed the stale numeric TTL claim from transcript-timer documentation. Twelve targeted tick, recompose, and portrait tests pass; Ruff lint/format and diff checks pass.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Renumbered from TASK-32301 after rebasing onto dev 0fcb79e596. The Library list-entry focus task was created on 2026-09-10 at 21:30; this task was created on 2026-09-11 at 02:15. The older task retains TASK-32301 under the TASK-19601 owner rule. Portrait task references in ADR-083 and the live-verification lesson now use TASK-32311.
