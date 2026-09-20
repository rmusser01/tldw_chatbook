---
id: TASK-32823
title: Keep selected MCP tool details current during catalog refresh
status: Done
assignee:
  - '@codex'
created_date: '2026-09-18 21:46'
updated_date: '2026-09-20 18:03'
labels:
  - mcp
  - ui
  - selection
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A refreshed Tools catalog must not leave obsolete selected-tool metadata or test controls in the inspector, and unchanged catalogs must preserve the user argument draft and focus.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Catalog refresh updates the selected tool details and clears removed tools without selecting a replacement.
- [x] #2 Unchanged definitions preserve the mounted argument draft and focus; changed definitions retire old test controls and permission previews with visible guidance.
- [x] #3 Targeted refresh, selection and prepared-test regressions plus bounded native evidence qualify the behavior; review ledgers and the draft PR record remaining scope.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. ADR path: backlog/decisions/161-component-pattern-library.md, backlog/decisions/170-table-repopulation-selection-boundary.md and backlog/decisions/032-local-agent-tool-permission-boundary.md. Reason: repair selected-detail projection and preview ownership within existing UI and service admission boundaries. 1. Resume saved commit 135f226888 from verified PR2730 merge 802809947b; reproduce the saved four catalog cases on this baseline. 2. Port the bounded refresh repair while preserving current permission navigation, then deterministically reproduce and fix delayed preview publication during awaited teardown. 3. Verify unchanged form/raw drafts and focus, changed/removed definitions, newer selection/focus, and adjacent prepared-preview behavior with targeted tests. 4. Complete bounded private-profile native dark/light captures, independent review, lifecycle and artifact guards; update ledgers and save a follow-up draft PR against dev. Final visual approval remains required for this new PR.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Selected MCP tool detail now reconciles catalog refresh without replacing unchanged argument forms, cursor, focus or previews. Changed/removed definitions retire prior controls, clear detail or show reopen guidance, and synchronously invalidate preview ownership before awaited teardown. Queued refresh and late completions preserve newer selection/focus; failed-context successors also retire prior workers. Port preserves current permission navigation and recovery guards. 475 targeted passes; one unchanged Workflows dimension-ratchet failure reproduced on merged baseline802809947b. Seven artifact guards, Ruff checks with no introduced diagnostics, independent review, and eight inspected current-source native captures passed. Evidence: Docs/superpowers/qa/2026-09-20-mcp-inspector-refresh/README.md. Existing ADR-161/170 and ADR-032 apply, no new ADR. Saved as draft PR2757 against dev. Current-head CI/Qodo and final visual approval remain; keep In Progress until closeout. Connected-runtime and compact/long-path work remain outside this task.

2026-09-20 owner approved PR2757 gallery and merge. Conflict-free current-dev integration 6e9e94c794 keeps all approved MCP sources/styles unchanged; all 13 refresh regressions pass again. Native source/runner hashes still match. PR is ready for review; current-head CI/Qodo and exact merge-tree verification remain. See approved-integration.json.

Qodo closeout plan: verify the alleged mount race against synchronous Textual registration; add focused permission-input refresh regressions and isolated reconciliation coverage; document the public refresh/token contract. Fix only reproduced ownership/currentness defects under existing ADR-161/170 and ADR-032, then rerun targeted tests and review.

Qodo follow-up: reproduced and fixed policy-only refresh equality; current effective state, exact-input rules and displayed profile-wide session grants participate. Added public API docs, 3 isolated reconciliation cases, 3 policy component cases and 2 real mount-boundary cases. Mount orphan allegation is unsupported by actual Textual 8.2.8 registration and both real-boundary probes; no speculative mount change. 316 current-source targeted passes and native006 lifecycle pass. Six gallery captures are pixel-identical; two differ only by blinking input caret. Independent review: no blockers. Evidence: Docs/superpowers/qa/2026-09-20-mcp-inspector-refresh/qodo/README.md. Current-head CI/review and merge receipt remain.

Closed through PR2757, merged into dev at 62d43190ce3ca21bda0f0ff03e5148971eedc7ca on 2026-09-20. Actual merge tree equals verified d93498014f. Current-head Fast Lane 1,152 tests and all applicable CI checks pass; 316 local targeted cases and final native evidence qualify review fixes. All Qodo threads resolved: documentation/coverage fixed, orphan-mount allegation declined with real Textual boundary tests and independent review. Owner visual approval preserved; no rebase conflicts. Merge receipt: Docs/superpowers/qa/2026-09-20-mcp-inspector-refresh/pr2757-closeout.json. Existing ADR-161/170 and ADR-032 apply.
<!-- SECTION:NOTES:END -->
