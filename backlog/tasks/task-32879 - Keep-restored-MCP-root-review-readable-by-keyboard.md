---
id: TASK-32879
title: Keep restored MCP root review readable by keyboard
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-20 18:22'
updated_date: '2026-09-20 18:38'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let compact-terminal users read every restored-root path and reach safe confirmation choices before applying fresh MCP defaults.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At 80x24 and 170x48 in dark and light themes both recovery choices are fully painted at entry and after scrolling or resizing.
- [x] #2 Keyboard users can scroll the complete literal long-path review while actions stay visible and no review action runs implicitly.
- [x] #3 Cancel and Escape retain historical authority; explicit confirmation still uses the existing owner checks and creates fresh Ask/local state only.
- [x] #4 Targeted regression checks and private native screenshots qualify the bounded dialog; unrelated confirmation dialogs retain their layout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve failing compact focus and keyboard-scroll evidence using real restored owners. 2. Compose a recovery-specific ConfirmationDialog body with a focusable VerticalScroll and fixed standard actions; left-align literal paths with existing tokens and remove post-mount inline layout writes. 3. Verify compact/wide dark/light rendering, full text through keyboard scrolling, cancellation and explicit confirmation against real private owners; run adjacent recovery and design governance checks. 4. Inspect native captures and lifecycle, request independent review, update the review ledgers and save a bounded PR against dev. ADR required: no. ADR path: N/A; existing backlog/decisions/126-complete-local-backup-and-recovery.md and backlog/decisions/161-component-pattern-library.md apply. Reason: presentation repair preserving existing recovery authority, modal dismissal and persistence contracts.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a recovery-specific ConfirmationDialog with a keyboard-scrollable, left-aligned body and persistent standard actions. Existing ADR-126/150/161, owner checks and native writes are preserved; no new ADR. Four meaningful red tests now pass; 116 distinct targeted passes, two reproduced baseline failures and seven passing preflight guards. Fixed the catalog test wait to follow token retirement plus busy settlement without weakening assertions. Sixteen inspected native captures qualify dark/light 80x24 and 170x48 using actual restored owners, normal shutdown, healthy private DBs and unchanged defaults. Independent review found no blockers. Evidence: Docs/superpowers/qa/2026-09-20-mcp-compact-review/README.md. Current-head CI, accumulated PR review and owner visual approval remain before merge; task stays In Progress until closeout.
<!-- SECTION:NOTES:END -->
