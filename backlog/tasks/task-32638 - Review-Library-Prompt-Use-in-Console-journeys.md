---
id: TASK-32638
title: Review Library Prompt Use in Console journeys
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 17:11'
updated_date: '2026-09-15 17:38'
labels:
  - library
  - prompts
  - console
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the component review through saved Prompt and Recipe handoffs, variable entry, System authorization and Console draft insertion while preserving existing guarded destination and privacy contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The header action and variable dialog support readable keyboard navigation and cancellation at wide and compact sizes in both themes.
- [x] #2 Direct User text, escaped braces, substituted values and original placeholders follow the shared grammar and append without replacing the Console draft or sending a request.
- [x] #3 System replacement requires explicit authorization; missing or stale targets and unavailable staging report usable recovery without modifying either lane.
- [x] #4 Recipe conversion remains a detached unsaved Prompt copy and source Prompt or Recipe records remain unchanged by insertion.
- [x] #5 Targeted tests, isolated native handoff and persistence evidence, documentation and review notes qualify the completed journeys.
- [x] #6 The System authorization checkbox paints only its state glyph beside the full label, and the Console System status chip reflects a successful replacement.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/053-prompt-variable-grammar-and-guarded-insertion.md; backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md; backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md (existing)
Reason: Verify and repair existing insertion grammar, keyboard interaction and recovery under the approved handoff and artifact contracts; no new ownership, persistence or service boundary.

1. Exercise saved User-only Prompt handoff and the shared variables/System dialog with production CSS and real SQLite at 170x48 and 80x24 in both themes. Reproduce defects with actual keyboard input before fixing.
2. Verify escaped literals, value substitution, original placeholders, explicit System authorization, Cancel and missing/stale targets. Keep Recipe conversion detached and source records unchanged. If the same direct-insert grammar defect exists in Console, repair that shared semantic path too.
3. Run affected dialog/Library/Console/handoff/parser and governance checks only. Verify the actual native Library-to-Console navigation and append/System state without sending provider requests, using a fresh private profile and normal-exit evidence.
4. Update guide, workflow audit, QA evidence and task notes; self-review and commit locally. No full suite or integration into dev.

Allocation: fresh fetch, reachable paths and 43 live worktrees max 32637; no content reference to 32638 across 314 refs. CLI offered occupied 32633, corrected before implementation.

Native paint follow-up: reproduce the clipped duplicate checkbox label and stale System status chip in targeted tests. Render the existing glyph-only checkbox while retaining its semantic label, and synchronize the existing control bar with the other System surfaces. Verify both in one native confirmation batch.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Direct Prompt insertion now renders compiled escaped literals once in Library and Console. The shared System checkbox paints only its glyph beside the full copy, and successful System replacement refreshes the existing Console status bar. Original-source, draft, one-shot, expiry, Recipe and System authorization contracts are preserved.

Verification: 369 distinct targeted tests passed, including four reproduced escape failures and four reproduced paint failures before fixes. Final native LinuxDriver run passed at 170x48 dark and 80x24 light with normal Ctrl+Q, app.run return, exit 0 and observed zsh. Read-only SQLite confirms four unchanged v1 source Prompts and zero messages. Rendered captures and exact test selections are in Docs/superpowers/qa/2026-09-15-prompt-console/README.md. No new Ruff diagnostics; changed test ranges and new files are formatted. Self-review and independent code review completed with no outstanding findings.

Updated insertion guide, workflow audit, QA evidence and testing lesson. Existing delegation checks now validate the current owner and async forwarding without counting signature lines. Native setup uses seeded saved fixtures; authoring, fault injection, Recipe conversion and durable System restart are not native claims. Compact inputs scroll above fixed actions; the System chip is horizontally offscreen initially. No full suite or provider request.

ADR required: no. Existing backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md, 053-prompt-variable-grammar-and-guarded-insertion.md, 086-library-adaptive-reader-shell.md, 094-console-turn-lifetime-and-navigation-boundary.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md apply. No new architectural boundary. Next component: Skills. Integration into dev remains pending.
<!-- SECTION:NOTES:END -->
