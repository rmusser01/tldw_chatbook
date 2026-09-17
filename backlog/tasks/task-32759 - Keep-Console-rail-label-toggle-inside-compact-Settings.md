---
id: TASK-32759
title: Keep Console rail label toggle inside compact Settings
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 22:27'
updated_date: '2026-09-17 22:35'
labels:
  - ui
  - settings
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 80x24 the rail-label checkbox extends outside the Settings detail pane even after focus scrolls it into view. Users need a fully visible toggle and readable label at compact terminal sizes without changing its staged-save behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The rail-label toggle and its full label and glyph remain inside the visible detail pane after keyboard focus at 80x24 and 190x55 in dark and light themes.
- [x] #2 Space stages the rail-label change while preserving the active runtime value until Save; existing success failure and revert behavior stays covered.
- [x] #3 A private native terminal journey confirms compact visibility and clean shutdown; targeted tests and token governance pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md and backlog/decisions/161-component-pattern-library.md
Reason: Directly compose the existing compact toggle tokens; no ownership or behavior boundary changes.
1. Retain the failing production-style dark/light and compact/wide checkbox regression from TASK-23150.
2. Size the rail-label checkbox with existing compact-control and spacing tokens, rebuild generated CSS, and preserve staged-save behavior.
3. Run focused rail-label and token/bundle checks, inspect private native compact/wide terminal captures, verify clean shutdown and unchanged default profile, and obtain independent review.
4. Record exact evidence and remaining Console Behavior/Storage review scope; save this bounded repair to PR #2704.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The compact rail-label checkbox now uses existing one-row toggle tokens. The compact Console inner card drops duplicate horizontal padding so all 32 columns paint within the clip; wide padding and staged-save semantics remain unchanged. Rebuilt generated Settings CSS. 45 distinct targeted cases pass (13 rail cases, 31 governance, one boot-budget), plus four inspected private native cells at 190x55/80x24 in dark/light. Verified native exit, released lock, default fingerprints, logs and 11 healthy private DBs. Ruff and independent read-only review pass. QA receipt: Docs/superpowers/qa/2026-09-17-settings-console-rail/README.md. Existing ADR-150/161 govern; no new ADR. Other Console Behavior and Storage review remains open.
<!-- SECTION:NOTES:END -->
