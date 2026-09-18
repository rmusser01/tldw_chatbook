---
id: TASK-32813
title: Restore CSS consolidation without increasing boot budgets
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-18 17:38'
updated_date: '2026-09-18 19:01'
labels:
  - ui
  - design-system
  - performance
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep feature and modal styles within the existing stylesheet source and boot-cost limits while preserving rendered behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All currently ungoverned class-level CSS declarations are consolidated or proven redundant without adding allowlist exceptions.
- [x] #2 Generated styles preserve the current cascade and rendered appearance for the affected controls, including focus, hover and disabled states.
- [x] #3 Existing boot-byte, stylesheet-source and selector-cost ratchets pass without increases; the destination-tour check runs in an isolated profile and proves route coverage.
- [ ] #4 Targeted consumer, governance, independent review and native visual evidence are recorded with explicit remaining limits and saved in the draft PR.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory the 26 out-of-allowlist declarations and reproduce source/byte/selector costs and the destination-tour setup failure. 2. Reuse existing build-time consolidation, preserving origin and specificity; shed equivalent generated trivia or redundant rules where required by boot budgets. Resolve static CSS constants without adding runtime evaluation to the builder. Repair the tour harness with the existing private-profile boundary and explicit route evidence. 3. Compare parsed/computed styles for affected consumers and run targeted consumer, token, source-count, boot-byte and fast-path checks. 4. Inspect native representative dark/light compact/wide views, obtain independent review and update evidence, ledgers and draft PR. ADR required: no. ADR path: backlog/decisions/150-design-token-system-and-design-language.md, backlog/decisions/161-component-pattern-library.md and backlog/decisions/097-boot-budget-ratchets.md. Reason: Mechanical consolidation and verification of existing architecture; stop and document an ADR before any new runtime loading boundary if evidence requires one.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Consolidated 25 class declarations through the existing widget-default build stream and removed one proven dead sibling-scoped alias. Quote-aware generated comment removal preserves tokens and provenance; indexed subjects keep selector cost at 274. The standalone RecoveryApp registers its screen CSS at the native default tier. Private consumer hosts now load their actual feature sheets and own profile setup before imports.

Evidence: Docs/superpowers/qa/2026-09-18-css-consolidation/README.md and GALLERY.md record 194 passing exact consumer cases, final serial guards, all 29 affected owners observed with zero computed-rule mismatches, and 28 inspected native captures. Bytes are 583097/608090; proven 15-route/modal source count is 46 with unchanged source limits. Native exit, lock, ten databases and default-state checks pass. Original failures remain visible; TASK-32815 tracks the intermittent empty-stack shutdown query and TASK-32816 tracks preserved compact Appearance clipping. Roleplay recovery's existing unstyled view remains component-review debt.

Independent review and baseline-relative Ruff/changed-range formatting receipts are retained. No full suite or connected-provider qualification was run. ADR required: no; existing ADR-150, ADR-161 and ADR-097 apply. Lesson recorded in backlog/docs/lessons-testing-evidence.md. Product/test/generated sheets, QA artifacts and component/MCP ledgers are updated. Draft PR2707 remains unmerged and requires its own visual review. Save verification follows push.
<!-- SECTION:NOTES:END -->
