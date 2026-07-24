---
id: TASK-538
title: Align unified first-time replay with top-level Logs navigation
status: Done
assignee: []
created_date: '2026-07-24 21:23'
updated_date: '2026-07-24 21:25'
labels:
  - ui
  - navigation
  - tests
  - logs
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent the unified-shell first-time replay from timing out on healthy Home chrome by deriving its expected navigation roster from the canonical shell destination model after Logs became top-level.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 First-time Home readiness recognizes the complete canonical navigation roster including Logs
- [x] #2 The replay still verifies exact navigation order and labels without a duplicated hard-coded destination list
- [x] #3 Focused first-time replay and shell-navigation tests plus Ruff and diff checks pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the first-time Home-chrome timeout and verify that the live 13-button navigation conflicts with the replay's stale 12-button constant.
2. Generate the replay's exact expected button ids and labels from `SHELL_DESTINATION_ORDER` and the production label-numbering helper.
3. Run the unified and product-maturity first-time replays plus focused master-shell navigation tests.
4. Run static checks, inspect the bounded diff, and request independent review before closeout.

ADR required: no
ADR path: backlog/decisions/015-shell-destination-ia.md
Reason: This is a stale replay-fixture repair that directly applies ADR-015's now-current 13-destination contract; it does not change navigation architecture or runtime behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the replay's stale 12-button literal with exact expected button ids and labels generated from `SHELL_DESTINATION_ORDER` and `nav_button_label`, keeping the integration assertion while removing the duplicate route roster.
- The deterministic 10-second Home-chrome timeout now passes in 4.5 seconds, and the replay continues through Console, Library, and Personas orientation checks.
- Verification: the unified replay, product-maturity first-time replay, master-shell navigation, and shell-destination suites passed 32 tests. Ruff, formatting, `compileall`, and `git diff --check` passed.
- Independent review also verified the replay, Nielsen consumer, master-shell order, and label helper (4 tests) and approved the single-source-of-truth coupling, imports, ADR-015 linkage, and bounded diff with no actionable findings.
<!-- SECTION:NOTES:END -->
