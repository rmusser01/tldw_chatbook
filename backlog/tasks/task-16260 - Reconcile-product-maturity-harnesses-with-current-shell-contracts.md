---
id: TASK-16260
title: Reconcile product-maturity harnesses with current shell contracts
status: Done
assignee:
  - '@codex'
created_date: '2026-08-14 17:04'
updated_date: '2026-08-14 17:08'
labels:
  - testing
  - ui
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore deterministic product-maturity coverage after Library handoff, Console setup, navigation overflow, and focus behavior evolved while preserving current user-visible contracts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library handoff coverage targets the mounted current canvas.
- [x] #2 Console clean-run coverage asserts the configured provider's actual recovery path.
- [x] #3 Overflow navigation and keyboard focus coverage exercise the current reachable controls.
- [x] #4 Affected focused modules pass with Ruff and diff checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the reproduced failures and classify each against current product contracts.
2. Update only stale harness selectors, copy expectations, overflow navigation, and focus entry assumptions.
3. Run the named failures and full affected test modules.
4. Run Ruff and diff checks and record closeout evidence.

ADR required: no

ADR path: N/A

Reason: test-contract reconciliation only; no production architecture or behavior change.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Replaced the retired Library handoff-detail selector with the mounted handoff canvas and aligned clean-run Console recovery with the fixture's selected OpenAI model and missing-key state.
- Updated navigation coverage for the current overflow modal and clip-ghost focus contract: off-screen destinations remain keyboard reachable through the complete modal list, while invisible strip buttons stay outside Tab order.
- Preserved strict RED evidence for all four stale harness assumptions. The four named regressions passed, followed by all five affected product-maturity modules: 26 passed with two existing dependency warnings.
- Ruff lint and format checks pass for the four changed tests, and `git diff --check` is clean. No production code or ADR changed.
<!-- SECTION:NOTES:END -->
