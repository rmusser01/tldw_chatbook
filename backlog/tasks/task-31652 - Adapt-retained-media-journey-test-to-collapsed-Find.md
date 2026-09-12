---
id: TASK-31652
title: Adapt retained media journey test to collapsed Find
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 16:58'
updated_date: '2026-09-05 17:16'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the retained adaptive-reader scenario useful against current UI behavior without rewriting its immutable historical evidence bundle.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The media journey explicitly opens current Find controls and verifies settled selected/loaded row facts.
- [x] #2 Retained TASK-23019 evidence files and hashes remain unchanged.
- [x] #3 Affected live harness and Find behavior checks pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce missing Find input and compare TASK31237 collapsed Find behavior with the retained TASK23019 scenario.
2. Adapt only the current test entry/focus hooks to press the real Find button and await mounted input; preserve scenario facts and archived bytes.
3. Run the focused media capability and Find behavior tests plus the closeout harness tests, scoped Ruff/format, and diff checks.
ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md (existing)
Reason: test-only compatibility with approved TASK31237 behavior, no reader contract or evidence rewrite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Adapted only the current test hooks: press the real Find action at entry/focus and await the real Delete readiness gate after selection. Preserved all retained scenario assertions and the complete hashed TASK23019 bundle byte-for-byte. Initial focused tests exposed missing Find; the fullfile run then exposed selected-count painting before Delete re-enabled, now explicitly awaited. Final full490 closeout tests passed142.80s using the original hermetic-runtime-compatible virtualenv; the dedicated Find-collapse case also passed. Scoped Ruff, changed-range formatting, diff checks, and independent review passed. No product or historical-evidence edits; ADR086 unchanged.
<!-- SECTION:NOTES:END -->
