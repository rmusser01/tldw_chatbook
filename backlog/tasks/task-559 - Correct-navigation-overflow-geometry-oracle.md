---
id: TASK-559
title: Correct navigation overflow geometry oracle
status: Done
assignee: []
created_date: '2026-07-25 18:07'
updated_date: '2026-07-25 18:09'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep the destination visual-parity gate aligned with Textual scroll clipping by comparing the scroll viewport to the docked overflow hint instead of treating an overflowing child widget's virtual region as painted geometry.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 At the 140-column default viewport the destination strip and overflow hint occupy non-overlapping regions
- [x] #2 The test proves the destination content genuinely overflows the strip
- [x] #3 The hint remains flush with the navigation bar's right edge
- [x] #4 Focused navigation and visual-parity tests pass
- [x] #5 Task notes record RED evidence and ADR applicability
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the RED geometry and capture the navigation, scroll viewport, virtual content, Settings child, and hint regions.
2. Replace the stale child-vs-sibling assertion with paint-boundary assertions for the clipped scroll viewport and docked hint.
3. Retain explicit proof that the virtual destination content overflows the strip and the hint remains docked.
4. Run the focused regression, navigation contract suite, complete visual-parity module, Ruff, formatter, and diff checks.
5. Self-review and record the corrected timing/geometry contract.

ADR required: no
ADR path: N/A
Reason: This is a test-oracle correction for Textual's existing scroll-clipping boundary and changes no application architecture or runtime behavior.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Corrected the default-size navigation geometry oracle to compare the destination strip's clipped viewport with the docked overflow hint. The test now also proves virtual destination content exceeds the viewport and that the hint stays flush with the navigation bar's right edge. RED evidence: Settings occupied virtual columns 122-131 while the strip clipped at column 126 and the hint began at 126; comparing the un-clipped child region incorrectly reported a painted overlap after Logs added six virtual columns. Verification: the focused regression plus master-shell and overflow-hint suites pass 21/21; the corrected case also passes in the complete visual-parity run. That broader run exposed separate stale Watchlists and Schedules contracts (71 passed, 12 failed), so no whole-module claim is made here. Ruff, formatter, and diff checks pass. Plan deviation: the complete-module gate revealed unrelated destination migration fallout rather than passing; that fallout remains in the active sweep. ADR required: no; test-oracle correction only. Modified: Tests/UI/test_destination_visual_parity_correction.py and this task file.
<!-- SECTION:NOTES:END -->
