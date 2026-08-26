---
id: TASK-17659
title: 'Console: one blank line above the composer'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - console
  - ux
dependencies:
  - task-17657
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner request 2026-08-17, following the breathing-room row below the composer (TASK-17657): one blank line above it too, so the bar floats clear of the status row. The margin sits on the composer's top edge, so whatever is above — the status chips at rest, or the staged-evidence/prompt-queue strips while active — gets the same one row of air; the strips remain the nearest neighbors above the gap (the shelf-adjacency contract updates from touching the composer to sitting immediately above its margin). Compact mode (< 35 rows) drops both gaps via the existing margin override.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 One blank row renders above the composer (below the status row in above-placement mode) and one below it, pinned by painted assertions at 150x44
- [x] #2 The prompt-queue shelf sits immediately above the composer's margin (one row of air between shelf and composer), contract test updated
- [x] #3 Compact mode drops both gaps; bottom-stack suites green; bundle rebuilt from source
- [x] #4 User Guide Console page stamp refreshed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: extend the single-separator contract — composer one row below the chips, painted blank rows on BOTH sides; watched fail against the pre-change bundle.
2. Composer margin `0 0 1 0` -> `1 0`; the existing compact override (`margin: 0`) drops both gaps.
3. Update the prompt-queue shelf adjacency pin; bundle rebuild; suites; live probes; docs stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One declaration: composer margin `0 0 1 0` -> `1 0` (bundle rebuilt); the TASK-17657 compact override already zeroes all edges. The single-separator contract now pins painted blank rows on both sides of the bar. The shelf-adjacency pin took two corrections worth recording: a size-parameterized expectation first (wrong — the shelf suite runs on the bundle-less ConsoleHarness, which cannot see stylesheet margins at any size), settled as adjacency to the composer's computed MARGIN box (`composer.styles.margin.top`), which is exact in every harness and leaves the painted gap to the bundled contract test. Live probes: above mode chips y39 / BLANK / composer y41 / BLANK / footer y43; below mode grid y38 / BLANK / composer y40 / BLANK / chips y42. Transcript region 32 -> 31 (the owner spent the row deliberately).

Files: `css/components/_agentic_terminal.tcss` (+ bundle), `Tests/UI/test_console_composer_collapse.py`, `Tests/UI/test_console_prompt_queue.py`, `Docs/User_Guide/console.md`.
<!-- SECTION:NOTES:END -->
