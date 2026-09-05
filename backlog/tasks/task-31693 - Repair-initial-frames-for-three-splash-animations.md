---
id: TASK-31693
title: Repair initial frames for three splash animations
status: Done
assignee:
  - '@codex'
created_date: '2026-09-05 18:36'
updated_date: '2026-09-05 18:46'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore animated startup for phonebooths, hypno swirl, and world map after their initial frame raises real markup or dimension errors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All three affected cards construct and render valid initial frames without static fallback.
- [x] #2 Existing animation behavior remains intact and the complete startup polish regression file passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve the three mounted startup RED cases and add direct frame regressions for literal phonebooth brackets, valid swirl colors, and initialized map dimensions. 2. Escape only literal booth characters, convert HSV to Rich-supported RGB with stdlib colorsys, and initialize WorldMap dimensions locally. Keep animation algorithms and fallback policy unchanged. 3. Run the full startup polish file and scoped checks, obtain parent review, document evidence, and commit only this task. ADR required: no. ADR path: N/A. Reason: routine repairs to existing effects without interface or UX changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Repaired three real initial-frame faults: Phonebooths explicitly escapes its literal bracket, Hypno Swirl converts its existing hue to supported RGB hex with stdlib colorsys, and WorldMap initializes its requested dimensions. Three direct frame regressions reproduced all failures through Textual Static.update before the fix and preserve dimensions, art, styles, and map cursor states. Full startup file: 97 passed; final four-file combined gate: 269 passed in 68.46s (/private/tmp/tldw-31693-31695-final.xml). Scoped Ruff and diff checks passed. Rich Text.from_markup alone accepted the malformed markup; testing now uses the actual display parser. Shared testing-evidence lesson owner was sent this incident. No new ADR required: existing effect algorithms and fallback policy remain unchanged.

Parent completed bounded final diff review with no actionable findings.
<!-- SECTION:NOTES:END -->
