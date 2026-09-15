---
id: TASK-32599
title: Restore bounded Library queries during focus changes and same-band resizing
status: Done
assignee:
  - '@codex'
created_date: '2026-09-15 02:48'
updated_date: '2026-09-15 04:19'
labels:
  - library
  - ui
  - audit
dependencies: []
references:
  - Docs/superpowers/reports/2026-09-14-library-workflow-audit.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The task-23025 query-budget gates reproduce failures at baseline 2939afda63: 23 Library queries across three non-crossing resize frames (expected zero), and five per Tab (ceiling one). Nearest production call-site instrumentation attributes most work to _active_library_rail and _library_focusable, plus four ordinary-rail width queries and one on_resize query. This is measured excess query work; visible latency has not been established.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Non-crossing resize frames meet the established zero-Library-query gate while actual layout-band crossings still apply the correct reader and rail geometry.
- [x] #2 Tab focus changes meet the established ceiling of one Library query and do not trigger whole-screen recomposition.
- [x] #3 Keyboard traversal, visible focus, reader return and restored rail widths remain correct with production styles at 80 and 120 columns.
- [x] #4 The targeted task-23025 gates pass without relaxing budgets merely to match the regression; any necessary contract change is supported by explicit behavior and measurement evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce the existing two query gates at af5a26a2af and attribute calls and complete-helper timings; identify true width-contract changes separately from no-op frames.
2. Reuse the existing validated positive-reference cache for active rail, focusability and resize chrome, and the existing compose-scoped adaptive-shell presence check. Preserve file-Notes authority, live focus-chain checks, exact width policy and cache invalidation.
3. Pin hidden/disabled/replaced control behavior and run both original query gates with production styles without increasing budgets. Assert real 36-to-35-cell rail changes still apply, then run the task-23025 file plus focused width/restoration and visible-focus cases.
4. Record before/after evidence and an isolated native traversal/resize check, review the diff, complete the task and commit locally.

ADR required: no
ADR path: backlog/decisions/086-library-adaptive-reader-shell.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: routine optimization using existing cache ownership and width/focus contracts. No storage, service, permission, UI structure or token changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Reused the existing validated Library reference cache for active rail, focusability and resize chrome; reused the existing adaptive-shell presence check for ordinary width application. The selector scan in focusability is gone, while the live focus chain still rejects hidden/disabled controls and the positive cache still recovers after removal/replacement. File Notes keeps its explicit active-rail authority.

Measured Library-attributed queries: 23 to 0 across the resize sequence, and 5 to 0 per measured Tab. The warm complete focusability helper median fell from 348.89 to 121.09 microseconds (seven batches of 500 calls). The already-cheap active-rail helper was slightly slower, 0.18 to 0.26 microseconds; this is not an end-to-end latency claim.

The same-band sequence legitimately changes the default width from 36 to 35 cells. The existing budgets are unchanged, and tests now assert the width and no whole-screen recomposition. Both gates also run with production CSS. Added hidden/disabled/replaced-control and Media-to-Search/RAG return coverage, preserving 24/48-cell saved widths at 120/80/120 columns.

Validation: full task-23025 file 22 passed; adjacent selection 10 passed, 5 failed. The older width family applies ordinary rules to the adaptive Collections route; a pinned baseline-method control reproduces its representative 24-cell failure. The other failure type is the already-hidden File Notes return control, also reproduced by the baseline control. These failures remain outside this repair. Native private-profile Tab, 80/120 resize, note opening, retained editor content/focus and Escape-to-selected-row return worked; exit 0. The existing missing log-widget startup error and optional warnings are recorded.

Ruff and formatting pass for the changed test and three archived probes. LibraryScreen retains exactly 205 pre-existing Ruff diagnostics with no additions; all four changed method fragments format cleanly. git diff --check passes. No full test sweep. Self-review confirmed width policy, permission rules and token values are unchanged.

Evidence, exact selections, baseline controls and timings: Docs/superpowers/qa/2026-09-14-library-query-budget/README.md. Lessons-testing-evidence records why same-band resize counts must be paired with actual cell geometry and complete-helper timings.

ADR required: no. Existing backlog/decisions/086-library-adaptive-reader-shell.md, 150-design-token-system-and-design-language.md and 161-component-pattern-library.md apply; this changes no ownership or service boundaries.
<!-- SECTION:NOTES:END -->
