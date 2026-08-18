---
id: TASK-17650
title: 'Console: delete the zero-information rows in the bottom stack (CSS-only)'
status: In Progress
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - console
  - ux
  - css
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console bottom stack renders ~10 chrome rows below the transcript, and a 2026-08-17 headless audit (150x44, dev `22d156155`, app bundle loaded) found two of them carry zero information and are pure CSS leaks:

1. A blank row between the status chips and the footer: `#console-status-chips` is composed with `classes="ds-panel"` and inherits `.ds-panel { margin: 0 0 1 0 }`; its id rule overrides height/padding/border but never `margin`. Sibling `ds-panel`s in the same stack (`#console-native-composer`, `#console-staged-evidence-strip`) all explicitly set `margin: 0` — the chips are the only one that forgot. Live mutation experiment confirmed `margin: 0` recovers exactly +1 transcript row.
2. `#console-native-transcript` draws its own `border: solid $ds-grid-line` — the innermost of THREE stacked bottom borders (transcript border, region inline frame, grid inline frame render on three consecutive rows). Compact mode (`-console-compact`) already sets this exact border to `none` with the reasoning "the inner frame costs rows; the region frame still surrounds the transcript" — the same argument applies at all sizes.

Both phantom rows survive compact mode today, so this fix helps most where rows are scarcest. This is Phase A of the bottom-stack de-clutter agreed with the owner on 2026-08-17 (NNG heuristic #8: aesthetic and minimalist design); the structural phases are tracked separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 No blank row renders between the status chips and the footer on the running Console screen (verified with the app CSS bundle loaded)
- [ ] #2 The transcript no longer draws its own inner border at any terminal size; the region frame still surrounds it (parity with the existing compact-mode treatment)
- [ ] #3 The transcript gains at least 3 rows at 150x44 versus the dev baseline, measured on the running screen (1 margin row + 2 transcript-border rows)
- [ ] #4 A regression test pins the recovered geometry and loads the real CSS bundle (the `StatusRowApp` pattern from Tests/UI/test_console_status_row_collapse.py:100 — NOT `ConsoleHarness`, which runs without the bundle and cannot see either defect)
- [ ] #5 Existing bottom-stack contract tests stay green; CSS edited only in source modules with the bundle rebuilt via build_css.py (bundle-sync guard green)
- [ ] #6 The Docs/User_Guide Console page is updated or its "Verified against" stamp refreshed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: headless row-map probe at 150x44 with the bundle loaded (before state).
2. RED: bundle-loading regression test asserting (a) chips margin is zero / no blank row above the footer, (b) transcript renders without its own border.
3. Edit css/components/_agentic_terminal.tcss: add `margin: 0;` to the `#console-status-chips` block; set `#console-native-transcript` border to none (decide whether the now-redundant compact override is removed or kept harmless).
4. Rebuild the bundle (python tldw_chatbook/css/build_css.py); bundle-sync guard.
5. GREEN: new test + targeted contract suites (status-row collapse, workbench contract, composer collapse, shell regions, command popup, non-obscuring focus).
6. Live headless before/after row map as evidence; update User Guide stamp.
<!-- SECTION:PLAN:END -->
