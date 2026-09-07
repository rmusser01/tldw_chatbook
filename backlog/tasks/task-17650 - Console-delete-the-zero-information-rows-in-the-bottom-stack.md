---
id: TASK-17650
title: 'Console: delete the zero-information rows in the bottom stack (CSS-only)'
status: Done
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

A blank row between the status chips and the footer: `#console-status-chips` is composed with `classes="ds-panel"` and inherits `.ds-panel { margin: 0 0 1 0 }`; its id rule overrides height/padding/border but never `margin`. Sibling `ds-panel`s in the same stack (`#console-native-composer`, `#console-staged-evidence-strip`) all explicitly set `margin: 0` — the chips are the only one that forgot. Live mutation experiment confirmed `margin: 0` recovers exactly +1 transcript row. The phantom row survives compact mode too, so this fix helps most where rows are scarcest.

Scope note (2026-08-17, during implementation): the audit's second candidate — removing `#console-native-transcript`'s own border — was reallocated to TASK-17651. Those border rows are NOT zero-information: the `:focus` rule recolors them to `$ds-focus-accent`, making them the transcript's keyboard-focus affordance, pinned by `test_console_transcript_focus_uses_stable_border_geometry` (Tests/UI/test_non_obscuring_focus_contract.py). Removing them requires a redesigned, dimensionally-stable focus treatment, which belongs with the frame-grammar work in TASK-17651, not a CSS cleanup.

This is Phase A of the bottom-stack de-clutter agreed with the owner on 2026-08-17 (NNG heuristic #8: aesthetic and minimalist design); the structural phases are tracked separately.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 No blank row renders between the status chips and the footer on the running Console screen (verified with the app CSS bundle loaded)
- [x] #2 The transcript gains at least 1 row at 150x44 versus the dev baseline, measured on the running screen
- [x] #3 A regression test pins the zero-margin contract and loads the real CSS bundle (the `StatusRowApp` pattern from Tests/UI/test_console_status_row_collapse.py:100 — NOT `ConsoleHarness`, which runs without the bundle and cannot see the defect)
- [x] #4 Existing bottom-stack contract tests stay green; CSS edited only in source modules with the bundle rebuilt via build_css.py (bundle-sync guard green)
- [x] #5 The Docs/User_Guide Console page is updated or its "Verified against" stamp refreshed
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Baseline: headless row-map probe at 150x44 with the bundle loaded (before state).
2. RED: bundle-loading regression test asserting the chips' computed margin is zero when mounted with the real `classes="ds-panel"`, plus a source+bundle CSS pin that the `#console-status-chips` block declares `margin: 0;`.
3. Edit css/components/_agentic_terminal.tcss: add `margin: 0;` to the `#console-status-chips` block (sibling precedent: `#console-native-composer`, `#console-staged-evidence-strip`).
4. Rebuild the bundle (python tldw_chatbook/css/build_css.py); bundle-sync guard.
5. GREEN: new test + targeted contract suites (status-row collapse, workbench contract, composer collapse, shell regions, command popup, non-obscuring focus).
6. Live headless before/after row map as evidence; update User Guide stamp.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
One-line CSS fix: `margin: 0;` added to the `#console-status-chips` block in `css/components/_agentic_terminal.tcss` (bundle rebuilt via build_css.py), cancelling the `.ds-panel` bottom margin that painted a permanent blank row between the chips and the footer — the same explicit cancel its bottom-stack siblings (`#console-native-composer`, `#console-staged-evidence-strip`) already carry.

Scope deviation, documented in the description: the audit's second candidate (removing `#console-native-transcript`'s border) was reallocated to TASK-17651 mid-implementation — those rows carry the transcript's keyboard-focus affordance (`:focus` recolors them; pinned by `test_console_transcript_focus_uses_stable_border_geometry`), so their removal needs a replacement focus treatment, not a CSS cleanup.

Tests (TDD, both watched RED first): new `PanelStatusRowApp` harness mounting the chips exactly as the screen does (`classes="ds-panel"`, neighbor below, real bundle via `CSS_PATH`) with a runtime margin + adjacency assertion (RED: `Spacing(bottom=1)`); and `margin: 0;` added to the source+bundle stylesheet contract in `test_status_row_stylesheet_contract_is_in_source_and_bundle` (RED: declaration missing). Note for future geometry tests: `ConsoleHarness` runs WITHOUT the app bundle and cannot see `.ds-panel` styles — bundle-loading harnesses are mandatory for this class of assertion.

Evidence: 808 passed across the 12 bottom-stack contract files (incl. bundle-sync guard). Headless row map at 150x44 (isolated config, worktree provenance asserted): pre-fix blank at y=42 between chips (y41) and footer (y43); post-fix no blank row, chips y42, footer y43, `#console-transcript-region` height 28 -> 29.

Files: `tldw_chatbook/css/components/_agentic_terminal.tcss`, `tldw_chatbook/css/tldw_cli_modular.tcss` (rebuilt), `Tests/UI/test_console_status_row_collapse.py`, `Docs/User_Guide/console.md` (stamp), this task file + TASK-17651 (scope reallocation).
<!-- SECTION:NOTES:END -->
