---
id: TASK-31795
title: Research Runs view renders collapsed/blank — missing layout CSS
status: Done
assignee: []
created_date: '2026-09-06 00:12'
labels:
  - bug
  - ui
  - research
dependencies: []
priority: high
---

## Description (the why)

Live UAT on dev tip (fresh profile) reproduced twice: Research tab →
"Runs" chip in the Research mode strip → the whole content area collapses
to ~1 squashed line. Nothing renders, nothing is reachable, no traceback
(silent layout failure). Only ~8 non-blank rows appear in a 52-row
terminal. This is a release blocker for the dev→main cut: the durable
Research Runs surface is unusable.

## Acceptance Criteria (the what)

- [x] Opening Research → Runs renders the full Runs view (toolbar,
      create-run row, status line, run list and detail panel) with
      nonzero, on-screen geometry at common terminal sizes.
- [x] Every Runs control (source select, refresh, create inputs/buttons,
      run actions, checkpoint/follow-up controls) is painted inside the
      screen content area, not pushed off-screen.
- [x] A regression test with production CSS asserts the Runs view's key
      containers lay out vertically with usable heights/widths.
- [x] CSS bundle stays in sync with its sources (preflight passes) and
      within the boot CSS byte ratchet.

## Implementation Plan (the how)

1. Reproduce with a production-CSS Textual harness and dump the Runs
   view's widget regions to find the collapse mechanism.
2. Root-cause: check git history for lost research CSS vs never-authored
   styles; inspect the global `.window` rules.
3. Fix in `css/features/_research_workspace.tcss` (already a CSS_MODULES
   entry): force `layout: vertical` on the Runs window (the global
   `.window` rule imposes horizontal) and author the missing internal
   layout rules (rows auto-height, body 1fr, list/detail split,
   input/select widths, scrollable detail panel). Rebuild the bundle
   with `build_css.py`.
4. Add a production-CSS regression test asserting the vertical stacking
   and on-screen geometry of the Runs containers and controls.
5. Verify live in tmux with an isolated scratch config (before/after
   captures), run targeted UI tests, `./scripts/preflight.sh`, PR to dev.

## Implementation Notes

Root cause: never-authored CSS, not a lost file. `ResearchWindow`
(UI/Research_Window.py, born unstyled in 7a6129009b) subclasses
`Vertical`, but ResearchScreen mounts it with `classes="window"` and the
global `.window` rule (css/layout/_windows.tcss) sets
`layout: horizontal`. App-tier class rules beat widget DEFAULT_CSS, so
the window's five vertical sections were laid out side-by-side: toolbar
and create-row collapsed to width 1 and `#research-body` was parked at
x=239 on a 220-column terminal (production-CSS harness region dump).
Git history shows no research layout tcss was ever deleted; the only
Runs rule ever authored was the mode-strip-era height override.

Fix (app tier, mirroring the `#logs-window { layout: vertical; }`
precedent): extended `css/features/_research_workspace.tcss` (already a
CSS_MODULES entry) with `layout: vertical` on
`ResearchScreen #research-window.window` plus the missing internal
layout — toolbar/create/action rows auto-height, `#research-body` 1fr,
run-list/detail split (30%/1fr), input/select widths, scrollable detail
panel, fixed-height checkpoint TextArea. Bundle regenerated with
`build_css.py` (+80 lines; boot CSS byte ratchet passes with ~23KB
headroom). A widget-tier DEFAULT_CSS baseline was considered and
rejected: the offending `.window` rule is app-tier, which beats any
DEFAULT_CSS regardless of specificity, so only an app-tier rule can fix
it; keeping one authority avoids a second drift surface.

Verification: live tmux on a scratch TLDW_CONFIG_PATH profile — before
(pristine dev bundle): 8 non-blank rows, exactly the UAT capture; after:
48 non-blank rows, every control painted, Refresh click round-trips
("Loaded 0 local research run(s)."), Workspace→Runs roundtrip stays
rendered. New production-CSS regression tests in
Tests/UI/test_research_workspace_geometry.py (vertical stacking +
usable regions at 220x50/120x30/80x24; painted-control containment)
fail 4/4 against the pristine bundle and pass with the fix. Research UI
test files: 150 passed, 1 pre-existing failure
(test_research_sources_region.py sort test, fails identically on the
pristine bundle). preflight.sh all green.

Files: css/features/_research_workspace.tcss, css/tldw_cli_modular.tcss
(generated), Tests/UI/test_research_workspace_geometry.py,
Docs/User_Guide/research_workspace.md (stamp), this task file.
