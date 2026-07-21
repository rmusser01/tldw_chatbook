---
id: TASK-359
title: Make the F6 rail focus stop visible
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F6 cycles 3 stops: rail -> transcript -> composer. Transcript and composer stops paint a clear accent (#0178D4) pane border, but the rail stop paints NO accent anywhere (verified by full-screen fg-color scan: accent zones empty at that stop). The only style delta is the 3-cell rail-collapse handle '◂' background changing #1e1e1e -> #272727 (~1.08:1 contrast, imperceptible). The FIRST Tab press after the rail stop also changed zero cells (two consecutive keyboard stops visually identical). The Inspector pane is never part of the F6 cycle.

**Repro:** Open Console (seeded home, 2050x1240). Focus composer. Press F6 once (rail stop): no accent border appears anywhere; compare with a second F6 (transcript: accent rows 14-68) and third (composer: accent rows 71-75). cell_attrs_row on row 13 shows only the '◂' handle bg shifting #1e1e1e->#272727.

**Verifier note:** Code-confirmed: F6 rail stop focuses #console-context-rail-collapse (CONSOLE_FOCUS_TARGETS_BY_PANE, chat_screen.py:372-377); .console-rail-collapse-button has no :focus rule (default Button bg shift only) and #console-left-rail:focus paints border $ds-column-line — identical to the region frame color, i.e. invisible by construction (_agentic_terminal.tcss:2170). Adjacent settled decision rail-layout-quiet-focus (task-149) covers only removing the loud focus boundary on the rail BODY scrollable, not the F6 pane-stop indicator (F6 convention is task-103, which never specified visible indication). Downgraded P1→P2: one invisible stop in an otherwise clear cycle; the umbrella focus-visibility finding carries P1.

**Source:** Console UX expert review 2026-07-20 (finding j6-f6-rail-stop-invisible; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-a06-cycle1.png`, `j6-a06-cycle2.png`, `j6-a06-cycle3.png`, `j6-a07-invisible-stop.png`, `j6-a08-rail-tab1.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every F6 stop is visibly distinguishable with the same pane-border treatment (accent border on the rail like transcript/composer), so the user always knows which pane owns focus
<!-- AC:END -->
