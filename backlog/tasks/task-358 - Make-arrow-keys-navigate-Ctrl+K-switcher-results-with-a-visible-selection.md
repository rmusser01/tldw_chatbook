---
id: TASK-358
title: Make arrow keys navigate Ctrl+K switcher results with a visible selection
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the Switch Session modal, ArrowDown from the search field (x3) changed nothing but the input cursor blink. After Tab into the results, ArrowDown (x2) still moved nothing — results are plain Buttons in a Vertical (no list widget), so navigation is Tab/Shift+Tab only. The focused result's only distinction is bg #272727 vs #1e1e1e on siblings (~1.08:1). Enter in the search box activates the FIRST result regardless of any focused result.

**Repro:** Press Ctrl+K. Press ArrowDown 3x (no visible change; buffer diff empty). Press Tab (first result gains bg #272727), then ArrowDown 2x (no change). Tab again moves to second result. Enter activates focused result.

**Verifier note:** Code-confirmed: console_session_switcher_modal.py BINDINGS are only escape/f2; results are plain Buttons in a Vertical, no list widget or arrow handling; phase3 plan (2026-07-04-console-keyboard-layer-phase3.md) is silent on arrow navigation. The Enter-activates-first-result part merely restates shipped design (ledger ctrl-k-switcher) — but the absent up/down navigation and the unfocused-vs-focused Button delta (no :focus rule for .console-switcher-result) are uncovered; ledger gap-not-exercised-2026-07 confirms the switcher was never live-tested before. Not a regression. Downgraded P1→P2: type-to-filter+Enter and Tab navigation still work.

**Source:** Console UX expert review 2026-07-20 (finding j6-switcher-arrows-dead; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-a09-ctrlk-open.png`, `j6-a10-ctrlk-down3.png`, `j6-a11-ctrlk-tab.png`, `j6-a11-ctrlk-tab-down2.png`, `j6-a12-ctrlk-tab2.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Quick-switcher idiom: Up/Down moves a clearly highlighted selection through the result list, Enter activates the highlighted item
- [ ] #2 The selection highlight meets a ~3:1 contrast ratio
<!-- AC:END -->
