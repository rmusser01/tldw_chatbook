---
id: TASK-383
title: Fix the Model header chip degenerating into an empty select on click
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The header chips look like static status text, but clicking 'Model: local-gemma' replaces the chip with an empty blue-outlined box and renders a floating fragment 'Model: local-gemma' below it, overlapping and truncating the 'Settings' button ('Settin') and the rail header. The dropdown offers only the literal chip text as its single option; there is no model list, title, or hint.

**Repro:** 1. On idle Console, click the 'Model: local-gemma' chip in the status row. 2. Chip becomes an empty outlined box; 'Model: local-gemma' floats below it over 'Settings' (truncated to 'Settin'). 3. Escape restores.

**Verifier note:** Mechanics misread but a real render defect remains. There is no Select and no dropdown: ConsoleChip is a focusable Static (console_control_bar.py:80-89) whose click focuses it (focus CSS deliberately lifts the 22-cell ellipsis so the full label shows) and whose tooltip carries the full label — the 'floating fragment' is the tooltip. The genuine defect visible in j5-41: the FOCUSED chip renders as an empty outlined box (label not visible at all), defeating the documented focus-expand-to-read-full-label purpose, while the tooltip overlaps and truncates the 'Settings' action button. Not covered by any ledger item (counter-chips-dim covers styling only). Downgraded to P3: Escape recovers, nothing functional lost.

**Source:** Console UX expert review 2026-07-20 (finding j5-model-chip-degenerate-select; P3, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J5 settings journey. Evidence: `j5-41-chip-click-clean.png`, `j5-01-baseline.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Either chips are static status (then they should not react at all), or they are controls (then they need a visible affordance and a well-formed popover that doesn't occlude the action strip). Half-interactive chips that open a degenerate one-item dropdown confuse both readings
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Served-app-reproduced: clicking the "Model: local-gemma" status chip focused it
and it rendered as an empty bordered box (row: "Provider: Llama_cpp  ┌──────────┐
Assistant:") with the label gone. ROOT CAUSE = the global
`*:focus { outline: solid $ds-focus-accent }` accessibility fallback OVERLAYS a
box onto the height-1 chip (outline draws over the content area, not reserving
space), so on a 1-row widget the box glyphs replace the label. The chip's own
`.console-control-chip:focus` already gives a NON-obscuring cue (high-contrast
$ds-focus-bg/$ds-focus-fg + bold underline), so the fix adds `outline: none` there
to suppress the destructive overlay. Served-app-VERIFIED: the focused chip now
shows "Model: local-gemma" with focus-fg + underline (cell-attr scan) instead of
an empty box. CSS-source contract test + `test_non_obscuring_focus_contract` (106)
still green (the chip keeps a compliant non-obscuring focus).
<!-- SECTION:NOTES:END -->
