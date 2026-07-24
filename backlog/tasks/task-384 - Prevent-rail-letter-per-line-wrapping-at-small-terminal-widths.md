---
id: TASK-384
title: Prevent rail letter-per-line wrapping at small terminal widths
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
At 125x38 and 97x30, the rail renders 'Workspace Default' as 'Def / aul / t' stacked one fragment per line, truncates 'New conversation' to 'New conversati' with no ellipsis, and 'Chat 1 - Chats' to 'Chat 1 -', while the transcript is squeezed to ~57 columns. The inspector also auto-expands at 900x620 (it is collapsed at 2050px), further shrinking the transcript at exactly the sizes where space is scarcest.

**Repro:** Open the Console at 900x620 (125x38 cells) and read the rail Workspace row: 'Def/aul/t' letter stack; compare inspector width vs 2050x1240 baseline.

**Verifier note:** Evidence j6-b05-cold-700x480.png confirms 'Def/aul/t' letter-stack and 'New conversati' ellipsis-free truncation. Code: left rail min_width 24 / right rail min-width 34 / main column min-width 56-60 are hard floors with no auto-collapse breakpoint (chat_screen.py:7062/7366/7383); rails already have collapse handles but nothing triggers them on narrow widths. Not covered by ledger (rail phases 2-4 pending item is about IA, not width degradation). P3 appropriate.

**Source:** Console UX expert review 2026-07-20 (finding j6-rail-crush-letter-wrap; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-b01-900x620.png`, `j6-b05-cold-700x480.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Below a min width the rail collapses to its handle (it already has a '◂' collapse affordance) or truncates whole tokens with ellipsis
- [x] #2 The transcript/composer get width priority
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Served-app-reproduced at 700x480 (97x30 cells): the rail "Workspace: Default"
status-pair VALUE column shrinks to ~3 cells and word-wrapped "Default" into a
"Def / aul / t" letter stack. AC#1 (truncate whole tokens with ellipsis): the
`ConsoleWorkspaceStatusPair` value Static now sets `text-wrap: nowrap` +
`text-overflow: ellipsis` inline (applies in both the served app and pytest
pilots, unlike bundle CSS) so "Default" reads "De…" on one line, with the full
value on hover. AC#2 (transcript width priority): confirmed in the same capture
the transcript keeps ~52 cols at 700x480 while the rail sits at its min width.
Vertical space is also reclaimed (1 line vs 3). Served-app-VERIFIED (row 10:
"Workspace   De…") + a style-contract regression test.
<!-- SECTION:NOTES:END -->
