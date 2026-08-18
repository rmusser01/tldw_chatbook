---
id: TASK-17658
title: 'Console: transcript message separators span the full width'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-17'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner report 2026-08-17: the lines separating each message in the conversation pane stop about 4/5 of the way across. Root cause: the separator was a fixed `"─" * 200` string in a Static — exactly 200 columns, so any transcript wider than 200 (the owner's terminal is ~250 columns) left the tail blank. Replaced with the stylesheet's `hatch: horizontal` fill, which paints edge to edge at any width; the widget's text renderable is now empty so no second, width-limited dash layer sits over the hatch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Message separators paint `─` from the transcript's left content edge to its right content edge at any terminal width, pinned by a painted assertion at 250 columns
- [x] #2 The superseded fixed-length contract test is replaced by the new one (empty renderable + painted full-width pin)
- [x] #3 Transcript and focus-contract suites green; bundle rebuilt from source
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: painted full-width pin at 250 cols (bundle harness) — watched fail on the 200-dash tail gap.
2. `.console-transcript-rule` gains `hatch: horizontal $ds-column-line`; `CONSOLE_TRANSCRIPT_RULE` becomes "" (kept as a named constant with the explanation).
3. Replace `test_console_transcript_widget_rules_are_long_enough_to_clip_full_width` (the old >= 160-dash pin) with the empty-renderable contract.
4. Bundle rebuild; suites; live probe in the real app at 250 cols.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Two-line production change: the hatch declaration in `_agentic_terminal.tcss` (+ rebuilt bundle) and the constant emptied in `console_transcript.py` (`to_plain_text` keeps its own width-parameterized rule for exports). RED-first painted pin at 250 columns; the old fixed-length test replaced with its inverse (dash text over the hatch would reintroduce a width-dependent seam). Live probe in the full app at 250 columns: both rules render uniform `─` across their whole 185-column transcript width, edge to edge. 98 transcript tests + 99 focus-contract tests green post-change.

Files: `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ bundle), `tldw_chatbook/Widgets/Console/console_transcript.py`, `Tests/UI/test_console_native_transcript.py`.
<!-- SECTION:NOTES:END -->
