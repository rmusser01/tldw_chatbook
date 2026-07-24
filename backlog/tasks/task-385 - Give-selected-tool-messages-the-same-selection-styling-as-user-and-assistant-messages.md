---
id: TASK-385
title: Give selected tool messages the same selection styling as user and assistant messages
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
Selecting a user or assistant message shows underline + lighter panel + action row (clearly visible). Selecting a Tool message ('⤷ spawned sub-agent…') shows only the action row; cell scans found no underline/bold/bg change on the message text itself, so with the action row near the fold the selection reads as invisible.

**Repro:** In a conversation whose latest replies include Tool rows, press k repeatedly and compare the selected styling of a Tool message vs a User message.

**Verifier note:** Code-confirmed CSS-ordering defect: .console-transcript-message-tool (color $ds-text-muted; text-style dim italic) is declared AFTER .console-transcript-message-selected (bold underline, $ds-focus-fg) at equal specificity (_agentic_terminal.tcss:2800-2810), so a selected tool row loses the underline/fg treatment and keeps only the near-invisible bg delta. transcript-visual and decision-selected-message-accent-border ledger items cover the selected treatment's design, not this per-role inconsistency. NEW, P3 correct.

**Source:** Console UX expert review 2026-07-20 (finding j6-selection-styling-inconsistent-tool-msgs; P3, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-a17-k2-select.png`, `j6-a18-select-user.png`, `j6-a23-after-r.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 All selectable message kinds share the same selected treatment
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause was a CSS cascade collision, not a missing style. A selected transcript
row carries both `.console-transcript-message-selected` and its role class. The
single-class `.console-transcript-message-tool` (`dim italic` muted) and
`-system` rules follow `-selected` in source order at EQUAL specificity, so for a
selected tool/system row they won the cascade and stripped the selection
treatment (`$ds-focus-fg` + `bold underline`) off the text — leaving only the
action row as a selection cue. FIX = two-class compound selectors
`.console-transcript-message-tool.console-transcript-message-selected` /
`…-system…-selected` that out-specify the muted rules and re-assert the focus
colour + bold underline, so every selectable message kind reads the same when
selected. Regenerated bundle. CSS-source contract test (source + bundle) mirrors
the `test_non_obscuring_focus_contract` pattern; full focus-contract suite green.
<!-- SECTION:NOTES:END -->
