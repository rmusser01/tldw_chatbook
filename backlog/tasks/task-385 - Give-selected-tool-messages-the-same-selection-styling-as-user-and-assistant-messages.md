---
id: TASK-385
title: Give selected tool messages the same selection styling as user and assistant messages
status: To Do
assignee: []
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
- [ ] #1 All selectable message kinds share the same selected treatment
<!-- AC:END -->
