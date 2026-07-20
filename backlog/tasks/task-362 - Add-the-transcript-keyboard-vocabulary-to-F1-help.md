---
id: TASK-362
title: Add the transcript keyboard vocabulary to F1 help
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
F1 replaces the whole 2050x1240 screen with ~15 lines of unstyled top-left text listing 5 actions and 7 shortcuts. It does not mention transcript keys (j/k select, Enter show-actions, c copy, e edit, r regenerate, Escape clear), F2 rename in the switcher, Shift+Enter newline, or Escape-to-composer. The in-transcript 'Guide:' line explains icon meanings (♻ ---> 👍👎 🗑) but not their key bindings either, so the c/e/r keys used by this journey are undiscoverable anywhere in the app.

**Repro:** Press F1 on the Console screen at 2050x1240 and compare the listed shortcuts against the transcript BINDINGS (j/k/enter/escape/c/e/r all missing).

**Verifier note:** Evidence j6-a34-f1-help.png matches code exactly: WorkbenchHelpPanel renders only the 5 actions + the 7 CONSOLE_WORKBENCH_SHORTCUTS (chat_screen.py:381-387) as an unstyled top-left dump on a 2050x1240 blank screen; transcript keys j/k/c/e/r/Enter/Esc (console_transcript.py:378-380), F2, Shift+Enter, Alt+M/Alt+1..9 all absent. Aggravating context: task-264 (settled) folded the old pane-contextual footer (which used to surface C/E/R when the transcript was focused, per contextual-footer ledger item) into these same 7 static shortcuts, so c/e/r are now genuinely undiscoverable — but no ledger item or task owns the help-content gap itself. NEW, P2 stands.

**Source:** Console UX expert review 2026-07-20 (finding j6-help-omits-keyboard-vocabulary; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-a34-f1-help.png`, `j6-a18-select-user.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Help presents the full keyboard map (grouped: panes, transcript, composer, modals) in a styled panel sized to content
- [ ] #2 The action-row guide teaches the single-key shortcuts
<!-- AC:END -->
