---
id: TASK-381
title: Provide a discoverable multiline-draft path in the composer
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Shift+Enter sent the draft ('line one' became a sent message - expected, since terminals deliver plain CR for Shift+Enter), Ctrl+J inserted nothing ('aaabbb' stayed one line), and F1 Help lists only 'Enter: send' with no multiline or attachment shortcuts at all. The Help overlay itself is a bare unformatted list.

**Repro:** Type text, press Shift+Enter (sends), Ctrl+J (nothing). Press F1 and read the shortcut list.

**Verifier note:** Nuanced: Shift+Enter→newline IS implemented (chat_screen.py:11194-11200) — the observed send is xterm.js delivering plain CR, the same harness class as the documented Alt limitation, so that sub-claim is a tool artifact. But the actionable gaps are real and unrecorded: no terminal-portable newline chord (ctrl+j falls through unhandled), and CONSOLE_WORKBENCH_SHORTCUTS (chat_screen.py:380-388) — the F1/help and footer source — lists only 'Enter send' with no newline, attach, or paste-behavior coverage. P3 correct.

**Source:** Console UX expert review 2026-07-20 (finding j3-no-multiline-composer-path; P3, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-67-shift-enter.png`, `j3-74-ctrl-j.png`, `j3-75-help.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Provide and document a newline chord that survives terminals (e.g. Ctrl+J) and mention paste behavior
- [ ] #2 Help should cover the composer's real capabilities (attach, paste-path, cap)
<!-- AC:END -->
