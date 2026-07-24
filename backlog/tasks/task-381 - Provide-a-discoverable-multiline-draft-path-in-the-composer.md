---
id: TASK-381
title: Provide a discoverable multiline-draft path in the composer
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
Shift+Enter sent the draft ('line one' became a sent message - expected, since terminals deliver plain CR for Shift+Enter), Ctrl+J inserted nothing ('aaabbb' stayed one line), and F1 Help lists only 'Enter: send' with no multiline or attachment shortcuts at all. The Help overlay itself is a bare unformatted list.

**Repro:** Type text, press Shift+Enter (sends), Ctrl+J (nothing). Press F1 and read the shortcut list.

**Verifier note:** Nuanced: Shift+Enter→newline IS implemented (chat_screen.py:11194-11200) — the observed send is xterm.js delivering plain CR, the same harness class as the documented Alt limitation, so that sub-claim is a tool artifact. But the actionable gaps are real and unrecorded: no terminal-portable newline chord (ctrl+j falls through unhandled), and CONSOLE_WORKBENCH_SHORTCUTS (chat_screen.py:380-388) — the F1/help and footer source — lists only 'Enter send' with no newline, attach, or paste-behavior coverage. P3 correct.

**Source:** Console UX expert review 2026-07-20 (finding j3-no-multiline-composer-path; P3, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J3 attachments journey. Evidence: `j3-67-shift-enter.png`, `j3-74-ctrl-j.png`, `j3-75-help.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Provide and document a newline chord that survives terminals (e.g. Ctrl+J) and mention paste behavior
- [x] #2 Help should cover the composer's real capabilities (attach, paste-path, cap)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC#1: Shift+Enter already inserts a newline but terminals deliver it as a plain
CR (send), so `on_key` now also accepts `ctrl+j` (a control code that survives
every terminal) as a portable newline chord -- same `composer.insert_text("\n")`
path. AC#2: the F1 help Composer group (CONSOLE_WORKBENCH_SHORTCUT_GROUPS) now
documents both newline chords plus the previously-undiscoverable capabilities --
Alt+V paste-image, Attach (up to 5 per message = the cap), and paste/drop a file
path to attach it. The compact footer set (flat CONSOLE_WORKBENCH_SHORTCUTS) is
unchanged. RED->GREEN: on_key ctrl+j newline test + a pure help-group coverage
test. Baseline: test_console_rag_action_without_service_stages_recoverable_blocker
is a load-contention flake (passes solo), unrelated to this change.
<!-- SECTION:NOTES:END -->
