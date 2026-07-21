---
id: TASK-340
title: Snapshot the composer draft synchronously on Enter so late keystrokes are not folded into the sent message
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux, keyboard]
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In a fresh Ctrl+T tab, with draft 'line one', a Shift+Enter (delivered by the web terminal as plain Enter = send) was followed ~0.4s later by typing 'line two'. The message that got sent was 'line oneline two' — the post-Enter keystrokes were folded into the sent message. Send side-effects (tab rename to '● line oneline', transcript user row) surfaced only ~3s after the keypress with no interim feedback; Backspace presses during the window did not visibly edit the draft.

**Repro:** Ctrl+T for a new tab. Type 'line one'. Press Enter (or Shift+Enter through xterm.js) and immediately type 'line two'. Wait: the sent user message reads 'line oneline two' and the tab is renamed accordingly.

**Verifier note:** Code-confirmed: Enter in the composer calls Button.press() on #console-send-message (chat_screen.py:11201-11209); the draft is only read via composer.draft_text() when the bubbled Button.Pressed handler finally runs (_send_console_message_from_visible_action:8866-8871), while printable keys processed meanwhile mutate the draft via on_key→insert_text — so post-Enter typing is folded into the sent message. The settled clear-at-submission-accept decision (decision-failed-sends-system-rows) governs when the composer CLEARS, not when the draft is CAPTURED; capture-at-keypress was never settled. Genuine message-integrity defect, P1 stands.

**Source:** Console UX expert review 2026-07-20 (finding j6-send-captures-late-keystrokes; P1, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-a31-shift-enter.png`, `j6-a32-post-shift-enter.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Enter snapshots and clears the draft synchronously at the moment of the keypress
- [ ] #2 Anything typed afterwards belongs to the next draft. First-send-in-new-tab latency should show immediate pending feedback
<!-- AC:END -->
