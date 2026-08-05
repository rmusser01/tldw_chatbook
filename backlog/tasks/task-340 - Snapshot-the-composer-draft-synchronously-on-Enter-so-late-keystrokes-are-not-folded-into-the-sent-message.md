---
id: TASK-340
title: >-
  Snapshot the composer draft synchronously on Enter so late keystrokes are not
  folded into the sent message
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 01:35'
labels:
  - console
  - ux
  - keyboard
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
- [x] #1 Enter snapshots and clears the draft synchronously at the moment of the keypress
- [x] #2 Anything typed afterwards belongs to the next draft. First-send-in-new-tab latency should show immediate pending feedback
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ConsoleComposerBar: add stash_draft_for_send() (capture segments+text, clear synchronously; None when empty) and restore_stashed_draft() (prepend stashed segments, preserving paste-segment state)
2. chat_screen Enter branch: stash at keypress, then Button.press(); handler consumes stash as the send payload (mouse click path unchanged, reads live draft)
3. Restore the stash on every non-send path: command/fallback dispatch (restore BEFORE dispatch so command semantics stay byte-identical), unknown-command hint, blocked-send gate, run-in-progress, controller-level refusal (result.accepted False)
4. Guard the two accept-time clear sites (_on_console_submission_accepted, should_clear_draft branch) so a stashed send never clears post-Enter typing
5. TDD: failing tests first for late-keystroke fold-in, blocked-send restore, unknown-command armed flow, paste-segment preservation, accepted-send-keeps-next-draft
<!-- SECTION:PLAN:END -->

## Implementation Notes

Enter now captures the draft synchronously at the keypress via a new
`ConsoleComposerBar.stash_draft_for_send()` (segments + canonical text +
paste provenance; composer clears immediately — that clear IS the instant
pending feedback) and hands the stash through `Button.press()` to the send
handler. Every non-send path restores it with `restore_stashed_draft()`
(stashed segments prepended ahead of anything typed since): command/fallback
dispatch (restored BEFORE dispatch so `/prompt`-style live-composer semantics
stay byte-identical), unknown-command hint (armed second-Enter compare still
works), blocked-send gate, run-already-running, and controller-level refusal
in `_submit_console_native_draft`. The two accept-time clear sites
(`_on_console_submission_accepted`, the `should_clear_draft` branch) skip
clearing for stashed sends so post-Enter typing survives as the next draft.
Mouse-click sends are unchanged (no keypress to race).

Verified: 4 new UI tests in `Tests/UI/test_console_send_draft_snapshot.py`
(fold-in reproduced RED first: 'line oneline two'), plus live served-app
check against llama-server — sent row clean, composer holding the post-Enter
text. Files: `Widgets/Console/console_composer_bar.py` (+`ConsoleDraftStash`),
`Widgets/Console/__init__.py`, `UI/Screens/chat_screen.py`.
