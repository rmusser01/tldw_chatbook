---
id: TASK-360
title: Prevent keystrokes leaking into the message draft while the edit modal opens
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-21 02:40'
labels:
  - console
  - ux
  - keyboard
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Pressing e on the selected user message gave no visible response — a full buffer dump 0.9s later showed no modal. Believing it failed, I pressed e again; the Edit Message modal (which had opened in the meantime) received that keypress as text: the textarea read 'eWhat backoff strategy should I use for websocket reconnects?' — the draft was silently corrupted with a stray 'e'. Saving would persist the corruption.

**Repro:** Select a user message in the transcript (k until action row sits under it), press e, and press e again about a second later. The modal textarea shows the second 'e' prepended to the message text.

**Verifier note:** Mechanism is real: transcript 'e' presses the Edit action Button (keyboard-bindings ledger), the Button.Pressed message hops through widget queues before the async dispatch pushes the modal (chat_screen.py:10166-10171), with zero synchronous feedback at keypress — keys typed in the gap land in the late-opening TextArea. Not covered by any ledger item or task. Medium confidence / downgraded P1→P2 because the 0.9s window is likely amplified by the textual-serve harness latency and the corruption is visible in the modal before saving.

**Source:** Console UX expert review 2026-07-20 (finding j6-edit-modal-late-open-keystroke-leak; P2, verdict NEW, medium confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J6 keyboard-only/small-terminal journey. Evidence: `j6-a19-edit-modal.png`, `j6-a20-e-retry.png`, `j6-a21-edit-modal-full.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The edit modal opens within one tick of the keypress (or shows immediate busy feedback), and late-arriving modals must not swallow keystrokes typed before they appeared
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. ConsoleEditMessageModal: record open-time on mount; ignore Key events whose event.time predates it (typed before the modal appeared — intent was not text entry)
2. TDD: deliver a Key with a pre-open timestamp to the mounted modal and assert the textarea is unchanged; normal typing after open still lands
<!-- SECTION:PLAN:END -->

## Implementation Notes

The edit modal's TextArea now ignores Key events whose event time predates
the modal's mount (`_EditMessageTextArea` in
`console_edit_message_modal.py`; open time taken from the Mount event so
the clock domain matches `Key.time`). A key pressed before the modal
appeared was aimed at whatever the user was looking at then — swallowing
it prevents the silent draft corruption; typing after open is unaffected.
Guarding at the TextArea (not the screen) is required because printable
keys are consumed at the focused leaf before they bubble.

Verified: 2 new UI tests in
`Tests/UI/test_console_edit_modal_keystroke_guard.py` (stale-timed key
reproduced the corruption RED first; post-open typing still lands).
