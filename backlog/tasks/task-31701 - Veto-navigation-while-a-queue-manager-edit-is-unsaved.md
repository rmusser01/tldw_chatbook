---
id: TASK-31701
title: Veto navigation while a queue-manager edit is unsaved
status: Done
assignee: []
created_date: '2026-09-05 12:10'
labels:
  - console
  - ux
dependencies:
  - task-31520
priority: medium
---

## Description (the why)

TASK-31520 made Console navigation lossless and retired the busy-fleet
confirmation dialog -- but that dialog had been the ACCIDENTAL protection
for one genuinely lossy case: an unsaved edit in the prompt-queue manager
modal. The navigation seam dismisses pushed screens before switching, so
navigating with a dirty queue edit open silently discards the user's
typed text (it always did when the fleet was idle; the busy-fleet dialog
merely happened to intercept some of those journeys). The repo's
convention for exactly this shape is the outgoing screen's
`flush_pending_work` veto -- Library's note editor already blocks
navigation on an unresolved dirty state and lets the user resolve and
retry.

## Acceptance Criteria (the what)

- [x] Navigating away from Console with a DIRTY queue-manager edit open is vetoed: the user stays, the modal and typed text are untouched, and a notification says why
- [x] A clean edit view, a saved edit, or no modal never vetoes -- navigation stays instant and lossless
- [x] The veto goes through the existing `flush_pending_work` seam (no new dialog, no confirm_navigation revival)
- [x] Guard tests cover the veto, the allow paths, and are mutation-tested (disabling the veto loop fails the guard)

## Implementation Plan (the how)

1. `ConsolePromptQueueModal.has_unsaved_edit()`: True only when an edit
   view is open AND its text diverges from the queued entry's current
   text (an unchanged view or a changed/vanished entry protects nothing).
2. `ChatScreen.flush_pending_work()`: walk the app screen stack for a
   `has_unsaved_edit` provider; veto with a warning notification.
3. One journey guard test covering no-modal / clean-edit / dirty-edit /
   saved-edit; mutation-test the veto.

## Implementation Notes

Implemented exactly per plan. The dirty check lives ON the modal (it
owns the controller, session id, and revision), so the screen hook stays
a generic stack walk -- any future modal with recoverable typed state
opts in by defining `has_unsaved_edit`. The veto fires through the same
`flush_pending_work` seam Library's note editor uses: app.py consults it
BEFORE dismissing overlays, so a veto leaves the modal, text, and focus
untouched. Quit is deliberately out of scope (confirm_quit already
warns, and quit cancels everything regardless).

Verified: the journey guard passes (no-modal allow, clean-edit allow,
dirty veto with modal/text preserved, save-then-allow); disabling the
veto loop fails it; prompt-queue + console reuse + suspend-contract +
parallel-runs suites 58 passed; ruff clean.

Files: `Widgets/Console/console_prompt_queue_modal.py`,
`UI/Screens/chat_screen.py`, `Tests/UI/test_console_prompt_queue.py`.
