---
id: TASK-18515
title: Console review-note management modal
status: Done
assignee:
  - '@Robert'
created_date: '2026-08-18'
updated_date: '2026-08-18 18:51'
labels:
  - console
  - notes
  - ui
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console's inline review-note marker (the small annotation row under a message with saved notes) is currently non-interactive except for a papercut: clicking it can disturb the transcript's message selection instead of surfacing the notes themselves, and there is no way to read, edit, or remove a note once it exists. Console needs a lightweight modal, reachable from the marker or a keyboard action, that lets a user review every note anchored to a message and manage it (edit the comment, delete a note, or delete the last one and have the marker itself disappear) without disturbing the surrounding transcript or its existing sidecar (citation-sources / feedback) event wiring.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clicking the review-note marker opens the notes modal instead of toggling the transcript's message selection
- [x] #2 Pressing `n` on a selected message that has notes opens the notes modal; pressing it on a selected message with no notes shows a toast instead
- [x] #3 Editing a note in the modal persists only the comment text, leaving the note's other fields untouched
- [x] #4 Deleting a note asks for confirmation first, then soft-deletes it
- [x] #5 Deleting a message's last remaining note removes its marker from the transcript
- [x] #6 The existing citation-sources and selection-feedback sidecar events are unaffected (covered by their pinned tests)
- [x] #7 Docs/User_Guide/console.md (or its dedicated console/*.md page) documents the marker click and `n` action
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Task 1 (this commit): clickable annotation marker + `n` keyboard action --
   ConsoleAnnotationMarker/ConsoleReviewNotesRequested in console_transcript.py,
   posting a request instead of toggling selection; toast when the selected
   message has no notes.
2. Task 2: wire ConsoleReviewNotesRequested on the owning ChatScreen to open a
   new ConsoleReviewNotesModal listing every note anchored to the message.
3. Task 3: per-note edit (comment-only persistence) and delete (confirm ->
   soft-delete; last note removes the marker) inside that modal, plus the
   Docs/User_Guide/console.md update.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 1 landed: ConsoleAnnotationMarker (Static subclass) replaces the
anonymous Static the annotations row used to build; its on_click stops the
event and posts ConsoleReviewNotesRequested(anchor_message_id) instead of
letting the click fall through to the transcript's message-selection click
handling. "console-transcript-annotations" was added to
PROTECTED_CLICK_CLASSES as a belt for the capture-reroute click path. The
transcript's BINDINGS gained a plain `n` -> action_open_review_notes entry
(no on_key branch needed -- printable-key bindings already fire while the
transcript holds focus); it posts the same request when the selected
message has notes, else toasts "No review notes on this message."
(severity="warning"). No modal exists yet -- that is Tasks 2/3, so every AC
above stays unchecked until they land.

Files: tldw_chatbook/Widgets/Console/console_transcript.py,
Tests/UI/test_console_annotation_markers.py (3 new pilot-driven tests:
marker click leaves selection alone and requests notes, `n` on a noted
selection requests notes, `n` on a note-less selection toasts and requests
nothing).
<!-- SECTION:NOTES:END -->

## Implementation Notes

Per-note Edit and soft-Delete for Console review notes, reached from the ✎
marker (click) or `n`. `ConsoleAnnotationMarker` replaces the anonymous
Static and joins `PROTECTED_CLICK_CLASSES` — closing the phase-4 papercut
where clicking the marker toggled message selection.
`ConsoleReviewNotesModal` takes injected `on_edit`/`on_delete` callables and
imports no DB code; the screen owns the off-thread fetch, never-raises
wrappers, and the preview reload. Two invariants are test-pinned: the
sidecar `user_feedback` row is byte-identical after edit AND delete, and
`quote_text` is never written.

Review findings fixed en route: comments citing the wrong (closed) task id;
a missing inflight latch against rapid double-trigger — the THIRD instance
of that class in this program (the non-exclusive-worker rationale keeps
getting copied from `_console_selection_feedback_flow` while its `_inflight`
guard does not; both halves are load-bearing).

**Live verification earned its keep**: it caught a marker surviving the
delete of its last note. The DB and sidecar were correct; the transcript
never re-rendered, because (1) the flow left the reload to a sync tick that
only runs during active runs — phase 4 only looked correct because writing
a note also dispatches a message — and (2) the refresh key omitted
annotation previews entirely (latent phase-4 gap). Both fixed and re-run
end to end on a clean profile.

Files: `Widgets/Console/console_transcript.py`,
`Widgets/Console/console_review_notes_modal.py` (new),
`UI/Screens/chat_screen.py`, `Tests/UI/test_console_annotation_markers.py`,
`Tests/UI/test_console_review_notes_modal.py` (new),
`Tests/UI/test_console_modal_dismissal.py`, user guide, ADR-068 amendment 6.
