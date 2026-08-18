---
id: TASK-18515
title: Console review-note management modal
status: In Progress
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
- [ ] #1 Clicking the review-note marker opens the notes modal instead of toggling the transcript's message selection
- [ ] #2 Pressing `n` on a selected message that has notes opens the notes modal; pressing it on a selected message with no notes shows a toast instead
- [ ] #3 Editing a note in the modal persists only the comment text, leaving the note's other fields untouched
- [ ] #4 Deleting a note asks for confirmation first, then soft-deletes it
- [ ] #5 Deleting a message's last remaining note removes its marker from the transcript
- [ ] #6 The existing citation-sources and selection-feedback sidecar events are unaffected (covered by their pinned tests)
- [ ] #7 Docs/User_Guide/console.md (or its dedicated console/*.md page) documents the marker click and `n` action
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
