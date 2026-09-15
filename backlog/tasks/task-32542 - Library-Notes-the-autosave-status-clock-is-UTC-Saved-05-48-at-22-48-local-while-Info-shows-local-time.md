---
id: TASK-32542
title: >-
  Library Notes: the autosave status clock is UTC ("Saved 05:48" at 22:48 local)
  while Info shows local time
status: Done
assignee:
  - '@claude'
created_date: '2026-09-13 06:46'
updated_date: '2026-09-14 14:45'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), assessor A (B recorded the same string), persona Jordan, Create workflow. P2 #6.

**What happened.** First note autosaves: status line "Saved 05:48" at 22:48 PDT (A 07; B 09 "Saved 05:48"). Later "Saved 05:54" beside Info's "Modified 2026-09-12 22:54 · just now" (A 18). A first-timer reads it as "saved seven hours ago" or "wrong". Captures: A 07, 18; B 09.

**Cause.** PROVEN that the note-session coordinator's clock is `datetime.now(timezone.utc)` (`tldw_chatbook/UI/Screens/library_screen.py:3866` and `:3870`, both the port and the coordinator); that the status line prints that value unconverted is INFERRED. Docs: notes.md's status states list "Saved" with no clock.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The status line's Saved time is rendered in the user's local time, or replaced by a relative form ("Saved just now")
- [x] #2 The status line and Info agree on the same save (same instant, same zone)
- [x] #3 A test pins the rendering for a non-UTC zone
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live under TZ=America/Los_Angeles: create a note, compare the status 'Saved HH:MM' with Info's 'Modified …'.
2. RED: session test under a TZ fixture with a fixed UTC clock -> 'Saved 22:48'; pin file test that the status line and Info render the same instant in one zone.
3. Fix: render saved_at through astimezone() (display only; the persisted modified_at is unchanged).
4. GREEN, live capture, guide.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The session coordinator's clock is `datetime.now(timezone.utc)` by design and
the persisted `modified_at` stays UTC; the status line printed that instant
unconverted. One seam — the new `DatabaseNoteSessionCoordinator.saved_status_message()`
— renders it through `astimezone()`, the same conversion Info's absolute label
already applies (`_absolute_local_label`). Display only: nothing persisted
changes, which the session test pins by asserting the baseline `modified_at`
is still `2026-09-13T05:48:00+00:00` while the status reads "Saved 22:48".

Live (235x52, PDT): a new note autosaved with the status line reading
"Saved 07:34" beside Info's "Modified 2026-09-14 07:34 · just now" at 07:34
PDT / 14:34 UTC, with the stored `last_modified` still
`2026-09-14T14:34:57.982Z`
(`wave4-caps/data-truth/data-23-saved-local-clock-status`,
`data-24-info-vs-status-clock`).

Files: `tldw_chatbook/Library/library_notes_session.py`,
`Tests/Library/test_library_notes_session.py`,
`Tests/UI/test_library_notes_w4_data_truth.py`,
`Docs/User_Guide/library/notes.md`.
<!-- SECTION:NOTES:END -->
