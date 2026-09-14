---
id: TASK-32542
title: >-
  Library Notes: the autosave status clock is UTC ("Saved 05:48" at 22:48 local)
  while Info shows local time
status: To Do
assignee: []
created_date: '2026-09-13 06:46'
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
- [ ] #1 The status line's Saved time is rendered in the user's local time, or replaced by a relative form ("Saved just now")
- [ ] #2 The status line and Info agree on the same save (same instant, same zone)
- [ ] #3 A test pins the rendering for a non-UTC zone
<!-- AC:END -->
