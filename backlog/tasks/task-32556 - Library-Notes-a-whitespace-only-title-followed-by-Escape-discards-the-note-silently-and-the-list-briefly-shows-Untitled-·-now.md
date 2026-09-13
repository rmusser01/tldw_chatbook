---
id: TASK-32556
title: >-
  Library Notes: a whitespace-only title followed by Escape discards the note
  silently, and the list briefly shows "Untitled · now"
status: To Do
assignee: []
created_date: '2026-09-13 06:48'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors, persona Riley. Residual of task-32133 (blocked-save Escape veto).

**What happened.** New note, whitespace-only title, Escape: the editor leaves to the list with no "Can't leave yet" toast and no receipt; the DB row is `Untitled`, content 0, `deleted=1` (discarded, as the guide documents for an untouched blank note); the list momentarily still showed "Untitled · now" (B 55 line 39 + DB; A 66). Captures: A 66; B 55.

**Cause.** PROVEN by DB read; the discard is documented behaviour, the silence is not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Leaving a blank note whose only content is a whitespace title shows the documented "Can't leave yet…" prompt, or a one-line "Empty note discarded" receipt
- [ ] #2 The list never paints a row for a note that is about to be discarded
<!-- AC:END -->
