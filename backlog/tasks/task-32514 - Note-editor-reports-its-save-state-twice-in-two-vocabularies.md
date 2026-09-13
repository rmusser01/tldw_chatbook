---
id: TASK-32514
title: Note editor reports its save state twice, in two vocabularies
status: To Do
assignee: []
created_date: '2026-09-11 17:25'
labels:
  - library
  - notes
  - ux
  - honesty
  - critique-notes-2026-09
  - rider
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The open note editor states its save state twice on one screen, in two
different wordings, from two different producers:

- the workbench authority line at the top reads `Saved 16:47 · Next: Keep
  editing; changes save automatically.` — built from `status_line`, which
  carries the timestamp the session records on a successful save
  (`Library/library_notes_session.py`, `f"Saved {saved_at.strftime('%H:%M')}"`);
- `#library-note-status`, two rows below it, reads plain `Saved` — built from
  `resolve_database_note_status_channels`, which has no timestamped form at
  all.

So the same fact is told twice and the more useful telling (when) is the one
the reader is least likely to treat as the save indicator. The critique's own
docs-vs-live table recorded the same split ("Status line 'N words · saved'" →
live "`Saved` / `Saved 04:07`"), and the wave-3 design line for task-32143
named the timestamped vocabulary (`Saved 16:24`) as the existing one — it is
not reachable from the widget that design named.

One vocabulary, one place. Which of the two lines keeps it is the decision;
`#library-note-status` is the one the guide documents as "the status line",
and task-32177 already removed one duplicate status widget for the same
reason.

Evidence: `wave3-caps/chrome-strip/10-235x52-strip-on-open.txt` and
`15-100x30-info-no-strip.txt` (both lines visible in one capture).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A given save state is worded one way on the note editor — the authority line and the status line never show two different sentences for the same state
- [ ] #2 The time of the last successful save is readable on whichever line keeps the save state, not only on the other one
- [ ] #3 The existing status-channel tests (task-32063, task-32133/32358, task-32177) stay green at equal or better strength
- [ ] #4 Live-verified at 235x52 and 100x30 across dirty → saving → saved, with captures
<!-- AC:END -->
