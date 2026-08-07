---
id: TASK-3020
title: Media select-mode polish batch from the P1 arc final review
status: To Do
assignee: []
created_date: '2026-08-07 12:20'
labels:
  - library
  - media
  - polish
  - uat-2026-08-06
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The `fix/library-uat-p1s` final whole-branch review (2026-08-07) triaged these as ride-as-follow-up
minors, all in the Media select-mode neighborhood shipped by task-2853/2856. File positions cite
that branch's head `6672ed276`.

1. `library_screen.py:8751-8769` — the bulk-delete confirm handler starts a bare `run_worker`; a
   fast double-press launches two workers over the same ids. `mark_as_trash` is idempotent but the
   second worker decrements the rail count again (floored at 0) → transient under-report. Fix:
   `exclusive=True` worker group or an in-flight guard.
2. Escape inconsistency between the two delete confirms: the single-item viewer confirm cancels on
   Escape ("back a step"); an armed bulk-delete confirm does not — Escape moves focus to the rail
   and leaves the armed "Delete N…?" row behind. Add a `confirming_bulk_delete` branch to the
   Escape chain (cancel first, like its own Cancel button).
3. Partial-failure bulk delete leaves nothing focused (the confirm row's Delete button is removed
   by the recompose; entry focus deliberately not re-armed). Focus the first still-checked row.
4. The skill editor's footer does not advertise its working `esc`/`ctrl+s`
   (`_register_footer_shortcuts` has no skill-editor branch) — under-advertising, now a visible
   asymmetry beside the note/prompt editors.
5. The single-item viewer delete never decrements the rail's "Media N" count (pre-existing) —
   now a visible inconsistency beside the bulk path, which does.
6. Copy: `library_screen.py:8846` uses "item(s)" while sibling strings compute singular/plural.
7. `Docs/User_Guide/library/media-and-conversations.md` stamp names only TASK-2857 though
   task-2853's content shipped on the same page (content correct; stamp lags).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A double-press on the bulk-delete confirm cannot run two delete workers or double-decrement the rail count
- [ ] #2 Escape cancels an armed bulk-delete confirm (parity with the single-item confirm), covered by a test
- [ ] #3 After a partial-failure bulk delete, keyboard focus lands on a still-checked row
- [ ] #4 The skill editor's footer advertises its working keys
- [ ] #5 Single-item viewer delete updates the rail count in place, like the bulk path
- [ ] #6 Pluralization and the docs stamp are corrected
<!-- AC:END -->
