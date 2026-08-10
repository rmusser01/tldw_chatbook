---
id: TASK-4026
title: Media overwrite=True silently un-trashes a deleted row
status: To Do
assignee: []
created_date: '2026-08-09 22:10'
labels:
  - media
  - data-integrity
  - recritique-2026-08-09
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the whole-branch review of task-4022's fix wave (2026-08-09), as an out-of-scope
observation — **pre-existing, not introduced by that work**.

`Media/local_media_reading_service.py:1836` ("create local reading item" — a different function
from the `_materialize_reading_import_row` path that task-4022 protected) calls
`add_media_with_keywords(..., overwrite=True)` without `restore_trashed`. If that hits a trashed
row via the full-content-update branch, `_media_payload` (`DB/Client_Media_DB_v2.py:3610-3629`)
hardcodes `is_trash: 0, trash_date: None, deleted: 0` unconditionally — so the row is silently
un-trashed with no explicit restore decision anywhere in the call chain.

This predates task-4022 entirely: `overwrite=True` has always meant "update in place regardless of
trash state". Task-4022 made the *non*-overwrite restore explicitly opt-in
(`restore_trashed: bool = False`), which throws the remaining implicit case into relief — the
overwrite path still resurrects without asking.

Decide the intended contract and make it explicit either way:
- if `overwrite=True` should also require an explicit restore decision, gate the un-trashing in
  `_media_payload` on the same opt-in flag and update the callers that genuinely want it;
- if resurrect-on-overwrite is correct, document it at `_media_payload` and at the callers, and
  cover it with a test so it stops looking like an oversight.

Audit every `overwrite=True` caller while deciding — the reading-list creator is the one the
review named, but it may not be alone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The intended behaviour of `overwrite=True` against a trashed row is decided and stated in code, not implied
- [ ] #2 Whichever way it is decided, a real-DB test pins it
- [ ] #3 Every `overwrite=True` caller is audited against that decision, and any that disagrees is fixed or justified in the notes
<!-- AC:END -->
