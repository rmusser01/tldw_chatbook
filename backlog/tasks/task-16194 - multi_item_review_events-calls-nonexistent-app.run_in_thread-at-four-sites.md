---
id: TASK-16194
title: 'multi_item_review_events calls nonexistent app.run_in_thread at four sites'
status: To Do
assignee: []
created_date: '2026-08-14 03:05'
labels:
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Event_Handlers/multi_item_review_events.py:172,256,262,296` call `app.run_in_thread(...)`, which does not exist in Textual 8.2.8 and is defined nowhere in the repo — every one of those code paths dies in AttributeError when reached. TASK-15471 found and fixed the identical bug in `collections_tag_events.py` (rename/merge/delete were all dead on dev) and verified this residue is real and untouched. Fix with the same pattern: `asyncio.to_thread` + the memory-db guard, and add tests that actually drive the four paths (the collections fix's test file is the reference — its born-red evidence was `AttributeError: '_FakeApp' object has no attribute 'run_in_thread'`). Surfaced during TASK-15471 (per-click I/O off-loop, PR #1625 merged `172ada448`) and its concurrency review; evidence in the session review record.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All four sites execute their threaded work successfully (no AttributeError)
- [ ] #2 Tests drive each repaired path, born-red against the current dev behavior
- [ ] #3 A grep confirms no other run_in_thread call sites remain in the repo
<!-- AC:END -->
