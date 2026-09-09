---
id: TASK-32054
title: >-
  Library Import: failed rows show a raw errno, lack 'Show details', and
  misreport skips
status: To Do
assignee: []
created_date: '2026-09-08 18:23'
labels:
  - library
  - import
  - ux
  - critique-8
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A failed job reads '✗ failed · reading-notes.md · Parse pool could not start: [Errno 28] No space left on device' on a disk with 80 GB free (the real cause was POSIX semaphore exhaustion); the documented 'Show details' row action is absent; files the forecast said 'will skip' become 'failed' with the pool reason; a 6-file batch stacks several 'Import finished — 1 failed' toasts; Retry appends '· attempt 2' where the guide says '· retry 1'. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 5.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Pool-start and OS-level failures map to a plain-language reason with a next step (for example a system resource limit with 'Restart the app, then Retry')
- [ ] #2 Every failed row offers 'Show details' with the underlying error available on demand
- [ ] #3 Unsupported files keep their skip reason and are not offered Retry
- [ ] #4 A batch produces one completion toast
- [ ] #5 The retry suffix matches the user guide (docs or copy updated, one of the two)
<!-- AC:END -->
