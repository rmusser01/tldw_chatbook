---
id: TASK-32050
title: >-
  Library Notes: opening an existing note never finishes ('Loading note…' with
  no timeout)
status: To Do
assignee: []
created_date: '2026-09-08 18:22'
labels:
  - library
  - notes
  - ux
  - critique-8
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Any note opened from the Notes list stays on 'Loading note… · Next: Wait for loading to finish.' indefinitely, with no failure state, Retry or Cancel; after the first hang every later open hangs too, and only notes created in the same session are editable. Reproduced five times across both assessors and two diagnostic sessions (a 60-byte note opened first also hangs). A thread dump during the hang shows the event loop idle and no worker thread running the load, so the load outcome is dropped or the coroutine is cancelled before it paints (_refresh_library_note_detail generation guards / open_session request token / _run_library_service_call's asyncio.run inside to_thread are the suspects). The notes guide was verified working on 2026-09-06 (fix/library-uat-31796-31797), bounding the regression window. This breaks the core write-reopen-summarise loop. Evidence: critique #8 snapshot `.impeccable/critique/2026-09-08T16-05-41Z__tldw-chatbook-ui-screens-library-screen-py.md` (dev 1c022378cb, 2026-09-08, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile). Register row 1.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Opening any existing note from the Notes list renders the editor with title and body on the seeded profile, including the 35 KB note
- [ ] #2 A note load that exceeds a deadline (about 3 s) shows the existing failed state with Retry instead of a permanent 'Loading note…'
- [ ] #3 After a failed or slow load, opening another note still works in the same session
- [ ] #4 The root cause is identified in the task notes and a regression test opens a stored note through the real note session port (not a fake service)
<!-- AC:END -->
