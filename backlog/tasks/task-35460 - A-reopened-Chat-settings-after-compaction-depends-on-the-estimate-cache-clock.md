---
id: TASK-35460
title: A reopened Chat settings after compaction depends on the estimate cache's clock
status: To Do
created_date: 2026-10-18 11:04
dependencies: []
labels:
- model-config-redesign
- follow-up
- flaky-test
priority: low
references:
- tldw_chatbook/UI/Screens/chat_screen.py
- Tests/Chat/test_console_session_settings.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 3 of the Phase 9 Wave 6 integration review (2026-10-18). The problem is pre-existing and unrelated to that wave's changes.

`Tests/Chat/test_console_session_settings.py::test_settings_active_compaction_close_anyway_keeps_provider_work_running_and_reopens_fresh` passes or fails depending on the wall clock:
- Interleaved serial runs at two commits passed 13 of 36 and 17 of 36.
- The pass rate for unchanged code drifted from 83% to about 31% within 25 minutes.

Whether the reopened Chat settings modal shows the fresh summary depends on whether `ConsoleContextReadSnapshot`'s `max_age` (`CONSOLE_SETTINGS_ESTIMATE_TTL_SECONDS = 1.0` in chat_screen.py) has run out between the first open's cache fill and the reopen. `compact_now` writes the new memory straight to the repository, which bumps none of the revisions in the snapshot's key. So within that one second the cache can still return the pre-compaction context.

This was proved by forcing `max_age=0`, which passed 4 of 4, and `max_age=1e6`, which failed 4 of 4 with "Summary for fresh-memory." against "original-memory.". The test runs in no PR lane, so CI does not see it.

It is still open whether a user can see the stale summary on a quick reopen (a product defect), or only the test can (a test defect). This task decides that first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 It is decided, with evidence, whether a Chat settings reopen within the estimate cache window after a compaction can show the pre-compaction summary in the running app, and the answer is recorded in this task
- [ ] #2 If it can, a compaction invalidates the cached context read, so a reopen right after it shows the new summary, pinned by a test that controls the clock
- [ ] #3 The test passes 20 of 20 serial runs, at any spacing, and does not depend on how much wall-clock time passes between the two opens
<!-- AC:END -->
