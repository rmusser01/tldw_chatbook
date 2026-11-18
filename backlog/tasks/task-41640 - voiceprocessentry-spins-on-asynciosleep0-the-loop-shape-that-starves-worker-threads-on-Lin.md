---
id: TASK-41640
title: "voice_process_entry spins on asyncio.sleep(0), the loop shape that starves worker threads on Linux"
status: To Do
assignee: []
created_date: 2026-11-17 19:05
labels:
- follow-up
- audio
- flaky
priority: low
references:
- tldw_chatbook/Audio/voice_process_entry.py
- backlog/docs/lessons-testing-evidence.md
dependencies:
- TASK-41317
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found during the TASK-41306..41320 follow-up burn-down (2026-11-17, merged as #3581..#3593). Checked on dev 97f8c58c0c; line numbers are that tree's, and paths are under `tldw_chatbook/` unless they start with `Tests/` or `backlog/`.

**Why.** TASK-40334 showed that an `await asyncio.sleep(0)` polling loop keeps the event loop on-CPU on Linux and starves threads waiting for the GIL. TASK-41317 removed one such loop (the relay publisher's `wait_idle`). `Audio/voice_process_entry.py:186` has the same shape.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The loop waits on an event, queue or future instead of re-queuing itself
- [ ] #2 A test counts loop turns during the wait (as TASK-41317's did) and fails on the old code
<!-- AC:END -->
