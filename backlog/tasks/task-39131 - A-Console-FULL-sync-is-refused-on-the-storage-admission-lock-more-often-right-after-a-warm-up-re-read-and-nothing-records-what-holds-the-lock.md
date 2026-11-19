---
id: TASK-39131
title: "A Console FULL sync is refused on the storage-admission lock more often right after a warm-up re-read, and nothing records what holds the lock"
status: To Do
created_date: 2026-11-07 02:03
labels:
- follow-up
- console
- performance
priority: low
references:
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- tldw_chatbook/UI/Console_Modules/config_sync.py
dependencies:
- TASK-38967
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and in TASK-38967's notes (round-11 branch fu/r11-console). Line numbers are dev's.

**The mechanism.**
- `_checked_display_status` (`console_spend_projection.py:751`) returns False when `storage._lock.acquire(blocking=False)` fails (`:841`). The whole FULL sync then replays.
- The warm-up re-read runs through `warm()` (`:496`), which calls `run()` (`:413`).

**What TASK-38967 measured.** In an interleaved arm on base:
- 8 of 35 warm-ups that re-read were refused on the lock, against 3 of 92 that did not;
- on the branch, 7 of 61 against 4 of 64.

**Why it is not settled.**
- The same notes warn "that correlation may be the tests' own": these files write config.
- Total lock refusals stayed at 11 on both arms while re-reads nearly doubled.
- TASK-38745 (Done) saw the same non-blocking refusal and noted "The holding thread was not captured."
- So the cost is a replayed render, and nothing yet shows who holds the lock at that moment.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Over a stated workload, every storage-admission lock refusal in a FULL render records the holder's thread and stack; counts with and without a preceding re-read are given
- [ ] #2 If the warm-up re-read path is the holder, a render after a re-read no longer refuses on the lock, with a deterministic test red before; otherwise the notes name the holder and the product is left unchanged
- [ ] #3 FULL-sync replays per workload are measured before and after
<!-- AC:END -->
