---
id: TASK-39145
title: "With config unparseable and the machine heavily loaded, the Console's core push can lose every non-blocking config-lock probe for 10 s or more and never run"
status: To Do
created_date: 2026-11-07 02:28
labels:
- follow-up
- console
- config
priority: low
references:
- tldw_chatbook/UI/Console_Modules/config_sync.py
- tldw_chatbook/UI/Console_Modules/console_spend_projection.py
- Tests/UI/test_console_projection_reads_off_loop.py
dependencies:
- TASK-38967
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, and in TASK-38967's notes (round-11 branch fu/r11-console). Line numbers are dev's.

**How the push waits.**
- `sync_console_core_state` (`UI/Console_Modules/config_sync.py:264`) probes both config locks without blocking (`:283-287`).
- If either is busy, it replays the whole FULL sync (`:292`) instead of pushing.

**Why it can starve.**
- With config unparseable, every off-loop config read holds those same locks through a rebuild that fails.
- So under heavy load the push can lose every probe.
- TASK-38967's notes record two failures of `Tests/UI/test_console_projection_reads_off_loop.py::test_an_unparseable_config_leaves_an_idle_console_idle` (`:602`) at load 48-122 in this mode: "the full sync never pushed", 4 FULL syncs and 0 core pushes in the 10 s deadline.

**What is already fixed.** Round-11 branch fu/r11-console fixes the separate replay loop (`rewarm_console_presentation`). This mode remains, and TASK-38967 leaves it as a proposed follow-up because making it deterministic needs a design change.

**Why it matters.** While config cannot be parsed, the Console never pushes its provider selection, and each attempt costs a whole FULL sync.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 With config unparseable and the config locks held continuously by failing off-loop reads (held deterministically in a test), the core push still completes, or reaches a defined state the user can see, within one bounded period; the test is red on dev
- [ ] #2 Lock contention no longer costs a whole FULL sync replay per attempt; replays are counted in the test, before and after
- [ ] #3 `test_an_unparseable_config_leaves_an_idle_console_idle` and the TASK-38967 replay pins stay green, and the unparseable-config test passes 10 of 10 at `-n 4` at load 40 or more, with the load recorded
<!-- AC:END -->

Possible directions (not required): a cold read that does not hold the config locks, or not replaying the whole sync on lock contention.
