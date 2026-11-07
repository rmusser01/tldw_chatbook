---
id: TASK-39145
title: "With config unparseable and the machine heavily loaded, the Console's core push can lose every non-blocking config-lock probe for 10 s or more and never run"
status: In Progress
assignee: ['@claude']
created_date: 2026-11-07 02:28
updated_date: '2026-11-07 07:17'
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

## Implementation Plan

1. Trace how the core push meets the config locks while config is unparseable, and what each acceptance criterion needs from the product.
2. Stop before changing code if AC#1's "held continuously" case leaves only a choice of new user-visible behaviour, and report NEEDS_CONTEXT with the options.

## Implementation Notes

**Not implemented: NEEDS_CONTEXT (2026-11-07).** No product or test change. The trace below shows that AC#1, in the case it names, needs a product decision.

**How the push meets the locks (base 8b762b52fa).**
- `config_sync.sync_console_core_state` probes `config._settings_rebuild_lock()` and `config._config_file_lock()` without blocking. It holds both through the push so that the push's live reads (`_provider_readiness_app_config` -> `load_settings()`, and any `get_cli_setting` under `_sync_console_chat_core_state`) can neither wait on the UI loop nor race a rebuild (TASK-36511).
- With config unparseable, no cache is ever installed. Every live read is a rebuild under both locks: it parses, checks the deduplicated aside copy (`_preserve_corrupt_config_aside`) and logs a traceback, then serves the last known good config. Off-loop readers (the readiness display's retries, FULL warm-ups, identity refreshes) take both locks for that whole rebuild.
- A lost probe calls `replay_whole_console_sync`, so each contention costs a whole FULL sync 0.2 s later (AC#2's cost).

**Why AC#1 needs a decision.** With both locks held continuously, which is the case AC#1 asks a test to hold, a push that keeps the TASK-36511 guarantee can never run. Its live reads would wait on the UI loop or rebuild config there. So the push can finish within a bounded period only by one of these:
1. **Complete without the locks.** Push from a config value that needs no lock: the last known good mapping, or one read off-loop and seeded into `_console_derivation_memo["app_config"]`. Every other live config read under `_sync_console_chat_core_state` must then be proved lock-free, or it can still block the UI loop. That is the "cold read that does not hold the config locks" direction, and it changes what TASK-36511's lock holding guarantees.
2. **Stop at a defined, visible state.** After a bounded period of contention, stop retrying and show the user that the provider selection could not be refreshed because the config file cannot be read. This is new Console copy and behaviour, which needs a User Guide entry and an owner ruling on the wording and placement.

Either way, AC#2 has a cheap part: on contention, retry only the push (on the readiness read's settle hook and the next FULL) instead of replaying the whole FULL. It is not done alone, because its end state for the held case is what (1) or (2) decides.

**Question for the owner.** For a config that stays unparseable while its locks stay busy, should the core push (1) complete from the last known good config without the config locks, or (2) stop after a bounded period and show a visible "config unreadable, selection not refreshed" state? If (2), where and with what copy?
