---
id: TASK-39131
title: "A Console FULL sync is refused on the storage-admission lock more often right after a warm-up re-read, and nothing records what holds the lock"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: '2026-11-07 07:13'
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
- [x] #1 Over a stated workload, every storage-admission lock refusal in a FULL render records the holder's thread and stack; counts with and without a preceding re-read are given
- [x] #2 If the warm-up re-read path is the holder, a render after a re-read no longer refuses on the lock, with a deterministic test red before; otherwise the notes name the holder and the product is left unchanged
- [x] #3 FULL-sync replays per workload are measured before and after
<!-- AC:END -->

## Implementation Plan

1. Find the holder first, with a scratch probe (not committed) loaded through `PYTEST_PLUGINS` so that xdist workers and private-profile children load it too. It proxies `storage_admission._lock`, recording each acquirer and, on a non-blocking refusal, the owner's thread and stack (`repr(RLock)` names the owner's ident; `sys._current_frames()` gives its frame). It also wraps the FULL's warm-up, `project_console_tick` and `_checked_display_status`, to tag each refusal with whether the FULL's warm-up re-read.
2. Workload: TASK-38967's three files at `-n 4`, interleaved base/head arms.
3. If the re-read path holds the lock, fix it with a deterministic test. Otherwise name the holder and leave the product unchanged.

## Implementation Notes

**Result: the warm-up re-read is never the holder, so the product is unchanged (AC#2's "otherwise" branch).**

**Probe (not committed).** `r12p_lock_probe` was loaded through `PYTEST_PLUGINS` by a runner that adds the scratch directory to `PYTHONPATH`, so xdist workers and private-profile children load it too. It installs a delegating proxy over `storage_admission._lock`, which records each acquirer's thread and frame. On a non-blocking refusal it records the owner from `repr(RLock)` (`owner=<ident>`), with that thread's name and current stack from `sys._current_frames()`, or the acquire-time frame when the owner released before capture. It also wraps `ConsoleReadinessConfigProjection.warm` / `_refresh`, `config_sync.prepare_console_presentation_sync`, `project_console_tick` and `_checked_display_status`, to tag each refusal with whether it fell inside a FULL render and whether that FULL's warm-up re-read.

**Workload.** TASK-38967's three files (`test_console_pending_facts_cold_readiness.py`, `test_console_pending_facts_direct_control_cold_readiness.py`, `test_first_persistence_getter_attribution.py`) at `-n 4` through run.sh, base 8b762b52fa and head interleaved: 7 base runs and 6 head runs, load 58-181. All runs passed.

**AC#1, counts.**

| | base | head |
|---|---|---|
| FULL renders | 132 | 116 |
| ... after a warm-up re-read | 47, lock refusals 0 | 36, lock refusals 0 |
| ... with no re-read | 85, lock refusals 2 | 80, lock refusals 4 |
| Storage-lock refusals anywhere (poll ticks, display reads included) | 25 | 17 |

Every one of the 6 FULL-render refusals recorded its holder's thread and stack; none had released the lock before capture:
- 2 x the test's agent run thread (`Thread-2 (_run)`) in `request_mcp_approvals` -> `console_interrupt_rounds.run_round` -> `storage_admission._acquire_storage` / `_scope`;
- 2 x an asyncio executor thread in the attention preparation read (`console_attention_preparation` -> `conversation_local_marks_service.list_console_unseen_marks`);
- 1 x an executor thread in the character-context preparation read (`character_context._read_database_scope_metadata_pair`);
- 1 x an executor thread in `ChaChaNotes_DB.get_transcript_annotations`.

Each of the last three was admitting its DB connection: `_Operation.check` resolves paths (`Path.resolve()`, filesystem syscalls) while it holds the lock. Across all 42 refusals, no holder stack contains a `console_spend_projection` frame, the readiness re-read's module. An earlier smoke run (first probe version, no acquire-time record) showed the same kind of holder: a repository participant's `core_operation` check.

**AC#3, FULL-sync replays per workload.** Base: 1 prepare refusal and 1 render replay over 7 runs. Head: 0 and 2 over 6 runs. This path's code is identical on both arms; the counts are the workload's noise. TASK-38967's correlation (8 of 35 re-read warm-ups refused) did not reproduce: 0 of 83 here. Its own notes suspected the tests' config writes.

Files: this task file and `backlog/docs/lessons-testing-evidence.md` (how to name an RLock's holder).
