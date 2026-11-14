---
id: TASK-38987
title: "Config writes take about 70x longer while another thread runs CPU-bound Python"
status: Done
assignee:
- '@claude'
created_date: 2026-11-06 01:42
updated_date: 2026-11-10 16:30
labels:
- follow-up
- performance
- config
- storage-admission
priority: medium
references:
- tldw_chatbook/config.py
- tldw_chatbook/Backup_Recovery/native_files.py
- tldw_chatbook/Utils/private_paths.py
- tldw_chatbook/Backup_Recovery/storage_admission.py
- tldw_chatbook/Chat/console_preparation_reads.py
- Tests/UI/test_llamacpp_snapshot_settings.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 8 of the 2026-10-23 burn-down; checked on dev 29592ecc9b, 2026-11-05. TASK-36530 (Done on its branch, which merges first) recorded this as "Accepted with a reason" and proposed a follow-up. On that branch an isolated save took 0.08 s with the main thread idle, and 2.6-11.8 s against a spinning thread at the default 5 ms switch interval.

**Measured on dev.** Each run is one `save_setting_to_cli_config("sync", "sync_log_retention_days", 90)` on a worker thread, through the scratch-profile wrapper, at load 9.6-13.3:

| Condition | Save time (3 runs) |
|---|---|
| Main thread idle | 0.091, 0.087, 0.084 s |
| A spinning Python thread, default 5 ms switch interval | 5.686, 6.334, 6.400 s (about 70x) |
| The same at a 1 ms interval | 1.441, 2.574, 3.845 s |

**Why.** One save makes 32,196 calls into posix/fcntl (counted with `sys.setprofile` c_call). 13,305 of them are syscalls that release the GIL (open, stat, fstat, close, lstat, read, listdir, flock). Every release waits out the busy thread's switch interval before the save gets the GIL back.

The calls come from storage admission and private-path verification. Grouped by the first app frame:

| Frame | Calls |
|---|---|
| `Backup_Recovery/native_files.py:17 pinned_directory` | 3,426 |
| `Utils/private_paths.py:676 _allocate_native_descriptor` | 1,419 |
| `Utils/private_paths.py:432 _walk_verified_parent` | 1,344 |
| `Utils/private_paths.py:697 _native_close` | 1,119 |
| `storage_admission.py:827 _content` | 1,104 |
| `storage_admission.py:812 _posture` | 1,084 |
| `bootstrap.py:270 _validate_ancestry_record` | 800 |

The entry points are `config.py:6455 save_setting_to_cli_config` and `config.py:6343 apply_settings_mutation_to_cli_config`.

**Precedent.** `Chat/console_preparation_reads.py:38-58 _preparation_handoffs` is a ref-counted scope that lowers the switch interval to `_READ_HANDOFF_LIMIT = 0.001` (`:33`) while it is active. It restores the old interval only if nothing else changed it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One config save on a worker thread, against a spinning Python thread at the default 5 ms switch interval, finishes in 1.0 s or less in 3 of 3 runs, with the load recorded (dev: 5.7-6.4 s)
- [x] #2 A save with an idle main thread stays at or under 0.10 s (dev: 0.084-0.091 s)
- [x] #3 If the fix changes the switch interval, the previous interval is restored after the write, including when the write raises and when two writes overlap
- [x] #4 If the fix cuts the number of calls instead, a counter test pins the GIL-releasing syscalls per save below dev's 13,305 and fails if they come back (N/A — the fix took the switch-interval route, AC#3; the call count is unchanged by design)
<!-- AC:END -->

## Implementation Plan

1. Route: the switch interval (AC#3), not the call count. Cutting storage admission's and private-path verification's syscalls means changing security checks; a scoped shorter interval leaves them as they are.
2. One owner for every such scope: `tldw_chatbook/Utils/switch_interval.py` `switch_interval_at_most(limit)`, ref-counted over a multiset of open limits. It installs the smallest open limit (keeping an already shorter interval), restores the interval in force before the first scope when the last one exits (normally or by raising), and leaves alone an interval other code set meanwhile. `console_preparation_reads._preparation_handoffs` delegates to it, so a preparation read and a config write that overlap without nesting can no longer leave the interval lowered (the precedent's own "only if nothing changed it" guard would skip its restore in that order).
3. Wrap `config._apply_literal_settings_transaction_locked`, the transaction every config mutation goes through (`apply_settings_mutation_to_cli_config`, `save_setting_to_cli_config` and the literal transaction), in that scope.
4. Tests in `Tests/test_config_persistence_owner.py` (laned): the write runs under the limit and restores it; a write that raises restores it; two overlapping writes keep it until the last ends; a non-nested overlap with a preparation read restores the original; an interval set by other code is kept. Red first.
5. Measure with the task's probe through run.sh (scratch HOME), base and head, idle and against a spinning thread, recording the load.

## Implementation Notes

**Route: the switch interval (AC#3).** Cutting the ~10,000 GIL-releasing syscalls means changing storage admission's and private-path verification's checks; a scoped shorter interval leaves every check as it is. AC#4 (the call-count route) does not apply.

**Fix.**
- `tldw_chatbook/Utils/switch_interval.py`: `switch_interval_at_most(limit)` is the one owner of such scopes. Under a lock it keeps a multiset of open limits, installs the smallest (keeping an interval that is already shorter), and when the last scope exits, normally or by raising, restores the interval in force before the first one. If other code set the interval while scopes were open, it is left alone.
- `config._apply_literal_settings_transaction_locked`, the transaction every config mutation runs through (`apply_settings_mutation_to_cli_config`, `save_setting_to_cli_config`, `save_settings_to_cli_config`, `delete_settings_from_cli_config` and the literal transaction), now runs the body (renamed `_apply_literal_settings_transaction_unscoped`) inside `switch_interval_at_most(0.0001)`. The overlap rule includes the config write lock wait: a second write waiting on the lock is inside its own scope.
- `console_preparation_reads._preparation_handoffs` (the precedent) delegates to the same owner. Its own copy restored only if nothing had changed the interval, so a preparation read and a config write that overlapped without nesting would have left 0.001 s installed; with one owner they cannot.
- Both modules import the owner on first use, which keeps it off the app-import path (687/688). It is resident at UI ready, because a read or write runs before ready: `MAX_TLDW_MODULES_AT_UI_READY` 1038 -> 1039 under the owner waiver, snapshot refreshed. The rename moved seven diagnostic rows' scope name only (`check_persistent_diagnostic_inventory.py --statements config.py`: 0 moved, 0 removed, 0 added); inventory re-pinned. `config.py` size budget 8983 -> 8989 (owner waiver).

**AC#3 tests** (`Tests/test_config_persistence_owner.py`, laned in "Run admission-sensitive suites 2"):
- `test_a_config_write_runs_with_a_short_switch_interval_then_restores_it`, `test_a_config_write_that_raises_restores_the_switch_interval` (a `BaseException` from `before_replace` escapes the transaction), `test_overlapping_config_writes_keep_the_short_interval_until_the_last_ends` (two threads; the second waits on the write lock inside its scope): red on base, `assert 0.005 < 0.001` for each.
- `test_a_preparation_read_ending_inside_a_config_write_still_restores` and `test_an_interval_other_code_sets_inside_a_config_write_is_kept` hold on base trivially, so their controls were temporary source edits: the precedent's own `_preparation_handoffs` put back gave `assert 0.001 == 0.005`; the helper's "only if we installed it" check removed gave `assert 0.005 == 0.002`. Both reverted.
- The five run in 5.0-6.0 s serially (3 runs, load 42-45); the step they join took 24-26 min of its 35 in recent CI runs.

**Measurements** (the task's probe through run.sh, scratch HOME: `save_setting_to_cli_config("sync", "sync_log_retention_days", v)` on a worker thread, a new value each save, one warm-up save first; base is `fu-base-r12`, run back to back with head). The filed numbers were at load 9.6-13.3; the machine was never that quiet during this round, and the convoy shrinks under load (lesson added), so base is always shown beside head:

| Load (1 min) | Base, spinner, 5 ms | Head, spinner, 5 ms | Base idle | Head idle |
|---|---|---|---|---|
| 29.5-31.8 | 0.87-1.91 s; 1.65-2.08 s | 0.42-0.57 s; 0.48-0.78 s | 0.11-0.17 s | 0.11-0.17 s |
| 45.9-58.3 | 0.39-0.82 s | 0.37-0.64 s | 0.15-0.30 s | 0.13-0.15 s |
| 69.5-70.0 | 0.40-0.75 s | 0.39-0.75 s | 0.28-0.48 s | 0.24-0.29 s |

- AC#1: head finished in 1.0 s or less in every spinning run (3 of 3 in each of five rounds), against base's 0.87-2.08 s at load 30-32.
- AC#2: not shown. Idle saves were 0.11-0.17 s on both base and head at load 30-34 (alternating runs, 10 each: head 0.114-0.139 s, base 0.110-0.154 s), above the criterion's 0.10 s only because of the load; the change does nothing to a save that has no GIL contender. A run at the filing's load (about 10) is still needed for the 0.10 s figure; left unticked.
- Call count (unchanged by design): 26,142 posix/fcntl calls per save in the scratch profile, 10,484 of them GIL-releasing (the filing counted 32,196 / 13,305 in its profile).

**Lesson.** `lessons-testing-evidence.md`: a GIL convoy's slowdown disappears when the machine is loaded; and saving the same value twice is a no-op write.

Files: `tldw_chatbook/Utils/switch_interval.py` (new), `tldw_chatbook/config.py`, `tldw_chatbook/Chat/console_preparation_reads.py`, `Tests/test_config_persistence_owner.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Tests/Performance/test_ui_ready_module_census.py`, `Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt`, `Docs/security/production-diagnostic-inventory.json`, `backlog/docs/lessons-testing-evidence.md`.

**Review r12s-rev1 (M4, M6).** The scope covered `_config_interprocess_lock`'s untimed polling wait, so while another process held the config lock the whole process ran at 0.1 ms (CPU-bound threads 28-47% slower, the review measured). The wait now runs inside `switch_interval_released()`, which lifts only the calling thread's scopes and reopens them after. The `*args, **kwargs` wrapper and the `_unscoped` rename are gone: the scope is the first context in the transaction's own `ExitStack`, so the typed signature is back and the diagnostic inventory is byte-identical to base again. The review asked for the scope after the lock; it stays ahead of it because 3,858 of a save's 10,484 GIL-releasing calls are the lock's admission and lock-file checks (a counting probe: 6,626 inside / 3,858 outside with the scope after the lock, 10,484 / 0 now).
- `test_a_write_waiting_on_another_process_lock_does_not_shorten_the_interval` holds the OS lock from the test and records the interval inside the write's wait. Red on 00243178b1 and with the release removed (`assert 9.999999999999999e-05 == 0.005`); green after, and the write still runs short once it holds the lock.
- `config.py` size budget 8989 -> 8990 (owner waiver).

**Review r12s-rev2 (M-b).** `switch_interval_released()` was not re-entrant: a release nested in a release, or one after a scope opened inside a release, raised `ValueError: list.remove(x)`. Each release now keeps its own snapshot and empties the thread's open-scope list while it lasts. `test_switch_interval_releases_nest_and_reopen` (laned file) was red on 6059958f8c with that `ValueError`, green after; the review's thread/exception probe and 8-thread stress probe show no violations.

**Low-load measurement (2026-11-07, load 14.6-17.0).** The machine reached the filing's range during this round, so base and head were measured back to back with the probe:
- Spinner, 5 ms: base 7.28-11.98 s and 7.36-12.85 s (the convoy); head 0.60-0.64 s and 0.59-0.69 s. AC#1 holds at the load where base shows the 70x slowdown.
- Idle: head 0.094-0.099 s (3 of 3 at or under 0.10 s), then 0.106-0.136 s and 0.093-0.136 s in two more rounds; base 0.104-0.105 s, 0.106-0.109 s and 0.105-0.138 s, above 0.10 s in every run at this load (the filing's 0.084-0.091 s was at load 10-13). AC#2 is left unticked for the owner: head shows no idle regression against base, but its absolute 0.10 s held in only one of three rounds.

**Quiet-host measurement after merge (2026-11-10, load 5.1-10.3, 18 CPUs; AC#2).** The task's probe (`idle`, a new value per save, one warm-up) through run.sh, alternating head = dev ae9f9e8d21 (this task merged) and base = dev aa96a3f48c (just before it), 95 saves each over eight rounds:
- Head: medians 0.058-0.065 s per round; 90 of 95 saves at or under 0.10 s, the rest 0.110-0.129 s.
- Base: medians 0.056-0.075 s; 92 of 95 at or under 0.10 s, the rest up to 0.144 s.
- Both sides show the same occasional host spike, and base's worst exceeds head's, so the tail is the machine, not the change. A save with an idle main thread runs at about 0.06 s, under the 0.10 s criterion and below the filing's 0.084-0.091 s. AC#2 ticked; AC#4 does not apply (switch-interval route).
