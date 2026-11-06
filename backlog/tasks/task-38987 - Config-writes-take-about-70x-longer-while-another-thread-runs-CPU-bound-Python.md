---
id: TASK-38987
title: "Config writes take about 70x longer while another thread runs CPU-bound Python"
status: To Do
created_date: 2026-11-06 01:42
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
- [ ] #1 One config save on a worker thread, against a spinning Python thread at the default 5 ms switch interval, finishes in 1.0 s or less in 3 of 3 runs, with the load recorded (dev: 5.7-6.4 s)
- [ ] #2 A save with an idle main thread stays at or under 0.10 s (dev: 0.084-0.091 s)
- [ ] #3 If the fix changes the switch interval, the previous interval is restored after the write, including when the write raises and when two writes overlap
- [ ] #4 If the fix cuts the number of calls instead, a counter test pins the GIL-releasing syscalls per save below dev's 13,305 and fails if they come back
<!-- AC:END -->
