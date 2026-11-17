---
id: TASK-41140
title: "Console Performance reds left by TASK-40323: held-pointer [skill], managed received custody over its 10 s bound, and three genuine-capture vars pins"
status: Done
assignee:
- '@claude'
created_date: 2026-11-15 02:20
updated_date: '2026-11-16 23:00'
labels:
- follow-up
- console
- tests
- performance
- dev-red
priority: medium
references:
- Tests/Performance/test_app_wiring_held_pointer_currentness.py
- Tests/Performance/test_console_managed_received_acceptance.py
- Tests/Performance/test_console_skill_discovery_genuine_capture.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed in round 14 of the 2026-10-23 burn-down (closeout, 2026-11-15). Checked against the round-14 stack top c7d4a41717 (on dev beab5c1e4e); line numbers are that tree's.

**Why.** These subprocess pins guard where skill reads and received-send custody run. Red, they cannot catch a regression in either.

**Evidence.**
- `test_app_wiring_held_pointer_currentness.py[skill]`: red at its script `:138`, `source.runtime_reads` still holds a read right after the trust task returns (empty 0.5 s later) (TASK-40323's notes).
- `test_console_managed_received_acceptance.py`: after the 077e5bfa09 and 064d5cb014 re-pins it reaches its end and fails its 10 s received-custody bound, 11.0 s at load 65-70; a trace-settlement thread was opening a ChaChaNotes connection through the private SQLite helper process (TASK-40323's notes).
- `test_console_skill_discovery_genuine_capture.py`: through run.sh at the stack top, 3 failed, 6 passed; the round-14 report names the same stale `vars(...)` globals pin that 064d5cb014 moved (`run_preparation_read` now lives in `Backup_Recovery.runtime_producer_lifetime`).

The stock skill-trust `[send_enter]` / `[send_button]` cases ("Cold Send did not reach original configuration SQL") are TASK-34563.35's behaviour, not this task's.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The genuine-capture nodes are re-pinned to the module that owns the read since 064d5cb014, red then green
- [x] #2 The held-pointer [skill] read lifetime is classified (test timing or a read that outlives its trust task) and fixed accordingly, red first
- [x] #3 The managed received-custody path is measured; the trace-settlement connection is moved off the custody window or the bound is justified with measurements
<!-- AC:END -->

## Implementation Plan

1. Run the three files at the base through run.sh and name each failure.
2. Re-pin the genuine-capture `vars(...)` globals pins to `Backup_Recovery.runtime_producer_lifetime` (064d5cb014), red then green.
3. Classify the held-pointer `[skill]` read lifetime (test timing vs a read outliving its trust task) and fix accordingly, red first.
4. Measure the managed received-custody path; move the trace-settlement connection off the custody window or justify the bound with measurements.
5. Update TASK-40323's notes; close it if every remaining criterion is met.

## Implementation Notes

All three files are green at base cc6006fcf5 plus this change, and the newly green nodes run on the recovered-suites lane. Base run (`-n 4`, load 41-45): 5 failed, 7 passed over the three files.

**AC#1, genuine capture.** The script pinned `callback.__globals__` / `request.__globals__` to `vars(console_preparation_reads)`; since 064d5cb014 the read is defined in `Backup_Recovery.runtime_producer_lifetime` and only re-exported, so the monitor never matched the invoke closure and the held control was never admitted. The script now pins `vars(runtime_producer_lifetime)` and asserts `physical_read.__globals__` is it. Red at base: `[held_popup_body]`, `[held_controller_body]`, `[held_controller_default]` failed at script :806 (`held_task is not None`); green after: 9 passed (`-n 4`, load 41-45).

**AC#2, held-pointer `[skill]`: test timing, not a read outliving its trust task.** A probe of the read left in `source.runtime_reads` at script :138 showed it was not the trust read (that one had retired and left the set before the task returned) but the Console runtime's `console-attention-refresh` read (`ConsoleRuntime.request_console_attention_refresh.<locals>.refresh`, task not done). `TldwCli.__init__` -> `_wire_chat_conversation_services` (`app_service_wiring.py`) starts that refresh since 2e6d7f64e8 (2026-10-18), and `source.runtime_reads` is the runtime's shared `_preparation_reads` set. The script now awaits the constructor's attention refresh task and asserts the set is empty before it arms the held read, so the final whole-set assertion keeps its meaning. Red at base: two runs failed at :138; green after: the file passed 2 of 2 in three runs (load 55).

**AC#3, managed received custody: measured; the bound is set from the measurements.** Three serial runs with a temporary 0.25 s await-chain sampler and a `prepare_in_helper` timer (both reverted), load 63-81 on 18 CPUs: release to custody-task end took 15.7, 20.7 and 21.0 s. In that window:
- pre-commit preparation (catalog, trust, configuration capture, ordinary admission) 3.8-6.8 s; durable commit, hooks, mix-in save, tool and provider composition about 3 s; the personal-context bootstrap 2.8-5.5 s; the provider worker and its finish 4.2-6.0 s;
- 50-68 SQLite connection opens through the private helper process, 0.06-0.43 s each;
- the `console-trace-settlement` thread's ChaChaNotes connection, 0.15 s at +15.0 s and 0.12 s at +20.1 s (outside the window in the third run).

Moving that connection off the window would save at most 0.15 s, so the bound was justified instead: the window is a whole cold first turn, and its personal-context step alone has a 10 s product budget (`CONSOLE_PRE_PROVIDER_SETUP_BUDGET_SECONDS`); the rest of the window measured at most 15.3 s. The custody bound (and its cleanup wait) is now `CUSTODY_BOUND = 30` s, with the measurements in a comment; the 240 s child limit, the held-native-read 10 s bound and the 1 s drain are unchanged. Red at base: 10 s bound timed out in five runs (load 41-81). Green after: 3 of 3 serial runs (call 78-98 s, load 80-93). Note for TASK-34563.35, which owns this control and recorded its "original 240/10/1 limits": only the custody 10 s changed.

**Lanes (TASK-40323 AC#3).** `[skill]` and the managed received file joined the recovered-suites job (`derived-artifacts.yml` and the `Tests/CI/test_ci_queue_pressure_contract.py` mirror). Serial with the lane's flags (`--storage-lease-guard --timeout=180`, load 91-97): `[skill]` 22.4 s (setup 3.4 + call 19.0), managed received 71.3 s. Tests/CI: 1640 passed, 1 skipped.

Files: `Tests/Performance/test_console_skill_discovery_genuine_capture.py`, `Tests/Performance/test_app_wiring_held_pointer_currentness.py`, `Tests/Performance/test_console_managed_received_acceptance.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.

**Review r15co-rev1 (2026-11-16): the custody wait is a hang bound.** The 30 s bound was justified only by serial runs and failed 2 of 2 times at `-n 4`. The reviewer's probe measured 15.1-23.8 s serially at load 55-61. Ruling: how fast a received Send must finish is TASK-34563.35's open requirement, not this test's. The wait is now `CUSTODY_HANG_BOUND = CHILD_LIMIT // 2` (120 s; `CHILD_LIMIT = 240` is the script's existing exercise limit, now named), with a comment saying it catches a hang, not latency; the cleanup wait uses the same bound and the failure label is `received_custody_task_hang`. TASK-34563.35's notes record that this test no longer enforces 10 s, with the measurements. At `-n 4` with its Performance peers (the genuine-capture and held-pointer files), load 84-98: the managed node passed 3 of 3 runs (call 138.0, 91.8 and 143.8 s). Runs 2 and 3 passed 12 of 12. Run 1 had 3 genuine-capture failures from load (`HelperTimeoutError('private_sqlite_helper_timeout')`, Textual `WaitForScreenTimeout`); that file runs on no lane.
