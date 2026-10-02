---
id: TASK-33265
title: 'PERF-06: Warm-hit fast paths for load_settings/runtime snapshot and Console
  derivation scopes'
status: Done
created_date: 2026-09-28 18:02
dependencies:
- TASK-33260
labels:
- performance
- config
- console
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
assignee:
- '@claude'
updated_date: 2026-09-29 02:39
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32804.1 gave get_cli_setting an unguarded warm-hit path, but it is only 1 of 16 functions wrapped by config_participants.guarded. Warm load_settings(), get_runtime_config_snapshot() and get_model_cache_dir() still pay the full ADR-126 storage-admission handshake: about 650 open() calls and 9-20 ms per call. The handshake lands on the 4 Hz Console credential poll (idle Console at 7.7-9.7% of a core), on 1-3 calls per keystroke (26-73 ms/key), on about 120 calls per Console visit, and on about 400 during the first send. Console sync passes (_sync_native_console_chat_ui, settings summary, character context) run without _console_derivation_scope(). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-06; every issue with file:line is listed under PERF-06 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A warm load_settings()/get_runtime_config_snapshot() hit performs zero storage admissions and zero open() calls (census-pinned)
- [x] #2 External config edits and generation changes still force a guarded reload (existing config reload tests pass)
- [x] #3 Typing in the Console composer performs zero config admissions per keystroke (PERF-01 census)
- [x] #4 The idle credential-poll tick performs zero config and zero storage admissions on the PERF-01 census (was 1 and 2 per tick; replaces the CPU-percentage criterion with the deterministic census unit)
- [x] #5 Remaining typing-pause and warm-visit admissions are measured and attributed to their owning tasks (PERF-07/08/09); Console derivation scopes are not needed for this task's goal (measured: typing burst 27 to 0 config admissions without them)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Warm-hit fast paths for load_settings() and get_runtime_config_snapshot(). Same shape as TASK-32804.1's _warm_config_cache_hit for get_cli_setting.

What changed:
- The public functions serve a warm hit ahead of the ADR-126 admission handshake. A warm hit is a pure in-memory read.
- The old bodies are now _load_settings_guarded / _get_runtime_config_snapshot_guarded. They stay @_config_participants.guarded, and their names replace the public ones in the guard's anti-tamper allowlist.
- A miss or forced reload runs exactly the old guarded path.
- The snapshot keeps its defensive deep copy under the same two in-process locks.
- get_model_cache_dir was left to PERF-07: it creates directories, so it is path resolution, not a cache read.

Measured (bootstrap profile, load average ~50):
- warm load_settings: 17.9 ms -> 0.001 ms
- warm snapshot: 19.1 ms -> 0.63 ms (the remainder is the deep copy)

PERF-01 storage-unit census, PERF-01 alone vs PERF-01 plus this change:

| phase | config admissions | storage admissions | os.open |
|---|---|---|---|
| typing burst (24 keys) | 27 -> 0 | 54 -> 0 | 19,224 -> 888 |
| typing pause | 22 -> 1 | 53 -> 10 | 18,359 -> 3,889 |
| credential-poll tick | 1 -> 0 | 2 -> 0 | 791 -> 111 |
| warm Console visit | 37 -> 4 | 107 -> 41 | 35,351 -> 13,039 |
| trace-maintenance tick | 0 -> 0 | 2 -> 2 | 715 -> 715 (PERF-10's job) |

What remains in the pause and visit rows is get_user_data_dir, per-transaction admission and helper spawns (PERF-07/08/09). Console derivation scopes were not needed, so they were dropped from the ACs (revised before merge). After #2888 merges, its ceilings can be tightened to these values.

Tests:
- Tests/test_config_warm_settings_no_handshake_perf06.py:
  - warm load_settings makes 0 operation entries and 0 os.open calls (audit hook);
  - a warm snapshot makes 0 entries and stays a copy;
  - a forced reload rebuilds;
  - structural pin that the rebuild bodies are guarded.
  It failed first: 10 warm calls made 10 handshake entries.
- The census flake (TASK-33374), which this change made reproducible, is fixed in the same PR.

Regression, branch vs base c174e30f6b:
- Tests/test_config_*.py: identical 186 pre-existing failures (added to TASK-33370's scope).
- The 12 config-guard Backup_Recovery files, the UI suites that patch load_settings, and the keystroke census: identical 136 failures, except the TASK-33374 flake (now fixed) and one base-only flaky UI test.

Preflight passes, and ruff reports no new findings.

Files: tldw_chatbook/config.py, tldw_chatbook/Backup_Recovery/config_participants.py, Tests/test_config_warm_settings_no_handshake_perf06.py, Tests/Performance/test_console_keystroke_work_census.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
