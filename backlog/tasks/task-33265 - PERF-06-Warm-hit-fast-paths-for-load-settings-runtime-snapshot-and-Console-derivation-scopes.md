---
id: TASK-33265
title: 'PERF-06: Warm-hit fast paths for load_settings/runtime snapshot and Console
  derivation scopes'
status: To Do
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
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-32804.1 gave get_cli_setting an unguarded warm-hit path, but it is only 1 of 16 functions wrapped by config_participants.guarded. Warm load_settings(), get_runtime_config_snapshot() and get_model_cache_dir() still pay the full ADR-126 storage-admission handshake: about 650 open() calls and 9-20 ms per call. The handshake lands on the 4 Hz Console credential poll (idle Console at 7.7-9.7% of a core), on 1-3 calls per keystroke (26-73 ms/key), on about 120 calls per Console visit, and on about 400 during the first send. Console sync passes (_sync_native_console_chat_ui, settings summary, character context) run without _console_derivation_scope(). Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-06; every issue with file:line is listed under PERF-06 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A warm load_settings()/get_runtime_config_snapshot() hit performs zero storage admissions and zero open() calls (census-pinned)
- [ ] #2 External config edits and generation changes still force a guarded reload (existing config reload tests pass)
- [ ] #3 Typing in the Console composer performs zero config admissions per keystroke (PERF-01 census)
- [ ] #4 Idle Console loop-thread CPU is under 2% of a core on the idle probe
- [ ] #5 The listed Console sync passes run inside a derivation scope
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
