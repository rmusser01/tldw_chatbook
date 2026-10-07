---
id: TASK-33275
title: 'PERF-16: TldwCli.__init__ diet - defer feature services and DBs, drop dead
  boot work'
status: To Do
created_date: 2026-09-28 18:03
dependencies:
- TASK-33266
- TASK-33268
labels:
- performance
- startup
- perf-audit-2026-09
priority: medium
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TldwCli.__init__ constructs about 15 services and opens 5 feature DBs before first paint, each paying get_user_data_dir, admission and a helper spawn (about 340 ms for the DBs alone). It also builds 18 interop services with zero consumers, the Evals orchestrator, Watchlists wiring, MCP stores and control plane, persona/dictionary services, collections capture, scheduling maintenance, the notes-sync runtime owner, TTSService, File-Notes-git and cold feature services. Dead boot work (RichLogHandler path, media-type prefetch, duplicate packages) still runs. Owner decision D5: delete zero-consumer services rather than lazy-load them. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-16; every issue with file:line is listed under PERF-16 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The listed services and feature DBs are constructed lazily or in the post-_ui_ready boot tier
- [ ] #2 Zero-consumer interop services and dead boot paths are removed per decision D5
- [ ] #3 Main-thread helper spawns before first paint fall from 12 to at most 2, and TldwCli() construction time falls by at least 50% on the boot probe
- [ ] #4 The boot worker census allowlist is updated for any work that moved
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
