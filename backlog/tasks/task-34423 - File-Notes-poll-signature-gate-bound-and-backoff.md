---
id: TASK-34423
title: File Notes poll signature gate bound and backoff
status: Done
created_date: 2026-10-07 02:41
updated_date: 2026-10-07 12:02
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 5 / F10: the 1.5s workspace poll does an unbounded full-tree walk with two lstats per file plus a full replica read per tick and session changes accumulate unboundedly
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Unchanged vault performs zero replica reads per tick,Reconcile fires exactly once per real change,Session change list capped at 500,Poll backoff to 6s when idle,Existing sync tests pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 11 (T11)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
File Notes poll hygiene: reconcile gate on a 5-field per-root signature (path, device, inode, size, mtime_ns) built from the walk that must run anyway (zero extra stats) — clean passes only arm the gate (warning-bearing results null the cache, degraded passes self-heal next tick; regression-tested for raise->recover with vault unchanged); ctime_ns omission documented as deliberate narrowing (replica stores only size+mtime; watcher's changed_ns has no counterpart — branch verified against replica schema). Walk bounds mirror the watcher (1000 files / 10000 entries / depth 32, deterministic truncation prefix, invisible tail never tombstoned, once-per-episode logging). Session changes bounded: append-only below the 500-record cap (preserving the append-only ledger contract used by commit-authority drift detection — controller-approved deviation from unconditional append-merge), same-(action,path) compaction-to-newest only in overflow, oldest-drop beyond; per-tick coalesce skipped via pending counter. Poll backoff 1.5 s -> 6.0 s after 4 quiet ticks with reset on change/error/offline. Evidence: 40-file vault x 5 idle ticks — list_active_files 5 -> 1, per-tick replica read + sort + session coalesce eliminated. 17 new tests + 185 core + 20 activation green; failure list byte-identical to baseline. Files: Notes/file_notes_service.py, Notes/file_notes_session_owner.py, Widgets/Library/library_file_notes_workspace.py, Tests/Notes/test_file_notes_poll_gating.py. Report: .superpowers/sdd/task-11-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
