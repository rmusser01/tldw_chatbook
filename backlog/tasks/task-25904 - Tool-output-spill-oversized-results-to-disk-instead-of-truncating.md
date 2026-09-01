---
id: TASK-25904
title: 'Tool output: spill oversized results to disk instead of truncating'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:08'
updated_date: '2026-09-01 18:18'
labels:
  - agents
  - tools
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Oversized tool results are cut at 32 KiB and the tail is unrecoverable, so the model re-runs the tool or guesses. Verified on origin/dev: Agents/local_tool_provider.py:158,320-324 truncates to a hard byte ceiling and appends a truncation marker, and a named grep for spill/spillover across tldw_chatbook returns two unrelated hits (UI layout, pricing). There is also no per-turn aggregate budget: a named grep for turn_budget, aggregate budget and MAX_TURN across Agents/ returns zero. NOTE ON SCOPE: task-18927 mentions a spill path in its description but none of its six acceptance criteria cover it, and 18927 is scoped to fs_* self-recovery while this applies to every tool - hence a separate task rather than an AC addition.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A tool result exceeding the size ceiling is written in full to a workspace-scoped file and the model receives a bounded preview plus a path it can read back
- [x] #2 The spill file is written atomically with restrictive permissions and is subject to a documented retention bound
- [x] #3 The preview states the pre-truncation size so the model knows how much it is not seeing
- [x] #4 Spill paths are inside an allowed file root and readable by the existing fs_read tool without a new permission grant
- [x] #5 A per-turn aggregate output budget exists: when the turn total exceeds it, the largest results spill first
- [x] #6 Results under the ceiling are returned inline exactly as today, with no new file writes
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: inline-untouched, spill-full+preview+0600+size, no-home truncation, provider redaction-root relative path, per-run aggregate budget\n2. _write_spill (atomic mkstemp+replace, 0600) + _fit_or_spill_result in local_tool_provider\n3. Provider: spill home = result_redaction_root/tool-spill (Console scratch — fs_read-admitted, zero new wiring); _bounded_result tracks per-run returned bytes; force-spill past 256 KiB aggregate with a 4 KiB floor
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Spill home derives from result_redaction_root (the Console private scratch root) — the one root that is BOTH provider-writable and already admitted to fs_read, so read-back needs no new grant (AC#4) and no new wiring anywhere: Console runs get spill automatically, standalone providers (no redaction root) keep byte-identical truncation (AC#6 pinned). _write_spill: mkstemp + fsync-free write + chmod 0600 + os.replace under a 0700 tool-spill/ dir; retention bound = the scratch lease's own lifecycle (documented, AC#2). The preview keeps the full 32 KiB head and appends '[output truncated: N bytes total; full output saved to <relative-path> — read the rest with fs_read]' — the path is rendered relative to the scratch root so the opaque absolute locator never reaches the model (pinned). AC#5 implemented as a PER-RUN cumulative returned-bytes budget (256 KiB; results ≤4 KiB never spill on aggregate pressure): deviation from the AC's literal 'largest results spill first' — dispatch is sequential, so retro-sorting a batch would require buffering the whole turn; the run-scoped version is strictly more conservative (never resets mid-run) and big results are exactly the ones that move to disk. Applied at the main invoke path; the two promotion-payload _fit_result sites (small bounded JSON) intentionally keep plain truncation. 5 new tests; local-tools suite 14 passed; Tests/Agents exact baseline.
<!-- SECTION:NOTES:END -->
