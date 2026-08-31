---
id: TASK-25904
title: 'Tool output: spill oversized results to disk instead of truncating'
status: To Do
assignee: []
created_date: '2026-08-31 15:08'
updated_date: '2026-08-31 15:11'
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
- [ ] #1 A tool result exceeding the size ceiling is written in full to a workspace-scoped file and the model receives a bounded preview plus a path it can read back
- [ ] #2 The spill file is written atomically with restrictive permissions and is subject to a documented retention bound
- [ ] #3 The preview states the pre-truncation size so the model knows how much it is not seeing
- [ ] #4 Spill paths are inside an allowed file root and readable by the existing fs_read tool without a new permission grant
- [ ] #5 A per-turn aggregate output budget exists: when the turn total exceeds it, the largest results spill first
- [ ] #6 Results under the ceiling are returned inline exactly as today, with no new file writes
<!-- AC:END -->
