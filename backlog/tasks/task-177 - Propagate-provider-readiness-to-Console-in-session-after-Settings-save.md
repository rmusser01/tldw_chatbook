---
id: TASK-177
title: Propagate provider readiness to Console in-session after Settings save
status: Done
assignee: []
created_date: '2026-07-12 02:47'
labels:
  - ux
  - console
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11: after saving a ready provider in Settings (toast confirms), returning to Console still shows the blocking 'Add an API key' setup card; only an app restart unlocks it. Readiness reads a stale config snapshot (app_config vs CLI-config seam). New users cannot reach first chat without restarting.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Saving provider settings and navigating to Console shows the ready state without restarting the app
- [x] #2 Setup-card readiness is derived from fresh config (or re-probed on Console entry)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch claude/uat-core-loop-2026-07 (PR #606, commits 6fd4a60f..88c0475b) with focused tests; re-verified live against llama.cpp on a fresh profile (remediation captures in Docs/superpowers/qa/core-loop-uat-2026-07).
<!-- SECTION:NOTES:END -->
