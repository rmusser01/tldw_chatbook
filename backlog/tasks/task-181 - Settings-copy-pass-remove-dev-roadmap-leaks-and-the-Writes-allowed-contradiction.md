---
id: TASK-181
title: >-
  Settings copy pass: remove dev-roadmap leaks and the Writes-allowed
  contradiction
status: Done
assignee: []
created_date: '2026-07-12 02:47'
labels:
  - ux
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Core-loop UAT 2026-07-11: the Scope Inspector shows 'Writes allowed: Yes' directly above 'Mutation replay: disabled - Writes remain blocked until explicit review, conflict, rollback, and audit gates are implemented.' Overview leads with a blocked Manual Sync v2 block and internal ownership boilerplate (Settings owns X, MCP owns Y). Reads as an architecture doc and undermines trust on first run.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No panel simultaneously claims writes are allowed and blocked
- [x] #2 Internal roadmap/ownership prose is removed or moved behind an expert view
- [x] #3 Overview leads with user-relevant readiness rather than a blocked power feature
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Fixed on branch claude/uat-core-loop-2026-07 (PR #606, commits 6fd4a60f..88c0475b) with focused tests; re-verified live against llama.cpp on a fresh profile (remediation captures in Docs/superpowers/qa/core-loop-uat-2026-07).
<!-- SECTION:NOTES:END -->
