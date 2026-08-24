---
id: TASK-21251
title: Keep Windows startup alive when Actor Pack cleanup is unsupported
status: In Progress
assignee: []
created_date: '2026-08-24 00:09'
updated_date: '2026-08-24 00:09'
labels:
  - bug
  - actor-packs
  - windows
dependencies: []
references:
  - >-
    backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent Actor Pack startup housekeeping from terminating the application on platforms where private staging authority cannot be verified, while preserving the existing fail-closed cleanup boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Constructing the Actor Pack import service does not raise when private staging is usable but platform verification is unavailable.
- [ ] #2 Startup cleanup does not enumerate, modify, or delete staged candidates unless private staging authority is verified.
- [ ] #3 Authenticated startup cleanup behavior on supported platforms remains unchanged.
- [ ] #4 Automated regression coverage exercises the unsupported-platform startup path and the supported cleanup path.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Capture the approved fail-closed startup behavior and link the governing Actor Pack ADR.
2. Add a regression test proving unsupported-platform cleanup is a non-destructive no-op during service construction.
3. Implement the smallest guard that skips startup sweeping when platform verification is unavailable while preserving errors for unusable staging.
4. Run focused Actor Pack tests, relevant startup coverage, and static checks.

ADR required: no
ADR path: backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md
Reason: This routine boot bug fix preserves ADR-074’s existing fail-closed authority boundary and introduces no new architecture or security policy.
<!-- SECTION:PLAN:END -->
