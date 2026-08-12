---
id: TASK-15674
title: Honor TLDW_CONFIG_PATH when persisting startup defaults
status: In Progress
assignee: []
created_date: '2026-08-12 06:35'
updated_date: '2026-08-12 15:37'
labels:
  - bug
  - config
  - privacy
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent app startup under an isolated config profile from writing normalized default keys into the user's default config file. This was reproduced during generated-video player UAT: the profile remained isolated for reads, but startup appended defaults to the unrelated real config; the exact pre-run file was restored from a validated snapshot.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Starting the real app with `TLDW_CONFIG_PATH` pointing to a scratch profile leaves the default user config byte-for-byte unchanged.
- [ ] #2 Defaults needed by the isolated run are written only to the effective profile path if persistence is required.
- [ ] #3 A regression test uses distinct profile and decoy default configs and proves no cross-profile write.
- [ ] #4 Existing no-override startup persistence behavior remains covered.
- [ ] #5 No config values or credentials are emitted in diagnostics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a real TldwCli startup-to-approved-quit subprocess regression with separate effective and decoy default profiles, deterministic network disablement, and privacy-safe persistence evidence.
2. Mutation-prove the regression against the production effective-config lookup while preserving current production code unchanged.
3. Correct the UAT, TASK-3401.14 notes, TASK-15674 wording, and live-verification lesson to distinguish observed drift from proven causality.
4. Run only touched-file and named config controls, then Ruff, temporary py_compile, privacy checks, and git diff --check.

ADR required: no
ADR path: N/A
Reason: regression-only characterization of the existing effective-config boundary; no new storage, security, runtime, or cross-module decision.
<!-- SECTION:PLAN:END -->
