---
id: TASK-32567
title: Publish version-only 0.2.2 release
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-14 14:16'
updated_date: '2026-09-14 14:20'
labels:
  - release
  - packaging
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish the owner-confirmed version-only successor to 0.2.1 through the existing app-only release process, preserving all existing user work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Application and companion source versions, README, and changelog agree on 0.2.2; application behavior and dependency constraints are unchanged.
- [ ] #2 Fresh build, app-only boundary, metadata, installed-distribution, and clean installation checks pass for the release source.
- [ ] #3 The verified source is on main, version 0.2.2 is published to PyPI, and its annotated tag and GitHub release identify that source.
- [ ] #4 Publication evidence and the inherited broader CI limitation are recorded accurately.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR
ADR path: backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md; backlog/decisions/032-immutable-installed-distribution-assets.md
Reason: This version-only release continues the approved app-only and immutable-package contracts without changing architecture or application behavior.
1. Prepare isolated origin/dev source 4631b60f8dd9623fc55bf16f4a37e29fcb1240c7 and confirm 0.2.2 is absent on both registries.
2. Synchronize package, runtime tuple, and companion source versions; update README, changelog, and current release documentation. Make strict-pin test fixtures derive the current version if required.
3. Run focused metadata and voice-version tests, build and inspect wheel/sdist, run installed-distribution checks, and obtain independent release review.
4. Merge the verified preparation to dev via required PR checks, publish to TestPyPI, and verify registry artifact hashes and a fresh installation.
5. Fast-forward main to the exact verified source, complete production gates and existing environment approval, then verify PyPI hashes and a fresh installation.
6. Create the annotated v0.2.2 tag and GitHub release at that exact source; record evidence and close this task. Preserve the known broader CI collection limitation without claiming a passing full suite.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prepared version-only 0.2.2 from 4631b60f8dd9623fc55bf16f4a37e29fcb1240c7. Root/runtime/tuple/native source versions, README, changelog, and current app-only docs agree. Dependency constraints and application behavior are unchanged. The initial focused run reproduced one stale strict-pin fixture failure with 41 passing tests; deriving its positive pin from TOML metadata preserved rejection of non-exact pins. Final focused verification: 147 passed. Fresh wheel/sdist build, twine checks, manifest checks, E9/F821 lint, test-file formatting, and whitespace checks pass. Independent review found no issues. Full targeted installed-distribution regression is running; publication remains pending. Existing ADR-032 and ADR-098 apply, with no new ADR required.
<!-- SECTION:NOTES:END -->
