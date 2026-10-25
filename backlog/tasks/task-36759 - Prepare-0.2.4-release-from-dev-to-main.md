---
id: TASK-36759
title: Prepare 0.2.4 release from dev to main
status: In Progress
created_date: 2026-10-25 17:36
assignee:
- '@codex'
labels:
- release
- packaging
updated_date: 2026-10-25 17:45
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the owner-confirmed 0.2.4 release from committed dev for review and promotion to main, preserving local work and recording verification blockers before publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Application, runtime, companion source, README, changelog, and current release documentation agree on 0.2.4 and summarize committed changes since 0.2.3.
- [ ] #2 Release metadata, app-only, build, artifact, and installed-distribution gates are run; the owner-requested full test suite is attempted and all results and blockers are recorded accurately.
- [ ] #3 Reviewable release-preparation and main-promotion draft PRs identify the source and remaining release gates; no production publication or merge is performed during preparation.
- [ ] #4 Built wheel and sdist contain served mobile, authentication and QR runtime assets and their vendored notice; the release checker rejects missing web assets and both installed command help entry points succeed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/032-immutable-installed-distribution-assets.md; backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md
Reason: Continue the existing immutable-package and app-only release contracts; no new runtime, dependency, schema or architectural decision.
1. Freeze origin/dev 1850d2599c58af9a0c1a8721e6c0a31e431d92fa in an isolated release branch and confirm origin/main is an ancestor. Preserve the dirty primary checkout.
2. Update root/runtime/companion versions, README, changelog and current app-only release docs to 0.2.4. Add the source-bound release plan with promotion and publication gates.
3. Run focused packaging checks, the requested full suite, fresh builds, manifest/twine checks and installed-distribution regressions. Record failures without changing unrelated features or weakening gates.
4. Self-review the bounded preparation diff. Push the release branch and open a draft preparation PR to dev plus a draft dev-to-main promotion PR. Document pending TestPyPI, CI/review and final main-merge approval; attach both PRs.
5. The fresh-wheel smoke exposed omitted mobile/auth/QR assets: extend explicit Web_Server package data and the existing artifact contract under ADR-032, add a failing built-artifact regression and a missing-asset mutation check, then rebuild and repeat the affected release gates. No web UI or runtime behavior is changed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Prepared release-only metadata and documentation from dev 1850d2599c58af9a0c1a8721e6c0a31e431d92fa. Root/runtime/tuple/companion source versions, README and current app-only documentation agree on 0.2.4; the changelog summarizes committed changes since 0.2.3 and warns about schema 75 to 84. Existing ADR-032 and ADR-098 apply; no new ADR. Main is already an ancestor, preserving its fixes. Baseline and updated focused release/app-only tests both passed all 42 cases, the standalone app-only checker passed, and E9/F821 lint, ruff format and whitespace checks passed. PyPI and TestPyPI both report latest 0.2.3 with 0.2.4 absent. Owner-requested full-suite and installed-distribution runs are in progress; publication is not authorized by preparation.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
