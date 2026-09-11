---
id: TASK-32311
title: Prepare and publish 0.2.1 from current dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-11 02:12'
updated_date: '2026-09-11 02:22'
labels:
  - release
  - packaging
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish the newer committed dev changes requested by the release owner while preserving the existing dirty checkout and superseding the pending older 0.2.0 upload.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Release metadata and changelog describe 0.2.1 from dev ed6fd5db0a with main reconciled.
- [ ] #2 Fresh package build, metadata and installed-distribution checks pass on the release source.
- [ ] #3 The release is integrated to main, published to PyPI, and tagged with verified installable artifacts.
- [x] #4 The README current-version statement matches the published release metadata.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/032-immutable-installed-distribution-assets.md
Reason: Release metadata and publication follow the existing installed-distribution contract and branch publishing workflow.
1. Reconcile main into pinned dev ed6fd5db0a in an isolated release worktree.
2. Set package and runtime version to 0.2.1 and summarize changes in CHANGELOG.md; TestPyPI already owns older 0.2.0 artifacts.
3. Build fresh distributions and run targeted metadata, installed-distribution, manifest, and entry-point checks.
4. Integrate release preparation to dev and verify its trusted TestPyPI publication and installation.
5. Integrate the verified source to main, publish through the existing PyPI environment gate, verify PyPI artifact hashes and installed entry points, and create source tag and GitHub release.
6. Record evidence and close this task after publication succeeds.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Prepared the release from dev ed6fd5db0a with main d0aa66ea97 reconciled without file changes. Updated pyproject.toml, runtime version tuple, and CHANGELOG.md to 0.2.1; existing ADR-032 applies. Fresh wheel/sdist build, twine check, manifest validation, and 24 metadata tests pass. Clean Python 3.12 dependency installation, pip check, tldw-cli --help, and tldw-serve --help pass. Installed-distribution regressions and publication remain in progress; this task is not complete.

PR review identified a stale README current-version statement. Corrected it to 0.2.1, updated the release plan, and rebuilt the wheel and sdist. Fresh twine/manifest checks and all 24 metadata tests pass after the documentation correction; runtime source is unchanged.
<!-- SECTION:NOTES:END -->
