---
id: TASK-32311
title: Prepare and publish 0.2.1 from current dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-11 02:12'
updated_date: '2026-09-11 02:38'
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
- [x] #2 Fresh package build, metadata and installed-distribution checks pass on the release source.
- [ ] #3 The release is integrated to main, published to PyPI, and tagged with verified installable artifacts.
- [x] #4 The README current-version statement matches the published release metadata.
- [x] #5 Library release verification uses current focus, empty-reader allocation, and destructive-action spacing contracts, with failing legacy assertions repaired without application behavior changes.
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
Prepared 0.2.1 from dev ed6fd5db0a, reconciled main d0aa66ea97 without file differences, and synchronized the subsequently merged dev a0b8f96416 Library reader fixes. Release metadata, README, changelog, and plan agree. ADR-032 governs the unchanged publishing and installed-asset contract.

Verification: fresh wheel/sdist build, twine check and manifest validation pass. Metadata: 24 passed. Installed-distribution regressions on the initial release source: 175 passed, 24 metadata cases deselected. Synced source rebuilt and passed metadata plus a clean Python 3.12 installation, pip check, and both CLI help commands. GitHub will repeat packaging on the final publication commit.

Expanded Library verification: 197 passed and five stale assertions failed; all five reproduced on pristine ed6fd5db0a. Repaired only tests: inspect underlined label content without the solid focus border; expect the 32-cell canvas inside a 36-cell pane; compare grid slot origins with exact neutral-zero and danger-two margins and require complete action labels. Five corrected cases and four shared focus-helper callers pass. Independent review approved these repairs. No new application behavior was introduced.

Publication and the source tag remain pending. Full test suite was not requested or run locally.

Final repair verification after focused formatting: 9 passed in 30.26s. Ruff E9/F821 checks pass for the touched Python modules; Ruff range-format checks pass for all changed functions and the version declarations. Existing unrelated whole-file formatting drift was not reformatted. Added the baseline-backed geometry/helper lesson. The width failure was a canvas-versus-pane mismatch (32 versus 36), not ignored custom widths.
<!-- SECTION:NOTES:END -->
