---
id: TASK-32506
title: Prepare and publish 0.2.1 from current dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-11 02:12'
updated_date: '2026-09-12 16:35'
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
- [x] #1 Release metadata and changelog describe 0.2.1 from current dev 71313cccd8 with main reconciled.
- [ ] #2 Fresh package build, metadata and installed-distribution checks pass on the release source.
- [ ] #3 The release is integrated to main, published to PyPI, and tagged with verified installable artifacts.
- [x] #4 The README current-version statement matches the published release metadata.
- [x] #5 Library release verification uses current focus, empty-reader allocation, and destructive-action spacing contracts, with failing legacy assertions repaired without application behavior changes.
- [ ] #6 All four runtime Canvas guide topics are required in both distributions; only the three fixed guide Markdown files are exempt from the development-document exclusion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/032-immutable-installed-distribution-assets.md
Reason: Release metadata and publication follow the existing installed-distribution contract and branch publishing workflow.
1. Reconcile main and current dev 71313cccd8 in the isolated release worktree.
2. Set package and runtime version to 0.2.1 and summarize changes in CHANGELOG.md; TestPyPI already owns older 0.2.0 artifacts.
3. Require all four Canvas guide resources and exempt only their fixed Markdown files from development-document exclusion; verify missing-resource rejection and installed guide loading. Build fresh distributions and run targeted metadata, installed-distribution, manifest, and entry-point checks.
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

Synced dev again to a5486ad4d4 (PR2585 Library shell fixes) to satisfy its strict up-to-date rule. Targeted shell and repaired-assertion verification: 28 passed, 1 existing xfailed (TASK-32302 conversation entry focus), no unexpected failures. Fresh synchronized build, twine/manifest gates, and 24 metadata tests pass. This expected failure is retained rather than changing product behavior in the release task.

Resumed 2026-09-12. Previous candidate 70658fb6c7 passed all GitHub checks, but dev advanced to 71313cccd8. Reconciled current dev, retaining its removal of the eager Textual compatibility-shim call while keeping release version 0.2.1. Preserved both sides of the appended testing-lessons conflict. Renumbered this release task from 32311 to 32506 because a different task with 32311 merged during the interruption; scanned all fetched remotes and current workspace tasks before choosing the new ID. Fresh release verification is in progress.

Fresh September 12 build and twine validation passed; manifest validation exposed the new Canvas guides being rejected as development Markdown. Release verification is pending a narrow checker correction under ADR-032 and the existing ADR-149 guide contract.

September 12 targeted verification: 34 metadata/Library checks and 80 Canvas/metadata/manifest checks passed (overlapping metadata coverage). New checker regressions first reproduced 9 failures before the narrow fix. Independent review found no blockers, conditional on final package checks. Fresh full targeted installed-distribution run is in progress.
<!-- SECTION:NOTES:END -->
