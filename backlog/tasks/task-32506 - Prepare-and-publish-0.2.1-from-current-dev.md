---
id: TASK-32506
title: Prepare and publish 0.2.1 from current dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-11 02:12'
updated_date: '2026-09-13 23:50'
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
- [x] #1 Release metadata and changelog describe 0.2.1 from current dev392ce191fd with main reconciled.
- [ ] #2 Fresh package build, metadata and installed-distribution checks pass on the release source.
- [ ] #3 The release is integrated to main, published to PyPI, and tagged with verified installable artifacts.
- [x] #4 The README current-version statement matches the published release metadata.
- [x] #5 Library release verification uses current focus, empty-reader allocation, and destructive-action spacing contracts, with failing legacy assertions repaired without application behavior changes.
- [x] #6 All four runtime Canvas guide topics are required in both distributions; only the three fixed guide Markdown files are exempt from the development-document exclusion.
- [x] #7 Application and native voice companion versions remain synchronized; release scope explicitly accounts for the unavailable physical voice qualification.
- [ ] #8 App-only distributions have no native AEC dependency; ordinary speech-recording dependencies remain and installed experimental duplex qualification stays false.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes (amend existing ADR-098)
ADR path: backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md
Packaging contract: backlog/decisions/032-immutable-installed-distribution-assets.md
Reason: The approved app-only exception removes the unavailable native dependency while preserving all duplex qualification gates. Follow the updated release plan for explicit app-only validation, targeted tests and publication.
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

Final local candidate f139056ef3 passed a fresh wheel/sdist build, twine and manifest checks, 178 installed-distribution regressions in 689.88 seconds, clean Python 3.12 pip check, and both CLI help smoke tests. The required GitHub Derived Artifacts run 34705705436 passed. PR2588 merged to dev as eab11880110d4997e256852ca915eaf00334e523; its only additional change versus the tested candidate is branch-protection baseline documentation. TestPyPI publishing run 34706422953 is building that exact merge commit. Runtime Canvas guide enforcement follows existing ADR-032 and ADR-149; no new architecture was introduced.

Later Voice AEC CI run 34705705442 built all five platform wheel families successfully but its assembly job rejected the stale 0.2.0 companion version and speech_recording pin. Cancelled pending TestPyPI run34706422953 before upload. Docs/Development/TTS/voice-aec-release.md also requires companion publication before the application; the checked-in rollout manifest marks all five platforms unqualified and Artifacts/voice_qualification reports are absent. Version correction is proceeding; app-only versus qualification release scope awaits owner input.

Voice version metadata is corrected locally to0.2.1 and the companion-pin test now derives the current app version rather than hard-coding0.2.0. Version-lock command and19 focused version/license tests pass; E9/F821 lint passes. Existing whole-file I001 and SIM117 findings are outside the changed function. TestPyPI run34706422953 is confirmed cancelled with no upload. Main remains d0aa66ea97. Owner scope decision is pending; no physical hardware or qualification run was attempted.

Owner approved app-only0.2.1 on2026-09-13, keeping experimental duplex disabled and removing its unavailable dependency. Synced current committed dev392ce191fd before implementation. Existing ADR-098 will be amended for the app-only release exception; companion publishing remains subject to its unchanged qualification gates.

Implemented the owner-approved app-only exception under the ADR-098 amendment. Removed the native companion from speech_recording while retaining ordinary recording dependencies; synchronized companion source version0.2.1. Explicit app-only validation rejects companion requirements across all groups (including direct URLs/case variants) and requires entirely unqualified packaged rollout. Default native release checks and runtime acoustic/qualification code remain unchanged. Fresh build/twine/manifest checks pass;193 focused tests pass. Independent review is complete. Local full distribution run aborted on ENOSPC, not a code assertion; removed only task-owned temporary installations and retained logs. The changed installed-runtime probe is running locally; complete installed-distribution regression remains a mandatory publishing-runner gate.
<!-- SECTION:NOTES:END -->
