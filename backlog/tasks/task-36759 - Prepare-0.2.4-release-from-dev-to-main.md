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
updated_date: 2026-10-25 17:59
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prepare the owner-confirmed 0.2.4 release from committed dev for review and promotion to main, preserving local work and recording verification blockers before publication.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Application, runtime, companion source, README, changelog, and current release documentation agree on 0.2.4 and summarize committed changes since 0.2.3.
- [x] #2 Release metadata, app-only, build, artifact, and installed-distribution gates are run; the owner-requested full test suite is attempted and all results and blockers are recorded accurately.
- [x] #3 Reviewable release-preparation and main-promotion draft PRs identify the source and remaining release gates; no production publication or merge is performed during preparation.
- [x] #4 Built wheel and sdist contain served mobile, authentication and QR runtime assets and their vendored notice; the release checker rejects missing web assets and both installed command help entry points succeed.
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
Prepared 0.2.4 from committed dev 1850d2599c58af9a0c1a8721e6c0a31e431d92fa, preserving the dirty primary checkout and all main commits. Synchronized root/runtime/tuple/companion source versions, README, changelog and current release documentation. The changelog covers the committed release delta and the schema 75-to-84 upgrade. A fresh install exposed eleven missing mobile/auth/QR web assets despite the old manifest gate passing; explicit package-data and both manifest contracts now include them and the exact vendored notice. Added real built-artifact coverage and wheel/sdist missing-mobile-page controls. ADR-032 and ADR-098 apply; no new ADR or dependency/runtime boundary change.

Verification: baseline/updated metadata and app-only checks each passed 42 tests; final affected packaging/entry-point/notice checks passed 61 tests; the complete corrected installed-distribution gate passed all 231 tests in 551.06 seconds. The committed-source build at 465279e39e passed Twine and the expanded manifest checker. A fresh dependency-resolved Python 3.12 install passed pip check, reports 0.2.4 from site-packages, and both commands return help with exit 0 from separate empty private profiles. E9/F821, formatting, whitespace and Backlog guards pass; default lint has the same 13 pre-existing findings on unchanged dev, with no new findings.

Owner-requested full suite stopped during collection: 11 errors and 18 collection skips, exit 2. Unchanged archived dev reproduces the same files and exceptions: nine stale/missing project-module/helper imports, plus two missing optional fido2 imports. No full-suite pass is claimed. The initial installed gate was superseded after discovering the asset gap; final results above are from the corrected candidate. Detailed commands, artifact SHA-256 values and local receipts are in Docs/superpowers/plans/2026-10-25-release-0.2.4.md. Recorded the manifest-enumeration gap and complete private-child isolation lesson in backlog/docs/lessons-testing-evidence.md.

Draft preparation PR #3245 and dev-to-main promotion PR #3246 are attached. No merge, tag, release or publication occurred. Keep the task In Progress: inherited full-suite collection blockers, CI/Qodo current-head review and TestPyPI verification remain before explicit owner approval of main promotion and the protected PyPI environment.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
