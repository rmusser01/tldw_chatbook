---
id: TASK-33803
title: Release 0.2.3 from latest dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 19:09'
updated_date: '2026-10-02 19:12'
labels:
  - release
  - packaging
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Publish current committed dev changes as a verified new release on main and PyPI, preserving main fixes and existing user work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Release metadata and changelog agree on 0.2.3 and summarize committed changes since 0.2.2.
- [ ] #2 Latest dev and existing main fixes are included, with required release checks and fresh installed-package verification passing.
- [ ] #3 The verified source is merged to main, published to PyPI, and identified by an annotated tag and GitHub release.
- [ ] #4 Final publication evidence and any material verification limitations are recorded.
- [x] #5 Release source-digest inputs refer to the current Console stylesheets, with existing regression checks passing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve latest committed dev and existing main fixes in an isolated release checkout.
2. Update version metadata and changelog to 0.2.3; review release scope and repair the stale digest input exposed by targeted checks.
3. Run targeted release checks, build distributions, and verify installed packages; obtain bounded review and required CI.
4. Merge verified preparation to dev, publish to TestPyPI, then merge dev to main for production publication.
5. Verify published artifacts and fresh installs, create annotated tag and GitHub release, and record evidence.
ADR required: no
ADR path: backlog/decisions/032-immutable-installed-distribution-assets.md; backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md
Reason: This release continues the existing app-only exception without architectural changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

Prepared 0.2.3 from dev f80d3e0090130658918ccd519d85ca700028ce98 and merged
main 64ab237bde673919d47e4f33f3b10cf14ccbf05d without conflicts, retaining the
evaluation case-sensitivity fix. Updated app/runtime/native source versions,
README, changelog, and app-only release documentation. Native qualification
and unavailable duplex voice remain unchanged under ADR-098; installed artifact
verification follows ADR-032. No new ADR is required for this release.

Targeted baseline: 63 passed. Expanded metadata, native boundary, digest, and
evaluation checks exposed three failures caused by the deleted generated
Console stylesheet. Replaced its stale digest entry with the current two
Console source sheets; the same 268 checks then passed. Builds, installed-package
checks, required CI, index publication, and final provenance are pending.

### Refresh after dev advanced

Preparation e6bb9c06ad8f1b55c4ceefd8dd6bb8cec7b33b0b passed the committed build,
268 focused tests, 184 installed-distribution tests, and a fresh dependency-resolved
wheel install with both CLI help commands. Required CI run 37052989900 passed:
922 UI tests; PR phases of 1183 and 123 passes; and derived-artifact contracts.
The protected merge correctly refused because dev advanced to
bb865f5cfeae4c9d8c068f588ee0d85ff28a0e13 while these checks ran.

Merged the additional Console Resend commits without conflicts, retained both
sets of testing lessons, and added Resend plus the documented, pre-existing
TASK-33662 relaunch-recovery limitation to the changelog. Refreshed affected
checks, installed boot probes, required CI, and publication remain pending.
