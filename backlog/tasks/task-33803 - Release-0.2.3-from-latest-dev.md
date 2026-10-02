---
id: TASK-33803
title: Release 0.2.3 from latest dev
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-02 19:09'
updated_date: '2026-10-02 20:32'
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
- [x] #6 Resend stays off the first-paint import path, with the unchanged UI-ready module budget and affected behavior checks passing.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Preserve latest committed dev and existing main fixes in an isolated release checkout.
2. Update version metadata and changelog to 0.2.3; repair the stale digest input exposed by targeted checks.
3. Defer all three eager Resend imports to their first-use call sites and extend the existing first-paint absence guard; verify the unchanged ADR-097 ratchet and Resend behavior.
4. Run targeted release checks, build distributions, verify installed packages, and obtain bounded review and required CI.
5. Merge verified preparation to dev, publish to TestPyPI, then merge dev to main for production publication.
6. Verify published artifacts and fresh installs, create annotated tag and GitHub release, and record evidence.
ADR required: no
ADR path: backlog/decisions/032-immutable-installed-distribution-assets.md; backlog/decisions/098-low-latency-speculative-duplex-voice-pipeline.md; backlog/decisions/097-boot-budget-ratchets.md
Reason: Existing app-only release policy and first-use import deferral implement accepted ADRs without changing boundaries.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
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
<!-- SECTION:NOTES:END -->

### Resend startup regression repaired

The refreshed preparation's Perf Guard exposed a new Resend eager-import cost:
1034 own modules at UI-ready against the unchanged 1033 ADR-097 ratchet.
A local run reproduced the same count and named module before editing. Moved
all three Resend imports to their actual first-use sites, adjusted the existing
UI test patch at the defining module, and added Resend to the existing
first-paint absence assertion. Nine startup/import checks passed at 1033/1033;
all 75 Resend unit/UI checks passed. Bounded review found no blockers. No
budget or snapshot was raised. Required CI and publication remain pending.

### Second latest-dev refresh

Dev advanced to 185c845fe836bf452e4beaaf8853162ce49b1e8d (PR #2958)
while preparation 835c1959fc906e250e952cfee851e7279983e5ca waited for
its final required job. Its UI and PR lanes passed, and CI boot ratchets
passed (20 passes, 2 skips; 1033/1033 UI-ready modules). The separate
non-required trace-maintenance storage check failed at 2.125 admissions
per tick against ceiling 2, identical to existing TASK-33621.44 and dev
baseline run 37015097693 at ancestor ee1c1e7365c232a184e129bef1dec15afd85b24f.
No ratchet was raised.

Merged the latest dev model-configuration Phase 5 changes without conflicts,
preserving the Resend deferral and release metadata. Added supported explicit
cloud-key checks, local-switcher probes, and shared connection-readiness
evidence to the changelog. This integrates already accepted ADR-012/033/114
functionality and makes no new architectural decision. Affected checks,
required CI, and publication remain pending.

Latest refresh evidence: 119 release metadata, app-only, source-digest and
UI-ready census checks passed, still 1033/1033 modules. Affected shared-evidence,
readiness wording, session settings, local-switcher probing, cloud-key checks
and Resend UI tests produced 410 passes and one identical baseline test-double
failure: test_new_chats_resolve_the_chat_defaults_pair lacks the existing
_console_default_settings_memo field on its SimpleNamespace. The single test
fails identically against archived unmodified dev 185c845; critical source
files were verified byte-for-byte against that commit before the control run.
Bounded integration review found no blockers. The changelog now also records
TASK-33621.44's periodic trace-maintenance performance limitation.

### Reconcile upstream's duplicate Resend startup fix

Dev advanced to ef8fd5d38a512be299af17b1e0d5b367a352a5d6 (#2962),
merging the same three first-use import deferrals already in this preparation.
Resolved the four overlapping code/test-name conflicts using the upstream
implementations, retaining our additional first-paint absence guard and
all release/main/digest changes. Preserved both sets of lessons and trimmed
an inherited extra blank line at the task file's end. No new feature or
changelog scope change. Bounded review found no blockers. All 75 Resend
checks and 119 release/native-boundary/digest/UI-ready checks passed;
the unchanged first-paint budget remains 1033/1033. Required CI is refreshed.

### Fourth latest-dev refresh

Preparation c29e028b387f6c66ba2d766c4a4e05f8cb478996 passed all three
required jobs in run 37068668711 (PR/UI fast lanes and derived contracts).
The protected merge refused because dev advanced during the final check to
ecc0a531c855bc9e80906bff90180fd2045f7159 (#2924, PERF-07). Merged that
data-directory and sensitive-input memoization without conflicts, preserving
release metadata, main fixes and all release guards. Updated the changelog's
configuration-performance summary. Existing approved ADR-126 D2 applies; no
new architectural decision. All 22 affected memo/remote-worker checks and
119 release/app-only/digest/first-paint checks passed; bounded integration
review found no blockers. Required CI is refreshed again.
