---
id: TASK-33803
title: Release 0.2.3 from latest dev
status: Done
assignee:
  - '@codex'
created_date: '2026-10-02 19:09'
updated_date: '2026-10-03 01:26'
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
- [x] #2 Latest dev and existing main fixes are included, with required release checks and fresh installed-package verification passing.
- [x] #3 The verified source is merged to main, published to PyPI, and identified by an annotated tag and GitHub release.
- [x] #4 Final publication evidence and any material verification limitations are recorded.
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
Published app-only 0.2.3 from latest committed dev ecc0a531c855bc9e80906bff90180fd2045f7159, retaining main's evaluation case-sensitivity fix. Updated app/runtime/native source versions, README, changelog and release guidance; repaired stale Console CSS digest inputs and retained upstream's three first-use Resend imports with a stronger first-paint absence guard. Experimental duplex remains unqualified; no native companion was published.

Release source/main cut: `f3aeb32fb3d230c0774c7fc349729d9f75c96366`, whose tree equals reviewed preparation `6ecebc13b2a553845453eece57884c372ef802a3`. Annotated `v0.2.3` points to that source. [PyPI](https://pypi.org/project/tldw-chatbook/0.2.3/) and [GitHub release](https://github.com/rmusser01/tldw_chatbook/releases/tag/v0.2.3) are published. Post-publication bookkeeping is on `codex/release-0.2.3-evidence`, preserving the immutable main/tag cut.

Validation: [required CI 37071829315](https://github.com/rmusser01/tldw_chatbook/actions/runs/37071829315) passed PR/UI/derived gates (933 UI; 1183 + 124 PR passes). [TestPyPI 37075969288](https://github.com/rmusser01/tldw_chatbook/actions/runs/37075969288) and [production 37081603061](https://github.com/rmusser01/tldw_chatbook/actions/runs/37081603061) each passed 24 metadata and all 184 installed-distribution regressions. Both workflow artifact hashes match their respective registries. Fresh registry installs passed `pip check`, version/tuple, installed-module origin, app-only gate with the dev flag set, and `tldw-cli --help` / `tldw-serve --help`.

Targeted integration evidence: 22 memo/worker, 119 release/digest/first-paint, 75 Resend and 410 readiness/session/probe passes. UI-ready remains 1033/1033 modules with no budget raised; bounded reviews found no blockers. Changed runtime/test files passed E9/F821 lint and the changed test range passed formatting. The full local suite was not requested or run.

Material limitations: ten broader message-action failures and one session-settings test-double failure were reproduced with matching assertions on unchanged dev controls. Non-required Perf Guard run 37071829232's only failure was the existing TASK-33621.44 trace-maintenance rate (2.125 admissions/tick, ceiling 2), also present in baseline run 37015097693; latency and boot ratchets passed. The changelog records this and TASK-33662's relaunch-recovery limitation. An unchanged upstream test-file wrap also fails whole-file formatting; release changes introduce no formatting diff there.

Existing ADR-032/098/097 govern artifact integrity, app-only scope and boot ratchets; integrated memoization follows approved ADR-126 D2. No new ADR or deviation in release scope. Detailed logs and control evidence are retained at `/private/tmp/tldw-release-0.2.3-evidence`.

Published SHA-256 hashes (each index was checked against its own workflow's artifacts):

| Index | Artifact | SHA-256 |
| --- | --- | --- |
| testpypi | `tldw_chatbook-0.2.3-py3-none-any.whl` | `5a6c3bb76c184157af5d1654b2e7d4805294e2da33bc49effa5bba3f668de7f5` |
| testpypi | `tldw_chatbook-0.2.3.tar.gz` | `289883ac929b27e3a6fa6273c01fe657c2bf5c672c1782903790250350fea34a` |
| pypi | `tldw_chatbook-0.2.3-py3-none-any.whl` | `525f3fafec01c2f84e4113d0a6b3fd1f328fa5eb7880049aae0c4af6084b0c5c` |
| pypi | `tldw_chatbook-0.2.3.tar.gz` | `734c4c3a0153ad6a59bc2f7981948d54253fde92e09a1283e880e2ad636cbb20` |
<!-- SECTION:NOTES:END -->
