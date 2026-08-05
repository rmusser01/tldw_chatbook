---
id: TASK-1720
title: Fold single-file source_url into the acquisition consent fingerprint
status: Done
assignee:
  - '@claude'
created_date: '2026-08-01 08:21'
updated_date: '2026-08-01 09:41'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual from TASK-1695's review: the consent fingerprint covers caller-supplied (ref, path, url) triples but not the single-file source_url fallback, so a descriptor whose source_url changes between preflight() and provision() — with no explicit source map — is not consent-gated. Inherited from closure_fingerprint(), which only ever hashed the ArtifactRef set; not introduced by 1695. Reachable if an ArtifactCatalog implementation is dynamic (mirror rotation, CDN rebalancing), since the protocol guarantees no immutability. Closing it means updating ~15 tests that hand-build AcquisitionConsent from the bare function.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A single-file descriptor whose source_url changes between grant() and provision() raises ConsentMismatchError
- [x] #2 Existing hand-built-consent tests are updated for the new contract rather than weakened
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Fold every resolved (ref, path, url) triple into _closure_fingerprint_with_sources (drop the caller-supplied-only filter). 2. Update the one call site in _aggregate_closure. 3. Update ~15 tests that hand-build AcquisitionConsent from the bare closure_fingerprint() to go through PreflightReport.grant() (via a new network-free grant_consent test helper) instead. 4. Add a regression test: single-file descriptor whose source_url changes between grant() and provision() raises ConsentMismatchError.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed the source_map filter from _closure_fingerprint_with_sources (acquisition.py) so it folds in every entry of the already-validated resolved_sources map, fallback-resolved single-file source_url entries included -- not just caller-supplied ArtifactSourceMap entries. This is no longer back-compatible with the bare closure_fingerprint(root, deps) for any closure with a not-yet-installed entry, so every test that hand-built an AcquisitionConsent from that bare function had to switch to going through a real PreflightReport.grant() (13 direct call sites across test_provision_install.py/test_provision_serialization.py/test_provision_fetch.py/test_provision_crash_recovery.py/provision_processes.py, plus test_source_map.py's own now-incorrect back-compat assertion). Added a shared, network-free grant_consent(svc, root, catalog, sources) helper (Tests/Model_Artifacts/acquisition_test_helpers.py) that grants through ArtifactAcquisitionService._aggregate_closure(...).grant() -- the same pure aggregation preflight() wraps with a gating probe -- so none of those call sites needed to start touching the network (several use unreachable example.test placeholder URLs by design). Replaced test_source_map.py's test_source_map_fingerprint_matches_plain_closure_fingerprint_when_absent (which asserted the OLD, buggy back-compat behavior) with test_single_file_fallback_fingerprint_differs_from_plain_closure_fingerprint and the literal AC #1 regression test, test_single_file_source_url_changed_after_consent_raises_consent_mismatch. Verified both new tests fail against the pre-fix code (git stash) for the expected reason before restoring the fix. Full suite green: Tests/Model_Artifacts/ + Tests/STT/test_boundaries.py, 432 passed.
<!-- SECTION:NOTES:END -->
