---
id: TASK-1712
title: Fold single-file source_url into the acquisition consent fingerprint
status: To Do
assignee: []
created_date: '2026-08-01 08:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Residual from TASK-1695's review: the consent fingerprint covers caller-supplied (ref, path, url) triples but not the single-file source_url fallback, so a descriptor whose source_url changes between preflight() and provision() — with no explicit source map — is not consent-gated. Inherited from closure_fingerprint(), which only ever hashed the ArtifactRef set; not introduced by 1695. Reachable if an ArtifactCatalog implementation is dynamic (mirror rotation, CDN rebalancing), since the protocol guarantees no immutability. Closing it means updating ~15 tests that hand-build AcquisitionConsent from the bare function.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A single-file descriptor whose source_url changes between grant() and provision() raises ConsentMismatchError
- [ ] #2 Existing hand-built-consent tests are updated for the new contract rather than weakened
<!-- AC:END -->
