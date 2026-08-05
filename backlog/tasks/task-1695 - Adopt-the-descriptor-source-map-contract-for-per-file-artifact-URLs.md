---
id: TASK-1695
title: Adopt the descriptor + source-map contract for per-file artifact URLs
status: Done
assignee:
  - '@claude'
created_date: '2026-08-01 07:02'
updated_date: '2026-08-01 08:21'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reconciliation item 2: the parallel TASK-595 branch passes an explicit credential-free source map alongside the exact descriptor, so per-file URLs never need to enter the frozen descriptor schema. Adopt that contract in acquisition.py, replacing the current hard CatalogError refusal of multi-file artifacts. Supersedes TASK-1693 (descriptor schema v2), which should be closed as unnecessary if this lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Multi-file artifacts provision successfully with per-file URLs supplied by the caller's source map
- [x] #2 No per-file url field is added to the frozen ArtifactFile schema
- [x] #3 TASK-1693 is closed as superseded
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Define ArtifactSourceMap type alias (Mapping[ArtifactRef, Mapping[str, str]]) in acquisition.py.
2. Replace _require_single_file with _resolve_file_sources: resolves+validates every declared file's URL (explicit map entry, or single-file source_url fallback), reusing service._validate_url so errors never quote URL text.
3. Thread sources through preflight()/provision()/_aggregate_closure(); resolve the full closure's not-yet-installed source map once and reuse it for fetch/pre-verify/_file_url (now a lookup).
4. Extend the consent fingerprint via a wrapper (_closure_fingerprint_with_sources) that folds in only CALLER-SUPPLIED (ref, path, url) triples, so it is byte-identical to the plain closure_fingerprint when sources is absent -- preserving back-compat with existing hand-built AcquisitionConsent call sites.
5. Write TDD coverage: multi-file end-to-end over fixture server, single-file back-compat, missing/extra/scheme/query/userinfo validation-at-preflight, consent-mismatch on URL swap, and a secret-hygiene extension.
6. Run full Tests/Model_Artifacts + Tests/STT/test_boundaries.py suite green before commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ArtifactSourceMap threaded through preflight()/provision(); per-file URLs resolve from the caller's credential-free map with a single-file source_url fallback for back-compat. Resolution is total, so _require_single_file is gone from both call sites and multi-file artifacts provision end-to-end (13 new tests incl. root+dependency composition). Validation runs inside _aggregate_closure — before consent and before any network on both the preflight and provision paths — reusing _validate_url to reject non-http(s), query strings, fragments, and userinfo without ever interpolating the URL into the error. Consent fingerprint extended over caller-supplied (ref, path, url) triples so a swapped origin invalidates stale consent. No ArtifactFile.url, no schema bump. 423 tests green.
<!-- SECTION:NOTES:END -->
