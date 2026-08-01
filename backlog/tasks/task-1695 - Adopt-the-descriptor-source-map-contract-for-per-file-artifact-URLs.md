---
id: TASK-1695
title: Adopt the descriptor + source-map contract for per-file artifact URLs
status: Done
assignee:
  - '@claude'
created_date: '2026-08-01 07:02'
updated_date: '2026-08-01 08:13'
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
Implemented the ArtifactSourceMap contract in acquisition.py: a plain type alias Mapping[ArtifactRef, Mapping[str, str]] (not a dataclass -- ArtifactRef is already used as a dict key elsewhere in this module/service.py, and every entry's shape is validated once, at consumption time, by _resolve_file_sources, so a dataclass wrapper would add construction machinery with no invariant left to enforce).

_resolve_file_sources (module-level, replaces _require_single_file) resolves+validates every declared file's URL per descriptor: an explicit sources entry, or -- single-file descriptors only, with no map entry for that file -- the pre-1695 descriptor.source_url fallback. Missing/extra/non-http(s)/query-string/userinfo all raise CatalogError naming the artifact and file path via service._validate_url (reused, not reimplemented), which never quotes URL text in its messages -- the credential-hygiene property falls out of that reuse rather than needing separate enforcement. Both call sites (_aggregate_closure at preflight/provision entry, and _fetch_artifact as direct-call defense-in-depth) now resolve through this shared function; _file_url is now a plain dict lookup against the pre-resolved map.

Consent fingerprint: _closure_fingerprint_with_sources wraps (does not modify) service.closure_fingerprint -- that function is also used by readiness records/manifest verification, which know nothing about source maps. Critical back-compat decision: the wrapper folds in ONLY (ref, path, url) triples the CALLER actually named in the sources argument, not every entry _resolve_file_sources filled in via the single-file fallback. This makes the fingerprint byte-identical to plain closure_fingerprint(root, deps) whenever sources is absent/empty -- which is what let ~15 existing tests across test_provision_install.py/test_provision_serialization.py/test_provision_crash_recovery.py that hand-build AcquisitionConsent via the bare closure_fingerprint() keep passing unmodified in behavior (only their local phase-stub signatures needed a resolved_sources=None parameter added). When a caller DOES pass sources, swapping a URL between preflight() and provision() changes the fingerprint and provision() raises ConsentMismatchError (test_source_map.py::test_source_url_changed_after_consent_raises_consent_mismatch).

Back-compat: sources is an optional keyword-only param on both preflight() and provision(), defaulting to None; a single-file descriptor with a bare source_url and sources omitted (or {}) resolves exactly as before TASK-1695.

Tests: new Tests/Model_Artifacts/test_source_map.py (13 tests) covers the headline multi-file end-to-end provision (and a root+dependency multi-file closure), both back-compat shapes, all five preflight-time validation failures (each proven to precede any network via a client_factory stub that raises if ever constructed), the consent-mismatch-on-URL-swap case and its identical-map sanity counterpart, and a direct fingerprint-equality proof against the plain closure_fingerprint. Extended test_credentials_and_boundaries.py with a secret-hygiene test scoped to this application's own log records (httpx/httpcore's own operational request-URL tracing is excluded -- logging a credential-free URL it is about to fetch is expected, not a leak; the spec's 'never contain tokens/cookies/signed URLs/query strings' rule is about secrets, not URL identity, and untestable request URLs cannot themselves carry the secret shapes we already reject at preflight). Updated two pre-existing tests' docstrings (test_preflight.py, test_provision_fetch.py) whose CatalogError assertions still hold but whose old TASK-596/1301-deferred rationale no longer applies.

Full suite: PYTHONPATH=<worktree> pytest Tests/Model_Artifacts/ Tests/STT/test_boundaries.py -q -> 423 passed (409 baseline + 14 new). TASK-1693 was already closed as superseded in a prior commit; AC #3 verified, not re-done.

Files: tldw_chatbook/Model_Artifacts/acquisition.py, tldw_chatbook/Model_Artifacts/__init__.py, Tests/Model_Artifacts/test_source_map.py (new), Tests/Model_Artifacts/test_credentials_and_boundaries.py, Tests/Model_Artifacts/test_provision_install.py, Tests/Model_Artifacts/test_provision_serialization.py, Tests/Model_Artifacts/test_preflight.py, Tests/Model_Artifacts/test_provision_fetch.py, Tests/Model_Artifacts/test_service.py.
<!-- SECTION:NOTES:END -->
