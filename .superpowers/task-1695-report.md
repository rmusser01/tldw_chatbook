# TASK-1695 report: descriptor + credential-free source-map contract for per-file artifact URLs

## Type chosen: `ArtifactSourceMap = Mapping[ArtifactRef, Mapping[str, str]]`

A plain nested `Mapping` type alias, not a dataclass. Justification:

- `ArtifactRef` is already used as a dict key elsewhere in this module
  (`resolve_catalog_closure`'s internal `resolved` dict) and in `service.py`,
  so this is the lowest-ceremony shape a caller can hand over directly as a
  dict literal (`{ref: {"a.bin": url_a}}`).
- Every entry's completeness and credential-free shape is validated exactly
  once, at consumption time, by the new `_resolve_file_sources` — so a
  dataclass wrapper would add construction/equality machinery with no
  invariant left for it to usefully enforce at construction time.
- `Mapping` (not `dict`) in the type signature documents read-only-by-contract
  without needing a runtime-enforced frozen wrapper.

## Implementation shape

- `_resolve_file_sources(descriptor, entries)` (module-level, replaces
  `_require_single_file`): resolves and validates every declared file's URL —
  an explicit `entries` entry, or (single-file descriptors only, no entry
  supplied) the pre-1695 `descriptor.source_url` fallback. Missing entries,
  extra entries (naming an undeclared path), non-`http(s)` schemes, query
  strings, and userinfo all raise `CatalogError`. Validation is delegated to
  `service._validate_url` (reused, not reimplemented) — its error messages
  never quote the URL text, so the credential-hygiene property (never leak a
  credential-shaped value into an error) falls out of reuse rather than
  needing separate enforcement.
- `_aggregate_closure` (shared by `preflight()`/`provision()`) now threads an
  optional `sources: ArtifactSourceMap | None` parameter, resolves the full
  not-yet-installed closure's source map once via `_resolve_file_sources`,
  and returns it alongside the existing closure/report/gating-targets tuple
  so `provision()`'s per-artifact loop reuses the exact validated map instead
  of re-deriving it.
- `_fetch_artifact` gained an optional `resolved_sources: Mapping[str, str] |
  None = None` parameter; it re-resolves via `_resolve_file_sources` up front
  (before `staging_dir.mkdir`, matching the pre-1695 "fail before touching
  anything" contract) as defense-in-depth for a source map or descriptor that
  changed shape between `preflight()`'s and `provision()`'s independent
  catalog re-walks, and for direct-call unit tests that bypass preflight
  entirely.
- `_file_url` is now a `@staticmethod` plain dict lookup against the already-
  resolved map, per the task's explicit instruction.
- `_preverify_artifact`/`_preverify_one_file` thread `resolved_sources`
  through unchanged so a pre-verify mismatch's internal refetch call reuses
  the same resolved map.
- `_require_single_file`'s two call sites (`_aggregate_closure`,
  `_fetch_artifact`) are both gone; the hard multi-file refusal no longer
  exists anywhere in the module.

## Fingerprint change

`_closure_fingerprint_with_sources(root, dependencies, source_map,
resolved_sources)` wraps (does not modify) `service.closure_fingerprint` —
that function is also used by readiness records and installed-manifest
verification, which know nothing about source maps and must not change shape
for this task.

**Critical design decision for back-compat**: the wrapper folds in ONLY
`(ref, path, url)` triples the CALLER actually named in the `sources`
argument — not every entry `_resolve_file_sources` filled in via the
single-file `source_url` fallback. Consequence: when a caller never passes
`sources` (the default, and what every pre-1695 test and any real caller not
yet aware of source maps does), the fingerprint is **byte-identical** to
plain `closure_fingerprint(root, deps)` — the exact formula this codebase
used before TASK-1695. When a caller *does* pass `sources`, swapping a
supplied URL between `preflight()` and `provision()` changes the fingerprint,
so `provision()` raises `ConsentMismatchError`
(`test_source_map.py::test_source_url_changed_after_consent_raises_consent_mismatch`).

This was not the first design tried. An initial version folded in *every*
resolved `(ref, path, url)` triple unconditionally (including single-file
fallback entries), which changed the fingerprint value even for the default
no-`sources` path — breaking ~15 existing tests across
`test_provision_install.py`, `test_provision_serialization.py`, and
`test_provision_crash_recovery.py` that hand-build `AcquisitionConsent` via
the bare `closure_fingerprint()` as a shortcut to avoid a full `preflight()`
call. The caller-supplied-only design fixes this while still satisfying the
spec's "cover credential-free source identities" requirement for the actual
new capability (explicit source maps).

## Back-compat decisions

- `sources` is an optional, keyword-only parameter on both `preflight()` and
  `provision()`, defaulting to `None`.
- A single-file descriptor with a bare `source_url` and `sources` omitted (or
  `{}`, or present but missing an entry for that one file) resolves exactly
  as before TASK-1695 — proven by
  `test_single_file_source_url_without_source_map_still_works` and
  `test_single_file_source_url_with_empty_source_map_still_works`.
- `_fetch_artifact`'s existing direct-call unit tests (11 call sites across
  `test_provision_fetch.py`, `test_provision_install.py`) needed no changes:
  they're all single-file and pass no `resolved_sources`, which still
  resolves via the fallback.
- ~15 tests that hand-construct `AcquisitionConsent` via the base
  `closure_fingerprint()` needed only a `resolved_sources=None` parameter
  added to their local phase-stub functions (`fake_fetch`, `paused_fetch`,
  etc.) — a mechanical signature fix, not a behavior change — because those
  stubs are called with 4 positional args by `provision()` now.

## Test evidence

New `Tests/Model_Artifacts/test_source_map.py` (13 tests):
- Headline: genuine 2-file artifact provisions end-to-end over a real
  fixture server with per-file URLs (`test_multi_file_artifact_provisions_end_to_end_with_source_map`),
  plus a root+dependency multi-file closure variant.
- Back-compat: single-file + `source_url`, no map / empty map (2 tests).
- Preflight-time validation, each proven to precede consent AND network via
  a `client_factory` stub that raises `AssertionError` if ever constructed:
  missing entry, extra entry, non-`http(s)` scheme, query string, userinfo
  (5 tests) — each also asserts the offending URL text never appears in the
  raised error's `str()`.
- `provision()`'s independent re-walk also catches a source map that changed
  shape after consent, before touching any staging directory.
- Consent-mismatch on URL swap, plus a same-map sanity counterpart proving
  the fingerprint is deterministic, not merely order-sensitive.
- Direct proof the fingerprint is byte-identical to
  `service.closure_fingerprint` when no `sources` is ever passed.

Extended `Tests/Model_Artifacts/test_credentials_and_boundaries.py` (1 test):
- `test_multi_file_source_map_urls_never_leak_into_state_manifests_or_errors`:
  provisions a multi-file artifact whose source-map URLs carry a unique path
  marker, then scans every file under the artifact store plus this
  application's own log records (httpx/httpcore's own operational
  request-URL tracing is deliberately excluded from the log scan — logging
  the exact credential-free URL it's about to fetch is expected operational
  visibility, not a leak; the spec's "never contain tokens/cookies/signed
  URLs/query strings" rule targets secrets, not URL identity, and we already
  reject any URL shaped like a secret at preflight time).

Two pre-existing tests' docstrings updated (not their assertions, which still
hold) to remove now-inaccurate "TASK-596/1301 will define this" language:
`test_preflight.py::test_preflight_multi_file_descriptor_raises_catalog_error`
and
`test_provision_fetch.py::test_fetch_multi_file_descriptor_raises_catalog_error_without_touching_anything`.

Full required suite:
```
PYTHONPATH=<worktree> pytest Tests/Model_Artifacts/ Tests/STT/test_boundaries.py -q
423 passed in ~34s
```
(409 baseline + 13 in `test_source_map.py` + 1 hygiene extension = 423.)

TASK-1693 was already `status: Done` (closed as superseded) in a prior
commit on this branch — AC #3 verified, not re-done.

## Deviations from the initial plan

- The consent-fingerprint design changed mid-implementation (see above):
  the first attempt (fold in every resolved triple unconditionally) broke
  back-compat with ~15 existing tests; the caller-supplied-only design fixes
  this and is what's committed.
- Added one extra defense-in-depth test beyond the task's explicit list
  (`test_resolution_failure_at_provision_also_precedes_any_side_effect`),
  proving `provision()`'s own independent re-walk — not just `preflight()` —
  also catches a source map that changed shape after consent, before any
  staging directory is created.

## Files touched

- `tldw_chatbook/Model_Artifacts/acquisition.py` — the core contract.
- `tldw_chatbook/Model_Artifacts/__init__.py` — exports `ArtifactSourceMap`.
- `Tests/Model_Artifacts/test_source_map.py` — new, 13 tests.
- `Tests/Model_Artifacts/test_credentials_and_boundaries.py` — +1 hygiene test.
- `Tests/Model_Artifacts/test_provision_install.py`,
  `test_provision_serialization.py` — phase-stub signature fixes
  (`resolved_sources=None` parameter).
- `Tests/Model_Artifacts/test_preflight.py`, `test_provision_fetch.py` —
  docstring accuracy fixes only.
- `Tests/Model_Artifacts/test_service.py` — `__all__` export-list test
  expectation updated for `ArtifactSourceMap`.
- `backlog/tasks/task-1695 - ...md` — plan, ACs checked, implementation notes,
  status set to Done.
