# PR #1165 fix report — finalization-seam / source-map review findings

Six verified code-review findings fixed on `feat/artifact-finalization-seam`
(TASK-1694 payload-subtree finalization seam + TASK-1695 credential-free
source map). Each behavioral fix below was proven test-first: the new
regression test was run against the pre-fix source (via a scoped `git
stash` of just the changed production file) and confirmed red, then
confirmed green after restoring the fix. The one pushback item (leading
underscore on `_ManagedDownloadStage`) was left alone per the assignment's
explicit instruction — a module-private class name is not a PascalCase
violation.

## P0 — credential sent cross-origin to a mapped per-file URL

**Root cause:** `ArtifactAcquisitionService._auth_headers(repository)`
(`acquisition.py`) resolved and attached a repository's bearer token
unconditionally, based only on `repository`, never on which URL the
request was actually going to. TASK-1695's per-file `ArtifactSourceMap`
lets a caller map an individual file to any absolute URL, including one on
a completely different origin than the descriptor's own `source_url` (a
third-party CDN or mirror). `fetch.stream_fetch` already strips
`Authorization` on a cross-origin *redirect* hop, but `_auth_headers` fed
it the header on the very first, non-redirected request to that
cross-origin URL — the exact case `CredentialResolver`'s own docstring
already promised was excluded ("attached ONLY to the request for the
entry's OWN origin"), just never actually implemented.

**Change (`acquisition.py`):** `_auth_headers` now takes `url` and
`source_url` and returns `None` whenever
`tldw_chatbook.Utils.egress.same_origin(url, source_url)` is `False` —
before even consulting the credential resolver. Both call sites
(`_fetch_one_file`'s real per-file fetch, and `_probe_gating`'s HEAD
probe — see P2 below) now pass the request URL and the descriptor's
`source_url` through.

**Covering tests** (`Tests/Model_Artifacts/test_credentials_and_boundaries.py`):
- `test_credential_attached_to_same_origin_mapped_file` — a per-file
  mapped URL on the SAME origin as `descriptor.source_url` still receives
  the token; both files download.
- `test_credential_withheld_from_cross_origin_mapped_file_but_both_download`
  — two real, differently-ported `FixtureArtifactServer` instances; the
  same-origin file requires and receives the token, the cross-origin file
  is public and receives NO `Authorization` header at all, and both files
  download and verify correctly.

**Verified red/green:** with `acquisition.py` reverted (`git stash`), the
cross-origin test fails —
`AssertionError: the credential must NOT have reached the cross-origin
mapped file` (the same-origin test still passes, as expected — it doesn't
exercise the vulnerability). With the fix restored, both pass.

## P1 — consent hole: single-file `source_url` fallback excluded from the fingerprint

**Root cause:** `_closure_fingerprint_with_sources` (`acquisition.py`)
only folded caller-supplied `(ref, path, url)` triples from the `sources`
argument into the consent fingerprint — deliberately, per its own
docstring, to stay byte-for-byte back-compatible with the pre-TASK-1695
bare `closure_fingerprint()`. But `_resolve_file_sources` resolves a
not-yet-installed **single-file** descriptor's URL from
`descriptor.source_url` even when the caller supplies no `sources` map at
all (the fallback path). That fallback-resolved URL was never folded in,
so a descriptor whose `source_url` changed between `preflight()`/`grant()`
and `provision()` (a dynamic `ArtifactCatalog` doing mirror rotation or
CDN rebalancing — the protocol guarantees no immutability) was NOT
consent-gated: `provision()` would silently fetch from the new origin
under stale consent. Backlog TASK-1712 tracked exactly this residual from
TASK-1695's own review.

**Change (`acquisition.py`):** `_closure_fingerprint_with_sources` no
longer takes a `source_map` parameter at all — it folds in **every**
`(ref, path, url)` triple present in the already-validated
`resolved_sources` mapping, fallback-resolved entries included. This is no
longer fingerprint-identical to the bare `closure_fingerprint(root, deps)`
for any closure containing a not-yet-installed entry (single-file or
multi-file, source map or fallback).

Fixing this broke 13 existing test call sites that hand-built
`AcquisitionConsent(closure_fingerprint=closure_fingerprint(...))`
directly (`test_provision_install.py` ×5, `test_provision_serialization.py`
×4, `test_provision_fetch.py` ×1, `test_provision_crash_recovery.py` ×2,
plus the shared `provision_processes.py` subprocess helper) and one test
(`test_source_map.py`) that asserted the old back-compat equivalence
directly. Per the assignment's instruction, these were updated to
construct consent the supported way rather than weakened: a new shared,
**network-free** `grant_consent(svc, root, catalog, sources=None)` helper
(`Tests/Model_Artifacts/acquisition_test_helpers.py`) grants through
`ArtifactAcquisitionService._aggregate_closure(...).grant()` — the same
pure aggregation the public `preflight()` wraps with a network gating
probe — so no call site needed to start touching the network (several
intentionally use unreachable `example.test` placeholder URLs). Two sites
that are timing-sensitive real subprocess crash tests
(`test_provision_crash_recovery.py`, `provision_processes.py`) use the
same network-free path for the same reason.

**Covering tests** (`Tests/Model_Artifacts/test_source_map.py`):
- `test_single_file_source_url_changed_after_consent_raises_consent_mismatch`
  — the literal TASK-1712 acceptance criterion: a single-file descriptor
  with NO explicit source map at all, whose `source_url` changes between
  `grant()` and `provision()`, raises `ConsentMismatchError`.
- `test_single_file_fallback_fingerprint_differs_from_plain_closure_fingerprint`
  — replaces the old (now-incorrect)
  `test_source_map_fingerprint_matches_plain_closure_fingerprint_when_absent`,
  which asserted the very back-compat behavior that was the bug.

**Verified red/green:** with `acquisition.py` reverted, both new tests
fail — the fingerprint-difference test asserts an equality that no longer
holds pre-fix (reversed), and the mismatch test raises `TransferError:
egress policy blocked...` instead of `ConsentMismatchError` (pre-fix,
`provision()` doesn't even detect the drift, so it proceeds into the real
fetch phase and fails there instead). Both pass after restoring the fix.

## P2a — gating probe targets `descriptor.source_url`, not the URLs actually fetched

**Root cause:** `_probe_gating`'s targets (`_aggregate_closure`) were one
`ArtifactPreflightEntry` per repository, each carrying
`entry.source_url = descriptor.source_url` — the descriptor's OWN URL,
never the resolved per-file URLs a caller-supplied source map actually
points `provision()` at. A gated `descriptor.source_url` therefore blocked
consent even when every real per-file URL was public, and a public
`descriptor.source_url` let consent through even when a mapped file's real
origin required auth.

**Change (`acquisition.py`):** added a `_GatingTarget` dataclass (`url`,
`repository`, `source_url`). `_aggregate_closure`'s gating-targets loop now
builds one representative target per distinct `(repository, origin_of(url))`
pair from the RESOLVED per-file URLs (deduped, so it stays one bounded
probe per distinct source — the design spec's requirement), carrying the
descriptor's `source_url` alongside purely for the P0 origin-binding check.
`_probe_gating` now probes `target.url` and passes `target.url`/
`target.source_url` through to the (now origin-bound) `_auth_headers`.

**Covering tests** (`Tests/Model_Artifacts/test_source_map.py`):
- `test_gated_mapped_file_detected_at_preflight_even_when_descriptor_url_is_public`
  — descriptor URL public, mapped files on a separate gated origin →
  `gating_errors` non-empty.
- `test_gated_descriptor_url_does_not_block_public_mapped_files` — inverse:
  descriptor URL gated, mapped files all public → `gating_errors == ()`
  and `.grant()` does not raise.

**Verified red/green:** with `acquisition.py` reverted, both fail —
`AssertionError: the gated mapped-file origin must have been probed` and
an unexpected non-empty `gating_errors` tuple, respectively. Both pass
after restoring the fix.

## P2b — download stages are never reclaimed (two related leaks)

**Root cause (i):** `ModelArtifactService._gc_staging`
(`service.py`) only recognized the legacy `staging/managed/` tree and
`install-*` tempdirs; a service-owned `download-<fingerprint>/` operation
(TASK-1694's layout) fell through as an "unrecognized top-level name" and
was left alone forever — including its real, potentially large, payload
bytes.

**Root cause (ii):** `_remove_incomplete_download_stage` — the cleanup
`_download_stage_for` calls on any failure while creating a new stage —
unconditionally returned once the marker file existed, regardless of
WHERE the failure happened. A failure after marker creation but before the
temp-to-canonical `os.rename` (e.g. an I/O error at publication) left the
fully-populated ephemeral `.download-<random>/` temp directory behind
forever: nothing, including the (now-fixed) reconcile GC, ever looks up a
stage by its temp name, only its canonical `download-<fingerprint>/` name.

**Change (`service.py`):**
- Added `_DOWNLOAD_STAGE_PREFIX`; `_gc_staging` now collects top-level
  `download-*` directories and hands them to a new `_gc_download_staging`,
  gated behind the same non-blocking `ACQUISITION_SESSION_LEASE_KEY`
  `_gc_managed_staging` already uses (a live `provision()` holds it for
  its entire run).
- Added `_download_stage_ownership(operation)`: proves ownership via
  self-consistency — the marker parses to the exact schema
  `_download_stage_for` writes, and its `descriptor_fingerprint` matches
  the fingerprint encoded in the operation directory's OWN name (reconcile
  has no catalog to check the marker against a live descriptor, unlike
  `_read_download_stage_marker`). Containment is checked alongside: exact
  expected entries, no symlinks anywhere in `payload/` (new
  `_assert_no_symlinks`, since payload holds arbitrary bytes, not JSON) or
  `state/` (reused `_validate_download_stage_state`).
  **Design choice** (explicitly called out per the assignment, since the
  finding left this open): a marker that is missing, unparseable, or
  self-inconsistent in ANY way returns `None` and the entry is
  **left completely alone**, never removed — more conservative than the
  legacy `managed/` GC (which reclaims a structurally-invalid entry
  outright), because this layout's payload can hold a large, real,
  in-progress download that a transient/corrupt marker read must never
  cost.
- `_gc_download_staging` reclaims an owned-and-contained operation only
  when its reference is **already installed**
  (`core.artifact_path(reference)` exists) — the one on-disk-provable
  "this exact stage will never be resumed again" signal available without
  a catalog (`provision()` itself skips the fetch/pre-verify/install
  phases entirely for an already-installed reference). A stage for a
  not-yet-installed reference is exactly the "valid and resumable" case
  and survives regardless of lease state or age — confirmed unchanged
  against the existing crash-recovery suite, which already asserts a
  not-yet-installed stage survives `reconcile()`.
- `_remove_incomplete_download_stage`: removed the early
  `if self._managed_path_exists(marker): return`. The existing
  `operation_identity` check already proves this call exclusively owns the
  ephemeral temp directory (unique `tempfile.mkdtemp` name, never looked
  up by anything else), so the marker is now unlinked (alongside the
  existing payload/state emptiness checks and rmdir cleanup) instead of
  blocking cleanup outright.

**Covering tests** (`Tests/Model_Artifacts/test_reconcile_staging_gc.py`):
- `test_download_stage_for_already_installed_reference_is_reclaimed_and_reported`
  — stage created, artifact separately installed, `reconcile()` removes
  the stage and reports it in `staging_removed`.
- `test_download_stage_for_not_yet_installed_reference_survives_reconcile`
  — same stage shape, reference NOT installed → untouched, empty
  `staging_removed`.
- `test_download_stage_with_unparseable_marker_is_left_alone_even_when_installed`
  — marker corrupted even though the reference IS installed (would
  otherwise qualify) → still untouched; pins the conservative
  "unprovable = leave alone" design choice.
- `test_download_stage_creation_failure_after_marker_leaves_no_orphan_temp_dir`
  — `os.rename` patched to fail only on the temp→canonical publish step;
  after the resulting `ArtifactStateError`, staging is empty.

**Verified red/green:** with `service.py` reverted, the "already
installed → reclaimed" and "post-marker failure → no orphan" tests fail
(`assert not True` / a leftover `.download-<random>` directory found);
the other two pass either way pre-fix (nothing recognized the layout at
all, so "survives" and "left alone" were trivially true) — they guard the
safety property going forward, not the leak itself. All four pass after
restoring the fix.

## P2c — tests hit the network (`test_source_map.py`)

**Root cause:** three tests built `sources`/descriptor URLs against
`https://example.test/...` and called the real `svc.preflight(...)` with
no `client_factory` override, so `_probe_gating`'s `check_url_or_raise_async`
performed genuine DNS resolution against an unreachable placeholder host —
non-deterministic in offline/sandboxed CI (one of them, the one at the
line the finding cited, even said so explicitly in its own docstring:
"preflight() ... legitimately reaches the network gating probe").

**Change (`test_source_map.py`):**
- `test_resolution_failure_at_provision_also_precedes_any_side_effect` and
  `test_source_url_changed_after_consent_raises_consent_mismatch` now use
  a real, loopback `FixtureArtifactServer` (with `trusted_origins`) for the
  URLs that reach `preflight()`'s gating probe; each still proves exactly
  what it proved before (provision()'s re-walk failure precedes any
  side effect; a swapped URL raises `ConsentMismatchError`), unaffected by
  which host the probe happens to hit.
- The third (`test_source_map_fingerprint_matches_plain_closure_fingerprint_when_absent`)
  was replaced outright by the P1 fix above — it asserted the exact
  back-compat behavior that was the bug, and its network-free replacement
  uses `grant_consent` instead of the real `preflight()`.

**Covering evidence:** full-file run
(`pytest Tests/Model_Artifacts/test_source_map.py -q`) is green with no
`Egress blocked (dns_failure)` warnings in captured output (previously
present for `example.test` on every run touching those three tests).

## P3 — duplicated test helpers

**Change:** new shared module `Tests/Model_Artifacts/acquisition_test_helpers.py`
holding:
- `_trusted(srv)` — deduped from six byte-identical copies
  (`test_stream_fetch.py`, `test_preflight.py`, `test_source_map.py`,
  `test_credentials_and_boundaries.py`, `test_provision_install.py`,
  `test_provision_fetch.py` — the finding named three, all six were
  actually identical, so all six were consolidated).
- `_two_file_descriptor(ref, source_url=..., *, role=...)` — the
  real-content 2-file descriptor from `test_source_map.py`, now also used
  by `test_credentials_and_boundaries.py` in place of its near-duplicate
  `_two_file_descriptor_for_hygiene`. `test_preflight.py`'s OWN
  `_two_file_descriptor` (a different, no-real-content helper whose only
  job is tripping `CatalogError`, per its own docstring) was deliberately
  left alone — not the same helper, not part of this duplication.
- `grant_consent(svc, root, catalog, sources=None)` — the network-free
  consent helper P1's fix needed at 15 call sites; centralizing it here
  avoided reproducing the same `_aggregate_closure(...).grant()`
  one-liner that many times.

All call sites updated; no behavioral change (confirmed by the full green
run below).

## Full suite

```
PYTHONPATH=<worktree> <main-checkout>/.venv/bin/pytest \
  Tests/Model_Artifacts/ Tests/STT/test_boundaries.py -q
...
432 passed in 38.67s
```

(423 baseline + 9 net-new regression tests: P0 ×2, P1 ×2 net [+2 new, -1
replaced], P2a ×2, P2b ×4 = 10 new, 1 removed = 9 net.)

## Not acted on

Per explicit instruction: the finding that `_ManagedDownloadStage` (a
leading-underscore, module-private class) violates PascalCase was not
acted on — a leading underscore marking module-private is idiomatic
Python, not a naming-convention violation.
