# Shared Model Artifact Core — Focused Design

**Date:** 2026-07-28
**Status:** Approved
**Task:** TASK-594
**ADR:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`

## Purpose

Provide the smallest provider-neutral filesystem service that can safely install,
identify, activate, lease, list, reconcile, and delete immutable ONNX and GGUF
artifacts. STT is the first consumer.

This replaces the abandoned earlier lifecycle branch, which grew beyond 21,000
added lines. No code will be cherry-picked from that branch.

## Scope

TASK-594 adds:

- immutable typed descriptors for roots and exact dependencies;
- per-file size and SHA-256 verification;
- immutable installed directories;
- atomic root-readiness and active-version records;
- canonical dependency-closure fingerprints;
- leased artifact handles and lease-aware deletion;
- installed inventory, reconciliation, and simple disk totals.

It does not add:

- HTTP downloads, resume support, credentials, or progress UI;
- a model browser or first-run wizard UI;
- arbitrary local model import;
- inference-runtime imports or model loading;
- content-addressed deduplication;
- migration of existing LLM or external model caches.

Those boundaries preserve TASK-595 for verified downloads and TASK-596 for the
reusable model-management UI.

## Shape

Add one production module:

`tldw_chatbook/Model_Artifacts/service.py`

It contains the frozen descriptor dataclasses, artifact handle, small result
types, and `ModelArtifactService`. The package `__init__.py` re-exports the public
types. Existing `leases.py` and `Utils/atomic_file_ops.py` are reused.

No database, repository interface, factory, background service, or cache is
introduced.

## Descriptor contract

Three frozen dataclasses are sufficient:

- `ArtifactRef`: `artifact_id`, immutable `revision`, and `variant`;
- `ArtifactFile`: contained POSIX-relative path, byte size, and SHA-256;
- `ArtifactDescriptor`: reference, stable `model_id`, root/dependency role,
  format, consumer, model family, upstream repository, immutable upstream
  revision, source URL, precision, expected installed bytes, license identifier,
  license URL, usage notice, runtime name/version constraint, supported
  OS/architecture values, provenance classes, required files, and exact
  dependency references.

Descriptor validation rejects empty identifiers, traversal, absolute paths,
backslashes, symlinks, duplicate paths, case-insensitive path collisions, invalid
digests, reserved manifest/state filenames, and conflicting duplicate
dependencies. `artifact_id`, `revision`, and `variant` use one canonical
lowercase ASCII path-component grammar and reject Windows-reserved names,
trailing dots/spaces, and case-fold aliases. Upstream repository names and
revisions remain separate metadata and are not used as store paths.

`expected_installed_bytes` must equal the sum of declared file sizes.
`integrity_verified` and `local_integrity_recorded` are mutually exclusive.
`source_url` is a credential-free provenance URL, never a signed/request URL;
userinfo and secret-bearing query data are not persisted in manifests.

Provenance is a tuple so `chatbook_curated` can coexist with either
`integrity_verified` or `local_integrity_recorded`.

Manifests and state records carry `schema_version = 1` and fail closed on
unsupported versions.

## Store layout

```text
<root>/
  artifacts/<artifact-id>/<revision>/<variant>/
    manifest.json
    <declared files>
  active/<artifact-id>.json
  ready/<artifact-id>/<revision>/<variant>.json
  staging/
  locks/
```

The service is the only code allowed to create, replace, or remove entries in
this managed store.

Installation copies a descriptor-backed source directory into a same-filesystem
temporary staging directory, rejects undeclared or unsafe files, verifies every
size and digest, writes the manifest, and renames into an absent final directory.
When the destination already exists, idempotent success requires both an
identical manifest and a fresh size/SHA-256 verification of every installed
payload file. A conflicting or corrupt destination is rejected. Populated
installed directories are never replaced in place. A failed copy or verification
never creates a loadable final directory and its operation-owned staging is
rolled back.

Copying into a unique, non-loadable staging directory does not hold the lifecycle
lease. Verification and promotion acquire the lifecycle lease plus the target
artifact's exclusive lease, recheck the destination, and use a same-filesystem
rename after the authoritative absent-destination check. This relies on the
managed store's sole-writer contract: Python's standard library has no portable
atomic no-replace directory rename across every supported platform, so
out-of-band mutation by a non-service writer is outside this correctness
boundary. This keeps slow file copying out of the authoritative writer critical
section while preserving one writer for installed state.

## Readiness, activation, and handles

`activate(root_ref)` resolves the complete transitive dependency closure from
installed manifests and rejects cycles or missing exact revisions. If no valid
readiness record already names that exact closure, activation verifies every
artifact before constructing one. A valid existing readiness record is reused
without rehashing multi-gigabyte payloads. Activation then writes:

1. the root readiness record, containing the exact ordered closure and its
   fingerprint;
2. the active selector for the artifact ID.

The precise invariant is: readiness is written only after every artifact in the
closure has been promoted and verified. The active selector may be written
afterward. If that selector write fails, atomic replacement preserves the prior
active selector; the newly ready revision remains installed but inactive.

Full payload hashing occurs during installation and readiness construction or
reconstruction, not on every inventory read, activation, or model acquisition.
Service-managed mutation invalidates affected readiness records. Out-of-band
store modification is detected by explicit reconciliation rather than by
rehashing every model load.

The closure fingerprint is SHA-256 over canonical JSON containing the sorted
root and dependency references. `ArtifactHandle` carries the root reference,
ordered closure, fingerprint, resolved paths, lease keys, and a resident identity
of `(root_ref, closure_fingerprint)`.

`acquire(root_ref)`:

1. reads the readiness record;
2. derives the lease set from the exact closure bound to that fingerprint and
   acquires shared TASK-505 leases for every member;
3. re-reads and compares readiness;
4. returns a context-managed leased handle only when it is unchanged.

This closes the load/delete race without a larger transaction system.

## Mutation and deletion

All authoritative store mutations use one private exclusive lifecycle lease.
This intentionally serializes uncommon promotion, activation, deletion, and
reconciliation operations.

- Promotion also takes the target artifact's exclusive lease.
- Activation takes one shared lease set over the exact root/dependency closure
  before verification and state writes. The exclusive lifecycle lease already
  serializes state writers, while shared closure leases allow activation when a
  resident model is safely reading a common dependency.
- Deletion takes the target artifact's exclusive lease.
- Reconciliation uses shared artifact leases when valid references or closures
  are available for verification. When malformed or missing state makes a
  closure unknowable, the lifecycle lease alone permits removal of that
  unreadable/orphaned derived record; reconciliation never deletes the
  corresponding payload directory automatically.

Deletion then acquires the target artifact's exclusive TASK-505 lease before
changing metadata. If a worker holds a shared root or dependency lease, deletion
fails clearly and changes nothing. After the exclusive lease is acquired, the
service removes readiness records that reference the target, clears an active
selector when it selects the target or any root whose readiness was invalidated,
and deletes the immutable directory.

The fixed order is lifecycle lease, then sorted artifact leases. Readers acquire
only sorted artifact leases, so there is no lock cycle.

## Inventory, reconciliation, and accounting

`list_installed()` scans manifests and state records without importing native
runtimes or hashing model payloads.

`reconcile()` runs under the lifecycle lease. It:

- removes active or readiness records that reference missing, invalid, or
  payload-corrupt artifacts;
- reconstructs readiness only after full size and hash verification succeeds;
- leaves corrupt installed directories visible but unloadable;
- reports observed staging entries, which may include an active pre-lifecycle
  install, and never deletes them automatically.

`disk_usage()` returns logical installed bytes, staging bytes, and filesystem
free bytes. It does not claim portable physical-allocation or quota accuracy.

## Public API

The initial public surface is limited to:

- `install(descriptor, source_directory)`;
- `activate(root_ref)`;
- `acquire(root_ref)`;
- `list_installed()`;
- `delete(ref)`;
- `reconcile()`;
- `disk_usage()`.

TASK-595 may add download-specific staging operations only when its resume
requirements are implemented.

## First-run and Settings integration

The active first-run wizard design is the UX reference:

- all setup is skippable and re-runnable;
- current installed/configured values prefill;
- failures are explicit and never trap the user;
- Summary reads persisted state rather than transient widget state.

TASK-594 contains no UI. TASK-595 supplies explicit-consent verified downloads.
TASK-596 supplies one reusable model picker/download panel for ongoing Settings
management and explicitly requires active revision/precision selection plus
onboarding reuse.

TASK-1301 adds a skippable **Speech transcription** step to the wizard's Full
track using that same panel.
It will show the recommended model and precision, source, license, download size,
destination, and free-space result; activate only after verification; and report
persisted installed state in Summary. Quick setup remains unchanged.

## Testing

Use one focused offline test module plus the existing TASK-505 process helpers.
Tests cover:

- descriptor validation and stable fingerprints;
- successful and failed verified installation;
- exact dependency readiness, readiness reuse, and atomic activation rollback;
- acquire/revalidate behavior;
- deletion blocked by shared leases and allowed after release/process death;
- reconciliation of missing, corrupt, and interrupted state;
- path containment and symlink rejection;
- installed/staging disk totals;
- absence of network and inference-runtime imports.

The Windows/Linux TASK-505 qualification gate remains open. Implementation may
proceed from the approved macOS evidence, but TASK-505 is not marked complete and
no cross-platform proof is claimed until native Windows and Linux evidence exists.
