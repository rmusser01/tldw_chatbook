# ADR-050: Use external Parakeet roots with managed VAD

Status: Accepted
Date: 2026-08-09
Related Tasks: TASK-598
Amends: ADR-025
Informed by: ADR-041

## Decision

Chatbook permits a user to select an exact catalog-known Parakeet v2 or v3
ONNX model directory and use it in place. The Parakeet root remains an
external, user-owned source: Chatbook does not copy, modify, delete, lease, or
label that directory as a managed installed artifact unless the user later
chooses the separate managed-copy action.

The selected model ID and precision resolve the authoritative Chatbook catalog
descriptor. The user does not create or select a bundle manifest. Every
descriptor-required model and external-data file must be a contained,
materialized regular file and match the descriptor's exact size and SHA-256.
Unknown or modified graphs are rejected by their bytes without parsing an
untrusted ONNX graph in the UI process or resident inference worker. Required
symlinks and irregular files remain unsupported.

Chatbook supplies the pinned Silero VAD dependency. If it is absent, selecting
an external Parakeet source presents consent for a VAD-only managed download;
the external Parakeet root is not downloaded or copied. The resident STT worker
holds shared leases for the exact managed VAD dependency for its full residency
interval. This creates an intentional mixed-ownership runtime closure:

- external descriptor-verified Parakeet root;
- managed, immutable, leased Silero VAD dependency.

External selections persist per exact model ID and precision. Resolution order
is an explicit per-job external directory, then the configured preferred source
for the exact descriptor (external or managed), then an active managed root
when no preference exists, then the existing verified legacy fallback when no
preference or managed root exists. A remembered but non-preferred external path
is not a resolver candidate. An explicit override or preferred source that is
missing, changed, or invalid fails clearly and does not silently fall through to
another source or provider.

The external descriptor reference, local metadata snapshot, and exact managed
VAD reference participate in resident-model identity and worker recycling.
Transcript provenance does not fabricate a managed root: `artifact_root` is
null, the VAD is recorded in `artifact_dependencies`, and no local path is
recorded. Persistent paths live only in dedicated local configuration; a
per-job override may live in the local job options required for restart/retry.
Paths do not enter generic logs, notifications, sanitized errors, or transcript
provenance.

Full descriptor hashing runs when an external directory is selected and once
before first use after each application restart. Within that application
process, identical concurrent verification is coalesced and an unchanged
device/inode/size/mtime snapshot may reuse the verified result. Cancelling one
waiter stops only that wait; shared hashing continues for remaining waiters and
is cancelled when no waiter remains. Any metadata change invalidates the
result, requires a new hash pass, and recycles a resident model. Hashing is
chunked, cancellable, and runs outside the Textual event loop. No verification
result is persisted as a durable integrity receipt.

The bounded in-memory result set includes persistent configured selections and
verified job-scoped selections owned by live Library batches. Repeated items in
one batch reuse an identical per-job snapshot; the batch releases its entries on
completion or cancellation. A restart always requires a new full hash pass.

Persistent external selection is atomic with VAD readiness: Chatbook commits
the new path and preference only after root verification and exact managed VAD
readiness succeed. Offline operation, VAD acquisition failure, or cancellation
leaves the previous source configuration unchanged.

The optional managed-copy action uses the existing artifact service to stage,
revalidate, and install only the Parakeet root. It reuses the already managed
VAD but does not write root readiness, update the active selector, or change the
source preference. The installed root is presented as activation-required, not
broken. Later explicit activation verifies the complete root-plus-VAD closure,
writes readiness last, switches the active selector, and only then changes the
exact source preference to managed. Root inventory permits that explicit
activation when an installed root manifest is valid but readiness is absent;
dependency-only entries never receive an activation action.

## Context

ADR-025 required the initial local ONNX import to copy a complete descriptor-
backed closure into the managed store. That provides strong immutable ownership
but prevents a user who already has a multi-gigabyte Parakeet model from using
it without duplicating those bytes.

The current application already routes a per-job Parakeet model directory into
batch transcription, but that path checks only expected filenames and a local
metadata snapshot. It does not verify catalog hashes, persist an exact v2/v3
and precision identity, or attach the managed VAD required by long-form
transcription. Extending that existing path is smaller and more useful than
building a second copy-required importer.

ADR-041 established direct external configuration as the first-class path for
transcribe.cpp, but cannot govern this case by itself. Parakeet participates in
automatic application routing, uses multi-file descriptor-known bundles, and
requires a managed dependency. Those differences require this explicit
amendment to ADR-025.

## Consequences

- Users can reuse v2/v3 INT8 or F32 model files where they already exist.
- A small verified VAD download may still be required and always requires
  explicit consent.
- External roots have weaker immutability than managed roots. A metadata and
  hashing boundary reduces accidental change and narrows races but cannot
  eliminate the interval in which the native runtime reopens user-owned files.
- Materialized files are required; symlink-based Hugging Face cache snapshots
  must be materialized into a regular directory before use.
- The managed artifact service needs a public exact-dependency lease surface;
  it does not register the external root as an artifact or create a fake root
  readiness record.
- The user-facing model lifecycle refuses VAD deletion while a configured
  external source depends on it. Core deletion remains protected by normal
  operation leases while the runtime is resident; an out-of-band deletion is
  reported as a missing dependency on next use.
- Dependency-only inventory entries are labeled as managed dependencies and
  are never offered an Activate action.
- The required CPU evidence matrix remains Linux x86_64, Linux aarch64,
  Windows x86_64, macOS arm64, and macOS x86_64, as defined by the parent STT
  design. Evidence may land incrementally, but unsupported hosts cannot be
  inferred from structural tests.
- The legacy `transcription.parakeet_onnx_model_dir` value is treated only as a
  v2 INT8 migration candidate and must pass the new descriptor validation.
- No provider-initiated download is introduced. Headless and already-enqueued
  transcription paths remain side-effect free.

## Alternatives considered

| Option | Reason rejected |
| --- | --- |
| Always copy the selected root into the managed store | Duplicates multi-gigabyte files and violates the direct-use requirement. |
| Require the user to supply a VAD directory | Exposes an internal dependency and makes a curated app dependency the user's maintenance burden. |
| Bundle VAD inside every application wheel | Adds binary redistribution, wheel-size, and release-policy work while duplicating the managed artifact lifecycle. |
| Register the external directory as a managed artifact | Creates ownership, deletion, readiness, and immutability claims the app cannot enforce. |
| Accept arbitrary ONNX bundles by parsing their graphs | Expands the trust boundary and requires a separately isolated, resource-bounded validator. |
| Persist a verification receipt and skip hashing after restart | Treats mutable filesystem metadata as durable integrity evidence and weakens descriptor verification. |

## Rollback

Disable external-source selection and clear the external source preference.
Managed Parakeet roots and the managed VAD remain valid. No transcript or
artifact migration is required because external paths never enter provenance
and external roots never become managed artifacts implicitly.

## Links

- [TASK-598 design](../../Docs/superpowers/specs/2026-08-09-task-598-external-parakeet-bundles-design.md)
- [ADR-025](025-shared-stt-artifacts-and-runtime-routing.md)
- [ADR-041](041-direct-local-gguf-before-managed-acquisition.md)
