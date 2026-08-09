# TASK-598 — External Parakeet ONNX Bundles with Managed VAD

Status: Approved design
Date: 2026-08-09
Task: TASK-598
Decision: [ADR-050](../../../backlog/decisions/050-external-parakeet-roots-with-managed-vad.md)

## Summary

Users can select catalog-known Parakeet v2 or v3 ONNX model directories and
transcribe directly from those directories. External roots remain user-owned;
Chatbook supplies and leases the verified managed Silero VAD dependency. A
managed copy remains an optional secondary action, never a prerequisite.

This design renovates the existing `parakeet_onnx_model_dir` path instead of
building a second copy-required importer. It preserves the shared executor,
artifact service, routing, provenance, and first-run model setup already landed
by the surrounding STT workstream.

## Current-state anchors

- `STT/parakeet_dispatch.py` already resolves an explicit model directory
  before a managed or verified legacy model, but snapshots expected filenames
  without checking descriptor hashes.
- `STT/executor.py` and `STT/executor_worker.py` already protect local sources
  with path-private metadata snapshots and managed roots with full-lifetime
  operation leases. They currently treat local roots and managed closures as
  mutually exclusive.
- `Local_Ingestion/parakeet_v2_artifact.py` already defines the exact v2/v3,
  INT8/F32, and Silero VAD descriptors and download source maps.
- Library already carries a per-job `transcription_model_dir` override.
- First-run setup and Lab Models already expose managed Parakeet acquisition and
  activation.

## Goals

- Use a selected v2/v3 INT8/F32 model directory without copying it.
- Persist external selections per exact catalog model and precision.
- Keep the existing per-job directory override.
- Verify all runtime-consumed model bytes against the catalog without parsing
  ONNX graphs.
- Supply VAD through the existing consented managed-download boundary.
- Hold a real managed VAD lease for the full resident runtime lifetime.
- Preserve path-private result provenance and deterministic routing.
- Offer an optional, safe managed copy after direct use works.

## Non-goals

- Arbitrary or modified ONNX graph admission.
- User-authored bundle manifests.
- User-supplied VAD directories.
- Symlinked Hugging Face cache snapshots.
- Provider-initiated or transcription-triggered downloads.
- A persistent external-model database or fake artifact registration.
- The deferred large STT evaluation harness from TASK-1023.

## Source ownership and resolution

An external source record contains only the exact catalog model ID, precision,
local directory, and preferred source (`external` or `managed`) required to
select it again. It is configuration, not an installed artifact. The config
supports one record for each exact model/precision pair so v2/v3 and INT8/F32
cannot alias one another.

Resolution for one Parakeet request is:

1. Explicit per-job external directory for the request's exact model/precision.
2. Matching persistent external selection.
3. Matching active managed root.
4. Existing verified legacy fallback.

An explicit or persistent external selection is authoritative. If it is gone,
changed, or invalid, resolution fails with recovery actions rather than silently
using a different model source or provider. The legacy singular configured
directory is considered only as a v2 INT8 migration candidate.

Activating a managed root changes the source preference only for that exact
descriptor. It may retain the external directory so the user can choose it
again, but managed activation never erases or modifies external files.

## Descriptor verification

The caller already knows the desired model ID and precision from routing or the
catalog row. Those values resolve the trusted descriptor; the external
directory supplies only bytes.

Verification performs the following for every descriptor-required file:

1. Validate and resolve the selected directory through the existing path
   boundary.
2. Form only catalog-declared relative paths.
3. Require each path to remain contained in the selected directory.
4. Reject missing paths, symlinks, and non-regular files.
5. Compare exact size.
6. Compute and compare SHA-256 in cancellable chunks.
7. Capture a path-private device/inode/size/mtime snapshot for dispatch.

The known graph digest also pins its external-data relationship, so no ONNX
parser is required. F32 external-data payloads are independently declared and
hashed. Unrelated regular files such as README or license text are ignored
because they are not given to the runtime.

Validation runs off the Textual event loop and reports determinate byte
progress. Concurrent checks for the same exact descriptor and unchanged
snapshot share one in-flight result. Cancelling one caller cancels only that
caller's wait; hashing continues for remaining callers and stops when no caller
still needs it. The in-memory cache is bounded to current configured selections
and is cleared when a source changes.

The app performs a full hash pass on selection and before the first use after
each restart. Later requests in the same process may reuse the result while
the metadata snapshot remains identical. The worker repeats the metadata check
immediately before native load and before resident reuse. A change forces a
new hash pass and resident recycle.

This narrows but cannot remove the race between validation and an external file
being reopened by the native runtime. The managed-copy action is the stronger
choice for users who need immutable store semantics.

## Managed VAD dependency

The external source never accepts a user-provided VAD path. Dispatch resolves
the exact Silero VAD reference declared by the Parakeet descriptor.

If the dependency is absent, interactive configuration runs a VAD-only
preflight and presents the existing consent information: source, revision,
license, size, destination, and free space. Provisioning uses the existing
acquisition service with activation disabled for the dependency descriptor.
No Parakeet root URL appears in that plan or transfer.

The artifact core exposes a public exact-dependency acquisition/lease result
that verifies the installed dependency under shared mutation protection and
holds its lease until closed. It does not create a fake root readiness record.
The executor request can therefore combine:

- a verified `LocalSourceSnapshot` for the external Parakeet root; and
- exact managed dependency references plus the managed store root.

The worker acquires those dependency leases before loading the runtime, passes
the managed VAD directory to `ParakeetOnnxRuntime.load`, and retains the lease
for resident reuse. Dependency-reference changes participate in resident
identity and force recycle.

The user-facing model lifecycle refuses VAD deletion while configured external
sources require it. The core's normal lease refusal protects active/resident
use. Out-of-band deletion is detected as a missing dependency on next use and
offers reinstall.

## Identity and provenance

Resident identity combines the exact catalog descriptor reference, precision,
external metadata snapshot token, device, and managed VAD reference. Paths stay
out of identity representations and logs.

Direct-external result provenance records:

- provider and model ID;
- precision and requested/effective device;
- requested/effective language;
- `artifact_root = null`;
- the exact VAD lease identity in `artifact_dependencies`.

It never records the external directory or fabricates a managed root revision.
UI wording is “External source · descriptor verified,” not Installed, Managed,
or Integrity verified. The exact path may appear only where the user explicitly
selects or edits it. Persistent paths live in dedicated local configuration;
per-job overrides may remain in local job options to preserve restart/retry.

## User experience

### First-run setup

For the selected exact Parakeet model and precision, show both:

- **Use model from disk…**
- **Review and install…**

Directory verification is cancellable and visibly reports progress. A missing
VAD produces a VAD-only consent step. The pending external choice becomes
persistent when the user continues; stale callbacks are fenced by selection
generation and screen lifetime. Cancelling leaves the prior source unchanged.

Directory verification remains available when the ONNX runtime is absent, but
the source is labeled **Runtime required** and cannot become usable until the
runtime is installed.

### Lab Models

Each exact Parakeet catalog row offers **Use from disk…**. External selections
appear in a clearly separate user-owned section with:

- Change directory
- Stop using
- Copy into managed store

A dependency-only VAD row is labeled **Managed dependency** and never receives
an Activate action.

### Library per-job override

The existing Parakeet-only directory field remains and gains a directory
picker. It does not modify global source configuration; the chosen path may
remain in that local job's options for restart and retry. Validation and any
needed VAD consent complete before a job record is created or enqueued.
Headless callers receive a structured `ModelNotInstalled` outcome and no
download side effect.

All three surfaces call one shared selection/validation/configuration flow.
They do not implement independent rules.

## Optional managed copy

After external verification and VAD readiness, **Copy into managed store**:

1. Preflights only the root bytes not already installed.
2. Obtains explicit consent for additional disk use.
3. Copies only catalog-declared root files through existing staging.
4. Revalidates staged bytes through `ModelArtifactService.install`.
5. Reuses the installed exact VAD dependency.
6. Activates root readiness only through the existing readiness-last boundary.

Copy failure or cancellation leaves the external source unchanged. If the exact
managed root already exists, no copy occurs and the existing activation control
is shown. The managed source becomes preferred only after explicit activation.

## Failure model

Stable path-safe outcomes distinguish:

- unsupported descriptor/model/precision;
- missing, irregular, symlinked, corrupt, or changed root files;
- missing managed VAD;
- VAD acquisition cancellation, failure, or contention;
- insufficient managed-copy disk space;
- managed-copy interruption or conflict;
- unavailable ONNX runtime.

Interactive recovery can reselect the directory, install VAD, select a managed
model, or retry with faster-whisper where the routing policy permits it. No
selected external source silently falls through after failure. Errors,
notifications, logs, job outcome summaries, and transcript provenance do not
include the path.

## Verification strategy

Use tiny injected descriptors for validator tests; do not create giant catalog
fixtures. Cover v2/v3, INT8/F32, external-data files, missing/wrong bytes,
symlinks, irregular nodes, mutation, cancellation within a hash chunk loop,
coalesced concurrent validation, harmless extra files, and cache invalidation.

Focused configuration and dispatch tests cover descriptor-keyed persistence,
legacy migration, per-job precedence, managed activation, no silent fallback,
and path-private failures.

Focused artifact/executor tests cover VAD-only consent, no root download,
exact dependency leasing, mixed external/managed runtime load, resident recycle,
dependency provenance, and user-facing deletion protection. Reuse the artifact
core's existing generic staging/crash/lease tests rather than duplicating them.

Mount one real directory-picker flow. First-run, Lab Models, and Library need
focused state/event tests proving that they call the same shared flow. Run one
real macOS external-mode transcription smoke through the app-owned shared
executor and managed VAD.

Windows and Linux remain required release evidence. Structural tests do not
close that gate; TASK-598 remains In Progress until those hosts are available
and the focused platform evidence passes.

Only affected focused tests, scoped lint, formatting checks for changed files,
and `git diff --check` run locally. The unrelated full suite is not part of this
task's local gate.

## ADR check

ADR required: yes
ADR path: `backlog/decisions/050-external-parakeet-roots-with-managed-vad.md`
Reason: direct external roots combined with managed dependencies change
artifact ownership, runtime identity, provenance, configuration, and deletion
boundaries established by ADR-025. ADR-041 is a precedent but does not cover
automatically routed multi-file Parakeet models or mixed ownership.

## Implementation boundary

No production code is authorized by this design alone. Implementation begins
only after this spec and ADR are reviewed and an approved task plan maps each
change to TASK-598 acceptance criteria.
