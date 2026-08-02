# TASK-597 — Bounded local GGUF artifact import: design

**Date:** 2026-08-01
**Task:** TASK-597 — Add bounded local GGUF artifact import
**Parent:** TASK-596 — Renovate the local model artifact browser
**Depends on:** TASK-594, TASK-595
**ADR:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Status:** approved section-by-section on 2026-08-01

## Outcome

Add one complete local-import path to **Lab → Models → Installed**. A user can
select a compatible GGUF file, have Chatbook inspect it without loading a native
runtime, copy it into managed storage, verify the managed bytes, and register an
exact immutable artifact for the pinned `transcribe.cpp==0.1.3` runtime.

The selected external file remains untouched and is never a runtime dependency.
Unknown compatible files are labeled **Local integrity recorded**, receive a
content-derived identity, and never become automatic STT routing candidates.

TASK-604 still owns the transcribe.cpp provider, curated catalog, optional
dependency, execution, and model-specific capability checks. Until that task
lands, successful import reads:

> Imported for transcribe.cpp. Provider setup is required before transcription.

TASK-597 does not claim that the imported model can transcribe immediately.

## Existing foundation

TASK-594 and TASK-595 already provide:

- strict immutable `ArtifactDescriptor` and `ArtifactRef` values;
- a format-neutral `ModelArtifactService` as the sole managed-store writer;
- isolated same-filesystem staging, per-file SHA-256 verification, immutable
  promotion, atomic active selectors, readiness records, and reconciliation;
- `install(..., consume_source=True)` for moving a verified service-owned stage
  into installation without keeping a second model-sized managed copy;
- an exclusive cross-process `ACQUISITION_SESSION_LEASE_KEY` used by managed
  downloads and staging reconciliation;
- Installed-view inventory, disk accounting, progress, activation, deletion,
  repair, and an off-event-loop legacy-model scan.

The current Installed view deliberately labels unmanaged files **Import is not
available yet**. TASK-597 replaces that dead end for GGUF files only.

## Scope

This task includes:

1. One explicit **Import GGUF…** action and `.gguf`-filtered file picker.
2. Backend enforcement that the selection is a regular, non-symlink file.
3. A bounded standard-library GGUF v3 structural reader.
4. A pinned declaration of the architecture names accepted by
   `transcribe.cpp==0.1.3`.
5. Disk preflight for one managed copy plus a fixed safety margin.
6. Copy-and-hash into a service-owned, lease-protected local-import stage.
7. Post-copy source-snapshot and GGUF revalidation.
8. Exact curated recognition where a suitable descriptor already exists;
   deterministic local-integrity descriptors otherwise.
9. Immutable installation, exact artifact activation, progress, cancellation,
   inventory refresh, and contained staging cleanup.

## Non-goals

- Importing ONNX, SafeTensors, PyTorch, or arbitrary model bundles.
- Loading a model or invoking any inference/native GGUF parser during import.
- Proving that tensor contents are behaviorally correct or malware-safe.
- Implementing the transcribe.cpp provider, curated model catalog, inference,
  settings, or first-run setup; TASK-604 owns those outcomes.
- Making transcribe.cpp a semantic default or silent fallback.
- Migrating server launch paths or retiring the legacy Download Models surface.
- Adding per-row import buttons, a second consent modal, resumable local copies,
  conversion, quantization, benchmarking, or transcription evaluation.
- Adopting an already-managed arbitrary remote GGUF into a runtime-specific
  descriptor; this task accepts an explicitly selected local file.

## ADR check

**ADR required:** no
**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Reason:** ADR-025 already authorizes managed local GGUF import, precise local
provenance, pinned transcribe.cpp compatibility, and curated-only automatic
routing. This design implements that boundary without changing it.

## Chosen architecture

Use a dedicated, Textual-free local importer composed over the existing
artifact core.

Rejected alternatives:

- Treating local files as downloads overloads network, credential, resume, and
  consent semantics with unrelated behavior.
- Adding a broad generic file-import API to `ModelArtifactService` expands the
  sealed core before another format needs it.
- Loading the native transcribe.cpp/ggml parser during selection would expose
  untrusted files to native code in the UI process and violate TASK-597.

The importer owns only admission, copying, and descriptor construction.
`ModelArtifactService` remains the sole writer that installs and activates a
managed artifact.

## Components

### `Model_Artifacts/gguf_import.py`

One focused module contains:

- immutable import metadata, plan, progress, and result values;
- typed local-import errors;
- the bounded GGUF reader;
- the pinned transcribe.cpp compatibility declaration;
- source identity checks, disk preflight, copy/hash, curated matching, local
  descriptor construction, installation, activation, and cleanup orchestration.

The module imports no native model runtime and performs no Textual work.

Its public operation is synchronous because Installed runs it in a threaded
worker. It accepts the selected `Path`, `ModelArtifactService`, curated
descriptors, a cancellation event, and an optional progress callback.

### `Model_Artifacts/service.py`

Make one narrow lifecycle addition: recognize service-created top-level
`local-import-*` staging directories. Reconciliation may delete such a stage
only when the global acquisition-session lease is free. A live import holds
that lease from before stage creation through stage cleanup, installation, and
activation.

No descriptor, installed-layout, readiness, activation, or download contract
changes.

### `UI/Screens/model_installed_view.py`

Add **Import GGUF…** beside Refresh and Repair. Selection returns intent; the
view owns the worker and operation lifetime. Parsing, disk probes, copying,
hashing, installation, and activation remain off the Textual event loop.

While importing, the view shows phase/byte progress and a Cancel action. Other
lifecycle actions are disabled. The file picker filter is convenience only;
the importer independently enforces the file contract.

Unmanaged `.gguf` rows change their hint to **Use Import GGUF… to manage this
file**. Other unmanaged formats remain unavailable.

### Shared progress display

Do not pass `AcquisitionProgress` directly into local import. Download progress
requires an `ArtifactRef`, but local import cannot mint its content-derived ref
until the copy hash is complete.

The existing progress widget instead accepts a tiny structural display shape:
phase, optional filename label, bytes done, and bytes total. Existing download
events continue to satisfy it. Local import adds `copy`, `verify`, `install`,
and `activate` phase labels without changing remote acquisition behavior.

## Pinned compatibility declaration

The admission declaration is derived from the explicit registry in
`transcribe.cpp` v0.1.3. Accepted `general.architecture` values are exactly:

- `canary`
- `canary_qwen`
- `cohere_asr`
- `funasr_nano`
- `gigaam`
- `granite_speech`
- `granite_speech_nar`
- `medasr`
- `moonshine`
- `moonshine_streaming`
- `parakeet`
- `qwen3_asr`
- `sensevoice`
- `voxtral`
- `voxtral_realtime`
- `whisper`

`general.architecture` must be a string. `stt.variant` is optional in the
pinned loader; if present it must also be a bounded string. This import gate
claims only that the file declares a family dispatched by the pinned runtime.
It does not claim that family-specific tensors, metadata, tasks, or languages
will load successfully. TASK-604 performs the real managed-handle load and
capability checks.

The declaration also records the five selected-runtime wheel targets:

- Linux x86_64
- Linux aarch64
- Windows x86_64
- macOS arm64
- macOS x86_64

Import code uses portable Python behavior on all five. An imported descriptor
records the current normalized supported OS/architecture pair rather than
flattening this non-Cartesian matrix into inaccurate combinations. An
unsupported current platform fails before staging.

TASK-604 must import and reuse this declaration rather than creating another
architecture or platform list.

## Bounded GGUF inspection

The reader uses `struct` and a seekable binary handle. It never imports ggml,
transcribe.cpp, or a third-party GGUF parser; allocates no tensors; and reads no
tensor payload.

Initial admission accepts GGUF v3 only. Other versions fail explicitly. The
reader validates magic, version, counts, typed key/value metadata, tensor-info
structure, alignment, and that the computed data-section start does not exceed
the regular file size.

Limits are named constants and checked before allocation, multiplication, or
iteration:

| Boundary | Limit |
|---|---:|
| bytes inspected before tensor data | 64 MiB |
| metadata entries | 4,096 |
| tensor entries | 65,536 |
| one string | 1 MiB UTF-8 bytes |
| cumulative metadata string/array payload | 64 MiB |
| one array | 1,000,000 elements |
| nested array depth | 2 |
| tensor dimensions | 4 |

Python integers avoid machine-integer wraparound, but every encoded length is
still compared against the remaining header budget and file size before use.
Unknown GGUF value types, invalid UTF-8 in retained identity fields, duplicate
required keys, truncated values, invalid alignment, or excessive structure
fail closed.

Only bounded display/admission values are retained:

- required `general.architecture`;
- optional `stt.variant`;
- optional `general.name`;
- optional numeric `general.file_type`.

Other well-formed values are skipped under the same budgets. Display strings
are stripped of control characters and length-limited before crossing into UI
state.

## Source-file boundary and TOCTOU handling

The importer applies the project path-validation boundary, then uses `lstat`
without resolving away the final component. It rejects symlinks and anything
other than a regular file.

The worker opens the file once, using no-follow behavior where the platform
provides it, and compares `fstat` with the pre-open identity. That same handle
is retained through parsing and copying. The source snapshot includes device,
inode/file identity, mode, size, modification time, and change time where the
platform exposes them.

After copying, `fstat` must still match the snapshot. A pathname replacement,
truncate, append, metadata mutation, or ordinary in-place write therefore
fails before installation. The staged copy is independently reparsed and its
complete bytes are hashed. Tests inject path replacement and in-place mutation
rather than claiming protection against a privileged attacker that can forge
all filesystem metadata.

The external path and original filename are never written to a manifest,
notification, or log. The staged payload uses the fixed portable name
`model.gguf` unless an exact curated descriptor requires another single-file
path.

## Disk preflight and staging ownership

Before stage creation, required free space is:

`selected file size + 64 MiB safety margin`

Free space is probed at the managed artifact root. It is checked once after
parse and again after acquiring the acquisition-session lease, immediately
before copying. An `ENOSPC` or other copy error remains authoritative even
after a successful probe.

The worker acquires the existing global acquisition-session lease
non-blocking. Contention reports that another model operation is in progress.
The lease is held across:

1. creation of a mode-0700 `local-import-*` operation directory;
2. copy/hash and staged verification;
3. `ModelArtifactService.install(..., consume_source=True)`;
4. exact activation; and
5. operation-owned cleanup.

The payload directory passed to `install()` contains only the one declared
GGUF. Because it lies under the service root, `consume_source=True` moves it
through the existing install stage and immutable promotion without another
model-sized retained copy.

On cancellation or failure, cleanup removes only that operation's proven
service-owned stage. Reconciliation reclaims a crashed `local-import-*` stage
only after the OS has released the acquisition lease. Arbitrary or
unrecognized staging entries remain untouched.

## Descriptor and provenance rules

After the staged copy is hashed, the importer searches the curated registry.
A curated match requires all of the following:

- root artifact role;
- GGUF format and `transcribe-cpp` consumer;
- exactly one payload file;
- no dependencies;
- exact size and SHA-256 match; and
- runtime constraint compatible with the pinned v0.1.3 declaration.

A match reuses the complete curated descriptor, including its source, license,
precision, platform, and provenance. If multiple curated descriptors match the
same bytes, import fails as an ambiguous registry defect.

Every other admitted file receives a deterministic descriptor:

- artifact ID: `local-gguf-<architecture>-<first-16-sha256>`;
- revision: the complete lowercase SHA-256;
- payload path: `model.gguf`;
- consumer/runtime: `transcribe-cpp` / exact `0.1.3`;
- model family: bounded `general.architecture`;
- model label: sanitized `general.name`, otherwise `<architecture> local GGUF`;
- variant/precision: normalized `general.file_type` number, otherwise
  `unknown`—never guessed from the filename;
- provenance: `LOCAL_INTEGRITY_RECORDED` only;
- license: `NOASSERTION`; embedded model metadata is not trusted as a license
  attestation;
- source and license URLs: fixed credential-free `.invalid` sentinels;
- platform: the current normalized pair, after wheel-target admission; and
- no dependencies.

No external path, semantic language role, default-provider setting, or
automatic-routing preference is persisted.

Activation writes only the atomic selector for the unique content-derived
artifact ID. It does not alter `provider=default`, Parakeet/faster-whisper
routing, or any STT configuration. Future automatic routing must continue to
require curated provenance; local-integrity artifacts are manual-only.

## Operation flow and cancellation

The complete flow is:

`Select → validate/open → parse → space check → acquire lease → recheck space →`
`create stage → copy+hash → source recheck → staged parse → descriptor →`
`commit point → install → activate exact ID → cleanup → refresh`

Selection itself is explicit consent, so this single-file local operation does
not add another modal.

Cancellation is observed before stage creation and between copy chunks. It is
also checked after staged verification, immediately before the commit point.
Before the commit point, cancellation removes the operation stage and installs
nothing. Once `install()` begins, Cancel is disabled and finalization runs to a
defined result; interrupting immutable promotion would be less safe than
finishing it.

If installation succeeds but activation fails, the complete verified artifact
remains installed and inactive. The UI reports that activation can be retried
and refreshes inventory. A complete immutable installation is not a partial
artifact and is not deleted as rollback.

Repeated import of identical uncurated bytes resolves to the same reference.
The importer may still copy once to calculate the digest, but core installation
is idempotent and does not create duplicate immutable artifacts.

## Error contract

Typed errors map to fixed user messages without raw exception or path text:

| Category | User outcome |
|---|---|
| unsafe selection | Select a regular, non-symlink GGUF file. |
| malformed or excessive GGUF | This GGUF is invalid or exceeds safe inspection limits. |
| unsupported version | This GGUF version is not supported; version 3 is required. |
| unsupported architecture | This GGUF does not declare a transcribe.cpp 0.1.3 architecture. |
| unsupported platform | transcribe.cpp is unavailable for this platform target. |
| insufficient space | More managed-storage space is required; required and free byte totals are shown. |
| busy | Another model download or import is in progress. |
| source changed | The selected file changed during import; select it again. |
| cancelled | Import cancelled; no model was installed. |
| installation failure | Import failed during managed installation; see the application log. |
| activation failure | The model was imported but could not be activated; retry activation from Installed. |

Logs record the phase and typed category, not the selected path or untrusted
metadata.

## Testing

Tests use small synthesized GGUF byte fixtures. They do not download or execute
a real model.

### Parser tests

- supported architecture with required and optional metadata;
- every pinned architecture spelling;
- bad magic, non-v3 version, truncation at every structural section;
- excessive metadata, tensors, strings, arrays, nesting, dimensions, or total
  header budget;
- wrong metadata types, invalid UTF-8, duplicate required keys, bad alignment,
  and header/data offset beyond EOF;
- unsupported architecture and sanitized display metadata;
- parser imports no native runtime and does not read tensor data.

### Import-service tests

- curated single-file match reuses the exact descriptor;
- unknown compatible file creates deterministic local-integrity provenance;
- ambiguous or ineligible curated matches fail or fall back as specified;
- symlink, directory, irregular file, wrong extension, and path-validation
  rejection;
- pathname replacement and source mutation before/during copy;
- insufficient first/second space probes and copy-time `ENOSPC`;
- cancellation before staging, during copy, and immediately before commit;
- staged reparse failure and corruption detected by core verification;
- cleanup removes only the operation-owned stage;
- live import stage survives reconciliation and abandoned stage is reclaimed;
- duplicate import is idempotent;
- activation failure retains a complete inactive artifact;
- no external path or untrusted license value reaches the manifest/log; and
- no semantic STT default or automatic-routing state changes.

### UI tests

- **Import GGUF…** opens the filtered picker;
- composition and mount perform no filesystem work;
- the import worker runs off the event loop;
- progress and Cancel states follow the pre-commit/commit boundary;
- lifecycle buttons are disabled during import;
- typed errors render without raw path/exception text;
- success and activation-failure copy is precise;
- inventory refreshes after every terminal result; and
- unmanaged GGUF/non-GGUF hints remain distinct.

### Platform gate

Pure parser/import tests run on every available wheel-supported CI target. Path
identity helpers receive focused POSIX and Windows-branch tests with injected
stat/open behavior. Immediate development may collect macOS evidence only;
Windows/Linux qualification remains a preserved release gate rather than an
unsupported claim.

## Acceptance-criteria coverage

| TASK-597 AC | Design coverage |
|---|---|
| #1 explicit regular file, no symlinks/irregular paths | Source-file boundary and UI selection |
| #2 bounded magic/version/metadata/runtime compatibility | Bounded GGUF inspection and pinned declaration |
| #3 preflight, stage, revalidate, hash, atomic activation | Disk/staging, operation flow, core install/activate |
| #4 uncurated local provenance, never automatic | Descriptor and provenance rules |
| #5 cancellation/mutation/failure containment | Cancellation, cleanup, and error contract |
| #6 focused security/lifecycle tests | Parser, importer, UI, and platform test sections |

## Backlog alignment

TASK-597 is a child of TASK-596 and depends directly on the landed foundations
TASK-594 and TASK-595. The task metadata links this design specification.

## References

- [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md)
- [Master STT design](2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md)
- [Model artifact browser design](2026-08-01-task-596-model-artifact-browser-design.md)
- [TASK-596.1 remote discovery design](2026-08-01-task-596-1-remote-model-discovery-design.md)
- [GGUF specification](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [transcribe.cpp v0.1.3 architecture registry](https://github.com/handy-computer/transcribe.cpp/blob/v0.1.3/src/transcribe-arch.cpp)
- [transcribe.cpp v0.1.3 loader](https://github.com/handy-computer/transcribe.cpp/blob/v0.1.3/src/transcribe-loader.cpp)
