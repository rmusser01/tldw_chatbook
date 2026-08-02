# TASK-597 — Direct local GGUF admission: revised design

**Date:** 2026-08-02
**Task:** TASK-597 — Validate explicit local transcribe.cpp GGUF files
**ADR:** `backlog/decisions/040-direct-local-gguf-before-managed-acquisition.md`
**Status:** revised after user approval to make transcription precede managed GGUF storage

## Outcome

Let a user point Chatbook at an existing local GGUF and use that exact file with
the optional transcribe.cpp provider. Do not make the user wait for copying,
artifact registration, curated catalogs, or managed downloads before local
transcription can work.

TASK-597 provides the small, native-runtime-free admission boundary that
TASK-604 will call when the user chooses a file and before request dispatch.
TASK-604 owns the picker, saved configuration, provider, and actual
transcription.

Managed GGUF import/download remains valuable, but it is a later enhancement
after the direct-path transcription loop works end to end.

## Decision

Use a direct local path first.

The selected file remains external and is not copied, hashed in full,
registered as an `ArtifactDescriptor`, activated, or represented as an
installed managed artifact. Chatbook stores the path only in provider
configuration. Transcript provenance, notifications, and logs never persist
the path.

Before request dispatch, the provider reopens and revalidates the current file.
If it is unavailable, became a symlink, or no longer passes
structural/runtime admission, loading fails clearly and the UI offers **Choose
another GGUF…**.

Only `model_path` persists. The source identity snapshot is per-admission and
is not configuration. If another compatible GGUF replaces the file at the same
path between requests, Chatbook treats it as the new current file and recycles
the resident worker model using the new snapshot. The design does not promise
to reject a compatible replacement across admissions or restarts.

This intentionally accepts an external-path dependency. It does not claim the
immutability, independent digest verification, safe deletion, or recovery
properties of the managed artifact store.

## Why the order changes

The previous design completed the artifact-store path before a provider could
transcribe. That optimized infrastructure sequencing rather than the user
outcome.

The direct path reuses the standard filesystem and the selected runtime. It
requires only bounded validation plus provider configuration. Managed copying
and downloading can later wrap the same provider by supplying a managed path;
the provider does not need a second inference implementation.

Rejected for this phase:

- **External artifact registration without copying:** adds lifecycle state but
  still cannot make the external file immutable.
- **Managed-store-first:** stronger guarantees, but delays usable transcription.
- **Native parsing in the UI process:** exposes selection-time UI to native
  parser faults and is unnecessary.

## TASK-597 scope

TASK-597 includes only:

1. A reusable, Textual-free local GGUF admission module.
2. Safe opening of one explicitly selected `.gguf` regular file.
3. A bounded standard-library GGUF v3 structural reader.
4. Exact transcribe.cpp v0.1.3 architecture and wheel-platform admission.
5. A small validated result containing the selected path, bounded metadata,
   file identity snapshot, and normalized platform pair.
6. Typed, path-safe failures and focused cross-platform tests.

TASK-597 excludes:

- artifact descriptors, curated matching, full-file hashing, provenance
  promotion, staging, copying, disk preflight, installation, activation,
  reconciliation, deletion, or repair;
- a model catalog or downloader;
- provider configuration, UI, native import, model load, or transcription;
- semantic default routing or silent fallback.

Any descriptor/store code already prototyped on the TASK-597 branch is removed
before the task merges. The reviewed bounded parser and compatibility
declaration remain.

## Components

### `Model_Artifacts/gguf_import.py`

Keep this existing module name to avoid churn, but reduce it to admission only:

- typed parser/admission errors;
- `GGUFMetadata` and a compact source-identity value;
- bounded `inspect_gguf(handle, file_size=...)`;
- pinned transcribe.cpp v0.1.3 architecture declaration;
- pure wheel-platform normalization/admission; and
- `validate_local_gguf(path)` for the provider boundary.

The module imports no Textual, artifact store, network client, transcribe.cpp,
ggml, or other native model runtime.

`validate_local_gguf` applies the project path validator without resolving away
the final component, uses `lstat`, rejects symlinks and non-regular files, opens
with no-follow behavior where available, compares `fstat`, inspects through the
same handle, and returns a bounded result. It closes the handle before return;
the result is admission evidence, not a lease or immutable guarantee.

### TASK-604 provider/configuration

TASK-604 consumes the admission API and owns the user-visible path:

- **Choose GGUF…** in transcribe.cpp provider settings;
- the same field exposed to the first-run setup wizard when that wizard lands;
- `[transcription.transcribe_cpp] model_path` as the explicit configuration;
- admission when selecting and before request dispatch;
- lazy import of `transcribe.cpp==0.1.3` in the heavy worker;
- one active inference through the app-owned local STT executor; and
- normalized results/provenance without the local path.

The path is never an automatic-routing candidate. transcribe.cpp remains an
exact manual provider. Eligible failures may offer the existing explicit
**Retry with faster-whisper** action; no fallback runs silently.

### TASK-1861: later managed GGUF acquisition

TASK-1861 owns the deferred work:

- representative curated catalog;
- verified managed downloads;
- managed local-file copy/import;
- full-file hashes and descriptor/provenance promotion;
- activation, recovery, deletion, and artifact-browser integration; and
- letting the same TASK-604 provider receive a managed path.

Direct local paths continue to work after managed acquisition lands.

## Data flow

1. User chooses a local `.gguf` in transcribe.cpp settings or the setup wizard.
2. UI calls TASK-597 admission off the Textual event loop.
3. On success, configuration stores the explicit path and shows the bounded
   architecture/model label. No model bytes are copied.
4. On transcription, TASK-604 reruns admission and compares the current
   snapshot with the resident worker model identity.
5. An unchanged snapshot may reuse the resident model; a changed snapshot
   recycles it. The heavy worker receives the current validated path/snapshot.
6. The provider loads that path, transcribes normalized 16 kHz mono audio, and
   returns the provider-neutral STT result.
7. Provenance records provider, architecture/model label, runtime version,
   precision reported by the file/runtime where available, device, language,
   and attempt lineage—but never the local path. `artifact_root` is null and
   `artifact_dependencies` is empty for direct-local execution.

## Pinned compatibility

Accepted `general.architecture` values for transcribe.cpp v0.1.3 remain exactly:

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

Supported wheel pairs remain:

- Linux x86_64
- Linux aarch64
- Windows x86_64
- macOS arm64
- macOS x86_64

Admission means only that the bounded header declares a family dispatched by
the pinned runtime on a wheel-supported platform. The native provider still
owns family-specific load and capability validation.

## Bounded GGUF inspection

The reviewed GGUF v3 parser remains unchanged in intent. It never reads tensor
payload or imports a native parser. It validates structural types, keys, tensor
information, alignment, offsets, retained metadata, and these pre-allocation
limits:

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

Only bounded semantic/display values cross the API. Semantic architecture is
preserved exactly; display labels are separately sanitized and capped.

## Source boundary and honest limitations

Admission records device/inode or platform file identity, mode, size,
modification time, and change time where exposed. It rejects a pathname swap
between `lstat` and `fstat` and parses through the pinned handle. That snapshot
exists only for the current admission and worker-residency decision; it is not
persisted as the expected identity of the configured path.

The provider repeats admission before request dispatch, but the native runtime
ultimately opens a path rather than the validated Python file descriptor. A
concurrent or privileged mutation after revalidation can therefore race native
open. This is an accepted direct-path limitation, not hidden as an immutable
guarantee. A compatible replacement between separate admissions is accepted as
the current configured model, not reported as tampering.

The user explicitly selected this local file. The UI says **Local file** rather
than **Installed** or **Integrity verified**. Managed acquisition later removes
the external-path and mutation limitations for users who choose it.

## Errors

Stable user actions are small and explicit:

| Failure | User-facing action |
|---|---|
| no path configured | **Choose GGUF…** |
| missing/unreadable current file | **Choose another GGUF…** |
| symlink or irregular file | choose a regular local `.gguf` |
| malformed/unsupported GGUF | choose a compatible transcribe.cpp v0.1.3 GGUF |
| unsupported platform | install/use a supported wheel target |
| optional runtime unavailable | install the transcribe.cpp extra |
| native load/capability failure | show exact failure; offer eligible explicit faster-whisper retry |

Raw selected paths and raw native exceptions are not logged, persisted in
provenance, or rendered in generic error copy.

## Direct-local provenance

Direct-local execution has no immutable artifact revision and cannot reproduce
the exact model bytes after the external file changes. It uses the already
nullable provider-neutral provenance contract:

- `artifact_root = null`;
- `artifact_dependencies = ()`;
- `model_id` is a bounded provider/model identity derived from admitted
  architecture and trusted runtime metadata;
- `precision` is the bounded admitted/runtime-reported value where available;
- provider, runtime version, device, language, capabilities, warnings, attempt,
  and retry lineage remain populated; and
- the path and source snapshot are never persisted.

ADR-040 explicitly amends ADR-025's immutable-root expectation for this manual
direct-local provider only. TASK-1861 restores immutable artifact provenance
when the user chooses managed acquisition.

## Tests

TASK-597 tests cover:

- all reviewed parser bounds and truncation cases;
- exact architecture and platform matrices;
- missing file, directory, FIFO/irregular file, final symlink, and pre-open
  replacement;
- same-handle `lstat`/`fstat` identity and typed path-safe errors;
- a successful validated direct-path result; and
- no artifact descriptor/store/native/UI imports.

TASK-604 tests later cover:

- picker/config save and restart round trip;
- admission rerun before request dispatch;
- missing/invalid path recovery and worker recycle when a later admission
  observes a different compatible source snapshot;
- lazy optional import, load, batch transcription, cancellation, worker crash,
  and shutdown;
- no local path in normalized provenance/logs;
- no automatic routing or silent fallback; and
- supported wheel-platform provider smoke.

## ADR check

**ADR required:** yes
**ADR path:** `backlog/decisions/040-direct-local-gguf-before-managed-acquisition.md`
**Reason:** this changes ADR-025's transcribe.cpp runtime input from managed-only
artifact handles to an explicitly configured direct local path for the first
usable release, and defers curated/managed acquisition.

## Completion boundary

TASK-597 is complete when a local GGUF can be safely admitted and described for
the pinned runtime without copying or registering it. It does not claim user
transcription by itself.

The next user-value task is the revised TASK-604 provider/configuration slice.
TASK-1861 managed GGUF acquisition is intentionally last.
