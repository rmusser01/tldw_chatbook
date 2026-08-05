# TASK-597 — Direct local GGUF admission: revised design

**Date:** 2026-08-02
**Task:** TASK-597 — Validate explicit local transcribe.cpp GGUF files
**ADR:** `backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md`
**Status:** revised after user approval to make transcription precede managed GGUF storage

## Outcome

Let a user point Chatbook at an existing local GGUF and use that exact file with
the optional transcribe.cpp provider. Do not make the user wait for copying,
artifact registration, curated catalogs, or managed downloads before local
transcription can work.

TASK-597 provides the small, native-runtime-free admission boundary that
TASK-604 will call when the user chooses a file and again inside the existing
spawn-isolated ingestion worker immediately before native model load. TASK-604
owns the picker, saved configuration, Library batch selector, production
wiring, provider, and actual transcription.

Managed GGUF import/download remains valuable, but it is a later enhancement
after the direct-path transcription loop works end to end.

## Decision

Use a direct local path first.

The selected file remains external and is not copied, hashed in full,
registered as an `ArtifactDescriptor`, activated, or represented as an
installed managed artifact. Chatbook stores the path only in provider
configuration. Transcript provenance, notifications, and logs never persist
the path.

Immediately before native model load, the ingestion worker reopens and
revalidates the current file. If it is unavailable, became a symlink, or no
longer passes structural/runtime admission, loading fails clearly and the UI
offers **Choose another GGUF…**.

Only `model_path` persists. The source identity snapshot is per-admission and
is not configuration. If another compatible GGUF replaces the file at the same
path between requests, Chatbook treats it as the new current file. The first
release loads once per batch-ingest job and has no resident-model identity to
recycle. The design does not promise to reject a compatible replacement across
admissions or restarts.

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

Any descriptor/store code already prototyped on the TASK-597 branch is retained
only as a private deferred reference module for TASK-1915. It is not exported,
registered, imported, called, or tested as active TASK-597 behavior. The
reviewed bounded parser and compatibility declaration remain in the active
admission module.

## Components

### `Model_Artifacts/gguf_admission.py`

Rename the unmerged prototype now so later managed import can use the word
`import` without colliding with this admission-only boundary:

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

### `Model_Artifacts/_deferred_gguf_managed_import.py`

The already-written store-facing descriptor prototype is preserved here for
TASK-1915 instead of being deleted. This file is deliberately dead in the
current release:

- its module docstring names TASK-1915 and says it is deferred reference code;
- `__all__` is empty;
- `Model_Artifacts/__init__.py` does not export it;
- no production module imports it and it has no registration or call site;
- TASK-597 does not use it for validation, descriptors, hashing, copying,
  activation, or routing; and
- it may import the active admission metadata/compatibility values rather than
  duplicate the bounded parser.

The file is retained source, not a supported API or a partially enabled managed
feature. TASK-1915 must review and test it against its then-current artifact
contracts before activation; preservation does not pre-approve the prototype.

### TASK-604 provider/configuration

TASK-604 consumes the admission API and owns the user-visible path:

- **Choose GGUF…** in transcribe.cpp provider settings;
- the same field exposed to the first-run setup wizard when that wizard lands;
- `[transcription.transcribe_cpp] model_path` as the explicit configuration;
- key-only atomic config persistence so saving the path does not log its value;
- a `transcribe-cpp` choice in the real Library audio/video ingest form;
- admission when selecting and inside the ingestion worker immediately before
  native model load;
- lazy import of `transcribe.cpp==0.1.3` in that worker;
- one native load per job, followed by authoritative capability discovery,
  construction of the exact per-job declaration and sealed registry, and
  coordinator verification before inference;
- one active inference through the existing spawn-isolated batch parse pool's
  heavy-lane cap; and
- normalized results/provenance without the local path.

The first usable release deliberately does not depend on TASK-601. It does not
keep a resident transcribe.cpp model or add fine-grained cross-process
cancellation. Existing app shutdown still terminates the parse pool, and the
pool monitor contains a worker crash by recycling that pool generation. Such a
crash can make other in-flight parse jobs retryable; TASK-601 later provides a
dedicated resident executor, stronger heavy/light isolation, and cooperative
per-request cancellation when those benefits justify the extra machinery.

The path is never an automatic-routing candidate. transcribe.cpp remains an
exact manual provider. Eligible failures may offer the existing explicit
**Retry with faster-whisper** action; no fallback runs silently.

### TASK-1915: later managed GGUF acquisition

TASK-1915 owns the deferred work:

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
3. On success, key-only configuration persistence stores the explicit path and
   shows the bounded architecture/model label. No model bytes are copied and
   no config log contains the path value.
4. The real Library audio/video form offers `transcribe-cpp`; submission carries
   that exact manual provider through the production batch-ingestion path.
5. Inside the existing spawn worker, TASK-604 reruns admission against the
   current path, lazily imports the pinned runtime, loads the model once, and
   reads authoritative native capabilities.
6. The worker constructs the exact per-job adapter declaration and sealed
   registry from those observed capabilities. Coordinator preflight returns
   the already-loaded observation, verifies exact equality with the declaration,
   rejects incompatible requests before inference, and reuses that model for
   the immediately following transcription. The adapter transcribes normalized
   16 kHz mono audio and closes the model at job end.
7. The existing parent-side writer persists transcript content and normalized
   provenance atomically.
8. A failed worker returns a bounded, path-safe STT failure/action envelope by
   extending the existing `error_detail` payload. The parent job record and
   Library failure UI preserve its stable failure code and only the eligible
   `choose_another_gguf` and `retry_faster_whisper` actions.
9. Provenance records provider, architecture/model identity, precision reported
   by the file/runtime where available, device, language, and attempt
   lineage—but never the local path. `artifact_root` is null and
   `artifact_dependencies` is empty for direct-local execution. The exact
   provider package remains pinned, but v1 does not add a runtime-version field
   to the persisted provenance schema.

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

OS/CPU wheel candidate pairs remain:

- Linux x86_64
- Linux aarch64
- Windows x86_64
- macOS arm64
- macOS x86_64

The pair check is necessary but not sufficient for package availability.
Pinned Linux wheels carry manylinux 2.27/2.28 tags, pinned macOS wheels carry a
macOS 11.0 floor, and unsupported ABIs such as musl may share one of the pairs
above without accepting a released wheel. TASK-597 therefore reports only the
normalized candidate pair. TASK-604's lazy runtime import/provider probe owns
the authoritative wheel/ABI availability result and clear recovery copy.

Admission means only that the bounded header declares a family dispatched by
the pinned runtime and the OS/CPU pair has a released native-wheel lane. The
native provider still owns final package availability, family-specific load,
and capability validation.

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
exists only as evidence for the current admission. It is not persisted as the
expected identity of the configured path; TASK-601 may later reuse it for a
resident-model decision.

The worker repeats admission immediately before model load, but the native
runtime ultimately opens a path rather than the validated Python file
descriptor. A concurrent or privileged mutation after revalidation can
therefore race native open. This is an accepted direct-path limitation, not
hidden as an immutable guarantee. A compatible replacement between separate
admissions is accepted as the current configured model, not reported as
tampering.

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
| native load/capability failure | show a sanitized specific failure and offer eligible explicit faster-whisper retry |

Raw selected paths and raw native exceptions are not logged, persisted in
provenance, or rendered in generic error copy. The admission result's path is
excluded from its representation, provider settings use the key-only atomic
config writer rather than the value-logging single-setting helper, and native
exceptions/log callbacks are sanitized at the adapter boundary.

Worker failures extend the existing bounded, picklable `error_detail` payload
with a stable STT failure code and an allowlisted action list. The parent job
record carries that envelope to the Library failure UI. It does not serialize
provider exceptions or introduce a second IPC/error transport.

## Direct-local provenance

Direct-local execution has no immutable artifact revision and cannot reproduce
the exact model bytes after the external file changes. It uses the already
nullable provider-neutral provenance contract:

- `artifact_root = null`;
- `artifact_dependencies = ()`;
- `model_id` is the deterministic bounded identity
  `local-gguf:<allowlisted-architecture>`;
- `precision` is the bounded admitted/runtime-reported value where available;
- provider, device, language, capabilities, warnings, attempt, and retry
  lineage remain populated; and
- the path and source snapshot are never persisted.

ADR-041 explicitly amends ADR-025's immutable-root expectation for this manual
direct-local provider only. TASK-1915 restores immutable artifact provenance
when the user chooses managed acquisition.

## Tests

TASK-597 tests cover:

- all reviewed parser bounds and truncation cases;
- exact architecture and platform matrices;
- missing file, directory, FIFO/irregular file, final symlink, and pre-open
  replacement;
- same-handle `lstat`/`fstat` identity and typed path-safe errors;
- a successful validated direct-path result; and
- path exclusion from result representations and errors; and
- no artifact descriptor/store/native/UI imports in the active admission
  module; and
- the deferred prototype is private, unexported, and has no active production
  import or call site.

TASK-604 tests later cover:

- picker/config save and restart round trip through key-only persistence;
- the actual Library selector, submission, spawn-worker provider call, parent
  writer, and persisted transcript/provenance path;
- admission rerun in the spawn worker immediately before native model load,
  capability-derived declaration/registry sealing after that single load, and
  exact coordinator equality verification before inference;
- missing/invalid path recovery and acceptance of a later compatible
  replacement at the same configured path;
- lazy optional import, final ABI availability, load, post-load capability
  validation, batch transcription, worker crash containment, and shutdown;
- bounded typed worker failure/action propagation through the existing
  `error_detail` payload to the Library failure UI;
- no local path in config logs, native logs/exceptions, normalized provenance,
  result representations, or generic error copy;
- no automatic routing or silent fallback; and
- supported wheel package-resolution/provider smoke, including Linux ABI
  coverage.

## ADR check

**ADR required:** yes
**ADR path:** `backlog/decisions/041-direct-local-gguf-before-managed-acquisition.md`
**Reason:** this changes ADR-025's transcribe.cpp runtime input from managed-only
artifact handles to an explicitly configured direct local path for the first
usable release, and defers curated/managed acquisition.

## Completion boundary

TASK-597 is complete when a local GGUF can be safely admitted and described for
the pinned runtime without copying or registering it. It does not claim user
transcription by itself. Preserving the private TASK-1915 prototype does not
make managed GGUF import part of TASK-597 or an active application capability.

The delivery order is TASK-597 admission, then TASK-604 usable Library batch
transcription through the existing isolated worker. TASK-601 is later executor
hardening/residency and is not a prerequisite for first use. TASK-1915 managed
GGUF acquisition is intentionally last.
