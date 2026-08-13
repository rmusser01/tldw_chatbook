# TASK-2062 — Managed and external GGUF sources

**Original date:** 2026-08-03
**Revised:** 2026-08-12
**Task:** TASK-2062
**Child tasks:** TASK-2062.1, TASK-2062.2, TASK-2062.3
**Parent spec:** `Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md`
**ADR:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Status:** approved section-by-section on 2026-08-12

This revision replaces the earlier TASK-2062 design. The earlier version used
filename-dependent identities, retained source paths in managed metadata, covered
providers that do not consume GGUF files, and did not retain managed leases through
stubborn server-process lifetimes. Those choices are superseded here.

## Outcome

Chatbook will make managed GGUF ownership optional rather than mandatory:

- A user can copy a local GGUF into Chatbook's managed artifact store. The managed
  copy receives locally recorded integrity, immutable content identity, safe
  activation, deletion protection, and recovery states.
- llama.cpp can launch either an exact managed GGUF or an arbitrary external GGUF
  outside the store.
- llamafile retains its embedded-model mode and can additionally launch an exact
  managed GGUF or an arbitrary external GGUF.
- vLLM and MLX remain unchanged. They accept Hugging Face model identifiers or model
  directories and are not forced through a GGUF-only artifact contract.
- Obsolete direct-write Models downloaders are removed only after Import and External
  source flows are usable.

The original local file remains user-owned. Chatbook never writes, renames, deletes,
or globally selects it merely because it was imported or used externally.

## Workstream and merge order

The implementation is split into three separately reviewed child PRs. They merge in
this order:

1. **TASK-2062.1 — Managed local GGUF import.** Add the generic import boundary and
   Installed-view UX.
2. **TASK-2062.2 — Runtime source modes.** Add managed/external llama.cpp sources and
   embedded/managed/external llamafile sources. This depends on TASK-2062.1.
3. **TASK-2062.3 — Legacy downloader removal.** Remove obsolete direct-write Models
   download surfaces after the replacement is available. This depends on both prior
   child tasks.

The PRs are independently reviewable but intentionally ordered. The downloader must
not disappear before the replacement paths work.

## Trust and authority vocabulary

The UI and persisted records use these exact meanings:

- **Managed · local integrity recorded:** Chatbook owns a copy, recorded its full
  digest locally, and verifies those owned bytes. This is not an upstream-publisher
  integrity claim and is not a runtime-compatibility claim.
- **External · integrity unknown:** Chatbook launches a user-owned file in place. It
  does not copy, delete, globally select, or continuously attest that file.
- **Embedded:** The llamafile executable contains its own model and receives no
  separate `-m` model argument.
- **Ready:** A managed artifact is structurally valid, installed, digest-verified,
  and activated in the managed store. Ready does not mean a particular runtime has
  proven support for its architecture or quantization.

## TASK-2062.1 — Managed local GGUF import

### Generic GGUF inspection

The existing `gguf_admission` reader combines bounded GGUF-v3 structural inspection
with the transcribe.cpp architecture allowlist. TASK-2062.1 separates those policies:

- The generic inspector validates the magic, version, bounded metadata and tensor
  tables, alignment, offsets, retained metadata types, and required
  `general.architecture` field without reading tensor payloads.
- The existing transcribe.cpp admission path continues to apply its architecture
  allowlist after generic inspection.
- Managed LLM import uses generic inspection only. It must not claim that the
  installed llama.cpp or llamafile build supports the imported architecture or
  quantization.

No new parser dependency or native runtime import is introduced.

### Local descriptor accommodation

`ArtifactDescriptor` currently requires credential-free HTTP(S) `source_url` and
`license_url` values. A local import has no truthful remote source or reviewed
license URL. The sealed descriptor boundary receives the narrowest accommodation:

- `source_url == ""` is permitted only when provenance is exactly
  `LOCAL_INTEGRITY_RECORDED`.
- `license_url == ""` is permitted only for that same local provenance and
  `license_id == "unknown"`.
- Remote, curated, and independently verified descriptors retain their existing
  non-empty HTTP(S) URL requirements.
- Empty local sources are never eligible for acquisition or download.

The design forbids `file://` URLs, fabricated `local.invalid` URLs, and absolute
source paths in descriptors, manifests, usage notices, logs, configuration, worker
descriptions, or stable operation state.

### Content identity

Every canonical identity field is deterministic from the staged bytes:

- The reference revision contains the full lowercase SHA-256, prefixed by
  `sha256-`.
- The artifact id uses a bounded digest-derived local-GGUF namespace. The full
  revision remains the exact immutable identity.
- The variant is derived from bounded GGUF metadata such as `general.file_type`,
  with a stable `imported` fallback.
- The display `model_id` uses bounded `general.name` metadata when present and a
  digest-derived label otherwise.
- The managed payload has the fixed portable name `model.gguf`.

The original basename is never identity. Identical bytes imported under different
filenames converge on the same exact `ArtifactRef`; changed bytes create a different
revision. The first committed canonical descriptor wins for an identical concurrent
import.

### Import transaction

The artifact service owns one narrow single-file import operation. It reuses the
existing lifecycle lease, per-artifact lease, operation-owned staging, staged digest
verification, manifest write, atomic promotion, readiness activation, reconciliation,
and crash-cleanup primitives. It does not create a second marker or garbage-collection
framework and does not wrap the input in a second temporary source directory.

The flow is:

```text
user selects a local GGUF
  -> lstat the selected node
  -> reject symlinks/reparse points, directories, and special files
  -> open once without following links where the platform supports it
  -> compare lstat/open-fstat identity
  -> stream once into operation-owned staging/model.gguf while hashing
  -> recheck the open source descriptor identity and mutation fields
  -> structurally inspect staging/model.gguf
  -> verify the staged size and full SHA-256
  -> derive the path-private descriptor from staged bytes
  -> lock the exact ArtifactRef
  -> converge with an identical installed destination or promote atomically
  -> activate readiness
  -> return the exact ArtifactRef without changing runtime preference
```

The source descriptor includes device/inode or the supported platform equivalent,
node type, size, modification time, and change time where available. Chatbook promises
that it does not write, rename, or delete the source. It does not promise that another
program cannot mutate it, and it does not claim that filesystem access time is
unchanged by a read.

The source is read once. The staged copy is authoritative: structural inspection and
the digest used by the managed descriptor are both checked against store-owned staged
bytes before promotion.

### Cancellation and commit point

Import is cancellable during open, copy, staged inspection, and verification. A
physical Cancel action remains stable and focusable while cancellation is valid.

Atomic promotion is the point of no return:

- Before promotion, cancellation or failure removes only the operation's private
  staging and releases its leases.
- When final promotion begins, the UI changes to **Finalizing** and disables Cancel.
- After promotion, cancellation does not delete the newly shared immutable artifact.
- Activation failure leaves the bytes installed and exposes an Activate recovery
  action.
- A cancelled concurrent importer never removes an artifact published by another
  importer.

Successful import activates readiness but never changes llama.cpp or llamafile source
mode, managed selection, or external selection.

### Import UI

The Installed view exposes a stable **Import GGUF…** action and an Import action on
eligible unmanaged GGUF rows. The confirmation surface shows:

- the user-selected filename and size;
- that a managed copy will be created;
- that Chatbook will leave the original in place;
- that license and runtime compatibility are not verified.

The absolute path may appear transiently in this user-owned selection surface. It is
not copied into persistent state or error output.

Progress updates an existing mounted status widget in place. Recomposition is limited
to state transitions that add or remove controls so repeated byte progress cannot
destroy keyboard focus. Terminal outcomes are Imported and ready, Already imported,
Installed — activation required, Cancelled, or a stable path-private failure with
Retry and Choose another file recovery.

## TASK-2062.2 — Runtime source modes

### Source state

llama.cpp exposes two mutually exclusive modes:

```text
Source: [ Managed GGUF | External GGUF ]
```

llamafile exposes three mutually exclusive modes:

```text
Source: [ Embedded | Managed GGUF | External GGUF ]
```

Mode, managed `ArtifactRef`, external path, and inactive-mode selections belong to the
screen/controller rather than recomposed widgets. Switching modes preserves inactive
values in memory, but only the active mode participates in validation and launch. No
new durable absolute-path setting is introduced.

Compatibility mapping is deterministic:

- an existing llama.cpp model path initializes External mode;
- an existing llamafile model path initializes External mode;
- llamafile with no model path initializes Embedded mode;
- existing values continue to behave as before rather than being silently imported;
- vLLM and MLX configuration and launch behavior remain unchanged.

### Managed launch

The managed selector shows a display label, quantization/variant, size, and local
integrity provenance. It stores the exact `ArtifactRef`, never a managed filesystem
path. The worker resolves the payload only after acquiring that exact reference.

Managed lease ownership follows one explicit handoff:

```text
launch worker acquires and owns exact artifact handle
  -> current server claim accepts ownership atomically
  -> process spawns and is published under that exact claim
  -> claim retains the handle for the entire live or stubborn process lifetime
  -> cleanup detaches the handle under the claim-identity lock
  -> cleanup closes the handle outside the lock, exactly once
```

If reservation, ownership transfer, or spawn fails, the worker retains responsibility
and closes the handle. After transfer, only identity-checked cleanup for that exact
claim may close it. A stale callback cannot close another generation's handle.

The existing `ServerLaunchClaim` receives one optional closable resource rather than a
second lease manager. Normal exit, failed exit, successful Stop, cancelled pre-spawn,
failed publication, and retained stubborn processes all use the existing lifecycle
owner. A process that cannot be proven dead keeps both its claim and lease. Deletion
performs its authoritative in-use recheck under the artifact service's own lock.

### External launch

External mode remains a permanent first-class escape hatch. A user may select any
regular GGUF outside the managed store without importing it.

The external validation worker:

- applies the same cross-platform no-follow/reparse-point and regular-file boundary;
- performs bounded generic GGUF structural inspection off the Textual loop;
- rechecks path identity immediately before spawn;
- does not hash a multi-gigabyte file on every launch;
- never acquires, activates, deletes, or writes managed-store state.

Because the external runtime ultimately reopens a pathname, this is an honest
best-effort validation rather than an immutability claim. The UI labels the authority:

> Outside Chatbook · integrity unknown
>
> This file is used in place and is not imported, copied, deleted, or selected
> globally.

External files never receive a Delete action from Chatbook.

### Embedded llamafile

Embedded mode launches the selected llamafile executable without a separate model
argument. This resolves the current UI/handler contradiction in which the UI describes
the model path as optional while the handler requires one.

### Running state

When a launch claim is pending or a server is running:

- Launch is disabled immediately to prevent duplicates;
- source mode and source-selection controls are disabled;
- the surface shows the exact active authority: Embedded, Managed, or External;
- Stop controls only the current process claim;
- changes become available only after process death is confirmed.

Import, source validation, activation, and server lifecycle use separate operation
lanes. Changing an unrelated setting does not silently cancel a copy. Starting a
replacement import cancels only the prior import; replacing validation cancels only
validation; Stop owns only the active server claim.

## Errors, privacy, and recovery

Filesystem, descriptor, lease, and subprocess exceptions are converted immediately
to stable path-private error codes and sanitized user copy. Exception context is
suppressed at the UI boundary. Worker descriptions are static and path-free. Logs may
record operation, provider, stable code, and exit status, but not raw commands,
absolute paths, stderr, or exception strings.

Runtime stderr is not parsed for classification because it is unstable and may echo
the source path. A runtime load failure maps to:

> The runtime could not load this GGUF. Check that its architecture and quantization
> are supported.

Only managed digest verification can label an artifact corrupt.

Required recovery includes:

- managed missing: choose another model or import again;
- managed corrupt: delete and import again;
- installed but not active: Activate;
- lease contention: retryable busy state;
- external missing/unreadable: preserve selection and Browse;
- external changed between validation and launch: require retry;
- malformed GGUF: reject before spawn;
- runtime incompatibility: preserve source and report a runtime failure;
- managed-store unavailable: keep External and llamafile Embedded usable.

Long names wrap or truncate without moving actions outside the Models shell. All
controls and recovery actions remain visible, painted, and keyboard-reachable at 80
columns. Progress never replaces the focused Cancel or Stop control.

## TASK-2062.3 — Legacy downloader removal

Only after TASK-2062.1 and TASK-2062.2 are usable, remove:

- `Widgets/HuggingFace` and its Models-browser wiring;
- the Download Models rail destination;
- the Transformers **Download New Model** controls;
- the `huggingface-cli download --local-dir` worker and dispatch path;
- an obsolete browser client module only when a final reference audit proves it dead.

Preserve:

- the separate Hugging Face inference provider and its configuration;
- provider-owned model-ID caching used by vLLM, MLX, or Transformers;
- Hugging Face model identifiers accepted by existing runtimes;
- arbitrary external model directories;
- legacy directory scanning and discovery of previously downloaded unmanaged GGUFs;
- `model_download_dir` as a legacy scan root, without describing it as a current
  managed-download destination.

Dead-path checks are scoped to the Models UI and its obsolete direct-write handler.
They must not impose a global ban on legitimate runtime-owned Hugging Face caching.

## Cross-platform safety

Local file admission uses an `lstat -> open -> fstat` identity comparison. POSIX uses
`O_NOFOLLOW` where available. Windows rejects symlinks and reparse points through its
supported stat/path boundary and performs the same post-open identity comparison.
Directories and non-regular files fail closed.

Native Linux, macOS, and Windows tests cover path replacement, symlink/reparse-point
rejection, cancellation, promotion cleanup, launch-claim ownership, and process
termination. Small synthetic GGUF-v3 fixtures and bounded helper subprocesses provide
the evidence; tests do not require a real llama runtime or a multi-gigabyte model.

## Verification strategy

Each child PR starts with focused failing tests and includes mutation evidence for its
load-bearing boundaries.

### TASK-2062.1

- source is never opened for write, renamed, or deleted;
- staged bytes, not source-path claims, are authoritatively inspected and verified;
- filename-derived identity mutation fails rename-idempotence tests;
- full-digest identity, same-content convergence, and changed-content revisions;
- malicious metadata bounds, malformed/truncated structures, and special files;
- cancellation before promotion, point-of-no-return Finalizing, activation failure,
  disk-full/permission failure, crash reconcile, and concurrent import cleanup;
- descriptor serialization proves no source path or fabricated URL persists;
- mounted production-CSS import, progress, Cancel, focus, and 80-column geometry.

### TASK-2062.2

- exact llama.cpp and llamafile source-mode matrices;
- arbitrary external GGUF launch remains reachable and download-free;
- managed selectors retain refs, not paths;
- lease ownership through pre-spawn cancellation, spawn failure, normal/failed exit,
  Stop, stubborn retained process, deletion race, and stale claim cleanup;
- a mutation that releases when the worker returns fails the stubborn-process test;
- inactive source values never enter launch;
- external validation is off-loop and store-free;
- existing source values map compatibly; vLLM and MLX regression nodes remain green;
- mounted production-CSS active-authority, physical Stop, focus, and 80-column tests.

### TASK-2062.3

- removed imports, actions, rail keys, handlers, subprocess construction, and dead
  browser references;
- a mutation leaving either direct downloader callable fails;
- legacy unmanaged scanning and import remain reachable;
- Hugging Face inference, model IDs, external directories, vLLM, and MLX regressions;
- empty-state and action-census mounted tests.

Every PR also runs scoped Ruff, changed-range formatting, compilation, diff checks,
added-line path/privacy scans, static worker-description/log scans, and exact affected
test files. The final parent gate runs the union of artifact-service, inventory,
Models UI, local-server lifecycle, llama.cpp, llamafile, vLLM, MLX, and downloader
retirement tests.

## Security and ownership invariants

1. Managed descriptors never persist the original local path.
2. Empty descriptor URLs are legal only for exact local-integrity provenance and are
   never fetchable.
3. Managed identity depends only on staged bytes and bounded metadata contained in
   those bytes.
4. The service is the sole writer under the managed artifact root.
5. A cancellable operation removes only its own staging.
6. Promotion is immutable and is never rolled back by a stale UI callback.
7. A managed runtime lease outlives the process it protects, never the reverse.
8. External files stay user-owned and cannot be deleted by Chatbook.
9. Raw paths, commands, stderr, and exception strings do not enter logs or notices.
10. Removing a Models downloader does not remove runtime-owned model-ID resolution.

## Non-goals

- No automatic import or migration of existing external GGUFs.
- No requirement that users move GGUF files into the managed store.
- No managed pickers for vLLM or MLX in this task.
- No GGUF runtime-compatibility database or automatic recommendation engine.
- No remote GGUF catalog or downloader replacement beyond the existing managed
  acquisition/catalog surfaces.
- No persistence of new external absolute-path settings.
- No new GGUF parser dependency.
- No automatic deletion of legacy caches or original files.
- No interpretation of runtime stderr as a stable machine-readable protocol.

## Rollback

- TASK-2062.1 can be disabled by hiding Import while leaving already installed
  immutable artifacts readable and deletable through the service.
- TASK-2062.2 can restore the prior external-path presentation without changing or
  deleting managed artifacts. It must not silently convert managed selection into an
  external managed-store path.
- TASK-2062.3 must not merge before the replacement paths. If the new flows regress,
  keep new direct downloads disabled rather than restoring unverified writes; users
  retain External and existing unmanaged files.
- No rollback deletes original files, external caches, or immutable managed records.

## Rejected alternatives

| Alternative | Why rejected |
| --- | --- |
| Force every GGUF into the store | Removes the explicitly required external-file escape hatch and needlessly duplicates very large user-owned models. |
| Store the original path as `file://` provenance | Persists private local information and falsely treats a mutable pathname as artifact provenance. |
| Fabricate an HTTP URL for local imports | Makes descriptor validation pass by lying about the source. |
| Use filename as artifact identity | Renaming identical bytes creates duplicate managed identities. |
| Use a short digest as the immutable revision | Weakens an exact content-identity contract for no material benefit. |
| Reuse the transcribe.cpp architecture allowlist for LLM import | Rejects valid LLM GGUF families based on a speech-runtime policy. |
| Build a second import staging/GC system | Duplicates service-owned locking, staging, reconcile, verification, and promotion machinery. |
| Resolve managed selection to a path in the UI | Loses lease ownership and exposes internal storage layout. |
| Release the lease when the launch worker returns | A retained stubborn process could outlive the bytes it is using. |
| Give llamafile only Managed/External modes | Erases its valid embedded-model execution mode. |
| Add managed pickers for vLLM and MLX | Changes unrelated HF-ID/directory providers and incorrectly treats them as GGUF-only. |
| Remove every Hugging Face cache/download call globally | Breaks legitimate provider-owned model-ID behavior outside the obsolete Models downloader. |

## Related work

- `TASK-596` delivered the Curated, Remote, and Installed browser foundation.
- `TASK-1915` may reuse the generic import boundary for managed transcribe.cpp GGUFs,
  but must continue applying its speech-specific admission policy and must not activate
  the deferred prototype wholesale.
- ADR-025 is amended by this design to extend shared artifact ownership to local LLM
  GGUFs while preserving explicit external authority.
