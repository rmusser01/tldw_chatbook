# Guided audio.cpp packages in Model Library

- **Task:** TASK-13207
- **Status:** Approved design and implementation plan; implementation in progress
- **Existing decisions:** ADR-050, ADR-051
- **Upstream artifact host:** `audio-cpp/audio.cpp-gguf` on Hugging Face
- **Pinned artifact commit:** `597048d9a920592808d7d4e2acd7b9c4596a143a`

## Summary

Chatbook will expose reviewed audio.cpp TTS and voice-cloning packages through
the existing Model Library artifact owner. The application catalog is static,
versioned with Chatbook, and pinned to one exact Hugging Face repository commit.
The repository hosts bytes; it does not define Chatbook compatibility.

An explicit install downloads and verifies one self-contained package root. It
does not download `audiocpp_server`, launch audio.cpp, select a TTS model, save
Settings, or alter Studio/profile/character defaults. A successful install
returns an exact artifact identity and managed root to the originating unsaved
Guided Settings draft. Settings then subjects that root to the same scanner and
review flow as a user-selected local package.

Managed packages retain the shared artifact owner's authority throughout their
lifecycle. A staged or live audio.cpp generation holds an artifact lease, so a
removal cannot delete package bytes beneath it. Removal is previewed against all
durable and live consumers and requires explicit resolution or acknowledgement
of every impact.

## Goals

- Support every declared family through the same recipe and catalog machinery,
  without family-specific installer branches.
- Complete the pinned 21-family, 67-package recipe accounting surface.
- Offer a Model Library row only for a package variant with complete official
  source, license, integrity, companion-file, and compatibility evidence.
- Preserve the existing local-package path for exact variants that are not
  present as a complete package in the official audio.cpp repository.
- Reuse the existing acquisition service, managed store, scanner, Settings
  draft ownership, profile repository, and runtime supervisor.
- Keep ordinary catalog rendering, Settings review, and normal CI network-free.

## Non-goals

- Installing, updating, compiling, or selecting an `audiocpp_server` binary.
- Browsing moving Hugging Face `main` at runtime.
- Treating every artifact in the audio.cpp repository as TTS-compatible.
- Inferring compatibility from a filename, repository presence, or successful
  checksum alone.
- Automatically saving Settings, choosing a global/Studio default, creating a
  profile, assigning a character, or launching audio.cpp after installation.
- Deduplicating companion files through symlinks, hardlinks, or a synthetic
  multi-root package view.

## Architecture decisions

### 1. Static pinned artifact manifest

A checked-in manifest declares the downloadable audio.cpp artifacts at
`597048d9a920592808d7d4e2acd7b9c4596a143a`. A small maintainer-only refresh
command accepts an **explicit 40-character Hugging Face commit**, reads that
immutable tree, and emits deterministic reviewable data. It never resolves
`main` on behalf of the running application.

The manifest stores one repository/commit header. Each package record contains
only artifact-source facts that cannot be derived from the recipe:

- recipe ID/revision and package variant;
- a stable artifact ID;
- upstream source path for every file;
- managed relative path for every file;
- byte size and SHA-256 for every file;
- artifact-specific license ID, license source, and usage notice.

Registry construction derives pinned `/resolve/<commit>/...` URLs,
`ArtifactRef.revision`, variant/precision, family, display metadata, runtime
constraint, tasks, and compatibility evidence from the manifest header plus the
exact recipe. Those values are not duplicated in the manifest. The join asserts
that the derived descriptor matches both authorities.

Large Hugging Face LFS files may use the exact LFS SHA-256 and declared size.
Small Git-managed files do not have a content SHA-256 in their Git object ID;
the refresh command downloads those bounded files and hashes their content.
The emitted manifest is then the runtime authority. Normal tests validate it
without contacting Hugging Face.

The repository-level `license: other` value is never projected as a package
license. Each artifact needs a separately reviewed license fact derived from
the exact upstream model/package evidence. An unknown or ambiguous license
keeps the artifact out of Model Library.

### 2. Recipe and artifact admission

The recipe registry remains the compatibility authority. It is expanded so all
67 package entries across the pinned 21-family release inventory are explicitly
classified. The implementation is data-driven: a new family is represented by
recipe data, not a new conditional installer path.

Two independent classifications remain separate:

- **Recipe support:** the existing `approved`, `explicitly_unsupported`, and
  `open_gap` states. Approved recipes enter the matching registry. By task
  completion, no release-accounting row may remain `open_gap`; every variant is
  either approved or carries a reviewed explicit-unsupported reason.
- **Artifact availability:** `downloadable` or `local_only`, evaluated only for
  approved recipes. Downloadable means the pinned official tree contains a
  complete reviewed closure. Local-only means the recipe remains supported for
  explicit local scanning but the pinned official tree cannot supply its exact
  closure.

The combined accounting projection therefore has exactly three public outcomes:
`downloadable` (approved recipe plus admitted artifact), `local_only` (approved
recipe without an admitted artifact), or `explicitly_unsupported`. It cannot
represent `open_gap` at task completion.

At the pinned artifact commit, the initial source-availability boundary is 56
GGUF variants with a named official primary file and 11 variants absent as an
exact package closure. The 11 start as local-only when their recipes are
approved:

- `supertonic_3_safetensors`;
- `pocket_tts_english_safetensors`;
- `chatterbox_safetensors`;
- `omnivoice_safetensors`;
- `qwen3_tts_1_7b_base_safetensors`;
- `qwen3_tts_0_6b_base_safetensors`;
- `voxcpm2_safetensors`;
- `index_tts2_safetensors`;
- `glm_tts_q8_0`;
- `outetts_1_0_1b_q8_0`; and
- `vietneu_tts_v3_turbo_q8_0`.

The 56 named-primary-file variants are not automatically downloadable. The
manifest audit must still prove their full required closure, license, digests,
and compatibility mapping; a failure leaves an approved recipe local-only or
classifies the recipe explicitly unsupported with evidence.

A Model Library row exists only when a pure catalog join proves all of these:

1. the exact recipe is approved;
2. the recipe names the artifact ID;
3. exactly one pinned descriptor supplies that ID and variant;
4. every required recipe file is present in the descriptor at the expected
   managed relative path;
5. every source has an exact size, SHA-256, and pinned URL;
6. license and usage facts are reviewed; and
7. displayed platform/backend compatibility comes from the recipe's exact
   evidence tuple.

Failure of any condition hides the row and fails the generated accounting
test. It does not weaken the corresponding local-package recipe. This is a
per-variant source-evidence decision, not a family limitation.

Only recipes whose task includes TTS or voice cloning enter this catalog.
ASR, diarization, music, source-separation, and unrelated artifacts present in
the same Hugging Face repository are excluded.

### 3. Self-contained managed package roots

Every downloadable recipe variant is one `ArtifactRole.ROOT` descriptor whose
files reproduce the exact directory layout the existing package scanner
expects. The source map may map files from deeper Hugging Face paths into those
recipe-relative managed paths.

Required companion files live in the same managed root as the selected model.
They are not separate artifact dependencies. The current scanner and generated
audio.cpp configuration accept one canonical root; composing multiple managed
roots through links or a generated view would add authority and removal races.
Small shared companions may therefore be duplicated between variants. This is
the deliberate minimal trade-off for an independently verifiable and removable
package root.

The existing `ArtifactFormat` enum is extended only if the approved manifest
contains a downloadable package whose primary format requires it. A local-only
Safetensors recipe does not justify speculative store-format expansion.

### 4. Durable managed artifact identity

`AudioCppAcceptedPackage` gains an optional frozen managed-artifact identity:

```text
(artifact_id, revision, variant)
```

All three values are present together or all are absent. User-selected local
packages keep the field absent. Model Library packages store the exact identity
in addition to the existing canonical-root, root-identity, configuration, and
weight evidence. The Settings format remains backward-compatible: existing
accepted packages decode as local/in-place packages.

The TTS configuration model does not import artifact-store domain classes.
Boundary code converts the frozen persisted value to and from `ArtifactRef`.

## Install and return flow

1. Guided Settings stages an opaque handoff token and its current draft
   generation, then navigates to Model Library. Existing screen-state
   persistence preserves the complete draft.
2. Model Library filters to the reviewed audio.cpp catalog while preserving the
   ordinary Library navigation and keyboard behavior.
3. Selecting Install invokes the existing acquisition `preflight()` and shared
   consent modal. The plan displays the exact commit, files/companions, total
   bytes, checksums, license, and source authority.
4. Confirmation invokes the existing `provision()` with `activate=False`.
   Installation verifies and promotes the exact root but does not create an
   active selector or TTS side effect.
5. Success produces only the exact `ArtifactRef`, managed root, and handoff
   token. The UI says **Installed — ready for review**, never Active or Running.
6. Settings accepts the result only when the handoff token and draft generation
   still match. A stale result remains safely installed but does not mutate a
   newer draft; the user may select it again from Model Library.
7. Settings scans the exact managed root through the existing bounded scanner.
   It requires one exact candidate whose recipe and identities match the
   returned artifact mapping.
8. The accepted candidate is merged into the current draft without replacing
   unrelated draft fields. The user must still review and Save.

Closing Settings or navigating elsewhere does not transfer worker ownership to
a removable widget. The existing durable Model Library/acquisition owner joins
or retains in-flight work, and a completed handoff is consumed at most once.

## Runtime lease flow

A managed artifact becomes active only at an existing deliberate runtime
boundary such as Start, Test, or an explicitly applied staged generation:

1. reconstruct the exact `ArtifactRef` from the accepted package;
2. call the artifact owner to activate and verify that exact root;
3. acquire the exact shared `LeasedArtifactHandle`;
4. resolve the canonical package path from the handle rather than trusting the
   persisted path alone;
5. re-run the existing accepted-package validation against that path; and
6. retain the handle for the complete staged/live generation lifetime.

Discarding a staged generation, definitively shutting down its child, or
failing before publication releases the handle only after owned runtime work is
joined. Local packages continue using the existing path-identity checks and do
not acquire artifact-store leases.

## Removal design

### Preview

Removal begins with an immutable preview bound to the exact artifact reference
and a fingerprint of all displayed evidence. The preview includes:

- saved Guided Settings packages and default model references;
- the current unsaved Guided Settings draft, when mounted;
- TTS profiles whose exact recipe/model dependency resolves through the
  package, including whether they carry clone reference material;
- character assignments reachable through those profiles;
- staged and live runtime generations;
- current artifact leases and other shared owners; and
- the exact installed bytes that would be removed.

The preview exposes bounded IDs, display labels, counts, and actions. Stable
errors and logs do not include complete private paths, reference transcript or
audio data, raw configuration, or collaborator exception graphs.

### Resolution classes

- **Hard blockers:** a staged/live generation or active artifact lease. The
  user must discard the stage, shut down the exact child, or wait for the owner
  to release. Removal never unlinks beneath a live generation.
- **Durable impacts:** saved/draft Settings, profiles, clone references, and
  assignments. The user may cancel, navigate to the owner, or explicitly
  acknowledge **Remove package; keep consumers unavailable**. Acknowledgement
  changes no consumer data.
- **Foreign/shared ownership:** any other artifact-store owner is a hard
  blocker until released. No force-delete path is added.

No resolution silently selects another model, changes a profile recipe, changes
an assignment, clears a global default, or deletes clone reference material.

### Commit

Confirmation carries only the preview fingerprint and explicit resolutions.
The artifact service gains one narrow removal-authority capability. It acquires
the existing locks exactly once in the service's established
`lifecycle -> artifact` order, pins the exact installed target, and exposes only
`commit()` and `close()`. The existing public `delete()` delegates to this same
capability for compatibility; it does not implement a second locking path.

The removal coordinator executes acquisition, revalidation, commit, and release
through one retained worker so authority is never split across widget or
executor lifetimes:

1. acquire the service-owned removal authority in the established lock order;
2. recompute the artifact and consumer snapshot while that exclusive authority
   is held;
3. reject any drift as **Review changed dependencies** and close without
   mutation;
4. call the authority's `commit()` to remove that exact self-contained root
   without reacquiring either lock;
5. re-read installed state and report the converged result; and
6. close the authority on every terminal path, retaining the worker if cleanup
   cannot yet settle.

The authority does not promise named process-owner enumeration from OS locks;
the current lease layer cannot provide it. The preview reports named Chatbook
owners from Settings/profile/runtime state and reports any remaining lease
contention generically as **Another operation is using this package**.

Mutations that save or stage a dependency on an installed managed artifact use
a transient shared artifact lease while validating and publishing that
mutation. This serializes them with the final removal boundary. Operations that
deliberately create an already-missing dependency remain allowed only through
their existing explicit inactive/missing-dependency contracts.

If deletion is interrupted after derived readiness is invalidated, the static
catalog remains authoritative and the managed manifest/tree is classified by
the existing inventory/reconcile path. The UI reports **Removal incomplete —
repair or retry** rather than claiming success. No mutable catalog entry is
created or orphaned.

## User-interface truth

Model Library keeps these dimensions separate:

- **Available to download** — the static catalog has a complete pinned source;
- **Integrity verified** — installed bytes match the manifest;
- **Recipe matched** — the scanner matched the exact recipe revision;
- **Compatibility** — Expected or Verified for the exact displayed
  audio.cpp/platform/backend evidence tuple;
- **Configured** — present in the unsaved or saved Guided Settings model list;
- **Running** — observed only from the runtime supervisor.

No single Active badge collapses these states. Every row states that the model
package does not include `audiocpp_server`. Required companions render as a
count plus a keyboard-expandable relative-file list. Long model names, large
counts, missing data, offline errors, and narrow terminals wrap or scroll
without hiding the primary action or recovery. Disabled destructive controls
carry a readable reason and meet the app's measured disabled-label contrast
floor.

## Error and recovery matrix

| Condition | Public result | Recovery |
| --- | --- | --- |
| Offline, timeout, 404, or source removed | Source unavailable; nothing installed | Retry or choose local package |
| Consent/catalog/source-map drift | Review changed download | Re-run preflight and consent |
| Insufficient space | Exact required/free byte summary | Free space, then retry |
| Checksum or size mismatch | Verification failed; package not installed | Retry from pinned source; staged bytes remain owner-controlled |
| Cancellation before promotion | Cancelled; no installed result | Resume/retry through existing acquisition state |
| Stale Settings return | Installed, not added to changed draft | Select installed package again |
| Scan no longer matches recipe | Review required | Reinstall, rescan, or use local/manual path |
| Removal preview drift | Dependencies changed | Reopen preview |
| Live/staged lease | Package in use | Shut down/discard/wait |
| Interrupted or partial deletion | Removal incomplete | Reconcile, repair, or retry |

Errors are bounded and value-independent. Control-flow exceptions retain their
semantics without carrying private source/configuration data through public
tracebacks.

## Testing and verification

### Hermetic automated tests

- Manifest parser rejects moving revisions, missing SHA-256/size/license data,
  duplicate paths/identities, traversal, non-TTS records, and incomplete file
  closures.
- Generated accounting covers all 21 families and 67 declared package entries;
  every entry is approved, explicitly unsupported, or a reviewed local-only
  variant, with no silent gap.
- Catalog joins reject unknown, duplicate, incompatible, or incomplete recipe
  mappings and never import the network-capable refresh path.
- Fake-source acquisition covers preflight, consent, resume, cancellation,
  source disappearance, checksum failure, exact-root install, and
  `activate=False`.
- Settings round-trip tests mutate unrelated draft fields during navigation and
  prove a valid return merges one package while a stale return changes nothing.
- Scanner tests prove managed layouts and local layouts enter the same review
  path.
- Runtime tests prove activate/acquire happens only at deliberate use, leases
  survive staged/live ownership, and release follows definitive shutdown.
- Removal matrices cover zero/many settings, profiles, clone references,
  assignments, staged/live generations, lease contention, drift, explicit
  acknowledgement, cancellation, and partial-delete recovery.
- UI Pilot tests cover keyboard order, narrow 80x24 rendering, long labels,
  expandable companions, blocked reasons, state-label separation, and no
  server-binary/default/launch side effects.
- Full exception-graph privacy tests use canaries for source paths, config,
  checksums, reference metadata, and collaborator failures.

No normal test contacts Hugging Face or downloads a large artifact. Tests use
small deterministic package fixtures with the same descriptor and scanner
shape.

### Opt-in UAT

With isolated HOME/XDG/config/data/model roots:

1. install one exact pinned official package through the real Model Library;
2. verify exact-root return and unrelated draft preservation;
3. review and Save without launching audio.cpp;
4. deliberately Start/Test with an exact compatible pre-provisioned
   `audiocpp_server`;
5. generate and play a sample;
6. verify removal is blocked while the generation lease is live;
7. shut down, preview durable impacts, acknowledge them, and remove;
8. verify consumers remain unchanged but report the package missing; and
9. verify shutdown leaves no owned worker, handle, child, task, staging entry,
   or generated artifact.

The UAT records pinned public identities, structural audio facts, and user
audibility confirmation without persisting private paths, transcript, audio, or
audio checksums.

## Migration and compatibility

- Existing accepted packages decode with no managed artifact identity and
  retain local/in-place behavior.
- The static artifact catalog is additive and does not mutate installed
  manifests on startup.
- A future catalog revision creates a different exact `ArtifactRef`; it never
  silently reinterprets an installed or accepted package.
- Removing catalog support does not delete installed bytes or rewrite durable
  Settings. The package becomes review-required/local-only until explicitly
  resolved.

## ADR assessment

- **ADR required:** yes, amendment only
- **ADR path:** `backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md`
- **Reason:** ADR-050 already chooses static recipes, shared Model Library
  ownership, exact package identity, and separation from server-binary ownership.
  TASK-13207 specifies the previously deferred pinned-source, managed-identity,
  lease-lifetime, and dependency-aware removal details. A second ADR would split
  one ownership decision across two authorities.

## Alternatives rejected

| Alternative | Why rejected |
| --- | --- |
| Runtime browsing of Hugging Face `main` | Makes available packages and source bytes change outside the installed Chatbook version. |
| Treat the official repository as compatibility authority | Repository presence proves neither exact runtime support nor the user's binary/platform tuple. |
| Use repository-level `license: other` for every row | Hides materially different package licenses and usage obligations. |
| Automatically activate after install | Conflates downloaded state with deliberate runtime selection and weakens no-default-change semantics. |
| Store only a managed filesystem path | Loses exact artifact ownership and cannot acquire the lease needed to block removal. |
| Separate companion artifacts composed through links | The scanner and generated config require one canonical package root; a composed view adds unnecessary namespace authority. |
| Force-delete leased artifacts | Can disrupt staged/live generations and violates the shared artifact owner's contract. |
| Rewrite affected profiles/assignments on removal | Silently changes durable user intent and risks deleting private clone reference assets. |
