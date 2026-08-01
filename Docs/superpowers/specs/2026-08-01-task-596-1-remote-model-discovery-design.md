# TASK-596.1 — Bounded Hugging Face GGUF discovery: design

**Date:** 2026-08-01
**Task:** TASK-596.1 — Add bounded Hugging Face GGUF discovery to the managed model browser
**Parent:** TASK-596 — Renovate the local model artifact browser
**Depends on:** TASK-595
**ADR:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Status:** approved section-by-section on 2026-08-01

## Outcome

Add a **Remote** row to Lab → Models that lets a user explicitly search
Hugging Face, select one LFS-backed GGUF file or complete standard GGUF shard
set, and download it through the existing managed-model preflight and
acquisition flow.

The downloaded model is intentionally inert. Chatbook records and inventories
the exact bytes, but does not claim that an arbitrary GGUF is compatible with a
runtime, ready for transcription, or eligible for automatic routing.

## Existing foundation

TASK-596 Phase 1 already provides:

- Curated and Installed model views in Lab;
- `ArtifactDescriptor`, `ArtifactCatalog`, and `ArtifactSourceMap` contracts;
- consent-driven `ArtifactAcquisitionService.preflight()` and `provision()`;
- resumable bounded fetches with per-hop egress checks and cross-origin
  credential stripping;
- shared plan, consent, progress, inventory, activation, and deletion controls;
- lazy view activation through `LLMManagementWindow._start_view_work()`.

Remote discovery is an adapter and UI over that foundation. It does not
renovate or reuse the legacy `Widgets/HuggingFace/` downloader, which continues
to exist behind **Download Models** until its separately approved retirement.

## Scope

This slice includes:

1. Explicit Hugging Face model search and exact `owner/repository` resolution.
2. Immutable repository resolution to a commit SHA.
3. Bounded discovery of LFS-backed GGUF files and standard complete shard sets.
4. Conversion of one selected candidate into an in-memory catalog and source
   map accepted by the managed acquisition service.
5. Managed preflight, consent, download, digest verification, installation,
   progress, and Installed-view refresh.
6. Existing configured Hugging Face credential support for gated and private
   repositories.

## Non-goals

- Making arbitrary GGUF files runnable or assigning a runtime.
- Adding arbitrary remote models to STT routing or provider selectors.
- Inspecting GGUF contents or inferring architecture, quantization, language,
  or runtime compatibility.
- Importing local GGUF files; TASK-597 owns that work.
- Adding or configuring a Hugging Face token in this screen.
- Search pagination, result caching, model-card/README rendering, compatibility
  detection, automatic activation, or download recommendations.
- General remote providers, mirrors, self-hosted repositories, or LAN origins.
- Adding `huggingface_hub` or another dependency.

Runtime adoption and validation remain with TASK-597 and TASK-604. This task
only makes remote GGUF bytes discoverable and safely managed.

## User flow

1. The user opens **Remote**. The idle view renders without filesystem or
   network I/O.
2. The user enters text and presses **Search**.
3. If the input is a valid exact `owner/repository` identifier, Chatbook
   resolves that repository directly. Otherwise it performs a bounded model
   search and shows at most 50 typed results.
4. Selecting a search result resolves that repository to an immutable commit
   and lists at most 100 eligible GGUF choices.
5. The user selects one single GGUF or complete shard set. Chatbook displays the
   repository, commit, files, total bytes, declared license state, provenance,
   and the warning **Runtime compatibility has not been verified.**
6. The existing managed preflight shows destination, free space, staging,
   immutable sources, sizes, digests, and provenance.
7. The user explicitly confirms. If no license was declared, confirmation stays
   disabled until the user acknowledges that fact.
8. The existing provision worker downloads and verifies the candidate without
   activating it.
9. Success reads: **Model downloaded and managed. Runtime compatibility has not
   been verified.** Installed refreshes and shows the model as downloaded but
   unconfigured.

Selecting any shard in a standard shard set represents the entire set. An
individual shard is never offered as an independently installable model.

## Component boundary

### `Model_Artifacts/remote_huggingface.py`

A narrow, Textual-free adapter owns Hugging Face request construction, bounded
response reading, strict JSON parsing, GGUF grouping, and conversion to managed
artifact values. It performs no writes and downloads no model payloads.

It exposes typed immutable values:

- `RemoteModelSummary`: bounded search-result metadata needed by the list.
- `RemoteGGUFCandidate`: one single file or complete shard set, including
  upstream paths, per-file sizes/digests, and total size.
- `ResolvedRemoteModel`: repository, immutable commit, license state, and
  bounded candidates.
- `ResolvedRemoteCatalog`: the one selected `ArtifactDescriptor` plus its
  complete `ArtifactSourceMap`; it structurally satisfies `ArtifactCatalog`.

The adapter uses existing `httpx`. It does not import the legacy downloader or
introduce a provider framework.

### `UI/Screens/model_remote_view.py`

`RemoteView` owns form state, workers, monotonic generations, the selected
in-memory catalog, and the managed install lifecycle. It reuses
`ModelInstallModal`, `ModelInstallProgress`, the existing progress messages,
`managed_service()`, and `EnvConfigCredentialResolver`.

The view follows the same ownership rule as Curated: widgets and modals return
intent, while the view owns search, resolution, preflight, and provision
workers. No network access occurs in `compose()`, `on_mount()`, or merely because
the parent Models screen was composed.

### Existing integration points

- Add `("remote", "Remote")` before **Download Models** in
  `MODELS_RAIL_SECTIONS`.
- Add the corresponding container and view mapping in
  `LLMManagementWindow`.
- `_start_view_work("remote", ...)` must not start a request; it may only leave
  the already composed view idle.
- Install progress and completion use the existing messages so Installed and
  the Lab status chip remain synchronized.

## Request and parsing rules

All metadata requests are explicit GET requests to the fixed HTTPS origin
`https://huggingface.co`:

- bounded model search through the Hugging Face models API;
- one repository-info request with file metadata when a result is selected or
  an exact repository is submitted.

Metadata responses do not follow redirects. Request URLs are constructed by the
adapter from validated components; URLs returned inside response data are never
trusted or followed.

The adapter enforces all of these limits while streaming decoded response
bytes, rather than trusting `Content-Length`:

| Limit | Value |
|---|---:|
| decoded metadata body | 2 MiB per response |
| search results | 50 |
| repository file entries inspected | 2,048 |
| GGUF candidates returned | 100 |
| shards in one set | 64 |

JSON values have exact expected types. Repository identifiers, commit SHAs, and
file paths are independently validated before any URL or descriptor is built.
A repository identifier must be one bounded `owner/repository` pair using the
portable Hugging Face identifier character set. The resolved commit must be an
immutable hexadecimal Git commit identity, not `main`, a tag, or user input.

A repository response containing more than 2,048 file entries is rejected; it
is never partially inspected. Eligible candidates are sorted by their exact
upstream path tuple before the 100-candidate display cap is applied. When that
cap truncates the list, the resolved value records the total and the UI says
that only the first 100 deterministic choices are shown.

## Eligible GGUF candidates

Only repository entries with all of the following are eligible:

- a `.gguf` filename;
- a non-negative declared size;
- a normalized 64-character lowercase LFS SHA-256 digest from file metadata;
- a path that can be safely represented as a URL source identifier.

An entry without LFS metadata, size, or digest is ignored and cannot be
installed. This means Hugging Face Git-LFS metadata provides the expected digest
used to verify the downloaded bytes. Because the digest comes from the same
origin as the payload metadata, it is recorded as local integrity, not
independent publisher verification.

Every standard shard member matches
`^(?P<stem>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$`.
Both numeric fields are exactly five digits, with
`1 <= index <= count <= 64`. Grouping includes the containing directory, stem,
and declared count. A shard group is eligible only when:

- every index from 1 through `NN` is present exactly once;
- every shard declares LFS size and SHA-256 metadata;
- every shard belongs to the same directory/stem/count group;
- `NN` does not exceed 64.

An incomplete, duplicate, inconsistent, or oversized group is rejected as a
group and none of its members is offered as a single file. Non-sharded GGUF
files remain independent candidates.

## Pinned source construction

Payload URLs are generated locally from:

1. the fixed Hugging Face HTTPS origin;
2. the validated repository identifier;
3. the repository's resolved commit SHA;
4. the validated and segment-encoded upstream file path.

No URL from repository metadata is reused. URLs contain no credentials, query,
fragment, mutable revision, or userinfo. The selected catalog supplies one
source-map entry for every managed file.

Hugging Face payload endpoints may redirect to storage hosts. The existing
managed fetcher remains authoritative: it checks every hop, permits only its
bounded redirect statuses, and strips authorization on every cross-origin hop.
As a shared security hardening, an initially HTTPS fetch must reject any
redirect hop whose scheme is not HTTPS; initially HTTP fixture/caller behavior
is otherwise unchanged. The Remote adapter does not implement a second redirect
loop.

## Managed identity and descriptor

Remote identifiers are deterministic but never use raw repository or file names
as managed directory identity:

- `artifact_id`: `hf-gguf-` plus the full lowercase SHA-256 of canonical JSON
  `{"repository": <exact resolved repo ID>, "paths": <exact paths sorted
  lexicographically>}`, encoded as UTF-8 with sorted keys and compact
  separators;
- `revision`: the full resolved repository commit SHA;
- `variant` and `precision`: `not-declared`.

The persisted `model_id` is a bounded, markup-disabled human label formed from
the repository ID and candidate filename/stem. It remains understandable in
Installed after the ephemeral search state is gone.

Upstream paths are display/source identifiers only. Managed payload names are
portable and deterministic:

- a single file becomes `model.gguf`;
- a shard set becomes `model-00001-of-000NN.gguf` through
  `model-NNNNN-of-NNNNN.gguf`.

The source map relates those managed paths to their exact pinned upstream URLs.
This avoids unsafe, Unicode-dependent, reserved, or case-colliding filesystem
identity while retaining conventional shard naming.

The descriptor is deliberately inert:

| Field | Value |
|---|---|
| role | `ROOT` |
| format | `GGUF` |
| consumer | `unassigned` |
| model family | `unassigned` |
| runtime name | `unassigned` |
| runtime version constraint | `none` |
| supported OS / architectures | `unassigned` sentinel values |
| provenance | `LOCAL_INTEGRITY_RECORDED` only |
| dependencies | none |
| usage notice | runtime compatibility is not verified; configuration is required |

The unassigned OS/architecture sentinels intentionally fail closed. They do not
claim support for every platform.

## License behavior

If model metadata declares a non-empty license identifier, the identifier is
shown. If it does not, the descriptor records `NOASSERTION`.

`ArtifactDescriptor` requires a non-empty `license_url`, but repository metadata
does not guarantee a pinned license-document URL. Remote descriptors therefore
store the pinned repository page as the review URL. User-facing copy calls this
the **source review page**, never a license document.

For `NOASSERTION`, the shared install modal accepts one optional required
acknowledgment. Remote supplies: **No license was declared. I reviewed the
source and want to continue.** The Install button stays disabled until checked.
Curated callers pass no acknowledgment and retain their current behavior.

## Credentials and private repositories

Both metadata requests and managed payload requests reuse
`EnvConfigCredentialResolver`, preserving the existing precedence:
`HUGGINGFACE_API_KEY`, then `HF_TOKEN`, then `[API] huggingface_api_key`.

The resolved token exists only in request memory. It is never included in a
descriptor, source map, artifact ID, consent fingerprint, error, notification,
or log. Metadata requests attach it only to `https://huggingface.co`. Managed
payload requests already attach it only when the source URL shares the
descriptor source origin; the fetcher strips it after a cross-origin redirect.

No token-entry or access-approval UI is added. A 401/403 explains that configured
Hugging Face access is required and offers retry. For a gated repository whose
terms have not been accepted, the pinned repository page is the recovery
destination outside this task's UI.

Exact repository submission is required for reliable private/new repository
access; free-text search is not treated as the only discovery path.

## Acquisition without activation

Today `ArtifactAcquisitionService.provision()` always installs and activates a
closure. Remote content must be installed without receiving an active selector.

Add a keyword-only `activate: bool = True` parameter:

- the default preserves every existing Curated and Parakeet caller;
- when true, behavior and progress are unchanged;
- when false, provision performs the complete fetch, pre-verify, and immutable
  install phases, then returns the installed root without calling
  `ModelArtifactService.activate()` or emitting an `activate` progress event.

Remote passes `activate=False`. This is an additive extension of ADR-025's
managed acquisition boundary, not a second installation path.

Immediately after install-only provisioning, the artifact has no active
selector and normally no readiness record. Existing `reconcile()` may later
create readiness for any structurally valid ROOT artifact, including an
unassigned Remote artifact; this task does not change that core behavior.

Remote inertness therefore never depends on `InstalledArtifact.ready`. Installed
derives `runtime_assigned` from the descriptor's consumer, runtime, OS, and
architecture fields: it is false when `consumer`, `model_family`, or
`runtime_name` equals `unassigned`, or either supported-platform tuple contains
the `unassigned` sentinel. An unassigned descriptor always says **Downloaded ·
runtime compatibility not verified**, even if Repair later sets `ready=True`.
Add an optional `allow_activation: bool = True` input to
`ModelActivationControls`; unassigned rows pass false, which omits only the
Activate button while preserving the existing lease-safe Delete action. No
other installed-row behavior changes.

## Concurrency and cancellation

Search and repository resolution each use a monotonic generation. A worker may
apply a result only if its generation and relevant input still match current UI
state. A new search invalidates any older search or resolution result.

Only one search/resolve request and one managed install operation are presented
as current. Search controls are disabled during preflight/provision. The
existing acquisition session lease remains the authority for concurrent model
downloads across views and processes.

Worker callbacks update widgets only through Textual's thread-safe message or
`call_from_thread` paths. A dismissed or recomposed view may discard stale
presentation results, but cannot weaken the acquisition service's staging and
lease guarantees.

## Errors and recovery

The adapter and view map typed failures to short sanitized messages:

| Failure | User-visible recovery |
|---|---|
| authentication or gated access | configure/verify Hugging Face access, then Retry |
| repository not found | check the exact ID or search again |
| timeout, rate limit, or network error | Retry |
| malformed or oversized metadata | repository cannot be safely inspected |
| no eligible LFS GGUF | explain the LFS size/digest requirement |
| incomplete shard set | name the candidate and missing shard indexes, bounded |
| insufficient disk or acquisition gating | existing preflight explanation |
| transfer or digest mismatch | existing managed-acquisition failure and Retry |

Logs record the operation and a bounded/hash-safe context, not raw search text,
private repository names, tokens, source bodies, or arbitrary upstream error
content. UI labels always use `markup=False`.

## Portability

The feature adds only pure Python over existing `httpx`, Textual, and the shared
artifact service. It introduces no native package, shell command, platform path,
or external downloader. Managed filenames are portable across the service's
supported filesystems.

The implementation contract covers every currently wheel-supported platform.
macOS evidence can be collected locally. Existing Windows and Linux gates remain
required when those runners are available; the task does not claim local test
coverage that was not run.

## Verification

Focused tests cover:

1. Search and exact-repository request construction, token scoping, strict
   parsing, body/result/file limits, and sanitized failures using
   `httpx.MockTransport` or equivalent injected clients.
2. Single GGUFs, complete shard sets, incomplete/duplicate/inconsistent shard
   sets, missing LFS metadata, candidate limits, and deterministic portable
   managed filenames.
3. Descriptor identity, `NOASSERTION`, pinned review/source URLs, provenance,
   unassigned runtime fields, catalog completeness, and source maps.
4. Existing cross-origin redirect behavior proving a Remote bearer token is
   stripped on the redirected payload request, plus rejection of an HTTPS to
   HTTP downgrade.
5. `provision(activate=False)` installs without an active selector or activate
   progress; the default still activates and preserves existing behavior.
6. Textual Pilot coverage proving opening Remote performs no request, explicit
   submission does, stale generations are ignored, unknown-license consent is
   gated, failures are retryable, and success refreshes Installed.
7. Installed state mapping renders unassigned Remote artifacts as downloaded but
   unconfigured, with no activation affordance and with deletion still present,
   both before and after reconciliation creates readiness.
8. One integration test from selected candidate through managed preflight and
   provision using mocked HTTP and a temporary artifact root; no real model
   download is required.

Static checks cover the changed files, and the affected Model Artifacts, Lab,
shared-widget, and Remote suites form the local regression gate.

## ADR check

**ADR required:** no
**ADR path:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Reason:** ADR-025 already owns remote artifact provenance, immutable managed
downloads, consent, activation, and runtime boundaries. The install-without-
activation option is an additive way to enforce that existing boundary for an
unassigned remote artifact; it does not introduce a new owner or storage model.
TASK-1723 does not apply because this slice permits only Hugging Face's public
HTTPS origin, not self-hosted or private-network origins.

## Rollback

Remote is an independent rail row. If discovery proves unreliable, remove or
disable that row and adapter while leaving Curated, Installed, the managed store,
and already downloaded immutable artifacts intact. Never fall back to the legacy
unverified downloader for a failed Remote install.

## References

- [Hugging Face Hub API model information](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)
- `Docs/superpowers/specs/2026-08-01-task-596-model-artifact-browser-design.md`
- `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
