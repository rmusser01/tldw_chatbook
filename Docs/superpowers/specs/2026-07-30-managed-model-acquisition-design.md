# Managed Model Acquisition (TASK-595) — Design

**Date:** 2026-07-30
**Status:** Approved (brainstorm complete; awaiting implementation plan)
**Chain:** TASK-594 (Done, the immutable artifact core) → **TASK-595 (this)** → TASK-596 (reusable setup controls) → TASK-1301 (wizard Speech step)
**Binding references:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`; `Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md` (acquisition sections); TASK-595's six ACs, adopted in full.

## Purpose

Consent-driven managed acquisition over the shared artifact lifecycle: curated
ONNX/GGUF bundles can be preflighted, downloaded with resume, verified,
activated, and recovered safely — with the TASK-594 core reused unmodified for
every integrity-bearing step.

## Decisions taken during brainstorm

1. **Full ACs as written** — no wizard-sufficient subset.
2. **Async service surface** — `httpx.AsyncClient` streaming in async methods;
   the blocking 594 core (leases, verify, promote) is reached via executor
   hops. One seam, stated explicitly.
3. **Consent is API-shaped, honestly scoped** — `PreflightReport.grant()` is
   the only non-deliberate way to obtain consent; Python cannot make the type
   unforgeable, so the worker-side guarantee is an import-boundary test
   (workers never import the acquisition modules), following
   `Tests/STT/test_boundaries.py`.
4. **Composition over the sealed core** (Approach A): a new
   `ArtifactAcquisitionService` composes `ModelArtifactService`; only three
   small, sync, agreed additions touch the core: `install(...,
   consume_source: bool = False)`; `reconcile()`'s staging GC; and a
   per-staging-dir `install()` lease (its `locks_path` property plus
   `ReconcileReport.staging_removed`) added mid-implementation as an
   approved in-flight fix for a real race between a crashed `install()`'s
   abandoned staging tempdir and a concurrent `reconcile()` pass (see
   Core additions below).

## Architecture

### Modules

- **`Model_Artifacts/acquisition.py`** — `ArtifactAcquisitionService(core, *,
  client_factory, credential_resolver, free_bytes_probe)` (all seams
  injectable for tests). Public methods: `preflight()`, `provision()`
  (named to avoid colliding with the core's lease-oriented `acquire()`).
- **`Model_Artifacts/fetch.py`** — `stream_fetch()`: the `egress.py` guarded
  hop loop (same SSRF policy, hop cap, credential stripping on cross-origin
  redirects) re-shaped to stream chunks to a staged file under a hard
  `max_bytes` bound instead of buffering bodies in memory. Supports `Range`
  resume and returns captured validators (ETag / Last-Modified).
- **Core additions (sync, small; three total, not two — see Decision 4):**
  - `ModelArtifactService.install(..., consume_source: bool = False)` —
    per-file `os.replace` when the source directory lies inside the service
    root; a source outside the root raises `ArtifactPathError` (no silent
    copy fallback). Within-root `EXDEV` (a bind-mount under the root)
    degrades to copy+delete for that file — correctness over the disk
    optimization. Halves peak disk during install.
  - `reconcile()` staging GC — deletes **true orphans only**: entries with a
    missing or unparseable sidecar (e.g. dead install-staging tmpdirs).
    Valid-sidecar staging is resumable state and is never GC'd (see
    Recovery). Deletions are containment-checked and counted in
    `ReconcileReport`.
  - Per-staging-dir `install()` lease — a reserved `#install-staging`
    lease key held for one `install()` call's whole staging tempdir
    lifetime, exposed via the new `ModelArtifactService.locks_path`
    property (so `ArtifactAcquisitionService`, in a different module, can
    take its own reserved-namespace leases against the same lock root) and
    surfaced in reports via `ReconcileReport.staging_removed`. Approved
    mid-implementation, not part of the original two-addition plan: without
    it, `reconcile()`'s staging GC could not tell a crashed `install()`'s
    abandoned tempdir apart from one still legitimately in flight, a real
    race between the two.

### Serialization: one acquisition-session lease

`provision()` holds a single **exclusive acquisition-session lease** (a
reserved key namespace in the existing `leases.py` machinery) for the whole
run, acquired **non-blocking**: a concurrent attempt in another process gets
an immediate typed `AcquisitionBusyError` rather than a hang. In-process,
the service serializes itself with an `asyncio.Lock`. The session key never
overlaps `install()`'s per-artifact lease keys, so the
lock-held-across-a-lock-taker deadlock class is structurally impossible.
`reconcile()`'s GC requires taking the session lease (non-blocking) before
touching staging.

### The flow (parakeet-spec steps, mapped)

1. **`await preflight(root, catalog)`** — walks the closure over **catalog
   descriptors** (`descriptor.dependencies`, recursively, with the core's
   cycle/conflict rules and `closure_fingerprint()` over the resolved set —
   NOT the core's `_resolve_closure`, which reads installed manifests).
   Aggregates a frozen `PreflightReport`; performs one bounded HEAD probe per
   repository so gated/authenticated repos fail here, with instructions,
   before any consent screen.
2. **Consent** — `report.grant()` → `AcquisitionConsent` carrying the closure
   fingerprint; raises `PreflightNotGrantableError` on gating or space
   failure. `provision()` recomputes the closure and refuses on fingerprint
   drift (`ConsentMismatchError`) — changed content means new preflight.
3. **Fetch** — per artifact in stable sorted order, each declared file
   streams into durable staging `staging/managed/<id>/<rev>/<variant>/<file>`
   with ONE sidecar `fetch-state.json` per artifact staging directory,
   mapping filename → `{validators, bytes_done, complete}` (a single atomic
   write path; the one thing GC validates). Durability order:
   `fsync(data)` **before** the atomic fsynced sidecar write — resume state
   may only claim durable bytes. Resume: strong validators match → `Range`
   continuation (`resume_from + written ≤ max_bytes`); weak (`W/`) or changed
   validators, or no `Range` support → restart that file from zero.
4. **Pre-verify** — acquisition streams SHA-256 over each fully staged file
   *before* install. A corrupt file resets only its own sidecar and refetches
   once per provision run, then fails typed. `install()` therefore only ever
   consumes verified bytes (its own verification remains as defense in
   depth) — and `consume_source` can never destroy good siblings over one bad
   file.
5. **Install** — executor hop to the core's existing
   `install(descriptor, source_directory=staged_dir, consume_source=True)`;
   every 594 guarantee applies unchanged. The artifact's download staging is
   gone after success (consumed).
6. **`activate(root)` last** — the core writes readiness only after
   verifying the whole closure; prior active version untouched until then.
7. **Idempotent completion** — an already-fully-installed closure provisions
   with zero fetch work (same consent ceremony; phases skip to
   verify/activate). This is also the crash-after-install recovery path.

### Cancellation semantics (stated honestly)

Task cancellation is honored at fetch chunk boundaries and between
artifacts. Lease acquisition is non-blocking with a short, fixed timeout
(0.1s for `provision()`'s acquisition-session lease; see
`NONBLOCKING_LEASE_TIMEOUT_SECONDS`) rather than a long poll, so no
`cancelled`-callable bridge into `leases.py` was ever implemented or is
needed — a wait that short has nothing meaningful to interrupt mid-flight.
`ArtifactOperationLease.cancelled` remains available as a constructor
parameter for a future caller that does need it, but no lease in this
acquisition path passes one. An `install()` already running in an executor
completes before cancellation takes effect — it is short relative to
transfers, and AC #3's guarantees hold regardless.

## API surface

```python
class ArtifactCatalog(Protocol):
    def descriptor(self, ref: ArtifactRef) -> ArtifactDescriptor: ...

@dataclass(frozen=True)
class ArtifactPreflightEntry:
    ref: ArtifactRef; source_url: str; repository: str; revision: str
    license_id: str; license_url: str; precision: str
    total_bytes: int; file_count: int; already_installed: bool

@dataclass(frozen=True)
class PreflightReport:
    root: ArtifactRef
    closure_fingerprint: str
    entries: tuple[ArtifactPreflightEntry, ...]   # stable sorted order
    download_bytes: int                           # remaining to transfer
    already_staged_bytes: int                     # best-effort resumable credit (sidecar
                                                  # bytes; server revalidation at provision
                                                  # may still force restarts — consent copy
                                                  # labels it "up to N already fetched")
    staging_overhead_bytes: int                   # computed for consume_source semantics
    retained_bytes: int                           # prior active kept during upgrade
    destination: Path
    free_bytes: int
    required_bytes: int                           # download + staging + retained + margin
    sufficient_space: bool
    gating_errors: tuple[str, ...]
    def grant(self) -> AcquisitionConsent: ...    # raises PreflightNotGrantableError

@dataclass(frozen=True)
class AcquisitionConsent:
    closure_fingerprint: str

@dataclass(frozen=True)
class AcquisitionProgress:
    phase: Literal["fetch", "pre-verify", "verify-install", "activate"]
    ref: ArtifactRef; file: str | None
    bytes_done: int; bytes_total: int             # real byte detail in fetch AND
                                                  # pre-verify (streaming SHA-256 knows
                                                  # its position); install/activate emit
                                                  # per-artifact indeterminate events

class ArtifactAcquisitionService:
    async def preflight(self, root: ArtifactRef, catalog: ArtifactCatalog) -> PreflightReport: ...
    async def provision(
        self, consent: AcquisitionConsent, catalog: ArtifactCatalog, *,
        progress: Callable[[AcquisitionProgress], None] | None = None,
    ) -> ArtifactRef: ...                         # the activated root
    # provision re-walks the catalog: the fingerprint drift-check REQUIRES an
    # independent re-resolution, and the freshly resolved descriptors supply
    # the file URLs — consent alone cannot carry enough to download safely.
```

`fetch.stream_fetch(url, destination, *, client, max_bytes, resume_from=0,
validators=None, headers=None, trusted_origins=frozenset(), on_chunk=None)
-> FetchResult(bytes_written, validators, resumed)`. The sidecar is owned by
acquisition, not fetch.

**Errors** extend the existing family: `AcquisitionError(ArtifactError)` →
`ConsentMismatchError`, `PreflightNotGrantableError`, `AcquisitionBusyError`,
`InsufficientSpaceError`, `GatedRepositoryError`, `TransferError` (artifact/
file context, a `retryable` flag, never URLs-with-tokens or header values),
`CatalogError` (unknown/invalid refs surface typed at preflight, never a raw
`KeyError` mid-provision).

## Credentials

Resolved per request via an injected resolver (env → config → keyring where
available, matching the parakeet spec); attached only to same-origin
requests (the fetch layer strips them on cross-origin hops); never written
to manifests, sidecars, or logs (log lines carry repository ids only).
Gated repos without working credentials fail at preflight.

## Recovery matrix

| Interruption | State left | Next provision |
|---|---|---|
| Network drop / cancel / crash mid-fetch | Partial file + durable sidecar; prior active untouched | Strong validators match → `Range` resume; else restart file |
| Upstream changed | Stale validators | Restart affected file |
| Corrupt payload | Bad staged file | Pre-verify catches; one refetch; typed failure; nothing installed |
| Crash mid-`consume_source` | Files split across stagings | Orphaned install-staging GC'd by `reconcile()`; provision re-checks sidecar-vs-files, refetches missing |
| Crash after install, before activate | Installed-but-inactive immutable versions | `already_installed` → skip to verify + activate |
| Crash mid-activate | Readiness written last (594 semantics) | Re-run activate; `acquire()` refuses root until readiness exists |
| Disk full mid-transfer | `TransferError`, staging retained | Free space; resume (provision re-checks free space at start) |

**Never-trap rule:** every failure is a typed, contextual error with a
`retryable` flag; callers always render Retry/Skip.

## Test plan (AC #6)

Local stdlib threading HTTP fixture server with switches: `Range` on/off,
ETag strong/weak/changing, mid-body disconnect, 401-until-token, wrong
bytes. On the existing `Tests/Model_Artifacts/` process harness:

1. Resume: kill mid-file → `Range` continuation; completed files untouched.
2. Changed validators → restart-from-zero of that file only.
3. Corrupt payload → pre-verify catch, one refetch, typed failure, nothing installed.
4. Concurrency: second process gets `AcquisitionBusyError` during overlap;
   after completion it converges via `already_installed` (the core's own
   concurrent-install safety is 594's tested ground).
5. Insufficient space: probe fails `grant()`; injected ENOSPC mid-write →
   `TransferError`, staging retained.
6. Crash recovery: `kill -9` at staged checkpoints → `reconcile()` GCs only
   orphans; valid-sidecar staging survives; resume completes.
7. Containment: GC never deletes outside `staging/managed`.
8. Import boundary: provider/worker modules import neither `acquisition`
   nor `fetch`.
9. Gated repos: 401 → preflight `gating_errors`; with token → success; log
   and sidecar scans prove the token appears nowhere.

## Non-goals (v1)

Retention-count/GC policy changes; download scheduling/parallel transfers
(files fetch sequentially); catalog contents and their storage (TASK-596/
1301); UI of any kind (596); mirroring the existing HuggingFace browser
downloader (ADR-025 disqualifies it).

## Open items for the implementation plan

- Reserved session-lease key: confirm `ArtifactLeaseKey` field validation
  accepts the reserved namespace or add a dedicated constructor.
- Whether `stream_fetch` lands beside `guarded_fetch_httpx` in `egress.py`
  or in `Model_Artifacts/fetch.py` importing egress's policy helpers —
  decide by import-cycle reality.
- Exact safety-margin constant for `required_bytes` (parakeet spec names the
  factors; pick and document the number).
- Per-file automatic refetch count is 1 by design; confirm no AC test needs
  a configurable knob.
