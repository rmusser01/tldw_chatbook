# TASK-596 — Model artifact browser: design

**Date:** 2026-08-01
**Task:** TASK-596 — Renovate the local model artifact browser
**Depends on:** TASK-595 (merged, PR #1157), TASK-1694/1695/1696 (merged, PRs #1165/#1167)
**ADR:** `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
**Status:** approved section-by-section on 2026-08-01; Phase 1 planned separately

## Why now

TASK-595 shipped a verified, resumable, consent-driven acquisition layer, and
TASK-1696 gave it its first production consumer. Two things follow:

1. The layer has exactly one hardcoded descriptor and one bespoke install
   dialog. Nothing enumerates what else could be installed.
2. The old GGUF browser still writes unverified downloads to disk — the
   behavior ADR-025 says not to keep.

This design replaces the second with a UI over the first.

## What exists today

Two disconnected model UIs:

**`HuggingFaceModelBrowser`** — Lab → Models → "Download Models", about 2,200
lines across five widgets (`Widgets/HuggingFace/`). Searches HF, renders a model
card, queues file downloads into `~/Downloads/tldw_models` through its own
`DownloadManager`. No pinned revision, no digests, no activation, no leases.
Alongside it, "Local Models" (`local_models_widget.py`) scans that directory with
a threaded `os.walk` and offers delete.

**`ParakeetV2InstallModal`** — Library, shipped in TASK-1696. `ModalScreen[bool]`
driving `preflight → plan → consent`, with `LibraryScreen` owning the provision
worker. One hardcoded descriptor, and no progress display.

## Goals

Satisfy TASK-596's eight acceptance criteria with a provider-neutral browser over
`ModelArtifactService` and `ArtifactAcquisitionService`, and give the install,
plan, progress, and activation controls a single implementation shared by the
browser, Library, the first-run wizard (TASK-1301), and Settings.

## Non-goals

- Replacing the local-server launch UIs (llama.cpp, vLLM, MLX, Ollama, …). Only
  the "Models" rail section changes.
- A new top-level destination. The browser lives inside Lab → Models.
- Any change to `Model_Artifacts` core semantics. TASK-594's core is sealed; the
  one additive change this design requires is stated explicitly below.

## Phasing

The end state is one browser for every model this application manages, reached in
three phases so that the risky server-path work lands last and can be dropped
without stranding the rest.

**Phase 1 — Curated + Installed, no network.**
Curated registry, the pure view-model, the shared widgets, the two offline views,
and the Library modal refactored onto the shared modal. Installed **replaces**
"Local Models" and absorbs its unmanaged scan. "Download Models" stays mounted,
untouched.

**Phase 2 — Remote.**
Search an external index, resolve a selection to a pinned revision with per-file
digests, then install through the same flow as Curated. Arbitrary repositories
install as `LOCAL_INTEGRITY_RECORDED`.

**Phase 3 — GGUF adoption.**
Import existing unmanaged files as artifacts; local-server launch paths resolve
from the service instead of user-typed paths; retire `DownloadManager` and the
`Widgets/HuggingFace/` browser.

This spec covers all three, but only Phase 1 is planned for implementation now.

## Placement and naming

The browser lives in **Lab → Models**, replacing the two rows of
`MODELS_RAIL_SECTIONS`' "Models" section (`llm_screen.py:42-48`) with:

```
Models
  Curated       descriptors this application vouches for; offline
  Installed     inventory, activation, deletion, disk use
  Remote        Phase 2
  Download Models   (legacy; removed in Phase 3)
```

Three rail rows rather than one row with tabs, matching the screen's existing
navigation idiom.

**Naming constraint.** `artifacts` is already the top-level destination for
generated outputs and Chatbooks (`artifacts_screen.py`), and `Model_Artifacts/`
is an internal package. **`artifact` never appears in user-visible copy in this
feature**; the UI says "model" throughout. Internal module and class names keep
the artifact vocabulary.

**One inventory, several entry points.** STT and LLM models appear in the same
Installed view. Library, the wizard, and Settings keep their own install entry
points, but all of them route through the shared controls and read the same
inventory, so no two surfaces can disagree about what is installed or active.

**Deep links are not free.** `lab_frame.py` has no initial-view or deep-link
support; rail rows set `active_view`, and `LLMManagementWindow.watch_active_view`
toggles a `-active` CSS class over `view_mapping`. If the wizard or Settings must
land on a specific view, that is a `PendingHandoffStore` channel
(`UI/Navigation/pending_handoff_store.py`) and explicit work — not a property the
rail provides.

## Views

**Curated.** `CuratedRegistry.list()` cross-referenced against
`ModelArtifactService.list_installed()` to mark what is already present. Pure and
offline.

**Installed.** `list_installed()` plus `disk_usage()`, plus a threaded scan of the
legacy download directory. Unmanaged files are listed as
`Unmanaged — integrity unknown`, greyed, with no actions until Phase 3 adds
Import. Omitting them would make a user with 40 GB of GGUF files see an
"Installed" screen listing only Parakeet and conclude the screen is broken.

**Remote.** Phase 2.

### Views mount eagerly; work must not

`LLMManagementWindow` composes every view up front and switches by CSS class, so
a view's `compose`/`on_mount` runs on screen construction. **No view may scan,
walk, or read inventory at compose time.** Data loads on first activation and on
explicit refresh.

## Data flow

Three reads, one write path, two mutations. The browser performs no I/O of its
own; everything goes through the two services.

The install path is one flow, shared by every entry point:

```
select (ArtifactRef + catalog)
  → preflight(root, catalog, sources=…)           threaded
  → render the plan from PreflightReport          pure, no I/O
  → user consents → report.grant()                raises if not grantable
  → provision(root, consent, catalog, progress=)  threaded
  → activate(root_reference)                      threaded
```

### AC #3 is derived entirely from `PreflightReport`

| Requirement | Source |
|---|---|
| full dependency closure | `report.entries` — one row per artifact |
| immutable source revision | `entry.repository`, `entry.revision` |
| license | `entry.license_id`, `entry.license_url` |
| precision | `entry.precision` |
| download bytes | `report.download_bytes`, `entry.total_bytes`, `entry.file_count` |
| staging requirement | `report.staging_overhead_bytes`, `report.already_staged_bytes` |
| destination | `report.destination` |
| free-space result | `report.free_bytes`, `report.required_bytes`, `report.sufficient_space` |

**Required additive change:** `ArtifactPreflightEntry` gains a `provenance`
field, so AC #2's three trust labels come from the same report as the bytes they
describe rather than being derived separately by each consumer. This is safe:
`closure_fingerprint(root, dependencies)` is computed purely from `ArtifactRef`
values (`service.py:826`), so consent fingerprints, resume state, and in-flight
downloads are unaffected by adding a field to the report.

### The plan is a gate, not a warning

`report.grant()` raises `PreflightNotGrantableError` when `gating_errors` is
non-empty or `sufficient_space` is false. Insufficient space or a gating error
therefore renders the Install control **disabled with the reason shown** — never
enabled-then-failed.

### Cost and threading

`preflight`, `provision`, `activate`, `disk_usage`, the unmanaged scan, and
Phase 2's remote search all run in `@work(thread=True)`. Two of these are easy to
misjudge:

- **`disk_usage()` walks the entire managed tree** — `_regular_tree_bytes` over
  both `artifacts/` and `staging/` on every call (`service.py:2219`). Threaded,
  cached, refreshed on demand; never per render.
- **`activate()` is not a checkbox.** It takes an exclusive lifecycle lease,
  resolves the closure, takes a shared lease set over every member, and calls
  `_verify_installed()` per member when readiness does not match — full digest
  verification while holding that exclusive lease (`service.py:1963-2012`). It
  needs a pending state, it can block on another process's lease, and it must
  refuse re-entry.

`list_installed()` is manifest-only and cheap by comparison, but is still disk
I/O and travels with the same worker.

### Progress

`provision(progress=…)` emits `AcquisitionProgress(phase, ref, file, bytes_done,
bytes_total)` across `fetch → pre-verify → verify-install → activate`. The
callback runs **on the worker thread** and posts a message rather than touching
widgets. `fetch.py` reads in 1 MiB chunks, so a 660 MB download produces roughly
660 events — about ten per second at a typical rate. Coalescing is optional; no
throttle is required.

**Progress must outlive its dialog.** The host screen owns the provision worker
precisely so a download survives dismissal of the consent modal, which means
progress cannot live only inside that modal. It renders in a persistent home —
the Installed view's row for that model, plus the Lab status chip — and the modal
mirrors that state while it happens to be open.

The Library Parakeet install currently passes no `progress` callback at all
(`run_parakeet_v2_provision` accepts one; `library_screen.py` omits it), so a
660 MB install today shows the user nothing. Phase 1 fixes this as a side effect
of the shared component.

### Mutations

`activate(root_reference)` is the AC #7 selector. Because activation is keyed by
`artifact_id` (`active_path`), exactly one revision/variant of a model is active
at a time, and selecting a precision means selecting the variant that carries it.
Revisions that are not `ready` are not offered.

`delete(reference)` reports blockers rather than forcing. **AC #5's idle
heavy-worker recycle is deferred:** nothing in the service can currently ask a
worker to unload a resident model, and the mechanism belongs to whoever owns the
heavy-worker pool. Until it exists, an artifact held by an active lease is
reported as blocked, with the reason named. An active lease is never bypassed and
an active job is never silently cancelled.

`reconcile()` is a user action behind an explicit "Repair" control, with its
`ReconcileReport` shown afterward. It can remove staging, so it never runs as a
side effect of opening a screen.

### Broken inventory rows are a defined state

`InstalledArtifact.descriptor` is `ArtifactDescriptor | None`, paired with
`error: str | None` — the service deliberately surfaces unreadable or corrupt
manifests as inventory entries. The Installed view renders these as first-class
rows ("unreadable manifest — Repair"). Any code path reaching for
`descriptor.license_id` without a guard crashes the screen, and these rows appear
only after a crash or partial delete, so they must be tested explicitly.

## Components

### Layer 1 — service-side, no Textual

**`Model_Artifacts/curated_registry.py`** (new). `list() -> tuple[ArtifactDescriptor, ...]`
and `descriptor(ref)`, so the registry structurally satisfies `ArtifactCatalog`.
It owns the answer to "what does this application vouch for", which today has no
owner: `ArtifactCatalog` (`acquisition.py:188`) is lookup-only, and the only
implementation is `ParakeetV2Catalog` inside `Local_Ingestion/`. Parakeet v2
registers into the registry instead of keeping a bespoke catalog class.

**`managed_model_artifact_root()` moves here** from
`Local_Ingestion/parakeet_v2_artifact.py`, whose own docstring already states it
is not Parakeet-specific. The Parakeet module re-exports it for its existing
callers.

### Layer 2 — pure view-model, no Textual, no I/O

**`UI/Screens/model_browser_state.py`**, following `first_run_setup_state.py`:

- `plan_rows(report)` — the AC #3 table
- `inventory_rows(installed, usage, unmanaged)` — including broken rows and
  deletion blockers
- `install_failure_message(exc, *, model_label)` — the typed-error mapping lifted
  from `library_screen.py:942`, whose Parakeet-specific strings become a label
  parameter

Unit-testable with plain pytest, no Pilot.

### Layer 3 — widgets

Shared, in `Widgets/ModelArtifacts/`:

| Widget | Responsibility |
|---|---|
| `ModelPlanPanel` | renders a `PreflightReport` via `plan_rows`; read-only |
| `ModelInstallProgress` | renders `AcquisitionProgress`; four phases, per-file detail |
| `ModelInstallModal` | `ModalScreen[bool]`: plan + consent; returns a decision, owns nothing |
| `ModelActivationControls` | active revision/precision selection, delete with blockers, pending states |

Browser-only, in `UI/Screens/`: `CuratedView`, `InstalledView`, and Phase 2's
`RemoteView`.

## Boundary rules

**R1 — the host screen owns the work; modals return decisions.** No widget or
modal calls `preflight`, `provision`, `activate`, or `delete`. They post intent
messages; the host screen runs the threaded worker and owns the operation's
lifetime. This is what TASK-1696 already does, and the reason is that a download
must survive its consent dialog being dismissed.

**R2 — `service` at module scope; `acquisition` and `fetch` only inside
functions.** `Tests/Model_Artifacts/test_credentials_and_boundaries.py:484` runs a
subprocess with an import-recording hook and fails if importing the STT worker
surface pulls in either module — catching even an import attempted and caught.
Consequently **`CuratedRegistry` must not inherit from `ArtifactCatalog`**: it
satisfies the Protocol structurally, importing it only under `TYPE_CHECKING`,
exactly as `ParakeetV2Catalog` documents. Subclassing it is a module-scope import
of `acquisition` and turns that test red as soon as anything worker-side reaches
the registry.

**R3 — an `ArtifactRef` is never constructed from user input.** Refs come from the
registry or the inventory. Phase 2's resolution step mints refs from a fetched
repository listing, not from typed text.

**R4 — no widget touches the filesystem.** The unmanaged scan is a threaded worker
in `InstalledView`.

## Acceptance criteria coverage

| AC | Where |
|---|---|
| #1 Curated / Remote / Installed over service and catalog interfaces | Phase 1 (Curated, Installed); Phase 2 (Remote) |
| #2 precise provenance labels, never implying malware safety | Phase 1, via the new `ArtifactPreflightEntry.provenance` and descriptor provenance |
| #3 install confirmation shows the full plan | Phase 1, `ModelPlanPanel` from `PreflightReport` |
| #4 installed inventory: revisions, dependencies, installed vs staging space, blockers | Phase 1, `InstalledView` + `disk_usage()` |
| #5 deletion may request an idle recycle; never bypasses a lease | Phase 1 delivers the blocker reporting; **the recycle request is deferred** pending a pool-owner mechanism |
| #6 off-event-loop work, bounded results, focused UI tests | Phase 1 for local work; Phase 2 for remote search |
| #7 select and persist active revision and precision | Phase 1, `ModelActivationControls` over `activate()` |
| #8 controls reusable by Settings and onboarding | Phase 1, proven by refactoring Library onto `ModelInstallModal` |

## Migration notes

- TASK-1696's tests pin `#parakeet-v2-install-modal`, `#parakeet-v2-install-modal-confirm`,
  and `#parakeet-v2-install-modal-cancel`. Refactoring Library onto the shared
  modal either preserves these ids or updates those tests deliberately; they are
  the regression net for the flow being refactored.
- `local_models_widget.py`'s threaded `os.walk` scan and delete flow are absorbed
  by `InstalledView` rather than rewritten.
- Phase 1 removes the "Local Models" rail row. "Download Models" and the whole of
  `Widgets/HuggingFace/` stay until Phase 3.

## Risks

**AC #5's recycle has no mechanism.** The riskiest criterion in the task is not UI
work. Phase 1 reports blockers honestly and defers the recycle request; if it is
required sooner, it needs its own design against the heavy-worker pool owner.

**Phase 2's honesty problem.** HF search returns repositories; installing needs a
pinned revision and per-file digests. If results are listed without resolution,
most of them are dead ends discovered only at install time. Resolution must
happen on selection, before the plan is shown — this is the hard part of Phase 2
and is designed there, not assumed away here.

**Phase 3 touches every local-server launch path.** Kept last deliberately.

## Testing

- **Pure view-model:** plain pytest over `model_browser_state` — plan rows from
  synthetic reports, inventory rows including `descriptor is None`, every branch
  of `install_failure_message` asserting the mapped text is present and raw
  exception text is absent.
- **Widgets:** Pilot tests for plan rendering, the disabled-Install gate on a
  non-grantable report, progress phases, and activation pending state.
- **Boundaries:** extend the existing import-boundary test to the new registry
  module; assert no view performs I/O at compose time.
- **Regression:** the TASK-1696 Parakeet flow keeps passing through the shared
  modal.
