# TASK-2062.2 Managed and External GGUF Source Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let llama.cpp launch an exact managed GGUF or an arbitrary external GGUF, and let llamafile launch Embedded, Managed GGUF, or External GGUF, without forcing external files into Chatbook's store.

**Architecture:** Keep source choice as transient `LLMManagementWindow` state, represented by one small pure source-state module. Extend the existing identity-bearing `ServerLaunchClaim` with one optional closable lease owner and a path-free authority label; launch workers validate external GGUFs in place or acquire an exact managed `ArtifactRef`, transfer the lease to the current claim, and then use the incumbent subprocess lifecycle for spawn, Stop, stubborn retention, and exact-generation cleanup.

**Tech Stack:** Python 3.11+, stdlib `dataclasses`/`enum`/`pathlib`/`threading`/`subprocess`, existing `ModelArtifactService`, existing bounded GGUF admission, Textual 8.x workers/Select/Pilot, pytest, Ruff, GitHub Actions.

## Global constraints

- Read `AGENTS.md`, TASK-2062.2, the approved TASK-2062 design, amended ADR-025, `backlog/docs/lessons-testing-evidence.md`, `backlog/docs/lessons-live-verification.md`, and `backlog/docs/lessons-backlog-hygiene.md` before editing.
- Use strict TDD: write each focused test first, run it to a genuine RED for the intended missing behavior, then add only the minimum production code needed for GREEN.
- llama.cpp has exactly **Managed GGUF** and **External GGUF** modes. llamafile has exactly **Embedded**, **Managed GGUF**, and **External GGUF** modes.
- External GGUF is permanent and first-class. It launches the user-owned file in place without import, copy, activation, deletion, managed-store write, or global source selection.
- UI/controller state may retain an external `Path` in memory, but no new durable absolute-path setting is added. Existing model-path values map to External; blank llamafile maps to Embedded.
- Managed UI state stores only an exact `ArtifactRef`. Resolve a managed payload path only in the launch worker after exact acquisition.
- A managed lease transfers atomically to the exact `ServerLaunchClaim` before spawn and remains there until that exact claim proves process death. A stubborn process retains both claim and lease.
- Detach a claim-owned resource under the lifecycle lock; close it outside the lock, idempotently and exactly once. A stale claim cannot detach or close a current claim's resource.
- External validation reuses the current no-follow/reparse-point `open_local_gguf()` and bounded `inspect_gguf_structure()` boundary off the Textual loop. It performs a final identity recheck immediately before spawn and never hashes tensor payload.
- Claim authority is path-free (`Embedded`, `Managed GGUF`, or `External GGUF`). Commands, paths, stderr, exception strings, and managed storage layout never enter logs, notifications, worker descriptions, or stable status.
- When a claim is pending or its process is live, Start and all source controls are frozen; Stop addresses only the exact current claim. Progress/status updates do not replace focused Stop.
- vLLM and MLX code, state, source contracts, and launch commands remain unchanged.
- Do not create a source-controller framework, lease manager, persisted managed path, compatibility database, automatic import, or second artifact inventory service.
- Regenerate `tldw_chatbook/css/tldw_cli_modular.tcss` from source CSS; never hand-edit its generated rules.
- Current `origin/dev` baseline note: the pre-change three-file mounted probe produced `2 failed, 22 passed, 7 errors`; the failures are stale Models rail/tooltip expectations and the errors are the existing periodic Ollama loopback probe colliding with the test network guard. New TASK-2062.2 tests must isolate that timer/probe and pass independently. Broader gates must report those baseline nodes honestly unless upstream fixes them.
- ADR required: yes. ADR path: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` (already amended for TASK-2062; do not create a duplicate ADR).

---

## File responsibility map

- `tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py` — pure source mode/selection types, deterministic legacy mapping, managed inventory filtering/display, exact managed payload resolution, and path-private source error taxonomy.
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py` — app-owned claim authority, atomic optional-resource transfer, exact-generation resource settlement, process spawn/Stop/stubborn lifetime.
- `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py` — llama.cpp/llamafile picker and start workers, external bounded inspection, managed acquisition/transfer, command construction, and path-private failure delivery.
- `tldw_chatbook/UI/LLM_Management_Window.py` — transient source selections, managed inventory worker, mutually exclusive mode controls, authority/status presentation, source-control fencing, and compatibility initialization.
- `tldw_chatbook/css/features/_llm-management.tcss` — compact source rows, selector/action containment, wrapping authority copy, and 80-column layout.
- `tldw_chatbook/css/tldw_cli_modular.tcss` — generated CSS bundle only.
- `Tests/LLM_Management/test_gguf_source_modes.py` — pure mode, compatibility, inventory, payload, privacy, and inactive-value tests.
- `Tests/LLM_Management/test_server_lifecycle_resources.py` — claim resource transfer and every terminal/stubborn/stale lifecycle.
- `Tests/LLM_Management/test_gguf_server_sources.py` — handler/worker source matrices, external store-free validation, managed exact-ref launch, cancellation, and command privacy.
- `Tests/UI/test_llm_gguf_source_modes.py` — mounted real-CSS source controls, 80-column geometry, keyboard access, focus identity, compatibility, and lifecycle fencing.
- `Tests/CI/test_task2062_2_gguf_source_evidence.py` — exact workflow shape and required-node ratchet.
- `.github/workflows/task-2062-2-gguf-source-evidence.yml` — read-only exact-head Linux/macOS/Windows evidence lane.
- `backlog/tasks/task-2062.2 - Add-managed-and-external-GGUF-source-modes-to-llama.cpp-and-llamafile.md` — plan link, checked ACs, final implementation notes, evidence, and status.

## Acceptance-criteria traceability

| Acceptance criterion | Planned evidence |
|---|---|
| AC1 llama.cpp Managed/External | Tasks 1, 3, and 4 pure matrix, worker command, and mounted mode tests |
| AC2 llamafile Embedded/Managed/External | Tasks 1, 3, and 4 exact three-mode and no-`-m` Embedded tests |
| AC3 arbitrary external remains in-place | Tasks 1, 3, and 4 store-spy, unchanged-byte/mtime, no-import/delete/activate/global-selection, and physical picker tests |
| AC4 exact ref and process-lifetime lease | Tasks 2, 3, and 5 transfer, spawn failure, normal/failed exit, Stop, stubborn, deletion-race, and stale-generation tests |
| AC5 authority and fencing | Tasks 2–4 claim authority, disabled controls, stale callback, replacement, and focused Stop tests |
| AC6 compatibility; vLLM/MLX unchanged | Tasks 1, 3, and 5 legacy mapping, exact command snapshots, and incumbent provider regression tests |
| AC7 80-column/native lifecycle | Tasks 4 and 5 production-CSS compositor/keyboard tests plus exact three-OS workflow |

### Task 1: Add the pure GGUF source-state boundary

**Files:**
- Create: `tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py`
- Create: `Tests/LLM_Management/test_gguf_source_modes.py`

**Interfaces:**
- Consumes: `ArtifactRef`, `ArtifactDescriptor`, `ArtifactFormat`, `ArtifactRole`, `InstalledArtifact`, `LeasedArtifactHandle`, and `ModelArtifactService`.
- Produces: `GGUFSourceMode`, `GGUFSourceSelection`, `ManagedGGUFChoice`, `initial_gguf_selection()`, `managed_gguf_choices()`, `acquire_managed_gguf()`, and `gguf_source_failure_message()`.
- Preserves: external path is transient and repr-hidden; managed selection contains an exact ref and no path.

- [ ] **Step 1: Write failing mode, compatibility, and privacy tests**

Add exact tests for:

```python
def test_legacy_source_values_map_without_importing():
    assert initial_gguf_selection("llamacpp", "outside.gguf").mode is GGUFSourceMode.EXTERNAL
    assert initial_gguf_selection("llamafile", "outside.gguf").mode is GGUFSourceMode.EXTERNAL
    assert initial_gguf_selection("llamafile", "").mode is GGUFSourceMode.EMBEDDED


def test_source_selection_preserves_inactive_values_without_exposing_path():
    selected = GGUFSourceSelection(
        mode=GGUFSourceMode.EXTERNAL,
        managed_ref=REF,
        external_path=Path("/private/sentinel.gguf"),
    )
    assert selected.for_mode(GGUFSourceMode.MANAGED).managed_ref == REF
    assert selected.for_mode(GGUFSourceMode.MANAGED).external_path.name == "sentinel.gguf"
    assert "/private/sentinel.gguf" not in repr(selected)
```

Cover invalid provider/mode combinations: llama.cpp rejects Embedded; llamafile accepts all three; blank llama.cpp remains External but invalid at launch rather than silently becoming Managed.

- [ ] **Step 2: Run the mode tests and record genuine RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_source_modes.py -k 'legacy or selection or mode'
```

Expected: import/collection failure because `gguf_source_modes.py` does not exist.

- [ ] **Step 3: Implement the minimum immutable state types**

Use `StrEnum` and frozen dataclasses; do not add a controller class:

```python
class GGUFSourceMode(StrEnum):
    EMBEDDED = "embedded"
    MANAGED = "managed"
    EXTERNAL = "external"


@dataclass(frozen=True)
class GGUFSourceSelection:
    mode: GGUFSourceMode
    managed_ref: ArtifactRef | None = None
    external_path: Path | None = field(default=None, repr=False)

    @property
    def authority(self) -> str:
        return {
            GGUFSourceMode.EMBEDDED: "Embedded",
            GGUFSourceMode.MANAGED: "Managed GGUF",
            GGUFSourceMode.EXTERNAL: "External GGUF",
        }[self.mode]

    def for_mode(self, mode: GGUFSourceMode) -> "GGUFSourceSelection":
        return replace(self, mode=mode)
```

`initial_gguf_selection(provider, existing_model_path)` implements only the approved deterministic mapping. Validate exact provider names and never touch filesystem/store state.

- [ ] **Step 4: Write failing managed-inventory and exact-payload tests**

Build mixed `InstalledArtifact` rows and assert:

- only ready root `ArtifactFormat.GGUF` entries are selectable;
- broken, dependency, non-GGUF, and not-ready entries are excluded;
- label contains descriptor `model_id`, `precision`, size, and `Managed · local integrity recorded` for local provenance;
- choice contains exact `ArtifactRef` and no managed path;
- `acquire_managed_gguf(service, REF)` calls `service.acquire(REF)`, returns the single declared root GGUF payload and the still-open leased handle;
- missing/corrupt/not-ready/multiple-GGUF shapes close the handle and raise a stable typed error;
- recursive string/repr scan excludes managed root paths and raw exceptions.

- [ ] **Step 5: Run the managed tests and record RED**

```bash
../../.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_source_modes.py -k 'managed or inventory or payload or failure'
```

Expected: failures for missing inventory/resolution functions.

- [ ] **Step 6: Implement inventory filtering and exact acquisition**

`managed_gguf_choices(installed)` is a pure tuple transformation. `acquire_managed_gguf(service, reference)`:

1. requires exact `ArtifactRef`;
2. calls `service.acquire(reference)` first, so authoritative deletion is blocked;
3. finds the matching exact ready root descriptor from `service.list_installed()`;
4. requires GGUF format and exactly one declared `.gguf` file;
5. resolves that relative declared filename beneath `dict(leased.handle.paths)[reference]`;
6. returns `(payload_path, leased)` without closing on success;
7. closes on every pre-return failure.

Do not add a service method or expose the path to UI code. Map known `ArtifactNotReadyError`, `ArtifactIntegrityError`, `ArtifactStateError`, and GGUF admission errors to stable path-free copy; unknown failures receive one generic recovery message.

- [ ] **Step 7: Run, mutate, restore, and commit**

```bash
../../.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_source_modes.py
```

Temporarily remove the exact-ready/root/GGUF predicate; the mixed-inventory test must fail. Restore and rerun GREEN.

```bash
git add tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py Tests/LLM_Management/test_gguf_source_modes.py
git commit -m "feat(models): define GGUF runtime source state"
```

### Task 2: Transfer one managed lease into the existing server claim

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py`
- Create: `Tests/LLM_Management/test_server_lifecycle_resources.py`

**Interfaces:**
- Extends: `ServerLaunchClaim` with `authority: str | None` and repr-hidden `_resource: object | None`.
- Produces: `attach_server_claim_resource(app, provider, claim, resource) -> bool`.
- Changes: `reserve_server_launch(..., authority=None)`; existing callers remain source-compatible.
- Preserves: existing provider/process identity and vLLM/MLX/Ollama behavior.

- [ ] **Step 1: Write failing atomic-transfer and stale-claim tests**

Use a fake closable resource with a close count. Assert:

```python
claim = reserve_server_launch(app, "llamacpp", authority="Managed GGUF")
assert attach_server_claim_resource(app, "llamacpp", claim, lease) is True
assert claim.authority == "Managed GGUF"
assert "private" not in repr(claim)

assert attach_server_claim_resource(app, "llamacpp", stale, stale_lease) is False
assert stale_lease.close_count == 0  # ownership never transferred
assert current_lease.close_count == 0
```

Also reject second resource attachment and cancellation-before-transfer. The worker remains responsible for closing a resource when transfer returns false.

- [ ] **Step 2: Run and record genuine RED**

```bash
../../.venv/bin/python -m pytest -q Tests/LLM_Management/test_server_lifecycle_resources.py -k 'attach or stale or authority'
```

Expected: missing signature/helper/field failures.

- [ ] **Step 3: Implement claim metadata and atomic attachment**

Add only:

```python
@dataclass(eq=False)
class ServerLaunchClaim:
    provider: str
    authority: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    _resource: Any | None = field(default=None, repr=False)
```

`attach_server_claim_resource` runs under `_lock(app)` and succeeds only for the exact current uncancelled claim with no existing resource. It does not close on rejection.

- [ ] **Step 4: Write failing terminal-lifecycle matrix**

Cover all exact outcomes:

- cancelled before spawn closes once when claim releases;
- `Popen` failure closes once;
- publication failure with process proven dead closes once;
- normal exit and failed exit close once after exact death;
- successful Stop closes once after death;
- stubborn cancelled process retains claim/resource while `poll() is None` and blocks a simulated artifact delete;
- later proven death clears exact process/claim and closes once;
- stale `release_server_claim`/`clear_server_process` cannot close current or stale resources;
- resource `close()` failure cannot leave lifecycle state wedged and is reported only by stable category.

Measure resource ownership directly, not only `server_is_active()`.

- [ ] **Step 5: Run matrix and record RED**

```bash
../../.venv/bin/python -m pytest -q Tests/LLM_Management/test_server_lifecycle_resources.py
```

Expected: resource remains open after terminal claims because settlement does not own it yet.

- [ ] **Step 6: Settle identity-checked resources outside the lock**

Refactor `release_server_claim()` and `clear_server_process()` through one private lock-held detach helper returning the exact resource. Remove claim/process state under the lock, then call `close()` after leaving the lock. Keep their public boolean return contract. `retain_cancelled_server_process()` must not detach anything. `run_server_subprocess()` continues using these existing functions, so every current provider retains identical behavior when `_resource is None`.

- [ ] **Step 7: Mutation-test stubborn retention and stale cleanup**

Temporarily close the resource in `run_server_subprocess()`'s worker `finally` regardless of `retained`; the stubborn test must fail. Restore. Temporarily remove exact claim identity from detach; the stale-generation test must fail. Restore.

- [ ] **Step 8: Run incumbent lifecycle regressions and commit**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/LLM_Management/test_server_lifecycle_resources.py \
  Tests/ProductionApp/test_llm_destination_actions.py::test_server_lifecycle_is_app_owned_and_root_worker_handler_is_retired \
  Tests/ProductionApp/test_llm_destination_actions.py::test_bounded_termination_kills_and_reaps_when_terminate_raises
```

```bash
git add tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py Tests/LLM_Management/test_server_lifecycle_resources.py
git commit -m "feat(models): retain managed GGUF leases for server claims"
```

### Task 3: Launch llama.cpp and llamafile from the selected authority

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py`
- Test: `Tests/LLM_Management/test_gguf_server_sources.py`

**Interfaces:**
- Consumes: `GGUFSourceSelection`, `acquire_managed_gguf()`, `open_local_gguf()`, `inspect_gguf_structure()`, and `attach_server_claim_resource()`.
- Changes: llama.cpp/llamafile start workers receive immutable source snapshots instead of a prebuilt model path command.
- Preserves: executable/host/port/additional-argument behavior and existing `run_server_subprocess()` owner.

- [ ] **Step 1: Write the failing exact source-mode command matrix**

Parameterize:

| Provider | Source | Required command |
|---|---|---|
| llama.cpp | Managed | `--model <leased payload>` |
| llama.cpp | External | `--model <selected external path>` |
| llamafile | Embedded | no `-m` and no model path |
| llamafile | Managed | `-m <leased payload>` |
| llamafile | External | `-m <selected external path>` |

Assert inactive managed refs/external paths never enter the command. Snapshot vLLM and MLX command builders before/after and assert unchanged.

- [ ] **Step 2: Write failing external-boundary tests**

Use a valid sparse GGUF outside the managed root and spies that fail if `managed_service`, `list_installed`, `acquire`, `activate`, `delete`, or import are called. Assert:

- validation and final identity recheck execute on the worker thread, not the event-loop thread;
- malformed, missing, special, symlink/reparse, and replacement files fail before `Popen`;
- selected bytes and mtime remain unchanged;
- successful launch passes the same outside path to `Popen`;
- no descriptor, active selector, manifest, or config is written;
- notices/logs/worker description omit the path, raw command, stderr, and exception string.

- [ ] **Step 3: Run the matrix/boundary and record RED**

```bash
../../.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_server_sources.py -k 'matrix or external or embedded'
```

Expected: current handlers require a model path for llamafile, do UI-loop `Path.is_file`, and have no managed source seam.

- [ ] **Step 4: Move source preparation into the existing thread worker**

Keep the current public button handlers. They capture executable, host, port, arguments, and `window.gguf_source_snapshot(provider)`, reserve the claim with `authority=selection.authority`, immediately sync controls, and schedule one static-description app-owned thread worker.

The worker does exactly one source branch:

```python
if selection.mode is GGUFSourceMode.EMBEDDED:
    model_path = None
elif selection.mode is GGUFSourceMode.EXTERNAL:
    with open_local_gguf(selection.external_path) as opened:
        inspect_gguf_structure(opened.handle, file_size=opened.identity.size_bytes)
        opened.recheck()
        model_path = opened.path
else:
    model_path, leased = acquire_managed_gguf(managed_service(), selection.managed_ref)
    if not attach_server_claim_resource(app, provider, claim, leased):
        leased.close()
        return stable_cancelled_result
```

Build only the selected provider's command, then call `run_server_subprocess()`. Embedded llamafile omits `-m`. If source preparation fails, release the exact claim, deliver only stable path-free recovery to the current mounted window, and leave the selection intact.

- [ ] **Step 5: Write failing managed ownership/cancellation tests**

Assert exact ref acquisition, transfer-before-`Popen`, failed transfer worker-close, pre-spawn Cancel, spawn failure, successful Stop, stubborn retention, and deletion race. In the deletion race, `service.delete(ref)` must report in-use until exact process death, then succeed. A stale prior worker completion cannot release a newer claim/lease or overwrite status.

- [ ] **Step 6: Run, mutate, restore, and commit**

```bash
../../.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_server_sources.py Tests/LLM_Management/test_server_lifecycle_resources.py
```

Required mutations:

1. validate external on the async handler instead of the worker — heartbeat/thread test fails;
2. transfer managed lease after `Popen` — transfer-order/deletion-race test fails;
3. include an inactive external path — command-matrix test fails;
4. require llamafile model in Embedded — Embedded test fails.

Restore every mutation and rerun GREEN.

```bash
git add tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py Tests/LLM_Management/test_gguf_server_sources.py
git commit -m "feat(models): launch GGUF servers from explicit sources"
```

### Task 4: Render and fence source controls in the Models destination

**Files:**
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py`
- Modify: `tldw_chatbook/css/features/_llm-management.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Create: `Tests/UI/test_llm_gguf_source_modes.py`

**Interfaces:**
- Owns: one `GGUFSourceSelection` per provider, one inventory generation, and managed choice labels.
- Produces: `gguf_source_snapshot(provider)`, mode/path/managed selection handlers, and one generation-fenced inventory worker.
- Reads: app-owned current `ServerLaunchClaim.authority` to render pending/running authority after screen replacement.

- [ ] **Step 1: Write failing mounted source matrix and compatibility tests**

Mount `TldwCli` with the periodic Ollama probe stubbed before mount. Under the real `TldwCli.CSS_PATH`, assert:

- llama.cpp shows exactly Managed GGUF / External GGUF;
- llamafile shows exactly Embedded / Managed GGUF / External GGUF;
- llama.cpp initial state is External;
- llamafile blank initial path is Embedded;
- a legacy/nonblank path maps to External and remains unchanged;
- switching modes preserves both inactive external path and managed exact ref in window memory;
- managed selector values are exact refs, labels show name/variant/size/provenance, and no managed absolute path appears in recursive widget/render text.

- [ ] **Step 2: Run and record genuine RED**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_llm_gguf_source_modes.py -k 'matrix or compatibility or preserves or selector'
```

Expected: missing mode/managed controls and state APIs.

- [ ] **Step 3: Compose the minimum source controls**

Use one compact `Select` for source mode and one mode-owned source region per provider:

- `Select` choices use path-free labels and enum string values;
- Managed region contains a `Select` of exact refs plus `Refresh managed models`;
- External region retains the current Input/Browse control and the exact authority copy:
  `Outside Chatbook · integrity unknown` and
  `This file is used in place and is not imported, copied, deleted, or selected globally.`;
- Embedded region contains only short explanatory copy;
- one always-mounted path-free status `Static` reports selected/pending/running authority and recovery.

Do not recompose for progress/status-only changes. Recompose only when the selected mode changes which region is displayed or inventory choices change.

- [ ] **Step 4: Load managed inventory off-loop with a generation fence**

Start a static-description thread worker on first activation of llama.cpp/llamafile and explicit Refresh. The worker constructs `managed_service()`, calls `list_installed()`, and applies `managed_gguf_choices()`. Apply only when window is still the current destination and generation matches. Store only choices/refs, never paths. Store failure disables only Managed mode/selector; External and Embedded stay usable.

- [ ] **Step 5: Write failing lifecycle-fencing and focus tests**

Hold the source worker/launch at deterministic gates and assert:

- accepted Start immediately disables Start, mode, managed selector/refresh, external input/Browse, executable source controls that would alter the launch, and enables Stop;
- visible authority comes from the app-owned claim and survives `LLMScreen.recompose()`;
- physical Stop cancels only the current claim;
- stale inventory/launch callbacks cannot alter newer selection/status;
- process death restores controls and focus to Start or the selected recovery control;
- repeated pending/running status refresh preserves the exact focused Stop widget.

- [ ] **Step 6: Add 80-column production-CSS keyboard/geometry evidence**

At 80x24 (and enough vertical room through scroll), physically Tab/Enter through both providers. Assert:

- mode, selector/Browse, Start, and Stop are compositor-painted and inside their real parent/content bounds;
- long managed labels wrap/truncate without pushing actions outside the shell;
- External authority copy is visible and path-free;
- Stop remains the same widget identity and keyboard-reachable while pending/running;
- vLLM/MLX view controls and values are byte-for-byte/behaviorally unchanged.

Use `app.export_screenshot()` for user-visible text evidence, not computed styles alone.

- [ ] **Step 7: Implement narrow CSS and regenerate the bundle**

Add only `.gguf-source-*` rules in `tldw_chatbook/css/features/_llm-management.tcss`: `1fr auto`/horizontal rows where safe, full-width stacked fallback at narrow content, auto-height wrapped authority/status, and no fixed widths wider than the content shell.

```bash
../../.venv/bin/python tldw_chatbook/css/build_css.py
```

Review generated selector deltas; do not hand-edit the bundle.

- [ ] **Step 8: Run, mutate, restore, and commit**

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_llm_gguf_source_modes.py Tests/LLM_Management/test_gguf_source_modes.py Tests/LLM_Management/test_gguf_server_sources.py
```

Required mutations:

1. omit `server_is_active`/claim fencing from one source control — mounted pending test fails;
2. render authority from current window selection instead of claim — replacement test fails;
3. unconditional status recompose — focused Stop identity test fails;
4. remove External authority copy — frame test fails.

Restore and rerun GREEN.

```bash
git add tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/css/features/_llm-management.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_llm_gguf_source_modes.py
git commit -m "feat(models): expose GGUF runtime source modes"
```

### Task 5: Pin native lifecycle evidence and regressions

**Files:**
- Create: `.github/workflows/task-2062-2-gguf-source-evidence.yml`
- Create: `Tests/CI/test_task2062_2_gguf_source_evidence.py`
- Modify only if the focused assertions require compatibility updates: `Tests/ProductionApp/test_llm_destination_actions.py`

- [ ] **Step 1: Write the failing workflow-shape test**

Assert the workflow has only `pull_request` to `dev` and `workflow_dispatch`, `contents: read`, Python 3.12, `fail-fast: false`, exactly `ubuntu-latest`, `macos-latest`, `windows-latest`, a bounded timeout, exact-head checkout, and only the required explicit node IDs.

Required nodes include:

- external no-follow/reparse/replacement and store-free launch;
- managed exact-ref acquisition/transfer;
- pre-spawn cancel and spawn failure;
- normal/failed exit;
- physical Stop;
- stubborn resource retention/deletion block;
- exact-death release/deletion success;
- stale claim cannot release newer resource;
- Embedded llamafile omits model argument;
- 80-column authority/focus/control containment;
- vLLM/MLX unchanged command regressions.

- [ ] **Step 2: Run and record RED**

```bash
../../.venv/bin/python -m pytest -q Tests/CI/test_task2062_2_gguf_source_evidence.py
```

Expected: workflow missing.

- [ ] **Step 3: Add the minimal read-only three-OS workflow**

Model it after `.github/workflows/task-2062-1-gguf-import-evidence.yml`. Do not add secrets, caches, model downloads, live runtimes, broad test directories, or write permissions. Install only editable package plus pytest/pytest-asyncio/pytest-timeout; run exact explicit nodes with `--timeout=60 -q`.

- [ ] **Step 4: Run the full focused local union**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/LLM_Management/test_gguf_source_modes.py \
  Tests/LLM_Management/test_server_lifecycle_resources.py \
  Tests/LLM_Management/test_gguf_server_sources.py \
  Tests/UI/test_llm_gguf_source_modes.py \
  Tests/CI/test_task2062_2_gguf_source_evidence.py \
  Tests/Model_Artifacts/test_service.py \
  Tests/Model_Artifacts/test_gguf_admission.py
```

Run exact incumbent vLLM/MLX handler suites and the isolated production lifecycle nodes. Do not hide the documented pre-existing Models rail/tooltip/Ollama-probe baseline failures; if upstream has not fixed them, record exact node IDs separately.

- [ ] **Step 5: Run static/privacy gates**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py \
  tldw_chatbook/UI/LLM_Management_Window.py \
  Tests/LLM_Management/test_gguf_source_modes.py \
  Tests/LLM_Management/test_server_lifecycle_resources.py \
  Tests/LLM_Management/test_gguf_server_sources.py \
  Tests/UI/test_llm_gguf_source_modes.py \
  Tests/CI/test_task2062_2_gguf_source_evidence.py
../../.venv/bin/python -m py_compile \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/gguf_source_modes.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py \
  tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events.py \
  tldw_chatbook/UI/LLM_Management_Window.py
git diff --check
```

Run Ruff format checks only on changed logical ranges when legacy full-file debt exists. Scan added lines and recursive captured UI/log/status/result structures for machine-local paths, commands, stderr, exception strings, managed paths, fabricated URLs, and persisted absolute-path settings.

- [ ] **Step 6: Run Impeccable and Ponytail review**

Run the repository's Impeccable detector exactly once on the changed production UI/CSS targets. Apply only concrete findings. Then review the whole diff under Ponytail: retain the one pure source module and one claim resource; delete any second controller, duplicated lease owner, speculative provider abstraction, or durable setting.

- [ ] **Step 7: Commit workflow/evidence shape**

```bash
git add .github/workflows/task-2062-2-gguf-source-evidence.yml Tests/CI/test_task2062_2_gguf_source_evidence.py
git commit -m "ci(models): verify GGUF runtime source lifecycles"
```

### Task 6: Final review, exact-head native evidence, and task closeout

**Files:**
- Modify: `Docs/superpowers/plans/2026-08-13-task-2062-2-gguf-source-modes.md` only for checked execution boxes/deviations if project practice requires it.
- Modify: `backlog/tasks/task-2062.2 - Add-managed-and-external-GGUF-source-modes-to-llama.cpp-and-llamafile.md`
- Create only if evidence needs a durable artifact: `Docs/Model_Evaluation/task-2062-2/README.md`

- [ ] **Step 1: Perform independent correctness and minimality review**

Review the exact implementation range against every AC and the approved spec. Rank Critical/Important/Minor issues. In particular re-audit:

- resource detach/close outside lock and exactly once;
- stubborn process retains lease;
- Stop/pre-spawn cancellation and stale generations;
- external path never enters managed service or deletion;
- active authority survives window replacement;
- External and Embedded remain usable when managed store fails;
- no vLLM/MLX change;
- 80-column physical keyboard/focus.

Fix every actionable issue with a genuine RED before production edits, rerun affected mutations, and commit corrections separately.

- [ ] **Step 2: Rebase onto current `origin/dev` and rerun frozen local gates**

Fetch, verify global Backlog IDs/task dependency, rebase, regenerate CSS if it conflicts, and rerun the exact focused union/static/privacy gates. Freeze the executable SHA only when the worktree is clean and gates are green apart from explicitly unchanged baseline failures.

- [ ] **Step 3: Push, open the PR, and run exact native evidence**

Push the branch, open a ready PR to `dev`, and confirm the TASK-2062.2 workflow binds to the frozen `headSha`. Wait for all three exact OS lanes. A native RED stops closeout: capture exact node/trace, diagnose, obtain review, TDD-fix, freeze a new SHA, and trigger a new full three-lane run rather than retrying only one lane.

- [ ] **Step 4: Address every PR review comment**

Inspect all review threads/checks, verify each finding technically, and TDD-fix every actionable issue. Resolve comments only after the corresponding pushed commit and fresh evidence. Rebase again if `dev` moved; rerun exact local/native gates on the final reviewed SHA.

- [ ] **Step 5: Close the Backlog task only after complete evidence**

Through Backlog CLI, check all seven ACs, add concise Implementation Notes covering approach/files/trade-offs/tests/native evidence, record ADR-025 as the governing ADR, and set TASK-2062.2 Done. Add a lessons entry only if implementation surfaced a genuinely generalizable incident; do not invent one.

- [ ] **Step 6: Commit closeout, push, merge, and clean the worktree**

Commit only the task/evidence docs, validate final task state and diff boundaries, push, ensure final PR checks/reviews are green, and merge to `dev`. Verify the merge commit contains TASK-2062.2, then remove the completed worktree only after confirming it is clean and merged.
