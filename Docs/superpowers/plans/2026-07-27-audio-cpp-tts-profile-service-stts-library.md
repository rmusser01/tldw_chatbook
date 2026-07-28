# audio.cpp TTS Profile Service and STTS Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users save successful native audio.cpp Playground generations as exact reusable profiles and manage them through a bounded STTS library without adding character assignment, roleplay routing, portability, legacy-profile execution, or managed-server behavior.

**Architecture:** Extend the existing native adapter boundary with structured voice observation, then let the app-owned `TTSService` produce one revision-coherent bounded capability snapshot through its existing writer-preferred admission coordinator. Add one app-owned `TTSProfileService` over the already-merged task-763 repository, preserve exact native request provenance in successful Playground artifacts, and mount a focused STTS profile-library widget that reuses the existing Playground for preview and generation. Repository lifecycle generations, provider-configuration revisions, catalog revisions, and UI request generations each remain explicit and are checked at their owning boundary.

**Tech Stack:** Python 3.12, asyncio, frozen dataclasses and protocols, httpx, SQLite through the existing serialized `TTSProfileRepository`, Textual 8, pytest/pytest-asyncio, Ruff, mypy, Backlog.md.

**Task:** `TASK-951`

**Dependency:** `TASK-763` / PR #977

**ADR required:** yes
**ADR paths:** `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md` and `backlog/decisions/028-character-tts-generation-profile-ownership.md`
**Reason:** Both existing ADRs were amended before this plan. ADR-023 governs exact request provenance and structured native capability observation; ADR-028 governs the app-owned profile service and expected repository-generation mutation contract. No new ADR, schema migration, store, dependency, or process-runtime decision is introduced.

---

## Scope boundary

This plan implements only approved Slice 2B:

- native audio.cpp requested-selection provenance for successful Playground
  artifacts;
- structured audio.cpp voice discovery before tuple projection or failure
  collapse;
- one bounded revision-coherent native capability snapshot;
- generation-safe profile create, update, duplicate, delete, list/search, and
  availability services;
- one lazy app-owned profile service;
- a focused STTS profile library and editor;
- exact remount-safe preview through the existing Playground;
- save-result-as-profile for successful native audio.cpp artifacts;
- bounded search, capability enrichment, cancellation, and stale-result
  suppression;
- minimal non-mutating repair guidance.

It does **not** implement:

- character identity or assignment UI;
- a character request resolver or roleplay routing;
- automatic speech or assigned-profile execution;
- character-card or standalone profile import/export;
- legacy-provider profile creation or execution;
- another TTS service, adapter registry, player, generation handler, artifact
  store, executor, or SQLite connection;
- audio.cpp binary discovery, launch, supervision, restart, or shutdown.

## Rebased baseline

The planning branch was rebased cleanly onto `origin/dev` at
`c171ae56a`, which already contains task 763 through merge commit
`e3a020f74`.

The first baseline command accidentally used Apple Python 3.9 and failed
collection because the project requires Python 3.11+. A task-local ignored
`.venv` was then created with Python 3.12.11 and the declared `.[dev]`
dependencies. Ruff and mypy were installed into that ignored task environment
because the repository's `dev` extra does not declare either verification
tool.

Focused supported-interpreter baseline:

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/UI/test_stts_playground_audio_cpp.py -q
```

Result before task-951 code: `559 passed, 3 failed, 2 warnings`.

All three failures are the same pre-existing shared-fixture defect:
`Tests/UI/test_screen_navigation.py::fake_runtime_policy` assigns the now
read-only `TldwCli.current_runtime_backend` property while constructing the
app. The affected task-scoped tests are:

- `test_app_constructs_one_closed_pure_profile_repository`;
- `test_app_constructs_one_tts_service`; and
- `test_app_construction_keeps_audio_cpp_import_and_all_adapters_lazy`.

Task 951 does not change that property or fixture. Final evidence must report
these baseline failures separately; it must not claim a wholly green gate,
silently fix the unrelated fixture, or treat a new failure as baseline.

A broad Ruff preflight over the inherited TTS/STTS tree reports 1,065 existing
findings, including legacy backends, tests, and `UI/STTS_Window.py`; even the
smaller set of existing TTS modules changed by this task has 107 inherited
findings under the current all-rules configuration. Task 951 therefore runs
full Ruff and Ruff format over every new file, fatal-error Ruff checks over
modified legacy files, mypy over the focused TTS service boundary, and
`git diff --check`. It must not create a thousand-line unrelated lint cleanup
or claim the inherited broad Ruff baseline is green.

## File responsibility map

| File | Responsibility |
| --- | --- |
| `backlog/tasks/task-951 - Add-audio.cpp-TTS-profile-service-and-STTS-library.md` | Atomic Slice 2B acceptance criteria, plan summary, and final evidence |
| `Docs/superpowers/plans/2026-07-27-audio-cpp-tts-profile-service-stts-library.md` | Executable task-951 plan |
| `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md` | Approved Slice 2B architecture and scope |
| `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md` | Native capability and exact-admission decision |
| `backlog/decisions/028-character-tts-generation-profile-ownership.md` | Profile service and lifecycle-generation ownership decision |
| `tldw_chatbook/TTS/adapter_types.py` | Structured voice protocol/result and immutable native capability snapshot |
| `tldw_chatbook/TTS/adapters/audio_cpp.py` | Status-aware voice fetch, cache, coalescing, and tuple compatibility projection |
| `tldw_chatbook/TTS/TTS_Generation.py` | Public service methods for exact native synthesis, capability snapshots, and revision decisions |
| `tldw_chatbook/TTS/request_admission.py` | Writer-preferred exact-admission, capability-lease, and configuration-revision linearization |
| `tldw_chatbook/TTS/playground_types.py` | Text-free requested-selection provenance, exact preview preset, and optional artifact field |
| `tldw_chatbook/TTS/profile_repository.py` | Expected-generation admission for profile-derived mutations |
| `tldw_chatbook/TTS/profile_errors.py` | Safe profile-service error codes |
| `tldw_chatbook/TTS/profile_service.py` | Native allowlist validation, CRUD orchestration, availability, and preview preset construction |
| `tldw_chatbook/TTS/__init__.py` | Public task-951 domain/service exports |
| `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py` | Native exact synthesis provenance; legacy generation remains unchanged |
| `tldw_chatbook/app.py` | One lazy app-owned `TTSProfileService` over existing owners |
| `tldw_chatbook/UI/stts_profile_library.py` | Focused profile library, editor dialog, list pipeline, status, and actions |
| `tldw_chatbook/UI/stts_playground_catalog.py` | Pure exact-preset control projection without first/default substitution |
| `tldw_chatbook/UI/STTS_Window.py` | Sidebar mount, one-shot preset handoff, and save-profile button integration |
| `Tests/TTS/test_audio_cpp_adapter.py` | Adapter structured status, cache, coalescing, and cancellation |
| `Tests/TTS/test_tts_profile_capabilities.py` | Coherent snapshot bounds, deadline, catalog movement, and lease cleanup |
| `Tests/TTS/test_tts_request_admission.py` | Exact provenance and configuration-revision decision ordering |
| `Tests/TTS/test_profile_repository.py` | Expected-generation mutation admission |
| `Tests/TTS/test_profile_repository_lifecycle.py` | Same-UUID/revision restore races against loaded mutations |
| `Tests/TTS/test_stts_playground_types.py` | Provenance and preset immutability |
| `Tests/TTS/test_stts_audio_cpp_generation.py` | Native-only provenance and unchanged legacy generation |
| `Tests/TTS/test_profile_service.py` | Native allowlist, save/edit/duplicate/delete, availability, and stale generations |
| `Tests/TTS/test_tts_app_ownership.py` | One lazy profile service with no independent lifecycle owner |
| `Tests/UI/test_stts_profile_library.py` | List/search/editor/actions, coalescing, stale publication, and store failure |
| `Tests/UI/test_stts_playground_audio_cpp.py` | Exact remount-safe preset, no substitution, warned unverified attempt, and save action |
| `Docs/Development/TTS/TTS_MODULE_GUIDE.md` | Developer-facing service/capability/lifecycle contract |
| `Docs/Features/Speech-Services-Guide.md` | User-facing save/manage/preview/repair behavior and external-server boundary |

## Public contracts fixed by this plan

Use these shapes unless a red test proves a smaller equivalent is required.
Names may move only to avoid an import cycle; semantics may not weaken.

```python
VoiceDiscoveryState = Literal["complete", "model_missing", "unverified"]


@dataclass(frozen=True, slots=True)
class TTSVoiceDiscoveryResult:
    provider_id: str
    model_id: str
    catalog_revision: int
    voices: tuple[str, ...]
    state: VoiceDiscoveryState


@runtime_checkable
class TTSStructuredVoiceAdapter(Protocol):
    async def observe_voices(
        self,
        model_id: str,
        refresh: bool = False,
    ) -> TTSVoiceDiscoveryResult: ...


CapabilitySnapshotState = Literal["complete", "unverified"]


@dataclass(frozen=True, slots=True)
class TTSNativeCapabilitySnapshot:
    provider_id: str
    configuration_revision: int
    state: CapabilitySnapshotState
    catalog: TTSProviderCatalog | None
    voice_results: Mapping[str, TTSVoiceDiscoveryResult]
```

`voice_results` contains only distinct models requested because at least one
profile uses an exact non-null voice. Server-default profiles never cause a
voice request. A `complete` snapshot has one catalog revision shared by every
authoritative voice result. `unverified` may retain safe partial observations
for diagnostics, but callers may not treat them as proof of absence.

```python
@dataclass(frozen=True, slots=True)
class TTSRequestedSelectionSnapshot:
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, Any]
    configuration_revision: int


ProfileAvailabilityState = Literal["available", "unavailable", "unverified"]


@dataclass(frozen=True, slots=True)
class TTSPlaygroundSelectionPreset:
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, Any]
    availability: ProfileAvailabilityState


@dataclass(frozen=True, slots=True)
class STTSGeneratedAudio:
    # Existing fields remain unchanged.
    requested_selection: TTSRequestedSelectionSnapshot | None = None
```

The requested-selection snapshot never contains submitted text, origins,
credentials, raw responses, or connection configuration. Legacy artifacts
always retain `requested_selection=None`.

```python
@dataclass(frozen=True, slots=True)
class TTSProfilePageSnapshot:
    repository_generation: int
    profiles: tuple[TTSGenerationProfile, ...]
    total: int


@dataclass(frozen=True, slots=True)
class LoadedTTSProfile:
    repository_generation: int
    profile: TTSGenerationProfile


@dataclass(frozen=True, slots=True)
class TTSProfileAvailability:
    profile_id: UUID
    state: ProfileAvailabilityState
    recovery_action: Literal["none", "refresh", "edit"]


@dataclass(frozen=True, slots=True)
class TTSProfileAvailabilitySnapshot:
    repository_generation: int
    configuration_revision: int
    catalog_revision: int | None
    profiles: tuple[TTSProfileAvailability, ...]
```

The service boundary is:

```python
class TTSProfileService:
    async def list_profiles(
        self,
        *,
        search: str | None = None,
        offset: int = 0,
    ) -> TTSProfilePageSnapshot: ...

    async def observe_availability(
        self,
        page: TTSProfilePageSnapshot,
    ) -> TTSProfileAvailabilitySnapshot: ...

    async def create_from_artifact(
        self,
        display_name: str,
        artifact: STTSGeneratedAudio,
    ) -> LoadedTTSProfile: ...

    async def update_profile(
        self,
        loaded: LoadedTTSProfile,
        draft: TTSProfileDraft,
    ) -> LoadedTTSProfile: ...

    async def duplicate_profile(
        self,
        loaded: LoadedTTSProfile,
        display_name: str,
    ) -> LoadedTTSProfile: ...

    async def assignment_count(self, loaded: LoadedTTSProfile) -> int: ...

    async def delete_profile(self, loaded: LoadedTTSProfile) -> None: ...

    def preview_preset(
        self,
        loaded: LoadedTTSProfile,
        availability: TTSProfileAvailability,
    ) -> TTSPlaygroundSelectionPreset: ...
```

The repository extends only mutation admission:

```python
await repository.create_profile(
    draft,
    profile_id=None,
    expected_generation=None,  # supplied for duplicate-from-loaded
)
await repository.update_profile(
    profile_id,
    expected_revision,
    draft,
    expected_generation=loaded_generation,
)
await repository.delete_profile(
    profile_id,
    expected_generation=loaded_generation,
)
```

`_admit_operation()` validates `expected_generation` under the existing
repository state lock before submitting the worker job. `None` is accepted
only for a create that is not derived from caller-held repository state.
There is no schema or migration change.

## Task 1: Freeze task scope, branch, plan, and baseline

**Files:**

- Create: `backlog/tasks/task-951 - Add-audio.cpp-TTS-profile-service-and-STTS-library.md`
- Create: `Docs/superpowers/plans/2026-07-27-audio-cpp-tts-profile-service-stts-library.md`
- Modify: `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md`
- Modify: `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md`
- Modify: `backlog/decisions/028-character-tts-generation-profile-ownership.md`

- [x] **Step 1: Rebase the approved design amendments onto current dev**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git merge-base --is-ancestor origin/dev HEAD
```

Expected: the two approved documentation commits replay cleanly above current
`origin/dev`; ancestor check exits `0`.

- [x] **Step 2: Create and start the atomic Backlog task**

Run:

```bash
backlog task 951 --plain
```

Expected: TASK-951 is In Progress, depends on completed TASK-763, and contains
only Slice 2B acceptance criteria.

- [x] **Step 3: Verify the worktree and create the task branch**

Run:

```bash
git -C ../.. check-ignore -q .worktrees
git status --short --branch
```

Expected: the project-local worktree directory is ignored and branch
`codex/task-951-tts-profile-library` contains no production changes.

- [x] **Step 4: Establish the supported-interpreter baseline**

Use the exact command and result recorded under **Rebased baseline**. Do not
repair the three unrelated fixture failures in task 951.

- [x] **Step 5: Commit the reviewed planning boundary**

```bash
git add \
  Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md \
  Docs/superpowers/plans/2026-07-27-audio-cpp-tts-profile-service-stts-library.md \
  backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md \
  backlog/decisions/028-character-tts-generation-profile-ownership.md \
  "backlog/tasks/task-951 - Add-audio.cpp-TTS-profile-service-and-STTS-library.md"
git commit -m "docs(tts): plan profile service and STTS library"
```

Expected: one planning commit above the two already approved design-amendment
commits.

## Task 2: Preserve structured voice authority at the adapter boundary

**Files:**

- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `Tests/TTS/test_adapter_types.py`
- Modify: `Tests/TTS/test_audio_cpp_adapter.py`

- [x] **Step 1: Write failing immutable-type and protocol tests**

Cover exact validation, frozen voice tuples, nonnegative catalog revision,
runtime protocol detection, and rejection of mutable or invalid state.

```python
result = TTSVoiceDiscoveryResult(
    provider_id="audio_cpp",
    model_id="supertonic",
    catalog_revision=4,
    voices=("voice-a",),
    state="complete",
)
assert result.voices == ("voice-a",)
assert isinstance(structured_adapter, TTSStructuredVoiceAdapter)
```

- [x] **Step 2: Run the type tests and verify red**

```bash
.venv/bin/python -m pytest Tests/TTS/test_adapter_types.py -q
```

Expected: failure because the structured result/protocol do not exist.

- [x] **Step 3: Add the minimal frozen result and optional runtime protocol**

Keep `TTSAdapter.get_voices()` unchanged. The optional protocol contains only
`observe_voices()` and is not added to legacy adapter requirements.

- [x] **Step 4: Write failing adapter behavior tests**

Prove:

- successful empty is `complete` and cached as such;
- missing exact model is `model_missing` without a voice HTTP request;
- timeout, transient transport failure, contract failure, reconfiguration, and
  shutdown are `unverified`;
- an ambiguous result is cached only as `unverified`, never as authoritative
  empty, and an explicit refresh can recover it;
- same-model/revision calls still coalesce;
- external `CancelledError` propagates and is neither cached nor published;
- `get_voices()` delegates and projects only `complete` to its tuple.

- [x] **Step 5: Run the adapter tests and verify red**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_adapter_types.py -q
```

Expected: new structured-status tests fail against tuple-only discovery.

- [x] **Step 6: Convert the existing cache and fetch path minimally**

Store `TTSVoiceDiscoveryResult` in `_VoiceCacheEntry` and
`_voice_shared_results`. `_fetch_voices()` converts only bounded internal
failures to `unverified`; it continues to re-raise `CancelledError`.
`get_voices()` becomes a compatibility projection over
`observe_voices()`. Both `complete` and `unverified` results enter the bounded
status-aware LRU cache, so repeated ordinary calls preserve the distinction
without repeatedly probing a failing optional endpoint. `refresh=True`
bypasses the cached status and performs a new observation. `model_missing` is
derived directly from the current authoritative catalog rather than stored as
a voice fetch result. Caller cancellation is neither shared nor cached.
Preserve the existing cache entry/byte limits, per-key locks, request
coalescing, privacy filters, and refresh generations.

- [x] **Step 7: Run green adapter tests and commit**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_adapter_registry.py -q
git add \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_audio_cpp_adapter.py
git commit -m "feat(tts): preserve structured audio cpp voice status"
```

## Task 3: Add coherent capability observation and exact native provenance

**Files:**

- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/request_admission.py`
- Modify: `tldw_chatbook/TTS/playground_types.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Create: `Tests/TTS/test_tts_profile_capabilities.py`
- Modify: `Tests/TTS/test_tts_request_admission.py`
- Modify: `Tests/TTS/test_stts_playground_types.py`
- Modify: `Tests/TTS/test_stts_audio_cpp_generation.py`

- [x] **Step 1: Write failing capability snapshot tests**

Use fake adapters and a controlled clock to prove:

- one expected-revision lease is acquired inside the coordinator read side;
- the read gate is released before catalog/voice network waits;
- at most four distinct exact-voice models run concurrently;
- server-default profiles cause no voice observation;
- all work shares one ten-second aggregate deadline;
- one catalog revision advance triggers one full retry;
- a second advance or unfinished observation returns `unverified`;
- caller cancellation propagates after cancellation-safe lease release;
- no concrete adapter or lease escapes the service API.

- [x] **Step 2: Run the new capability tests and verify red**

```bash
.venv/bin/python -m pytest Tests/TTS/test_tts_profile_capabilities.py -q
```

Expected: import/attribute failures for the missing snapshot API.

- [x] **Step 3: Implement one bounded service snapshot**

Add:

```python
await service.get_native_capability_snapshot(
    "audio_cpp",
    exact_voice_model_ids,
)
```

The coordinator captures configuration revision and acquires one matching
registry lease under `_gate.read()`, then exits that gate. The service observes
one fresh catalog, deduplicates model IDs, runs voice observations under a
four-slot semaphore, compares catalog revision after observations, and retries
the complete snapshot at most once within the original deadline. Lease release
uses the existing retained cleanup helper and preserves caller cancellation.
The ten-second deadline is a module-owned Slice 2B constant, not a new public
configuration surface; tests control the private clock/deadline seam.

- [x] **Step 4: Write failing exact-admission provenance tests**

Prove:

- native exact admission freezes the request and configuration revision under
  the same read side used to acquire the lease;
- the requested-selection snapshot has no text;
- controls or response metadata cannot alter it;
- `require_current_configuration_revision()` waits behind a queued writer and
  makes one decision before returning;
- a change visible at that decision rejects stale provenance;
- the coordinator gate is released before repository work;
- legacy generation cannot create requested-selection provenance.

- [x] **Step 5: Run exact-admission tests and verify red**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_stts_audio_cpp_generation.py -q
```

Expected: failures for missing exact synthesis and artifact provenance.

- [x] **Step 6: Add exact synthesis and revision-decision service methods**

Add public service methods that delegate to its existing coordinator:

```python
response, selection = await service.synthesize_exact(
    request,
    progress_sink,
)
await service.require_current_configuration_revision(
    provider_id,
    expected_revision,
)
```

`synthesize_exact()` reserves service capacity, enters `_gate.read()`, captures
the current revision, acquires the matching operation, creates the immutable
text-free requested-selection snapshot, exits the gate, and synthesizes.
The response continues to own the admitted lease through close.

- [x] **Step 7: Attach provenance only to successful native artifacts**

Change `_generate_audio_cpp()` to call `synthesize_exact()` and set the
artifact's optional `requested_selection`. Do not modify `_generate_legacy()`
beyond tests that assert the field remains `None`.

- [x] **Step 8: Run green capability/provenance tests and commit**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_tts_registry_service.py -q
git add \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/__init__.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_stts_audio_cpp_generation.py
git commit -m "feat(tts): admit exact native profile provenance"
```

## Task 4: Reject stale profile-derived repository mutations

**Files:**

- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `Tests/TTS/test_profile_repository.py`
- Modify: `Tests/TTS/test_profile_repository_lifecycle.py`

- [x] **Step 1: Write failing expected-generation admission tests**

Cover exact integer validation and the three mutation paths:

```python
await repository.update_profile(
    loaded.profile_id,
    loaded.revision,
    draft,
    expected_generation=loaded_result.generation,
)
await repository.delete_profile(
    loaded.profile_id,
    expected_generation=loaded_result.generation,
)
await repository.create_profile(
    duplicate_draft,
    expected_generation=loaded_result.generation,
)
```

Pause admission around restore and use a replacement store containing the same
UUID and profile revision. Every pre-restore update, delete, and duplicate must
raise safe code `stale` before enqueue and leave the replacement unchanged.

- [x] **Step 2: Run repository tests and verify red**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py -q
```

Expected: new calls fail because public methods lack expected-generation
admission.

- [x] **Step 3: Add the minimal state-lock comparison**

Validate an exact nonnegative generation at the public boundary. Thread it to
`_submit_operation()` and `_admit_operation()`. Under `_state_lock`, compare it
with `_generation` after checking lifecycle state and before
`executor.submit()`. Raise the existing safe `stale` error on mismatch.

Do not add a table, column, migration, second lock, or worker round trip.
Update existing direct repository tests to pass the generation returned by the
read/create that produced their caller state.

- [x] **Step 4: Run green repository gates and commit**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_schema.py -q
git add \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py
git commit -m "fix(tts): fence loaded profile mutations by store generation"
```

## Task 5: Add the native-only profile service

**Files:**

- Modify: `tldw_chatbook/TTS/profile_errors.py`
- Create: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Create: `Tests/TTS/test_profile_service.py`

- [x] **Step 1: Write failing validation, availability-allowlist, and
  save-from-artifact tests**

Prove:

- only exact `audio_cpp`, `wav`, speed `1.0`, and empty options are accepted;
- a descriptor's `native=True` does not bypass the explicit allowlist;
- a mixed page excludes unsupported-provider models from the native capability
  snapshot request;
- an all-unsupported page classifies every row without a capability lookup;
- legacy or missing requested-selection provenance is rejected;
- save reads only the immutable artifact snapshot;
- save does not call catalog or voice discovery;
- stale configuration revision is rejected at the coordinator decision;
- a later reconfiguration does not roll back an admitted repository create;
- errors expose bounded codes without text, paths, endpoints, credentials, or
  raw upstream data.

- [x] **Step 2: Run service tests and verify red**

```bash
.venv/bin/python -m pytest Tests/TTS/test_profile_service.py -q
```

Expected: import failure for the missing service.

- [x] **Step 3: Implement immutable page, loaded, and availability values**

Keep these in `profile_service.py`; do not add persistence fields. Freeze every
tuple/mapping and validate state/code values at construction.

- [x] **Step 4: Implement save, list, and availability minimally**

`list_profiles()` delegates to the repository with fixed `limit=50`.
`observe_availability()` first confirms the page's repository generation,
then rejects any row outside the exact Slice 2B executable contract
(`provider_id == "audio_cpp"`, WAV, speed `1.0`, and empty options) as
`unavailable` without capability lookup. A descriptor's `native=True` does not
expand that allowlist. For structurally supported rows, ask `TTSService` for
one capability snapshot containing only distinct exact-voice models.
Classification is:

- `available`: exact model exists and format is compatible; server-default is
  declared or exact voice appears in a `complete` result;
- `unavailable`: the provider/profile contract is unsupported or fresh
  authoritative catalog/model/voice evidence rejects the exact selection;
- `unverified`: catalog/voice/configuration/repository state is ambiguous.

Recheck repository generation and snapshot configuration revision before
returning. Return those revisions with the availability rows so the UI can
reject an older enrichment after a newer page/refresh has been requested. Do
not persist availability or add a second cache.

- [x] **Step 5: Write failing edit, duplicate, count, and delete tests**

Prove:

- rename-only is determined by comparing submitted generation fields with the
  immutable loaded profile, not by a UI flag;
- rename-only may proceed while capabilities are unverified;
- generation edits and duplicate require a fresh authoritative capability
  snapshot plus matching configuration-revision decision;
- duplicate copies the immutable loaded version under a new name/UUID/revision
  1 even if the source later changes;
- update/delete/duplicate supply the loaded repository generation;
- count/publication rejects a changed repository generation;
- transactional assignment protection remains the final delete authority.

- [x] **Step 6: Implement mutations and preview preset**

Release the coordinator read side before calling any repository method.
`preview_preset()` copies only exact persisted generation values and the
availability state; it performs no synthesis.

- [x] **Step 7: Run green service tests and commit**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_tts_profile_capabilities.py -q
git add \
  tldw_chatbook/TTS/profile_errors.py \
  tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_profile_service.py
git commit -m "feat(tts): add native generation profile service"
```

## Task 6: Bind one lazy profile service to existing app owners

**Files:**

- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`

- [x] **Step 1: Write failing ownership tests**

Prove:

- app construction does not instantiate `TTSProfileService`;
- concurrent first use joins one lazy construction;
- construction first obtains the existing app-owned open repository;
- the service receives exactly `app._tts_profile_repository` and
  `app.tts_service`;
- store-open failure returns `None` without affecting ordinary TTS;
- the profile service owns no close task, adapter, registry, executor,
  connection, or shutdown call;
- app shutdown continues to close only the existing repository and TTS service.

Use focused fakes that do not call the known failing shared
`_build_test_app()` fixture until that unrelated baseline is fixed elsewhere.

- [x] **Step 2: Run ownership tests and verify red**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_tts_app_ownership.py \
  -k "profile_service or profile_repository_ensure or owned_tts_cleanup" -q
```

Expected: new profile-service ownership tests fail.

- [x] **Step 3: Add one lazy app-owned service**

Add `_tts_profile_service: TTSProfileService | None = None` and a small
`_ensure_tts_profile_service()` method. It awaits
`_ensure_tts_profile_repository()`, constructs once on the event loop, and
returns `None` on unavailable storage. It adds no close method because its
dependencies are already app-owned and closed by existing shutdown.

- [x] **Step 4: Run green ownership tests and commit**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_tts_app_ownership.py \
  -k "profile_service or profile_repository_ensure or owned_tts_cleanup" -q
git add tldw_chatbook/app.py Tests/TTS/test_tts_app_ownership.py
git commit -m "feat(tts): own one lazy generation profile service"
```

Report the three known full-file baseline failures separately if still present.

## Task 7: Build the bounded STTS profile library

**Files:**

- Create: `tldw_chatbook/UI/stts_profile_library.py`
- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Create: `Tests/UI/test_stts_profile_library.py`

- [x] **Step 1: Write failing initial-library and failure-isolation tests**

Mount `STTSWindow` with a fake profile service. Prove:

- a **Voice profiles** sidebar item mounts the focused library;
- unavailable profile storage shows stable recovery copy;
- Playground, settings, audiobook, dictation, and legacy generation remain
  mountable;
- one page contains at most 50 rows;
- repository rows appear before the controlled availability future completes;
- selected profile actions remain disabled until a row is selected.

- [x] **Step 2: Run UI tests and verify red**

```bash
.venv/bin/python -m pytest Tests/UI/test_stts_profile_library.py -q
```

Expected: the library module/view does not exist.

- [x] **Step 3: Implement the focused library shell**

Use a selectable `DataTable`, one search `Input`, Previous/Next controls, one
status/detail region, and explicit Preview/Edit/Duplicate/Refresh/Delete
buttons. Keep the editor in a focused modal class inside the new module.
Do not add profile logic to the 5,000-line `STTS_Window.py`.

- [x] **Step 4: Write failing coalescing and stale-publication tests**

Use controlled futures to prove:

- debounce completes before repository submission;
- at most one active page pipeline and one latest pending query exist;
- intermediate keystrokes do not enqueue shielded repository operations;
- late page results are rejected across UI request, repository generation,
  page, search, and unmount;
- late availability is additionally rejected across configuration/catalog
  revision changes;
- cancellation and unmount settle retained work without unhandled tasks.

- [x] **Step 5: Implement one coalesced page pipeline**

Store only:

- one active page task/token;
- one latest pending `(search, offset)` request;
- the currently rendered page generation;
- current row availability for that rendered page.

Publish repository rows immediately, then start availability enrichment.
Do not add a long-lived catalog/voice cache to the widget or service.

- [x] **Step 6: Write failing editor/action tests**

Cover create-from-artifact handoff, edit conflict, rename-only, generation
edit, duplicate new name/UUID, assignment count, protected delete, refresh,
minimal repair, and value-independent errors. Confirm all actions pass the
exact loaded profile token back to the service.

- [x] **Step 7: Implement editor and actions**

The editor preserves exact opaque model/voice values. It does not silently
replace unavailable values, expose character assignment, or synthesize.
Delete confirmation shows the advisory count; repository conflict remains the
final authority.

- [x] **Step 8: Run green library tests and commit**

```bash
.venv/bin/python -m pytest \
  Tests/UI/test_stts_profile_library.py \
  Tests/UI/test_stts_capability_state.py \
  Tests/UI/test_stts_settings_widget.py -q
git add \
  tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/UI/STTS_Window.py \
  Tests/UI/test_stts_profile_library.py
git commit -m "feat(stts): add bounded generation profile library"
```

## Task 8: Reuse the Playground for exact preview and save-result flow

**Files:**

- Modify: `tldw_chatbook/TTS/playground_types.py`
- Modify: `tldw_chatbook/UI/stts_playground_catalog.py`
- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Modify: `tldw_chatbook/UI/stts_profile_library.py`
- Modify: `Tests/TTS/test_stts_playground_types.py`
- Modify: `Tests/UI/test_stts_playground_audio_cpp.py`
- Modify: `Tests/UI/test_stts_profile_library.py`

- [x] **Step 1: Write failing exact-preset projection tests**

Prove:

- preset values survive Playground remount;
- an absent exact model is injected visibly rather than replaced by the first
  catalog model;
- an absent exact voice is injected visibly rather than replaced by Server
  default;
- profile-originated unavailable state disables generation and points to Edit;
- unverified state permits only an explicit warned exact attempt;
- no preview generates automatically;
- the preset association ends on user edits to provider/model/voice/format/
  speed/options.

- [x] **Step 2: Run preset tests and verify red**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_stts_playground_types.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  -k "preset or profile_preview or exact" -q
```

Expected: missing preset and current first/default substitution failures.

- [x] **Step 3: Add the one-shot STTSWindow handoff**

`STTSWindow` owns one `_pending_playground_preset`. A profile-library preview
message sets it and switches `current_view` to `playground`. The next
`TTSPlaygroundWidget` constructor receives and consumes it. No global,
class-static, config, or handler state stores the preset.

Add a pure exact projection helper in `stts_playground_catalog.py`; keep the
ordinary selector projection unchanged for non-profile use.

- [x] **Step 4: Write failing save-result tests**

Prove:

- a successful native artifact exposes **Save result as profile**;
- changing selectors after generation cannot alter the saved draft;
- legacy artifacts and failed/native-in-progress states never expose the
  action;
- save opens a name dialog, calls the lazy app-owned service once, and reports
  conflict/stale/store-unavailable errors safely;
- save does not issue another catalog/voice request;
- navigation/remount continues to reuse handler-owned artifact and player
  cleanup.

- [x] **Step 5: Implement the minimal save action**

The button reads only `current_audio_artifact.requested_selection`.
It obtains `app._ensure_tts_profile_service()` lazily and delegates to
`create_from_artifact()`. The existing handler retains artifact lifetime; the
widget does not create or copy audio files.

- [x] **Step 6: Run green preview/save and legacy regression tests**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/UI/test_stts_profile_library.py \
  Tests/TTS/test_legacy_bridge.py \
  Tests/TTS/test_stts_export_security.py -q
```

Expected: all task-added and existing tests pass; legacy artifacts still have
no profile provenance.

- [x] **Step 7: Commit exact preview and save flow**

```bash
git add \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/UI/stts_playground_catalog.py \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/UI/stts_profile_library.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/UI/test_stts_profile_library.py
git commit -m "feat(stts): save and preview exact audio cpp profiles"
```

## Task 9: Verify, document, review, and close task 951

**Files:**

- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Modify: `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md` only if implementation required a documented non-semantic clarification
- Modify: `backlog/decisions/023-tts-adapter-registry-and-audio-cpp-runtime-boundary.md` only if implementation would otherwise contradict the accepted decision
- Modify: `backlog/decisions/028-character-tts-generation-profile-ownership.md` only if implementation would otherwise contradict the accepted decision
- Modify: `backlog/tasks/task-951 - Add-audio.cpp-TTS-profile-service-and-STTS-library.md`

- [x] **Step 1: Run the focused task-951 gate**

```bash
.venv/bin/python -m pytest \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_adapter_registry.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/TTS/test_legacy_bridge.py \
  Tests/TTS/test_stts_export_security.py \
  Tests/UI/test_stts_profile_library.py \
  Tests/UI/test_stts_playground_audio_cpp.py \
  Tests/UI/test_stts_capability_state.py \
  Tests/UI/test_stts_settings_widget.py -q
```

Expected: every task-added test and every previously green baseline test
passes. Report the three named pre-existing shared-fixture failures separately
if they remain; no additional failure is accepted.

- [x] **Step 2: Run broader TTS/STTS regression**

```bash
.venv/bin/python -m pytest Tests/TTS Tests/UI/test_stts_*.py -q
```

Expected: no new failure relative to the recorded baseline and optional
dependency skips remain explicit.

- [x] **Step 3: Run static and boundary checks**

```bash
.venv/bin/python -m ruff check \
  --output-format concise \
  tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/TTS/profile_service.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_profile_service.py \
  Tests/UI/test_stts_profile_library.py
.venv/bin/python -m ruff format --check \
  tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/TTS/profile_service.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_profile_service.py \
  Tests/UI/test_stts_profile_library.py
.venv/bin/python -m ruff check \
  --select E9,F63,F7,F82 \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/profile_errors.py \
  tldw_chatbook/TTS/__init__.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/UI/stts_playground_catalog.py \
  Tests/TTS/test_adapter_types.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/UI/test_stts_playground_audio_cpp.py
.venv/bin/python -m compileall -q \
  tldw_chatbook/TTS \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py
.venv/bin/python -m mypy \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/profile_service.py
git diff --check
```

Expected: task-scoped checks pass. Any inherited baseline must be identified by
exact file/line and must not be represented as task-created success.

- [x] **Step 4: Run scope and privacy audit**

```bash
git diff --name-only origin/dev...HEAD
rg -n -i \
  "subprocess|popen|server\\.json|managed audio|character_tts_assignments|card export|card import" \
  tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/UI/STTS_Window.py
rg -n \
  "source_text|base_url|api_key|credential|raw upstream" \
  tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/playground_types.py
```

Expected: no managed-process, assignment, roleplay, portability, credential,
origin, or submitted-text persistence enters Slice 2B. Existing
`STTSGeneratedAudio.source_text` remains an artifact-lifetime field and never
enters `TTSRequestedSelectionSnapshot` or profile persistence.

- [x] **Step 5: Update guides and perform UAT**

Document:

- external audio.cpp prerequisite;
- generate and play a complete WAV;
- save the successful result;
- search/reload/edit/duplicate/preview/delete;
- unavailable versus unverified behavior;
- profile-store failure isolation;
- no character assignment or managed server in this slice.

Run isolated-config UAT against the user-started external audio.cpp server.
Slice 2B UAT stops after profile persistence and management; do not claim
character roleplay speech.

- [x] **Step 6: Request independent code and scope review**

Use `superpowers:requesting-code-review` over `origin/dev...HEAD`. Address every
verified Critical, Important, and Minor finding with fresh tests. Re-run the
focused/static gates after the last amendment.

- [x] **Step 7: Rebase on latest dev and repeat final verification**

```bash
git fetch origin dev
git rebase origin/dev
git diff --check origin/dev...HEAD
```

Repeat Steps 1–4 after the rebase. Do not rely on pre-rebase evidence.

- [x] **Step 8: Complete task hygiene**

Only after every acceptance criterion and Definition of Done item is proven:
first set `--notes` to a concise summary of the implemented behavior plus the
exact test, static-analysis, UAT, review, and rebase evidence observed in Steps
1–7. Then check the proven items and move the task:

```bash
backlog task edit 951 \
  --check-ac 1 --check-ac 2 --check-ac 3 --check-ac 4 \
  --check-ac 5 --check-ac 6 --check-ac 7 --check-ac 8 \
  --check-ac 9 --check-ac 10 --check-ac 11 --check-ac 12 \
  --check-ac 13 --check-ac 14 \
  --check-dod 1 --check-dod 2 --check-dod 3 \
  --check-dod 4 --check-dod 5 --check-dod 6 \
  --status Done
```

- [x] **Step 9: Commit final documentation**

```bash
git add \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md \
  Docs/superpowers/plans/2026-07-27-audio-cpp-tts-profile-service-stts-library.md \
  "backlog/tasks/task-951 - Add-audio.cpp-TTS-profile-service-and-STTS-library.md"
git commit -m "docs(tts): complete profile service and STTS library"
```

## Execution rule

Use TDD for every behavior change. Run each named failing test before its
implementation and run the matching green gate afterward. Keep commits scoped
to the tasks above. If implementation requires character identity, assignment,
roleplay routing, portability, managed process behavior, legacy-profile
execution, another persistence store, or a second runtime owner, stop and amend
the accepted scope before coding it.
