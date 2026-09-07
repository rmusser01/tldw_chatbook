# TASK-13204 Guided Clone Admission and Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan sequentially. Apply superpowers:test-driven-development to every behavior change and stop at the review checkpoints.

**Goal:** Safely execute stored audio.cpp clone-reference profiles against only the exact compatible Guided Managed child, using immutable admission snapshots and private operation-scoped WAV materializations that survive through response close and are then definitively removed.

**Architecture:** Keep profile/reference authority in `TTSProfileService` and the v3 repository, process/lease authority in `TTSService` and `AudioCppSupervisor`, and HTTP authority in `AudioCppAdapter`. Exact-profile resolution reads and validates the full reference under repository generation/profile revision fences before provider work. The admitted operation asks the native adapter for a generation-fenced Guided Managed capability decision, creates one opaque POSIX-private materialization, sends it only through typed native request fields, attaches cleanup to the response, and relies on the existing response-held adapter lease to drain restart, replacement, and shutdown. A focused materializer owns only directory/file/lock lifecycle; it does not manage processes, profiles, catalogs, or UI state.

**Tech Stack:** Python 3.11+, `asyncio`, `pathlib`, `os`, POSIX `fcntl.flock`, standard-library filesystem APIs, existing Textual/TTS registry and audio.cpp adapter, pytest, Ruff, mypy.

**Normative design:** `Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md` (`GM-ARCH-005`, `GM-LIFE-005`, `GM-VOICE-003`–`005`, `GM-ERR-001`–`003`, `GM-TEST-003`–`004`, `GM-AC-014`, `GM-AC-023`, `GM-AC-027`).

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/051-private-tts-clone-reference-assets.md`

**Reason:** ADR-051 already fixes typed clone admission, exact reference snapshots, Guided Managed-only local-path authority, private operation materialization, ownership locks, and definitive cleanup. ADR-023 and ADR-028 remain the existing lifecycle and profile-ownership boundaries. This task directly implements those accepted decisions without changing architecture.

**Deliberate exclusions:** No clone setup/editor UI, transient Speech Lab audition, profile save/assignment workflow, ordinary profile export, portable voice bundle, Model Library changes, new model recipes, External clone upload, user-provided `server.json` clone paths, Windows ACL/reparse-point parity, or live UAT. Those belong to TASK-13205, TASK-13206, TASK-13208, and later guided-model tasks.

---

### Task 1: Make recipe voice/reference combinations explicit

**Files:**
- Modify: `tldw_chatbook/TTS/audio_cpp_recipes.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `Tests/TTS/test_audio_cpp_recipes.py`

- [ ] **Step 1: Write failing recipe-policy tests**

Add a bounded recipe policy that distinguishes native voice only, reference only, either native voice or reference, and an explicitly declared combined form. The combined form means both inputs are required and both upstream fields are emitted; neither field is a fallback and neither silently overrides the other. Keep it synthetic/reserved until pinned upstream evidence supports a production recipe. Test constructor validation, sealed registry round trips, every accepted initial recipe, clone-only families, and rejection of contradictory capability/reference declarations.

```python
@pytest.mark.parametrize(
    ("policy", "voice", "has_reference", "accepted"),
    [
        (AudioCppVoiceReferencePolicy.NATIVE_ONLY, "voice-a", False, True),
        (AudioCppVoiceReferencePolicy.REFERENCE_ONLY, None, True, True),
        (AudioCppVoiceReferencePolicy.EITHER, "voice-a", True, False),
        (
            AudioCppVoiceReferencePolicy.BOTH_REQUIRED_COMBINED,
            "voice-a",
            True,
            True,
        ),
    ],
)
def test_recipe_policy_admits_only_declared_combinations(...): ...
```

Use the initial recipe truth already accepted by TASK-13200: Supertonic recipes are native/default-voice-only; PocketTTS GGUF recipes require a reference; PocketTTS safetensors accepts a native voice or a reference, but not both. Keep a synthetic recipe fixture for the explicit combined case so both-field behavior is tested without inventing a production recipe claim.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_recipes.py -q
```

Expected: import/constructor failures for the missing combination policy.

- [ ] **Step 3: Implement the minimal immutable policy**

Add one enum and one pure decision method on the accepted recipe projection:

```python
class AudioCppVoiceReferencePolicy(StrEnum):
    NATIVE_ONLY = "native_only"
    REFERENCE_ONLY = "reference_only"
    EITHER = "either"
    BOTH_REQUIRED_COMBINED = "both_required_combined"

def admits_voice_reference(
    self,
    *,
    has_voice: bool,
    has_reference: bool,
) -> bool: ...
```

Validate exact booleans and require recipe capability/reference fields to agree with the policy. `BOTH_REQUIRED_COMBINED` has one exact projection: require both inputs and emit both `voice` and `voice_ref`/`reference_text`, with no fallback or override semantics. Every production recipe in this task must reject both; only a synthetic test recipe exercises the reserved combined contract. Keep `AudioCppReferenceRequirement` only where existing Settings/UX projection needs it; derive or cross-check it rather than creating two independent truths.

- [ ] **Step 4: Run tests and mutation-check fail-closed admission**

Run the Step 2 command. Temporarily make `EITHER` accept both fields and confirm the matrix fails, then restore the correct branch.

- [ ] **Step 5: Commit the recipe contract**

```bash
git add tldw_chatbook/TTS/audio_cpp_recipes.py tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_audio_cpp_recipes.py
git commit -m "feat(tts): define audio cpp clone admission policy"
```

### Task 2: Freeze exact profile/reference execution snapshots

**Files:**
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/character_request_resolver.py`
- Modify: `tldw_chatbook/TTS/default_profile_request_resolver.py`
- Modify: `tldw_chatbook/TTS/effective_settings.py`
- Modify: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`
- Modify: `Tests/TTS/test_profile_service.py`
- Modify: `Tests/TTS/test_character_request_resolver.py`
- Modify: `Tests/TTS/test_default_profile_request_resolver.py`
- Modify: `Tests/TTS/test_effective_settings.py`
- Modify: `Tests/TTS/test_console_speech_snapshot_admission.py`

- [ ] **Step 1: Write failing exact-snapshot tests**

Cover reference-free and reference-bearing assigned/default profiles; profile/reference edit and delete between the summary read and exact reference read; repository generation replacement; hostile/malformed repository results; cancellation; and safe context-free error mapping. Assert the resolved value freezes:

- profile UUID and revision;
- repository generation;
- provider/model/native voice selection;
- exact reference UUID, canonical digest, bounded transcript, metadata, and immutable WAV bytes.

Also add the missing regression that `TTSProfileService` preserves a profile's metadata-only reference summary during collaborator canonicalization.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_character_request_resolver.py \
  Tests/TTS/test_default_profile_request_resolver.py \
  Tests/TTS/test_effective_settings.py \
  Tests/TTS/test_console_speech_snapshot_admission.py -q
```

- [ ] **Step 3: Add one exact profile-execution read**

Extend the private repository protocol and `TTSProfileService` with a bounded exact reference read using the existing repository API:

```python
async def get_reference(
    self,
    profile_id: UUID,
    *,
    expected_revision: int,
    expected_generation: int,
) -> TTSCloneReference: ...
```

Canonicalize the returned private type without exposing its fields in `repr` or errors. Fix `_canonicalize_exact_profile()` to carry and validate `reference`. In each resolver, first load the exact profile/assignment summary, then—only when a summary exists—call `get_reference()` with the loaded generation and revision, verify summary UUID/metadata equality, and return one immutable resolution containing the full private reference. The repository fences make any intervening edit/delete/reopen fail rather than silently substituting data.

Extend `TTSCharacterProfileSelection` and `TTSDefaultProfileSelection` with exact `profile_id` and optional `TTSCloneReference`. The resolved/base public `TTSRequest` must never contain the stored reference object, WAV bytes, transcript, digest, or path; only the later internal post-admission clone request may carry redacted typed materialized fields. Never place reference data in generic options, logs, toasts, or selection provenance. Update the event handler to pass the frozen private value into effective admission.

- [ ] **Step 4: Run tests and prove the revision/generation guards discriminate**

Run the Step 2 command. Temporarily omit `expected_revision` or `expected_generation` from a resolver reference read and confirm the corresponding race regression fails, then restore it.

- [ ] **Step 5: Commit exact profile admission snapshots**

```bash
git add tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/character_request_resolver.py \
  tldw_chatbook/TTS/default_profile_request_resolver.py \
  tldw_chatbook/TTS/effective_settings.py \
  tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py \
  Tests/TTS/test_profile_service.py Tests/TTS/test_character_request_resolver.py \
  Tests/TTS/test_default_profile_request_resolver.py \
  Tests/TTS/test_effective_settings.py \
  Tests/TTS/test_console_speech_snapshot_admission.py
git commit -m "feat(tts): freeze exact clone profile snapshots"
```

### Review checkpoint A

- [ ] Re-read Tasks 1–2 against AC #1–#2 and `GM-VOICE-003`–`004`.
- [ ] Confirm no reference value enters `TTSRequest.options`, logs, public errors, or public provenance.
- [ ] Run `git diff --check` and the Task 1–2 test commands.

### Task 3: Implement private operation materialization and orphan cleanup

**Files:**
- Create: `tldw_chatbook/TTS/profile_reference_materialization.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Create: `Tests/TTS/test_profile_reference_materialization.py`

- [ ] **Step 1: Write the POSIX ownership and attack matrix**

Use only isolated `tmp_path` roots. Cover:

- lazy root creation with owner-private modes;
- pre-existing current-user roots/recognized children with permissive modes are
  narrowed through an already-open descriptor before use;
- foreign-owned, substituted, or identity-changing roots/children fail closed
  and are preserved;
- opaque recognized directory names and opaque WAV names;
- exact canonical byte publication and `0600` file mode;
- an ownership lock held for the materialization lifetime;
- idempotent close and exact-directory removal;
- cancellation/failure at each creation, write, fsync, lock, response-close, and removal step;
- startup cleanup of a recognized unlocked orphan;
- preservation of a recognized live-locked directory;
- preservation of unknown names, regular files, symlinks, nested symlinks, FIFOs/devices, hard-link anomalies, and merely old directories;
- concurrent first cleanup/materialization;
- sanitized errors and exception graphs containing no root/path/reference/transcript/digest/bytes.

The POSIX implementation should skip with an explicit unsupported-platform result on Windows; TASK-13208 owns Windows ACL/reparse parity.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_materialization.py -q
```

Expected: import failure for the new materializer.

- [ ] **Step 3: Implement one narrow owner**

Create a lazy async facade over descriptor-bound synchronous filesystem work:

```python
@dataclass(frozen=True, slots=True, repr=False)
class MaterializedTTSCloneReference:
    voice_ref: Path
    reference_text: str
    _owner: TTSCloneMaterialization

    def __repr__(self) -> str:
        return "MaterializedTTSCloneReference(<private>)"

class TTSCloneReferenceMaterializer:
    async def materialize(
        self,
        reference: TTSCloneReference,
    ) -> MaterializedTTSCloneReference: ...

    async def close(self) -> None: ...
```

Use `asyncio.to_thread` only behind a retained task owner. Cancellation must join the in-flight worker; if it published a materialization, close that exact owner before propagating cancellation. On POSIX, create the configured runtime root and one recognized versioned child with `0700`, open them without following links, and descriptor-verify directory type, stable device/inode identity, `st_uid == os.geteuid()`, and owner-only traversal. If a pre-existing root/recognized child is owned by the current effective user but has permissive mode bits, narrow it to `0700` through the open descriptor and verify the result; never chmod, traverse, clean, or replace a foreign-owned/substituted object. Create a private lock file and hold an exclusive `flock`, create the WAV with `O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC` and `0600`, write the exact canonical bytes through the descriptor, fsync file and directory, and revalidate identities/ownership before publication. Keep the lock descriptor open until cleanup. Use a random opaque operation token only; no profile/reference UUID or digest in names.

Perform the startup orphan sweep lazily once per materializer instance before first publication. Enumerate only immediate recognized versioned children using `lstat`/descriptor-relative operations, require current-effective-user ownership and stable no-follow descriptor identity, refuse symlinks and non-directories, acquire the exact lock nonblocking, and remove only a fully recognized unlocked directory. A foreign-owned, identity-changing, or otherwise unqualified entry is preserved. Age is never authority. Retain creation, sweep, and cleanup work against cancellation with the repository's existing retained-task helper pattern. `close()` seals new creation and joins every in-flight creation/sweep/cleanup task before reporting terminal ownership. Map ordinary failures to one stable materialization error outside the caught exception context; preserve `BaseException` control flow.

- [ ] **Step 4: Run tests and mutation-check ownership guards**

Run the Step 2 command. Separately remove the nonblocking lock proof, no-follow directory check, and effective-owner/mode qualification; confirm the live-owner, symlink, foreign-owner, and permissive-root regressions fail, then restore them.

- [ ] **Step 5: Commit private materialization**

```bash
git add tldw_chatbook/TTS/profile_reference_materialization.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_profile_reference_materialization.py
git commit -m "feat(tts): materialize private clone references"
```

### Task 4: Add generation-fenced internal native clone capability and payload fields

**Files:**
- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Modify: `Tests/TTS/test_audio_cpp_adapter.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`

- [ ] **Step 1: Write failing adapter admission/payload tests**

Cover:

- native voice only, reference only, and synthetic explicitly combined payloads;
- required-reference omission and undeclared both-field rejection;
- exact model-to-accepted-package/recipe identity;
- Guided Managed app-owned generation success;
- External and Managed user-JSON rejection before any HTTP request, child launch,
  or materialization;
- wrong recipe revision, wrong model, stale process generation, child exit, replacement, and source switch;
- exact internal request types, opaque live-owner identity, and private `repr`;
- forged internal clone fields passed directly to the adapter, arbitrary local
  paths, copied admission objects, and closed/stale materialization handles all
  fail before HTTP;
- no clone fields in `options`, catalog, server JSON, metadata, or public errors;
- HTTP/debug/body logging suppression for `voice_ref` and `reference_text`.

Assert the exact upstream JSON keys are `voice_ref` and `reference_text`; `voice` is included only when the recipe policy admits it.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_audio_cpp_managed_integration.py -q
```

- [ ] **Step 3: Add a private typed native clone request**

Keep public `TTSRequest` unchanged and reject any attempt to smuggle clone fields through `options` or ordinary service ingress. Add an internal exact request type accepted only by the optional audio.cpp clone protocol; do not add loose path/text keys or reopen `options`:

```python
@dataclass(frozen=True, slots=True, repr=False)
class _AdmittedAudioCppCloneRequest:
    request: TTSRequest
    materialization: MaterializedTTSCloneReference
    capability: AudioCppCloneCapabilityAdmission
    provider_revision: int
    applied_provider_generation: int
    recipe_id: str
    recipe_revision: int
    process_generation: int
```

The materializer retains the only live owner record for its opaque handle. The adapter capability carries a per-adapter private identity and is single-use/generation-fenced. Construction alone is never authority: the materializer must confirm the handle is currently live and exact-owned, and the adapter must confirm the capability was issued by this adapter for this exact model/recipe/process generation and has not been copied, consumed, closed, or invalidated. Validate exact types and use permanently redacted representations. Public `TTSService.synthesize()` continues to accept exact `TTSRequest` only and has no filesystem-path field to forge.

Expose a two-phase optional native-adapter contract. First, a synchronous/side-effect-free source-authority preflight rejects reference-bearing requests unless the adapter configuration is Managed Guided and is bound to the app-owned supervisor; it performs no HTTP, launch, catalog refresh, or filesystem work. Only after that preflight succeeds may the operation call `ensure_ready()`. A second post-ready method then returns a private single-use capability admission for the exact request model/accepted recipe/current owned process. Registry provider revision/applied-generation authority remains on the lease and is not inferred by the adapter. `AudioCppAdapter` must:

- require `mode == managed` and non-`None` Guided settings for references;
- resolve the exact applied `guided_packages` model and validate the accepted recipe in the sealed registry;
- freeze recipe ID/revision and current managed process generation;
- apply the recipe's voice/reference policy;
- recheck the exact process generation and recipe admission immediately before constructing/sending the payload;
- reject External and user-JSON references in the preflight without touching the
  network, launching a child, or invoking the materializer.

Serialize only the typed live materialization fields into the POST body. Keep raw body logging suppressed and normalize capability/generation failures to stable context-free `TTSOperationError` values.

- [ ] **Step 4: Run tests and prove source/process fences discriminate**

Run the Step 2 command. Temporarily allow clone fields when `guided_settings is None`, omit the process-generation recheck, and accept a directly constructed/copy-stale materialization handle; confirm the External/user-JSON, replacement, and forgery regressions fail, then restore all guards.

- [ ] **Step 5: Commit native clone admission**

```bash
git add tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_audio_cpp_managed_integration.py
git commit -m "feat(tts): admit typed audio cpp clone requests"
```

### Review checkpoint B

- [ ] Re-read Tasks 3–4 against AC #3–#4, #6–#7 and ADR-051.
- [ ] Confirm the materializer has no process/catalog/profile authority and the adapter has no repository/filesystem-sweep authority.
- [ ] Confirm External/user-JSON rejection occurs before materialization or HTTP.
- [ ] Run `git diff --check` and the Task 3–4 test commands.

### Task 5: Bind materialization to admitted operation and response lifetime

**Files:**
- Modify: `tldw_chatbook/TTS/adapter_registry.py`
- Modify: `tldw_chatbook/TTS/request_admission.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/adapter_bootstrap.py`
- Modify: `Tests/TTS/test_tts_request_admission.py`
- Modify: `Tests/TTS/test_tts_registry_service.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`

- [ ] **Step 1: Write failing end-to-end lifecycle tests**

Cover the full admitted operation:

1. resolver-provided private reference is frozen;
2. the provider lease freezes the exact registry configuration revision and
   applied provider generation as distinct axes;
3. adapter capability accepts the exact recipe/model/voice/reference combination;
4. materialization occurs only after that acceptance;
5. synthesis sees the exact typed fields;
6. successful response retains both materialization and registry lease until `aclose()`;
7. cleanup finishes before the lease is released and a waiting restart/shutdown proceeds.

Also cover public ingress explicitly: `TTSService.synthesize()` and admission
accept exact public `TTSRequest` only. A forged internal clone request, copied
capability, arbitrary path, or pre-populated clone-like generic option is
rejected before adapter use, HTTP, launch, or materialization.

Add races for cancellation before materialization, during write, after publication, during HTTP, during response streaming/close, saved/staged settings publication, explicit apply, process-generation replacement, unexpected child exit, timeout, adapter failure, invalid response, abandoned admitted operation, and app service close. A staged/saved generation must not alter the already-admitted applied generation; an explicit apply waits for the admitted lease and affects only later operations. Assert exact cleanup and zero retained materializations/tasks after `wait_closed()`.

For clone responses, explicitly cover the current close-signal-after-synthesis and app-shutdown paths that otherwise start response close and direct registry-resource release in parallel. Prove restart/shutdown cannot bypass the protected response-close chain or release its lease while the materialization owner is still retained.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_audio_cpp_managed_integration.py -q
```

- [ ] **Step 3: Extend admitted operations, not the registry**

Have effective resolution carry a private clone execution snapshot only for an exact character/default profile. `TTSRequestAdmissionCoordinator` passes that snapshot into `_AdmittedTTSOperation`; Studio/global/explicit requests remain unchanged.

Extend `TTSAdapterLease` to freeze the slot's exact configuration revision and applied generation under the same registry lock that selects/increments the active adapter record. Those immutable lease fields are registry-owned evidence and must not be read later from mutable slot state or fabricated by `AudioCppAdapter`. Existing callers remain source-compatible.

Inside `_AdmittedTTSOperation.synthesize()` under the existing adapter admission scope:

- for a reference-bearing request, run the native adapter's side-effect-free
  source-authority preflight before readiness;
- only after preflight acceptance, call `ensure_ready()`;
- ask the native adapter's post-ready method to admit the exact
  recipe/model/process combination;
- materialize the already-frozen canonical reference;
- compose the private admitted clone request from the exact lease revision and
  applied generation, adapter-issued recipe/process capability, and live
  materialization handle;
- call the optional adapter clone synthesis method; ordinary public
  `TTSRequest` synthesis remains unchanged;
- on success, append materialization close to the adapter response so it runs
  after underlying stream/adapter cleanup and before wrapper resource release;
- on every failure/control-flow path, retain and join exact cleanup before releasing operation resources;
- if cleanup also fails, preserve the primary stable error and retain cleanup ownership for service `wait_closed()` rather than leaking a path or raw exception.

The External and Managed user-JSON regressions must assert zero HTTP calls, zero
supervisor launches, and zero materializer calls—not merely a final error code.

Staged-save and explicit-apply regressions must assert all three generation
axes independently: a saved/staged provider generation never contaminates the
lease's applied generation, the old admitted operation finishes on its captured
adapter/process generation, and only a later post-apply operation observes the
new applied generation.

Make the protected close chain authoritative for clone responses: underlying stream/adapter close, then materialization cleanup, then registry lease/capacity release. The existing `start_resource_release()` escape used by service shutdown and the late close-signal path must not bypass that chain for a clone response; have it join/delegate to the same retained close task (or an equivalent explicit barrier). Non-clone response behavior remains unchanged. A bounded foreground `close()` may return at its existing deadline, but `wait_closed()` retains the chain and no provider transition or terminal supervisor stop may acquire the clone-held lease early.

Construct one lazy `TTSCloneReferenceMaterializer` in app adapter bootstrap using an owner-private runtime child below the existing user-data area. It must perform no filesystem work until the first accepted clone request. Seal new materializations when `TTSService.close()` starts and definitively join the materializer after admitted responses/operations drain. Do not create a second quota or lifecycle gate.

- [ ] **Step 4: Run tests and mutation-check response-lifetime retention**

Run the Step 2 command. Temporarily clean the directory immediately after adapter `synthesize()` returns and confirm a response-close/restart drain regression fails, then restore response-owned cleanup.

- [ ] **Step 5: Commit operation lifecycle integration**

```bash
git add tldw_chatbook/TTS/adapter_registry.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/adapter_bootstrap.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_audio_cpp_managed_integration.py
git commit -m "feat(tts): retain clone materializations through response close"
```

### Task 6: Close privacy, cleanup, and shutdown interleavings

**Files:**
- Modify: `tldw_chatbook/TTS/audio_cpp_supervisor.py` if generation-scoped
  diagnostic suppression is implemented there
- Modify: `Tests/TTS/test_tts_logging_privacy.py`
- Modify: `Tests/TTS/test_profile_reference_materialization.py`
- Modify: `Tests/TTS/test_tts_registry_service.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`

- [ ] **Step 1: Add privacy and terminal-state regression matrices**

Use unique canary path, transcript, digest, UUID, body, and raw collaborator exception values. Walk every raised exception's `str`, `repr`, `args`, notes, cause, and context graph; capture Loguru, stdlib HTTP/asyncio debug logs, bounded child diagnostics, response metadata, and public selection provenance. None may contain a canary.

Have a fake child echo the materialization path and transcript into stdout and stderr both chunked across reads and delayed until after the HTTP response/response close. Prove the exact process generation enters private-output suppression before the clone POST and retains it until generation retirement, so buffered or delayed child output cannot republish request fields after per-response cleanup. Prefer a generation-scoped content-suppressed diagnostic posture over retaining raw transcript/path redaction tokens: once any clone request is admitted for a child generation, store only safe generic diagnostic markers for that generation while preserving ring bounds and lifecycle metadata. A later process generation starts with the normal sanitized ring until it admits a clone request.

For each phase, assert the phase-appropriate bounded public error type/code, safe recovery semantics, severed exception graph, and absence of provider/model/voice/reference fallback—not only canary absence. Profile/reference resolution uses the existing bounded `CharacterTTSResolutionError`/profile error mapping; native validation, source/capability, process-generation, transport, and timeout use stable `TTSOperationError` values; terminal cleanup/shutdown uses the existing sanitized lifecycle aggregation. Do not force all phases through one taxonomy.

Exercise completion, response close, cancellation, timeout, startup/restart drain, replacement, exit during request, shutdown after admission, shutdown during materialization, cleanup retry, and app close. Assert lifecycle actions do not finish while the adapter could still read the materialization, and `wait_closed()` does not finish while any owned lock, directory, cleanup task, adapter lease, client, or process remains.

- [ ] **Step 2: Run focused tests and verify failures reveal missing guards**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_profile_reference_materialization.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_tts_app_ownership.py -q
```

- [ ] **Step 3: Make only the minimal fixes exposed by the matrix**

Normalize ordinary materializer/adapter/repository errors outside caught exception contexts, preserve cancellation and other control-flow signals, retain cleanup tasks with the existing service task ownership, and enforce service close ordering only where the test proves a dependency. If the child diagnostics seam is used, install generation-scoped clone-output suppression before the request and clear it only when that exact process generation is retired. Do not add new public error taxonomies unless an existing stable `TTSOperationError` code cannot express the phase.

- [ ] **Step 4: Run tests and privacy mutation guards**

Run the Step 2 command. Temporarily include the request payload or materialization path in one adapter/materializer diagnostic and confirm the canary test fails, then restore containment.

- [ ] **Step 5: Commit lifecycle/privacy hardening**

```bash
git add Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_profile_reference_materialization.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_tts_app_ownership.py \
  tldw_chatbook/TTS/audio_cpp_supervisor.py \
  tldw_chatbook/TTS
git commit -m "test(tts): harden clone cleanup and privacy races"
```

### Review checkpoint C

- [ ] Re-read the complete branch against all eight TASK-13204 acceptance criteria.
- [ ] Confirm every materialization owner has an exact terminal cleanup path and every cleanup is ordered after adapter readability ends.
- [ ] Confirm no work from TASK-13205/13206/13208 entered the branch.
- [ ] Run `git diff --check` and all Task 1–6 focused commands.

### Task 7: Document, verify, review, and close TASK-13204

**Files:**
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Modify: `backlog/tasks/task-13204 - Admit-and-materialize-guided-audio.cpp-clone-references-safely.md`
- Modify: `backlog/docs/lessons-*.md` only if implementation produced a concrete reusable incident

- [ ] **Step 1: Update implementation-truth documentation**

Document:

- exact profile/reference snapshot and recipe/source/process admission;
- Guided Managed app-owned-only local materialization;
- typed `voice_ref`/`reference_text` payload fields;
- owner-private local plaintext operation lifetime and definitive cleanup;
- External/user-JSON inactivity and recovery guidance;
- startup orphan-cleanup proof and POSIX-only scope;
- redacted diagnostics/public provenance.

Do not claim clone setup UI, transient audition, bundle portability, Windows parity, or live UAT.

- [ ] **Step 2: Run the complete task verification**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_types.py \
  Tests/TTS/test_profile_reference_audio.py \
  Tests/TTS/test_profile_reference_storage.py \
  Tests/TTS/test_profile_reference_repository.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_character_request_resolver.py \
  Tests/TTS/test_default_profile_request_resolver.py \
  Tests/TTS/test_effective_settings.py \
  Tests/TTS/test_audio_cpp_recipes.py \
  Tests/TTS/test_audio_cpp_contract.py \
  Tests/TTS/test_audio_cpp_adapter.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_tts_registry_service.py \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_profile_reference_materialization.py \
  Tests/TTS/test_console_speech_snapshot_admission.py \
  Tests/TTS/test_tts_app_ownership.py -q

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/TTS tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py \
  Tests/TTS

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/profile_reference_materialization.py \
  tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/character_request_resolver.py \
  tldw_chatbook/TTS/default_profile_request_resolver.py \
  tldw_chatbook/TTS/effective_settings.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py \
  tldw_chatbook/TTS/TTS_Generation.py

git diff --check
```

If a listed test path differs in the current tree, use the existing corresponding module and record the substitution. Any environment-only failure must be separately identified; do not count it as passing evidence.

- [ ] **Step 3: Request code review and address every validated finding**

Use `superpowers:requesting-code-review` against the branch diff from its merge base. Reproduce each finding before changing code, apply `superpowers:receiving-code-review`, add a regression, rerun the affected matrix, and repeat review until no Critical/Important/Minor findings remain.

- [ ] **Step 4: Complete Backlog Definition of Done**

Check all acceptance criteria only after verification, add concise Implementation Notes naming the implemented boundaries and exact test evidence, record the ADR check, update a lessons file only for a real reusable incident, and set TASK-13204 to Done through Backlog CLI.

- [ ] **Step 5: Commit closeout**

```bash
git add Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md \
  'backlog/tasks/task-13204 - Admit-and-materialize-guided-audio.cpp-clone-references-safely.md'
git commit -m "docs(tts): close guided clone admission task"
```

The branch is then ready for the repository's normal PR/rebase/review/merge workflow. Do not merge or start TASK-13205 unless the user explicitly requests it.
