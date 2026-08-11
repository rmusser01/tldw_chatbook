# TASK-13205 Clone Setup and Character Voice Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan sequentially. Apply superpowers:test-driven-development to every behavior change, use impeccable and textual-tui for the mounted interface work, and stop at the review checkpoints.

**Goal:** Let a Speech Lab user set up and audition a reference-required Guided audio.cpp model, save the exact successful reference as a reusable Voice Profile, explicitly assign it to a character, and hear that character's later Console/Roleplay Speak request through the one lazy managed child.

**Architecture:** Keep canonical reference bytes session-only until explicit save. Before generation, the mounted Speech Lab pane owns one immutable setup draft and passes an exact path-free canonical reference through the existing retained STTS request owner. After the STTS handler accepts a successful complete-WAV result, the pane drops that exact setup draft and retains only a sanitized operation-ID/current-result projection; the handler is the sole `STTSGeneratedAudio` and canonical-reference owner. Play and Save address that handler-owned current artifact by operation ID, never by a second UI-held artifact reference. `TTSService` extends its existing typed Guided clone admission to accept the canonical audition source, while the existing `TTSCloneReferenceMaterializer` remains the only owner of temporary request paths. Reference-bearing Voice Profile previews carry only repository/profile identity through UI navigation and resolve the exact private reference again at service admission. Profile save performs one atomic profile-plus-reference repository mutation, then optionally hands the newly saved profile to the existing Roleplay character-assignment UI, where choosing it from the existing Voice Profile selector remains the explicit assignment action. Console's existing store-issued message snapshot, character resolver, one-child lifecycle, and playback path remain the Roleplay execution authority.

**Tech Stack:** Python 3.11+, `asyncio`, Textual 8.x, immutable dataclasses, SQLite transactions, existing TTS adapter registry/audio.cpp supervisor/materializer, pytest Pilot tests, Ruff, mypy.

**Normative design:** `Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md` (`GM-ARCH-001`, `GM-VOICE-006`, `GM-UX-010`–`014`, Journey 2, Journey 8, `GM-ERR-001`–`003`, `GM-TEST-006`–`007`, `GM-AC-015`, `GM-AC-020`–`026`).

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/028-character-tts-generation-profile-ownership.md`, `backlog/decisions/040-speech-lab-current-result-and-auto-play.md`, `backlog/decisions/051-private-tts-clone-reference-assets.md`

**Reason:** ADR-028 already owns reusable profiles and explicit character assignments, ADR-040 owns the single Speech Lab current result and autoplay behavior, and ADR-051 already fixes canonical private references, typed Guided admission, exact-request materialization, and cleanup. This task connects those accepted owners without creating a new storage, provider, or UI authority.

**Deliberate exclusions:** No External or user-JSON clone path, no generic provider clone redesign, no ordinary export of reference bytes, no explicit voice bundle (TASK-13206), no Model Library work, no Windows ACL claim (TASK-13208), no new audio.cpp recipe, and no second Roleplay profile/assignment store.

---

### Task 1: Add a path-free transient clone-audition contract

**Files:**
- Modify: `tldw_chatbook/TTS/playground_types.py`
- Modify: `tldw_chatbook/TTS/effective_settings.py`
- Modify: `tldw_chatbook/TTS/request_admission.py`
- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/profile_reference_materialization.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/UI/Speech/speech_synthesis_mixin.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `Tests/TTS/test_stts_playground_types.py`
- Modify: `Tests/TTS/test_tts_request_admission.py`
- Modify: `Tests/TTS/test_profile_service.py`
- Modify: `Tests/TTS/test_profile_reference_materialization.py`
- Modify: `Tests/TTS/test_stts_audio_cpp_generation.py`
- Modify: `Tests/UI/test_speech_synthesis_wiring.py`

- [ ] **Step 1: Write failing immutable-ingress and admission tests**

Add tests for one exact transient clone value captured by `STTSPlaygroundRequest`, including exact type validation, permanently redacted `repr`, deep isolation from later UI draft edits, and rejection of a clone reference for any provider other than audio.cpp. Cover Guided Managed success plus External/user-JSON rejection before HTTP, child launch, or materialization. Assert a canonical transient reference and a stored `TTSCloneReference` both reach the same materializer contract, but a raw path, arbitrary bytes, generic option, copied internal admission, or stale process generation does not.

Also add a reference-bearing Voice Profile preview regression. Extend the navigation preset and STTS event with only exact profile UUID, repository generation, and profile revision—never the BLOB, transcript, digest, or a path. The UI and event handler must not call `get_reference()` or receive `TTSCloneReference`. For a preview synthesis only, the app-owned handler passes the coordinator a narrow async reference-resolver callback bound to `TTSProfileService`; the handler never invokes it or receives its result. The coordinator invokes it with the exact token before acquiring a registry/provider lease, exact-type validates the returned private reference, and proves a stale revision/generation acquires no lease and performs no readiness, HTTP, launch, or materialization work. A reference-free preset keeps its current path.

```python
def test_playground_clone_snapshot_is_path_free_and_redacted() -> None:
    snapshot = STTSPlaygroundCloneSnapshot(
        draft_revision=3,
        canonical_reference=canonical_reference,
    )
    assert "PRIVATE" not in repr(snapshot)
    assert not hasattr(snapshot, "source_path")

@pytest.mark.asyncio
async def test_transient_clone_uses_existing_typed_materialization_lifetime(...):
    response, selection = await service.synthesize_effective(
        text="hello",
        studio_draft=studio_draft,
        studio_preferences=studio_preferences,
        clone_audition=snapshot,
    )
    ...
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_profile_reference_materialization.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/UI/test_speech_synthesis_wiring.py -q
```

- [ ] **Step 3: Implement one narrow transient selection**

Add a private-value `STTSPlaygroundCloneSnapshot` containing only a nonnegative draft revision and exact `CanonicalTTSCloneReference`; its representation must never expose bytes, transcript, digest, or source path. Add it as an optional exact field on `STTSPlaygroundRequest`. Reject it unless the request is audio.cpp and Studio-effective selection is complete.

Extend `TTSRequestAdmissionCoordinator.synthesize_effective()` and `TTSService.synthesize_effective()` with one typed `clone_audition` argument. It is mutually exclusive with character/default/profile-preview references and is valid only when the resolved selection remains the exact audio.cpp Studio draft model. Include it in the existing side-effect-free Guided-source preflight before readiness. Keep the existing `_ResolvedTTSCloneExecution` for stored and admission-resolved preview profiles, where real profile UUID/repository generation/profile revision exist. Add a separate exact private `_ResolvedTransientTTSCloneExecution` for `CanonicalTTSCloneReference`, tagged and exact-type checked at every dispatch boundary. Never fabricate a profile UUID or repository revision for a transient audition. Keep public `TTSRequest` and generic `options` unchanged.

Extend `TTSPlaygroundSelectionPreset` with optional exact identity tokens only. `TTSProfileService.preview_preset()` builds those tokens from `LoadedTTSProfile`; `speech_synthesis_mixin.py` rechecks the preset identity/current UI selection and posts only that typed preview-profile token. The STTS handler forwards the token plus a private async resolver callable bound to the already app-owned profile service; it does not call the resolver. Define a narrow `TTSProfileReferenceResolver` protocol in admission code rather than giving `TTSService` ownership of, or construction responsibility for, `TTSProfileService` (which already depends on `TTSService`). Before provider lease acquisition, `TTSRequestAdmissionCoordinator` invokes the resolver with exact UUID/repository generation/profile revision, exact-type validates the returned reference, and keeps it below the TTS service boundary. Stale, missing, throwing, or forged resolver results fail closed with a bounded error and zero provider work. Do not flatten a reference-bearing profile into ordinary Studio selection, and do not put private reference values in Textual navigation context or STTS messages.

Extend `TTSCloneReferenceMaterializer.materialize()` to accept exactly the two validated private reference types (`TTSCloneReference` and `CanonicalTTSCloneReference`) and normalize their canonical fields internally. Do not add a second filesystem owner, path API, lifecycle task set, or cleanup order. The response must retain the materialization and registry lease exactly as TASK-13204 already guarantees.

- [ ] **Step 4: Mutation-check the source and exact-selection fences**

Temporarily allow the clone snapshot on External, on a different resolved model, and without the source preflight. Confirm the zero-HTTP/zero-launch/zero-materialization and model-mismatch tests fail, then restore the guards.

- [ ] **Step 5: Commit the transient admission contract**

```bash
git add tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/effective_settings.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/profile_reference_materialization.py \
  tldw_chatbook/TTS/TTS_Generation.py tldw_chatbook/TTS/__init__.py \
  tldw_chatbook/UI/Speech/speech_synthesis_mixin.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_profile_reference_materialization.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/UI/test_speech_synthesis_wiring.py
git commit -m "feat(tts): admit transient guided clone auditions"
```

### Task 2: Persist one successful reference atomically with its profile

**Files:**
- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/playground_types.py`
- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `Tests/TTS/test_profile_reference_repository.py`
- Modify: `Tests/TTS/test_profile_service.py`
- Modify: `Tests/TTS/test_tts_request_admission.py`
- Modify: `Tests/TTS/test_stts_audio_cpp_generation.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`

- [ ] **Step 1: Write failing repository/service tests**

Cover an atomic `create profile + reference` transaction, normalized-name and UUID collision, quota failure, cancellation, injected failure between profile/reference writes, stale repository generation, malformed canonical reference, and exact successful-artifact provenance. Assert an ordinary failure commits neither row. For caller cancellation, match the repository's retained shielded-worker contract: cancellation before admission/commit leaves zero rows; cancellation after admitted work may settle to both rows, but never one row, and lifecycle close joins the retained worker. Assert the service never reopens a source path and cannot save a failed, stale, or reference-free artifact as a clone profile.

```python
@pytest.mark.asyncio
async def test_create_profile_with_reference_rolls_back_both_rows(...):
    ...

@pytest.mark.asyncio
async def test_create_clone_profile_uses_exact_successful_artifact(...):
    loaded = await service.create_clone_from_artifact("Mira", artifact)
    stored = await service.get_reference(...)
    assert stored.wav_bytes == successful_canonical.wav_bytes
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_repository.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_audio_cpp_managed_integration.py -q
```

- [ ] **Step 3: Add one repository-owned atomic mutation**

Add `create_profile_with_reference()` to the repository protocol and implementation with an explicit required `expected_generation`. Validate that generation, the exact draft, and canonical WAV before queueing; recheck generation inside the serialized transaction; insert profile and reference atomically; enforce reference count/byte quotas there; and return one canonical `TTSGenerationProfile` whose reference summary matches the committed row. Define and test the new profile revision once; do not emulate create-then-edit as two public mutations.

Define a private immutable `TTSCloneGenerationEvidence` carrying the exact canonical reference plus admitted model, recipe ID/revision, provider configuration revision/applied generation, and process generation. It must be permanently redacted and absent from logs, response metadata, and public provenance. The evidence must originate inside `_AdmittedTTSOperation` from the exact adapter-issued `AudioCppCloneCapabilityAdmission` only after `synthesize_clone()` succeeds; a passive runtime observation is not success authority.

Expose that evidence to the retained STTS worker through a dedicated internal synthesis-result channel (for example, `synthesize_effective_with_evidence()` returning a typed internal result), while preserving the existing public two-tuple `synthesize_effective()` API for current callers. The handler attaches the evidence to `STTSGeneratedAudio` only after the complete WAV passes structural validation. The artifact field must be `repr=False` and exact-type validated. Add `TTSProfileService.create_clone_from_artifact()` to verify the successful artifact's exact selection/reference/model/configuration/recipe evidence, then call the atomic repository mutation. Keep the existing reference-free `create_from_artifact()` unchanged.

- [ ] **Step 4: Prove the transaction and source-independence guards discriminate**

Temporarily split the mutation into two transactions and delete the artifact-carried reference. Confirm rollback and deleted/replaced-source tests fail, then restore the atomic exact-artifact path.

- [ ] **Step 5: Commit exact clone-profile persistence**

```bash
git add tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  Tests/TTS/test_profile_reference_repository.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_audio_cpp_managed_integration.py
git commit -m "feat(tts): save successful clone references atomically"
```

### Review checkpoint A

- [ ] Re-read Tasks 1–2 against AC #3–#6, ADR-051, and `GM-VOICE-006`.
- [ ] Confirm no source path enters an immutable request, artifact, profile, log, error, metadata, or provenance line.
- [ ] Confirm no profile exists before explicit save and failed atomic save leaves the successful transient result retryable.
- [ ] Run `git diff --check` and both focused test commands.

### Task 3: Build the Speech Lab reference-required setup and immutable action projection

**Files:**
- Modify: `tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playground_pane.py`
- Create: `tldw_chatbook/UI/Speech/speech_clone_setup.py`
- Modify: `tldw_chatbook/UI/Speech/speech_synthesis_mixin.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playback_mixin.py`
- Modify: `tldw_chatbook/UI/Speech/speech_catalog_mixin.py`
- Modify: `tldw_chatbook/UI/Speech/speech_profile_mixin.py`
- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/playground_types.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_speech_playground_pane.py`
- Modify: `Tests/UI/test_speech_playground_pane_lifecycle.py`
- Modify: `Tests/UI/test_speech_synthesis_wiring.py`
- Modify: `Tests/UI/test_speech_profile_navigation.py`
- Modify: `Tests/TTS/test_stts_audio_cpp_generation.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`

- [ ] **Step 1: Write failing projection, validation, and workflow tests**

Use the real `SpeechPlaygroundPane` in Pilot tests. Cover:

- stopped reference-required Guided model → `Start & Set Up Voice`;
- deliberate start/test establishes matching server/catalog before showing setup;
- ready matching model with no draft → `Create Voice & Generate` disabled with a current reason;
- existing compatible profile remains reachable through the existing Voice Profiles library/Preview path and says Voice Profile, never User Profile or Persona;
- WAV picker cancellation, oversized/invalid/nonregular input, missing/oversized transcript, exact recipe guidance, and plaintext warning;
- field-specific errors preserve the other field and focus the invalid control;
- canonicalization runs off the event loop, is revision-fenced, and a late result cannot replace a newer draft/provider/model;
- replacing/clearing/unmounting closes the exact staged owner and mounts no hidden focusable controls;
- successful result acceptance drops only the matching setup draft, leaves no artifact/canonical reference in the pane, and routes Play through the sanitized current operation ID;
- keyboard-only and 80x24/100x30 geometry keep setup, primary action, error, and current result reachable.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_speech_playground_pane.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  Tests/UI/test_speech_synthesis_wiring.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_audio_cpp_managed_integration.py -q
```

- [ ] **Step 3: Add a focused clone-setup component and extend the existing pure projection**

First extend the passive `AudioCppRuntimeObservation` with a bounded, path-free selected-model clone setup projection derived by `TTSService` from the exact saved/applied Guided recipe evidence. It contains only the public model ID, recipe/family display labels, recipe revision, and voice/reference policy needed by the UI. It must not expose package/model/config paths or make the UI query adapter-private settings. Fence it to the same saved/applied/process/catalog identities as the runtime action projection.

Implement one compact `SpeechCloneSetup` component, mounted only when that exact selected audio.cpp catalog model's accepted Guided recipe requires or permits a reference. It owns presentation and input events, not canonical bytes, processes, profiles, or persistence. Show:

- `Choose reference WAV` and safe basename-free selected status;
- bounded transcript `TextArea` with remaining/limit guidance;
- exact recipe/family/reference guidance from the accepted recipe projection;
- plain copy that audio/transcript are local plaintext, filesystem controls are not encryption, and deletion is best effort;
- a `Use an existing Voice Profile` handoff to the existing Voice Profiles library/Preview flow, preserving the unsaved clone draft and avoiding a second profile picker; and
- one `Create Voice & Generate` action.

Extend the existing immutable audio.cpp action projection so the visible label, operation, disabled reason, tooltip, progress label, and focus target all come from the same observation plus clone-draft snapshot. The click handler must execute the stored projected operation. Starting a stopped reference-required model starts/tests only; generation begins only after a valid staged reference exists and the exact matching catalog/model remains current.

Canonicalize with `canonicalize_reference_wav()` in a retained worker using the pane's monotonically increasing provider/model/draft revision. Publish only if all revisions still match. Before generation, keep the canonical object in one pane-owned draft slot; replacing or clearing drops the prior object. On unmount/app close, seal/cancel the worker, join it, and clear only the staged draft without touching source files. The pane must not retain `STTSGeneratedAudio` or any other object that owns the successful canonical reference. It stores only the current operation ID and sanitized playback/save projection. Current successful artifacts remain under the STTS handler's existing lease/release lifecycle and are cleared by result replacement, explicit retirement, or handler/app close.

Extend `STTSPlaygroundGenerateEvent` handling so the immutable request snapshot carries the exact canonical clone draft to Task 1 admission. A successful complete WAV attaches that same canonical object to the handler-owned artifact. Acceptance as the handler's current result atomically publishes a sanitized operation-ID projection and tells the pane to drop the matching setup draft; a stale acknowledgement cannot clear a newer draft. Failures retain the prior playable handler artifact and keep the valid setup draft for retry. Result replacement, discard, and handler/app close release the sole successful canonical owner.

- [ ] **Step 4: Rebuild CSS and mutation-check stale/busy handling**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python \
  tldw_chatbook/css/build_css.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m \
  tldw_chatbook.css.check_bundle_sync
```

Delete the provider/model/draft revision check and temporarily let Start generate immediately. Confirm the late-result and start-before-reference tests fail, then restore the guards.

- [ ] **Step 5: Commit the Speech Lab clone setup**

```bash
git add tldw_chatbook/UI/Speech/audio_cpp_runtime_card.py \
  tldw_chatbook/UI/Speech/speech_playground_pane.py \
  tldw_chatbook/UI/Speech/speech_clone_setup.py \
  tldw_chatbook/UI/Speech/speech_synthesis_mixin.py \
  tldw_chatbook/UI/Speech/speech_playback_mixin.py \
  tldw_chatbook/UI/Speech/speech_catalog_mixin.py \
  tldw_chatbook/UI/Speech/speech_profile_mixin.py \
  tldw_chatbook/UI/STTS_Window.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/__init__.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  tldw_chatbook/css/features/_lab.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_speech_playground_pane.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  Tests/UI/test_speech_synthesis_wiring.py \
  Tests/UI/test_speech_profile_navigation.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_audio_cpp_managed_integration.py
git commit -m "feat(speech): add guided clone setup and audition"
```

### Task 4: Save the exact current voice and hand off explicit assignment

**Files:**
- Modify: `tldw_chatbook/UI/Speech/speech_profile_mixin.py`
- Modify: `tldw_chatbook/UI/Speech/speech_playback_mixin.py`
- Modify: `tldw_chatbook/UI/stts_profile_library.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`
- Modify: `Tests/UI/test_speech_playground_pane.py`
- Modify: `Tests/UI/test_speech_playground_pane_lifecycle.py`
- Modify: `Tests/UI/test_stts_profile_library.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Modify: `Tests/UI/test_speech_profile_navigation.py`

- [x] **Step 1: Write failing save/review/handoff tests**

Cover Save as Voice Profile visibility only for a sanitized projection of the handler's idle successful clone artifact; failed generation, stale settings, late replaced operation ID, active generation, and reference-free result cannot expose a false clone-save action. Cover modal cancellation, name validation, save failure/retry, exact reference persistence without source reopen, and stale modal completion after result replacement/discard. The review must offer `Save unassigned` and `Save & choose character`; neither changes the global default nor any assignment. The latter navigates to Roleplay, preserves the new profile as a suggested choice, and still requires the user to select a character and explicitly choose that profile from the existing Voice Profile selector.

- [x] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_speech_playground_pane.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  Tests/UI/test_stts_profile_library.py \
  Tests/UI/test_personas_workbench.py \
  Tests/UI/test_speech_profile_navigation.py -q
```

- [x] **Step 3: Extend the existing profile-save and Roleplay owners**

Replace the name-only save modal for clone-eligible results with a concise review result containing the validated name and one explicit post-save choice. Keep the existing modal and service path for reference-free artifacts. The pane captures only the sanitized current operation ID before opening the modal. On submit it posts an operation-ID save request to the STTS handler; the handler verifies that ID still names its current artifact, rechecks exact artifact/configuration evidence, and calls `create_clone_from_artifact()` while it retains sole artifact ownership. The pane never receives or captures the artifact/canonical object. Replacement, discard, or close invalidates the ID, so a stale modal completion cannot save.

For `Save & choose character`, post a bounded, non-authoritative Roleplay suggestion containing the created profile UUID plus the repository generation and profile revision returned by the atomic save. Add `PersonasScreen.apply_navigation_context()` support that opens Characters/Voice & Speech, records that exact suggestion separately from persisted assignment state, and focuses stable guidance. The Roleplay owner compares it with a freshly loaded profile page and clears/refuses the suggestion if UUID, repository generation, or profile revision is stale. The existing widget has no separate Assign button: changing its Voice Profile `Select` is the explicit assignment action. Therefore the handoff must never write or preselect the `Select` value. Once a character and exact fresh profile page/availability are current, render path-free guidance that names/highlights the saved profile as the suggested choice and focus the selector; assignment occurs only when the user deliberately selects it. Clear the suggestion after that selection, explicit dismissal, source change, stale/unavailable resolution, or screen teardown.

Do not list characters from Speech Lab, construct `CharacterRef` there, or duplicate Roleplay's local/server authority capture. Do not mutate `[app_tts].default_profile_id`.

- [x] **Step 4: Mutation-check that navigation is not assignment**

Temporarily call `set_assignment()` on navigation or conflate suggested and persisted selection. Confirm the no-silent-assignment/global-default and stale-authority tests fail, then restore the explicit existing action boundary.

- [x] **Step 5: Commit save and assignment handoff**

```bash
git add tldw_chatbook/UI/Speech/speech_profile_mixin.py \
  tldw_chatbook/UI/Speech/speech_playback_mixin.py \
  tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py \
  Tests/UI/test_speech_playground_pane.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  Tests/UI/test_stts_profile_library.py Tests/UI/test_personas_workbench.py \
  Tests/UI/test_speech_profile_navigation.py
git commit -m "feat(tts): save and assign clone voice profiles"
```

### Review checkpoint B

- [x] Re-read Tasks 3–4 against AC #1–#6 and `GM-UX-010`–`014`.
- [x] Inspect real mounted layouts at 80x24, 100x30, and 120x35; assert neighbors remain on-screen after dynamic labels reflow.
- [x] Confirm every disabled action has current reason/tooltip and every visible action executes its own stored projection.
- [x] Confirm `User Profiles`, Personas, Voice Profiles, and character assignments remain distinct labels and owners.
- [x] Run `git diff --check`, CSS bundle sync, and both focused test commands.

### Task 5: Prove assigned character Roleplay Speak uses the exact clone revision lazily

**Files:**
- Modify: `Tests/TTS/test_console_speech_snapshot_admission.py`
- Modify: `Tests/TTS/test_console_audio_cpp_native.py`
- Modify: `Tests/TTS/test_console_speak_autoplay.py`
- Modify: `Tests/UI/test_uat_first_time_character_chat.py`
- Modify: `Tests/Chat/test_console_context_compaction.py`
- Modify production only if a failing end-to-end test exposes a missing seam in:
  - `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`
  - `tldw_chatbook/Chat/console_speech.py`
  - `tldw_chatbook/UI/Screens/chat_screen.py`
  - `tldw_chatbook/Chat/console_context_compaction.py`

- [x] **Step 1: Add a production-boundary lazy-roleplay regression**

Build the smallest real owner chain: a profile repository containing a reference-bearing audio.cpp profile, an explicit local character assignment, a Console store-issued assistant message snapshot carrying that exact `CharacterRef`, the real `TTSHandler` resolution path, and a fake/provisioned native adapter. Assert:

- browsing the profile library, Roleplay character list, and assignment state launches no child and performs no synthesis;
- pressing Speak captures exact character/profile/reference/repository revision before startup;
- first use starts or joins exactly one eligible Guided Managed child;
- the adapter receives the exact assigned model and canonical reference, never the global/default voice;
- a complete structurally valid WAV is played through the existing Console playback path;
- profile/reference edit, assignment change, message edit, or child replacement before admission fails stale rather than switching identity;
- an edit after admission affects only the next Speak request.

- [x] **Step 2: Run focused tests and verify whether production is already green**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_console_speech_snapshot_admission.py \
  Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/UI/test_uat_first_time_character_chat.py -q
```

If the new regression passes on the existing runtime, make no production change. The purpose is to prove the already-shipped TASK-13204 seam through the user-facing Roleplay/Console path, not redesign it.

- [x] **Step 3: Fix only a demonstrated missing seam, if any**

Use `superpowers:systematic-debugging` before changing production. Preserve store-issued message snapshot validation, character resolver authority, one-child startup joining, stable refusal/global-override behavior, and existing playback ownership. Do not add a second Roleplay Speak implementation.

The existing first-time-character UAT exposed a rebased Console compaction defect:
character sessions intentionally begin with an assistant greeting, while the new
compaction grouper assumed every prefix began with a user message. The bounded fix
keeps the greeting in the live conversation but excludes pre-user rows from
compactable exchange units.

- [x] **Step 4: Mutation-check character specificity and passivity**

Temporarily remove assignment precedence or make browsing call startup. Confirm the exact-profile and passive-browsing tests fail, then restore the boundaries.

- [x] **Step 5: Commit Roleplay journey coverage**

```bash
git add Tests/TTS/test_console_speech_snapshot_admission.py \
  Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/UI/test_uat_first_time_character_chat.py \
  Tests/Chat/test_console_context_compaction.py \
  tldw_chatbook/Chat/console_context_compaction.py \
  tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py \
  tldw_chatbook/Chat/console_speech.py \
  tldw_chatbook/UI/Screens/chat_screen.py
git commit -m "test(tts): cover lazy assigned clone roleplay speech"
```

### Task 6: Close races, cleanup, privacy, and accessibility

**Files:**
- Modify: `Tests/TTS/test_tts_logging_privacy.py`
- Modify: `Tests/TTS/test_tts_request_admission.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`
- Modify: `Tests/UI/test_speech_playground_pane_lifecycle.py`
- Modify: `Tests/UI/test_speech_live_render_defects.py`
- Modify production files only where the matrix proves a defect

- [x] **Step 1: Add the cross-owner failure matrix**

Use unique canaries for source path, transient materialization path, transcript, digest, WAV bytes, prompt, and raw collaborator errors. Assert none appear in log records, toasts, diagnostics, artifact metadata/provenance, screenshots/text captures, or exception `str`/`repr`/`args`/notes/cause/context graphs.

Cover cancellation during file pick/canonicalization/admission/materialization/HTTP/result delivery/profile save/navigation; provider/model/profile/reference changes during each phase; failed passive runtime observation; generation retry; playback failure; pane/app close; and response cleanup. Assert:

- the last playable result is retained on every later failure;
- only the handler-owned successful current result authorizes Save as Voice Profile;
- the pane drops the matching setup canonical when success is accepted and never retains the successful artifact; failures keep the setup draft for retry;
- result replacement/discard and handler/app close make the successful canonical unreachable from both handler and pane, and stale operation-ID modal completion cannot save;
- the request materialization is gone before the response lease releases;
- no busy label/control remains stranded after failure;
- focus returns to the invalid field, retry, Play, Voice Profile selector, or stable primary action as appropriate;
- live regions announce phase transitions once, not progress spam;
- diagnostics remain focusable/scrollable and narrow layouts have no clipped action.

- [x] **Step 2: Run focused tests and verify RED/green boundaries**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  Tests/UI/test_speech_live_render_defects.py -q
```

- [x] **Step 3: Apply only failures demonstrated by the matrix**

Normalize ordinary collaborator errors outside caught exception contexts, retain exact cleanup tasks through close, and fence late UI publication with provider/model/draft/operation revisions. Keep source files user-owned: clearing a staged clone forgets Chatbook's canonical in-memory object and private materialization only; it never deletes the user's WAV.

- [x] **Step 4: Run accessibility/CSS gates**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m \
  tldw_chatbook.css.check_bundle_sync
git diff --check
```

- [x] **Step 5: Commit hardening**

```bash
git add tldw_chatbook Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_tts_request_admission.py Tests/TTS/test_tts_app_ownership.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  Tests/UI/test_speech_live_render_defects.py
git commit -m "test(tts): harden clone workflow races and privacy"
```

### Review checkpoint C

- [x] Re-read the complete diff against all nine TASK-13205 acceptance criteria.
- [x] Confirm no TASK-13206 bundle or TASK-13208 Windows scope entered the branch.
- [x] Confirm the current-result owner, profile repository, Roleplay assignment owner, request admission, and materializer each retain only their own authority.
- [x] Run `git diff --check`, CSS sync, all Task 1–6 focused commands, and mutation checks.

### Task 7: Document, verify, UAT, review, and close TASK-13205

**Files:**
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Create: `Docs/superpowers/qa/audio-cpp-clone-workflow-2026-08-11/live-uat.md`
- Modify: `backlog/tasks/task-13205 - Add-clone-setup-and-character-voice-workflows.md`
- Modify: `backlog/docs/lessons-*.md` only if implementation produces a concrete reusable incident

- [ ] **Step 1: Update implementation-truth documentation**

Document the reference-required Speech Lab flow, local-plaintext warning, transient-vs-profile ownership, exact successful-artifact save, explicit Roleplay assignment, lazy character Speak behavior, failure recovery, and cleanup. State that External/user-JSON clone paths, ordinary reference export, voice bundles, Windows parity, and non-evidenced recipes remain unsupported/deferred.

- [ ] **Step 2: Run the complete automated verification**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_types.py \
  Tests/TTS/test_profile_reference_audio.py \
  Tests/TTS/test_profile_reference_storage.py \
  Tests/TTS/test_profile_reference_repository.py \
  Tests/TTS/test_profile_reference_materialization.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_audio_cpp_recipes.py \
  Tests/TTS/test_audio_cpp_contract.py \
  Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/TTS/test_console_speech_snapshot_admission.py \
  Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/TTS/test_stts_playground_types.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/UI/test_speech_playground_pane.py \
  Tests/UI/test_speech_playground_pane_lifecycle.py \
  Tests/UI/test_speech_synthesis_wiring.py \
  Tests/UI/test_stts_profile_library.py \
  Tests/UI/test_personas_workbench.py \
  Tests/UI/test_speech_profile_navigation.py \
  Tests/UI/test_uat_first_time_character_chat.py \
  Tests/UI/test_speech_live_render_defects.py -q

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/TTS tldw_chatbook/UI/Speech \
  tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py \
  tldw_chatbook/Event_Handlers/STTS_Events/stts_events.py \
  tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py \
  tldw_chatbook/Chat/console_speech.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/TTS Tests/UI

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/playground_types.py \
  tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/profile_reference_materialization.py \
  tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/UI/Speech/speech_clone_setup.py \
  tldw_chatbook/UI/Speech/speech_profile_mixin.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m \
  tldw_chatbook.css.check_bundle_sync
git diff --check
```

Record any environment-only failures separately; never count them as passing evidence.

- [ ] **Step 3: Run clean-profile real-process UAT**

Use an isolated config, data directory, profile database, and provisioned exact supported `release-0.5.1` binary/model/reference tuple. Never point the schema-v3 build at the developer's live profile store. Record exact app commit, binary tag/commit/origin/identity, OS/architecture/backend, recipe/package/model identities, the redacted generated-config digest, safe reference metadata/digest only, structural WAV metadata, one PID/startup-join evidence, assignment/profile revisions, and definitive teardown. After discard and app exit, explicitly prove no owned child, process handle, supervisor/client/task, handler artifact, or private request materialization remains.

Perform the real user journey: Guided save without launch → Speech Lab `Start & Set Up Voice` → choose bounded WAV/transcript → `Create Voice & Generate` → Play and human audible confirmation → `Save as Voice Profile` → open Roleplay, select a character, and explicitly choose the saved Voice Profile → start/continue a character Console session → Speak the character response → Play and human audible confirmation → shutdown/app exit. Confirm passive profile/character browsing does not start audio.cpp. Evidence must contain no source/materialization path, transcript, prompt, or audio bytes.

- [ ] **Step 4: Request review and address every validated finding**

Use `superpowers:requesting-code-review` against the merge-base diff. Apply `superpowers:receiving-code-review`: reproduce each finding, add a failing regression, make the smallest fix, rerun the affected matrix, and repeat until no Critical/Important/Minor findings remain.

- [ ] **Step 5: Complete Backlog Definition of Done**

Check every acceptance criterion only after automated verification and UAT evidence exist. Add concise Implementation Notes naming the ownership boundaries, exact tests/UAT, and ADR check. Add a lessons entry only for a real reusable incident. Set TASK-13205 to Done through Backlog CLI.

- [ ] **Step 6: Commit closeout**

```bash
git add Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md \
  Docs/superpowers/qa/audio-cpp-clone-workflow-2026-08-11/live-uat.md \
  'backlog/tasks/task-13205 - Add-clone-setup-and-character-voice-workflows.md'
# If Step 5 recorded a real reusable incident, stage the exact changed
# backlog/docs/lessons-*.md file in this commit too.
git commit -m "docs(tts): close clone character workflow task"
```

The branch is then ready for the repository's normal PR/rebase/review/merge workflow. Do not begin TASK-13206 until TASK-13205 is merged and the user explicitly continues.
