# Guided audio.cpp Model Library Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every reviewed audio.cpp TTS package discoverable through the existing Model Library, install exact pinned package roots without side effects, return them to the originating Guided Settings draft, lease them during use, and remove them only after a truthful dependency review.

**Architecture:** Keep the recipe registry as compatibility authority and add one static, pinned artifact-source manifest joined into the existing curated registry only after recipe accounting is complete. Reuse `ArtifactAcquisitionService`, `ModelArtifactService`, the existing audio.cpp scanner, complete Settings panel snapshots, typed handoff owners, and the generated-launch lifetime. Add only one inactive-root lease seam, one narrow consumer-lease coordinator, and one removal authority/probe that preserve the artifact store's lifecycle→artifact lock order. Model Library and Settings exchange typed, opaque, process-local handoffs, while all persisted package evidence remains backward compatible.

**Tech Stack:** Python 3.11, standard-library JSON/urllib/hashlib, Pydantic v2 frozen models, Textual 8, existing Model Artifacts store/acquisition/lease APIs, pytest/pytest-asyncio.

**Required implementation disciplines:** `@superpowers:test-driven-development`, `@ponytail`, `@textual-tui`, and `@impeccable` for the mounted UI tasks; `@superpowers:verification-before-completion` before every completion claim.

**ADR required:** no new ADR.

**ADR path:** `backlog/decisions/050-audio-cpp-generated-model-setup-ownership.md` (amended by the approved design); `backlog/decisions/051-private-tts-clone-reference-assets.md` remains applicable to private clone-reference ownership.

**Reason:** TASK-13207 implements the package ownership and runtime boundaries already decided in ADR-050/051; the approved design amendment records the only new long-lived decision.

---

## File structure

### New files

- `tldw_chatbook/TTS/audio_cpp_artifact_catalog.py` — parse the checked-in source manifest, join it to exact recipes, derive curated descriptors/source maps, and expose the three-state accounting projection without network work.
- `tldw_chatbook/TTS/audio_cpp_artifact_manifest.json` — reviewed source-only facts pinned to one Hugging Face repository commit.
- `scripts/refresh_audio_cpp_artifact_manifest.py` — maintainer-only explicit-revision refresh/audit command; never imported by runtime code.
- `tldw_chatbook/UI/Navigation/audio_cpp_model_handoff.py` — frozen, redacted request/result values shared by Settings and Model Library.
- `tldw_chatbook/TTS/audio_cpp_artifact_dependencies.py` — bounded dependency snapshot/fingerprint plus the narrow app-injected shared-root lease coordinator; no UI, repository storage, or singleton.
- `Tests/TTS/test_audio_cpp_artifact_catalog.py` — manifest parsing, recipe join, accounting, and network-isolation tests.
- `Tests/TTS/test_audio_cpp_artifact_dependencies.py` — consumer projection, fingerprint, privacy, and drift tests.
- `Tests/UI/test_audio_cpp_model_library_handoff.py` — process-local handoff, full-draft preservation, stale return, and install-result tests.
- `Docs/superpowers/qa/audio-cpp-model-library-2026-08-13/live-uat.md` — sanitized opt-in clean-root evidence.

### Existing files to modify

- `tldw_chatbook/TTS/audio_cpp_recipes.py` and `Tests/TTS/test_audio_cpp_recipes.py` — complete the 67-row support accounting with data-only recipes/reasons.
- `tldw_chatbook/Model_Artifacts/curated_registry.py` and `Tests/Model_Artifacts/test_curated_registry.py` — register only admitted audio.cpp descriptors.
- `tldw_chatbook/TTS/audio_cpp_guided_config.py`, `tldw_chatbook/TTS/audio_cpp_recipes.py`, `tldw_chatbook/TTS/audio_cpp_package_scanner.py`, and their focused tests — persist and validate optional managed artifact identity.
- `tldw_chatbook/UI/Navigation/pending_handoff_store.py` and its tests — add typed request/result channels with exact detached-copy validation.
- `tldw_chatbook/UI/Screens/model_curated_view.py`, `tldw_chatbook/UI/Screens/llm_screen.py`, and existing Model Library tests — filter audio.cpp rows, provision with `activate=False`, and deliver exact installed results.
- `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py`, `tldw_chatbook/UI/Screens/settings_screen.py`, and Settings tests — preserve the complete non-secret draft, stage/consume the handoff, scan the exact root, and merge one candidate without saving.
- `tldw_chatbook/TTS/audio_cpp_guided_launch.py`, `tldw_chatbook/TTS/audio_cpp_supervisor.py`, and focused runtime tests — acquire and retain managed artifact handles for staged/live lifetimes and preserve cleanup authority for retry.
- `tldw_chatbook/Model_Artifacts/service.py`, `tldw_chatbook/Model_Artifacts/__init__.py`, and `Tests/Model_Artifacts/test_service.py` — add inactive-root acquisition, the single-lock removal authority/probe, and delegate existing `delete()` to it.
- `tldw_chatbook/TTS/profile_service.py`, `tldw_chatbook/TTS/voice_bundle_service.py`, `tldw_chatbook/UI/Screens/model_installed_view.py`, their tests, and app ownership tests — serialize consumer mutations, preview/revalidate audio.cpp consumers, and commit removal without recursive lock acquisition.
- `tldw_chatbook/UI/LLM_Management_Window.py`, `tldw_chatbook/css/_lab.tcss`, generated `tldw_chatbook/css/tldw_cli_modular.tcss`, and UI tests — render truthful state dimensions, companion disclosure, narrow layouts, and blocked removal actions.
- `tldw_chatbook/app.py` and `Tests/TTS/test_tts_app_ownership.py` — provide existing app-owned collaborators to the Model Library/removal boundary and verify shutdown ordering; do not create a second artifact service.
- `backlog/tasks/task-13207 - Integrate-guided-audio.cpp-packages-with-Model-Library.md`, ADR-050, relevant user docs, and lessons only if an evidenced reusable trap appears — final task/decision/documentation hygiene.

---

### Task 1: Add the bounded manifest schema and maintainer refresh tool

**Files:**
- Create: `tldw_chatbook/TTS/audio_cpp_artifact_manifest.json`
- Create: `tldw_chatbook/TTS/audio_cpp_artifact_catalog.py`
- Create: `scripts/refresh_audio_cpp_artifact_manifest.py`
- Create: `Tests/TTS/test_audio_cpp_artifact_catalog.py`
- Modify: `pyproject.toml`
- Modify: `MANIFEST.in`
- Modify: `Tests/Packaging/test_installed_distribution.py`

- [ ] **Step 1: Write failing manifest-boundary tests**

Cover exact repository/commit validation, duplicate JSON/package/path facts, traversal and drive paths, controls, malformed URLs, missing size/SHA/license facts, bounded manifest dimensions/bytes, and a guard that runtime loading succeeds while `urllib.request.urlopen` raises if touched. Recipe admission is deliberately deferred until Task 3, after Task 2 has completed the registry.

```python
def test_manifest_is_pinned_and_runtime_load_is_network_free(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _fail_network)
    catalog = load_audio_cpp_artifact_source_manifest()
    assert catalog.repository == "audio-cpp/audio.cpp-gguf"
    assert catalog.commit == "597048d9a920592808d7d4e2acd7b9c4596a143a"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_artifact_catalog.py -q`

Expected: collection/import failure because `audio_cpp_artifact_catalog` and the manifest do not exist.

- [ ] **Step 3: Implement the minimal source-only manifest model**

Use frozen dataclasses and strict exact-type checks. Keep repository and commit once at the top level; package records contain only recipe ID/revision/variant, stable artifact ID, source/managed paths, sizes, SHA-256 values, and reviewed license/usage facts.

```python
@dataclass(frozen=True, slots=True)
class AudioCppArtifactSourceFile:
    source_path: str
    managed_path: str
    size_bytes: int
    sha256: str

@dataclass(frozen=True, slots=True)
class AudioCppArtifactPackage:
    recipe_id: str
    recipe_revision: int
    package_variant: str
    artifact_id: str
    license_id: str
    license_url: str
    usage_notice: str
    files: tuple[AudioCppArtifactSourceFile, ...]
```

Do not duplicate family, display name, runtime constraint, task, backend evidence, format, precision, or model path; derive them later from the recipe. Check in a valid pinned header with an empty `packages` list so this task can be green without prematurely deciding that the incomplete recipe registry is authoritative.

- [ ] **Step 4: Add the explicit-revision maintainer refresh command**

Use only the standard library. Require `--commit` to match exactly 40 lowercase hex characters, fetch only immutable revision endpoints, follow only validated same-origin/same-repository/same-commit `rel="next"` pagination with page and aggregate-byte limits, stream/hash bounded Git-managed files, accept strict exact-type LFS SHA-256/size facts, preserve existing human-reviewed license fields by package key, and emit deterministic sorted JSON to stdout or an explicit output path. It must refuse `main`, missing reviewed license data, cross-origin/drifted pagination, and unknown file shapes.

- [ ] **Step 5: Package the runtime manifest and prove deterministic refresh**

Declare the JSON explicitly in both wheel package-data and sdist `MANIFEST.in`. Assert wheel and sdist membership and successful network-free loading from an isolated wheel install. Also assert the checked-in manifest is the exact pinned header plus an empty package list and two refreshes over the same recorded multi-page upstream fixture emit byte-identical sorted JSON. Task 3 will audit and populate real package rows only after Task 2 makes every reviewed recipe available.

- [ ] **Step 6: Run tests and static checks**

Run:

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_artifact_catalog.py -q
../../.venv/bin/python -m ruff check tldw_chatbook/TTS/audio_cpp_artifact_catalog.py scripts/refresh_audio_cpp_artifact_manifest.py Tests/TTS/test_audio_cpp_artifact_catalog.py
../../.venv/bin/python -m mypy tldw_chatbook/TTS/audio_cpp_artifact_catalog.py
../../.venv/bin/python -m pytest Tests/Packaging/test_installed_distribution.py -q
```

Expected: all pass; no test contacts Hugging Face.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/TTS/audio_cpp_artifact_catalog.py tldw_chatbook/TTS/audio_cpp_artifact_manifest.json scripts/refresh_audio_cpp_artifact_manifest.py Tests/TTS/test_audio_cpp_artifact_catalog.py pyproject.toml MANIFEST.in Tests/Packaging/test_installed_distribution.py Docs/superpowers/plans/2026-08-13-audio-cpp-model-library-integration.md
git commit -m "feat(tts): add audio cpp artifact manifest foundation"
```

### Task 2: Close the 67-package recipe support accounting

**Files:**
- Modify: `tldw_chatbook/TTS/audio_cpp_recipes.py:70-1076`
- Modify: `Tests/TTS/test_audio_cpp_recipes.py`

- [ ] **Step 1: Write the complete accounting RED test**

```python
def test_release_accounting_has_no_open_recipe_gap():
    assert len(AUDIO_CPP_RELEASE_ACCOUNTING) == 67
    assert {row.family for row in AUDIO_CPP_RELEASE_ACCOUNTING} == EXPECTED_21_FAMILIES
    assert not [row for row in AUDIO_CPP_RELEASE_ACCOUNTING if row.state == "open_gap"]
```

Also assert each approved row resolves one exact recipe, each unsupported row carries a reviewed non-empty reason/evidence reference, and artifact availability is evaluated only after support classification.

- [ ] **Step 2: Run the accounting tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_recipes.py -q`

Expected: failure showing the current 52 `open_gap` rows.

- [ ] **Step 3: Add recipes as inert data, not family branches**

Extend the existing `_recipe(...)` data construction from the pinned audio.cpp model specs. Add only validation signals the bounded scanner already supports. If a variant cannot be represented truthfully, classify it `EXPLICITLY_UNSUPPORTED` with evidence rather than weakening the matcher.

Keep the two axes independent. At this point availability is not yet joined to the still-empty checked-in manifest; Task 3 performs that join after the recipe registry is complete:

```python
support = AudioCppRecipeSupportState.APPROVED
```

No `if family == ...` installer/runtime/UI branches are allowed.

- [ ] **Step 4: Add the complete support-state projection test**

Assert every one of the 67 rows is exactly `approved` or `explicitly_unsupported`, every unsupported row has evidence, and every approved row is accepted by the bounded scanner. The later artifact join may classify an approved row as downloadable or local-only without changing support truth.

- [ ] **Step 5: Run focused recipe/scanner tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_recipes.py Tests/TTS/test_audio_cpp_package_scanner.py -q
```

Expected: all pass, with exactly 21 families/67 rows and zero open gaps.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/TTS/audio_cpp_recipes.py Tests/TTS/test_audio_cpp_recipes.py Tests/TTS/test_audio_cpp_package_scanner.py
git commit -m "feat(tts): close audio cpp recipe accounting"
```

### Task 3: Join admitted audio.cpp packages into the existing curated registry

**Files:**
- Modify: `tldw_chatbook/TTS/audio_cpp_artifact_catalog.py`
- Modify: `tldw_chatbook/TTS/audio_cpp_recipes.py` (admitted artifact IDs only)
- Modify: `tldw_chatbook/Model_Artifacts/curated_registry.py:18-91`
- Modify: `Tests/TTS/test_audio_cpp_recipes.py` (admitted artifact IDs only)
- Modify: `Tests/Model_Artifacts/test_curated_registry.py`
- Modify: `Tests/TTS/test_audio_cpp_artifact_catalog.py`
- Modify: `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` (compose-purity integration fix)
- Modify: `Tests/Wizards/test_first_run_speech_step.py` (compose-purity regression test)

Plan deviation: Task 3's expanded curated registry exposed that the speech
setup helpers unnecessarily loaded the full registry during composition. The
narrow integration fix projects the same exact Parakeet choices directly from
the existing routing policy and canonical precision constant.

- [ ] **Step 1: Write failing exact-join tests**

For every visible audio.cpp descriptor assert: one approved exact recipe; exact required-file closure; pinned URLs; source-map keys equal descriptor file paths; `consumer == "audio_cpp"`; role ROOT; derived family/precision/runtime/backend data; and no ASR/music/diarization entries. Reject manifest rows with unknown recipe IDs/revisions only here, when the completed registry exists. Add the final generated 67-row cross-axis assertion: approved+admitted → `downloadable`, approved+not admitted → `local_only`, explicitly unsupported → `explicitly_unsupported`, with exactly one outcome per row.

- [ ] **Step 2: Run the join tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_curated_registry.py Tests/TTS/test_audio_cpp_artifact_catalog.py -q`

Expected: audio.cpp descriptors are absent because the checked-in package list is still empty.

- [ ] **Step 3: Audit and populate the pinned manifest against the completed recipes**

Audit `audio-cpp/audio.cpp-gguf` at `597048d9a920592808d7d4e2acd7b9c4596a143a`. Admit a variant only after proving the complete recipe-required closure, exact source and managed paths, size/SHA-256, and artifact-specific license/usage evidence. Approved package variants absent as exact closures remain outside the manifest and therefore local-only; explicitly unsupported rows remain unsupported regardless of manifest absence. Do not invent URLs or companion files. Run the refresh tool against recorded immutable fixture metadata and review the deterministic diff before accepting it.

- [ ] **Step 4: Implement one pure manifest↔recipe join**

```python
def audio_cpp_curated_entries(
    registry: AudioCppRecipeRegistry = AUDIO_CPP_RECIPE_REGISTRY,
) -> tuple[tuple[ArtifactDescriptor, dict[str, str]], ...]:
    ...
```

Derive `ArtifactRef.revision` from the pinned manifest commit and `variant` from recipe precision/package data. Build each `/resolve/<commit>/<quoted-path>` URL locally. Fail the whole generated catalog test on an incomplete/duplicate mismatch, but do not hide or weaken a local recipe.

- [ ] **Step 5: Register entries in `curated_registry()`**

Reuse `CuratedRegistry.register`; add no audio.cpp-specific registry class and no runtime network lookup.

- [ ] **Step 6: Run registry, descriptor, and acquisition preflight tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_curated_registry.py Tests/Model_Artifacts/test_preflight.py Tests/TTS/test_audio_cpp_artifact_catalog.py -q
```

Expected: all pass, including the final 67-row `downloadable|local_only|explicitly_unsupported` projection with no open or duplicate outcome.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/TTS/audio_cpp_artifact_catalog.py tldw_chatbook/Model_Artifacts/curated_registry.py Tests/Model_Artifacts/test_curated_registry.py Tests/TTS/test_audio_cpp_artifact_catalog.py
git commit -m "feat(models): catalog reviewed audio cpp packages"
```

### Task 4: Persist optional managed artifact identity through the scanner

**Files:**
- Modify: `tldw_chatbook/TTS/audio_cpp_guided_config.py:300-358`
- Modify: `tldw_chatbook/TTS/audio_cpp_recipes.py:439-479`
- Modify: `tldw_chatbook/TTS/audio_cpp_package_scanner.py:522-897`
- Modify: `Tests/TTS/test_audio_cpp_guided_config.py`
- Modify: `Tests/TTS/test_audio_cpp_package_scanner.py`
- Modify: `Tests/TTS/test_audio_cpp_guided_launch.py`

- [ ] **Step 1: Write backward-compatibility and invariant RED tests**

Test legacy packages without managed identity, exact managed round-trip, malformed partial identity, recipe/variant disagreement, and scanner acceptance only when the returned managed root and recipe match the expected `ArtifactRef`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_guided_config.py Tests/TTS/test_audio_cpp_package_scanner.py -q`

Expected: managed identity type/argument is missing.

- [ ] **Step 3: Add a boundary-owned frozen identity**

```python
class AudioCppManagedArtifactIdentity(_FrozenModel):
    artifact_id: str
    revision: str
    variant: str

class AudioCppAcceptedPackage(_FrozenModel):
    ...
    managed_artifact: AudioCppManagedArtifactIdentity | None = None
```

Keep Model Artifact domain imports out of the persisted TTS config. Convert to/from `ArtifactRef` only at the handoff/runtime boundary.

- [ ] **Step 4: Extend `AudioCppPackageCandidate.accept()` minimally**

Add optional `managed_artifact` and verify its variant/recipe mapping before constructing the accepted package. Local picker calls remain unchanged and persist `None`.

- [ ] **Step 5: Add exact-root scanner admission**

Add an optional expected managed identity/root contract to the async boundary, not a second scanner. It must yield one exact candidate or fail closed; it may not accept a sibling candidate, changed canonical root, or recipe drift.

- [ ] **Step 6: Run focused config/scanner/launch tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_guided_config.py Tests/TTS/test_audio_cpp_package_scanner.py Tests/TTS/test_audio_cpp_guided_launch.py -q
```

Expected: all pass and legacy serialized config remains unchanged.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/TTS/audio_cpp_guided_config.py tldw_chatbook/TTS/audio_cpp_recipes.py tldw_chatbook/TTS/audio_cpp_package_scanner.py Tests/TTS/test_audio_cpp_guided_config.py Tests/TTS/test_audio_cpp_package_scanner.py Tests/TTS/test_audio_cpp_guided_launch.py
git commit -m "feat(tts): retain managed audio cpp artifact identity"
```

### Task 5: Add typed Settings↔Model Library handoffs and inactive installation

**Files:**
- Create: `tldw_chatbook/UI/Navigation/audio_cpp_model_handoff.py`
- Modify: `tldw_chatbook/Model_Artifacts/service.py`
- Modify: `tldw_chatbook/Model_Artifacts/__init__.py`
- Modify: `tldw_chatbook/UI/Navigation/pending_handoff_store.py:48-360`
- Modify: `tldw_chatbook/UI/Screens/model_curated_view.py:38-430`
- Modify: `tldw_chatbook/UI/Screens/llm_screen.py:140-1765`
- Modify: `Tests/State/test_pending_handoff_store.py`
- Create: `Tests/UI/test_audio_cpp_model_library_handoff.py`
- Modify: `Tests/UI/test_model_curated_view.py`
- Modify: `Tests/UI/test_llm_screen_lab_adoption.py`
- Modify: `Tests/Model_Artifacts/test_service.py`

- [ ] **Step 1: Write handoff and inactive-provision RED tests**

Cover detached request/result copies, invalid token/ref/root, one-time claim/acknowledge, audio-only catalog filtering, already-installed selection, `activate=False`, no active selector, no server launch/default mutation, terminal worker ownership across recompose/unmount, and an installed-but-inactive root leased/verified without readiness creation.

- [ ] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/UI/test_audio_cpp_model_library_handoff.py Tests/UI/test_model_curated_view.py Tests/UI/test_llm_screen_lab_adoption.py -q`

Expected: missing handoff types/channels and audio.cpp flow.

- [ ] **Step 3: Add frozen redacted handoff values**

```python
@dataclass(frozen=True, slots=True)
class AudioCppModelLibraryRequest:
    token: str
    draft_revision: int

@dataclass(frozen=True, slots=True)
class AudioCppModelLibraryResult:
    token: str
    draft_revision: int
    artifact_id: str
    revision: str
    variant: str
    canonical_root: str = field(repr=False)
```

Add two explicit `HandoffChannel` values. `_copy_value` reconstructs each field and rejects hostile subclasses/partial values; no private package contents enter repr/logs.

- [ ] **Step 4: Add the service-owned inactive-root lease seam**

Add `ModelArtifactService.acquire_installed_root(reference) -> LeasedArtifactHandle`. It acquires the exact root's shared artifact lease, verifies the installed ROOT descriptor/tree/inventory while continuously held, resolves the canonical managed root, and returns the existing root-only leased handle shape. It must not read or write readiness, activate a selector, traverse dependencies implicitly, or accept a DEPENDENCY descriptor. A genuinely absent exact root raises a new bounded `ArtifactNotInstalledError`; lease contention and invalid/corrupt state remain distinct failures. This is the sole handoff/Save path for `activate=False` installs; do not misuse `acquire()` (readiness required) or `acquire_dependencies()` (wrong role).

- [ ] **Step 5: Add one audio.cpp presentation mode to CuratedView**

Filter by `descriptor.consumer == "audio_cpp"`; render recipe/compatibility/companion facts derived from the joined catalog. Keep the existing shared install message and lifecycle owner. An installed audio.cpp row exposes **Use installed package** rather than Active.

- [ ] **Step 6: Branch only on consumer behavior in LLMScreen**

For `consumer == "audio_cpp"`, call the existing provision path with `activate=False`, skip Parakeet preference writes, produce the result handoff, and show **Installed — ready for review**. Existing Parakeet/remote behavior remains unchanged. Do not add per-family branches.

- [ ] **Step 7: Run handoff/install tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/State/test_pending_handoff_store.py Tests/UI/test_audio_cpp_model_library_handoff.py Tests/UI/test_model_curated_view.py Tests/UI/test_llm_screen_lab_adoption.py Tests/Model_Artifacts/test_provision_install.py Tests/Model_Artifacts/test_service.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/UI/Navigation/audio_cpp_model_handoff.py tldw_chatbook/Model_Artifacts/service.py tldw_chatbook/Model_Artifacts/__init__.py tldw_chatbook/UI/Navigation/pending_handoff_store.py tldw_chatbook/UI/Screens/model_curated_view.py tldw_chatbook/UI/Screens/llm_screen.py Tests/Model_Artifacts/test_service.py Tests/State/test_pending_handoff_store.py Tests/UI/test_audio_cpp_model_library_handoff.py Tests/UI/test_model_curated_view.py Tests/UI/test_llm_screen_lab_adoption.py
git commit -m "feat(tts): install audio cpp packages without activation"
```

### Task 6: Preserve and merge the complete Guided Settings draft

**Files:**
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py:444-4148`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py:1980-2340,13140-15070`
- Modify: `Tests/UI/test_audio_cpp_model_library_handoff.py`
- Modify: `Tests/UI/test_speech_tts_settings_ownership_closeout.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`

- [ ] **Step 1: Write the full-draft preservation RED test**

Mount real Settings, edit unrelated audio.cpp fields, another Speech/TTS field, and multiple Realtime fields, open Model Library, save/restore the Settings screen snapshot, return an exact result, and assert only one `guided_packages` entry/default candidate changed. Add a mutation while the Model Library is open for each draft family, stale token, stale draft revision, duplicate model ID, scan mismatch, leave/cancel, and late-result-after-unmount cases.

- [ ] **Step 2: Verify current behavior is RED**

Run: `../../.venv/bin/python -m pytest Tests/UI/test_audio_cpp_model_library_handoff.py Tests/UI/test_speech_tts_settings_ownership_closeout.py -q`

Expected: the complete Speech/TTS draft does not survive `SettingsScreen.save_state()` today.

- [ ] **Step 3: Define and persist one complete validated non-secret panel snapshot**

Add a frozen `SpeechTTSPanelDraftSnapshot` owned by the panel that contains deep copies of the live/original `GlobalSpeechTTSState`, live/original `_RealtimeSettingsDraft`, selected provider, and a monotonic `draft_revision`. Extend `save_state()`/`restore_state()` to round-trip that single value. Reject malformed/restored values through the existing pure `GlobalSpeechTTSState`, `AudioCppSettingsConfig`, and Realtime field validators. Do not serialize credentials, workers, package bytes, or handoff tokens into disk config.

Increment `draft_revision` on every real draft mutation, including Realtime widget collection, provider/default/profile/package edits, an explicit user Reset/Restore action, and result merge. Snapshot comparison may suppress increments when collection is value-identical. Merely rehydrating a newly mounted Settings screen from `SettingsScreen.save_state()` must restore the captured revision exactly and must not increment it; otherwise every valid return would become stale. The request token binds the revision captured after mounted values are collected, so any actual edit made while Model Library is open makes the returning result stale.

- [ ] **Step 4: Add the explicit Model Library action**

The panel collects mounted values, increments/stages its draft revision, asks the Settings shell to stage `AudioCppModelLibraryRequest`, and navigates to `llm` with the curated audio.cpp context. Add one exact navigation bypass so `flush_pending_work()` preserves this draft instead of forcing Save/Discard; every other navigation retains the existing confirmation behavior.

- [ ] **Step 5: Consume and merge the result**

On restored Settings mount, claim the result only when token and complete panel `draft_revision` match. Acquire the Task 5 inactive-root handle, require its canonical root to equal the detached result's root for the same exact `ArtifactRef`, call the existing scanner once while the lease is held with the expected recipe/artifact identity, require one exact candidate, and hold that lease through the in-memory merge. Leave the preserved Realtime live/original drafts byte-for-byte equivalent, recompose focus-safely, acknowledge the claim, and keep Save explicit. Stale results are acknowledged with **Installed, not added to this changed draft** and mutate nothing.

- [ ] **Step 6: Run mounted Settings/navigation tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_audio_cpp_model_library_handoff.py Tests/UI/test_speech_tts_settings_ownership_closeout.py Tests/UI/test_settings_configuration_hub.py Tests/TTS/test_audio_cpp_package_scanner.py -q
```

Expected: all pass; no save, provider acquisition, activation, or process launch occurs during handoff.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py tldw_chatbook/UI/Screens/settings_screen.py Tests/UI/test_audio_cpp_model_library_handoff.py Tests/UI/test_speech_tts_settings_ownership_closeout.py Tests/UI/test_settings_configuration_hub.py
git commit -m "feat(settings): review installed audio cpp packages in draft"
```

### Task 7: Hold managed artifact leases across Save, stage, and live runtime

**Files:**
- Modify: `tldw_chatbook/TTS/audio_cpp_guided_launch.py:88-744`
- Modify: `tldw_chatbook/TTS/audio_cpp_supervisor.py:1085-1525`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py:2950-3135`
- Modify: `tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py:2810-2965`
- Modify: `Tests/TTS/test_audio_cpp_guided_launch.py`
- Modify: `Tests/TTS/test_audio_cpp_supervisor.py`
- Modify: `Tests/TTS/test_stts_settings_reconfiguration.py`
- Modify: `Tests/UI/test_audio_cpp_model_library_handoff.py`

- [ ] **Step 1: Write lease-lifetime RED tests**

Prove: Settings Save validates under the Task 5 inactive-root shared lease; deliberate staging explicitly activates then acquires exact managed refs; local packages acquire none; a staged package blocks removal; a live child retains the handle until definitive stop; discarded/replaced stages release after cleanup; cancellation and cleanup failure retain ownership safely.

- [ ] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_guided_launch.py Tests/TTS/test_audio_cpp_supervisor.py Tests/TTS/test_stts_settings_reconfiguration.py -q`

Expected: managed artifact identity is currently ignored by runtime.

- [ ] **Step 3: Add one lease-owning launch bundle**

Extend `AudioCppGeneratedLaunchArtifact` (the existing staged/live cleanup owner) to retain exact `LeasedArtifactHandle` objects. For each managed package: reconstruct `ArtifactRef`, activate deliberately, acquire, derive the canonical root from the handle, then run the existing accepted-package scanner/identity validation. Local packages keep the current path flow.

- [ ] **Step 4: Make generated cleanup retryable without losing authority**

Change `AudioCppGeneratedLaunchArtifact.cleanup()` so a failed exact file/directory cleanup does not set `_cleaned` or close its descriptor/lease authority. Split the supervisor record's current one-shot `cleanup_called` truth into "runtime hooks settled" and "generated artifact settled": hooks run at most once, but exact artifact cleanup may retry. Do not clear `_generation` after child/output join if generated-artifact cleanup fails. Seal relaunch as unavailable, retain that exact generation as the cleanup owner, and retry through existing `stop()`/`close()`/`wait_closed()` shutdown ownership. Ordering is: child and drains conclusively join, runtime hooks settle once, generated config cleanup succeeds, then managed handles close, then the generation may be dropped. A retry or handle-close failure keeps all remaining ownership; a later successful retry releases it. Do not add a parallel lease registry or background retry loop.

- [ ] **Step 5: Fence Settings persistence**

Acquire inactive-root shared handles for the union of exact managed references in the original saved state and proposed draft before final save revalidation, and hold them through publication result/rollback. That before/after union prevents removing or replacing a package entry while its artifact is under removal authority. Apply the same lease around every in-memory Settings action that adds, removes, or replaces a `managed_artifact` identity (handoff merge, scan adoption, reset/removal); ordinary text edits that do not change artifact identity need only advance `draft_revision`. Save must not activate packages. Launch uses the deliberate activate+acquire path only when Start/Test/runtime staging begins. Do not merely validate then release before the write.

- [ ] **Step 6: Run runtime, cancellation, cleanup-retry, and store contention tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_guided_launch.py Tests/TTS/test_audio_cpp_supervisor.py Tests/TTS/test_stts_settings_reconfiguration.py Tests/Model_Artifacts/test_service.py -q
```

Expected: all pass; a competing exclusive removal times out while staged/live ownership exists, cleanup failure keeps removal blocked, and exact cleanup restoration followed by shutdown retry releases the handle.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/TTS/audio_cpp_guided_launch.py tldw_chatbook/TTS/audio_cpp_supervisor.py tldw_chatbook/TTS/TTS_Generation.py tldw_chatbook/Widgets/Settings_Widgets/speech_tts_settings_panel.py Tests/TTS/test_audio_cpp_guided_launch.py Tests/TTS/test_audio_cpp_supervisor.py Tests/TTS/test_stts_settings_reconfiguration.py Tests/UI/test_audio_cpp_model_library_handoff.py
git commit -m "feat(tts): lease managed audio cpp packages during use"
```

### Task 8: Add a non-recursive artifact removal authority

**Files:**
- Modify: `tldw_chatbook/Model_Artifacts/service.py:1119-1310`
- Modify: `tldw_chatbook/Model_Artifacts/__init__.py`
- Modify: `Tests/Model_Artifacts/test_service.py`
- Modify: `Tests/Model_Artifacts/test_operation_leases_process.py`

- [ ] **Step 1: Write removal-authority and contention-probe RED tests**

Cover established lifecycle→artifact acquisition order, cross-process contention, exact target pinning, `commit()` once, `close()` idempotence, cleanup failure retention, control-flow propagation, and the public `delete()` delegating without reacquiring the locks. Also cover a non-mutating `probe_removal_availability(reference)` that uses the same lifecycle→artifact order, releases both leases immediately, and returns only available/busy bounded truth without owner, PID, or lock-path details.

- [ ] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py Tests/Model_Artifacts/test_operation_leases_process.py -q`

Expected: no authority API exists.

- [ ] **Step 3: Extract current delete lock ownership into one capability**

```python
class ArtifactRemovalAuthority:
    def commit(self) -> None: ...
    def close(self) -> None: ...

class ModelArtifactService:
    def probe_removal_availability(self, reference: ArtifactRef) -> ArtifactRemovalAvailability:
        ...  # non-mutating, same lock order, generic available/busy result

    def acquire_removal_authority(self, reference: ArtifactRef) -> ArtifactRemovalAuthority:
        ...  # lifecycle EXCLUSIVE, then exact artifact EXCLUSIVE, once

    def delete(self, reference: ArtifactRef) -> None:
        with self.acquire_removal_authority(reference) as authority:
            authority.commit()
```

The authority calls `_delete_under_leases()` directly and never calls `delete()`. The probe does not call deletion or return an authority; it simply attempts and releases the same ordered leases with the non-blocking timeout. Keep errors/results bounded as available/busy, `ArtifactInUseError`, or `ArtifactStateError`; do not expose lock paths or promise OS process names. The final authority acquisition and preview revalidation remain authoritative because the probe can become stale immediately.

- [ ] **Step 4: Mutation-check recursive acquisition**

Temporarily make `commit()` call public `delete()` and verify the lock-order test fails/times out; restore the direct under-lock call and rerun green.

- [ ] **Step 5: Run service/process tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/Model_Artifacts/test_service.py Tests/Model_Artifacts/test_operation_leases.py Tests/Model_Artifacts/test_operation_leases_process.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Model_Artifacts/service.py tldw_chatbook/Model_Artifacts/__init__.py Tests/Model_Artifacts/test_service.py Tests/Model_Artifacts/test_operation_leases_process.py
git commit -m "feat(models): add exact artifact removal authority"
```

### Task 9: Project and revalidate every audio.cpp removal dependency

**Files:**
- Create: `tldw_chatbook/TTS/audio_cpp_artifact_dependencies.py`
- Create: `Tests/TTS/test_audio_cpp_artifact_dependencies.py`
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/voice_bundle_service.py`
- Modify: `Tests/TTS/test_profile_service.py`
- Modify: `Tests/TTS/test_voice_bundle_service.py`
- Modify: `tldw_chatbook/UI/Screens/model_installed_view.py:1020-1210`
- Modify: `Tests/UI/test_model_installed_view.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`

- [ ] **Step 1: Write the dependency matrix RED tests**

Cover zero/one/many saved packages, restored unsaved draft packages, global defaults, profiles, clone references, assignments, staged/live generations, exact artifact leases, unrelated variants, unavailable profile repository, and canary values in errors/logs. Assert clone bytes/transcripts, complete paths, raw settings, checksums, and collaborator exceptions never enter the public preview/error graph. Add races for profile create/update/delete, clone import/reference mutation, assignment set/remove, and Settings merge/save while removal is waiting or authoritative.

- [ ] **Step 2: Run tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_artifact_dependencies.py Tests/UI/test_model_installed_view.py -q`

Expected: no complete audio.cpp dependency preview exists.

- [ ] **Step 3: Implement one immutable bounded preview**

```python
@dataclass(frozen=True, slots=True)
class AudioCppArtifactRemovalPreview:
    reference: ArtifactRef
    fingerprint: str
    settings_labels: tuple[str, ...]
    profile_labels: tuple[str, ...]
    assignment_count: int
    clone_reference_count: int
    staged_or_live: bool
    generic_lease_blocked: bool
```

Derive the fingerprint by length-framed incremental hashing of canonical bounded consumer fields; exclude the volatile advisory `generic_lease_blocked` bit from that fingerprint. Read collaborators through existing app-owned Settings/TTS/profile/runtime seams; no new repository or artifact-service singleton. Obtain `generic_lease_blocked` from `ModelArtifactService.probe_removal_availability()` only for the pre-confirmation projection and treat busy as a hard blocker. Once `ArtifactRemovalAuthority` is acquired, recompute consumer evidence without calling the probe (the authority itself proves lease exclusion) and compare the stable consumer fingerprint. Never let the authority self-contend through its own probe.

- [ ] **Step 4: Serialize every artifact-consumer mutation with the same shared root lease**

Add one narrow `AudioCppArtifactLeaseCoordinator` beside the dependency projection. Given a validated recipe/model requirement, it first resolves the exact joined catalog `ArtifactRef` when that requirement is downloadable, then prefers/equality-checks any matching persisted `managed_artifact` identity from the injected immutable saved Settings snapshot. It calls `ModelArtifactService.acquire_installed_root()` for the deduplicated exact references. Catch only `ArtifactNotInstalledError` to preserve the existing inactive/missing-dependency profile/import path; contention, corruption, and identity disagreement fail boundedly before mutation. Local-only and explicitly unsupported requirements produce no managed-store lease. Inject this existing app-owned coordinator into `TTSProfileService` and `TTSVoiceBundleService`, rather than importing Settings or an artifact singleton into the repository.

Hold the exact shared root lease through each repository commit that can change removal evidence: profile create/update/delete, reference-bearing clone create/import, assignment set/remove, and bundle import. For deletion and assignment changes, include the loaded pre-mutation profile requirement; for create/update/import, include validated post-mutation requirements as well. Settings identity-changing merge/reset/save actions already use the same root seam from Tasks 6–7. The coordinator must deduplicate exact references, acquire in canonical sorted order, and release only after commit/rollback settles. Tests must prove an acquired removal authority blocks each mutation before repository or in-memory draft writes, while a mutation already holding the shared root lease completes before removal acquires authority and then causes preview fingerprint drift. Repository internals remain artifact-store agnostic.

- [ ] **Step 5: Add preview/confirm to InstalledView**

Audio.cpp managed roots use the detailed dialog. Hard blockers disable confirmation with exact recovery copy. Durable impacts require explicit **Remove package; keep consumers unavailable** acknowledgement; no profile/default/assignment/reference mutation occurs.

- [ ] **Step 6: Commit under one retained worker**

Acquire `ArtifactRemovalAuthority`, recompute the preview under authority, compare fingerprint and explicit resolutions, reject drift, call `authority.commit()`, reread inventory, then close. Retain/join worker and authority on cancellation/cleanup failure. Report unknown lease owners only as **Another operation is using this package**.

- [ ] **Step 7: Add concurrency and mutation tests**

Test each profile/reference/assignment/bundle/settings mutation between preview and confirm, a new stage before authority acquisition, every mutation attempt while authority is held, cancellation before/after commit, and cleanup retry. Assert no mutation crosses the authority boundary, preview drift forces a fresh review, and consumers remain byte/value identical after successful removal.

- [ ] **Step 8: Run removal/store/profile tests**

Run:

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_audio_cpp_artifact_dependencies.py Tests/UI/test_model_installed_view.py Tests/TTS/test_profile_service.py Tests/TTS/test_voice_bundle_service.py Tests/TTS/test_tts_app_ownership.py Tests/Model_Artifacts/test_service.py -q
```

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add tldw_chatbook/TTS/audio_cpp_artifact_dependencies.py tldw_chatbook/TTS/profile_service.py tldw_chatbook/TTS/voice_bundle_service.py tldw_chatbook/UI/Screens/model_installed_view.py tldw_chatbook/app.py Tests/TTS/test_audio_cpp_artifact_dependencies.py Tests/TTS/test_profile_service.py Tests/TTS/test_voice_bundle_service.py Tests/UI/test_model_installed_view.py Tests/TTS/test_tts_app_ownership.py
git commit -m "feat(tts): review dependencies before package removal"
```

### Task 10: Harden Model Library truth, accessibility, and recovery

**Files:**
- Modify: `tldw_chatbook/UI/Screens/model_curated_view.py`
- Modify: `tldw_chatbook/UI/Screens/model_installed_view.py`
- Modify: `tldw_chatbook/UI/LLM_Management_Window.py`
- Modify: `tldw_chatbook/css/_lab.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_model_curated_view.py`
- Modify: `Tests/UI/test_model_installed_view.py`
- Modify: `Tests/UI/test_llm_screen_lab_adoption.py`

- [ ] **Step 1: Write mounted UI RED tests first**

At 80x24 and the narrow real pane, cover keyboard order, long family/model names, many companions, expandable relative-file list, offline/checksum/space/stale-handoff/removal-blocked states, disabled reason tooltips, focus restoration, and separation of Available/Integrity/Recipe/Compatibility/Configured/Running.

- [ ] **Step 2: Run focused Pilot tests and verify RED**

Run: `../../.venv/bin/python -m pytest Tests/UI/test_model_curated_view.py Tests/UI/test_model_installed_view.py Tests/UI/test_llm_screen_lab_adoption.py -q`

Expected: new controls/states are absent or clipped.

- [ ] **Step 3: Implement the smallest truthful projection**

Keep state derivation frozen and outside `compose()`. Render no Active badge for audio.cpp installs. Always state **Model package only — audiocpp_server is not included**. Use one disclosure for companions and one vertically stackable action area; no new global keybinding.

- [ ] **Step 4: Apply scoped CSS and regenerate the bundle**

Use existing tokens, auto-height/vertical scroll, visible focus, and app-tier disabled styles. Measure disabled label contrast ≥3:1 across the five shipped themes using the existing compositor test pattern.

- [ ] **Step 5: Run accessibility and CSS gates**

Run:

```bash
../../.venv/bin/python -m pytest Tests/UI/test_model_curated_view.py Tests/UI/test_model_installed_view.py Tests/UI/test_llm_screen_lab_adoption.py -q
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
```

Expected: all pass, every action is contained/reachable, and generated CSS is byte-current.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/model_curated_view.py tldw_chatbook/UI/Screens/model_installed_view.py tldw_chatbook/UI/LLM_Management_Window.py tldw_chatbook/css/_lab.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_model_curated_view.py Tests/UI/test_model_installed_view.py Tests/UI/test_llm_screen_lab_adoption.py
git commit -m "feat(ui): expose truthful audio cpp package lifecycle"
```

### Task 11: Verify the end-to-end contract and close TASK-13207

**Files:**
- Create: `Docs/superpowers/qa/audio-cpp-model-library-2026-08-13/live-uat.md`
- Modify: relevant speech/model-library user docs discovered by `rg -n "audio\.cpp|Model Library" Docs README.md`
- Modify: `backlog/tasks/task-13207 - Integrate-guided-audio.cpp-packages-with-Model-Library.md`
- Modify only if evidenced: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run the focused feature matrix**

```bash
../../.venv/bin/python -m pytest \
  Tests/Model_Artifacts/test_curated_registry.py \
  Tests/Model_Artifacts/test_preflight.py \
  Tests/Model_Artifacts/test_provision_install.py \
  Tests/Model_Artifacts/test_service.py \
  Tests/TTS/test_audio_cpp_artifact_catalog.py \
  Tests/TTS/test_audio_cpp_artifact_dependencies.py \
  Tests/TTS/test_audio_cpp_recipes.py \
  Tests/TTS/test_audio_cpp_guided_config.py \
  Tests/TTS/test_audio_cpp_package_scanner.py \
  Tests/TTS/test_audio_cpp_guided_launch.py \
  Tests/TTS/test_audio_cpp_supervisor.py \
  Tests/TTS/test_stts_settings_reconfiguration.py \
  Tests/UI/test_audio_cpp_model_library_handoff.py \
  Tests/UI/test_model_curated_view.py \
  Tests/UI/test_model_installed_view.py \
  Tests/UI/test_llm_screen_lab_adoption.py \
  Tests/TTS/test_tts_app_ownership.py -q
```

Expected: all pass; normal tests use only small deterministic fixtures.

- [ ] **Step 2: Run static, formatting, CSS, and diff gates**

```bash
git diff --name-only -z origin/dev...HEAD -- '*.py' | xargs -0 ../../.venv/bin/python -m ruff check
git diff --name-only -z origin/dev...HEAD -- '*.py' | xargs -0 ../../.venv/bin/python -m ruff format --check
../../.venv/bin/python -m mypy \
  tldw_chatbook/TTS/audio_cpp_artifact_catalog.py \
  tldw_chatbook/TTS/audio_cpp_artifact_dependencies.py \
  tldw_chatbook/TTS/audio_cpp_guided_config.py \
  tldw_chatbook/TTS/audio_cpp_guided_launch.py \
  tldw_chatbook/Model_Artifacts/service.py
../../.venv/bin/python tldw_chatbook/css/check_bundle_sync.py
git diff --check origin/dev...HEAD
```

Expected: all clean. The NUL-safe changed-file list avoids linting the entire historical tree and mislabeling unrelated baseline issues.

- [ ] **Step 3: Run the opt-in clean-root UAT**

With isolated HOME/XDG/config/data/model/runtime roots, install one small exact pinned package through the real Model Library, return it to an unrelated dirty Guided draft, Save, deliberately Test/Start with a pre-provisioned compatible `audiocpp_server`, verify generation, then exercise blocked and acknowledged removal. Record exact safe commit/artifact identities and structural output facts; never record private paths, transcript, audio bytes, or secrets. If an exact small official package or server is unavailable, mark that portion partial rather than substituting a fake and claiming UAT.

- [ ] **Step 4: Perform review and mutation checks**

Request correctness/security review, then independently kill at least these mutants: `activate=False` removed; stale draft generation accepted; runtime handle released before child stop; removal authority calls public `delete()`; removal fingerprint not rechecked. Restore and rerun each named test green.

- [ ] **Step 5: Update docs, task notes, and acceptance criteria truthfully**

Document install-vs-activate truth, exact pinned commit, local-only variants, runtime lease ownership, and removal recovery. Add a lesson only if implementation produced a reusable evidenced trap. Add concise Implementation Notes, record ADR-050/051, check only criteria with fresh evidence, then run task hygiene.

- [ ] **Step 6: Run final verification after documentation changes**

Run the focused matrix, static gates, CSS sync, `git diff --check`, and Backlog task hygiene again. Do not mark Done if UAT or any acceptance criterion remains partial.

- [ ] **Step 7: Commit closeout**

```bash
git add Docs backlog tldw_chatbook Tests scripts
git commit -m "docs(tts): close guided model library integration"
```

---

## Explicitly skipped complexity

- No runtime Hugging Face browser or moving-`main` resolver.
- No family-specific installer, scanner, runtime, or removal subclass.
- No second curated registry, artifact store, lease registry, or download service.
- No companion-file symlink/hardlink composition or multi-root package view.
- No automatic Settings Save, global/default selection, profile rewrite, assignment rewrite, reference deletion, server install, server launch, or force-delete.
- No speculative Safetensors artifact-store format until an admitted downloadable manifest entry requires it.
