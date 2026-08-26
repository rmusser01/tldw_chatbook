# TASK-13206 Clone Voice Bundle Portability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan sequentially. Apply superpowers:test-driven-development to every behavior change, use impeccable and textual-tui for mounted interface work, and stop at review checkpoints.

**Goal:** Let users safely export one saved audio.cpp clone voice as an explicitly warned portable bundle, import it as an exact/reviewed/inactive profile without hidden mutation, and use it only with the exact recipe/model dependency.

**Architecture:** Keep ordinary profile JSON sanitized and add one strict four-entry bundle codec below an app-owned `TTSVoiceBundlePortabilityService`. Schema v4 persists immutable recipe provenance beside private clone references; the repository remains the only migration/profile/reference mutation owner, while the bundle service owns hostile-source inspection sessions, deterministic export, and retained cleanup. UI receives only opaque session handles and bounded review facts, and runtime admission independently rechecks exact saved/applied recipe and process-generation evidence before private materialization or synthesis.

**Tech Stack:** Python 3.11+, `asyncio`, Textual 8.x, immutable dataclasses, SQLite, standard-library ZIP primitives plus explicit layout validation, POSIX descriptor/no-follow containment, pytest/Hypothesis, Ruff, mypy.

**Normative design:** `Docs/superpowers/specs/2026-08-11-audio-cpp-clone-voice-bundle-portability-design.md`

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/028-character-tts-generation-profile-ownership.md`, `backlog/decisions/029-local-private-data-boundary.md`, `backlog/decisions/051-private-tts-clone-reference-assets.md`

**Reason:** ADR-051 already owns clone-reference storage, migration, runtime admission, privacy, and portability and has been amended for this protocol. ADR-028 keeps assignment explicit and ADR-029 supplies the private local-data boundary.

**Deliberate exclusions:** No character-card embedding, automatic assignment/default change, model installation, recipe substitution, External/user-JSON clone transport, encrypted/signed bundles, standalone sanitized-v2 import, general archive framework, Windows support without verified ACL parity, or invented provenance for migrated references.

---

### Preparation checkpoint: Record the exact legacy mypy baseline

Before Task 1 changes production Python, record normalized current diagnostics for the four plan-owned legacy modules that are not clean on `origin/dev`. The normalized comparison ignores line-number movement but not a new error code/message.

```bash
set +e
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/adapters/audio_cpp.py tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/UI/STTS_Window.py tldw_chatbook/app.py 2>&1 \
  | sed -E 's/:[0-9]+(:[0-9]+)?:/:LINE:/' \
  | grep -vE '^(Found [0-9]+ errors|Success:)' \
  | sort -u > /tmp/task13206-mypy-legacy-baseline.txt
set -e
```

Verify the file is nonempty and contains only diagnostics already present on the exact starting commit. This is the during-implementation comparison only; the Final review checkpoint regenerates it from the exact post-rebase merge-base so upstream changes cannot be hidden or misattributed. Do not treat either baseline as permission to add another error.

### Task 1: Add strict ordinary-wire v2 and recipe-provenance domain types

**Files:**
- Modify: `tldw_chatbook/TTS/profile_portability.py`
- Modify: `tldw_chatbook/TTS/profile_reference_types.py`
- Modify: `tldw_chatbook/TTS/profile_types.py`
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `Tests/TTS/test_profile_portability.py`
- Modify: `Tests/TTS/test_profile_reference_types.py`
- Modify: `Tests/TTS/test_profile_repository.py`
- Modify: `Tests/TTS/test_profile_service.py`

- [ ] **Step 1: Write failing wire/domain tests**

Prove reference-free exports remain byte-for-byte wire v1, reference-bearing exports produce only exact wire v2 plus `"reference":{"status":"omitted"}`, and decoding returns bounded `reference_omitted` with no profile/mutation. Reject missing/extra/malformed v2 fields. Add immutable `TTSCloneRecipeRequirement(recipe_id, recipe_revision, model_id)` tests for exact grammar, positive 32-bit revision, model bound, redacted representation, all-null legacy provenance, half-present rejection, and model/profile incoherence.

```python
def test_reference_bearing_ordinary_export_is_exact_sanitized_v2() -> None:
    payload = portable_profile_payload(portable_with_reference)
    assert payload == {
        "schema_version": 2,
        **expected_v1_selection,
        "reference": {"status": "omitted"},
    }
    assert "sha256" not in json.dumps(payload)

def test_v2_omission_decode_is_skip_without_profile() -> None:
    result = decode_portable_profile(exact_v2_payload)
    assert result.status is PortableProfileDecodeStatus.REFERENCE_OMITTED
    assert result.profile is None
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_portability.py \
  Tests/TTS/test_profile_reference_types.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_service.py -q

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/profile_portability.py \
  tldw_chatbook/TTS/profile_reference_types.py \
  tldw_chatbook/TTS/profile_types.py \
  tldw_chatbook/TTS/profile_service.py
```

Expected: failures for absent v2 omission handling and recipe requirement.

- [ ] **Step 3: Implement the exact domain extension**

Keep `PortableTTSProfile` and v1 serialization unchanged for reference-free profiles. Give the serializer explicit reference-presence rather than assembling JSON in UI. Decode only exact v2 and return a typed skip result; do not make it importable. Add the frozen recipe requirement to `TTSCloneReferenceSummary` and `TTSCloneReference`; canonicalizers reconstruct it and reject half-present/mismatched values. Do not expose paths, bytes, transcript, digest, assignment, runtime, or generated configuration.

- [ ] **Step 4: Mutation-check compatibility/privacy**

Temporarily add digest/recipe to v2 and confirm the allowlist test fails. Route reference-free export through v2 and confirm byte compatibility fails; restore both.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/profile_portability.py \
  tldw_chatbook/TTS/profile_reference_types.py \
  tldw_chatbook/TTS/profile_types.py tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_profile_portability.py Tests/TTS/test_profile_reference_types.py \
  Tests/TTS/test_profile_repository.py Tests/TTS/test_profile_service.py
git commit -m "feat(tts): define clone bundle provenance domain"
```

### Task 2: Migrate private profile storage to schema v4 with recoverable publication

**Files:**
- Create: `tldw_chatbook/TTS/migrations/v3_to_v4.py`
- Modify: `tldw_chatbook/TTS/profile_schema.py`
- Modify: `tldw_chatbook/TTS/profile_reference_storage.py`
- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify: `Tests/TTS/test_profile_schema.py`
- Modify: `Tests/TTS/test_profile_reference_storage.py`
- Modify: `Tests/TTS/test_profile_repository_lifecycle.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`

- [ ] **Step 1: Write failing schema/publication tests**

Work in four red-green microcycles, committing no production behavior until its focused test is red: (A) exact nullable recipe columns, both-null-or-both-valid checks, and v3 row/domain equivalence; (B) v3→v4 plus v0/v1/v2 multi-hop/private candidates, null migrated provenance, restore candidates, and newer-schema refusal; (C) exact pre-v3/pre-v4 backup contents, aliases, pre-publication cancellation, replacement, active reopen, and failures during each retained-backup publication, proving active plus every prior backup restore together; and (D) interrupted journal recovery and total-storage-failure bounded-unavailable state with recovery artifacts retained. After each microcycle, run only its new named tests, implement the minimum behavior, rerun green, then run the complete Step-2 command. Use isolated `tmp_path` stores only.

```python
def test_v3_to_v4_preserves_reference_and_leaves_recipe_unknown(tmp_path: Path) -> None:
    migrated = open_profile_store(seed_v3_reference_store(tmp_path))
    row = migrated.execute(RECIPE_REFERENCE_SELECT).fetchone()
    assert row["recipe_id"] is None
    assert row["recipe_revision"] is None

@pytest.mark.asyncio
async def test_post_replace_failure_restores_active_and_backups(...):
    ...
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_schema.py Tests/TTS/test_profile_reference_storage.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/DB/test_private_sqlite_inventory.py -q
```

- [ ] **Step 3: Implement private-candidate migration/publication**

Advance schema to v4; add recipe columns only on a private candidate. Extend full row/domain validation and reopen the candidate before publication. Under exclusive ownership, prepare boundary snapshots by SQLite online backup and retain active/prior backups under rollback identity through active replacement, directory fsync, reopen/validation, and durable publication/fsync of every prepared pre-v3/pre-v4 backup. Release rollback identities only after all those publications succeed. Any failure after active reopen or during any backup publication restores/fsyncs the active store and every prior retained backup before returning. Record old/new private identities in one bounded owner-private journal before the point of no return. Suppress cancellation only across completion-or-restoration and re-deliver it afterward. A current-v4 startup keeps one shared lease continuously across descriptor/no-follow journal-absence proof, exact version read, and live open; any journal, legacy, newer, or uncertain observation escalates to exclusive ownership before recovery or migration. Cleanup atomically quarantines exact artifacts in the finite tombstone namespace without truncating potentially aliased inodes; nonzero tombstones remain private, non-authoritative, and non-reusable. Register all new private artifacts/operations.

- [ ] **Step 4: Mutation-check publication ordering**

Release a prior backup before active reopen and confirm the post-replace regression fails. Allow cancellation inside replacement and confirm authoritative-store recovery fails; restore both.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/migrations/v3_to_v4.py \
  tldw_chatbook/TTS/profile_schema.py tldw_chatbook/TTS/profile_reference_storage.py \
  tldw_chatbook/TTS/profile_repository.py tldw_chatbook/DB/private_sqlite.py \
  backlog/docs/sqlite-private-owner-inventory.md Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_reference_storage.py Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/DB/test_private_sqlite_inventory.py
git commit -m "feat(tts): migrate clone provenance storage to v4"
```

### Review checkpoint A

- [ ] Check Tasks 1–2 against AC #1, #8, #9 and ADR-051.
- [ ] Confirm v1 is byte-compatible and migration never infers recipe from Settings.
- [ ] Confirm pre-publication cancellation preserves active bytes and post-publication always completes/restores.
- [ ] Run `git diff --check` and both focused commands.

### Task 3: Add repository-owned provenance mutations and atomic import decisions

**Files:**
- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/profile_reference_storage.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `Tests/TTS/test_profile_reference_repository.py`
- Modify: `Tests/TTS/test_profile_repository.py`
- Modify: `Tests/TTS/test_profile_repository_lifecycle.py`
- Modify: `Tests/TTS/test_profile_service.py`

- [ ] **Step 1: Write failing mutation/edit tests**

Use three red-green microcycles: (A) provenance on new/replaced v4 references, migrated-null compatibility, model coherence, and display-name-only edits; (B) one pure saved/applied Guided configuration + recipe-registry snapshot API with exact/missing/mismatch/pending outcomes and assertions of zero adapter/lease/supervisor/health/network/Settings work; and (C) the complete Create/Reuse/Copy × exact dependency/missing dependency × inactive-consent cross-product, including UUID/name/private-reference conflicts, explicit copy, stale destination, concurrent mutation, cancellation, and failure between profile/recipe/reference writes. Assert Reuse is permitted only for complete public/private equality with a usable exact provenance requirement; migrated null-provenance references never qualify as exact reuse. Run each new named test red, add only its minimum implementation, rerun green, then run Step 2. The repository command returns exact reuse/create atomically and never assigns/defaults.

```python
@pytest.mark.asyncio
async def test_import_rechecks_collision_and_creates_atomically(...):
    result = await repository.commit_bundle_import(reviewed_command)
    assert result.kind == "created"
    assert await repository.assignment_count(result.profile.profile_id) == 0
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_repository.py Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py Tests/TTS/test_profile_service.py -q
```

- [ ] **Step 3: Implement one exact repository command**

Extend reference create/set APIs with exact recipe evidence for new writes. Add a pure `TTSService` snapshot returning bounded saved/applied guided model/recipe/pending facts without touching an adapter. Add a private repository command containing reviewed UUID/name/generation fields, requirement, canonical reference, expected generations/revisions, and exact-reuse/copy decision. Canonicalize before queueing; re-read collisions and private equality inside one worker transaction. Return `reused`, `created`, or bounded `stale_inspection` with refreshed safe repository collision/profile facts only—the repository has no configuration authority. Task 5 separately recomputes pure dependency facts and combines them into a replacement single-use review session, so changed visible facts require confirmation again. Preserve shielded repository cancellation/join behavior.

- [ ] **Step 4: Mutation-check atomicity/edit fences**

Move collision lookup outside `BEGIN IMMEDIATE` and confirm the race test fails. Allow a model edit and confirm the generation-field invariant fails; restore both.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/profile_repository.py tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/profile_reference_storage.py tldw_chatbook/TTS/TTS_Generation.py \
  Tests/TTS/test_profile_reference_repository.py Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py Tests/TTS/test_profile_service.py
git commit -m "feat(tts): commit clone bundle profiles atomically"
```

### Task 4: Build the deterministic bundle codec and hostile archive validator

**Files:**
- Create: `tldw_chatbook/TTS/voice_bundle_codec.py`
- Create: `Tests/TTS/test_voice_bundle_codec.py`
- Create: `Tests/TTS/test_voice_bundle_hostile_archives.py`
- Modify: `tldw_chatbook/TTS/__init__.py`

- [ ] **Step 1: Write failing codec/hostile tests**

Use four red-green microcycles: (A) deterministic exact writer bytes/order/metadata/JSON/LF/transcript/WAV; (B) positive STORE/DEFLATE metadata matrix—flags `0` or bit 11, exact needed/created versions, Unix regular modes without special bits, and DOS non-directory/non-volume attributes; (C) structural rejection for encryption/other flags, invalid combinations, comments/extras/descriptors/ZIP64/multipart, duplicate/normalized/path names, special files, central/local disagreement, overlap/prefix/trailing bytes; and (D) CRC/size/checksum/count/ratio/streaming limits, malformed/deep/nonfinite JSON, invalid text, dependency disagreement, canonical WAV, privacy graphs, and bounded Hypothesis generation. For each microcycle run the smallest new node red, implement minimally, rerun green, then run Step 2.

```python
def test_writer_emits_exact_deterministic_four_member_bundle() -> None:
    assert encode_clone_voice_bundle(canonical_input) == encode_clone_voice_bundle(canonical_input)
    assert member_names(encode_clone_voice_bundle(canonical_input)) == EXPECTED_MEMBER_ORDER

@pytest.mark.parametrize("mutator", HOSTILE_ARCHIVE_MUTATORS)
def test_hostile_archive_is_rejected_without_extraction(mutator) -> None:
    with pytest.raises(TTSVoiceBundleError, match="bundle_invalid"):
        inspect_clone_voice_bundle(mutator(valid_bundle))
```

- [ ] **Step 2: Run and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_voice_bundle_codec.py Tests/TTS/test_voice_bundle_hostile_archives.py -q
```

- [ ] **Step 3: Implement fixed-format streaming codec**

Define frozen safe bundle/profile/manifest values and central limits. Validate EOCD, central directory, and local headers before using `zipfile` only as a bounded decompressor; never call extraction APIs or trust paths. Stream exactly four members through counters to caller-chosen sinks, then validate strict JSON, transcript, and canonical WAV. Writer uses exact ZIP_STORED metadata. Errors expose bounded codes and sever private values from representations/exception graphs.

- [ ] **Step 4: Mutation-check layout/counters**

Skip local-header name comparison and confirm disagreement fails; trust declared size without stream counter and confirm quota mutation fails; restore.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/voice_bundle_codec.py tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_voice_bundle_codec.py Tests/TTS/test_voice_bundle_hostile_archives.py
git commit -m "feat(tts): validate portable clone voice bundles"
```

### Task 5: Add the app-owned retained portability service

**Files:**
- Create: `tldw_chatbook/TTS/voice_bundle_service.py`
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/TTS/test_voice_bundle_service.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`

- [ ] **Step 1: Write failing session/containment/publication/lifecycle tests**

Use five red-green microcycles: (A) POSIX owner/type/mode/no-follow/identity, safe same-user narrowing, foreign/substituted refusal, cleanup, and separate source mutation/replacement tests at every authority boundary—initial open, copy in progress, post-copy inspection, modal review interval, commit reopen, commit copy, and final pre-repository fingerprint comparison; (B) cap four, expiry, single-use/replay/foreign handle plus refusal by `copy.copy`, `copy.deepcopy`, and pickle round-trip; (C) no retained extraction while modal open, cancellation/unmount, retained worker settlement, and pure dependency inspection with zero adapter/lease/supervisor/launch/health/HTTP/Settings work; (D) acknowledgement, deterministic export, destination substitution, atomic no-replace/0600/fsync/no overwrite; and (E) app construction and close/join before repository. Run each named node red, implement minimally, rerun green, then Step 2.

```python
@pytest.mark.asyncio
async def test_session_is_private_bounded_and_single_use(...):
    review = await service.inspect(source)
    assert "source" not in repr(review.handle)
    await service.commit(review.handle, reviewed_choice)
    with pytest.raises(TTSVoiceBundleError, match="stale_inspection"):
        await service.commit(review.handle, reviewed_choice)

@pytest.mark.asyncio
async def test_stale_commit_returns_new_review_session_with_refreshed_safe_facts(...):
    replacement = await service.commit(review.handle, reviewed_choice)
    assert replacement.status == "stale_inspection"
    assert replacement.review.handle != review.handle
    assert review.handle.is_redacted
```

- [ ] **Step 2: Run and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_voice_bundle_service.py Tests/TTS/test_tts_app_ownership.py -q
```

- [ ] **Step 3: Implement retained ownership**

Create `TTSVoiceBundlePortabilityService` with service identity, sealed admission, retained task set, an opaque token that cannot be copied/deep-copied/pickled, expiry, and `close()`/`wait_closed()`. A shielded worker descriptor-pins/copies/validates into an app-owned 0700 operation root, deletes first-pass extracted files before returning safe facts, and privately retains only source identity/fingerprint/evidence. Cancellation awaits settlement and identity cleanup. Commit consumes once, fully revalidates, obtains the Task-3 pure dependency snapshot, and submits Task 3's repository command. When the repository reports stale repository facts, the service recomputes dependency facts, invalidates the old handle, and returns a new bounded inspection session with refreshed safe combined facts; commit never proceeds until the user confirms that new review. Export publishes a 0600 temporary sibling with atomic no-replace. App construction and exact close ordering live in `app.py`, and shut this owner before the profile repository.

Validate the pure dependency snapshot as the producer emits it: exact requires
only the exact applied requirement and may have missing/drifted queued settings,
missing/mismatch may have either pending value, and pending
requires queued configuration plus the exact saved requirement. Keep the
pending flag coherent with saved/applied generations without treating it as a
state alias, and prove inspection/commit perform zero provider work.

Treat any nonempty verified owner-private operation root on first use or
restart as `cleanup_failed`, preserve every occupant, and do not infer cleanup
authority from a recognized filename. Recovery requires Chatbook to be exited
before the user manually inspects the app-owned portability root and removes
only confirmed residue; no runtime path crosses the public error boundary.

The successful atomic no-replace publication is export's non-cancellable point
of no return. Pre-publication cancellation retains the randomized `0600`
temporary sibling and propagates with no final; pathname cleanup in a
user-selected parent is not exact-safe. Its bounded recovery identifies that a
hidden randomized sibling may remain and requires user-verified manual removal.
Post-publication cancellation is deferred
while exact final identity/mode/content and parent fsync converge, then returns
successful publication. Publish by atomic no-clobber rename so the temporary
sibling is consumed at the PONR, and never unlink any export pathname;
POSIX cannot make a `stat`-then-`unlink` sequence substitution-safe.

- [ ] **Step 4: Mutation-check lifecycle/identity**

Use unretained `to_thread` and confirm cancellation leaks ownership; use path cleanup after substitution and confirm cleanup test fails; remove the post-PONR final-unlink prohibition and confirm the substitution/no-unlink regression fails; restore.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/voice_bundle_service.py tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/profile_repository.py tldw_chatbook/app.py \
  Tests/TTS/test_voice_bundle_service.py Tests/TTS/test_tts_app_ownership.py
git commit -m "feat(tts): own clone bundle inspection lifecycle"
```

### Review checkpoint B

- [ ] Check Tasks 3–5 against AC #2–#6, #10–#11, ADR-029/051.
- [ ] Confirm no general extraction, digest oracle, path, transcript, audio, member name, or collaborator error crosses the boundary.
- [ ] Confirm sessions retain no extraction while reviewed and every cancel/close joins work.
- [ ] Run `git diff --check` and Tasks 3–5 commands.

### Task 6: Enforce exact dependency truth and runtime clone admission

**Files:**
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/request_admission.py`
- Modify: `tldw_chatbook/TTS/adapter_types.py`
- Modify: `tldw_chatbook/TTS/TTS_Generation.py`
- Modify: `tldw_chatbook/TTS/adapters/audio_cpp.py`
- Modify: `Tests/TTS/test_profile_service.py`
- Modify: `Tests/TTS/test_tts_request_admission.py`
- Modify: `Tests/TTS/test_audio_cpp_contract.py`
- Modify: `Tests/TTS/test_audio_cpp_managed_integration.py`
- Modify: `Tests/TTS/test_stts_audio_cpp_generation.py`

- [ ] **Step 1: Write failing availability/admission tests**

Use four red-green microcycles consuming Task 3's pure snapshot: (A) availability projection for exact/missing/mismatch/pending/unknown recipe plus migrated-null advisory, including the exact blocker precedence matrix `damaged/profile-invalid` → `provider/configuration unavailable` → `recipe_missing|recipe_mismatch|recipe_pending_apply` → none, while the provenance advisory remains visible beside whichever primary blocker wins; (B) pre-provider comparison with zero registry/adapter work; (C) side-effect-free adapter preflight before readiness; and (D) post-ready process-generation fence before materialization/speech HTTP, no fallback, and staged-save isolation. Run each new named node red, implement minimally, rerun green, then Step 2.

```python
@pytest.mark.asyncio
async def test_recipe_mismatch_blocks_before_provider_lease(...):
    with pytest.raises(TTSOperationError, match="dependency_missing"):
        await service.synthesize_effective(...)
    assert registry.acquire_calls == 0
    assert transport.requests == []
```

- [ ] **Step 2: Run and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_service.py Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_audio_cpp_contract.py Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_stts_audio_cpp_generation.py -q
```

- [ ] **Step 3: Implement separate local and admitted gates**

Consume the pure saved/applied guided model/recipe/pending snapshot created in Task 3. Extend availability with blocking dependency reason/action plus independent provenance advisory. Before provider lease compare persisted requirement to applied snapshot. Under exact lease require side-effect-free audio.cpp configured recipe/model preflight before readiness. After readiness match adapter-issued recipe/model/process generation before materialization. Keep registry revision/applied generation distinct from process generation; reject caller evidence.

- [ ] **Step 4: Mutation-check ordering**

Move first check after adapter acquisition and confirm zero-work test fails. Skip final process fence and confirm drift materializes/sends; restore.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/TTS/profile_service.py tldw_chatbook/TTS/request_admission.py \
  tldw_chatbook/TTS/adapter_types.py tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/TTS/adapters/audio_cpp.py Tests/TTS/test_profile_service.py \
  Tests/TTS/test_tts_request_admission.py Tests/TTS/test_audio_cpp_contract.py \
  Tests/TTS/test_audio_cpp_managed_integration.py Tests/TTS/test_stts_audio_cpp_generation.py
git commit -m "feat(tts): enforce clone recipe dependencies"
```

### Task 7: Add warning-gated Voice Profile workflows and shared action truth

**Files:**
- Modify: `tldw_chatbook/UI/stts_profile_library.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/UI/Speech/speech_profile_mixin.py`
- Modify: `tldw_chatbook/UI/STTS_Window.py`
- Modify: `tldw_chatbook/css/features/_lab.tcss`
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_stts_profile_library.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Modify: `Tests/UI/test_speech_profile_navigation.py`

- [ ] **Step 1: Write failing mounted/a11y tests**

Use five red-green microcycles: (A) sanitized default export and secondary warning-gated bundle export; (B) import warning mounted before the picker with acknowledgement receiving initial focus, plus truthful disabled Windows import/export actions/copy; (C) safe review facts and the complete Create/Reuse/Copy × exact/missing dependency × inactive-consent projection, including refusal of Reuse for migrated null provenance, plus stale successor review requiring reconfirmation; (D) cancellation/unmount/late workers and one immutable visible-label/executed-operation projection; and (E) blocker precedence/provenance advisory, shared library/Personas projection, inactive assignment refusal, no default/assignment change, and keyboard/80x24/100x30 accessibility. Run each Pilot node red, implement minimally, rerun green, then Step 2.

```python
@pytest.mark.asyncio
async def test_reference_export_defaults_sanitized_and_bundle_requires_ack(...):
    assert app.focused.id == "bundle-warning-ack"
    assert continue_button.disabled is True

@pytest.mark.asyncio
async def test_missing_dependency_import_is_inactive_unassigned(...):
    assert row.status == "Needs compatible model"
    assert assignment is None
```

- [ ] **Step 2: Run and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_stts_profile_library.py Tests/UI/test_personas_workbench.py \
  Tests/UI/test_speech_profile_navigation.py -q
```

- [ ] **Step 3: Implement in existing library**

Add focused modals/action dataclasses in `stts_profile_library.py`; no second screen. Ordinary Export asks whether a reference exists and chooses v1/sanitized-v2 without reading BLOB. Bundle operations delegate filesystem/private work to the portability service; modal retains only handle/safe facts. Derive label, disable reason, tooltip, operation, and recovery from one immutable action projection. Reuse availability/dependency projection in Personas and refuse unavailable assignment. Add non-sensitive CSS and regenerate bundle.

- [ ] **Step 4: Mutation-check action truth**

Make visible Reuse invoke Copy and confirm action test fails. Enable inactive assignment and confirm Personas test fails; restore.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py \
  tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/UI/Speech/speech_profile_mixin.py \
  tldw_chatbook/UI/STTS_Window.py tldw_chatbook/css/features/_lab.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_stts_profile_library.py \
  Tests/UI/test_personas_workbench.py Tests/UI/test_speech_profile_navigation.py
git commit -m "feat(tts): add clone voice bundle workflows"
```

### Review checkpoint C

- [ ] Check Tasks 6–7 against AC #6, #9, #12 and design action table.
- [ ] Confirm labels/operations share a projection, acknowledgement is operation-local, and no import assigns/defaults.
- [ ] Confirm no reserved/global/terminal-convention binding was added.
- [ ] Run CSS sync, `git diff --check`, and Tasks 6–7 commands.

### Task 8: Complete privacy, docs, full verification, and isolated UAT

**Files:**
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Create: `Docs/superpowers/qa/audio-cpp-clone-voice-bundle-portability-2026-08-11/live-uat.md`
- Modify: `backlog/tasks/task-13206 - Add-explicit-clone-voice-bundle-portability.md`
- Modify: `backlog/docs/lessons-testing-evidence.md` only for a concrete new evidenced lesson
- Modify: any plan-listed source/test required by final review

- [ ] **Step 1: Add cross-cutting privacy/lifecycle regressions**

Inject source/destination/staging paths, member names, transcript, WAV, checksum, provider config/origin, and collaborator-error canaries across codec/service/repository/runtime/UI. Assert absence from logs/events/notifications/metrics/repr/public messages/full exception graphs. Add composite shutdown ownership and old-reader regressions.

- [ ] **Step 2: Run complete scoped/static verification**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_portability.py Tests/TTS/test_profile_reference_types.py \
  Tests/TTS/test_profile_schema.py Tests/TTS/test_profile_reference_storage.py \
  Tests/TTS/test_profile_reference_repository.py Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py Tests/TTS/test_profile_service.py \
  Tests/TTS/test_voice_bundle_codec.py Tests/TTS/test_voice_bundle_hostile_archives.py \
  Tests/TTS/test_voice_bundle_service.py Tests/TTS/test_tts_request_admission.py \
  Tests/TTS/test_audio_cpp_contract.py Tests/TTS/test_audio_cpp_managed_integration.py \
  Tests/TTS/test_stts_audio_cpp_generation.py Tests/UI/test_stts_profile_library.py \
  Tests/UI/test_personas_workbench.py Tests/UI/test_speech_profile_navigation.py \
  Tests/TTS/test_tts_app_ownership.py Tests/DB/test_private_sqlite_inventory.py -q

task13206_python_files=$(git diff --name-only --diff-filter=ACMR origin/dev...HEAD -- '*.py')
test -n "$task13206_python_files"
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check $task13206_python_files

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS/migrations/v3_to_v4.py \
  tldw_chatbook/TTS/voice_bundle_codec.py tldw_chatbook/TTS/voice_bundle_service.py \
  Tests/TTS/test_voice_bundle_codec.py Tests/TTS/test_voice_bundle_hostile_archives.py \
  Tests/TTS/test_voice_bundle_service.py

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/profile_portability.py tldw_chatbook/TTS/profile_reference_types.py \
  tldw_chatbook/TTS/profile_types.py tldw_chatbook/TTS/migrations/v3_to_v4.py \
  tldw_chatbook/TTS/profile_schema.py tldw_chatbook/TTS/profile_reference_storage.py \
  tldw_chatbook/TTS/profile_repository.py tldw_chatbook/TTS/profile_service.py \
  tldw_chatbook/TTS/voice_bundle_codec.py tldw_chatbook/TTS/voice_bundle_service.py \
  tldw_chatbook/TTS/request_admission.py tldw_chatbook/TTS/adapter_types.py \
  tldw_chatbook/TTS/TTS_Generation.py \
  tldw_chatbook/DB/private_sqlite.py tldw_chatbook/UI/stts_profile_library.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_character_tts_widget.py \
  tldw_chatbook/UI/Speech/speech_profile_mixin.py

set +e
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/adapters/audio_cpp.py tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/UI/STTS_Window.py tldw_chatbook/app.py 2>&1 \
  | sed -E 's/:[0-9]+(:[0-9]+)?:/:LINE:/' \
  | grep -vE '^(Found [0-9]+ errors|Success:)' \
  | sort -u > /tmp/task13206-mypy-legacy-final.txt
set -e
comm -13 /tmp/task13206-mypy-legacy-baseline.txt \
  /tmp/task13206-mypy-legacy-final.txt \
  > /tmp/task13206-mypy-new-errors.txt
test ! -s /tmp/task13206-mypy-new-errors.txt

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.css.check_bundle_sync
git diff --check
```

Expected: all pass. Run the full repository suite before PR creation; never reuse old evidence.

- [ ] **Step 3: Perform two-launch isolated audible UAT**

Use two independent temporary HOME/XDG/config/data/profile/model-package/generated-config/runtime roots. Set `TLDW_TEST_MODE`, `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `TLDW_CONFIG_PATH`, and scratch `[paths].data_dir` before importing app code. Never open the developer profile database.

Launch A: create/audibly verify clone; inspect sanitized v2; prove acknowledgement gates bundle publication; export. Launch B: prove no visibility of A's dependency; import inactive; restart and observe **Needs compatible model**; configure exact pre-provisioned dependency; refresh/generate/play with human confirmation. Prove assignment/default unchanged and shutdown has zero sessions/tasks/staging/output/partial rows. Record commit/platform/schema/bundle/recipe/model/sanitized metadata only—no transcript, audio, source/bundle path, checksum, or staging path.

- [ ] **Step 4: Update docs and close task**

Document safe export, bundle warning, inactive recovery, exact recipe admission, v4 downgrade using stable pre-v4 sibling, POSIX support, and shutdown ownership. Add concise Implementation Notes, check ACs only with evidence, link ADR-051, and set TASK-13206 Done through Backlog CLI. Add a lesson only for an actual reusable incident.

- [ ] **Step 5: Commit closeout**

```bash
git add Docs/Features/Speech-Services-Guide.md Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/superpowers/qa/audio-cpp-clone-voice-bundle-portability-2026-08-11/live-uat.md \
  'backlog/tasks/task-13206 - Add-explicit-clone-voice-bundle-portability.md'
git commit -m "docs(tts): close clone voice bundle portability"
```

### Final review checkpoint

- [ ] Request independent code review against the design and all twelve ACs.
- [ ] Fix every validated finding test-first and rerun affected suites.
- [ ] Rebase onto latest `origin/dev`, then regenerate the legacy mypy baseline from that exact merge-base in a clean detached worktree and prove the feature tree adds no normalized diagnostic:

```bash
task13206_merge_base=$(git merge-base HEAD origin/dev)
task13206_baseline_root=$(mktemp -d /tmp/task13206-mypy-base.XXXXXX)
git worktree add --detach "$task13206_baseline_root/worktree" "$task13206_merge_base"

set +e
(
  cd "$task13206_baseline_root/worktree"
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
    tldw_chatbook/TTS/adapters/audio_cpp.py tldw_chatbook/UI/Screens/personas_screen.py \
    tldw_chatbook/UI/STTS_Window.py tldw_chatbook/app.py 2>&1
) | sed -E 's/:[0-9]+(:[0-9]+)?:/:LINE:/' \
  | grep -vE '^(Found [0-9]+ errors|Success:)' \
  | sort -u > /tmp/task13206-mypy-post-rebase-baseline.txt

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/adapters/audio_cpp.py tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/UI/STTS_Window.py tldw_chatbook/app.py 2>&1 \
  | sed -E 's/:[0-9]+(:[0-9]+)?:/:LINE:/' \
  | grep -vE '^(Found [0-9]+ errors|Success:)' \
  | sort -u > /tmp/task13206-mypy-post-rebase-final.txt
set -e

comm -13 /tmp/task13206-mypy-post-rebase-baseline.txt \
  /tmp/task13206-mypy-post-rebase-final.txt \
  > /tmp/task13206-mypy-post-rebase-new-errors.txt
test ! -s /tmp/task13206-mypy-post-rebase-new-errors.txt
git worktree remove "$task13206_baseline_root/worktree"
rmdir "$task13206_baseline_root"
```

- [ ] On that exact rebased tree, rerun the scoped matrix, inventories, CSS sync, changed-file/new-file static gates, full suite, and `git diff --check`.
- [ ] Recheck task-ID uniqueness across every remote/worktree before merge.
- [ ] Create the PR only after task status, ACs, notes, ADR links, UAT, and verification are truthful.
