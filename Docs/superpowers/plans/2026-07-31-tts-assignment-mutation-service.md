# Exact Character TTS Assignment Mutation Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete TTS Slice 3A with exact `CharacterRef` assignment set/replace and detach mutations that preserve caller-held repository generation, profile revision, capability authority, and expected-current-assignment state.

**Architecture:** Extend the existing `TTSProfileRepository` mutation methods with mandatory compare-and-set inputs and verify them inside the existing `BEGIN IMMEDIATE` transaction. Add two narrow `TTSProfileService` operations that validate immutable caller-held values, perform fresh audio.cpp capability admission before assignment, and forward every expectation to the repository without adding a second assignment store, UI, or speech resolver.

**Tech Stack:** Python 3.11+, asyncio, SQLite, immutable dataclasses, pytest/pytest-asyncio, Ruff, mypy.

**ADR required:** no

**ADR path:** `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`

**Reason:** ADR-037 already requires lifecycle-generation, profile-revision, and expected-current-assignment compare-and-set semantics for Slice 3A.4. This plan implements that accepted boundary without changing schema, ownership, provider/runtime boundaries, or user-facing application structure.

---

## Scope and file map

- `tldw_chatbook/TTS/profile_repository.py`
  - Make assignment mutation expectations mandatory at the public boundary.
  - Recheck lifecycle generation, selected profile revision, and current assignment inside the final transaction.
  - Preserve exact idempotent detach and bounded repository errors.
- `tldw_chatbook/TTS/profile_service.py`
  - Extend the repository protocol with the exact mutation contract.
  - Add service-owned set/replace and detach operations over caller-held immutable values.
  - Reuse the existing authoritative audio.cpp capability observation.
- `Tests/TTS/test_profile_repository.py`
  - Characterize public-input admission, transactional compare-and-set behavior, rollback, lifecycle fencing, and assignment isolation.
- `Tests/TTS/test_profile_service.py`
  - Characterize ordering, capability fencing, forwarded expectations, stale-state failures, hostile collaborator results, and bounded diagnostics.
- `Tests/TTS/test_tts_profile_capabilities.py`
  - Keep the focused capability-integration repository fake conformant with the runtime-checkable profile-service protocol.
- `Tests/TTS/test_tts_app_ownership.py`
  - Keep the focused app-ownership repository fake conformant without adding a lifecycle owner to `TTSProfileService`.
- `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
  - Document the completed Slice 3A mutation boundary and explicitly deferred Slice 3B behavior.
- `backlog/tasks/task-617.4 - Add-exact-character-TTS-assignment-mutation-service.md`
  - Record the plan, completed acceptance criteria, verification evidence, ADR reuse, and concise implementation notes.

No migration, schema, application composition, Textual UI, Console event, speech resolver, Persona behavior, Sync contract, character-card payload, or managed audio.cpp file belongs in this diff.

### Task 1: Pin the repository compare-and-set contract with failing tests

**Files:**
- Modify: `Tests/TTS/test_profile_repository.py`
- Modify: `tldw_chatbook/TTS/profile_repository.py`

- [ ] **Step 1: Update the repository test call sites to supply caller-held expectations**

Use the profile result's exact generation and revision. Creation expects an explicitly unassigned target; replacement expects the exact previously assigned profile:

```python
created = await repository.create_profile(_draft("Assigned"))
assigned = await repository.set_assignment(
    character_ref,
    created.value.profile_id,
    expected_generation=created.generation,
    expected_profile_revision=created.value.revision,
    expected_current_profile_id=None,
)
```

Detach carries the exact profile ID observed by the caller:

```python
await repository.remove_assignment(
    character_ref,
    expected_generation=assigned.generation,
    expected_profile_id=assigned.value.profile_id,
)
```

- [ ] **Step 2: Add public-boundary validation tests**

Extend `test_invalid_public_inputs_fail_before_worker_submission` and the mutated-value tests so exact `int`, `UUID`, and `CharacterRef` values are required for:

```python
expected_generation
expected_profile_revision
expected_current_profile_id
expected_profile_id
```

Assert invalid values never submit worker work or create a database path, and errors expose only a bounded code.

- [ ] **Step 3: Add red tests for transactional set/replace compare-and-set**

Cover:

```python
async def test_set_assignment_requires_expected_unassigned_state() -> None: ...
async def test_replace_assignment_requires_exact_observed_profile() -> None: ...
async def test_set_assignment_rejects_stale_selected_profile_revision() -> None: ...
async def test_assignment_expectations_are_checked_inside_final_transaction() -> None: ...
```

The transaction test should arrange a queued operation or external committed update before mutation admission completes, then prove the stored assignment is unchanged when either the profile revision or current assignment differs from the caller-held expectation.

Add a separate deterministic lifecycle interleaving: wrap the worker assignment
entry point with a barrier that is reached after `_worker_operation` preflight
but before `_worker_set_assignment` opens its transaction. Advance the
repository generation under `_state_lock`, release the barrier, and assert the
operation reports `stale` **and the assignment row was never written**. This
pins the transaction-body generation check rather than relying on admission or
publication fencing.

- [ ] **Step 4: Add red tests for exact detach**

Cover all three required outcomes:

```python
# Matching assignment: remove it.
# No assignment: idempotent success.
# Different replacement assignment: conflict and preserve it.
```

Also retain exact authority isolation: the same `character_id` under another authority is untouched.

- [ ] **Step 5: Run the new repository tests and verify they fail for the missing contract**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository.py \
  -q
```

Expected: failures show that `set_assignment` and `remove_assignment` do not yet accept or enforce the new mandatory expectations; unrelated repository tests remain green.

- [ ] **Step 6: Commit the red repository contract**

```bash
git add Tests/TTS/test_profile_repository.py
git commit -m "test(tts): pin assignment mutation compare-and-set contract"
```

### Task 2: Implement repository-side transactional admission

**Files:**
- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Test: `Tests/TTS/test_profile_repository.py`

- [ ] **Step 1: Add mandatory public mutation parameters**

Use these signatures:

```python
async def set_assignment(
    self,
    character_ref: CharacterRef,
    profile_id: UUID,
    *,
    expected_generation: int,
    expected_profile_revision: int,
    expected_current_profile_id: UUID | None,
) -> ProfileStoreResult[CharacterTTSAssignment]: ...

async def remove_assignment(
    self,
    character_ref: CharacterRef,
    *,
    expected_generation: int,
    expected_profile_id: UUID,
) -> ProfileStoreResult[None]: ...
```

Canonicalize every input before worker submission. Pass `expected_generation` to `_submit_operation` and snapshot all UUID/identity values so caller mutation after enqueue cannot change worker behavior.

- [ ] **Step 2: Recheck lifecycle generation within the final transaction**

Add one private worker helper that acquires `_state_lock`, evaluates `_worker_state_error_locked(expected_generation)`, and raises the existing bounded repository error. Call it from the body of both assignment transactions after `BEGIN IMMEDIATE` and before reading or changing rows.

Do not add a generation column or a second lock. Restore remains the lifecycle owner, and publication retains its existing post-worker generation check.

- [ ] **Step 3: Enforce profile revision and current-assignment CAS for set/replace**

Inside `_worker_set_assignment`:

```python
stored_profile = self._worker_get_profile(connection, profile_id)
if stored_profile.revision != expected_profile_revision:
    raise _repository_error("conflict")

existing = self._worker_get_persisted_assignment(connection, character_ref)
actual_profile_id = None if existing is None else existing.assignment.profile_id
if actual_profile_id != expected_current_profile_id:
    raise _repository_error("conflict")
```

Only then perform the existing insert/upsert and exact round-trip verification. Preserve the original `created_at` on replacement and monotonic `updated_at`.

- [ ] **Step 4: Enforce exact idempotent detach**

Inside `_worker_remove_assignment`:

```python
existing = self._worker_get_persisted_assignment(connection, character_ref)
if existing is None:
    return
if existing.assignment.profile_id != expected_profile_id:
    raise _repository_error("conflict")
```

Delete with the full identity tuple plus `profile_id` in the `WHERE` clause, require exactly one affected row, and verify the exact assignment is absent before commit.

- [ ] **Step 5: Run the repository suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  -q
```

Expected: all tests pass, including rollback, restore-generation, mutation-snapshot, corruption, and interprocess lifecycle regressions.

- [ ] **Step 6: Run static checks for the repository increment**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository.py
```

Expected: both commands exit 0.

- [ ] **Step 7: Commit the repository implementation**

```bash
git add tldw_chatbook/TTS/profile_repository.py Tests/TTS/test_profile_repository.py
git commit -m "feat(tts): fence character assignment mutations"
```

### Task 3: Pin the profile-service mutation behavior with failing tests

**Files:**
- Modify: `Tests/TTS/test_profile_service.py`
- Modify: `Tests/TTS/test_tts_profile_capabilities.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`
- Modify: `tldw_chatbook/TTS/profile_service.py`

- [ ] **Step 1: Extend the fake repository with the exact protocol**

Record every forwarded value:

```python
async def set_assignment(
    self,
    character_ref: CharacterRef,
    profile_id: UUID,
    *,
    expected_generation: int,
    expected_profile_revision: int,
    expected_current_profile_id: UUID | None,
) -> ProfileStoreResult[CharacterTTSAssignment]: ...

async def remove_assignment(
    self,
    character_ref: CharacterRef,
    *,
    expected_generation: int,
    expected_profile_id: UUID,
) -> ProfileStoreResult[None]: ...
```

Permit deterministic injected conflicts, malformed results, and generation changes before publication.

Also add non-executing `set_assignment` and `remove_assignment` methods to
`_AvailabilityRepository` in `test_tts_profile_capabilities.py` and
`FocusedRepository` in `test_tts_app_ownership.py`. Those fakes are passed to
the runtime-checkable protocol during `TTSProfileService` construction; their
new methods must raise `AssertionError("not used")` so these tests continue to
prove capability observation and app composition do not invoke mutations.

- [ ] **Step 2: Add red set/replace service tests**

Define the service boundary as:

```python
async def set_assignment(
    self,
    character_ref: CharacterRef,
    loaded: LoadedTTSProfile,
    expected_current: CharacterTTSAssignment | None,
) -> CharacterTTSAssignment: ...
```

Test:

- exact `CharacterRef`, loaded generation/revision, and expected current assignment are forwarded;
- `None` means explicitly unassigned rather than “read current state for me”;
- an expected assignment must target the same `CharacterRef`;
- fresh authoritative audio.cpp capability is required for every new assignment;
- unverified, unavailable, stale-configuration, and stale-repository outcomes perform no repository mutation;
- generation is checked before capability work, after capability work, and after repository publication;
- repository conflicts remain bounded conflicts;
- malformed repository results fail as `operation_failed` without exposing authority, character, endpoint, path, credential, or submitted text.

- [ ] **Step 3: Add red detach service tests**

Define:

```python
async def detach_assignment(
    self,
    assignment: CharacterTTSAssignment,
    repository_generation: int,
) -> None: ...
```

Test that it:

- forwards the exact assignment `CharacterRef`, exact profile UUID, and caller-held generation;
- does no capability or provider work;
- treats repository `None` as success;
- rejects a stale generation before repository work;
- preserves repository conflict for a replacement assignment;
- rejects forged or malformed assignment and repository values with bounded errors.

- [ ] **Step 4: Run the service tests and verify they fail for missing methods**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_service.py \
  -q
```

Expected: only the new assignment-service cases fail because the service methods and protocol entries do not yet exist.

- [ ] **Step 5: Commit the red service contract**

```bash
git add \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_app_ownership.py
git commit -m "test(tts): pin exact assignment service behavior"
```

### Task 4: Implement the minimal assignment service

**Files:**
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Test: `Tests/TTS/test_profile_service.py`

- [ ] **Step 1: Import and canonicalize existing assignment domain values**

Reuse `CharacterRef` and `CharacterTTSAssignment`; do not introduce a generic assistant reference or assignment revision type. Add private exact-copy validators that:

```python
canonical_ref = CharacterRef(value.source, value.authority_id, value.character_id)
canonical_assignment = CharacterTTSAssignment(
    canonical_ref,
    exact_profile_id,
)
```

Require exact built-in/domain types and exact canonical field equality so mutated frozen objects and subclasses fail closed.

- [ ] **Step 2: Extend `_ProfileRepositoryProtocol`**

Add only `set_assignment` and `remove_assignment` with the signatures implemented in Task 2. Do not add repository ownership, UI helpers, resolver APIs, cleanup jobs, or hidden controls.

- [ ] **Step 3: Implement `set_assignment`**

Order the operation exactly:

1. Canonicalize `CharacterRef`, `LoadedTTSProfile`, and optional expected assignment.
2. Require the expected assignment to target the same canonical reference.
3. Check the caller-held repository generation.
4. Build an exact `TTSProfileDraft` from the loaded profile's generation fields.
5. Call the existing `_require_authoritative_capability` even when the profile was previously available.
6. Recheck repository generation.
7. Call the repository with generation, selected revision, and expected current profile ID.
8. Admit only a result with the exact generation, `CharacterRef`, and selected profile ID.
9. Recheck generation before returning the canonical assignment.

Never substitute the repository's current generation, reload the current assignment, or select another profile/model/voice.

- [ ] **Step 4: Implement `detach_assignment`**

Canonicalize the caller-held assignment and exact nonnegative generation, reject stale generation before repository work, then call `remove_assignment` with its exact reference/profile ID. Admit only an exact-generation result whose value is `None`.

Detach performs no capability observation because it removes selection rather than authorizing a generation profile.

- [ ] **Step 5: Run focused service and capability tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  -q
```

Expected: all tests pass and existing profile create/edit/duplicate/preview behavior remains unchanged.

- [ ] **Step 6: Run static checks for the service increment**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_service.py \
  Tests/TTS/test_profile_service.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS/profile_service.py \
  Tests/TTS/test_profile_service.py
```

Expected: both commands exit 0.

- [ ] **Step 7: Commit the service implementation**

```bash
git add tldw_chatbook/TTS/profile_service.py Tests/TTS/test_profile_service.py
git commit -m "feat(tts): add exact character assignment service"
```

### Task 5: Document the boundary and run the complete slice gate

**Files:**
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `backlog/tasks/task-617.4 - Add-exact-character-TTS-assignment-mutation-service.md`
- Test: `Tests/TTS/test_profile_repository.py`
- Test: `Tests/TTS/test_profile_repository_lifecycle.py`
- Test: `Tests/TTS/test_profile_service.py`
- Test: `Tests/TTS/test_profile_types.py`
- Test: `Tests/TTS/test_tts_profile_capabilities.py`
- Test: `Tests/TTS/test_tts_app_ownership.py`

- [ ] **Step 1: Update the developer guide**

Add a short **Slice 3A assignment mutation service** subsection explaining:

- assignment identity remains exact `(source, authority_id, character_id)`;
- set/replace requires caller-held repository generation, profile revision, expected current assignment, and fresh authoritative capability;
- detach is absence-idempotent but replacement-safe;
- repository transaction checks remain final authority;
- the slice adds no UI, resolver, automatic speech, Persona inheritance, portability, Sync, or managed audio.cpp behavior.

- [ ] **Step 2: Run the task-critical test union**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_types.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_tts_profile_capabilities.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/TTS/test_console_speech_snapshot_admission.py \
  Tests/TTS/test_console_audio_cpp_native.py \
  -q
```

Expected: all tests pass; snapshot admission and native complete-WAV behavior remain unchanged.

- [ ] **Step 3: Run broader profile and UI regressions**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_store_lock.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/UI/test_stts_profile_library.py \
  -q
```

Expected: all tests pass with only documented pre-existing warnings/skips.

- [ ] **Step 4: Run final static and diff checks**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/profile_service.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_service.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/profile_service.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_service.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/profile_service.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/profile_service.py
git diff --check origin/dev...HEAD
```

Expected: all task-owned checks pass. Any broader baseline issue must be reproduced on untouched `origin/dev` and recorded rather than silently attributed to this slice.

- [ ] **Step 5: Confirm scope**

Run:

```bash
git diff --name-only origin/dev...HEAD
```

Expected: only the task, plan, guide, two profile modules, two focused mutation
test files, and the two protocol-fake regression files are changed.

- [ ] **Step 6: Complete the Backlog record**

Check all acceptance criteria only after verification, add concise implementation notes with exact commands/results, retain the ADR-037 link, and set TASK-617.4 to `Done` via the Backlog CLI.

- [ ] **Step 7: Commit documentation and closeout**

```bash
git add \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/superpowers/plans/2026-07-31-tts-assignment-mutation-service.md \
  "backlog/tasks/task-617.4 - Add-exact-character-TTS-assignment-mutation-service.md"
git commit -m "docs(tts): close assignment mutation service slice"
```

- [ ] **Step 8: Request final code review and prepare the PR**

First run:

```bash
git status --short
```

Expected: no output; every task-owned file is committed.

Then use `superpowers:requesting-code-review`, address every valid finding with
focused regression coverage, rerun the affected and final gates, rebase onto
latest `origin/dev`, push the branch, and open one PR for TASK-617.4.
