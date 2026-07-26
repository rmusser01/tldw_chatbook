# TTS Generation Profile Domain and Repository Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the durable, authority-scoped, concurrency-safe local persistence foundation for reusable TTS generation profiles without adding profile-management UI, character assignment UI, roleplay routing, portability, or managed audio.cpp behavior.

**Architecture:** Add immutable profile-domain values in `tldw_chatbook/TTS`, a dedicated schema/version module, a small portalocker-backed shared/exclusive store lease, and one async repository that owns a single-thread executor and at most one long-lived SQLite connection. The repository carries a monotonic lifecycle generation through every queued operation/result, fails stale work closed across restore, and owns online backup plus bounded atomic restore. The application owns one lazily opened repository; Backup All reaches it through its online-backup API instead of copying an open SQLite file.

**Tech Stack:** Python 3.12, frozen dataclasses, SQLite 3, `concurrent.futures.ThreadPoolExecutor`, `asyncio`, portalocker, Textual workers, pytest/pytest-asyncio, Ruff, mypy, Backlog.md.

**Task:** `TASK-761`

**ADR required:** yes
**ADR path:** `backlog/decisions/028-character-tts-generation-profile-ownership.md`
**Reason:** This slice creates a new versioned store and defines profile data ownership, authority-scoped assignments, lifecycle serialization, interprocess exclusion, backup/restore, and cross-module application ownership.

---

## Scope boundary

This plan implements only approved Slice 2A:

- profile-domain validation and immutable values;
- dedicated versioned SQLite schema;
- profile and assignment repository operations;
- optimistic revisions, bounded pagination, assignment counts, and joined reads;
- one serialized off-event-loop lifecycle lane;
- cooperative interprocess shared/exclusive locking;
- online backup and bounded explicit restore;
- profile DB path configuration;
- application ownership and Backup All integration.

It does **not** implement:

- the Slice 2B STTS profile library or profile service;
- saving a Playground result as a profile;
- character-editor controls or authority acquisition;
- Console authorship snapshots or assigned-profile speech;
- character-card import/export;
- automatic speech;
- legacy-provider profile execution;
- server synchronization;
- managed audio.cpp launch, supervision, restart, or shutdown.

## File responsibility map

| File | Responsibility |
| --- | --- |
| `backlog/decisions/028-character-tts-generation-profile-ownership.md` | Canonical ownership, identity, lifecycle, and rollback decision |
| `backlog/tasks/task-761 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md` | Atomic Slice 2A acceptance criteria and delivery evidence |
| `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md` | Approved design with the collision-free ADR-028 reference |
| `tldw_chatbook/TTS/profile_errors.py` | Safe structured profile/repository failures |
| `tldw_chatbook/TTS/profile_types.py` | Immutable domain values and boundary validation |
| `tldw_chatbook/TTS/profile_schema.py` | Schema version, migration registry, connection setup, candidate validation, and row codecs |
| `tldw_chatbook/TTS/profile_store_lock.py` | One bounded portalocker-backed shared/exclusive lock primitive |
| `tldw_chatbook/TTS/profile_repository.py` | Serialized worker, lifecycle generations, CRUD, assignments, backup, and restore |
| `tldw_chatbook/TTS/__init__.py` | Public profile-domain/repository exports only |
| `tldw_chatbook/config.py` | `get_tts_profiles_db_path()` |
| `tldw_chatbook/app.py` | One lazily opened application-owned repository and bounded shutdown |
| `tldw_chatbook/UI/Tools_Settings_Window.py` | Backup All calls repository online backup and records the resulting entry |
| `Tests/TTS/test_profile_types.py` | Name/identifier/options/profile/CharacterRef validation |
| `Tests/TTS/test_profile_schema.py` | New schema, migration, unsupported/corrupt/partial schema, and row validation |
| `Tests/TTS/test_profile_store_lock.py` | Shared/exclusive lock behavior, timeout, cleanup, and process exclusion |
| `Tests/TTS/test_profile_repository.py` | Profile CRUD, pagination, conflicts, assignments, counts, and joined snapshots |
| `Tests/TTS/test_profile_repository_lifecycle.py` | Worker ownership, state/generation transitions, cancellation, backup, restore, and stale work |
| `Tests/TTS/test_profile_backup_integration.py` | Config path, app ownership, Backup All, and no raw profile DB copy |
| `Tests/TTS/test_tts_app_ownership.py` | Existing TTS ownership plus profile-repository construction/close invariants |
| `Docs/Development/TTS/TTS_MODULE_GUIDE.md` | Profile-store module and lifecycle developer contract |
| `Docs/Features/Speech-Services-Guide.md` | User-facing local profile-store backup/restore and privacy boundary |

## Repository contracts fixed by this plan

Use these public shapes unless a red test proves a smaller equivalent is
required:

```python
@dataclass(frozen=True, slots=True)
class TTSProfileDraft:
    display_name: str
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, JsonValue] = field(default_factory=dict)

    @property
    def normalized_name(self) -> str: ...


@dataclass(frozen=True, slots=True)
class TTSGenerationProfile:
    profile_id: UUID
    display_name: str
    normalized_name: str
    provider_id: str
    model_id: str
    voice_id: str | None
    response_format: str
    speed: float
    options: Mapping[str, JsonValue]
    revision: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class CharacterRef:
    source: Literal["local", "server"]
    authority_id: str
    character_id: str


@dataclass(frozen=True, slots=True)
class ProfileStoreResult(Generic[T]):
    generation: int
    value: T
```

Repository methods are asynchronous and return `ProfileStoreResult[T]`:

```python
await repository.open()
await repository.create_profile(draft, profile_id=None)
await repository.get_profile(profile_id)
await repository.list_profiles(search=None, limit=50, offset=0)
await repository.update_profile(profile_id, expected_revision, draft)
await repository.delete_profile(profile_id)
await repository.assignment_count(profile_id)
await repository.set_assignment(character_ref, profile_id)
await repository.remove_assignment(character_ref)
await repository.get_assigned_profile(character_ref)
await repository.backup_to(destination)
await repository.restore_from(candidate, timeout_seconds=5.0)
await repository.close()
```

Every normal data or backup operation:

1. captures the current lifecycle generation before enqueue;
2. verifies state/generation again on the repository worker before touching
   SQLite;
3. verifies generation before publishing the result;
4. returns the generation beside the immutable value;
5. raises a safe structured repository error when stale or unavailable.

Lifecycle operations are admitted under the lifecycle/state locks instead.
Restore changes state and advances generation atomically at admission, before
any restore worker job is enqueued, so work admitted under the prior generation
cannot pass its worker preflight.

### Lifecycle transition contract

`closed` has two internal phases: an initial, reopenable state before the first
`open()` and a terminal state after definitive `close()`. The terminal flag is
not a fifth public state.

| Admission/result | From | To | Generation | Worker/connection rule |
| --- | --- | --- | --- | --- |
| construction | — | initial `closed` | starts at `0` | no executor thread, lease, filesystem, or SQLite I/O |
| first/retry `open()` succeeds | initial `closed` or `unavailable` | `open` | increment once for the attempt | lazily start the executor; acquire shared lease before opening the one long-lived connection |
| first/retry `open()` fails | initial `closed` or `unavailable` | `unavailable` | retain the attempt's incremented generation | keep enough executor ownership for an explicit retry; never create over an invalid existing store |
| idempotent `open()` | `open` | `open` | unchanged | return the current generation without another lease or connection |
| restore admitted | `open` | `restoring` | increment synchronously before enqueue | reject new work, cancel not-started old-generation jobs, and wait only for tracked old-generation work within the caller deadline |
| restore quiescence fails before worker I/O | `restoring` | `open` | retain the new generation | leave the existing connection/shared lease in place and suppress every older result |
| restore fails before replacement and rebind succeeds | `restoring` | `open` | retain the new generation | original file remains authoritative; reacquire shared lease before reopening |
| replacement and rebind succeed | `restoring` | `open` | retain the new generation | close scoped validation connection, reacquire shared lease, then open the long-lived connection |
| rebind/reopen fails | `restoring` | `unavailable` | retain the new generation | retain recovery evidence and never create a blank database |
| first definitive `close()` admission | any nonterminal state | terminal `closed` | increment once | reject/cancel pending work, drain tracked running work, close connection/lease on the worker, then shut down the executor off-loop |
| idempotent `close()` | terminal `closed` | terminal `closed` | unchanged | no executor recreation; `open()` now fails with a structured terminal-state error |

Generations never decrement or get reused. A failed restore may return to
`open`, but it remains on the newly admitted generation and invalidates every
older result. A failed open may be retried from `unavailable`; a terminally
closed repository may not.

## Task 1: Freeze the decision record, task, and baseline

**Files:**

- Modify: `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md`
- Create: `backlog/decisions/028-character-tts-generation-profile-ownership.md`
- Create: `backlog/tasks/task-761 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md`
- Create: `Docs/superpowers/plans/2026-07-26-tts-profile-domain-repository-lifecycle.md`

- [x] **Step 1: Commit the approved planning boundary**

```bash
git add \
  Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md \
  Docs/superpowers/plans/2026-07-26-tts-profile-domain-repository-lifecycle.md \
  backlog/decisions/028-character-tts-generation-profile-ownership.md \
  "backlog/tasks/task-761 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md"
git commit -m "docs(tts): plan generation profile repository lifecycle"
```

Completed as planning commit `af16c04db` after the clean rebase below.

- [x] **Step 2: Rebase and verify the isolated base**

Run:

```bash
git fetch origin dev
git rebase origin/dev
git status --short --branch
git rev-parse HEAD
git merge-base --is-ancestor origin/dev HEAD
```

Expected: branch `codex/tts-profile-domain-repository` contains only the
planning commit on top of the then-current `origin/dev`, has no unrelated
changes, and the ancestor check exits `0`. Record the exact rebased base SHA;
do not rely on the pre-plan `60241b2` base because `dev` may advance while the
plan is reviewed.

Recorded 2026-07-26 base: `origin/dev` at `6095fade8`; the ancestor check
passed and the branch contained only the planning commit.

- [x] **Step 3: Record the focused rebased baseline**

Run:

```bash
../../.venv/bin/python -m pytest Tests/TTS -q
```

Expected: no regression on the rebased `origin/dev`; record the exact pass,
skip, and failure counts. The pre-rebase reference at `60241b2` was
`929 passed, 14 skipped` with one environment warning, but that count is not a
substitute for the rebased gate.

Recorded rebased result: `929 passed, 14 skipped, 1 warning in 210.58s`. The
warning is the existing Requests dependency-version warning in the shared
virtual environment.

- [x] **Step 4: Verify ADR and task hygiene**

Run:

```bash
backlog task 761 --plain
rg -n "ADR-02(7|8)|027-character|028-character" \
  Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md \
  backlog/decisions/028-character-tts-generation-profile-ownership.md \
  "backlog/tasks/task-761 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md"
```

Expected: TASK-761 is In Progress; every profile-ownership reference uses
ADR-028; unrelated ADR-027 remains untouched.

Verified: TASK-761 is In Progress, ADR-028 is linked consistently, and
unrelated ADR-027 was not modified.

## Task 2: Add safe errors and immutable profile-domain validation

**Files:**

- Create: `tldw_chatbook/TTS/profile_errors.py`
- Create: `tldw_chatbook/TTS/profile_types.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Create: `Tests/TTS/test_profile_types.py`

- [ ] **Step 1: Write failing name and identifier tests**

Cover:

- surrounding whitespace is trimmed from display names;
- uniqueness uses `NFKC(trimmed).casefold()`;
- composed/decomposed and non-ASCII case-fold equivalents collide;
- empty and over-128-character names fail;
- Unicode `Cc`, `Cf`, `Cs`, U+FDD0–FDEF, and code points ending FFFE/FFFF fail;
- provider IDs use the existing canonical lower-snake identifier contract and
  are at most 64 characters;
- exact model/voice IDs are non-empty, unmodified, and at most 256 characters;
- provider-neutral speed rejects booleans/non-numbers/non-finite values and
  accepts only the inclusive range `0.25..4.0`, stored as `float`;
- response format is stripped, lowercased, bounded to 1–32 characters, and
  must satisfy `^[a-z][a-z0-9_]{0,31}$`;
- persisted profile revisions reject booleans/non-integers and must be
  positive;
- `CharacterRef` accepts only `local`/`server` with bounded authority and
  character IDs.

Representative test:

```python
def test_profile_name_uses_nfkc_casefold_uniqueness() -> None:
    first = TTSProfileDraft(
        display_name="  Café  ",
        provider_id="audio_cpp",
        model_id="supertonic-3",
        voice_id="M1",
        response_format="wav",
        speed=1.0,
    )
    second = TTSProfileDraft(
        display_name="CAFE\u0301",
        provider_id="audio_cpp",
        model_id="supertonic-3",
        voice_id="M1",
        response_format="wav",
        speed=1.0,
    )

    assert first.display_name == "Café"
    assert first.normalized_name == second.normalized_name
```

- [ ] **Step 2: Run the boundary tests to verify RED**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_types.py -q
```

Expected: import failure because the profile-domain modules do not exist.

- [ ] **Step 3: Implement bounded JSON options**

Implement a canonical JSON validator that:

- accepts an object with string keys and JSON scalars/lists/objects;
- rejects bytes, sets, tuples used as input, non-string keys, and non-finite
  floats;
- rejects more than four container levels;
- rejects canonical UTF-8 output over 16 KiB;
- freezes mappings with `MappingProxyType` and arrays as tuples;
- emits canonical JSON with `sort_keys=True`, compact separators,
  `ensure_ascii=False`, and `allow_nan=False`.

For `provider_id == "audio_cpp"`, reject non-empty options, non-`wav` format,
or speed other than exactly `1.0`.

- [ ] **Step 4: Implement immutable domain values**

Add:

- `TTSProfileDraft`;
- `TTSGenerationProfile`;
- `CharacterRef`;
- `CharacterTTSAssignment`;
- `AssignedTTSProfileSnapshot`;
- `TTSProfilePage`;
- `ProfileStoreResult[T]`;
- `ProfileBackupReceipt` and `ProfileRestoreReceipt`;
- repository-state enum;
- UTC timestamp and UUID validators.

Apply the provider-neutral speed, normalized response-format, and positive
revision validators both when constructing values directly and when decoding
persisted rows. For `audio_cpp`, apply its stricter WAV/speed/options contract
after the provider-neutral checks.

Keep error copy safe and value-independent. Do not put display names,
identifiers, authorities, paths, or raw JSON in exception messages.

- [ ] **Step 5: Run GREEN and static checks**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_types.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_errors.py \
  tldw_chatbook/TTS/profile_types.py \
  Tests/TTS/test_profile_types.py
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_chatbook/TTS/profile_errors.py \
  tldw_chatbook/TTS/profile_types.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_profile_types.py
git commit -m "feat(tts): add generation profile domain contracts"
```

## Task 3: Add the dedicated versioned SQLite schema

**Files:**

- Create: `tldw_chatbook/TTS/profile_schema.py`
- Create: `Tests/TTS/test_profile_schema.py`

- [ ] **Step 1: Write failing new-store and schema tests**

Assert a new file reaches `PRAGMA user_version = 1` and contains:

```sql
tts_generation_profiles
character_tts_assignments
```

Assert profile name uniqueness, assignment composite primary key, assignment
foreign key with delete restriction, assignment profile index, and
`PRAGMA foreign_keys = ON`.

- [ ] **Step 2: Add fail-closed fixture tests**

Create fixtures for:

- valid v1;
- version 0 with partial profile tables;
- version newer than supported;
- v1 missing a required column/index/foreign key;
- corrupt/non-SQLite bytes;
- v1 with failing `quick_check` or `foreign_key_check`.

Expected behavior: no fixture is deleted, recreated, or silently upgraded
outside the explicit migration registry.

- [ ] **Step 3: Run schema tests to verify RED**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_schema.py -q
```

Expected: import failure because `profile_schema.py` does not exist.

- [ ] **Step 4: Implement schema version 1 and migration registry**

Use:

```python
CURRENT_PROFILE_SCHEMA_VERSION = 1
MIGRATIONS = {0: _migrate_v0_to_v1}
```

Rules:

- a truly empty version-0 store may migrate to v1 transactionally;
- version 0 with any profile-owned table is a partial schema error;
- versions above current fail closed;
- every supported step is applied in order and advances `user_version` only
  after its DDL succeeds;
- existing v1 validates table columns, uniqueness, index, foreign key,
  `quick_check`, and `foreign_key_check`;
- connections use row factory, foreign keys, WAL, and an explicit busy timeout;
- candidate validation is read-only and never migrates the candidate.

- [ ] **Step 5: Add row codecs**

Implement exact encode/decode helpers for UUIDs, UTC timestamps, canonical
options JSON, profiles, assignments, and joined snapshots. Row decoding must
re-run domain validation and raise a structured corrupt-data error instead of
returning partially trusted values.

- [ ] **Step 6: Run GREEN and commit**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_schema.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_schema.py \
  Tests/TTS/test_profile_schema.py
git add tldw_chatbook/TTS/profile_schema.py Tests/TTS/test_profile_schema.py
git commit -m "feat(tts): add versioned profile store schema"
```

## Task 4: Add cooperative shared/exclusive profile-store locking

**Files:**

- Create: `tldw_chatbook/TTS/profile_store_lock.py`
- Create: `Tests/TTS/test_profile_store_lock.py`

- [ ] **Step 1: Write failing lease tests**

Cover:

- two shared leases coexist;
- an exclusive lease times out while a shared lease is held;
- a shared lease times out while an exclusive lease is held;
- timeout/check interval reject negative, zero, or non-finite values;
- acquisition failure closes the file handle;
- release is idempotent;
- context cleanup preserves a primary body error and adds cleanup notes;
- safe errors contain no database or lock path.

- [ ] **Step 2: Add a spawned-process exclusion test**

Use `multiprocessing.get_context("spawn")`. A child opens the same lock file in
shared mode and reports readiness through a pipe. The parent must fail a
bounded exclusive acquisition before the child releases it. The test must not
rely on `fork` semantics and must terminate/join the child in `finally`.

- [ ] **Step 3: Run tests to verify RED**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_store_lock.py -q
```

- [ ] **Step 4: Implement the minimal portalocker lease**

Use one stable adjacent lock file derived from the validated database path.
Use non-blocking portalocker shared/exclusive flags with a monotonic deadline
and bounded poll interval. Keep this implementation profile-store-specific;
do not refactor `Model_Artifacts/leases.py` or expose artifact terminology.

- [ ] **Step 5: Run GREEN and commit**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_store_lock.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_store_lock.py \
  Tests/TTS/test_profile_store_lock.py
git add \
  tldw_chatbook/TTS/profile_store_lock.py \
  Tests/TTS/test_profile_store_lock.py
git commit -m "feat(tts): lock profile stores across processes"
```

## Task 5: Add the serialized repository lifecycle

**Files:**

- Create: `tldw_chatbook/TTS/profile_repository.py`
- Create: `Tests/TTS/test_profile_repository_lifecycle.py`

- [ ] **Step 1: Write failing lifecycle tests**

Cover:

- constructor performs no filesystem or SQLite I/O;
- `open()` acquires one shared lease and creates one connection on the
  repository worker;
- repeated `open()` and `close()` are idempotent;
- operations before open, while restoring, after close, or while unavailable
  fail with structured state errors;
- all SQL traces run on one non-event-loop thread;
- definitive close rejects new work before draining/closing the connection and
  shared lease;
- failed open reports `unavailable` without silently creating a replacement;
- retrying open from `unavailable` may recover;
- initial `closed` permits the first open, terminal `closed` rejects reopen,
  and executor shutdown happens exactly once off the event loop;
- the state/generation outcomes match the lifecycle transition table above;
- every result carries the generation captured for its operation.

- [ ] **Step 2: Run lifecycle tests to verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository_lifecycle.py \
  -q
```

- [ ] **Step 3: Implement one-worker ownership**

Use a dedicated `ThreadPoolExecutor(max_workers=1)`. Only worker methods may:

- acquire/release the store lease;
- create/use/close the SQLite connection;
- execute schema work;
- touch database, WAL, SHM, stage, or recovery files.

Protect state/generation with a small thread lock and lifecycle transitions
with one async lock. Normal operation submission captures generation, checks
it before SQL, and checks it again before returning `ProfileStoreResult`.

- [ ] **Step 4: Make cancellation observable and drainable**

Shield submitted worker futures. If the awaiting caller is cancelled, retain
and drain the worker completion before allowing lifecycle shutdown to report
closed. Do not claim that Python thread work was cancelled. A cancelled write
may complete transactionally, but its stale/cancelled result must not populate
later caches.

- [ ] **Step 5: Run GREEN and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository_lifecycle.py \
  -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py
git add \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py
git commit -m "feat(tts): serialize profile repository lifecycle"
```

## Task 6: Implement profile CRUD, pagination, and optimistic revisions

**Files:**

- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Create: `Tests/TTS/test_profile_repository.py`

- [ ] **Step 1: Write failing CRUD tests**

Cover:

- repository-generated UUID and optional caller-supplied UUID;
- initial revision `1` and UTC timestamps;
- exact round-trip of provider/model/voice/format/speed/options;
- normalized-name conflicts across whitespace, normalization, and case-folding;
- `get_profile()` missing behavior;
- list order by normalized name then UUID;
- bounded limit `1..100`, nonnegative offset, total count, and escaped search;
- duplicate UUID conflict;
- failed create/update rolls back fully.

- [ ] **Step 2: Write failing optimistic-update tests**

Two editors load revision 1. Editor A updates to revision 2. Editor B's update
with expected revision 1 must return a conflict and leave the stored revision-2
row unchanged. Updating display-name spelling without changing the normalized
name is allowed for the same profile.

- [ ] **Step 3: Run repository tests to verify RED**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_repository.py -q
```

- [ ] **Step 4: Implement minimal transactional CRUD**

Use parameterized SQL only. Distinguish missing rows, optimistic conflicts,
UUID conflicts, and normalized-name conflicts with safe structured codes.
Never interpolate identifiers or echo user values in errors/logs.

- [ ] **Step 5: Run GREEN and commit**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_repository.py -q
../../.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository.py
git add \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository.py
git commit -m "feat(tts): persist generation profiles transactionally"
```

## Task 7: Implement authority-scoped assignments and joined reads

**Files:**

- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `Tests/TTS/test_profile_repository.py`

- [ ] **Step 1: Write failing assignment tests**

Cover:

- one assignment per exact `(source, authority_id, character_id)`;
- the same character ID under two local authorities does not collide;
- the same character ID under two server authorities does not collide;
- replacing an assignment updates only the exact CharacterRef;
- assignment count reflects all authorities;
- delete is blocked while any assignment exists;
- explicit remove is idempotent;
- foreign keys reject a missing profile;
- joined read returns one immutable profile UUID/revision snapshot;
- a profile edit after the joined read does not mutate the returned snapshot.

- [ ] **Step 2: Run the assignment subset to verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository.py \
  -k "assignment or joined or delete_in_use" \
  -q
```

- [ ] **Step 3: Implement transactional assignment operations**

Add parameterized upsert/remove/count/join operations. Do not add authority
derivation, character lookups, target-status probes, automatic cleanup, or UI
logic; later slices own those policies.

- [ ] **Step 4: Run GREEN and commit**

```bash
../../.venv/bin/python -m pytest Tests/TTS/test_profile_repository.py -q
git add \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository.py
git commit -m "feat(tts): scope profile assignments by authority"
```

## Task 8: Implement online backup, generation-safe restore, and recovery

**Files:**

- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `Tests/TTS/test_profile_repository_lifecycle.py`
- Modify: `Tests/TTS/test_profile_store_lock.py`

- [ ] **Step 1: Write failing online-backup tests**

Start a controlled write and backup on the same serialized lane. The backup
must be a valid SQLite database at schema v1 and contain either the complete
pre-write or complete post-write transaction—never a torn row set. Reject a
destination equal to the live DB or lock file.

- [ ] **Step 2: Write failing restore validation tests**

For corrupt, partial, unsupported-version, integrity-failing, foreign-key
failing, and structurally valid candidates containing a profile row that fails
domain decoding (for example a non-positive revision), assert:

- live DB bytes/logical contents are unchanged;
- repository reopens the existing DB and returns to `open`;
- lifecycle generation advances;
- no fresh empty DB is created;
- safe error copy contains no candidate/live/recovery path.

- [ ] **Step 3: Write restore failure-injection tests**

Inject each boundary deterministically rather than depending on timing:

- quiescence timeout while one old-generation operation is running: restore
  performs no file/lease mutation, returns to `open` on the advanced
  generation, suppresses the old result, and becomes usable after the running
  operation drains;
- pre-restore recovery-backup failure: no replacement occurs, the stage is
  cleaned, the original store is rebound under a shared lease, and state is
  `open`;
- atomic `os.replace()` failure: the original logical contents remain, the
  stage is cleaned, an already-created recovery copy is retained, and the
  original store is rebound under a shared lease;
- post-replacement shared-lease acquisition or long-lived reopen/validation
  failure: state is `unavailable`, the recovery copy is retained, and no
  fallback path creates an empty database.

Assert the exact state, generation, connection/lease ownership, stage cleanup,
recovery-copy policy, and safe error code for every branch.

- [ ] **Step 4: Write the stale-generation interleaving**

Monkeypatch a private worker operation to pause:

1. start one running write;
2. enqueue a second old-generation write;
3. call restore and prove state becomes `restoring` and generation advances
   synchronously before any restore worker body runs;
4. release the running write;
5. prove its result is not published;
6. prove the queued write never executes;
7. prove the restored candidate is authoritative.

- [ ] **Step 5: Write interprocess restore exclusion**

A spawned second process holds the shared store lease. `restore_from()` must
time out before staging/replacement, reacquire its shared lease, reopen the
original store, and remain usable.

- [ ] **Step 6: Implement admission and bounded quiescence**

At `restore_from()` entry:

1. validate a positive finite timeout and compute one monotonic deadline;
2. under the lifecycle lock and small state lock, atomically require `open`,
   set `restoring`, advance generation, close normal admission, and snapshot
   the registered old-generation submissions;
3. cancel every not-started old-generation future; every wrapper also checks
   generation before SQL so a cancellation race still becomes a no-op;
4. await only snapshot futures that could not be cancelled and are not already
   complete, shielded and with only the deadline's remaining time;
5. if any tracked future cannot complete by the deadline, publish `open` on the
   advanced generation without touching the connection or shared lease, retain
   the future for normal draining, and return a quiescence-timeout error;
6. enqueue the restore worker body only after every snapshotted old-generation
   future is cancelled or complete and time remains.

At most one old operation can be running because the executor has one worker.
Its transaction may finish after a timeout, but its old-generation result can
never publish. No restore file or lock mutation is allowed before successful
quiescence.

- [ ] **Step 7: Implement the restore worker and race-free rebind**

On the repository worker:

1. checkpoint and close the live connection;
2. release this process's shared lease;
3. acquire the exclusive store lease using only the caller deadline's
   remaining time; do not stage or replace if no time remains;
4. copy the candidate through SQLite online backup into a same-directory stage
   file and validate the staged snapshot;
5. create a same-directory timestamp/nonce pre-restore online backup of the
   current store;
6. after the successful WAL checkpoint and while exclusive ownership is held,
   remove only the target database's known empty/stale `-wal` and `-shm`
   sidecars so they cannot be replayed against the replacement;
7. fsync the stage where supported and atomically replace the live database
   with it;
8. while still exclusive, open a scoped replacement connection, validate it,
   and close it—never retain this connection across lock handoff;
9. release exclusive ownership, then acquire the long-lived shared lease with a
   separate bounded rebind timeout;
10. only after shared ownership succeeds, open and validate the long-lived
    connection against the file that is then authoritative;
11. publish `open` with the admitted generation and invalidate repository
    consumers.

The release/reacquire gap is safe because there is no open long-lived
connection in it. If another cooperating process replaces the database first,
shared acquisition waits and this repository opens only the latest file after
that process releases exclusive ownership. Do not open under exclusive and
carry that connection past the handoff.

After quiescence but before replacement, every failure releases exclusive
ownership if acquired, reacquires a shared lease, and only then
reopens/validates the current store. If that rebind succeeds, publish `open` on
the advanced generation; otherwise publish `unavailable`. Once a recovery copy
exists, retain it on any failed restore. If replacement succeeds but shared
reacquisition or reopen fails, report `unavailable`, retain the recovery copy,
and never create a blank store automatically.

Always clean unmatched stage files. Never delete the recovery copy
automatically after a failed restore.

- [ ] **Step 8: Run restore/lifecycle tests**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_store_lock.py \
  -q
```

Expected: all pass, including spawned-process coverage.

- [ ] **Step 9: Commit**

```bash
git add \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_store_lock.py
git commit -m "feat(tts): back up and restore profile stores safely"
```

## Task 9: Add profile DB path and one lazy application owner

**Files:**

- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Modify: `Tests/TTS/test_tts_app_ownership.py`
- Create: `Tests/TTS/test_profile_backup_integration.py`

- [ ] **Step 1: Write failing path tests**

Assert:

- default path is
  `<get_user_data_dir()>/tldw_chatbook_tts_profiles.db`;
- `[database].tts_profiles_db_path` accepts a validated custom path;
- traversal/invalid path input fails through the existing path validator;
- path resolution happens at repository construction, not module import.

- [ ] **Step 2: Write failing app-ownership tests**

Assert:

- app construction creates one closed repository without opening SQLite or
  starting a thread;
- `_ensure_tts_profile_repository()` opens the same instance lazily and is
  idempotent;
- open failure leaves the app alive with repository state `unavailable`;
- `_close_tts_profile_repository()` is idempotent;
- the outer shutdown/unmount `finally` closes both the profile repository and
  TTS service;
- no handler or widget constructs a second repository.

- [ ] **Step 3: Run to verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_tts_app_ownership.py \
  -q
```

- [ ] **Step 4: Implement config and app ownership**

The repository constructor must remain pure. Add lazy ensure/close methods to
`TldwCli`; do not place profile-store open on the startup critical path. Log
only operation phase and safe error type/code—never DB path or raw exception
text.

- [ ] **Step 5: Run GREEN and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_tts_app_ownership.py \
  -q
git add \
  tldw_chatbook/config.py \
  tldw_chatbook/app.py \
  tldw_chatbook/TTS/__init__.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/TTS/test_profile_backup_integration.py
git commit -m "feat(tts): own one lazy profile repository"
```

## Task 10: Include the profile store in Backup All

**Files:**

- Modify: `tldw_chatbook/UI/Tools_Settings_Window.py`
- Modify: `Tests/TTS/test_profile_backup_integration.py`

- [ ] **Step 1: Write a failing Backup All integration test**

Use a fake repository whose `backup_to()` records the destination and creates a
valid small file. Assert Backup All:

- obtains the app-owned repository through the lazy ensure method;
- calls `backup_to()` exactly once;
- records `TTS Profiles` in `backup_info.json`;
- waits for profile backup before reporting success;
- reports a partial failure without claiming the profile DB was backed up when
  repository backup fails;
- never calls `shutil.copy2()` with the profile DB as source.

- [ ] **Step 2: Run to verify RED**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py \
  -k "backup_all" \
  -q
```

- [ ] **Step 3: Refactor only the Backup All orchestration**

Keep legacy database behavior unchanged. Move its blocking work into one
awaited Textual thread worker that returns the timestamped directory and
entries. Then await the profile repository's online backup on its own
serialized worker, write the final manifest off-loop, and notify success only
after both paths finish.

Do not add individual profile DB buttons or raw restore UI in this slice.
Explicit restore remains the repository API until the profile-management UI
ships.

- [ ] **Step 4: Run GREEN and commit**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py \
  -q
git add \
  tldw_chatbook/UI/Tools_Settings_Window.py \
  Tests/TTS/test_profile_backup_integration.py
git commit -m "feat(tts): include profiles in Backup All"
```

## Task 11: Documentation, full verification, and task closeout

**Files:**

- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Modify: `backlog/tasks/task-761 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md`

- [ ] **Step 1: Update documentation**

Document:

- local profile DB filename and optional config path;
- local-only ownership and exclusions;
- name normalization and optimistic revisions;
- shared/exclusive process behavior;
- Backup All online-backup semantics;
- explicit restore failure/recovery behavior;
- no STTS library/character assignment/runtime routing yet;
- no provider connection details or managed audio.cpp behavior.

- [ ] **Step 2: Run focused tests**

```bash
../../.venv/bin/python -m pytest \
  Tests/TTS/test_profile_types.py \
  Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_store_lock.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_tts_app_ownership.py \
  -q
```

Expected: all pass.

- [ ] **Step 3: Run the broad TTS regression suite**

```bash
../../.venv/bin/python -m pytest Tests/TTS -q
```

Expected: the exact rebased baseline recorded in Task 1 plus the new passing
tests; no regression delta. Do not compare against the superseded pre-rebase
`929 passed, 14 skipped` reference count.

- [ ] **Step 4: Run static verification**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/TTS/profile_errors.py \
  tldw_chatbook/TTS/profile_types.py \
  tldw_chatbook/TTS/profile_schema.py \
  tldw_chatbook/TTS/profile_store_lock.py \
  tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/TTS/__init__.py \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Tools_Settings_Window.py \
  Tests/TTS/test_profile_types.py \
  Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_store_lock.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_tts_app_ownership.py

../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS/profile_errors.py \
  tldw_chatbook/TTS/profile_types.py \
  tldw_chatbook/TTS/profile_schema.py \
  tldw_chatbook/TTS/profile_store_lock.py \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_types.py \
  Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_store_lock.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_backup_integration.py

../../.venv/bin/python -m compileall -q \
  tldw_chatbook/TTS \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Tools_Settings_Window.py

../../.venv/bin/python -m mypy \
  tldw_chatbook/TTS/profile_errors.py \
  tldw_chatbook/TTS/profile_types.py \
  tldw_chatbook/TTS/profile_schema.py \
  tldw_chatbook/TTS/profile_store_lock.py \
  tldw_chatbook/TTS/profile_repository.py

git diff --check
```

Expected: task-scoped checks pass. Compare any broad `app.py` or
`Tools_Settings_Window.py` baseline diagnostic with unchanged `origin/dev`
rather than fixing unrelated debt.

- [ ] **Step 5: Run repository-wide tests and classify baseline**

```bash
../../.venv/bin/python -m pytest -q
```

Expected: record the exact result. If existing repository failures remain,
compare the exact failures with an untouched `origin/dev` worktree. Do not mark
TASK-761 Done unless the repository Definition of Done is genuinely satisfied.

- [ ] **Step 6: Perform privacy and scope audits**

```bash
rg -n -i \
  "message text|source_text|authority_id|api[_ -]?key|token|base_url|server_url|binary|server\\.json|subprocess|Popen|create_subprocess" \
  tldw_chatbook/TTS/profile_*.py \
  tldw_chatbook/app.py \
  tldw_chatbook/UI/Tools_Settings_Window.py

git diff --name-only origin/dev...HEAD
```

Expected: authority fields exist only in assignment persistence/contracts;
logs/errors/backup manifests contain no authority, profile content, provider
origin, credential, or raw path. No managed-process behavior appears.

- [ ] **Step 7: Request independent code review**

Use `superpowers:requesting-code-review`. Address every validated finding with
TDD and rerun the affected plus broad verification.

- [ ] **Step 8: Update TASK-761**

Check each acceptance criterion only when evidence exists. Add concise
Implementation Notes with:

- approach and files;
- ADR-028;
- schema/lifecycle/backup decisions;
- test/static evidence;
- baseline comparison;
- explicit deferred Slice 2B/3/4 and managed-process boundaries.

Set status to Done only when every task DoD and repository DoD item is met.

- [ ] **Step 9: Final commit**

```bash
git add \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md \
  "backlog/tasks/task-761 - Add-TTS-generation-profile-domain-and-repository-lifecycle.md"
git commit -m "docs(tts): document profile repository lifecycle"
```
