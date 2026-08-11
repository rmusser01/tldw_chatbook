# TASK-13203 Private Clone-Reference Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Advance the TTS profile repository to schema v3 with canonical private clone-reference WAV storage, safe migration, bounded atomic mutations, qualified backup/restore, and per-profile damage isolation.

**Architecture:** Keep `TTSProfileRepository` as the sole lifecycle and transaction owner. Add one focused audio-admission module for regular-file/no-follow WAV canonicalization, one focused reference-persistence module for metadata and streamed SQLite BLOB operations, and one v2→v3 migration module; existing profile reads gain only reference metadata summaries and never load audio or transcript. The repository performs migration backup under its exclusive lease, and full reference qualification only at mutation, exact payload read, backup, and restore boundaries.

**Tech Stack:** Python 3.11+, `asyncio`, `sqlite3`/`Connection.blobopen`, standard-library RIFF/WAVE parsing and writing, SHA-256, pytest, Ruff, mypy.

**Normative design:** `Docs/superpowers/specs/2026-08-09-audio-cpp-guided-model-setup-design.md` (`GM-VOICE-001`, `002`, `007`–`009`; `GM-TEST-004`; `GM-AC-013`, `016`, `017`, `023`).

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/051-private-tts-clone-reference-assets.md`

**Reason:** ADR-051 already fixes the profile-v3 schema, local-plaintext privacy boundary, source-path exclusion, migration backup, backup/restore, and damage-isolation decisions. TASK-13203 implements that accepted decision without changing request materialization, UI, or portable-bundle ownership.

**Deliberate exclusions:** No adapter request fields, temporary materialization, Speech Lab clone UI, character assignment workflow, ordinary export wire-v2 change, explicit voice bundle, Model Library integration, Windows ACL claim, or additional audio.cpp recipes. Those belong to TASK-13204–TASK-13212.

---

### Task 1: Define the private reference domain contract

**Files:**
- Create: `tldw_chatbook/TTS/profile_reference_types.py`
- Modify: `tldw_chatbook/TTS/profile_types.py`
- Modify: `tldw_chatbook/TTS/profile_errors.py`
- Modify: `tldw_chatbook/TTS/__init__.py`
- Create: `Tests/TTS/test_profile_reference_types.py`
- Modify: `Tests/TTS/test_profile_types.py`

- [ ] **Step 1: Write failing domain tests**

Cover exact-type validation, immutable UUID/timestamp/metadata validation, transcript character and UTF-8 bounds, safe `repr`, frozen values, pickle round trips, and an optional metadata-only summary on `TTSGenerationProfile`. Assert the full payload's `repr` contains no transcript, bytes, digest, or UUID.

```python
def test_private_reference_repr_discloses_no_private_value() -> None:
    reference = TTSCloneReference(...)
    rendered = repr(reference)
    assert "PRIVATE TRANSCRIPT" not in rendered
    assert canonical_wav.hex() not in rendered
    assert str(reference.reference_id) not in rendered
    assert reference.sha256 not in rendered
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_types.py \
  Tests/TTS/test_profile_types.py -q
```

Expected: collection/import failures for the new reference types.

- [ ] **Step 3: Add minimal immutable types and named global limits**

Create exact immutable domain values:

```python
@dataclass(frozen=True, slots=True)
class TTSCloneReferenceSummary:
    reference_id: UUID
    byte_length: int
    duration_ms: int
    sample_rate_hz: int
    channels: int
    sample_encoding: Literal["pcm_s16le"]
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True, slots=True, repr=False)
class TTSCloneReference:
    summary: TTSCloneReferenceSummary
    reference_text: str
    sha256: str
    wav_bytes: bytes

    def __repr__(self) -> str:
        return "TTSCloneReference(<private>)"
```

Use named, mutation-tested limits instead of scattered literals. Initial global admission limits are implementation safety bounds, not recipe claims:

- source file: 64 MiB;
- canonical WAV: 32 MiB;
- duration: 60 seconds;
- sample rate: 8,000–96,000 Hz;
- channels: mono or stereo;
- transcript: 4,096 Unicode scalar values and 16 KiB UTF-8;
- repository: 256 references and 512 MiB canonical bytes.

Reject NUL/control/format/surrogate/noncharacter transcript code points, trim only outer whitespace, and otherwise preserve the exact admitted text. Add stable validation/repository codes (`reference_text`, `reference_invalid`, `reference_quota`, `reference_unavailable`) without placing private values in messages.

Add `reference: TTSCloneReferenceSummary | None = None` as a defaulted field on `TTSGenerationProfile`; existing non-reference equality and constructors must remain unchanged.

- [ ] **Step 4: Run tests and mutation-check the privacy guard**

Run the command from Step 2. Temporarily restore dataclass-generated `repr` and confirm the privacy test fails; then restore the safe implementation.

- [ ] **Step 5: Commit the domain boundary**

```bash
git add tldw_chatbook/TTS/profile_reference_types.py \
  tldw_chatbook/TTS/profile_types.py tldw_chatbook/TTS/profile_errors.py \
  tldw_chatbook/TTS/__init__.py Tests/TTS/test_profile_reference_types.py \
  Tests/TTS/test_profile_types.py
git commit -m "feat(tts): define private clone reference types"
```

### Task 2: Canonicalize bounded WAV sources without persisting paths

**Files:**
- Create: `tldw_chatbook/TTS/profile_reference_audio.py`
- Create: `Tests/TTS/test_profile_reference_audio.py`

- [ ] **Step 1: Write the WAV/source-admission matrix**

Generate small deterministic fixtures in memory. Cover valid PCM16 mono/stereo WAVs; reordered/padded chunks; metadata and unknown chunks; odd chunk padding; truncated RIFF/fmt/data; duplicate fmt/data; compressed and unsupported widths; rate/channel/duration/source/canonical limits; FIFO/directory/symlink; source replacement and mutation between validation and final identity check; cleanup; control-flow propagation; and context-free safe errors.

Assert canonical output consists only of one RIFF/WAVE header, one canonical 16-byte `fmt ` chunk, and one `data` chunk, with correct sizes and no arbitrary source metadata.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_audio.py -q
```

Expected: import failure for `profile_reference_audio`.

- [ ] **Step 3: Implement a bounded, standard-library canonicalizer**

Expose one source-boundary function and one byte-boundary validator:

```python
def canonicalize_reference_wav(
    source_path: Path,
    reference_text: str,
) -> CanonicalTTSCloneReference:
    """Pin one regular source, stream/validate PCM16 frames, and return no path."""

def validate_canonical_reference_wav(
    payload: bytes | BinaryIO,
) -> TTSCloneReferenceAudioMetadata:
    """Require the exact canonical RIFF/fmt/data representation."""
```

Open with `O_RDONLY | O_CLOEXEC | O_NONBLOCK | O_NOFOLLOW` where available; compare `lstat`, `fstat`, size, device/inode/mode/mtime/ctime before and after; reject non-regular files; read in bounded chunks; do not use the path again as authority. Parse RIFF sizes with overflow-safe arithmetic. Accept only uncompressed PCM16; preserve admitted sample rate/channels/frame bytes; write a deterministic metadata-free WAV. Compute SHA-256 over canonical bytes. Return a sensitive canonical value whose representation is redacted and whose fields contain no source path.

- [ ] **Step 4: Run tests and prove the source-race guard discriminates**

Run the Step 2 command. Temporarily remove the final source-identity comparison and confirm the replacement/mutation regression fails, then restore it.

- [ ] **Step 5: Commit canonical audio admission**

```bash
git add tldw_chatbook/TTS/profile_reference_audio.py \
  Tests/TTS/test_profile_reference_audio.py
git commit -m "feat(tts): canonicalize private clone references"
```

### Task 3: Add the exact schema-v3 migration and metadata projection

**Files:**
- Create: `tldw_chatbook/TTS/migrations/v2_to_v3.py`
- Create: `tldw_chatbook/TTS/profile_reference_storage.py`
- Modify: `tldw_chatbook/TTS/profile_schema.py`
- Modify: `Tests/TTS/test_profile_schema.py`
- Create: `Tests/TTS/test_profile_reference_storage.py`

- [ ] **Step 1: Write failing v3 schema and migration tests**

Cover exact table/column/index/foreign-key DDL; one-to-one uniqueness; `ON DELETE CASCADE`; UUID/timestamp/metadata decoding; no transcript/BLOB/digest in summary projections; schema-object mutation guards; v0→v3, v1→v3, and populated v2→v3 climbs; v2 profile/assignment domain equivalence; transaction rollback on DDL, row, and post-migration validation failure; and newer-version refusal.

- [ ] **Step 2: Run schema tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_reference_storage.py -q
```

Expected: schema version/table assertions fail at v2.

- [ ] **Step 3: Implement v2→v3 DDL and exact validation**

Create `tts_profile_clone_references` with `profile_id` primary key, unique immutable `reference_id`, canonical BLOB, transcript, lower-case 64-hex digest, bounded integer metadata, encoding, and UTC timestamps. The migration creates only this table/index and advances `user_version` to 3.

Extend the schema manifest to distinguish the v2 pre-migration shape from the exact v3 shape. Capture encoded profile/assignment snapshots before migration, run v2→v3 inside the existing `BEGIN IMMEDIATE`, run full `PRAGMA integrity_check` plus `foreign_key_check`, validate the complete v3 schema and empty reference table before commit, and assert the original domain snapshots are byte-for-byte equivalent. The full integrity/schema/domain checks execute before the migration transaction commits; any failure rolls back the DDL and `user_version` change to v2.

Add a metadata-only `PROFILE_WITH_REFERENCE_SELECT` and decoder. Repository query integration is deliberately deferred to Task 5, where every ordinary `get_profile`, list, collision, and assignment join is changed together; those queries must select reference metadata but never `wav_bytes`, `reference_text`, or `sha256`.

- [ ] **Step 4: Run tests and mutate the exact-schema guard**

Run the Step 2 command. Change one FK/delete rule or add one unexpected column and confirm the schema test fails before restoring it.

- [ ] **Step 5: Commit schema v3**

```bash
git add tldw_chatbook/TTS/migrations/v2_to_v3.py \
  tldw_chatbook/TTS/profile_reference_storage.py \
  tldw_chatbook/TTS/profile_schema.py \
  Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_reference_storage.py
git commit -m "feat(tts): migrate profile storage to v3"
```

### Task 4: Retain a validated owner-private v2 migration backup

**Files:**
- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `tldw_chatbook/DB/private_sqlite.py`
- Modify: `backlog/docs/sqlite-private-owner-inventory.md`
- Modify: `Tests/TTS/test_profile_repository_lifecycle.py`
- Modify: `Tests/DB/test_private_sqlite_inventory.py`

- [ ] **Step 1: Write lifecycle/migration-backup regressions**

Starting only from isolated `tmp_path` v2 stores, cover: exclusive lease before backup/migration; fixed owner-private sibling backup identity; SQLite online backup rather than raw copy; backup validation; retained backup after success; an existing backup that is domain-equivalent to the current v2 source; an existing backup that is valid but stale after later v2 edits; atomic fresh-backup replacement; backup/validation/fsync/migration/close/lease-release failures; concurrent open; cancellation; domain equivalence; POSIX `0600`; no real profile path; and safe public errors/exception graphs.

- [ ] **Step 2: Run focused lifecycle/inventory tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/DB/test_private_sqlite_inventory.py -q
```

- [ ] **Step 3: Implement one guarded migration transaction boundary**

When `peek_profile_store_schema_version()` returns exactly 2, acquire the existing exclusive `ProfileStoreLease`, open/validate v2 without migration, and compare any existing retained backup against the current source using exact schema version, full integrity, and encoded profile/assignment domain snapshots. Reuse it only when those snapshots are equivalent. Otherwise publish a fresh current-source online backup to a private temporary target, validate it as exact v2 and domain-equivalent to the current source, fsync it, atomically replace the stable retained-backup path, and fsync the parent; a failed fresh publication leaves the prior file intact and blocks migration. Only then invoke the transactional v3 migration/full-integrity/domain validation. Do not migrate under the long-lived shared lease. Keep the retained v2 backup inert and never include its path in public errors/logs.

Add the new owner and backup operation to `private_sqlite.py`, the curated inventory, and its ratchet tests. Do not add raw `sqlite3.connect`, `.backup`, or unregistered parent creation.

- [ ] **Step 4: Run tests and prove backup-before-migration ordering**

Run Step 2. Mutate the sequence to migrate before the backup and confirm the ordering regression fails; restore it.

- [ ] **Step 5: Commit guarded migration publication**

```bash
git add tldw_chatbook/TTS/profile_repository.py \
  tldw_chatbook/DB/private_sqlite.py \
  backlog/docs/sqlite-private-owner-inventory.md \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/DB/test_private_sqlite_inventory.py
git commit -m "feat(tts): retain profile v2 migration backup"
```

### Task 5: Add atomic reference mutations and streamed BLOB access

**Files:**
- Modify: `tldw_chatbook/TTS/profile_reference_storage.py`
- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `Tests/TTS/test_profile_repository.py`
- Modify: `Tests/TTS/test_profile_repository_lifecycle.py`
- Create: `Tests/TTS/test_profile_reference_repository.py`

- [ ] **Step 1: Write failing repository contract tests**

Cover attach, replace, remove, metadata-only get/list/assignment reads, exact payload read, parent profile revision increment, immutable/new reference UUID semantics, cascade delete, expected repository generation/revision conflicts, missing profile/reference, transaction rollback, concurrent mutations, cancellation/control-flow cleanup, per-reference and aggregate count/byte quotas, replacement quota delta, and unchanged existing profile/assignment behavior.

Instrument `sqlite3.Connection.blobopen`/the storage seam to prove payload reads and writes use bounded chunks. Assert ordinary list/open does not call `blobopen` and does not select the sensitive columns.

- [ ] **Step 2: Run focused repository tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_repository.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py -q
```

- [ ] **Step 3: Implement repository-owned operations**

Add async public methods with safe immutable inputs:

```python
async def set_reference(
    self,
    profile_id: UUID,
    canonical: CanonicalTTSCloneReference,
    *,
    expected_revision: int,
    expected_generation: int,
) -> ProfileStoreResult[TTSGenerationProfile]: ...

async def remove_reference(...same fences...) -> ProfileStoreResult[TTSGenerationProfile]: ...

async def get_reference(
    self,
    profile_id: UUID,
    *,
    expected_revision: int,
    expected_generation: int,
) -> ProfileStoreResult[TTSCloneReference]: ...
```

Perform quota queries, `zeroblob` allocation, chunked `blobopen` writes, reference publication, and parent profile revision/timestamp update inside one repository transaction. Every `sqlite3.Blob` handle is closed before transaction commit; a short write, close failure, cancellation, or any later mutation failure rolls back the `zeroblob`, reference row, and parent revision atomically. A replacement receives a new immutable reference UUID and quota accounting subtracts the old row. `get_reference` checks parent revision/generation, streams the exact BLOB through a definitively closed handle, revalidates length/digest/canonical WAV/metadata, and returns the private redacted value. Translate SQLite/OS/internal errors after leaving the raw catch so private detail is unreachable through `__context__`.

- [ ] **Step 4: Run tests and mutation-check quota/summary guards**

Run Step 2. Remove aggregate-byte accounting and remove the metadata-only query assertion in turn; confirm their dedicated tests fail, then restore both.

- [ ] **Step 5: Commit atomic reference operations**

```bash
git add tldw_chatbook/TTS/profile_reference_storage.py \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_reference_repository.py
git commit -m "feat(tts): persist clone references atomically"
```

### Task 6: Qualify backups/restores and isolate damaged live references

**Files:**
- Modify: `tldw_chatbook/TTS/profile_reference_storage.py`
- Modify: `tldw_chatbook/TTS/profile_schema.py`
- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `Tests/TTS/test_profile_backup_integration.py`
- Modify: `Tests/TTS/test_profile_repository_lifecycle.py`
- Modify: `Tests/TTS/test_profile_reference_repository.py`

- [ ] **Step 1: Write failing qualification and isolation tests**

Cover backup round-trip with full BLOB/transcript, restored exact bytes, full digest/WAV/metadata/count/byte quota validation before publication, default and explicit backup timeouts, deadline interruption during backup BLOB scans, restore deadline interruption, corrupt BLOB/digest/metadata/transcript, aggregate quota damage, structural corruption, ambiguous ownership, cleanup/rollback failure, and safe errors.

For a live store, corrupt one safely attributable reference after open: its exact read must become `reference_unavailable`, the repository must remain open, unrelated profiles and references must remain usable, and replacement/removal of that exact profile reference must recover it. Structural or ambiguous corruption must still make the repository unavailable.

- [ ] **Step 2: Run focused backup/restore tests and verify RED**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_reference_repository.py -q
```

- [ ] **Step 3: Add full reference qualification only at required boundaries**

Add a bounded keyword-only `timeout_seconds` to `backup_to()` (with a safe default and the existing monotonic/deadline validation pattern), capture its deadline at operation admission, and propagate that exact deadline through online backup and standalone qualification. Extend standalone snapshot qualification to stream every reference BLOB and verify canonical structure, digest, stored metadata, transcript, total count, and total bytes under that deadline progress guard. Backup qualifies the completed snapshot before publication and never publishes after deadline expiry; restore continues to qualify the staged candidate before replacement and the rebound store afterward under its existing deadline.

Keep normal open metadata-focused. Maintain an in-memory generation-local set of safely attributable damaged profile IDs, populated only by an exact reference read failure; clear the matching marker after successful replacement/removal and clear all markers on repository generation replacement/close. Do not reinterpret structural SQLite/schema failures as isolatable row damage.

- [ ] **Step 4: Run tests and prove backup/restore cannot skip BLOB qualification**

Run Step 2. Temporarily bypass full reference qualification for backup and restore separately and confirm the corrupt-digest fixtures publish only under the mutant; then restore the guard.

- [ ] **Step 5: Commit qualification and isolation**

```bash
git add tldw_chatbook/TTS/profile_reference_storage.py \
  tldw_chatbook/TTS/profile_schema.py \
  tldw_chatbook/TTS/profile_repository.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_reference_repository.py
git commit -m "feat(tts): qualify clone reference backups"
```

### Task 7: Document privacy, downgrade, and compatibility truth

**Files:**
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/Features/Speech-Services-Guide.md`
- Modify: `Tests/TTS/test_profile_portability.py`
- Modify: `Tests/TTS/test_tts_logging_privacy.py`
- Modify: `Tests/TTS/test_speech_tts_release_evidence.py`

- [ ] **Step 1: Add failing privacy/compatibility mutation guards**

Assert source paths never enter the DB schema/rows, profile representation, ordinary portability, diagnostics, log records, public exceptions, or nested exception graphs. Assert ordinary profile export remains byte/behavior compatible for non-reference profiles and does not accidentally serialize the new summary. Exercise loguru with a list-appending sink, not `capsys`.

- [ ] **Step 2: Run focused tests and verify RED where documentation/evidence is absent**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_portability.py \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_speech_tts_release_evidence.py -q
```

- [ ] **Step 3: Update documentation without claiming later slices**

Document that reference audio/transcript are local plaintext protected by filesystem ownership rather than encryption; backups contain the same sensitive data; deletion is best effort and not forensic erasure; Windows privacy posture remains unverified until TASK-13208; older builds refuse v3; downgrade requires closing Chatbook and restoring the retained v2 backup with loss of post-migration changes. State that storage alone does not yet enable clone generation, UI setup, or voice-bundle portability.

- [ ] **Step 4: Run tests and inspect the user-facing truth**

Run Step 2 plus `git diff --check`. Confirm docs do not say reference execution or cloning UI shipped in TASK-13203.

- [ ] **Step 5: Commit privacy and downgrade documentation**

```bash
git add Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md \
  Tests/TTS/test_profile_portability.py \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_speech_tts_release_evidence.py
git commit -m "docs(tts): document private clone reference storage"
```

### Task 8: Verify, review, and close TASK-13203

**Files:**
- Modify: `backlog/tasks/task-13203 - Add-private-clone-reference-profile-storage-and-migration.md`
- Modify only if a real incident warrants it: `backlog/docs/lessons-testing-evidence.md` or `backlog/docs/lessons-live-verification.md`

- [ ] **Step 1: Run the complete branch-relevant suite**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_reference_types.py \
  Tests/TTS/test_profile_reference_audio.py \
  Tests/TTS/test_profile_reference_storage.py \
  Tests/TTS/test_profile_reference_repository.py \
  Tests/TTS/test_profile_schema.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_repository_lifecycle.py \
  Tests/TTS/test_profile_backup_integration.py \
  Tests/TTS/test_profile_types.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_profile_portability.py \
  Tests/TTS/test_profile_store_lock.py \
  Tests/TTS/test_tts_logging_privacy.py \
  Tests/TTS/test_speech_tts_release_evidence.py \
  Tests/DB/test_private_sqlite.py \
  Tests/DB/test_private_sqlite_inventory.py -q
```

Do not launch Chatbook or point any ad-hoc script at the real profile store: this task bumps schema v2→v3. Every migration/UAT fixture must use `tmp_path` or an explicitly isolated temporary repository.

- [ ] **Step 2: Run static and generated gates**

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/TTS Tests/TTS Tests/DB/test_private_sqlite_inventory.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check \
  tldw_chatbook/TTS Tests/TTS Tests/DB/test_private_sqlite_inventory.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/profile_reference_types.py \
  tldw_chatbook/TTS/profile_reference_audio.py \
  tldw_chatbook/TTS/profile_reference_storage.py \
  tldw_chatbook/TTS/profile_schema.py \
  tldw_chatbook/TTS/profile_repository.py
git diff --check
```

- [ ] **Step 3: Run self-review and requested review**

Use `superpowers:requesting-code-review`. Review specifically for schema/backup ordering, BLOB allocation/overflow, source TOCTOU, transaction rollback, cancellation/BaseException cleanup, exception-context severing, metadata-only ordinary reads, privacy/logging, and scope creep into TASK-13204+.

- [ ] **Step 4: Close task hygiene only after evidence is green**

Check all eight acceptance criteria, add concise Implementation Notes including the existing ADR-051 link and deviations from this plan, and run:

```bash
backlog task edit 13203 -s Done --notes "Implemented profile-v3 private clone-reference storage, guarded v2 migration backup, atomic quota-bound BLOB operations, qualified backup/restore, damage isolation, and privacy/downgrade documentation. ADR-051 applies; no new ADR."
```

- [ ] **Step 5: Commit closeout**

```bash
git add 'backlog/tasks/task-13203 - Add-private-clone-reference-profile-storage-and-migration.md'
git commit -m "docs(backlog): close task 13203"
```

---

## Execution checkpoints

After Tasks 2, 4, and 6, pause for a focused review before continuing. Do not combine this task with request materialization (TASK-13204), clone UX/character workflows (TASK-13205), explicit bundle portability (TASK-13206), or Model Library integration (TASK-13207).
