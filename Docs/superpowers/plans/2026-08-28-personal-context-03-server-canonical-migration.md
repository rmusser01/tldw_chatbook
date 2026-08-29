# Personal Context 03 — Server Canonical Storage and Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make tldw_server an encrypted canonical Personal Context peer with an authenticated API, fenced migration from legacy personalization data, lossless compatibility routes, and governed server-side context/tools.

**Architecture:** tldw_server pins the same Shared Profile Core artifact as Chatbook and extends the existing per-user `Personalization.db`. A server master key wraps per-profile keys; one `PersonalContextService` owns canonical mutations. A per-user migration state machine converts legacy rows before compatibility routes project the canonical service. Runtime context and MCP tools consume the same service and never write storage directly.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic 2, SQLite, `cryptography` AESGCM, scrypt, HMAC-SHA-256, Shared Profile Core, Unified MCP, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md`

## ADR check

```text
ADR required: yes
ADR path: backlog/decisions/099-personal-context-profile-authority-sync-and-encryption.md
Reason: This plan establishes server key custody, canonical encrypted storage,
legacy migration/cutover, authenticated service contracts, and runtime boundaries.
```

## Global Constraints

- Complete Plans 01 and 02 and pin the identical released Shared Core version.
- Work in the tldw_server repository. Extend
  `tldw_Server_API/app/core/DB_Management/Personalization_DB.py`; do not create a
  second per-user database or make Sync V2 the live canonical store.
- The only supported V1 server key source is an explicitly configured 32-byte
  master key. A missing, changed, or invalid key yields `Locked`; it never
  generates a replacement.
- Authenticate and resolve the per-user database before any profile lookup or
  decrypt operation. Cross-user object IDs return the same not-found response.
- Canonical bodies, kinds, provenance, proposals, Undo, migration snapshots,
  compatibility staging, and future Sync outbox bodies remain encrypted at rest.
- One service owns all canonical mutations. New API routes, compatibility
  routes, migration, MCP tools, and runtime integration never issue object-table
  SQL directly.
- Migration is per-user, forward-only, fenced with durable phases, idempotent,
  and has no dual-write interval.
- Legacy semantic memory IDs, content, and timestamps survive migration.
  Response style and preferred format become canonical preferences. Runtime
  tuning, episodic/session memory, topics, Companion data, Persona state, and
  auth identity remain in their existing authorities.
- No Sync domains are added in this plan; Plan 04 owns replication.

---

### Task 1: Add server key custody and encrypted canonical repository

**Files (tldw_server repository):**
- Modify: `pyproject.toml`
- Modify: `tldw_Server_API/app/core/DB_Management/Personalization_DB.py`
- Create: `tldw_Server_API/app/core/Personalization/personal_context_crypto.py`
- Create: `tldw_Server_API/app/core/Personalization/personal_context_key_provider.py`
- Create: `tldw_Server_API/app/core/Personalization/personal_context_repository.py`
- Create: `tldw_Server_API/app/core/Personalization/personal_context_repository_models.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_crypto.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_key_custody.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_repository.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py`

**Interfaces:**
- Consumes: pinned Shared Core package and the existing `PersonalizationDB` path
  and SQLite policy.
- Produces:
  - `ServerProfileKeyProvider.load(profile_id) -> ProfileKeyMaterial`
  - `ServerProfileKeyProvider.create(profile_id) -> ProfileKeyMaterial`
  - `PersonalContextRepository.initialize_schema() -> None`
  - encrypted manifest/scope/record/proposal version and head operations
  - `PersonalizationDB.transaction(immediate: bool = False)`
  - `ProfileStorageLockedError`, `ProfileIntegrityError`

- [ ] **Step 1: Write failing key and repository tests**

```python
def test_missing_master_key_locks_existing_profile(tmp_path, monkeypatch, manifest):
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", base64_key(b"a" * 32))
    repo = repository_at(tmp_path)
    repo.commit_manifest(manifest)

    monkeypatch.delenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY")
    with pytest.raises(ProfileStorageLockedError):
        repository_at(tmp_path).get_manifest(manifest.profile_id)


def test_changed_master_key_never_replaces_profile_key(tmp_path, monkeypatch, record):
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", base64_key(b"a" * 32))
    repository_at(tmp_path).commit_record_version(record, expected_version_id=None)
    monkeypatch.setenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", base64_key(b"b" * 32))
    with pytest.raises(ProfileStorageLockedError):
        repository_at(tmp_path).get_record(record.record_id)
```

Add tests for random 96-bit nonces, unique per-version DEKs, AES-GCM associated
data binding object/profile/version IDs, HMAC verification before parse, key
version rotation, stale optimistic writes, content-free tombstones, transaction
rollback, profile isolation, malformed ciphertext, and reopen through
`PersonalizationDB.for_user()`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_crypto.py \
  tldw_Server_API/tests/Personalization/test_personal_context_key_custody.py \
  tldw_Server_API/tests/Personalization/test_personal_context_repository.py \
  tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py -v
```

Expected: imports fail because the server repository modules do not exist.

- [ ] **Step 3: Add the schema to the existing database**

Add versioned tables through `PersonalizationDB._ensure_schema()`:

```sql
CREATE TABLE personal_context_profile_keys (
    profile_id TEXT PRIMARY KEY,
    key_version INTEGER NOT NULL,
    integrity_key_version INTEGER NOT NULL,
    wrapped_profile_key BLOB NOT NULL,
    wrap_nonce BLOB NOT NULL,
    wrapped_integrity_key BLOB NOT NULL,
    integrity_wrap_nonce BLOB NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE personal_context_object_versions (
    profile_id TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    parent_version_id TEXT,
    schema_version INTEGER NOT NULL,
    key_version INTEGER NOT NULL,
    nonce BLOB NOT NULL,
    wrapped_dek BLOB NOT NULL,
    wrapped_dek_nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    integrity_tag TEXT NOT NULL,
    payload_size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, object_type, object_id, version_id)
);
```

Add a head table with `(profile_id, object_type, object_id)` primary key and
`current_version_id`, plus encrypted receipt/runtime tables. Clear columns are
limited to opaque routing IDs, schema/key versions, parent linkage, timestamps,
and byte sizes. Lifecycle, kind, semantic key, visibility, sync mode, labels,
and content stay in ciphertext.

Expose a parameterized `transaction(immediate=True)` context manager so the
repository can use `BEGIN IMMEDIATE` without reaching into `_connect()`.

- [ ] **Step 4: Implement key custody and envelope encryption**

```python
class ServerProfileKeyProvider:
    ENV_NAME = "TLDW_PERSONAL_CONTEXT_MASTER_KEY"

    def require_master_key(self) -> bytes:
        raw = os.getenv(self.ENV_NAME, "").strip()
        try:
            key = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ProfileStorageLockedError("invalid server profile master key") from exc
        if len(key) != 32:
            raise ProfileStorageLockedError("server profile master key must be 32 bytes")
        return key
```

Use AES-256-GCM with random 12-byte nonces. Generate separate random 32-byte
per-profile envelope-encryption and integrity keys plus a random DEK for every
object version. Wrap both profile keys with the server master key and each DEK
with the envelope-encryption key. Bind opaque IDs and versions as associated
data. Verify the Shared Core keyed integrity tag with the integrity key before
model parse. Ordinary encryption-key rotation rewraps DEKs and leaves the
separately wrapped integrity key unchanged; integrity-key rotation uses a
versioned full rebaseline. Do not log key bytes, ciphertext, canonical bodies,
or exception values that may contain them.

- [ ] **Step 5: Prove the no-plaintext boundary**

Create one canary in each kind, proposal, runtime policy, Undo receipt, and WAL
path. Check decoded database pages, WAL, SHM, logs, diagnostics, temporary
files, and exception text. The canaries may appear only after explicit decrypt
inside the test process.

- [ ] **Step 6: Run tests and commit**

Run the four tests from Step 2. Then:

```bash
git add pyproject.toml \
  tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/core/Personalization/personal_context_crypto.py \
  tldw_Server_API/app/core/Personalization/personal_context_key_provider.py \
  tldw_Server_API/app/core/Personalization/personal_context_repository.py \
  tldw_Server_API/app/core/Personalization/personal_context_repository_models.py \
  tldw_Server_API/tests/Personalization/test_personal_context_crypto.py \
  tldw_Server_API/tests/Personalization/test_personal_context_key_custody.py \
  tldw_Server_API/tests/Personalization/test_personal_context_repository.py \
  tldw_Server_API/tests/Personalization/test_personal_context_plaintext_canary.py
git diff --cached --check
git commit -m "feat: add encrypted server personal context store"
```

---

### Task 2: Add the canonical server service and authenticated API

**Files:**
- Create: `tldw_Server_API/app/core/Personalization/personal_context_service.py`
- Create: `tldw_Server_API/app/core/Personalization/personal_context_runtime_policy.py`
- Create: `tldw_Server_API/app/core/Personalization/personal_context_export.py`
- Create: `tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py`
- Create: `tldw_Server_API/app/api/v1/schemas/personal_context.py`
- Create: `tldw_Server_API/app/api/v1/endpoints/personal_context.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/content.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_service.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_auth_boundary.py`

**Interfaces:**
- Consumes: Task 1 repository, authenticated user dependency, Shared Core.
- Produces:
  - `personal_context_service_for_user(user_id) -> PersonalContextService`
  - lifecycle/status/scope/record/proposal/review/export/purge methods matching
    Chatbook semantics
  - `/api/v1/personal-context/status`
  - `/api/v1/personal-context/manifest`
  - `/api/v1/personal-context/scopes`
  - `/api/v1/personal-context/records`
  - `/api/v1/personal-context/proposals`
  - `/api/v1/personal-context/runtime`
  - `/api/v1/personal-context/export`
  - `/api/v1/personal-context/purge`

- [ ] **Step 1: Write failing service and route tests**

```python
def test_cross_user_record_id_is_not_decrypted(client, users, seeded_profiles):
    response = client.get(
        f"/api/v1/personal-context/records/{seeded_profiles[users.b].record_id}",
        headers=users.a.headers,
    )
    assert response.status_code == 404
    assert response.json() == {"detail": "Personal context record not found"}


def test_stale_update_returns_machine_readable_conflict(client, user, record):
    response = client.patch(
        f"/api/v1/personal-context/records/{record.record_id}",
        headers=user.headers,
        json={"expected_version_id": "stale", "payload": record.payload.model_dump()},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "profile_version_conflict"
```

Add tests for every operational state, strict schemas, 16 KiB payload ceiling,
default/max search limits of 5/20, workspace ownership, no user-only agent view,
proposal review, runtime enablement, recovery/plaintext exports, local-copy
delete refusal on the server, typed errors, and purge confirmation/generation.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py \
  tldw_Server_API/tests/Personalization/test_personal_context_auth_boundary.py -v
```

Expected: service, schemas, and routes are missing.

- [ ] **Step 3: Implement one mutation service**

```python
class PersonalContextService:
    def update_record(
        self,
        record_id: str,
        mutation: RecordMutation,
        expected_version_id: str,
    ) -> ProfileRecord:
        self._require_writable()
        current = self._repository.require_record(record_id)
        replacement = mutation.apply(
            current,
            now=self._clock(),
            version_id=self._ids.new(),
        )
        self._require_no_same_scope_key_collision(replacement, excluding=record_id)
        return self._repository.commit_record_version(
            replacement,
            expected_version_id=expected_version_id,
        )
```

Mirror Chatbook lifecycle and validation. Keep server runtime enablement and
scope authority local to the server and encrypted; do not expose those values
as canonical Sync objects. Enforce per-user scope access in the service even
after the route dependency has authenticated.

- [ ] **Step 4: Implement authenticated routes and error mapping**

Dependencies resolve `user_id` before constructing the repository. Response
models use Shared Core-compatible fields and reject unknown input. Map Locked,
Migration required, Migrating, Sync attention, Review, Purge pending, and
Unsupported to stable status/error codes without leaking storage details.
Return only bounded lists and never include raw profile bodies in logs.

- [ ] **Step 5: Run API tests and commit**

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py \
  tldw_Server_API/tests/Personalization/test_personal_context_auth_boundary.py -v
git add tldw_Server_API/app/core/Personalization/personal_context_service.py \
  tldw_Server_API/app/core/Personalization/personal_context_runtime_policy.py \
  tldw_Server_API/app/core/Personalization/personal_context_export.py \
  tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py \
  tldw_Server_API/app/api/v1/schemas/personal_context.py \
  tldw_Server_API/app/api/v1/endpoints/personal_context.py \
  tldw_Server_API/app/api/v1/router_groups/minimal.py \
  tldw_Server_API/app/api/v1/router_groups/content.py \
  tldw_Server_API/tests/Personalization/test_personal_context_service.py \
  tldw_Server_API/tests/Personalization/test_personal_context_endpoints.py \
  tldw_Server_API/tests/Personalization/test_personal_context_auth_boundary.py
git diff --cached --check
git commit -m "feat: expose canonical personal context API"
```

---

### Task 3: Build the fenced per-user legacy migration

**Files:**
- Create: `tldw_Server_API/app/core/Personalization/personal_context_migration.py`
- Create: `tldw_Server_API/app/core/Personalization/personal_context_migration_models.py`
- Modify: `tldw_Server_API/app/core/DB_Management/Personalization_DB.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_migration.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_historical_reopen.py`
- Test fixtures: `tldw_Server_API/tests/Personalization/fixtures/personalization_pre_context_v1.sql`

**Interfaces:**
- Consumes: existing legacy tables and Task 2 service.
- Produces:
  - `PersonalContextMigrationService.inspect(user_id) -> MigrationAssessment`
  - `migrate(user_id) -> MigrationReceipt`
  - durable phases `required`, `snapshot_created`, `canonical_written`,
    `validated`, `plaintext_cleanup_pending`, `complete`, `failed_locked`
  - encrypted recovery snapshot retained for at most seven days

- [ ] **Step 1: Write truthful historical-schema tests**

```python
def test_legacy_semantic_ids_content_and_timestamps_survive_migration(historical_db):
    legacy = historical_db.semantic_memory("memory-7")
    receipt = migration_service(historical_db).migrate("user-1")
    migrated = canonical_service(historical_db).get_record("memory-7")
    assert migrated.payload.text == legacy.content
    assert migrated.created_at.isoformat() == legacy.created_at
    assert migrated.kind == RecordKind.LEGACY_UNCLASSIFIED
    assert receipt.semantic_memory_count == 1


def test_second_migration_is_idempotent(historical_db):
    first = migration_service(historical_db).migrate("user-1")
    second = migration_service(historical_db).migrate("user-1")
    assert second.migration_id == first.migration_id
    assert canonical_service(historical_db).record_count() == first.record_count
```

Build the fixture with the actual pre-feature SQL, including rows that exercise
missing newer columns. Reopen it through `PersonalizationDB.for_path()` so both
`_ensure_schema()` and `_migrate_schema()` execute. Test interruption after each
phase, concurrent workers, invalid master key, corrupt row, duplicate semantic
IDs, default response-style values, purge marker, and no accidental migration
of episodic, topic, Companion, Persona, or auth records.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_migration.py \
  tldw_Server_API/tests/Personalization/test_personal_context_historical_reopen.py -v
```

Expected: the migration service and durable state table are missing.

- [ ] **Step 3: Implement the durable migration state machine**

```python
def migrate(self, user_id: str) -> MigrationReceipt:
    with self._db.transaction(immediate=True) as cursor:
        state = self._states.require_or_create(cursor, user_id)
        if state.phase == "complete":
            return state.receipt
        self._require_key_available()
        self._advance_inside_fence(cursor, state)
    return self._finish_plaintext_cleanup(user_id)
```

Create an encrypted, checksummed recovery snapshot before canonical writes.
Map each semantic memory to `legacy_unclassified` with the same ID, content,
tags, pinned/hidden status in provenance/controls, and original timestamps.
Create deterministic semantic keys only for response style and preferred format;
do not invent semantic keys for unclassified text. Migrate non-default style and
format values as preferences; preserve explicit defaults only when the legacy
profile was enabled. Leave all runtime tuning in the existing profile row.

Wire the authenticated per-user dependency to call
`PersonalContextMigrationService.ensure_current(user_id)` before returning the
canonical service. This is lazy automatic storage maintenance for that user's
database, not a new opt-in. If another process owns the fence, return the typed
Migrating state. Requests for other users continue against their separate
per-user databases.

- [ ] **Step 4: Validate, clean plaintext, and recover safely**

Validate IDs, counts, hashes, and successful Shared Core parse before cleanup.
Enable `PRAGMA secure_delete=ON`, remove migrated semantic bodies, checkpoint
the WAL, and perform `VACUUM` only in the separate
`plaintext_cleanup_pending` phase because SQLite cannot vacuum inside the
transaction. Any interruption resumes that phase while canonical APIs remain
locked. Crypto-shred the snapshot DEK immediately after successful post-cutover
validation, with seven days as the absolute maximum when validation or recovery
remains incomplete. Never roll back to dual writes.

- [ ] **Step 5: Run migration tests and commit**

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_migration.py \
  tldw_Server_API/tests/Personalization/test_personal_context_historical_reopen.py -v
git add tldw_Server_API/app/core/Personalization/personal_context_migration.py \
  tldw_Server_API/app/core/Personalization/personal_context_migration_models.py \
  tldw_Server_API/app/core/DB_Management/Personalization_DB.py \
  tldw_Server_API/app/api/v1/API_Deps/personal_context_deps.py \
  tldw_Server_API/tests/Personalization/test_personal_context_migration.py \
  tldw_Server_API/tests/Personalization/test_personal_context_historical_reopen.py \
  tldw_Server_API/tests/Personalization/fixtures/personalization_pre_context_v1.sql
git diff --cached --check
git commit -m "feat: migrate legacy personalization into personal context"
```

---

### Task 4: Cut legacy routes over to a lossless compatibility adapter

**Files:**
- Create: `tldw_Server_API/app/core/Personalization/personal_context_compat.py`
- Modify: `tldw_Server_API/app/api/v1/API_Deps/personalization_deps.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/personalization.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/personalization.py`
- Modify: `tldw_Server_API/tests/Personalization/test_personalization_endpoints.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_compat.py`

**Interfaces:**
- Consumes: completed migration and canonical service.
- Produces: `LegacyPersonalizationAdapter` projections for existing endpoint
  request/response models, with deprecation headers and no direct legacy writes.

- [ ] **Step 1: Write failing compatibility-equivalence tests**

```python
def test_legacy_memory_update_preserves_unknown_canonical_fields(adapter, canonical_record):
    before = canonical_record.model_dump()
    adapter.update_memory(canonical_record.record_id, content="new text")
    after = adapter.service.get_record(canonical_record.record_id).model_dump()
    assert after["payload"]["text"] == "new text"
    assert after["controls"] == before["controls"]
    assert after["provenance"]["source"] == before["provenance"]["source"]


def test_lossy_legacy_write_fails_closed(adapter, structured_preference):
    with pytest.raises(LegacyProjectionLossError):
        adapter.update_memory(structured_preference.record_id, content="flattened")
```

Cover list/get/create/update/delete/import/validate/profile preference behavior,
legacy pagination, preserved IDs, disabled feature flag, migration-required and
locked responses, purged users, deprecation headers, and identical auth
isolation. Assert endpoint modules no longer call semantic-memory write methods.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_compat.py \
  tldw_Server_API/tests/Personalization/test_personalization_endpoints.py -v
```

Expected: legacy routes still depend directly on `PersonalizationDB`.

- [ ] **Step 3: Implement compatibility projections**

Legacy semantic-memory reads project only canonical
`legacy_unclassified` records. Known legacy writes become canonical service
mutations with expected-version checks. Response style and preferred format
project the two canonical preference records; runtime fields continue through
the legacy runtime-config service. A request that cannot be represented without
discarding canonical fields returns `409 legacy_projection_loss`.

After the per-user fence reports `complete`, no legacy route writes
`semantic_memories`, `response_style`, or `preferred_format`. This is a cutover,
not a mirrored dual write.

- [ ] **Step 4: Run compatibility tests and commit**

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_compat.py \
  tldw_Server_API/tests/Personalization/test_personalization_endpoints.py -v
git add tldw_Server_API/app/core/Personalization/personal_context_compat.py \
  tldw_Server_API/app/api/v1/API_Deps/personalization_deps.py \
  tldw_Server_API/app/api/v1/endpoints/personalization.py \
  tldw_Server_API/app/api/v1/schemas/personalization.py \
  tldw_Server_API/tests/Personalization/test_personalization_endpoints.py \
  tldw_Server_API/tests/Personalization/test_personal_context_compat.py
git diff --cached --check
git commit -m "refactor: route legacy personalization through personal context"
```

---

### Task 5: Add immutable server context and governed MCP profile tools

**Files:**
- Create: `tldw_Server_API/app/core/Personalization/personal_context_runtime.py`
- Create: `tldw_Server_API/app/core/MCP_unified/modules/implementations/personal_context_module.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/server.py`
- Modify: `tldw_Server_API/app/core/MCP_unified/module_surface.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/chat.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/persona.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_runtime.py`
- Test: `tldw_Server_API/app/core/MCP_unified/tests/test_personal_context_module.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_chat_integration.py`

**Interfaces:**
- Consumes: Task 2 service, Shared Core tool contracts, authenticated run scope.
- Produces:
  - `build_personal_context_block(user_id, workspace_scope_id, token_budget) -> PersonalContextBlock`
  - opt-in `PersonalContextModule` with `profile_search`, `profile_get`,
    `profile_propose`, `profile_update`, `profile_promote`
  - immutable personal-context section in ordinary chat and Persona planning

- [ ] **Step 1: Write failing context and tool-boundary tests**

```python
def test_workspace_override_replaces_global_key_in_server_context(runtime, records):
    block = runtime.build_block(records.user_id, records.workspace_scope_id, 8192)
    assert "workspace concise" in block.text
    assert "global detailed" not in block.text


@pytest.mark.asyncio
async def test_default_server_tool_authority_creates_proposal(module, mcp_context):
    result = await module.execute_tool(
        "profile_propose",
        {"kind": "preference", "subject": "response.detail", "value": "concise"},
        context=mcp_context,
    )
    assert result["status"] == "proposal_created"
    assert module.service.list_records(scope_ids=(mcp_context.scope_id,)) == ()
```

Add tests for user-only omission, expired/archived/deleted omission, 12 KiB/10%
budget, conflict omission, explicit request precedence, locked/disabled behavior,
read-only/propose/direct-write catalogs, trusted current-message evidence, no
cross-user/workspace access, quotas, approval impossibility, and tool-module
registration disabled by default. Also prove an external MCP caller cannot
claim direct-write evidence: only an in-process chat turn carrying a
server-created run stamp may receive `direct_write`; generic MCP sessions are
intersected down to propose/read-only.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_personal_context_module.py \
  tldw_Server_API/tests/Personalization/test_personal_context_chat_integration.py -v
```

Expected: runtime builder and MCP module do not exist.

- [ ] **Step 3: Implement one deterministic context builder**

Order active agent-visible records by workspace overlay, global base, kind
priority, explicit user priority, then stable ID. Suppress an unresolved
same-scope key and use the last mutually acknowledged occupant when available.
Serialize a delimited data section with an instruction that the current request
and system policy outrank it. Return only an immutable string snapshot.

Call the builder once per request after auth/scope resolution. Append the same
snapshot to ordinary chat's system-layer assembly and Persona's `_propose_plan`
input; do not merge it into Companion knowledge, semantic memory, or persisted
turn metadata. Preview/diagnostics expose counts and IDs only.

- [ ] **Step 4: Implement the opt-in Unified MCP module**

Build tool JSON Schema from Shared Core. Resolve user/scope/message evidence
from the authenticated MCP context, not tool arguments. Intersect runtime-local
authority with operational status. Default to `propose`; expose update only for
`direct_write` and a server-created in-process chat run stamp. Store only
evidence hash and message reference. Register behind
`MCP_ENABLE_PERSONAL_CONTEXT_MODULE=true` and add the module risk surface.

- [ ] **Step 5: Run integration tests and commit**

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_personal_context_module.py \
  tldw_Server_API/tests/Personalization/test_personal_context_chat_integration.py -v
git add tldw_Server_API/app/core/Personalization/personal_context_runtime.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/personal_context_module.py \
  tldw_Server_API/app/core/MCP_unified/server.py \
  tldw_Server_API/app/core/MCP_unified/module_surface.py \
  tldw_Server_API/app/api/v1/endpoints/chat.py \
  tldw_Server_API/app/api/v1/endpoints/persona.py \
  tldw_Server_API/tests/Personalization/test_personal_context_runtime.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_personal_context_module.py \
  tldw_Server_API/tests/Personalization/test_personal_context_chat_integration.py
git diff --cached --check
git commit -m "feat: add governed server personal context runtime"
```

---

### Task 6: Verify live migration, recovery, and operator documentation

**Files:**
- Create: `Docs/Operations/personal-context-profile.md`
- Modify: `Docs/Product/Personalization_Memory_Layer_PRD.md`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_recovery.py`
- Test: `tldw_Server_API/tests/Personalization/test_personal_context_durable_owner_inventory.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: tested backup/key/migration/recovery runbook and deprecation record.

- [ ] **Step 1: Write failing recovery and durable-owner tests**

Test recovery export/import with the correct passphrase, wrong passphrase,
missing/changed server key, interrupted migration, seven-day snapshot expiry,
purge receipt, and a canary scan across DB/WAL/SHM, Sync DB, logs, caches,
exports, crash inputs, and temporary files.

- [ ] **Step 2: Run the targeted server feature suite**

```bash
pytest tldw_Server_API/tests/Personalization/test_personal_context_*.py \
  tldw_Server_API/tests/Personalization/test_personalization_endpoints.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_personal_context_module.py -v
```

Expected: all Personal Context, compatibility, migration, recovery, and MCP
tests pass. This is targeted; do not run the full repository suite without the
user's approval.

- [ ] **Step 3: Perform a real scratch-server probe**

Start a server with a scratch user database root and explicit scratch master
key. First call capabilities/health, then an authenticated control endpoint.
Create legacy semantic memory, restart through the production initialization
path, migrate, read through both APIs, mutate through each API, exercise context
and propose tools, rotate the profile key, restart, and verify no plaintext
canary. Record redacted request/status evidence; never record profile bodies.

- [ ] **Step 4: Document operations and supersession**

Document master-key creation/backup/rotation, Locked recovery, migration phase
recovery, encrypted snapshot retention, backup inclusion, purge interaction,
legacy route deprecation/sunset criteria, server-trusted TLS threat model, and
the fact that `Personalization_Memory_Layer_PRD.md` is superseded for human
profile facts while episodic/Companion/Persona systems remain separate.
Inventory pre-migration plaintext backups and external snapshots, define their
expiry/removal procedure, and state plainly that the new system cannot
retroactively encrypt copies outside its control.

- [ ] **Step 5: Commit**

```bash
git add Docs/Operations/personal-context-profile.md \
  Docs/Product/Personalization_Memory_Layer_PRD.md \
  tldw_Server_API/tests/Personalization/test_personal_context_recovery.py \
  tldw_Server_API/tests/Personalization/test_personal_context_durable_owner_inventory.py
git diff --cached --check
git commit -m "docs: complete personal context server operations"
```

## Plan 03 completion gate

- tldw_server and Chatbook parse the same canonical fixtures and schema version.
- Existing users migrate once behind a durable per-user fence with preserved
  semantic IDs/content/timestamps and no dual-write interval.
- Missing or changed key material locks data and never replaces it.
- New and legacy APIs share one mutation service and remain user-isolated.
- Server chat/Persona context is immutable and profile MCP tools are optional,
  scope-bound, proposal-first, and evidence-gated.
- All default durable owners are encrypted or content-free, and recovery/key
  operations have a tested runbook.
