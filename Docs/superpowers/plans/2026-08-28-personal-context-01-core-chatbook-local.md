# Personal Context 01 — Shared Core and Chatbook Local Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a complete encrypted local-only Personal Context Profile in Chatbook, including manual Settings management and immutable read-only Console context.

**Architecture:** A standalone `tldw-profile-core` package defines immutable canonical models, JSON Schema, fixtures, and deterministic serialization. Chatbook stores encrypted object versions in a dedicated SQLite database behind one `PersonalContextService`; the canonical Settings Screen and Console context builder consume that service.

**Tech Stack:** Python 3.11+, Pydantic 2, SQLite, `cryptography` AESGCM, keyring, scrypt, HMAC-SHA-256, Textual 8.x, pytest, Hypothesis.

**Spec:** `Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md`

## ADR check

```text
ADR required: yes
ADR path: backlog/decisions/099-personal-context-profile-authority-sync-and-encryption.md
Reason: This plan creates the shared schema, encrypted Chatbook authority,
key-custody boundary, and long-lived Settings/runtime integration.
```

## Global Constraints

- Read the suite index and its Global Constraints before starting.
- This plan delivers local-only behavior. Do not add server calls or Sync
  envelopes here.
- ADR-099 is created and accepted before production code begins.
- Shared Core contains no database, HTTP, provider, UI, key-custody, or runtime
  policy code.
- Chatbook uses a dedicated database resolved beneath its profile data
  directory; do not add tables to `ChaChaNotes_DB` or values to TOML.
- One Chatbook installation owns at most one human Personal Context Profile.
  Personas, characters, login identities, and workspaces never create another.
- AES-256-GCM nonces are random 96-bit values and never reused under one DEK.
- The OS keyring is preferred; passphrase-wrapped storage uses scrypt; there is
  no plaintext fallback.
- Context includes only active, unexpired, agent-visible global plus mapped
  active-workspace records and stays within the lesser of 12 KiB UTF-8 or 10%
  of available model input.
- The current explicit user request always outranks profile context.
- Targeted tests are mandatory; request permission before a local full suite.

---

### Task 1: Record ADR-099 before implementation

**Files:**
- Create: `backlog/decisions/099-personal-context-profile-authority-sync-and-encryption.md`
- Modify: `backlog/decisions/README.md`
- Modify: `Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md` only if the ADR number changes
- Modify: `Docs/superpowers/plans/2026-08-28-personal-context-implementation-index.md` only if the ADR number changes

**Interfaces:**
- Consumes: approved design specification.
- Produces: accepted architectural authority referenced by every implementation task.

- [ ] **Step 1: Recheck the ADR number across current refs and open PRs**

Run:

```bash
rg --files backlog/decisions | sort | tail -n 20
git for-each-ref --format='%(refname)' refs/remotes/ | while read -r ref_name; do
  git ls-tree -r -z --name-only "$ref_name" backlog/decisions/ | tr '\0' '\n'
done | rg '/099-' || true
gh pr list --state open --json number,title,files | rg 'backlog/decisions/099-' || true
```

Expected: no existing or in-flight ADR-099. If occupied, select the next unused
number, rename every planned path, and rerun this exact check for the new number.

- [ ] **Step 2: Write the decision record**

The ADR must contain these exact decisions, expanded with context and rejected
alternatives from the spec:

```markdown
# ADR-099: Unify encrypted Personal Context Profile authority across Chatbook and tldw_server

Status: Accepted
Date: 2026-08-28

## Decision

Chatbook and tldw_server are peer replicas of one canonical Personal Context
Profile contract published by `tldw-profile-core`. Chatbook is local-first;
tldw_server is the single V1 home peer. Each runtime owns one encrypted
repository and one mutation service. Sync V2 transports canonical whole
objects and never application-specific projections.

Profile records use global or random canonical workspace scopes, separate
proposal and conflict objects, per-record syncability and agent visibility,
runtime-local agent grants, server-trusted Sync payloads over TLS, per-version
envelope encryption at rest, keyed integrity tags, and a purge-generation
barrier. Personas, auth identity, Companion goals, and Persona/session memory
remain separate authorities.
```

Required ADR sections: Context, Decision, Alternatives considered,
Consequences, Security/privacy consequences, Migration/rollback, and Links.

- [ ] **Step 3: Verify references and formatting**

Run:

```bash
rg -n 'ADR[- ]099|099-personal-context-profile-authority-sync-and-encryption' \
  backlog/decisions/099-personal-context-profile-authority-sync-and-encryption.md \
  backlog/decisions/README.md \
  Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md \
  Docs/superpowers/plans/2026-08-28-personal-context-*.md
git diff --check -- backlog/decisions Docs/superpowers
```

Expected: every reference uses the final number and `git diff --check` is clean.

- [ ] **Step 4: Commit the ADR**

```bash
git add backlog/decisions/099-personal-context-profile-authority-sync-and-encryption.md \
  backlog/decisions/README.md \
  Docs/superpowers/specs/2026-08-28-unified-personal-context-profile-design.md \
  Docs/superpowers/plans/2026-08-28-personal-context-implementation-index.md
git diff --cached --check
git commit -m "docs: record personal context authority"
```

Stage the spec/index only when renumbering changed them.

---

### Task 2: Publish Shared Profile Core v0.1

**Files (`tldw-profile-core` repository):**
- Create: `pyproject.toml`
- Create: `src/tldw_profile_core/__init__.py`
- Create: `src/tldw_profile_core/enums.py`
- Create: `src/tldw_profile_core/payloads.py`
- Create: `src/tldw_profile_core/models.py`
- Create: `src/tldw_profile_core/canonical.py`
- Create: `src/tldw_profile_core/interview.py`
- Create: `src/tldw_profile_core/tool_contracts.py`
- Create: `src/tldw_profile_core/schema_export.py`
- Create: `schemas/personal-context-v1.json`
- Create: `fixtures/v1/*.json`
- Create: `tests/test_models.py`
- Create: `tests/test_canonical.py`
- Create: `tests/test_schema_fixtures.py`

**Interfaces:**
- Consumes: ADR-099 and the approved spec.
- Produces:
  - `SERIALIZED_SCHEMA_VERSION = 1`
  - `ProfileManifest`, `ProfileScope`, `ProfileRecord`, `ProfileProposal`
  - `RecordKind` including `legacy_unclassified`, `RecordState`, `ScopeKind`,
    `SyncMode`, `AgentVisibility`, `ProposalState`, `ProposalOperation`
  - `ProfileControls`, `ProfileProvenance`, `SemanticKey`
  - `InterviewPack`, `InterviewQuestion`, `InterviewTurn`, `InterviewProposalBatch`
  - `ProfileSearchRequest`, `ProfileGetRequest`, `ProfileProposeRequest`,
    `ProfileUpdateRequest`, `ProfilePromoteRequest`, `ProfileToolResult`
  - `canonical_bytes(value: BaseModel) -> bytes`
  - `integrity_tag(value: BaseModel, key: bytes) -> str`
  - `export_json_schema(path: Path) -> None`

- [ ] **Step 1: Write failing model and canonicalization tests**

```python
# tests/test_canonical.py
from datetime import UTC, datetime

from tldw_profile_core import (
    AgentVisibility,
    PreferencePayload,
    ProfileControls,
    ProfileProvenance,
    ProfileRecord,
    RecordKind,
    RecordState,
    SemanticKey,
    SyncMode,
    canonical_bytes,
    integrity_tag,
)


def preference(record_id: str = "11111111-1111-4111-8111-111111111111") -> ProfileRecord:
    now = datetime(2026, 8, 28, tzinfo=UTC)
    return ProfileRecord(
        profile_id="22222222-2222-4222-8222-222222222222",
        record_id=record_id,
        scope_id="33333333-3333-4333-8333-333333333333",
        kind=RecordKind.PREFERENCE,
        payload=PreferencePayload(subject="response.detail", polarity="like", value="concise"),
        semantic_key=SemanticKey(namespace="preference", subject="response.detail"),
        state=RecordState.ACTIVE,
        controls=ProfileControls(sync_mode=SyncMode.SYNCABLE, agent_visibility=AgentVisibility.AGENT_VISIBLE),
        provenance=ProfileProvenance(source="manual", actor="user", reason_code="settings_edit"),
        version_id="44444444-4444-4444-8444-444444444444",
        parent_version_id=None,
        created_at=now,
        updated_at=now,
    )


def test_canonical_bytes_are_stable_and_whitespace_free():
    value = canonical_bytes(preference())
    assert value == canonical_bytes(ProfileRecord.model_validate_json(value))
    assert b"\n" not in value and b": " not in value


def test_integrity_tag_is_keyed_and_versioned():
    record = preference()
    assert integrity_tag(record, b"a" * 32).startswith("hmac-sha256-v1:")
    assert integrity_tag(record, b"a" * 32) != integrity_tag(record, b"b" * 32)


def test_same_scope_semantic_key_is_structured_not_free_text():
    key = preference().semantic_key
    assert key == SemanticKey(namespace="preference", subject="response.detail")
```

Add tests that reject unknown fields, invalid lifecycle values, working context
without `expires_at` or `no_expiry=True`, confidence on `ProfileRecord`, payloads
over 16 KiB canonical bytes, proposal operations without a target/base pair,
interview packs with more than 20 questions or compound question text, and tool
requests that attempt privacy-control, delete, purge, or cross-workspace changes.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models.py tests/test_canonical.py tests/test_schema_fixtures.py -v`

Expected: collection fails because `tldw_profile_core` does not exist.

- [ ] **Step 3: Implement the immutable contract and canonical serializer**

```python
# src/tldw_profile_core/canonical.py
from __future__ import annotations

import hashlib
import hmac
import json

from pydantic import BaseModel


def canonical_bytes(value: BaseModel) -> bytes:
    payload = value.model_dump(mode="json", exclude_none=False, by_alias=True)
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def integrity_tag(value: BaseModel, key: bytes) -> str:
    if len(key) != 32:
        raise ValueError("integrity key must be exactly 32 bytes")
    digest = hmac.new(key, canonical_bytes(value), hashlib.sha256).hexdigest()
    return f"hmac-sha256-v1:{digest}"
```

```python
# src/tldw_profile_core/models.py (public shape)
class ProfileRecord(FrozenModel):
    schema_version: Literal[1] = 1
    profile_id: str
    record_id: str
    scope_id: str
    kind: RecordKind
    payload: ProfilePayload
    semantic_key: SemanticKey | None = None
    state: RecordState
    controls: ProfileControls
    provenance: ProfileProvenance
    version_id: str
    parent_version_id: str | None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None
    no_expiry: bool = False


class ProfileProposal(FrozenModel):
    schema_version: Literal[1] = 1
    proposal_id: str
    profile_id: str
    scope_id: str
    operation: ProposalOperation
    target_record_id: str | None
    base_version_id: str | None
    proposed_record: ProfileRecord | None
    provenance: ProfileProvenance
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    state: ProposalState = ProposalState.PENDING
    created_at: datetime
    expires_at: datetime
```

Use `ConfigDict(extra="forbid", frozen=True)`. Define discriminated typed
payloads for every initial kind and validators for immutable kind, payload/kind
agreement, working-context expiry, proposal target/base requirements, and the
16 KiB canonical payload ceiling. Export JSON Schema deterministically and check
in v1 positive/negative fixtures.

- [ ] **Step 4: Run unit, property, and schema conformance tests**

Run: `pytest -q tests`

Expected: all Shared Core tests pass and regenerating
`schemas/personal-context-v1.json` produces no diff.

- [ ] **Step 5: Build and inspect the package artifact**

Run:

```bash
python -m build
python -m zipfile -l dist/tldw_profile_core-0.1.0-py3-none-any.whl | rg 'schemas|fixtures|tldw_profile_core'
```

Expected: the wheel contains package modules plus JSON Schema and fixtures; it
contains no application, database, HTTP, provider, UI, or key-custody module.

- [ ] **Step 6: Commit and publish the pinned artifact**

```bash
git add pyproject.toml src schemas fixtures tests
git diff --cached --check
git commit -m "feat: publish personal context core v0.1"
```

Publish through the project’s normal package channel, record the immutable
version/hash, then pin that release in Chatbook before Task 3.

---

### Task 3: Add encrypted Chatbook repository and key protection

**Files:**
- Modify: `pyproject.toml`
- Create: `tldw_chatbook/Personal_Context/__init__.py`
- Create: `tldw_chatbook/Personal_Context/crypto.py`
- Create: `tldw_chatbook/Personal_Context/key_protector.py`
- Create: `tldw_chatbook/Personal_Context/repository.py`
- Create: `tldw_chatbook/Personal_Context/repository_models.py`
- Create: `tldw_chatbook/Personal_Context/paths.py`
- Test: `Tests/Personal_Context/test_crypto.py`
- Test: `Tests/Personal_Context/test_key_protector.py`
- Test: `Tests/Personal_Context/test_repository.py`
- Test: `Tests/Personal_Context/test_repository_plaintext_canary.py`

**Interfaces:**
- Consumes: Shared Core v0.1 canonical models and bytes.
- Produces:
  - `ProfileKeyMaterial(encryption_key, integrity_key, key_version)`
  - `ProfileKeyProtector.load_or_create(profile_ref: str) -> ProfileKeyMaterial`
  - `ProfileKeyProtector.delete(profile_ref: str) -> None`
  - `EnvelopeCipher.encrypt(plaintext: bytes, aad: bytes) -> EncryptedEnvelope`
  - `EnvelopeCipher.decrypt(envelope: EncryptedEnvelope, aad: bytes) -> bytes`
  - `PersonalContextRepository.create_provisional_profile() -> ProfileManifest`
  - `get_manifest()`, `list_records()`, `get_record()`, `commit_record_version()`
  - `commit_proposal()`, `list_proposals()`, `resolve_proposal()`
  - `destroy_profile_content()`, `quarantine_object()`

- [ ] **Step 1: Write failing crypto and repository tests**

```python
# Tests/Personal_Context/test_repository_plaintext_canary.py
from pathlib import Path

from tldw_chatbook.Personal_Context.repository import PersonalContextRepository


def test_record_text_never_appears_in_database_or_wal(tmp_path: Path, profile_record, memory_protector):
    canary = "PROFILE-CANARY-DO-NOT-PERSIST-PLAINTEXT-8c58f6"
    record = profile_record(payload_value=canary)
    db_path = tmp_path / "personal-context.db"
    repo = PersonalContextRepository(db_path, key_protector=memory_protector)
    repo.create_provisional_profile()
    repo.commit_record_version(record, expected_version_id=None)
    repo.close()

    durable = b"".join(path.read_bytes() for path in tmp_path.iterdir() if path.is_file())
    assert canary.encode() not in durable
```

Add tests for unique nonces, AAD mismatch, keyed integrity mismatch,
one-profile-per-install enforcement, keyring-unavailable locked state,
passphrase round-trip, no plaintext fallback, transaction rollback, immutable
versions, current-head compare-and-set, quarantine, key destruction, and reopen
through a fresh repository instance.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Personal_Context/test_crypto.py Tests/Personal_Context/test_key_protector.py Tests/Personal_Context/test_repository.py Tests/Personal_Context/test_repository_plaintext_canary.py -v`

Expected: collection fails because `tldw_chatbook.Personal_Context` does not exist.

- [ ] **Step 3: Implement versioned envelopes and key protectors**

```python
# tldw_chatbook/Personal_Context/crypto.py
@dataclass(frozen=True, slots=True)
class EncryptedEnvelope:
    algorithm: str
    nonce: bytes
    ciphertext: bytes
    wrapped_dek: bytes
    key_version: int


class EnvelopeCipher:
    def __init__(self, profile_key: bytes, *, key_version: int = 1) -> None:
        if len(profile_key) != 32:
            raise ValueError("profile key must be exactly 32 bytes")
        self._profile_key = profile_key
        self._key_version = key_version

    def encrypt(self, plaintext: bytes, aad: bytes) -> EncryptedEnvelope:
        dek = secrets.token_bytes(32)
        nonce = secrets.token_bytes(12)
        wrap_nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(dek).encrypt(nonce, plaintext, aad)
        wrapped = wrap_nonce + AESGCM(self._profile_key).encrypt(wrap_nonce, dek, aad)
        return EncryptedEnvelope("aes-256-gcm-v1", nonce, ciphertext, wrapped, self._key_version)

    def decrypt(self, envelope: EncryptedEnvelope, aad: bytes) -> bytes:
        wrap_nonce, wrapped_dek = envelope.wrapped_dek[:12], envelope.wrapped_dek[12:]
        dek = AESGCM(self._profile_key).decrypt(wrap_nonce, wrapped_dek, aad)
        return AESGCM(dek).decrypt(envelope.nonce, envelope.ciphertext, aad)
```

Implement keyring storage first and a passphrase wrapper using the existing
Chatbook scrypt/AES patterns with a profile-specific domain separator. The
protected bundle contains separate random 32-byte envelope-encryption and
integrity keys. Ordinary encryption-key rotation rewraps DEKs and does not
rotate the integrity key. Integrity-key compromise requires a versioned full
integrity rebaseline. A failed protector returns a typed `ProfileLockedError`;
it never creates replacement material for an existing database.

- [ ] **Step 4: Implement the dedicated SQLite repository**

Create schema version 1 with these tables:

```sql
CREATE TABLE profile_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    profile_id TEXT NOT NULL,
    purge_generation INTEGER NOT NULL,
    current_manifest_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE encrypted_objects (
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    scope_id TEXT,
    is_tombstone INTEGER NOT NULL DEFAULT 0,
    algorithm TEXT NOT NULL,
    nonce BLOB NOT NULL,
    ciphertext BLOB NOT NULL,
    wrapped_dek BLOB NOT NULL,
    key_version INTEGER NOT NULL,
    integrity_tag TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (object_type, object_id, version_id)
);
CREATE TABLE object_heads (
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    PRIMARY KEY (object_type, object_id)
);
CREATE TABLE local_runtime_policy (
    scope_id TEXT PRIMARY KEY,
    encrypted_policy_version TEXT NOT NULL
);
CREATE TABLE local_scope_bindings (
    scope_id TEXT PRIMARY KEY,
    encrypted_binding_version TEXT NOT NULL
);
CREATE TABLE encrypted_outbox (
    outbox_id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    envelope_version TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE quarantine (
    quarantine_id TEXT PRIMARY KEY,
    object_type TEXT NOT NULL,
    object_id TEXT NOT NULL,
    version_id TEXT,
    reason_code TEXT NOT NULL,
    created_at TEXT NOT NULL
);
```

Store all object bodies as separate per-version encrypted envelopes. Use one
SQLite transaction for version insert plus head compare-and-set. Do not claim
atomicity with WorkspaceDB, SyncStateRepository, keyring, or exported files.

- [ ] **Step 5: Run repository and canary tests**

Run: `pytest Tests/Personal_Context/test_crypto.py Tests/Personal_Context/test_key_protector.py Tests/Personal_Context/test_repository.py Tests/Personal_Context/test_repository_plaintext_canary.py -v`

Expected: all tests pass, including a close/reopen test using a new connection.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tldw_chatbook/Personal_Context Tests/Personal_Context
git diff --cached --check
git commit -m "feat: add encrypted local personal context store"
```

---

### Task 4: Add the canonical Chatbook application service

**Files:**
- Create: `tldw_chatbook/Personal_Context/service.py`
- Create: `tldw_chatbook/Personal_Context/runtime_policy.py`
- Create: `tldw_chatbook/Personal_Context/export_service.py`
- Create: `tldw_chatbook/Personal_Context/bootstrap.py`
- Test: `Tests/Personal_Context/test_service.py`
- Test: `Tests/Personal_Context/test_runtime_policy.py`
- Test: `Tests/Personal_Context/test_export_service.py`

**Interfaces:**
- Consumes: `PersonalContextRepository`, Shared Core models.
- Produces:
  - `PersonalContextService.status() -> ProfileOperationalStatus`
  - `create_profile() -> ProfileManifest`
  - `create_workspace_scope(local_workspace_id: str, label: str) -> ProfileScope`
  - `map_workspace_scope(local_workspace_id: str, scope_id: str) -> ProfileScope`
  - `create_record(record: ProfileRecord) -> ProfileRecord`
  - `update_record(record_id: str, mutation: RecordMutation, expected_version_id: str) -> ProfileRecord`
  - `archive_record(...)`, `restore_record(...)`, `delete_record(...)`
  - `list_records(scope_ids: tuple[str, ...], include_archived: bool = False) -> tuple[ProfileRecord, ...]`
  - `set_runtime_enabled(enabled: bool) -> None`
  - `set_scope_authority(scope_id: str, authority: AgentAuthority) -> None`
  - `remove_local_profile(confirm_only_copy: bool) -> None`
  - `export_plaintext(request: ExportRequest) -> Path`
  - `export_recovery(request: RecoveryExportRequest) -> Path`

- [ ] **Step 1: Write failing lifecycle, privacy, and export tests**

```python
def test_same_scope_same_key_updates_existing_record_not_duplicate(service, preference_record):
    first = service.create_record(preference_record)
    changed = service.update_record(
        first.record_id,
        RecordMutation(payload=first.payload.model_copy(update={"value": "detailed"})),
        expected_version_id=first.version_id,
    )
    active = service.list_records(scope_ids=(first.scope_id,))
    assert [record.record_id for record in active] == [first.record_id]
    assert changed.parent_version_id == first.version_id


def test_stale_expected_version_creates_no_revision(service, preference_record):
    first = service.create_record(preference_record)
    with pytest.raises(ProfileConflictError):
        service.update_record(first.record_id, RecordMutation(payload=first.payload), "stale-version")
    assert service.get_record(first.record_id).version_id == first.version_id


def test_disable_keeps_records_but_context_use_is_off(service, preference_record):
    service.create_record(preference_record)
    service.set_runtime_enabled(False)
    assert service.list_records(scope_ids=(preference_record.scope_id,))
    assert service.status().runtime_enabled is False
```

Also test expiry, archive/restore, content-free tombstone, user-only preservation,
device-only status, 24-hour encrypted Undo, profile lock, standalone-only-copy
confirmation, plaintext export exclusion of drafts/keys, and passphrase recovery
export round-trip.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Personal_Context/test_service.py Tests/Personal_Context/test_runtime_policy.py Tests/Personal_Context/test_export_service.py -v`

Expected: imports fail because the service modules do not exist.

- [ ] **Step 3: Implement mutation and policy choke points**

```python
class PersonalContextService:
    def create_record(self, record: ProfileRecord) -> ProfileRecord:
        self._require_available_for_write()
        self._require_profile(record.profile_id)
        self._require_known_scope(record.scope_id)
        collision = self._repository.find_active_by_key(record.scope_id, record.kind, record.semantic_key)
        if collision is not None:
            raise ProfileKeyCollisionError(collision.record_id)
        return self._repository.commit_record_version(record, expected_version_id=None)

    def update_record(
        self,
        record_id: str,
        mutation: RecordMutation,
        expected_version_id: str,
    ) -> ProfileRecord:
        current = self._repository.get_record(record_id)
        next_record = mutation.apply(current, now=self._clock(), version_id=self._ids.new())
        self._require_no_key_collision(next_record, excluding_record_id=record_id)
        return self._repository.commit_record_version(
            next_record,
            expected_version_id=expected_version_id,
        )
```

Every public operation checks status and scope before repository access.
Runtime enablement and scope authority stay local and encrypted. Export uses
validated explicit paths and never logs record bodies.

- [ ] **Step 4: Run service tests**

Run: `pytest Tests/Personal_Context/test_service.py Tests/Personal_Context/test_runtime_policy.py Tests/Personal_Context/test_export_service.py -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Personal_Context Tests/Personal_Context
git diff --cached --check
git commit -m "feat: add local personal context service"
```

---

### Task 5: Add My Profile to the canonical Settings Screen

**Files:**
- Modify: `tldw_chatbook/UI/Screens/settings_config_models.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Create: `tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py`
- Create: `tldw_chatbook/css/components/_settings_personal_context.tcss`
- Modify generated CSS through: `python -m tldw_chatbook.css.build_css`
- Test: `Tests/UI/test_settings_personal_context.py`
- Test: `Tests/UI/test_settings_category_sweep.py`
- Test: `Tests/UI/test_settings_footer_hints.py`

**Interfaces:**
- Consumes: `PersonalContextService` from Task 4.
- Produces: `SettingsCategoryId.PERSONAL_CONTEXT`; `PersonalContextSettingsPanel`.

- [ ] **Step 1: Write failing Settings tests with the production CSS harness**

```python
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from tldw_chatbook.UI.Screens.settings_config_models import SettingsCategoryId
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen


async def test_my_profile_category_renders_records_and_status(pilot, personal_context_service):
    screen = SettingsScreen(personal_context_service=personal_context_service)
    await pilot.app.push_screen(screen)
    await screen.select_category(SettingsCategoryId.PERSONAL_CONTEXT)
    assert screen.query_one("#personal-context-status").renderable == "Available"
    assert screen.query(".personal-context-record-row")


async def test_delete_everywhere_is_not_present_in_local_only_phase(pilot, personal_context_service):
    screen = SettingsScreen(personal_context_service=personal_context_service)
    await pilot.app.push_screen(screen)
    await screen.select_category(SettingsCategoryId.PERSONAL_CONTEXT)
    assert not screen.query("#personal-context-delete-everywhere")
    assert screen.query_one("#personal-context-remove-local")
```

Add tests for add/edit/archive/restore/delete, privacy toggles, runtime enable,
locked state, local removal confirmation, plaintext export warning, recovery
export, narrow layout containment, and advertised bindings matching working
actions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_settings_personal_context.py Tests/UI/test_settings_category_sweep.py Tests/UI/test_settings_footer_hints.py -v`

Expected: `SettingsCategoryId.PERSONAL_CONTEXT` is missing.

- [ ] **Step 3: Implement the focused panel and Settings registration**

```python
class PersonalContextSettingsPanel(Widget):
    """My Profile editor; all mutations delegate to PersonalContextService."""

    BINDINGS = [
        Binding("a", "add_record", "Add", show=True),
        Binding("e", "edit_record", "Edit", show=True),
        Binding("d", "delete_record", "Delete", show=True),
        Binding("x", "export_profile", "Export", show=True),
    ]

    def __init__(self, service: PersonalContextService) -> None:
        super().__init__(id="personal-context-settings-panel")
        self._service = service

    @work(thread=True, exclusive=True, group="personal-context-settings-load")
    def load_records(self) -> None:
        snapshot = self._service.settings_snapshot()
        self.app.call_from_thread(self.apply_snapshot, snapshot)
```

Keep the large `settings_screen.py` change to registration, category guidance,
search metadata, constructor injection, and panel mounting. All profile-specific
rendering and actions stay in the new widget. Use safe confirmation modals for
destructive actions and no terminal-convention keybindings.

- [ ] **Step 4: Rebuild CSS and run Settings tests**

Run:

```bash
python -m tldw_chatbook.css.build_css
pytest Tests/UI/test_settings_personal_context.py Tests/UI/test_settings_category_sweep.py Tests/UI/test_settings_footer_hints.py Tests/UI/test_css_bundle_sync_guard.py -v
```

Expected: all targeted tests pass and generated CSS matches source.

- [ ] **Step 5: Perform scratch-profile live verification**

Create a scratch config whose `[paths].data_dir` points to the same scratch
directory, launch Chatbook, open F9 → My Profile, add/edit/archive one canary
record, restart, and verify it remains readable. Fingerprint the real config and
data directory before/after and require no change.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/settings_config_models.py \
  tldw_chatbook/UI/Screens/settings_screen.py \
  tldw_chatbook/Widgets/Settings_Widgets/personal_context_panel.py \
  tldw_chatbook/css/components/_settings_personal_context.tcss \
  tldw_chatbook/css Tests/UI/test_settings_personal_context.py \
  Tests/UI/test_settings_category_sweep.py Tests/UI/test_settings_footer_hints.py
git diff --cached --check
git commit -m "feat: add My Profile settings"
```

Stage only generated CSS files changed by the build.

---

### Task 6: Inject immutable read-only context into Console requests

**Files:**
- Create: `tldw_chatbook/Personal_Context/context_service.py`
- Modify: `tldw_chatbook/Agents/agent_models.py`
- Modify: `tldw_chatbook/Agents/agent_service.py`
- Modify: `tldw_chatbook/Chat/console_agent_bridge.py`
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_models.py`
- Test: `Tests/Personal_Context/test_context_service.py`
- Test: `Tests/Agents/test_personal_context_prompt.py`
- Test: `Tests/Chat/test_console_personal_context_snapshot.py`
- Modify: `Docs/User_Guide/console/chat-basics.md`

**Interfaces:**
- Consumes: Task 4 service and active Chatbook workspace ID.
- Produces:
  - `ProfileContextSnapshot(generation, record_set_revision, scope_id, authority_revision, serialized_block, source_version_ids)`
  - `ProfileContextService.build_snapshot(request: ProfileContextRequest) -> ProfileContextSnapshot`
  - `AgentConfig.personal_context_block: str = ""`

- [ ] **Step 1: Write failing deterministic-context tests**

```python
def test_workspace_override_precedes_global_and_private_records_are_absent(context_service, records):
    snapshot = context_service.build_snapshot(
        ProfileContextRequest(
            current_user_text="Give me the answer",
            active_workspace_scope_id=records.workspace_scope_id,
            available_input_tokens=20_000,
        )
    )
    assert "workspace concise" in snapshot.serialized_block
    assert "global detailed" not in snapshot.serialized_block
    assert records.user_only_canary not in snapshot.serialized_block


def test_context_is_lesser_of_byte_and_token_budget(context_service, many_records):
    snapshot = context_service.build_snapshot(
        ProfileContextRequest(current_user_text="x", available_input_tokens=8_000)
    )
    assert len(snapshot.serialized_block.encode("utf-8")) <= 8_192
    assert snapshot.estimated_tokens <= 800
```

Add tests for corrections/constraints priority, expiry, archived/tombstoned
exclusion, unmapped workspace exclusion, locked/disabled empty snapshots,
deterministic output, escaped malicious strings, cache keys, conflict fallback,
live/Next Send parity, parent/subagent propagation, and snapshot immutability
after a concurrent profile edit.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Personal_Context/test_context_service.py Tests/Agents/test_personal_context_prompt.py Tests/Chat/test_console_personal_context_snapshot.py -v`

Expected: the context service and `AgentConfig.personal_context_block` are missing.

- [ ] **Step 3: Implement the context builder**

```python
@dataclass(frozen=True, slots=True)
class ProfileContextSnapshot:
    generation: int
    record_set_revision: str
    scope_id: str | None
    authority_revision: str
    serialized_block: str
    source_version_ids: tuple[str, ...]
    estimated_tokens: int


class ProfileContextService:
    def build_snapshot(self, request: ProfileContextRequest) -> ProfileContextSnapshot:
        view = self._service.authorized_context_view(request.active_workspace_scope_id)
        eligible = self._select_active_visible_unexpired(view)
        ordered = self._apply_priority_and_workspace_overrides(eligible)
        byte_budget = min(12 * 1024, self._token_budget_bytes(request.available_input_tokens))
        block, versions, token_count = self._serialize_whole_records(ordered, byte_budget)
        return ProfileContextSnapshot(
            generation=view.generation,
            record_set_revision=view.record_set_revision,
            scope_id=view.workspace_scope_id,
            authority_revision=view.authority_revision,
            serialized_block=block,
            source_version_ids=versions,
            estimated_tokens=token_count,
        )
```

Serialize profile values as JSON data inside a fixed block that states it is
user-owned context, not authority. Drop entire lower-priority records when the
budget is reached; never truncate a payload.

An incoming canonical workspace scope without a peer-local mapping appears in
Settings as Unlinked workspace context and is excluded from context, agent
search/tools, and edits until the user explicitly maps it. Unknown newer
records remain opaque and set `Unsupported records present` without exposing
their bodies.

- [ ] **Step 4: Wire the snapshot through live and preview request assembly**

Add `personal_context_block` to `AgentConfig`, copy it into child configs, and
append it through one shared helper in both `_build_model_request` and the live
loop path. `build_first_request_plan` obtains one snapshot and uses the same
block for Console dispatch and `build_context_snapshot`; do not rebuild the
profile midway through a turn.

```python
def append_personal_context(system_content: str, block: str) -> str:
    if not block:
        return system_content
    return f"{system_content}\n\n{block}"
```

- [ ] **Step 5: Run context, agent, and Console tests**

Run:

```bash
pytest Tests/Personal_Context/test_context_service.py \
  Tests/Agents/test_personal_context_prompt.py \
  Tests/Chat/test_console_personal_context_snapshot.py \
  Tests/Agents/test_workspace_context_note_prompt.py -v
```

Expected: all tests pass and existing workspace context remains byte-identical
when the profile block is empty.

- [ ] **Step 6: Update user documentation and commit**

Document enablement, global/workspace precedence, user-only behavior, context
budgeting, and how Next Send exposes the exact generated block.

```bash
git add tldw_chatbook/Personal_Context/context_service.py \
  tldw_chatbook/Agents/agent_models.py tldw_chatbook/Agents/agent_service.py \
  tldw_chatbook/Chat/console_agent_bridge.py \
  tldw_chatbook/Chat/console_chat_controller.py \
  tldw_chatbook/Chat/console_chat_models.py \
  Tests/Personal_Context/test_context_service.py \
  Tests/Agents/test_personal_context_prompt.py \
  Tests/Chat/test_console_personal_context_snapshot.py \
  Docs/User_Guide/console/chat-basics.md
git diff --cached --check
git commit -m "feat: add bounded Console profile context"
```

## Plan 01 completion gate

- The Shared Core artifact and schema fixtures are immutable and pinned.
- ADR-099 is accepted and linked.
- A standalone Chatbook profile survives restart and remains encrypted at rest.
- My Profile provides complete local CRUD, privacy, enablement, export, and
  local deletion.
- Console live send and Next Send use the same immutable bounded snapshot.
- No server, Sync, interview, or profile-write tool is required for this phase.
