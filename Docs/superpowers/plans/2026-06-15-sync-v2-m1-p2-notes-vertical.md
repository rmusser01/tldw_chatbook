# Sync v2 M1 — Phase 2: notes.note Vertical Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Get a real `notes.note` upsert→push→pull→tombstone round-trip green against the live M1 server (`:8076`), asserting server materialization (`apply_status == applied`), with a verified re-apply no-op.

**Architecture (additive superset):** `SyncV2Envelope` and the push/pull response models are shared by the whole sync subsystem (`LocalFirstSyncService`, the existing builder/applier). So P2 evolves them **additively** — add the M1 fields (`object_id`, `object_revision`, `base_*`, `schema_version`, `deleted`, `encryption_metadata`) and widen the literals (`tombstone`, dotted domains, `server_trusted_v1`) *alongside* the legacy fields/values, with compat fallbacks (`object_id` ⇄ `entity_id`, `payload` ⇄ `payload_clear`). The legacy `client_private_v1` path keeps working (it is the M3-parked path); the new M1 notes path is built on top. Every commit stays green. This mirrors how the server's own transition layer works.

**Scope:** the `notes.note` vertical primitives + a focused, real round-trip proof. P2 deliberately does **not** rewire the full `LocalFirstSyncService.sync_once` / `ManualSyncControlService` / outbox UI flow onto M1 — that integration (and chat) is later. The round-trip is driven by a focused `NotesM1SyncFlow` using the conformed client + builder + applier + mirror against a concrete local notes store.

**Tech Stack:** Python 3.11+, Pydantic v2, httpx, pytest. Run pytest only via `.venv/bin/python -m pytest` (system python breaks collection).

**Spec:** `Docs/superpowers/specs/2026-06-12-sync-v2-client-m1-conformance-design.md`
**ADR:** `backlog/decisions/008-sync-v2-client-m1-contract-alignment.md`
**Server contract:** `Sync_V2_M1.md` @ `992e89a03`. Materializer facts (verified): `notes.note` payload requires non-empty `title` (str) + `content` (str, falls back to `body`); server stores `object_hash = payload_hash`; updates/tombstones require `base_object_revision` + `base_object_hash` matching the server head; idempotent if `object_revision`+`payload_hash` match.

**Server launch for live tasks** (Task 7) — the M1 server fail-closes bootstrap (412) without attestation:
```bash
: "${SINGLE_USER_API_KEY:?Set SINGLE_USER_API_KEY before starting the M1 server}"
d=~/Documents/GitHub/tldw_server2/.worktrees/sync-v2-m1-next
PY=~/Documents/GitHub/tldw_server2/.venv/bin/python
cd "$d" && PYTHONPATH="$d" AUTH_MODE=single_user \
  SINGLE_USER_API_KEY="$SINGLE_USER_API_KEY" TLDW_API_PORT=8076 \
  SYNC_V2_SERVER_TRUSTED_ENABLED=true SYNC_V2_AT_REST_ENCRYPTION_MODE=managed_storage \
  nohup "$PY" -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8076 \
  > /tmp/syncqa-v2/server.log 2>&1 &
```

**Universal rules for every task:**
- `git add` ONLY the exact files listed in that task's commit step. NEVER `git add -A`/`.` (uncommitted CLAUDE.md + untracked scratch files must not be committed). Run `git status` before committing.
- Follow existing style in the files you touch.

---

## File Structure
- `tldw_chatbook/tldw_api/sync_schemas.py` — additive M1 fields on `SyncV2Envelope`; widen literals; add `idempotent`/`apply_errors`/`server_cursor`/`from_cursor` to push/pull responses.
- `tldw_chatbook/Sync_Interop/hashing.py` (new) — versioned canonical payload hash.
- `tldw_chatbook/Sync_Interop/notes_mirror.py` (new) — focused per-object mirror (object_id → object_revision, object_hash, server_cursor).
- `tldw_chatbook/Sync_Interop/notes_local_store.py` (new) — `NotesSyncLocalStore` protocol + `InMemoryNotesStore`.
- `tldw_chatbook/Sync_Interop/envelope_builder.py` — add `build_notes_note_upsert` / `build_notes_note_tombstone` (M1, cleartext, base from mirror).
- `tldw_chatbook/Sync_Interop/domain_adapters/notes_m1.py` (new) + `envelope_applier.py` — dotted `notes.note` apply route (cleartext, idempotent, soft-delete).
- `tldw_chatbook/Sync_Interop/notes_m1_flow.py` (new) — `NotesM1SyncFlow` orchestrating bootstrap→build→push→pull→apply for notes.
- Tests under `Tests/Sync_Interop/`.

---

## Task 1: Additively extend `SyncV2Envelope` + literals to the M1 superset

**Files:** Modify `tldw_chatbook/tldw_api/sync_schemas.py`. Test: `Tests/Sync_Interop/test_envelope_m1_superset.py` (create).

- [ ] **Step 1: Write the failing test**

Create `Tests/Sync_Interop/test_envelope_m1_superset.py`:

```python
"""P2: SyncV2Envelope supports the M1 superset additively (legacy still works)."""

from tldw_chatbook.tldw_api import SyncV2Envelope


def test_m1_notes_envelope_round_trips_canonical_fields():
    env = SyncV2Envelope(
        client_envelope_id="dev1:notes.note:note_1:h",
        dataset_id="ds_1",
        device_id="dev1",
        domain="notes.note",
        object_id="note_1",
        operation="tombstone",
        adapter_version=1,
        schema_version=1,
        object_revision=2,
        base_server_cursor=10,
        base_object_revision=1,
        base_object_hash="sha256:prev",
        deleted=True,
        payload={"deleted_at": "2026-06-15T00:00:00Z"},
        payload_hash="sha256:cur",
        encryption_metadata={"policy": "server_trusted_v1"},
    )
    dumped = env.model_dump(mode="json")
    assert dumped["object_id"] == "note_1"
    assert dumped["domain"] == "notes.note"
    assert dumped["operation"] == "tombstone"
    assert dumped["object_revision"] == 2
    assert dumped["base_object_hash"] == "sha256:prev"
    assert dumped["payload"] == {"deleted_at": "2026-06-15T00:00:00Z"}
    assert dumped["encryption_metadata"]["policy"] == "server_trusted_v1"


def test_object_id_falls_back_to_entity_id_and_vice_versa():
    # Legacy construction with entity_id still populates object_id.
    legacy = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds", domain="notes",
        entity_id="e_1", operation="upsert", adapter_version=1, payload_hash="sha256:x",
    )
    assert legacy.object_id == "e_1"
    assert legacy.entity_id == "e_1"
    # M1 construction with object_id exposes entity_id too.
    m1 = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds", domain="notes.note",
        object_id="o_1", operation="upsert", adapter_version=1, payload_hash="sha256:x",
    )
    assert m1.entity_id == "o_1"
    assert m1.object_id == "o_1"


def test_payload_mirrors_payload_clear():
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds", domain="notes.note",
        object_id="o", operation="upsert", adapter_version=1, payload_hash="sha256:x",
        payload={"title": "T", "content": "B"},
    )
    assert env.payload_clear == {"title": "T", "content": "B"}


def test_legacy_client_private_envelope_still_valid():
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds", domain="chat",
        entity_id="m1", operation="upsert", adapter_version=1,
        payload_clear={}, payload_hash="sha256:x",
        encryption_policy="client_private_v1", payload_ciphertext="abc",
    )
    assert env.encryption_policy == "client_private_v1"
    assert env.operation == "upsert"
```

- [ ] **Step 2: Run to confirm failure**

`.venv/bin/python -m pytest Tests/Sync_Interop/test_envelope_m1_superset.py -q`
Expected: FAIL — `object_id`, `object_revision`, `base_object_hash`, `tombstone` op, `notes.note` domain, `server_trusted_v1` policy not accepted.

- [ ] **Step 3: Widen the literals**

In `tldw_chatbook/tldw_api/sync_schemas.py`, replace the three literal definitions (currently ~lines 97-100) so they are **supersets** (keep all legacy values, add M1 values):

```python
SyncV2Domain = Literal[
    # legacy coarse domains (parked client_private path)
    "notes", "chat", "workspaces", "source_cache", "media",
    # M1 dotted domains
    "notes.note", "chat.conversation", "chat.message", "attachment.ref",
    "workspaces.workspace", "workspaces.source_ref", "source_cache.entry",
    "media.item", "media.keyword", "media.keyword_link",
]
SyncV2Operation = Literal[
    "upsert", "delete", "link", "unlink", "resolve_conflict",  # legacy
    "append", "tombstone",  # M1
]
SyncV2DatasetScope = Literal["personal", "workspace"]
SyncV2EncryptionPolicy = Literal[
    "client_private_v1", "server_trusted", "shared_workspace_v1",  # legacy
    "server_trusted_v1", "passphrase_wrapped_v1", "device_wrapped_v1",  # M1
]
```
(Leave `SYNC_V2_DOMAINS`/`SYNC_V2_OPERATIONS`/`SYNC_V2_ENCRYPTION_POLICIES` lists as-is.)

- [ ] **Step 4: Add M1 fields + compat to `SyncV2Envelope`**

In the `SyncV2Envelope` class (currently ~line 350), make `entity_id` optional and add the M1 fields + validators. Replace the field block so it reads:

```python
    client_envelope_id: str
    dataset_id: str
    domain: SyncV2Domain
    entity_id: str | None = None
    object_id: str | None = None
    parent_id: str | None = None
    operation: SyncV2Operation
    adapter_version: int = Field(1, ge=1)
    schema_version: int = Field(1, ge=1)
    device_id: str | None = None
    client_profile_id: str | None = None
    stable_key: str | None = None
    client_timestamp: str | None = None
    created_at_client: str | None = None
    server_timestamp: str | None = None
    received_at_server: str | None = None
    server_sequence: int | None = Field(None, ge=0)
    server_cursor: int | None = Field(None, ge=0)
    client_sequence: int | None = Field(None, ge=0)
    base_version: str | int | None = None
    entity_version: str | int | None = None
    object_revision: int | None = Field(None, ge=0)
    base_server_cursor: int | None = Field(None, ge=0)
    base_object_revision: int | None = Field(None, ge=0)
    base_object_hash: str | None = None
    deleted: bool = False
    dependencies: list[dict[str, Any]] = Field(default_factory=list)
    routing_metadata: dict[str, Any] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    payload_ciphertext: str | None = None
    payload_clear: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str
    payload_size_bytes: int | None = Field(None, ge=0)
    encryption_policy: SyncV2EncryptionPolicy = "client_private_v1"
    encryption_metadata: dict[str, Any] = Field(default_factory=dict)
    status: str | None = None
```

Then update/extend the model so `object_id`/`entity_id` and `payload`/`payload_clear` stay in sync. Add a `model_validator(mode="after")` (keep the existing `_reject_clear_private_payload` validator but make it run only for `client_private_v1`):

```python
    @model_validator(mode="after")
    def _sync_m1_aliases(self) -> "SyncV2Envelope":
        if self.object_id is None and self.entity_id is not None:
            object.__setattr__(self, "object_id", self.entity_id)
        if self.entity_id is None and self.object_id is not None:
            object.__setattr__(self, "entity_id", self.object_id)
        if not self.payload and self.payload_clear:
            object.__setattr__(self, "payload", dict(self.payload_clear))
        if not self.payload_clear and self.payload:
            object.__setattr__(self, "payload_clear", dict(self.payload))
        return self
```

If the existing `_reject_clear_private_payload` validator unconditionally inspects `payload_clear`, guard it: `if self.encryption_policy != "client_private_v1": return self` at its top (so server_trusted cleartext notes payloads are allowed). Keep `model_config` with `extra="ignore"` (add if absent). Ensure `entity_version` is no longer force-defaulted in a way that rejects M1 envelopes — M1 envelopes may omit it.

- [ ] **Step 5: Run to confirm pass**

`.venv/bin/python -m pytest Tests/Sync_Interop/test_envelope_m1_superset.py -q` → PASS (4 tests).

- [ ] **Step 6: Guard against regressions across the subsystem**

`.venv/bin/python -m pytest Tests/Sync_Interop Tests/tldw_api -q` → PASS. If any legacy test breaks because `entity_id` is now optional or a validator changed, fix the model (not the legacy tests) so legacy envelopes still validate exactly as before. Report any legacy test that needed a genuine update.

- [ ] **Step 7: Commit**
```
git add tldw_chatbook/tldw_api/sync_schemas.py Tests/Sync_Interop/test_envelope_m1_superset.py
git commit -m "Sync v2 P2: extend SyncV2Envelope + literals to M1 superset (additive)"
```

---

## Task 2: Conform push/pull response models (additive)

**Files:** Modify `tldw_chatbook/tldw_api/sync_schemas.py` (`SyncV2PushResponse`, the accepted-entry model, `SyncV2PullResponse`). Test: `Tests/Sync_Interop/test_push_pull_m1_responses.py` (create).

- [ ] **Step 1: Write the failing test**

Create `Tests/Sync_Interop/test_push_pull_m1_responses.py`:

```python
"""P2: client parses the live M1 push/pull response shapes."""

from tldw_chatbook.tldw_api import SyncV2PushResponse, SyncV2PullResponse

LIVE_PUSH_RESPONSE = {
    "dataset_id": "ds_1",
    "server_cursor": 129,
    "accepted": [
        {
            "client_envelope_id": "c1",
            "envelope_id": "srv_env_129",
            "server_cursor": 129,
            "object_id": "note_1",
            "object_revision": 1,
            "apply_status": "applied",
        }
    ],
    "idempotent": [],
    "rejected": [],
    "conflicts": [],
    "apply_errors": [],
    "next_cursor": "129",
}

LIVE_PULL_RESPONSE = {
    "dataset_id": "ds_1",
    "from_cursor": 0,
    "next_cursor": "130",
    "has_more": False,
    "envelopes": [
        {
            "envelope_id": "srv_env_130",
            "client_envelope_id": "c2",
            "dataset_id": "ds_1",
            "server_cursor": 130,
            "domain": "notes.note",
            "operation": "upsert",
            "object_id": "note_1",
            "schema_version": 1,
            "payload": {"title": "T", "content": "B"},
            "payload_hash": "sha256:x",
            "object_revision": 1,
            "deleted": False,
            "encryption_metadata": {"policy": "server_trusted_v1"},
        }
    ],
}


def test_push_response_parses_m1_buckets():
    resp = SyncV2PushResponse.model_validate(LIVE_PUSH_RESPONSE)
    assert resp.accepted[0].object_id == "note_1"
    assert resp.accepted[0].object_revision == 1
    assert resp.accepted[0].apply_status == "applied"
    assert resp.server_cursor == 129
    assert resp.idempotent == []
    assert resp.apply_errors == []


def test_pull_response_parses_m1_envelopes():
    resp = SyncV2PullResponse.model_validate(LIVE_PULL_RESPONSE)
    assert resp.from_cursor == 0
    env = resp.envelopes[0]
    assert env.object_id == "note_1"
    assert env.payload == {"title": "T", "content": "B"}
    assert env.server_cursor == 130
```

- [ ] **Step 2: Run to confirm failure**

`.venv/bin/python -m pytest Tests/Sync_Interop/test_push_pull_m1_responses.py -q` → FAIL (no `apply_status`/`object_id` on accepted; no `server_cursor`/`idempotent`/`apply_errors`/`from_cursor`).

- [ ] **Step 3: Extend the models**

In `tldw_chatbook/tldw_api/sync_schemas.py`:

Find `SyncV2PushAcceptedEnvelope` (the accepted-entry model) and add fields:
```python
    object_id: str | None = Field(None, validation_alias=AliasChoices("object_id", "entity_id"))
    object_revision: int | None = Field(None, ge=0)
    apply_status: str | None = None
    server_cursor: int | None = Field(None, ge=0, validation_alias=AliasChoices("server_cursor", "server_sequence"))
```
(Keep existing fields; add `model_config = ConfigDict(populate_by_name=True, extra="ignore")` if not present.)

Add an apply-error entry model near it:
```python
class SyncV2ApplyError(BaseModel):
    client_envelope_id: str | None = None
    object_id: str | None = None
    domain: str | None = None
    error_code: str | None = None
    message: str | None = None
    model_config = ConfigDict(extra="ignore")
```

Extend `SyncV2PushResponse`:
```python
    server_cursor: int | None = Field(None, ge=0)
    idempotent: list[SyncV2PushAcceptedEnvelope] = Field(default_factory=list)
    apply_errors: list[SyncV2ApplyError] = Field(default_factory=list)
```
(Keep `dataset_id`, `accepted`, `rejected`, `conflicts`, `next_cursor`; ensure `model_config` has `extra="ignore"`.)

Extend `SyncV2PullResponse`:
```python
    from_cursor: int | None = Field(None, ge=0)
```
(Keep the rest; `extra="ignore"`.)

Export `SyncV2ApplyError` from `tldw_chatbook/tldw_api/__init__.py` (import block + `__all__`).

- [ ] **Step 4: Run to confirm pass**

`.venv/bin/python -m pytest Tests/Sync_Interop/test_push_pull_m1_responses.py -q` → PASS.

- [ ] **Step 5: Regression guard + commit**

`.venv/bin/python -m pytest Tests/Sync_Interop Tests/tldw_api -q` → PASS.
```
git add tldw_chatbook/tldw_api/sync_schemas.py tldw_chatbook/tldw_api/__init__.py Tests/Sync_Interop/test_push_pull_m1_responses.py
git commit -m "Sync v2 P2: extend push/pull response models to M1 (idempotent/apply_errors/cursor)"
```

---

## Task 3: Canonical payload-hash helper

**Files:** Create `tldw_chatbook/Sync_Interop/hashing.py`. Test: `Tests/Sync_Interop/test_canonical_hash.py`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Sync_Interop/test_canonical_hash.py`:

```python
from tldw_chatbook.Sync_Interop.hashing import canonical_payload_hash, HASH_VERSION


def test_hash_is_deterministic_and_key_order_independent():
    a = canonical_payload_hash({"title": "T", "content": "B"})
    b = canonical_payload_hash({"content": "B", "title": "T"})
    assert a == b
    assert a.startswith("sha256:")


def test_hash_changes_with_content():
    assert canonical_payload_hash({"title": "T"}) != canonical_payload_hash({"title": "U"})


def test_hash_version_constant_exists():
    assert isinstance(HASH_VERSION, int) and HASH_VERSION >= 1
```

- [ ] **Step 2: Run to confirm failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `tldw_chatbook/Sync_Interop/hashing.py`:

```python
"""Canonical, versioned payload hashing shared by Sync v2 clients.

The server stores ``object_hash = payload_hash`` verbatim, so single-client push is
safe with any deterministic hash. This canonical form exists for cross-client parity
(chat.message dedupe, restore/preview local-inventory comparison): all chatbook
clients must hash identical payloads to identical digests.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

HASH_VERSION = 1


def canonical_payload_hash(payload: Mapping[str, Any]) -> str:
    """Return ``sha256:<hex>`` over the canonical JSON encoding of ``payload``.

    Canonical form: UTF-8 JSON with sorted keys and compact separators.
    """
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"
```

- [ ] **Step 4: Run to confirm pass** — `.venv/bin/python -m pytest Tests/Sync_Interop/test_canonical_hash.py -q` → PASS.

- [ ] **Step 5: Commit**
```
git add tldw_chatbook/Sync_Interop/hashing.py Tests/Sync_Interop/test_canonical_hash.py
git commit -m "Sync v2 P2: add versioned canonical payload-hash helper"
```

---

## Task 4: Per-object notes mirror

**Files:** Create `tldw_chatbook/Sync_Interop/notes_mirror.py`. Test: `Tests/Sync_Interop/test_notes_mirror.py`.

The mirror tracks, per `object_id`, the last server-acknowledged `object_revision`, `object_hash` (= payload_hash), and `server_cursor`, so the builder can populate base metadata for updates/tombstones and the applier can detect re-applies. Focused SQLite-backed store (`:memory:` or file).

- [ ] **Step 1: Write the failing test**

Create `Tests/Sync_Interop/test_notes_mirror.py`:

```python
from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror, MirrorRecord


def test_mirror_upsert_and_get():
    m = NotesMirror(":memory:")
    assert m.get("ds", "note_1") is None
    m.record("ds", "note_1", object_revision=1, object_hash="sha256:a", server_cursor=10)
    rec = m.get("ds", "note_1")
    assert isinstance(rec, MirrorRecord)
    assert rec.object_revision == 1
    assert rec.object_hash == "sha256:a"
    assert rec.server_cursor == 10


def test_mirror_record_is_idempotent_upsert():
    m = NotesMirror(":memory:")
    m.record("ds", "n", object_revision=1, object_hash="sha256:a", server_cursor=10)
    m.record("ds", "n", object_revision=2, object_hash="sha256:b", server_cursor=11)
    rec = m.get("ds", "n")
    assert rec.object_revision == 2 and rec.object_hash == "sha256:b" and rec.server_cursor == 11


def test_mirror_scopes_by_dataset():
    m = NotesMirror(":memory:")
    m.record("ds1", "n", object_revision=1, object_hash="h", server_cursor=1)
    assert m.get("ds2", "n") is None
```

- [ ] **Step 2: Run to confirm failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `tldw_chatbook/Sync_Interop/notes_mirror.py`:

```python
"""Per-object Sync v2 mirror for whole-object domains (notes.note in P2).

Stores the last server-acknowledged revision/hash/cursor per object so the builder
can fill base_object_revision/base_object_hash on updates and tombstones, and the
applier can recognise already-applied envelopes.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MirrorRecord:
    object_revision: int
    object_hash: str
    server_cursor: int


class NotesMirror:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes_object_mirror (
                dataset_id TEXT NOT NULL,
                object_id TEXT NOT NULL,
                object_revision INTEGER NOT NULL,
                object_hash TEXT NOT NULL,
                server_cursor INTEGER NOT NULL,
                PRIMARY KEY (dataset_id, object_id)
            )
            """
        )
        self._conn.commit()

    def record(
        self,
        dataset_id: str,
        object_id: str,
        *,
        object_revision: int,
        object_hash: str,
        server_cursor: int,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO notes_object_mirror
                (dataset_id, object_id, object_revision, object_hash, server_cursor)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_id, object_id) DO UPDATE SET
                object_revision=excluded.object_revision,
                object_hash=excluded.object_hash,
                server_cursor=excluded.server_cursor
            """,
            (dataset_id, object_id, object_revision, object_hash, server_cursor),
        )
        self._conn.commit()

    def get(self, dataset_id: str, object_id: str) -> MirrorRecord | None:
        row = self._conn.execute(
            "SELECT object_revision, object_hash, server_cursor FROM notes_object_mirror "
            "WHERE dataset_id=? AND object_id=?",
            (dataset_id, object_id),
        ).fetchone()
        if row is None:
            return None
        return MirrorRecord(
            object_revision=row["object_revision"],
            object_hash=row["object_hash"],
            server_cursor=row["server_cursor"],
        )

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run to confirm pass** — `.venv/bin/python -m pytest Tests/Sync_Interop/test_notes_mirror.py -q` → PASS.

- [ ] **Step 5: Commit**
```
git add tldw_chatbook/Sync_Interop/notes_mirror.py Tests/Sync_Interop/test_notes_mirror.py
git commit -m "Sync v2 P2: add per-object notes mirror (revision/hash/cursor)"
```

---

## Task 5: Notes local store + M1 build/apply adapters

**Files:** Create `tldw_chatbook/Sync_Interop/notes_local_store.py`; add builder methods to `tldw_chatbook/Sync_Interop/envelope_builder.py`; create `tldw_chatbook/Sync_Interop/domain_adapters/notes_m1.py`; wire it in `envelope_applier.py`. Test: `Tests/Sync_Interop/test_notes_m1_adapters.py`.

- [ ] **Step 1: Write the failing test**

Create `Tests/Sync_Interop/test_notes_m1_adapters.py`:

```python
from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier


def _builder(mirror):
    return SyncEnvelopeBuilder(
        dataset_id="ds_1", device_id="dev_1", dataset_key=b"x" * 32, notes_mirror=mirror,
    )


def test_build_notes_upsert_new_object_has_no_base():
    mirror = NotesMirror(":memory:")
    env = _builder(mirror).build_notes_note_upsert(note_id="n1", title="T", content="B")
    assert env.domain == "notes.note"
    assert env.operation == "upsert"
    assert env.object_id == "n1"
    assert env.payload == {"title": "T", "content": "B"}
    assert env.encryption_policy == "server_trusted_v1"
    assert env.encryption_metadata == {"policy": "server_trusted_v1"}
    assert env.base_object_revision is None and env.base_object_hash is None


def test_build_notes_upsert_update_uses_mirror_base():
    mirror = NotesMirror(":memory:")
    mirror.record("ds_1", "n1", object_revision=1, object_hash="sha256:prev", server_cursor=5)
    env = _builder(mirror).build_notes_note_upsert(note_id="n1", title="T2", content="B2")
    assert env.base_object_revision == 1
    assert env.base_object_hash == "sha256:prev"
    assert env.base_server_cursor == 5


def test_build_notes_tombstone_requires_mirror_base():
    mirror = NotesMirror(":memory:")
    mirror.record("ds_1", "n1", object_revision=2, object_hash="sha256:p", server_cursor=9)
    env = _builder(mirror).build_notes_note_tombstone(note_id="n1")
    assert env.operation == "tombstone" and env.deleted is True
    assert env.base_object_revision == 2 and env.base_object_hash == "sha256:p"


def test_apply_notes_upsert_creates_local_note_and_updates_mirror():
    store = InMemoryNotesStore()
    mirror = NotesMirror(":memory:")
    applier = SyncEnvelopeApplier(local_store=store, notes_mirror=mirror, dataset_id="ds_1")
    # Simulate a pulled server envelope.
    from tldw_chatbook.tldw_api import SyncV2Envelope
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="upsert", adapter_version=1, payload={"title": "T", "content": "B"},
        payload_hash="sha256:cur", object_revision=1, server_cursor=10,
        encryption_metadata={"policy": "server_trusted_v1"},
    )
    result = applier.apply(env)
    assert result["status"] == "applied"
    assert store.get("n1") == {"title": "T", "content": "B", "deleted": False}
    assert mirror.get("ds_1", "n1").object_revision == 1

    # Re-applying the same envelope is a verified no-op.
    result2 = applier.apply(env)
    assert result2["status"] in {"applied", "noop"}
    assert store.upsert_calls == 1  # not applied twice


def test_apply_notes_tombstone_soft_deletes():
    store = InMemoryNotesStore()
    store.upsert_note("n1", {"title": "T", "content": "B"}, object_revision=1)
    mirror = NotesMirror(":memory:")
    applier = SyncEnvelopeApplier(local_store=store, notes_mirror=mirror, dataset_id="ds_1")
    from tldw_chatbook.tldw_api import SyncV2Envelope
    env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n1",
        operation="tombstone", adapter_version=1, deleted=True, payload={"deleted_at": "t"},
        payload_hash="sha256:t", object_revision=2, server_cursor=11,
    )
    applier.apply(env)
    assert store.get("n1")["deleted"] is True
```

- [ ] **Step 2: Run to confirm failure** — import + signature errors.

- [ ] **Step 3: Implement the local store**

Create `tldw_chatbook/Sync_Interop/notes_local_store.py`:

```python
"""Local notes store interface for Sync v2 apply, plus an in-memory implementation.

The protocol is the seam the apply adapter writes through. P2 ships an in-memory
implementation for round-trip verification; a ChaChaNotes-backed implementation is a
later integration step.
"""
from __future__ import annotations

from typing import Any, Protocol


class NotesSyncLocalStore(Protocol):
    def upsert_note(self, object_id: str, payload: dict[str, Any], *, object_revision: int) -> None: ...
    def soft_delete_note(self, object_id: str, *, object_revision: int) -> None: ...
    def get(self, object_id: str) -> dict[str, Any] | None: ...


class InMemoryNotesStore:
    """Minimal in-memory notes store implementing NotesSyncLocalStore."""

    def __init__(self) -> None:
        self._notes: dict[str, dict[str, Any]] = {}
        self.upsert_calls = 0
        self.delete_calls = 0

    def upsert_note(self, object_id: str, payload: dict[str, Any], *, object_revision: int) -> None:
        self.upsert_calls += 1
        record = {"title": payload.get("title", ""), "content": payload.get("content", payload.get("body", "")), "deleted": False}
        self._notes[object_id] = record

    def soft_delete_note(self, object_id: str, *, object_revision: int) -> None:
        self.delete_calls += 1
        existing = self._notes.get(object_id, {"title": "", "content": ""})
        existing["deleted"] = True
        self._notes[object_id] = existing

    def get(self, object_id: str) -> dict[str, Any] | None:
        return self._notes.get(object_id)
```

- [ ] **Step 4: Add builder methods**

In `tldw_chatbook/Sync_Interop/envelope_builder.py`: add an optional `notes_mirror` param to `__init__` (default `None`) stored as `self.notes_mirror`, import `canonical_payload_hash` from `tldw_chatbook.Sync_Interop.hashing`, and add:

```python
    def build_notes_note_upsert(self, *, note_id: str, title: str, content: str) -> SyncV2Envelope:
        payload = {"title": title, "content": content}
        return self._notes_note_envelope(note_id=note_id, operation="upsert", payload=payload, deleted=False)

    def build_notes_note_tombstone(self, *, note_id: str, deleted_at: str | None = None) -> SyncV2Envelope:
        payload = {"deleted_at": deleted_at or "", "reason": "user_deleted"}
        return self._notes_note_envelope(note_id=note_id, operation="tombstone", payload=payload, deleted=True)

    def _notes_note_envelope(self, *, note_id, operation, payload, deleted) -> SyncV2Envelope:
        payload_hash = canonical_payload_hash(payload)
        base = self.notes_mirror.get(self.dataset_id, note_id) if self.notes_mirror is not None else None
        return SyncV2Envelope(
            client_envelope_id=f"{self.device_id}:notes.note:{note_id}:{payload_hash}",
            dataset_id=self.dataset_id,
            device_id=self.device_id,
            domain="notes.note",
            object_id=note_id,
            operation=operation,
            adapter_version=self.adapter_version,
            schema_version=1,
            deleted=deleted,
            payload=payload,
            payload_hash=payload_hash,
            base_object_revision=base.object_revision if base else None,
            base_object_hash=base.object_hash if base else None,
            base_server_cursor=base.server_cursor if base else None,
            encryption_policy="server_trusted_v1",
            encryption_metadata={"policy": "server_trusted_v1"},
        )
```

- [ ] **Step 5: Implement the M1 notes apply adapter + wire the applier**

Create `tldw_chatbook/Sync_Interop/domain_adapters/notes_m1.py`:

```python
"""Server-trusted M1 apply adapter for notes.note (cleartext, idempotent)."""
from __future__ import annotations

from typing import Any, Callable

from tldw_chatbook.tldw_api import SyncV2Envelope


class NotesM1SyncAdapter:
    def apply(
        self,
        envelope: SyncV2Envelope,
        *,
        local_store: Any,
        notes_mirror: Any,
        dataset_id: str,
        record_conflict: Callable[..., dict[str, Any]],
    ) -> dict[str, Any]:
        object_id = envelope.object_id or envelope.entity_id
        cursor = envelope.server_cursor or 0
        revision = envelope.object_revision or 0
        # Idempotent re-apply: mirror already at this revision+hash.
        existing = notes_mirror.get(dataset_id, object_id)
        if existing is not None and existing.object_revision == revision and existing.object_hash == envelope.payload_hash:
            return {"status": "noop", "object_id": object_id}

        if envelope.operation == "tombstone" or envelope.deleted:
            local_store.soft_delete_note(object_id, object_revision=revision)
        else:
            local_store.upsert_note(object_id, dict(envelope.payload), object_revision=revision)
        notes_mirror.record(
            dataset_id, object_id,
            object_revision=revision, object_hash=envelope.payload_hash, server_cursor=cursor,
        )
        return {"status": "applied", "object_id": object_id}
```

In `tldw_chatbook/Sync_Interop/envelope_applier.py`, extend `SyncEnvelopeApplier.__init__` to accept `notes_mirror=None` and `dataset_id=None` (keep `dataset_key` optional with a default of `None`), register the M1 adapter, and route dotted domains. Add to `_adapters`: `"notes.note": NotesM1SyncAdapter()` (import it). In `apply`, when the adapter is `NotesM1SyncAdapter`, call it with `local_store=self.local_store, notes_mirror=self.notes_mirror, dataset_id=self.dataset_id, record_conflict=self._record_conflict` (the legacy adapters keep their existing `dataset_key=...` call signature). Keep the unsupported-domain conflict path for unknown domains. Do not break the legacy coarse-domain routing.

- [ ] **Step 6: Run to confirm pass** — `.venv/bin/python -m pytest Tests/Sync_Interop/test_notes_m1_adapters.py -q` → PASS (5 tests).

- [ ] **Step 7: Regression guard + commit**

`.venv/bin/python -m pytest Tests/Sync_Interop Tests/tldw_api -q` → PASS.
```
git add tldw_chatbook/Sync_Interop/notes_local_store.py tldw_chatbook/Sync_Interop/envelope_builder.py tldw_chatbook/Sync_Interop/domain_adapters/notes_m1.py tldw_chatbook/Sync_Interop/envelope_applier.py Tests/Sync_Interop/test_notes_m1_adapters.py
git commit -m "Sync v2 P2: notes.note M1 build + apply adapters + local store"
```

---

## Task 6: `NotesM1SyncFlow` orchestration

**Files:** Create `tldw_chatbook/Sync_Interop/notes_m1_flow.py`. Test: `Tests/Sync_Interop/test_notes_m1_flow.py` (uses a mocked client).

The flow ties the pieces together: push a list of built envelopes, process the response (accepted → update mirror with returned `object_revision`/`server_cursor`; idempotent → leave; apply_errors/rejected/conflicts → surface), then pull from a cursor and apply each envelope.

- [ ] **Step 1: Write the failing test**

Create `Tests/Sync_Interop/test_notes_m1_flow.py`:

```python
import pytest
from unittest.mock import AsyncMock

from tldw_chatbook.Sync_Interop.notes_m1_flow import NotesM1SyncFlow
from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror
from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder


@pytest.mark.asyncio
async def test_push_updates_mirror_from_accepted(monkeypatch):
    mirror = NotesMirror(":memory:")
    store = InMemoryNotesStore()
    builder = SyncEnvelopeBuilder(dataset_id="ds_1", device_id="dev_1", dataset_key=b"x"*32, notes_mirror=mirror)
    client = AsyncMock()
    client.push_sync_v2_envelopes = AsyncMock(return_value=type("R", (), {
        "accepted": [type("A", (), {"client_envelope_id": "x", "object_id": "n1", "object_revision": 1, "server_cursor": 7, "apply_status": "applied"})()],
        "idempotent": [], "rejected": [], "conflicts": [], "apply_errors": [], "next_cursor": "7",
    })())
    flow = NotesM1SyncFlow(client=client, builder=builder, mirror=mirror, local_store=store, dataset_id="ds_1", device_id="dev_1")

    env = builder.build_notes_note_upsert(note_id="n1", title="T", content="B")
    result = await flow.push([env])

    assert result["accepted"] == 1
    assert mirror.get("ds_1", "n1").object_revision == 1
    assert mirror.get("ds_1", "n1").server_cursor == 7


@pytest.mark.asyncio
async def test_pull_applies_and_advances_cursor(monkeypatch):
    from tldw_chatbook.tldw_api import SyncV2Envelope
    mirror = NotesMirror(":memory:")
    store = InMemoryNotesStore()
    builder = SyncEnvelopeBuilder(dataset_id="ds_1", device_id="dev_2", dataset_key=b"x"*32, notes_mirror=mirror)
    client = AsyncMock()
    pulled_env = SyncV2Envelope(
        client_envelope_id="c", dataset_id="ds_1", domain="notes.note", object_id="n9",
        operation="upsert", adapter_version=1, payload={"title": "T", "content": "B"},
        payload_hash="sha256:cur", object_revision=1, server_cursor=12,
    )
    client.pull_sync_v2_envelopes = AsyncMock(return_value=type("P", (), {
        "envelopes": [pulled_env], "next_cursor": "12", "has_more": False,
    })())
    flow = NotesM1SyncFlow(client=client, builder=builder, mirror=mirror, local_store=store, dataset_id="ds_1", device_id="dev_2")

    result = await flow.pull(cursor=0)

    assert result["applied"] == 1
    assert store.get("n9")["title"] == "T"
    assert result["next_cursor"] == "12"
```

- [ ] **Step 2: Run to confirm failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `tldw_chatbook/Sync_Interop/notes_m1_flow.py`:

```python
"""Focused notes.note push/pull/apply flow against an M1 Sync v2 server."""
from __future__ import annotations

from typing import Any

from tldw_chatbook.Sync_Interop.envelope_applier import SyncEnvelopeApplier
from tldw_chatbook.tldw_api import SyncV2Envelope, SyncV2PushRequest


class NotesM1SyncFlow:
    def __init__(self, *, client, builder, mirror, local_store, dataset_id: str, device_id: str) -> None:
        self.client = client
        self.builder = builder
        self.mirror = mirror
        self.local_store = local_store
        self.dataset_id = dataset_id
        self.device_id = device_id
        self._client_sequence = 0

    async def push(self, envelopes: list[SyncV2Envelope]) -> dict[str, Any]:
        for env in envelopes:
            self._client_sequence += 1
            env.client_sequence = self._client_sequence
        request = SyncV2PushRequest(dataset_id=self.dataset_id, device_id=self.device_id, envelopes=envelopes)
        response = await self.client.push_sync_v2_envelopes(request)
        for accepted in response.accepted:
            oid = accepted.object_id
            if oid is None:
                continue
            self.mirror.record(
                self.dataset_id, oid,
                object_revision=accepted.object_revision or 0,
                object_hash=self._hash_for(envelopes, accepted.client_envelope_id),
                server_cursor=accepted.server_cursor or 0,
            )
        return {
            "accepted": len(response.accepted),
            "idempotent": len(getattr(response, "idempotent", []) or []),
            "rejected": len(response.rejected),
            "conflicts": len(response.conflicts),
            "apply_errors": len(getattr(response, "apply_errors", []) or []),
            "next_cursor": response.next_cursor,
        }

    async def pull(self, *, cursor: int) -> dict[str, Any]:
        response = await self.client.pull_sync_v2_envelopes(
            dataset_id=self.dataset_id, device_id=self.device_id, cursor=str(cursor), domains=["notes.note"],
        )
        applier = SyncEnvelopeApplier(local_store=self.local_store, notes_mirror=self.mirror, dataset_id=self.dataset_id)
        applied = 0
        for env in response.envelopes:
            result = applier.apply(env)
            if result.get("status") == "applied":
                applied += 1
        return {"applied": applied, "next_cursor": response.next_cursor, "has_more": response.has_more}

    @staticmethod
    def _hash_for(envelopes: list[SyncV2Envelope], client_envelope_id: str) -> str:
        for env in envelopes:
            if env.client_envelope_id == client_envelope_id:
                return env.payload_hash
        return ""
```

Note: `pull_sync_v2_envelopes` currently takes a `cursor` kwarg — confirm its signature in `client.py` and match it (it accepts `cursor: str | None`). If the client's pull does not yet pass `cursor`/`domain` through to the request correctly for the M1 server, adjust the client method minimally and note it.

- [ ] **Step 4: Run to confirm pass** — `.venv/bin/python -m pytest Tests/Sync_Interop/test_notes_m1_flow.py -q` → PASS.

- [ ] **Step 5: Regression guard + commit**

`.venv/bin/python -m pytest Tests/Sync_Interop Tests/tldw_api -q` → PASS.
```
git add tldw_chatbook/Sync_Interop/notes_m1_flow.py Tests/Sync_Interop/test_notes_m1_flow.py
git commit -m "Sync v2 P2: NotesM1SyncFlow push/pull/apply orchestration"
```

---

## Task 7: Live round-trip exit gate (controller-run)

**Files:** Create `/tmp/syncqa-v2/p2_roundtrip.py` (not committed). This is run by the controller, not a subagent — it needs the live server.

- [ ] **Step 1: Ensure the server is running** with attestation (see header launch block); confirm capabilities curl returns `sync-v2-m1`.

- [ ] **Step 2: Write the round-trip script** `/tmp/syncqa-v2/p2_roundtrip.py`:

```python
import asyncio, os, sys, uuid
from tldw_chatbook.tldw_api.client import TLDWAPIClient
from tldw_chatbook.tldw_api import SyncV2ProfileBootstrapRequest
from tldw_chatbook.Sync_Interop.notes_mirror import NotesMirror
from tldw_chatbook.Sync_Interop.notes_local_store import InMemoryNotesStore
from tldw_chatbook.Sync_Interop.envelope_builder import SyncEnvelopeBuilder
from tldw_chatbook.Sync_Interop.notes_m1_flow import NotesM1SyncFlow

BASE, KEY = "http://127.0.0.1:8076", os.environ["SINGLE_USER_API_KEY"]


async def main():
    client = TLDWAPIClient(base_url=BASE, token=KEY)
    client.bearer_token = KEY
    boot = await client.bootstrap_sync_v2_profile(SyncV2ProfileBootstrapRequest(device_name="p2-A", mode="offline_sync"))
    dataset_id, device_a = boot.dataset.dataset_id, boot.device.device_id

    mirror_a = NotesMirror(":memory:")
    builder_a = SyncEnvelopeBuilder(dataset_id=dataset_id, device_id=device_a, dataset_key=b"x"*32, notes_mirror=mirror_a)
    flow_a = NotesM1SyncFlow(client=client, builder=builder_a, mirror=mirror_a, local_store=InMemoryNotesStore(), dataset_id=dataset_id, device_id=device_a)

    note_id = f"note_{uuid.uuid4().hex[:8]}"
    up = builder_a.build_notes_note_upsert(note_id=note_id, title="Trip notes", content="Outline.")
    r1 = await flow_a.push([up])
    print("PUSH upsert:", r1)
    assert r1["accepted"] == 1 and r1["apply_errors"] == 0

    # Pull onto a second device (device_b) and confirm it materialises.
    boot_b = await client.bootstrap_sync_v2_profile(SyncV2ProfileBootstrapRequest(device_name="p2-B", mode="offline_sync", device_id=None))
    device_b = boot_b.device.device_id
    mirror_b, store_b = NotesMirror(":memory:"), InMemoryNotesStore()
    builder_b = SyncEnvelopeBuilder(dataset_id=dataset_id, device_id=device_b, dataset_key=b"x"*32, notes_mirror=mirror_b)
    flow_b = NotesM1SyncFlow(client=client, builder=builder_b, mirror=mirror_b, local_store=store_b, dataset_id=dataset_id, device_id=device_b)
    r2 = await flow_b.pull(cursor=0)
    print("PULL onto B:", r2, "| note:", store_b.get(note_id))
    assert store_b.get(note_id) and store_b.get(note_id)["title"] == "Trip notes"

    # Re-apply is a no-op.
    r2b = await flow_b.pull(cursor=0)
    print("RE-PULL onto B (idempotent):", r2b, "upsert_calls:", store_b.upsert_calls)

    # Update from A (revision bump uses mirror base), then tombstone.
    up2 = builder_a.build_notes_note_upsert(note_id=note_id, title="Trip notes", content="Updated outline.")
    r3 = await flow_a.push([up2]); print("PUSH update:", r3); assert r3["accepted"] == 1 and r3["conflicts"] == 0
    tomb = builder_a.build_notes_note_tombstone(note_id=note_id, deleted_at="2026-06-15T00:00:00Z")
    r4 = await flow_a.push([tomb]); print("PUSH tombstone:", r4); assert r4["accepted"] == 1

    r5 = await flow_b.pull(cursor=0)
    print("PULL tombstone onto B:", r5, "| deleted:", store_b.get(note_id) and store_b.get(note_id)["deleted"])
    assert store_b.get(note_id)["deleted"] is True

    await client.close()
    print("\nP2_ROUNDTRIP_OK: True")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Run it** — `PYTHONPATH=<repo> .venv/bin/python /tmp/syncqa-v2/p2_roundtrip.py`. Expected: all asserts pass, `P2_ROUNDTRIP_OK: True`.

- [ ] **Step 4: Triage any server-shape mismatch.** If push/pull responses or the pull `cursor`/`domain` param shape differ from what the client sends, adjust the client `push_sync_v2_envelopes`/`pull_sync_v2_envelopes` request construction minimally to match the live server, add/adjust a unit test in `Tests/Sync_Interop/test_push_pull_m1_responses.py`, and re-run. Capture the real server payloads into the unit tests so the contract is pinned. Commit any such client fix with the other P2 work.

- [ ] **Step 5: Record the result** on backlog task #24:
```
backlog task edit 24 --notes "P2 GREEN: notes.note upsert->push->pull(device B)->update->tombstone round-trip materialises live vs :8076 with apply_status==applied and idempotent re-apply. Harness: /tmp/syncqa-v2/p2_roundtrip.py"
```

---

## Self-Review (completed at authoring time)

- **Spec coverage:** P2 spec items — build+apply adapters (Task 5), per-object mirror (Task 4), push/pull/tombstone round-trip asserting `apply_status==applied` + re-apply no-op (Tasks 6-7), canonical hashing spec defined/exercised (Tasks 3,5,7). Envelope/response schema conformance needed to carry M1 data is Tasks 1-2.
- **Deferred (noted):** full `LocalFirstSyncService`/`ManualSyncControlService`/outbox rewire to M1 and the ChaChaNotes-backed `NotesSyncLocalStore` are a later integration phase; P2 proves the vertical with `InMemoryNotesStore`. The `server_sync_service.py` `supported_domains` fallback cleanup (P1 carryover) and the `pull_sync_v2_envelopes` None-param issue should be folded in when Task 6 touches the client pull method.
- **Placeholder scan:** every code/edit step shows concrete content.
- **Type consistency:** `NotesMirror`/`MirrorRecord`, `NotesM1SyncAdapter`, `NotesM1SyncFlow`, `InMemoryNotesStore`, `canonical_payload_hash`/`HASH_VERSION`, builder `build_notes_note_upsert`/`build_notes_note_tombstone`, applier `SyncEnvelopeApplier(local_store=, notes_mirror=, dataset_id=)` used consistently across tasks.
- **Additive-safety:** Tasks 1,2,5 each end with a `Tests/Sync_Interop Tests/tldw_api` regression guard so the legacy client_private path stays green.
