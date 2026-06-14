# Sync v2 Client — M1 Contract Conformance Design

Date: 2026-06-12
Status: Draft (pending user spec review)
ADR: `backlog/decisions/008-sync-v2-client-m1-contract-alignment.md`
Server contract: `tldw_server2` `Docs/API/Sync_V2_M1.md` @ commit `992e89a037244e5e5cdf58ad47245c89866e373f`

## Purpose

Conform the Chatbook Sync v2 client to the server's locked M1 contract so that manual
Notes + Chat sync round-trips end-to-end against a real server, closing backlog task #24
(TASK-70.5/70.6). Per ADR 008: server contract is canonical, encryption posture is
`server_trusted_v1`, `client_private_v1`/recovery is parked for M3.

## Background: what was verified

The live server (`codex/sync-v2-m1-next`, launched on `:8076` with
`SINGLE_USER_API_KEY`) was driven with the real `TLDWAPIClient`. Findings:

- The client cannot parse `GET /api/v1/sync/capabilities` — `protocol_version` is
  string `"sync-v2-m1"` (client expects `int`), and `encryption_policies` is
  `["server_trusted_v1"]` (client literal lacks it).
- The server's `SyncV2Envelope` carries a transition compat layer
  (`_with_transition_aliases`: `entity_id`→`object_id`, `server_sequence`→`server_cursor`,
  `payload_clear`→`payload`, `client_timestamp`→`created_at_client`) and tolerates int
  `2` protocol versions, `extra="ignore"`.
- The server **hard-rejects** (no alias): any encryption policy ≠ `server_trusted_v1`,
  any operation ∉ `{upsert, append, tombstone}`, missing `object_id`, and incomplete
  base metadata on whole-object updates/tombstones.

Conclusion: the conformance work is **not** an envelope-identity rebuild. It is
(a) widen client response parsing, (b) flip encryption to server-trusted cleartext,
(c) remap domain/operation vocabulary on **both** the build and apply sides,
(d) add base-revision/identity bookkeeping, (e) conform the pull/apply path
(`envelope_applier`: dotted routing, no-decryption, idempotent object_id→local-row
apply), then (f) per-domain QA asserting materialization (`apply_status == applied`),
not just acceptance. Canonical field names are emitted (not relying on the temporary
aliases).

## Scope

In scope (M1 domains): `notes.note`, `chat.conversation`, `chat.message`,
`attachment.ref` (metadata only — no blob transfer). Endpoint flow:
`/profile/bootstrap` → `/push` → `/pull` → `/restore/preview` → `/conflicts/resolve`
→ `/repair`.

Deferred / parked: `client_private_v1` encryption + key recovery (M3 gate);
`workspaces.*`, `source_cache.*`, `media.*` domains and their adapters; binary blob
transfer; the `/attachments` ciphertext-upload client method.

## Component design

### 1. `tldw_chatbook/tldw_api/sync_schemas.py` — schema conformance
- `SyncV2CapabilitiesResponse`: `protocol_version` / `min_supported_protocol_version`
  → `str` (default `"sync-v2-m1"`); accept the server's full capability shape
  (`encryption`, `blob_transfer`, `max_*`, `supports_*`, `compatibility_flags`,
  `warnings`).
- `SyncV2Domain` → dotted literal matching the server's `SyncDomain`.
- `SyncV2Operation` → `Literal["upsert", "append", "tombstone"]`.
- `SyncV2EncryptionPolicy` → include `server_trusted_v1` (and the strict M3 policies as
  accepted-but-unused values).
- `SyncV2Envelope`: emit canonical M1 fields — `object_id`, `parent_id`,
  `schema_version`, `payload` (cleartext), `payload_hash`, `object_revision`,
  `base_server_cursor`, `base_object_revision`, `base_object_hash`, `client_sequence`,
  `created_at_client`, `deleted`, `encryption_metadata.policy = "server_trusted_v1"`.
  Retain `client_envelope_id`, `dataset_id`, `device_id`. Condition the
  `client_private_v1`-only clear-payload rejection validator on policy (keep it for the
  parked M3 path; it is a no-op for `server_trusted_v1`), rather than deleting it.
- Add request/response models for `/profile`, `/profile/bootstrap`, `/restore/preview`,
  `/repair` matching the server. Conflict actions → `overwrite/duplicate_rename/skip`.

### 2. `tldw_chatbook/tldw_api/client.py` — transport
- Fix `get_sync_v2_capabilities` (parse string protocol).
- Add `bootstrap_sync_v2_profile` / `get_sync_v2_profile` (the durable M1 flow).
- `push_sync_v2_envelopes` / `pull_sync_v2_envelopes`: align request/response to the
  cursor model (`base_server_cursor`, `cursor`, `next_cursor`, `from_cursor`,
  `has_more`, `accepted/idempotent/rejected/conflicts/apply_errors`).
- Add `restore_preview_sync_v2`, `repair_sync_v2`, conflicts list/resolve to M1 shapes.
- Park behind M3 gate: `register_sync_v2_device`, `enroll_sync_v2_dataset`,
  recovery-bundle and attachment-ciphertext-upload methods (kept, not on M1 path).

### 3. `tldw_chatbook/Sync_Interop/envelope_builder.py` + build-side `domain_adapters/*` (push)
- Emit dotted domains; split `chat` producer into `chat.conversation` (upsert/tombstone)
  and `chat.message` (append/tombstone, dedupe by `object_id`+`payload_hash`).
- Replace `_encrypted_envelope` with a `_server_trusted_envelope` builder: cleartext
  `payload` per the M1 per-domain payload schemas (notes: title/body/tags/updated_at;
  conversation: title/model/character_id/updated_at; message:
  conversation_id/role/sender/content/created_at; attachment.ref: the 7 required keys).
  Note: only `attachment.ref` payload keys are validated at the *envelope* level; the
  notes/conversation/message payload shape is enforced by the server **materializer**
  (apply step), so shape bugs surface as `apply_errors`, not push rejections.
- `delete` → `tombstone` (payload `deleted_at`/`reason`, `deleted: true`).
- `payload_hash` / canonical hashing: the server stores `object_hash = envelope.payload_hash`
  verbatim and checks `base_object_hash == head.payload_hash` (verified in server
  `v2/_lineage.py` + materializers). So a single client only needs to **store the
  payload_hash it sent** and echo it as the next `base_object_hash` — no server-algorithm
  matching. A **canonical, versioned JSON hashing spec** (sorted keys, compact separators,
  a `hash_version`) is required only for cross-client parity: `chat.message` dedupe across
  devices and `restore/preview` `local_inventory` `object_hash` comparison. Define it once
  and reuse it on both build and inventory sides.
- Park `media`/`workspaces`/`source_cache` build adapters (M3+).

### 3b. `tldw_chatbook/Sync_Interop/envelope_applier.py` + apply-side adapters (pull)
The apply path is half the round-trip and currently coarse-keyed; it must conform too:
- `SyncEnvelopeApplier` routes by `envelope.domain` with adapters keyed `"notes"`/`"chat"`
  and takes a `dataset_key: bytes` for client-private decryption. Re-key to dotted domains
  (`notes.note`, `chat.conversation`, `chat.message`, `attachment.ref`), split `chat` into
  conversation vs message apply, and add a **no-decryption cleartext path** for
  `server_trusted_v1` (the `dataset_key` becomes M3-gated, not required for M1).
- **object_id ↔ local-row mapping** (the hardest correctness surface): idempotent
  upsert keyed by `object_id`, create-if-missing for objects originating on another
  device, and re-apply safety (applying the same envelope twice is a no-op). Persist the
  mapping in the per-object mirror.
- **tombstone → local soft-delete** (ChaChaNotes soft-deletion), not hard delete.
- **Ordering:** a pulled `chat.message` may arrive before its parent `chat.conversation`
  exists locally; apply must tolerate out-of-order parents (create a placeholder
  conversation or defer/re-queue) rather than dropping the message.

### 4. Orchestration — cursor/revision bookkeeping
- `sync_state_repository` / `sync_state`: persist
  - the **server-assigned identity**: `device_id` and `dataset_id` returned by
    `/profile/bootstrap` (bootstrap with an omitted `device_id` returns a generated one
    that MUST be saved before any push), and a per-device monotonic `client_sequence`
    counter;
  - per-dataset `server_cursor`;
  - a per-object mirror (`object_id` → last synced `server_cursor`, `object_revision`,
    `payload_hash` [≡ object_hash], local-row id) for whole-object domains, so the builder
    can populate `base_server_cursor`/`base_object_revision`/`base_object_hash` and the
    applier can map incoming envelopes to local rows.
- `ManualSyncControlService` / `LocalFirstSyncService`: drive bootstrap → push (with
  `client_sequence` + base metadata) → apply accepted (record server cursor/revision/
  hash) → pull from persisted cursor → apply to `ChaChaNotes.db` → surface conflicts.
- `conflict_review`: remap action vocabulary and copy to `overwrite/duplicate_rename/skip`.

### 5. M3 parking
- Module-level gate (e.g. `SYNC_V2_CLIENT_PRIVATE_ENABLED = False`) guarding
  `crypto.py`, `key_recovery_service.py`, and the client-private branches of
  `restore_service.py` / the recovery-bundle client method. Unit tests for those paths
  stay, marked/skipped behind the gate.

## Data flow

**Push:** local change → domain producer builds a `server_trusted_v1` cleartext envelope
with `object_id` and (for updates/tombstones on whole-object domains) base metadata from
the per-object mirror → `ManualSyncControlService` batches with `client_sequence` +
`base_server_cursor` → `POST /push` → handle all five response buckets:
`accepted` (update mirror with returned `server_cursor`/`object_revision`, store the sent
`payload_hash` as the new base), `idempotent` (already current — treat as success, no
state change), `conflicts` (record for review), `rejected` (surface error), and
`apply_errors` (envelope was accepted but server materialization failed — surface, do not
mark the object synced).

**Pull:** `GET /pull?cursor=<persisted>` → envelopes in server-cursor order → apply to
`ChaChaNotes.db` (notes/conversations/messages) and update the per-object mirror →
persist `next_cursor` (global cursor only after an unfiltered pull).

**Conflicts:** push/pull report base-state-mismatch conflicts → `conflict_review` →
`POST /conflicts/resolve` with `overwrite`/`duplicate_rename`/`skip`.

**Restore:** `POST /restore/preview` with local inventory → plan (safe_applies,
tombstones, conflicts) → apply.

## Phases (each its own PR + backlog task)

- **P0** — ADR 008 + this spec (committed; this PR).
- **P1** — Schema + transport conformance: widen response parsing, flip encryption,
  remap domain/operation literals, add bootstrap/profile. Exit: client parses
  capabilities; **`/profile/bootstrap` succeeds (not fail-closed on encryption
  attestation)** and the server-assigned `device_id`/`dataset_id` persist; unit tests
  green.
- **P2** — `notes.note` vertical: build + **apply** adapters + per-object mirror +
  push/pull/tombstone. Exit: real notes upsert→push→pull→tombstone round-trip green vs
  `:8076`, asserting **`apply_status == applied`** (not just `accepted`); canonical
  hashing spec defined and exercised; re-applying a pulled envelope is a verified no-op.
- **P3** — `chat.conversation` (upsert/tombstone) + `chat.message` (append/dedupe),
  including out-of-order parent handling on apply; assert `apply_status == applied`.
- **P4** — `attachment.ref` metadata envelopes (no blobs).
- **P5** — conflicts/resolve + restore/preview + repair.
- **P6** — Full live round-trip QA closeout (task #24); confirm `client_private_v1`/
  recovery parked behind the gate; update docs/backlog.

## Testing

- Per-phase unit tests: schema validation (round-trip parse of real server payloads),
  build-adapter payload shape, apply-adapter idempotency (re-apply is a no-op) and
  out-of-order-parent handling, cursor/mirror/identity bookkeeping, canonical-hash
  determinism.
- Live round-trip driver against `:8076` (the `/tmp/syncqa-v2/driver.py` harness),
  extended per phase.
- Update existing `Tests/` `Sync_Interop`/`tldw_api` suites to M1 shapes; gate
  client-private tests behind the M3 flag.
- Bandit on touched production code (repo DoD).

## Risks & open items

- **Moving target:** QA runs against an in-progress server branch with a temporary
  compat shim. Mitigation: pin server commit `992e89a03`; emit canonical field names so
  the client survives shim removal; verify `/profile/bootstrap` does not fail-closed on
  encryption attestation before P2 (server `_default_encryption` reports `ready: True`,
  but the real bootstrap path must be confirmed).
- **Open decision (recorded, adjustable):** migrate to `/profile/bootstrap` vs keep the
  current `/datasets/enroll`+`/devices/register` flow. Recommendation: `/profile/bootstrap`
  (durable M1 flow); the enroll/register endpoints may be transition-only.
- **chat.message edits:** M1 has only `append`/`tombstone` for messages. Editing/
  regenerating a message must be modeled as tombstone-old + append-new (or deferred);
  decide in P3.
- **cross-client hash parity:** `object_hash ≡ payload_hash` and the server stores the
  client's hash verbatim, so single-client push is safe once the client stores its own
  sent hash. Parity bites only across clients — `chat.message` dedupe and
  `restore/preview` `local_inventory` comparison — so the canonical hashing spec must be
  versioned and shared. Confirm empirically in P2/P3.
- **apply path is half the work:** `envelope_applier.py` + apply-side adapters
  (idempotent object_id→local-row upsert, tombstone soft-delete, out-of-order parents)
  are as much of the effort as the build side; `accepted` ≠ `applied`.
