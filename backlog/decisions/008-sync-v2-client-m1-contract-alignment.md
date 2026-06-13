# ADR 008 - Align the Chatbook Sync v2 client to the server's locked M1 contract

Date: 2026-06-12
Status: Accepted

## Context

Actual-use QA of the Chatbook Sync v2 manual-sync milestone (TASK-70.5/70.6, backlog
task #24) was blocked on the assumption that no `tldw_server2` build implemented the
Sync v2 API. That assumption is false. The worktree branch `codex/sync-v2-m1-next`
(`tldw_server2`, pinned at commit `992e89a037244e5e5cdf58ad47245c89866e373f`,
2026-06-07) serves the full Sync v2 endpoint surface and a documented, locked contract:
`tldw_Server_API/Docs/API/Sync_V2_M1.md` ("Sync v2 M1 API Contract", Status: Locked
for M1 implementation).

When the real chatbook `TLDWAPIClient` was driven against that live server, it failed
at the first call — `GET /api/v1/sync/capabilities` — because the chatbook client and
the server are on **divergent protocol revisions**:

| aspect | chatbook client (current) | server locked M1 contract |
| --- | --- | --- |
| `protocol_version` | int `2` (`ge=2`) | string `"sync-v2-m1"` |
| domains | coarse: `notes`, `chat`, `workspaces`, `source_cache`, `media` | dotted: `notes.note`, `chat.conversation`, `chat.message`, `attachment.ref` (+workspaces/source_cache/media fine-grained) |
| operations | `upsert`, `delete`, `link`, `unlink`, `resolve_conflict` | `upsert`, `append`, `tombstone` |
| encryption | `client_private_v1` (encrypted `payload_ciphertext`) | `server_trusted_v1` only at M1; `client_private_v1` is M3 work |
| conflict actions | `accept_local`/`accept_remote`/`merge`/`dismiss` | `overwrite`/`duplicate_rename`/`skip` |

Two important nuances were found in the running server source
(`tldw_Server_API/app/api/v1/schemas/sync_v2_models.py`):

1. The running build carries a **transition/compatibility layer**
   (`_with_transition_aliases`) that maps the chatbook client's field vocabulary
   server-side (`entity_id`→`object_id`, `server_sequence`→`server_cursor`,
   `payload_clear`→`payload`, `client_timestamp`→`created_at_client`) and even
   normalizes int `2` protocol versions to `"sync-v2-m1"`. This shim is explicitly
   transitional and must not be treated as durable contract.
2. The server's envelope validator (`_validate_m1_contract`) **hard-rejects** any
   encryption policy other than `server_trusted_v1` and any operation outside
   `upsert`/`append`/`tombstone`. No alias papers over those; the client must conform.

The server roadmap is explicit that `server_trusted_v1` is the M1 encryption posture
(backed by deployment at-rest encryption attestation), that M2 must not require
`client_private_v1`, and that `client_private_v1` (client-side, opaque-to-server
encryption) is M3 work (`Docs/Design/Sync_V2_M1_Implementation_Decisions.md`,
`Sync_V2_M2_*.md`, `Sync_V2_M3_*.md`).

## Decision

1. **The server's locked `Sync_V2_M1.md` contract is canonical.** The chatbook Sync v2
   client conforms to it; the server does not adapt to the client.
2. **The chatbook manual-sync milestone uses `server_trusted_v1`** (cleartext `payload`
   + `encryption_metadata.policy = "server_trusted_v1"`).
3. **`client_private_v1` and the client-side key-recovery work (task-24) are parked
   behind an M3 feature gate** — kept in-tree, removed from the M1 sync path, and
   re-targeted to align with the server's M3 milestone when it lands.
4. **The client emits the canonical M1 field names** (`object_id`, `server_cursor`,
   `payload`, dotted domains, `tombstone`/`append`) rather than relying on the server's
   transition aliases, so the client remains correct after the `-m1-next` shim is
   removed.
5. **Domains outside M1's four** (`workspaces.*`, `source_cache.*`, `media.*`) are
   deferred along with their existing client adapters; M1 scope is `notes.note`,
   `chat.conversation`, `chat.message`, `attachment.ref` (metadata only; no blob
   transfer).

The contract itself is not re-documented here; `Sync_V2_M1.md` at the pinned server
commit is the source of truth. The conformance design and phasing live in
`Docs/superpowers/specs/2026-06-12-sync-v2-client-m1-conformance-design.md`.

## Alternatives considered

- **Server adapts to the client (`client_private_v1` at M1).** Rejected: the server's
  M1 milestone is locked on `server_trusted_v1`; client-private is explicitly M3. This
  would re-block QA on server-side M3 work and invert the canonical direction.
- **Negotiate encryption per-dataset from advertised capabilities.** Rejected for now:
  most code and the most complex contract surface for no M1 benefit, since the server
  offers only `server_trusted_v1` at M1. Revisit at M3.
- **Lean on the server's transition aliases and keep client field names.** Rejected as
  the durable approach: the alias layer is transitional and may be removed; conforming
  to canonical names is the only forward-safe choice. (The aliases remain a useful
  safety net during migration.)
- **Rebuild the client's envelope identity model from scratch.** Rejected as
  unnecessary: the client's one-id-per-object, revision-based identity model maps
  cleanly to `object_id`/`object_revision`; this is a field rename plus base-revision
  bookkeeping, not a remodel.

## Consequences

- The client gains a `server_trusted_v1` cleartext envelope path; `client_private_v1`
  encryption/recovery becomes dead-but-gated until M3.
- New per-object local bookkeeping (last synced `server_cursor`/`object_revision`/
  `object_hash`) is required for whole-object domains to populate base metadata and
  detect conflicts.
- QA is conducted against an in-progress server branch behind a temporary compat shim;
  the spec pins the server commit and verifies `/profile/bootstrap` encryption
  attestation before per-domain QA.
- Manual Notes + Chat sync can round-trip end-to-end against the live server, unblocking
  task #24. task-119 is reframed: the server endpoints exist; the work is client
  conformance, not new server endpoints.

## Links

- Design spec: `Docs/superpowers/specs/2026-06-12-sync-v2-client-m1-conformance-design.md`
- Backlog: task #24 (QA closeout), task-119 (server contract gap, reframed)
- Server contract: `tldw_server2` `Docs/API/Sync_V2_M1.md` @ `992e89a03`
- Chatbook roadmap: `Docs/superpowers/specs/2026-05-26-chatbook-sync-v2-completion-roadmap-design.md`
