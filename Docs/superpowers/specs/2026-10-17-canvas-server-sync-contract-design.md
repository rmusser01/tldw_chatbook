# Canvas server synchronization contract design

Date: 2026-10-17. Status: **approved Chatbook target contract; implementation and server adoption not complete**.
Related task: [TASK-31003](../../../backlog/tasks/task-31003%20-%20Define-server-synchronization-contract-for-Canvas-artifacts.md).
Related decision: [accepted ADR-239](../../../backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md).
Inspected Chatbook base:
`272cad956855feda4d78efb9db9fce7e909d53db`.

The user approved this contract and splitting client/server implementation into
follow-up tasks on 2026-10-17. This document does not activate synchronization
or claim server support. Contract approval and feature activation remain
separate milestones; TASK-31003 covers the contract, not the complete feature.

ADR required: yes.
ADR path: backlog/decisions/239-canvas-sync-replica-and-lifecycle-contract.md (Accepted).
Reason: new cross-runtime service contract, replica ownership, conflict and
deletion policy, a durable revision-order/schema change, and version negotiation.
ADR-121, ADR-124, ADR-008 and the Personal Context ADR-102 are applicable prior
decisions, not approval of this proposal. Personal Context's encryption/key
package and profile purge mechanism are not automatically Canvas requirements.

## Evidence and constraints

- `Docs/Architecture/canvas.md`, `Canvas/service.py`, `Canvas/repository.py`,
  schema 66→67 and 67→68: Canvas belongs to exactly one conversation. Source,
  revision IDs, title, runtime profile, parent, sequence, actor and origin are
  immutable. The source is the durable artifact; render plans are derived.
  Temporary histories are process-only and promote atomically with messages.
- `Canvas/service.py::_resolution_key` already orders reachable revisions by
  `(origin_message_position_on_active_path, sequence, revision_id)`; explicit
  reachable historical selection overrides the default. Reopen hints and browser
  selections are local. Foreign or missing origins cannot be guessed/remapped.
- Current schema has `UNIQUE(canvas_id, sequence)` and a strict lower-sequence
  parent guard; `append_revision` allocates `max(sequence)+1` locally. Import
  validation additionally requires gapless unique sequences. Two offline peers
  can therefore create different immutable revisions numbered 2. A plain new
  transport adapter cannot combine those graphs losslessly in today's schema.
- `Sync_Interop/chat_outbox_producer.py` demonstrates durable source intents and
  exact intent-to-outbox receipts across separate databases.
  `sync_state_repository.py::sync_v2_local_outbox` is the outbound queue.
  `sync_log` is not that queue, and Canvas currently writes no sync-log records.
- `tldw_api/client.py` already exposes `/api/v1/sync/capabilities`, profile,
  device, dataset, push, pull, restore-manifest and conflicts transport. The
  client does not retry internally. The active local-first service chunks by
  **envelope count**, uses stable envelope IDs and batch idempotency keys, and
  records partial outcomes. A transport acceptance is not necessarily an
  applied product mutation; Canvas must check `apply_status`/`apply_errors`.
- `tldw_api/sync_schemas.py` accepts string domains, exposes read/write adapter
  version maps, and ignores unknown capability fields. Its default envelope
  payload ceiling is 262,144 bytes. Its conflict-action enum still includes
  legacy names (`accept_local`, etc.); new code must conform to the server's
  canonical `overwrite`/`duplicate_rename`/`skip` contract rather than rely on
  this mismatch. A string domain being parsable is not implementation support.
- Actual companion-server source was inspected read-only:
  `tldw_server` at `204a13724d0484eecb942d16fe82a28f08e68346`, and
  `tldw_server2` at `81258987760de25161fa4eb050c725e0b4618080`, under
  `/Users/macbook-dev/Documents/GitHub/`. Both contain the Sync v2 schemas and
  service, and neither declares Canvas artifact domains/adapters. Their Notes
  moodboard `canvas` field is unrelated. In `app/api/v1/schemas/sync_v2_models.py`
  domains are a closed Literal, operations are `upsert`/`append`/`tombstone`, and
  supported domains exclude Canvas. `app/core/Sync/v2/service.py` defaults to a
  262,144-byte envelope ceiling. `materializers/chat.py` handles
  `parent_message_id`, but that alone does not prove full Console variant/tree
  round-trip compatibility. These are source observations, not a live-server
  compatibility qualification or a promise about another deployment.

## Approaches

| Approach | Benefit | Cost / conclusion |
| --- | --- | --- |
| **Two explicit Sync v2 domains: document lifecycle + immutable revision** | Reuses existing authenticated dataset transport, outbox, cursors, restore and conflict review; transfers each source once; keeps identity and ownership explicit | Requires server adapters and a small but real revision-order/schema/archive change for offline siblings. **Recommended.** |
| Whole Canvas graph snapshots or archive blobs | Can reuse current graph validation; simple conceptual compare-and-swap | A conversation graph can contain 50 MiB of source, retransmits history on every edit, needs blob/manifest admission, and still cannot combine conflicting immutable sequence numbers in today's tables. Useful for existing manual export/import, not the initial incremental contract. |
| Put source/graphs in `chat.message` metadata or `attachment.ref` | Avoids declaring new domain names | Violates ADR-121's artifact/transcript boundary, relies on metadata-only attachment behavior, permits older chat writers to discard Canvas state, and hides deletion/branch semantics. Reject. |

Keep export/import as the available portability mechanism until both peers
implement and qualify the new contract. Do not add an independent Canvas REST
CRUD path or silently tunnel Canvas through an existing accepted domain.

## Ownership and eligibility

One runtime-owned `CanvasService` remains the authorized Canvas mutation owner
in Chatbook. A sync adapter validates transport and invokes a Canvas-domain
ingress operation; it does not write SQL directly, call browser controllers,
compile/render HTML, or become another artifact owner. The companion server
must provide the equivalent inert validation/persistence boundary behind Sync
v2; this is external implementation work, not an existing server feature.

Each Canvas owner is `(authenticated account, server profile, dataset,
conversation_id)`. The server checks membership/authorization before inspecting
source. A claimed `conversation_id`, `parent_id` or origin is data to verify,
not a permission grant. A workspace label, current Console screen, browser
session, source HTML, and model tool argument cannot enroll a conversation.

Only durable conversations explicitly enrolled in that dataset's Chat sync
may contribute Canvas intents. Both the conversation object and its complete
branching message graph, including historical soft-deleted origin messages,
must be available in the same dataset before dependent revisions apply.
Temporary chats, uncommitted assistant turns and discarded stages never enter
an outbox. Promotion becomes eligible only after the existing message+Canvas
transaction commits and its exact origin-ID map is proven. Disabling sync
stops exchange; it does not delete artifacts. Runtime execution enablement is
independent from source portability.

Canvas and revision UUIDs are preserved on all replicas. Ownership never moves.
Existing explicit import-as-new is the operation that deliberately remaps the
whole conversation/message/Canvas/revision graph. Equal source hashes do not
collapse different IDs. Reusing an ID with different immutable content,
ownership, parent, origin, title or profile is an integrity conflict.

## Contract domains and exact data

All names, the `canvas` capability object and the Canvas-specific semantic rules
below are **approved target protocol extensions**, not present server features.
The enclosing protocol remains `sync-v2-m1`; the Canvas adapter/schema is v1.

| Domain | Object identity / owner | Operations | Payload |
| --- | --- | --- | --- |
| `canvas.document` | `object_id = canvas_id`; `parent_id = conversation_id` | `upsert`, `tombstone` | `canvas_id`, `conversation_id`, immutable `created_at`, lifecycle `epoch_id`, `deleted_at`, bounded observed revision-ID set on deletion, and explicit restore provenance when restoring |
| `canvas.revision` | `object_id = revision_id`; `parent_id = canvas_id` | `append` | Exact complete immutable record shown below; never an HTML patch, rendered plan, executable runtime bundle, or mutable head pointer |

Document deletion is conversation-global, as today. Titles remain revisioned;
rename appends a `user_rename` revision with unchanged source/profile. There is
no independent revision-delete command. Preserve the existing immutable nullable
revision `deleted_at` field when receiving historical archives/records; it does
not authorize arbitrary revision tombstones. Conversation deletion and purge
take precedence over the document state.

`content_sha256` is the existing bare lowercase SHA-256 of exact UTF-8 source;
`payload_hash` is `sha256:` plus the canonical whole-payload digest. Canonical
payload bytes use the current Chatbook hashing convention: JSON with sorted
keys, compact separators, `ensure_ascii=True`, `allow_nan=False`, UTF-8 output.
There is no Unicode normalization, newline rewrite, float, timestamp rewrite,
or HTML repair during transport. `payload_size_bytes` counts those exact bytes.
Timestamps retain the immutable stored value; validation accepts only the
documented bounded timestamp representation. A server's envelope receipt time
does not replace an artifact's creation/deletion timestamps.

The examples use actual computed source and payload hashes. A revision append
uses `object_revision = 1` because the **revision object** is immutable; its
Canvas `sequence` is unrelated to the Sync object's version. Updates/tombstones
of the mutable document use the acknowledged `base_server_cursor`,
`base_object_revision`, and `base_object_hash` triple. Omit legacy alias fields
on canonical wire requests even if a current model emits them internally.

```json
{
  "dataset_id": "dataset-personal",
  "device_id": "device-a",
  "idempotency_key": "canvas-push-7f1855d0",
  "envelopes": [{
    "client_envelope_id": "55555555-5555-4555-8555-555555555555",
    "dataset_id": "dataset-personal",
    "domain": "canvas.revision",
    "operation": "append",
    "object_id": "33333333-3333-4333-8333-333333333333",
    "parent_id": "22222222-2222-4222-8222-222222222222",
    "adapter_version": 1,
    "schema_version": 1,
    "object_revision": 1,
    "base_server_cursor": null,
    "base_object_revision": null,
    "base_object_hash": null,
    "deleted": false,
    "routing_metadata": {"observed_document_epoch_id": "epoch-a"},
    "payload": {
      "revision_id": "33333333-3333-4333-8333-333333333333",
      "canvas_id": "22222222-2222-4222-8222-222222222222",
      "conversation_id": "11111111-1111-4111-8111-111111111111",
      "parent_revision_id": null,
      "sequence": 1,
      "title": "Hello",
      "runtime_profile": "canvas-v1",
      "source": "<h1>Hello</h1>",
      "content_sha256": "e2c6c0ea7c7900c31f953e48d30d5e839801ab90630d751e7c8426ed5859da47",
      "source_bytes": 14,
      "actor_kind": "assistant",
      "origin_message_id": "44444444-4444-4444-8444-444444444444",
      "origin_turn_id": "turn-1",
      "created_at": "2026-10-17T18:00:00Z",
      "deleted_at": null
    },
    "payload_hash": "sha256:3c33958221fec3d6e1d735bf25331ffa7b4ba1adb72d128e5f4bba4c7c6c1e96",
    "payload_size_bytes": 539,
    "encryption_metadata": {"policy": "server_trusted_v1"}
  }]
}
```

A document creation payload is:

```json
{
  "canvas_id": "22222222-2222-4222-8222-222222222222",
  "conversation_id": "11111111-1111-4111-8111-111111111111",
  "created_at": "2026-10-17T18:00:00Z",
  "epoch_id": "epoch-a",
  "deleted_at": null,
  "observed_revision_ids": [],
  "restore_from": null
}
```

Use a persisted random UUID for `epoch_id` in the implementation; the short
values in examples are labels. The epoch remains constant through ordinary
appends/deletion and changes only on explicit restore. A tombstone preserves
identity/creation/epoch, sets `deleted_at`, and includes the sorted revision IDs
the deleting peer had observed. Restore is `upsert` based on the exact current
tombstone triple with a new epoch and `restore_from` containing that triple.
An upsert without this provenance cannot revive a tombstone.

Dependencies are derived from these typed payload references and verified at
ingress. Existing untyped `dependencies` dictionaries alone do not prove
ordering or ownership. Push conversation first, then messages in parent order,
then documents and revisions in parent order. The first document and root may
arrive separately, but a rootless document is incomplete and never advertised
as ready. Missing references produce a bounded retryable dependency result;
foreign references produce a terminal ownership error. Pull applies the same
rules into a durable pending inbox so page boundaries never drop an early
child. A pull cursor may advance only with durable custody of every envelope
in that page, not after a failed or memory-only apply.

## Revision graphs, offline order, and visible conflicts

The proposed v1 revision-order amendment permits sibling revisions to share a
positive `sequence`, retains each original value, requires exactly one root at
sequence 1, and requires every child sequence to be strictly greater than its
same-Canvas parent's sequence. Distinct sequence values still cover every value
from 1 through the maximum, bounded at 100; only uniqueness changes. No sequence
is reassigned after sync. New local appends still allocate `max(sequence)+1`.
Change the local unique index to a non-unique lookup index and adjust complete
graph import validation to allow ties while retaining contiguous distinct
values, acyclicity, root, parent and count checks. The current resolver's UUID
tie-break is already deterministic; no CRDT, content merge or second ordering
key is needed.

This is a durable contract change, not just an index optimization. It requires
a schema migration and a new Canvas-bearing archive format/contract version
because a current 3.0 importer rejects equal sequences. Keep valid existing 3.0
archives readable; old applications must reject the new format without partial
import. Do not reuse format 3.0 for a graph its stated rules cannot represent.
The final version number and migration number are allocated at implementation
time against the then-current branch. Do not alter immutable existing rows.
Affected contracts include `Canvas/repository.py`, `Canvas/service.py`,
`DB/ChaChaNotes_DB.py` plus its migration/fresh-schema definitions,
`Chatbooks/chatbook_models.py` and `chatbook_importer.py`, backup schema
validation, and their repository/service/migration/archive fixtures. Canonical
export ordering becomes `(sequence, revision_id)` where tied rows were formerly
impossible. Sync producers/appliers/state, typed client capability/conflict
schemas, Console durable settlement/promotion and canonical sync documentation
also need integration; none of these callers should be assumed covered by only
adding a domain name.

For the same fully materialized graph and same ordered active message ancestry,
each peer resolves the same default revision with the existing tuple. Different
active branches can legitimately resolve different revisions. Active leaf,
reopen hint, browser state, selected historical revision and selection generation
stay local; receiving data never switches the user's branch or opens a browser.
Canvas sync eligibility requires a proven lossless message-tree adapter,
including stable origin IDs and parent/variant ancestry. Merely advertising
`chat.message` is insufficient. Missing ancestors remain incomplete, never
re-parented or projected onto the currently visible chat.

| Situation | Proposed deterministic outcome |
| --- | --- |
| Identical revision ID and exact immutable payload | Idempotent no-op; preserve original receipt |
| Same revision ID, any immutable field differs | Integrity conflict; no overwrite, both exact candidates retained for review |
| Offline appends on distinct message branches | Keep both immutable branches; resolve against each active ancestry |
| Concurrent siblings on the same origin message/parent, including rename vs update | Admit both within quotas; show durable conflict review with both revision IDs/titles/digests and the deterministic default. Do not field-merge title/HTML or imply automatic user approval |
| User chooses a sibling as the new shared continuation | Append an explicit `user_import` resolution revision from the chosen source/profile under the active origin, using a new ID and `max(sequence)+1`; keep both siblings and link the conflict resolution to that revision. Local historical selection alone does not resolve shared history |
| Independent Canvas creations with equal titles/source | Preserve separate UUIDs; no fuzzy deduplication |
| Soft delete versus update in the same epoch | Delete dominates visibility; retain the revision/history, and record a `delete_vs_update` review when the delete's observed-ID set omitted that update. Same result regardless of arrival order; no clock-based winner |
| Two deletes in the same epoch | Both converge on deleted state; retain the first authoritative tombstone receipt and record the other as equivalent, not an implicit restore |
| Restore versus an old offline update/delete | Only restore based on the exact tombstone creates the next epoch. Old-epoch commands cannot change lifecycle; retain late updates as conflicts and never resurrect the document |
| Conversation tombstoned | All its Canvases are hidden irrespective of per-document state; immutable history remains available to authorized restore, not to active Canvas tools |
| Capacity exceeded after combining offline branches | Retain local edits/pending incoming candidate and show a quota conflict; do not prune history, claim full convergence, or acknowledge a rejected write as synced |

A restore offers the retained known graph, including known concurrent updates,
for explicit review. Late old-epoch candidates require a new explicit update or
import after restore; no automatic replay under the new epoch. Conflict records
and pending candidates are separate durable sync state, not edits to revision
content. A peer with unresolved local or incoming candidates must say pending /
conflicted, not claim the graphs are fully synchronized. User-visible conflict
summaries exclude source; source inspection is an explicit authorized action.

## Deletion, purge, and stale devices

Soft deletion is reversible and never destroys revisions. A deleted document
cannot accept a source update as an implicit restore. Even if an old client
deletes/restores the owning conversation without Canvas support, the server
must enforce the parent lifecycle independently: restore reveals only document
identities that were not independently deleted. Omitting Canvas from a chat
payload never means delete or empty the Canvas graph.

Hard-purging a conversation records a content-free, dataset-scoped retired owner
ID barrier and cancels/removes its content-bearing pending work before deleting
Canvas payloads. The server rejects future writes for that retired conversation
ID. A restored external backup must import as a new identity or go through an
explicit qualified recovery procedure; a stale sync push cannot recreate it.
Keep the minimal retired-ID barrier for the dataset's lifetime. If an operator
ever compacts it, all older devices/cursors must be revoked and require a fresh
baseline first. No TTL-only tombstone expiration is allowed.

Local-only removal, synchronized soft delete and delete-everywhere/purge must
remain distinguishable operations. Purge acknowledgement does not claim that
an offline/stolen device or external backup has been erased. The server must
define backup retention and content erasure behavior before advertising purge
support. This proposed contract does not expand browser-side revocation claims.

## Outbox, retry and partial failure

Commit the artifact mutation and an immutable source intent in the owning
ChaChaNotes transaction. Reconcile exact committed source intents into the
existing Sync v2 outbox with source receipts; use a persisted stable
`client_envelope_id` per intent. The source DB and sync-state DB are separate:
do not claim a distributed transaction. A crash before reconciliation is
recovered from the source intent; a crash after enqueue is deduplicated by the
receipt. Never reconstruct an old wire snapshot from whichever revision is
currently selected. Incoming application uses a no-echo origin marker.

Reuse existing per-profile/principal/workspace/dataset outbox scoping and policy
gates. Revalidate binding, ownership, purge barriers and current negotiated
capabilities before each send. A source intent must not leak to a replacement
server/account after settings or runtime identity changes. Hard purge must fence
and clean content copies in both source and sync-state stores, rather than leave
replayable source envelopes behind a local row deletion.

Retry the exact immutable envelope and batch identity after an uncertain
transport outcome. The server deduplicates both stable envelope ID and revision
object ID + exact payload hash; changing bytes under an existing envelope ID is
an idempotency error. Rebatching can change the batch key but not envelope IDs.
Timeouts, disconnects, 429/503 and missing dependencies retain pending state;
honor Retry-After and use the existing caller retry policy. Auth failures pause
for reauthentication; malformed payloads, foreign ownership and unsupported
versions require review/capability correction; no infinite automatic retry.

Drain an envelope only after a matching idempotent or accepted **applied**
receipt, with no contradictory apply error. A server pending/failed/conflict
projection is not a successful sync. Process unrelated successes independently.
Persist the incoming envelope or conflict candidate before moving its cursor;
replay is idempotent after crashes at any apply/receipt boundary.

## Capability negotiation and limits

Proposed addition to the **existing** capabilities response (illustrative
fragment; no new endpoint):

```json
{
  "protocol_version": "sync-v2-m1",
  "domains": ["chat.conversation", "chat.message", "canvas.document", "canvas.revision"],
  "operations": {
    "canvas.document": ["upsert", "tombstone"],
    "canvas.revision": ["append"]
  },
  "supported_adapter_versions": {"canvas.document": [1], "canvas.revision": [1]},
  "writable_adapter_versions": {"canvas.document": [1], "canvas.revision": [1]},
  "max_envelope_payload_bytes": 4194304,
  "canvas": {
    "contract_version": "canvas-sync-v1",
    "message_tree_contract": "chat-message-tree-v1",
    "revision_order": "sibling-sequence-ties-v1",
    "max_source_bytes": 524288,
    "max_title_bytes": 4096,
    "max_canvases_per_conversation": 10,
    "max_revisions_per_canvas": 100,
    "max_source_bytes_per_conversation": 52428800,
    "retired_owner_barrier": true
  }
}
```

Negotiate the complete domain group, exact adapter/schema versions, operation
sets, message-tree contract, revision-order rule, deletion fencing and effective
limits. Missing/malformed/partial capability information disables Canvas
exchange; it never becomes success because the endpoint responds 200. Recheck
dataset writability after device enrollment changes. Do not assert existing
servers advertise this object, or require all older clients to understand it.

Preserve current durable caps: 10 Canvases/conversation, 100 revisions/Canvas,
512 KiB exact UTF-8 source/revision, 50 MiB source/conversation, 4 KiB UTF-8 title,
and existing bounded IDs/origin turns/path length. Count retained deleted
history as current repository quotas do. Reject invalid/deep/cyclic graphs,
invalid UTF-8, digest/size mismatches and foreign origins before mutation.
The receiving peer independently enforces its supported limits.

The present 256 KiB transport limit cannot promise a 512 KiB Canvas round-trip.
JSON escaping can expand valid source up to sixfold. A full v1 advertisement
requires at least 4 MiB payload capacity for one bounded revision plus its
metadata; this is a required server deployment/admission change. Validate the
actual serialized payload, routing/dependency overhead and HTTP request/page
size independently; count-only batching is insufficient. Push at most one
source-bearing envelope per request initially, avoiding a 100×4 MiB allocation.
Pull pagination needs the equivalent bounded byte budget or a server hard cap.
Do not reduce source quotas, truncate, split source into undocumented fragments,
or enable a small-doc-only feature under the full v1 capability. Deployment
needs coordinated body limits at the API server and proxy.

## Compatibility and the execution boundary

| Client / server combination | Behavior |
| --- | --- |
| Current client + current server | Current local-only Canvas behavior; manual archives remain available |
| New client + server without the complete v1 group | Canvas stays local with a bounded unsupported-capability explanation; ordinary chat sync can continue |
| Older client + new server | Older client requests only supported domains; server preserves Canvas records independently. No absent-field deletion, Canvas stripping, source substitution, or reinterpretation as attachments |
| New client with older revision-order schema/archive reader | No v1 Canvas enrollment until its storage capability is upgraded; no partial import of tied graphs |
| Compatible sync adapter + unknown/revoked/missing runtime profile | Preserve exact source/profile/history and synchronize them inertly; local execution stays unavailable |
| Newer Canvas adapter/schema payload | Do not interpret, edit, render, downgrade or echo it through an older adapter. Preserve server data; keep a bounded opaque pending record if unexpectedly delivered, and report unsupported version |
| Device upgrades to Canvas after advancing a chat-only dataset cursor | Perform a replay/baseline for the newly enrolled domains from their beginning; do not start at the chat cursor and silently miss history. Replaying the whole existing dataset from zero is acceptable if current per-dataset cursor storage is retained |

Cross-device portability intentionally discloses source to the authenticated
home server under the existing `server_trusted_v1` policy and TLS, with server
at-rest encryption readiness required as ADR-008 requires. Do not claim E2EE or
inherit Personal Context's device-only flags, crypto package, keys or agent
permissions. The opt-in surface must make server readability clear.

Sync never evaluates HTML/JS, starts QuickJS/a browser, fetches a profile/library
URL, installs an executable manifest, opens a preview, submits composer input,
or grants filesystem/network/browser authority. Profile identifiers are inert
strings; runtime admission remains the receiving application's packaged local
catalog and revocation policy. Capabilities, sessions, tokens, selection epochs,
confirmation receipts, cached plans and runtime grants do not synchronize.
Ordinary logs, diagnostics, failure strings, review summaries and metadata
surfaces keep source out; use bounded reason codes and opaque IDs/digests.

## Required implementation acceptance cases

These are future behavioral tests, not tests of this proposal's wording.

1. Create, update and rename round-trip exact source bytes, both hashes, IDs,
   parent, sequence, title/profile, actor, origin message/turn and timestamps;
   equal source under different IDs stays distinct. Same-ID divergent immutable
   data fails without overwriting either candidate.
2. Rollback/cancel creates no source intent or outbound source. Temporary chats
   never enqueue; promotion+rollback/retry preserves the exact origin map.
3. Deliver root/parent/origin dependencies after their children and across pull
   pages/crashes. Nothing is dropped, foreign origins never apply, and no early
   graph is labeled ready. Soft-deleted historical origins remain linkable.
4. Two real offline replicas append sequence-2 siblings, both on distinct message
   branches and on the same origin. The merged graph preserves both IDs/values;
   every shared active ancestry resolves identically. Historical pinning and
   navigation remain local; same-origin conflicts are visible and resolvable by
   one new immutable continuation.
5. Exercise rename-vs-update and delete-vs-update in both delivery orders.
   Deletion hides without source loss; restore requires exact current tombstone
   provenance; late old-epoch commands cannot undo restore or resurrect delete.
6. Old client conversation deletion/restoration preserves independent Canvas
   deletion, and ordinary chat updates never strip unknown Canvas records.
   Hard purge plus an offline replay or restored backup cannot reuse the retired
   owner ID. Verify cleanup of source-bearing local outbox/inbox/conflict copies.
7. Inject a crash before source reconciliation, after enqueue, after server
   commit before response, after local apply before cursor save, and before
   receipt. Exact retries neither duplicate revisions nor lose pending work.
   Mixed accepted/rejected/conflict/apply-pending responses drain only proven
   applied entries, including idempotent responses.
8. Test absent/partial/malformed capabilities, read-only dataset, enrollment
   changes, newer adapter/schema, old client, unknown/revoked profile and upgrade
   baseline. Ordinary chat continues without false Canvas sync success.
9. Boundary-test escaped Unicode/control source at 512 KiB, 4 KiB titles,
   100-revision/10-Canvas/50 MiB aggregate quotas, combined offline overage,
   maximum actual envelope/request/page bytes, invalid digest/UTF-8 and cycles.
   Quota failures preserve committed local work without silent pruning.
10. Canary source through sync success/error/conflict/retry paths: no source in
    logs, metadata or normal diagnostics; no network egress beyond the authorized
    sync service, no runtime/browser creation and no capability/grant imported.
11. Upgrade genuine current databases with rollback/retry and fresh-schema
    parity. Read existing 3.0 archives; round-trip a tied-sequence graph in the
    new archive contract; a current reader rejects it atomically. Never reuse
    schema/archive identifiers from an unmerged branch.
12. Qualify create/update/delete/restore/conflict/retry against a pinned real
    companion-server build, not only mocks. Verify the HTTP/proxy body caps and
    both persistence backends the server advertises. No server qualification
    claim is established by this design review.

## Recorded decisions, external dependencies and delivery split

The user's 2026-10-17 approval accepts decisions 1–3. Decision 4 remains a
server follow-up requirement, not a blocker for inert client foundations:

1. **Ownership/protocol:** accept the two new domains, home-server readability,
   conversation-only enrollment and no second CRUD surface.
2. **Revision order:** accept equal-sequence siblings with the existing
   deterministic resolver, including the required schema and archive-contract
   amendments. If this is rejected, automatic graph combination is out of scope;
   an alternative must explicitly preserve disconnected conflict candidates.
3. **Lifecycle/conflicts:** accept delete-dominant visibility, explicit
   tombstone-based restore epochs, durable visible sibling conflicts and
   dataset-lifetime retired-owner barriers.
4. **External owner:** identify the companion-server maintainer/release that will
   implement and lock the same contract, complete message-tree preservation,
   capability/version negotiation, byte-bounded transport and purge retention.
   Read-only availability of a server checkout is not authorization to change it
   and is not evidence those features exist.

An approved ADR/spec must name the exact final wire capability/version names,
conflict error codes, owner-barrier retention policy and server build before a
client implementation can claim compatibility. The suggested names and
semantics here are concrete review proposals, not reserved public identifiers.

Approved TASK-31003 acceptance-scope amendment:

- Retain AC1 as an explicit **documented and approved** contract, satisfied by
  this design, ADR-239 and the user's recorded approval.
- Reword AC2–7 as requirements the approved contract must specify, with explicit
  examples/conflict outcomes and compatibility/security rules.
- Reword AC8 as a concrete implementation conformance-test matrix. Do not check
  it off as though the automated feature tests have run.
- Reword AC9 to require an approved ADR, current architecture/Sync documentation,
  and migration/archive compatibility requirements; actual feature migrations
  belong to implementation work.
- Track server implementation, Chatbook storage/archive changes and Sync
  integration/real-server qualification as separately created atomic tasks after
  the decision is approved. Do not invent task IDs or mark the present mixed
  implementation criteria Done on the strength of a design document.

The original implementation requirements remain assigned to the following
atomic tasks; no design checklist is evidence that their feature tests passed.

| Task | Independently reviewable deliverable | Earlier dependencies |
| --- | --- | --- |
| TASK-35376 | Client sibling-sequence storage, migration and archive compatibility | TASK-31003 |
| TASK-35377 | Server inert Canvas domains, immutable persistence and ownership | TASK-31003 |
| TASK-35378 | Server lifecycle epochs, conflict retention and retired-owner barriers | TASK-35377 |
| TASK-35379 | Server complete message-tree capability and byte-bounded admission | TASK-35378 |
| TASK-35380 | Client committed source intents and outbox reconciliation | TASK-35376 |
| TASK-35381 | Client durable inbox, domain ingress, conflicts and lifecycle apply | TASK-35380 |
| TASK-35382 | Client negotiated enrollment, consent and byte-bounded exchange | TASK-35379, TASK-35381 |
| TASK-35383 | Pinned real-server two-replica conformance and release qualification | TASK-35382 |

Original implementation AC2–4 are covered by ownership/replica tasks
35376/35377/35380/35381; AC5 by 35378/35381; AC6–7 by
35377/35379/35381/35382; AC8–9 by the per-task tests/docs and final 35383
qualification. The twelve acceptance cases above remain the release matrix.

No production sync or server change is activated by this approval record. Server
repository selection, canonical contract adoption, real-server tests and release
activation remain follow-up gates. Read-only server evidence is not a write grant.
