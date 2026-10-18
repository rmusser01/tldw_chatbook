# ADR-239: Canvas sync replica and lifecycle contract

Status: Proposed — user approval and companion-server agreement required
Date: 2026-10-17
Related task: TASK-31003
Related decisions: ADR-008, ADR-121, ADR-124

## Context

Canvas is local-first and currently excluded from server synchronization.
Existing Sync v2 supports authenticated dataset transport and an outbox,
but neither inspected companion-server checkout implements Canvas domains.
The current unique per-Canvas revision sequence also cannot preserve two
offline siblings numbered alike, and the server's default 256 KiB envelope
limit cannot carry every valid 512 KiB Canvas source.

## Proposed decision

Reuse the existing Sync v2 transport with explicit `canvas.document`
lifecycle and immutable `canvas.revision` domains. Enroll only durable
conversations whose complete message tree belongs to the synchronized dataset.
Preserve exact identities, source bytes, origins and profiles. The existing
`server_trusted_v1` policy means the authenticated home server can read source;
the enrollment surface must state that explicitly.

Permit equal-sequence sibling revisions through a future storage migration
and new archive contract while retaining all immutable existing values and
the current deterministic resolver. Preserve competing revisions and surface
conflicts. Deletion dominates visibility; restoration requires explicit
tombstone provenance, and retired conversation IDs fence stale replay.

Use transaction-owned source intents, existing outbox reconciliation and
idempotent receipts. Negotiate the complete domain/version/message-tree group
and byte limits before exchange; unsupported peers keep Canvas local. Imported
source never executes or grants a browser/runtime capability.

## Alternatives

Whole-graph snapshots retransmit large histories and do not solve sequence
collisions. Putting Canvas in message metadata or attachment references loses
the explicit ownership, lifecycle and compatibility boundary. Retain manual
archives until coordinated client/server support qualifies.

## Consequences and approval boundary

This proposal requires coordinated server adapters and deployment limits,
client storage/archive compatibility work, lifecycle/conflict review and real
replica qualification. It does not approve or implement those changes.
TASK-31003 retains its current open acceptance criteria; its distinction between
contract design and full implementation needs an explicit scope decision.

The [reviewable contract](../../Docs/superpowers/specs/2026-10-17-canvas-server-sync-contract-design.md)
contains concrete payloads, conflict outcomes, compatibility rules, limits,
server evidence and twelve implementation conformance cases. Approval must
cover source readability, revision-order changes and lifecycle policy before
implementation planning. Current runtime profiles and sync exclusion remain.
