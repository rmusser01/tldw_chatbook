# ADR-105: Consume portable Notes organization and layer Agent Lessons on ordinary Notes

Status: Accepted

Date: 2026-08-29

Related Tasks: TASK-24307, TASK-24308, TASK-24309

Related Spec: [Agent Lessons and Notes Organization Sync Design](../../Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md)

Amends: [ADR-073](073-notes-sync-round-trip-and-interoperability-constraints.md) and [ADR-059](059-notes-folder-import-and-device-local-sync-ownership.md)

## Context

Chatbook already stores hierarchical, ownership-aware Database Notes folders and
keywords locally, but its Sync-v2 client currently synchronizes note content only.
ADR-073 therefore kept folder membership device-local until a separately versioned
folder contract existed.

`tldw_server` now publishes one indivisible Notes organization group containing
`notes.keyword`, `notes.keyword_link`, `notes.keyword_collection`,
`notes.keyword_collection_link`, `notes.folder`, and `notes.folder_link`. The group
defines stable resource identities, deterministic link identities, hierarchy and
case-fold collision rules, non-cascading tombstones, and suppression-preserving
folder membership. Creating a different Chatbook contract would duplicate an
already-deployed server boundary and make synchronized Notes organization diverge.

Agents also need a durable way to reuse verified solutions, including failed
attempts and their explanations. Ordinary Notes already provide user ownership,
lexical search, permissions, and note synchronization. The missing pieces are
portable organization, exact search scopes, transactional organization-aware saves,
and trusted runtime guidance that treats retrieved lessons as untrusted data.

## Decision

1. **Consume the server's complete six-domain group.** Chatbook enrolls, validates,
   materializes, and publishes all six Notes organization domains as one capability.
   It does not advertise partial group support and does not introduce another server
   domain. Server schema-v1 payloads, UUIDv4 resource identities, canonical link
   hashes, hierarchy rules, and tombstone behavior are the interoperability authority.

2. **Keep local and portable identities separate.** Existing local primary keys stay
   local. Organization resources receive stable UUIDv4 sync identities and links use
   the server's deterministic identities. Identity allocation is durable and distinct
   from adoption or publication state.

3. **Enroll before publishing and never guess adoption.** A client applies the full
   bootstrap/history and resolves legacy adoption before declaring the group ready.
   Equal visible names or paths with different identities are reviewable candidates,
   not automatic matches. Missing `notes.note` or `chat.conversation` dependencies
   keep affected links local and unpublished.

4. **Record synchronization intent in the Notes transaction.** A Notes mutation and
   its immutable, version-bound envelope intents commit in the same ChaChaNotes
   transaction. Payload, operation, routing metadata, source version, and causal
   predecessor are immutable at owner commit. When an offline successor cannot yet
   know its predecessor's accepted cursor, it remains undispatched; after the
   predecessor is acknowledged, its complete optimistic base triple is bound exactly
   once in the Notes database before enqueue. The dispatcher never publishes a partial
   base. Restore is persisted at mutation time as
   `routing_metadata.restore_intent: true`, never inferred later by the dispatcher.
   A separate dispatcher copies those intents verbatim to the general outbox and
   acknowledges them idempotently only after `apply_status=applied`; a server
   `superseded` result is terminal but cannot create an owner head or unblock a
   successor because the server may have materialized no object state.
   Durable insertion sequence, not timestamp or UUID ordering, preserves causality.
   No transaction is claimed across two SQLite files.

5. **Preserve ADR-059's filesystem boundary.** Portable Notes folders and membership
   do not carry physical paths, hashes, watcher state, bindings, claims, recovery
   content, or filesystem authority. Device-private lasting folder sync remains owned
   by ADR-059 and ADR-073. Source-managed memberships retain local provenance while
   portable suppression follows the server contract.

6. **Use ordinary Notes for Agent Lessons.** `Agent_Lessons` is a user-manageable
   default root folder. The spelling-exact keyword `agent-lesson` is the authoritative
   discovery marker, so rename or movement of the folder does not hide lessons. User
   removal of the marker or deletion of a note removes it from Agent Lessons discovery.
   No separate memory store, embedding index, ranking system, or automatic conversation
   capture is added.

7. **Make lesson saves guided, verified, and conflict-safe.** Agents search before
   saving, update only materially identical root-cause lessons, include verification
   evidence and failed attempts, and use both note and organization concurrency
   preconditions. The organization token covers effective local memberships as well as
   synchronized heads/receipts, so local-only edits cannot evade stale-write detection.
   Additive keyword assurance never removes user keywords, and ordinary lesson updates
   preserve current folder membership.

8. **Hold incomplete lesson organization locally.** When organization is not ready or
   the canonical keyword needs review, the note and a content-free pending receipt
   commit locally while every normal dispatcher excludes that note. Finalization
   atomically creates immutable note/resource/link intents. A folder-only collision may
   become a durable non-blocking placement review once the keyword can publish.

9. **Keep lessons outside the instruction authority chain.** Agent Lessons guidance is
   a capability-aware trusted runtime suffix, but note bodies remain ordinary untrusted
   tool-result data. They cannot grant permission, authorize commands, expand scope, or
   enter system or project instructions. High-confidence credential formats are
   rejected at the agent-authored lesson-save boundary without logging rejected content.

## Required Boundaries

- Revalidate Chatbook contract fixtures against the current server `dev` contract before
  implementation and at integration time.
- Use real historical ChaChaNotes schemas for migration tests; stamping a partial
  synthetic schema with an old version is not evidence.
- Prove crash behavior across the Notes intent store and general outbox with two real
  SQLite files and deterministic failure points.
- Distinguish explicit resource deletion from descendants hidden by a deleted ancestor;
  never emit accidental cascading portable tombstones.
- Keep spelling-exact Agent Lessons keyword discovery separate from the server's
  case-fold uniqueness validation.
- Seed synchronized profiles only after group readiness and local-only profiles after
  local schema readiness. Concurrent untouched seeds may converge automatically; any
  user edit, membership, acknowledgement, or different spelling requires review.
- Retrieved note content stays in tool-result data and is never interpolated into a
  trusted prompt or instruction owner.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Add a seventh Chatbook/server folder domain | Duplicates the deployed six-domain server contract and creates incompatible organization authority. |
| Discover lessons by folder name alone | User rename, movement, or deletion would make synchronized lessons disappear from the agent workflow. |
| Automatically capture lessons from conversations | Persists noise, secrets, and unverified diagnoses without an explicit verified-save boundary. |
| Build a dedicated memory/vector service | Duplicates Notes storage, permissions, synchronization, and retrieval before lexical Notes search proves insufficient. |
| Publish a pending note before its organization is ready | Exposes a half-classified remote note and recreates the existing local-mutation/outbox loss window. |
| Infer same-name objects are the same portable resource | Can merge unrelated user organization and contradicts the server's explicit identity contract. |

## Consequences

### Benefits

- Notes folders, keywords, collections, and memberships become portable through the
  server contract already designed for them.
- Verified agent lessons are visible and user-owned ordinary Notes that other permitted
  agents can find across devices.
- User organization changes, offline work, interrupted enrollment, and dispatcher
  crashes have explicit conflict or recovery states instead of silent data loss.
- Filesystem authority remains device-private and does not leak into portable payloads.

### Accepted trade-offs

- Chatbook must implement and test all six organization domains before advertising any
  of them.
- Legacy organization requires stable-ID migration and potentially user-visible
  adoption review.
- A small local intent/receipt state machine is required to close the cross-database
  publication gap.
- Exact marker conflicts and edited seed races may require user review rather than
  automatic convergence.
- Lexical discovery can retain duplicates; agents cross-reference them rather than
  relying on risky automatic merge heuristics.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md)
- [ADR-008: Sync-v2 Client M1 Contract Alignment](008-sync-v2-client-m1-contract-alignment.md)
- [ADR-030: Local Library Agent Tool Boundary](030-local-library-agent-tool-boundary.md)
- [ADR-032: Local Agent Tool Permission Boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-055: Library Destructive Action Reversibility Rule](055-library-destructive-action-reversibility-rule.md)
- [ADR-059: Notes Folder Import and Device-Local Sync Ownership](059-notes-folder-import-and-device-local-sync-ownership.md)
- [ADR-073: Notes Sync Round-trip and Interoperability Constraints](073-notes-sync-round-trip-and-interoperability-constraints.md)
