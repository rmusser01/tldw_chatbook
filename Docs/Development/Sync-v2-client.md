# Sync-v2 client runtime

This page is the canonical runtime reference for Chatbook's Sync-v2 client.
It describes the portable Notes organization group implemented by
[ADR-105](../../backlog/decisions/105-portable-notes-organization-and-agent-lessons.md).
The server contract remains the interoperability authority.

## Notes organization is one capability

Chatbook consumes these six schema-v1 domains as one indivisible group, in this
order:

1. `notes.keyword`
2. `notes.keyword_link`
3. `notes.keyword_collection`
4. `notes.keyword_collection_link`
5. `notes.folder`
6. `notes.folder_link`

The server must advertise adapter version 1 for the complete group and enroll
the group with its `notes.note` and `chat.conversation` dependencies. Chatbook
refuses a partial advertisement, partial enrollment, a response for another
device, or any encryption policy other than `server_trusted_v1`. It never
claims readiness for a subset.

`notes.note` is not changed by this feature. Organization envelopes refer to
the existing note and conversation identities instead of embedding content.

## Identity and validation

Local database primary keys remain device-local. Every keyword, collection,
and folder instead receives a stable canonical UUIDv4 `sync_id`; migration
does not derive that ID from a mutable name, path, or local integer. Two
devices can therefore materialize one portable object under different local
primary keys.

Relationship IDs are deterministic hashes of canonical JSON containing the
domain, ordered identity members, and schema version. Incoming links are
recomputed before materialization. Payload validation is strict: unknown
fields, invalid UUIDs, invalid hierarchy references, invalid names, and link
IDs that do not match their members fail closed.

Portable name uniqueness uses the server's case-fold rules. The existing local
folder UI may apply stricter normalization; an unrepresentable local collision
becomes a review item rather than an implicit merge.

## Enrollment and legacy adoption

The first manual sync for an eligible local-first server profile advances a
durable state machine:

1. Persist a device-local bootstrap identity before contacting the server.
2. Bootstrap or resume the complete server dataset.
3. Pull and apply the complete organization history before any local publish.
4. Present content-free adoption reviews for same-name or same-path objects
   whose identities differ.
5. Inventory adopted legacy organization in bounded, restartable phases.
6. Declare the group ready only when server state, local state, inventory, and
   reviews are all complete.

Bootstrap, pull, review, and inventory checkpoints live in the Notes database.
An interruption resumes from the last committed object rather than rebuilding
the inventory from a newer snapshot. `merge`, `rename_local`, and `keep_local`
are explicit adoption choices. `keep_local` remains unpublished for later
mutations as well as for the initial inventory.

Adoption review is available with Manual Sync in **Settings**. While a review
is open, or while the server is still initializing, Manual Sync reports a
blocked or conflict state and does not drain organization intents. Resolve the
review and run Manual Sync again to resume enrollment.

## Dependency behavior

Keyword links can target an enrolled note or conversation. Folder links target
an enrolled note. A relationship whose content dependency is absent is kept
local and reviewable; it is not published with a dangling reference. The
inventory checkpoint retains this result so a restart does not forget why the
relationship was skipped.

Incoming envelopes are applied in dependency order. Missing resources,
hierarchy cycles, case-fold collisions, and missing content dependencies are
reported as bounded conflicts instead of being guessed or partially applied.

## Local mutation and publication recovery

A Notes organization mutation and its immutable intent commit in one
ChaChaNotes transaction. The intent contains its domain, object identity,
operation, payload, routing metadata, source version, causal predecessor, and
optimistic base. A separate dispatcher copies the intent to the general
Sync-v2 outbox because the Notes database and SyncState database are separate
SQLite owners.

There is deliberately no claimed cross-database transaction. Recovery is
idempotent at each boundary:

- a crash before outbox copy leaves the Notes intent pending;
- a crash after copy reuses `intent_id` as `client_envelope_id` and does not
  enqueue a duplicate;
- a crash after a server acknowledgement replays the durable receipt and marks
  the Notes intent acknowledged;
- only an `applied` acknowledgement advances the portable owner head and
  unblocks its causal successor;
- `superseded` is terminal for that envelope but does not invent a server head
  or release a successor whose complete base is unknown.

Pending intents for one object publish in insertion order. A successor is held
until its predecessor has an accepted cursor, revision, and hash; the dispatcher
never publishes a partial optimistic base. Restore intent is recorded with the
owner mutation rather than inferred later.

## Folder deletion, restore, and suppression

Deleting a folder emits a tombstone only for the explicitly deleted folder.
Descendant folders and memberships remain active portable objects, although
the local tree hides them beneath the deleted ancestor. Restoring the ancestor
uses explicit restore intent and makes the dormant descendants and memberships
effective again without recreating them.

Effective folder membership is:

```text
(active manual membership UNION active source-managed membership)
MINUS portable suppressions
```

Removing one provenance does not tombstone the portable link while another
active provenance still makes it effective. A source-managed unlink emits a
link tombstone only when the effective union becomes empty. Portable
suppression is synchronized, but source-owner provenance remains local.

## Filesystem boundary

Portable Notes organization carries logical names, hierarchy, and membership
only. It never carries absolute or relative physical paths, file hashes,
watcher state, bindings, claims, recovery content, source-owner identifiers,
or filesystem authority. Lasting folder sync and Folder Files remain
device-private systems governed by ADR-059 and ADR-073.

## Operational checks

When Manual Sync does not advance, inspect its status in **Settings**:

- **initializing**: the server has not completed its bootstrap snapshot;
- **adoption review/conflict**: resolve the named content-free review;
- **dependency missing**: enroll or synchronize the referenced note or
  conversation before retrying;
- **failed**: retain the checkpoint and retry after the reported server or
  validation problem is fixed;
- **ready**: the complete six-domain group may publish and pull.

Do not repair these states by deleting checkpoints or rewriting object IDs.
Retry through Manual Sync so the client can resume its durable state. For
schema or integration verification, always use a disposable `HOME`, XDG
directories, config file, data directory, and database. A schema-bumping
checkout must never be launched against a real profile shared with an older
worktree.

## Contract references

- [ADR-105](../../backlog/decisions/105-portable-notes-organization-and-agent-lessons.md)
- [ADR-059](../../backlog/decisions/059-notes-folder-import-and-device-local-sync-ownership.md)
- [ADR-073](../../backlog/decisions/073-notes-sync-round-trip-and-interoperability-constraints.md)
- Client contract: `tldw_chatbook/Sync_Interop/notes_organization.py`
- Enrollment and dispatch: `tldw_chatbook/Sync_Interop/notes_organization_sync_service.py`
- Materialization: `tldw_chatbook/Notes/notes_organization_repository.py`
