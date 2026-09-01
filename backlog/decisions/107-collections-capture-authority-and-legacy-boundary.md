# ADR-107: Separate Collections capture authority from Media and legacy containers

Status: Accepted

Date: 2026-08-31

Related Task: TASK-18919

Related Spec: [Library Collections Capture Reader Design](../../Docs/superpowers/specs/2026-08-31-library-collections-capture-reader-design.md)

Extends: [ADR-030](030-local-library-agent-tool-boundary.md), [ADR-055](055-library-destructive-action-reversibility-rule.md), [ADR-067](067-library-top-level-pagination-contracts.md), and [ADR-086](086-library-adaptive-reader-shell.md)

## Context

Chatbook's current Collections implementation stores arbitrary named containers and direct
memberships in `library_collections` and `library_collection_items`. That model was designed as a
Library grouping concept. The authoritative `tldw_server` Collections product instead treats its
Reading List as a Pocket/Instapaper-style capture domain with URLs, readable content, status,
favorites, tags, notes, highlights, saved searches, archive copies, and optional Media provenance.

The existing Media reading seam does not resolve this mismatch. Its Local save path creates Media
records and lacks the complete capture model; its normalizer intentionally discards capture-owned
fields. Reusing it would make Media the implicit owner and would preserve the semantic error that
TASK-18919 exists to remove.

The application also supports explicit Local and Server Library sources. A server-only design
would make Collections an exception to that grammar. A merged view would be unsafe because local
database identities and server ids are meaningful only under their profile, principal, and
workspace authority.

This decision changes durable storage, migration, source identity, service boundaries,
cross-database references, legacy-data treatment, and long-lived Library structure, so an ADR is
required.

## Decision

1. **Define Collections as capture and reading, not generic membership.** The Library Collections
   destination represents saved URL captures and their reading lifecycle. A capture is not a
   Library item type. A Media link is optional provenance and never capture identity.

2. **Follow one explicit active authority.** Local and Server use the same capture reader contract,
   but only one dataset is active. Local resolves from the profile's configured Collections
   database. Server resolves from the selected server profile, authenticated principal, and
   workspace/dataset scope. Switching authority replaces the dataset; no browse, count, search,
   saved-search, or mutation path merges them.

3. **Qualify every identity by authority.** Selection, cache, page, detail, mutation, receipt, and
   asynchronous request identities contain an opaque authority key plus source-owned id. The key
   incorporates the resolved Local profile/database or complete Server profile/principal/workspace
   authority. Raw private identifiers are not displayed or logged. A delayed result applies only
   when destination, authority, scope, item, revision/epoch, and generation all match.

4. **Add a dedicated Local capture repository to the existing database file.**
   `LibraryCollectionsDB` advances additively to schema v2 with capture items, tags, highlights,
   saved searches, linked-Note references, offline-copy metadata, and FTS5-owned search objects.
   Capture rows include a monotonic revision for optimistic compare-and-swap. Count and page reads
   share one read transaction. Filtering occurs before deterministic 20-row slicing, and every
   sort has a stable id tie-breaker.

5. **Make the v2 migration atomic and leave v1 physically untouched.** Migration acquires a
   bounded immediate write lock, rechecks version, creates all v2 objects, and records version 2 in
   one transaction. It does not rename, delete, import, or update `library_collections` or
   `library_collection_items`. Failure leaves v1 usable. The existing configurable
   `library_collections_db_path` remains the one Collections database setting.

6. **Mirror canonical-URL upsert semantics inside one authority.** Canonical URL is unique within
   a Local authority. Resave merges tags, preserves existing nonempty fields when input is absent,
   applies explicit supported updates, and replaces extracted content deterministically. It never
   deduplicates across authorities. A capture is committed before Local extraction completes;
   failed or interrupted extraction preserves it with Retry.

7. **Keep large offline copies out of SQLite.** Managed private files own archive bodies. SQLite
   owns validated relative metadata, hash, size, content type, and state. File publication is
   atomic, and an unavailable managed-file seam disables Save Offline Copy rather than storing an
   unbounded blob or claiming success.

8. **Use external references instead of cross-database foreign keys.** Media provenance and linked
   Notes store authority-qualified external references and are validated through their owning
   services. Missing or unauthorized targets are reported as unavailable and never cascade-delete
   the capture.

9. **Create a capture-specific service boundary.** A single scope service normalizes a dedicated
   Local repository and the authenticated Reading List API into capture summary/detail/page and
   capability contracts. It does not reuse `MediaReadingScopeService` or its normalizer. Capability
   differences remain explicit; unsupported, denied, offline, stale, and malformed responses fail
   closed with distinct reasons.

10. **Retire the old product without silently deleting its data.** Generic-container UI and
    mutations stop being current Collections. Compatibility mutations return
    `legacy_read_only`. No old tool or route name is redirected to new captures. A clearly labelled
    legacy adapter may provide bounded read-only listing/detail and complete streaming JSON export.
    Tool, RAG, Home, and rail inventories must not describe legacy containers as current captures.

11. **Adopt the shared adaptive Library reader structure.** Collections uses the ADR-086 Library,
    Items, and permanent Work shell. Library and Items are independently collapsible. Responsive
    state is transient, Library collapses before Items, and reclaimed Library width expands Items
    toward the shared comfort cap before surplus reaches Work. Collections owns its data, modes,
    actions, errors, and `[library.collections_reader]` list preferences.

12. **Keep archive status distinct from offline copy and hard deletion.** Move to Archive changes
    reading status and owes the ADR-055 receipt/Undo pattern. Save Offline Copy creates an archive
    snapshot. Hard delete is separately capability-gated and requires permanence copy; it does not
    delete external Media or Note targets.

## Required boundaries

- Page size is 20 under ADR-067; a current-page client slice is not paging.
- Only the active scope carries an exact count. Complete tag/domain facets require a future
  aggregate contract and cannot be inferred from returned rows.
- Saved-search query keys and sort values are allowlisted; arbitrary stored JSON never becomes
  SQL.
- Local SQLite and extraction work run off the Textual event loop, except explicit in-memory test
  seams that cannot cross connections.
- Capture persistence and extraction are separate outcomes. Refresh/extraction failure cannot
  erase a committed capture or the last good readable content.
- Server transport failure without an authoritative response is an unknown save outcome, while
  canonical-URL idempotence makes an explicit retry safe.
- Clean HTML is sanitized before rendering. Active content, scripts, and surprise remote asset
  fetches are forbidden.
- Logs and diagnostics exclude bodies, notes, highlights, credentials, raw paths, private stable
  ids, and sensitive URL query data.
- The umbrella server workflows for templates, digests, and full import/export management remain
  outside this reader task.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Continue arbitrary Library containers | It contradicts the authoritative server product and user intent. |
| Use Media Read Later as the Local authority | It creates Media records, loses capture-owned fields, and makes provenance into identity. |
| Support Server only | It breaks the active Library source contract and removes an approved Local workflow. |
| Merge Local and Server results | It makes identity, counts, saved searches, and mutations ambiguous across principals and profiles. |
| Create a second Collections database file | It adds another setting and recovery location without improving table ownership. |
| Automatically migrate v1 memberships | Generic members are not necessarily URL captures, so conversion would silently change meaning. |
| Rename or delete v1 tables | It makes rollback and explicit user recovery harder and destroys evidence needed for export. |
| Reuse Media scope models | They intentionally normalize away capture detail and retain the wrong Local storage owner. |
| Put offline bodies in SQLite | Large blobs would inflate the shared database and complicate atomic publication and cleanup. |
| Pretend complete tag/domain facets | The existing server list contract does not provide aggregate evidence. |
| Build a new or application-wide pane framework | ADR-086 already owns the Library structure; a new framework would duplicate it. |

## Consequences

### Benefits

- Collections finally matches the server's capture/reading meaning.
- Local remains useful without corrupting Media ownership.
- Source switching, delayed requests, and restored selection cannot cross authority boundaries.
- Exact search and paging remain reachable and truthful at scale.
- Capture saves survive extraction failure and restart interruption.
- Legacy generic Collections remain recoverable without continuing to shape the product.
- The reader feels consistent with Media and other migrated Library destinations.

### Accepted trade-offs

- Local gains a new durable schema and extraction lifecycle to maintain.
- Local and Server capabilities are not identical and require visible capability reporting.
- Existing generic Collection tools/routes require a compatibility retirement pass.
- Legacy records are exported, not automatically transformed.
- Managed offline copies add a small file lifecycle beside SQLite metadata.
- A single task crosses UI, service, database, compatibility, and verification seams and may need
  atomic implementation slices while preserving this one product contract.

## Rollback plan

The adaptive Collections route and capture service can be disabled while leaving schema-v2 tables
and managed files intact. Because migration is additive and v1 tables are untouched, the labelled
legacy reader/export adapter remains available. Rollback must not reclassify v1 containers as
current Collections or point new capture APIs at Media; it may only hide the new route until the
capture implementation is repaired.

## Links

- [Approved Collections capture reader design](../../Docs/superpowers/specs/2026-08-31-library-collections-capture-reader-design.md)
- [ADR-030: Local Library agent tool boundary](030-local-library-agent-tool-boundary.md)
- [ADR-055: Library destructive-action reversibility](055-library-destructive-action-reversibility-rule.md)
- [ADR-067: Library top-level pagination contracts](067-library-top-level-pagination-contracts.md)
- [ADR-086: Shared adaptive Library reader shell](086-library-adaptive-reader-shell.md)
