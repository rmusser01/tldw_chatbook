# ADR-113: Separate Collections capture authority from Media and legacy containers

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
database identities and server ids are meaningful only under their Local profile/database or
Server profile/principal authority.

This decision changes durable storage, migration, source identity, service boundaries,
cross-database references, legacy-data treatment, and long-lived Library structure, so an ADR is
required.

## Decision

1. **Define Collections as capture and reading, not generic membership.** The Library Collections
   destination represents saved URL captures and their reading lifecycle. A capture is not a
   Library item type. A Media link is optional provenance and never capture identity.

2. **Follow one explicit active authority.** Local and Server use the same capture reader contract,
   but only one dataset is active. Local resolves from the profile's configured Collections
   database. Server resolves from the selected server profile and authenticated principal. The
   current Reading API has no workspace/dataset scope, so a local workspace switch does not change
   Server Collections authority. Switching authority replaces the dataset; no browse, count,
   search, saved-search, or mutation path merges them.

3. **Qualify every identity by authority.** Selection, cache, page, detail, mutation, receipt, and
   asynchronous request identities contain an opaque authority key plus source-owned id. The key
   incorporates the resolved Local profile/database or Server profile/principal authority. Raw
   private identifiers are not displayed or logged. A delayed result applies only when destination,
   authority, scope, item, revision/epoch, and generation all match.

4. **Add a dedicated Local capture repository to the existing database file.**
   `LibraryCollectionsDB` advances additively to schema v3: v2 adds capture items, tags,
   highlights, saved searches, linked-Note references, offline-copy metadata, and FTS5-owned
   search objects; v3 adds persisted extraction owner/lease fields so separate supported
   processes recover only expired claims.
   Capture rows include a monotonic revision for optimistic compare-and-swap. Count and page reads
   share one read transaction. Filtering occurs before deterministic 20-row slicing, and every
   sort has a stable id tie-breaker.

5. **Require coherent Server pages before enabling Server browse.** The current server evaluates
   count and rows in separate statements, so its page envelope cannot satisfy ADR-067 under a
   concurrent writer. The existing server list operation must read both inside one database
   snapshot/transaction and pass a controlled concurrency test. A passing build advertises exact
   boolean `capabilities.hasReadingSnapshotPagesV1: true` through the existing
   `/api/v1/config/docs-info` discovery response. Chatbook enables Server browse only for exact
   `true` under the current profile/principal capability snapshot; missing, false, or malformed
   evidence is unsupported with reason `server_page_snapshot_unavailable`. This adds no new public
   endpoint. Local remains available.

6. **Make capture migrations atomic and leave v1 physically untouched.** Migration acquires a
   bounded immediate write lock, rechecks version, creates all v2 objects, and adds the v3
   extraction-lease fields atomically. It does not rename, delete, import, or update
   `library_collections` or `library_collection_items`. Failure leaves v1 usable. The existing
   configurable `library_collections_db_path` remains the one Collections database setting. A
   version greater than 3 fails capture reads/writes with `schema_too_new` and is never stamped or
   repaired.

7. **Mirror canonical-URL upsert semantics inside Local authority.** Canonical URL is unique within
   a Local authority. Resave merges tags, preserves existing nonempty fields when input is absent,
   applies explicit supported updates, and replaces extracted content deterministically. It never
   deduplicates across authorities. A capture is committed before Local extraction completes;
   each active claim has an opaque persisted owner and renewable expiry, and only expired or
   migrated-unowned processing rows become interrupted. Failed or interrupted extraction
   preserves it with Retry. Current Server save extracts before
   persistence and supplies default status/favorite values. An unknown Server outcome is never
   retried automatically; explicit retry warns that it is duplicate-safe but not state-idempotent.

8. **Keep large offline copies out of SQLite.** Managed private files live under an owner-only,
   authority-derived root inside the profile's private data directory. SQLite owns normalized
   relative metadata, hash, size, content type, and state. Every operation rejects absolute paths,
   symlinks, and root escape. Initial limits are 50 MiB per copy and 1 GiB per Local authority.
   Quota and staging reservation are one transaction and count ready/staging capacity. A staging
   row whose final file was published before a crash is hash/size-validated into ready or cleaned
   and failed. Purge tombstones plus off-loop, bounded, cursor-resumable startup scavenging make
   publication and hard-delete recovery crash-safe. An unavailable seam disables Save Offline Copy
   rather than storing an unbounded blob or claiming success.

9. **Use external references instead of cross-database foreign keys.** Media provenance and linked
   Notes store authority-qualified external references and are validated through their owning
   services. Missing or unauthorized targets are reported as unavailable and never cascade-delete
   the capture.

10. **Create a capture-specific service boundary.** A single scope service normalizes a dedicated
   Local repository and the authenticated Reading List API into capture summary/detail/page and
   capability contracts. It does not reuse `MediaReadingScopeService` or its normalizer. Capability
   state is `unknown`, `supported`, or `unsupported(reason)` and is cached by server
   profile/principal/API version only after positive version, safe-probe, or exact-feature response
   evidence. Unknown destructive/data-creating actions remain disabled. Unsupported, denied,
   offline, stale, and malformed responses fail closed with distinct reasons.

11. **Retire the old product without silently deleting its data.** Generic-container UI and
    mutations stop being current Collections. Compatibility mutations return
    `legacy_read_only`. No old tool or route name is redirected to new captures. Whenever v1 data
    exists, **More > Legacy Collections data…** provides bounded active/deleted inspection and a
    schema-versioned, stable-order JSON export from one coherent read snapshot, atomically published
    to a validated user path. The recovery entry remains until a later explicit migration ADR;
    tool, RAG, Home, and rail inventories must not describe legacy containers as current captures.

12. **Adopt the shared adaptive Library reader structure.** Collections uses the ADR-086 Library,
    Items, and permanent Work shell. Library and Items are independently collapsible. Responsive
    state is transient, Library collapses before Items, and reclaimed Library width expands Items
    toward the shared comfort cap before surplus reaches Work. Collections owns its data, modes,
    actions, errors, and `[library.collections_reader]` list preferences.

13. **Keep archive status distinct from offline copy and hard deletion.** Move to Archive changes
    reading status and owes the ADR-055 receipt/Undo pattern. Save Offline Copy creates an archive
    snapshot. Hard delete is separately capability-gated and requires permanence copy; it does not
    delete external Media or Note targets.

## Required boundaries

- Page size is 20 under ADR-067; a current-page client slice is not paging.
- Server page shape validation does not prove snapshot coherence; Server browse remains disabled
  until the existing operation's count/rows transaction is positively established.
- Only the active scope carries an exact count. Complete tag/domain facets require a future
  aggregate contract and cannot be inferred from returned rows.
- Saved-search query keys and sort values are allowlisted; arbitrary stored JSON never becomes
  SQL.
- Local SQLite and extraction work run off the Textual event loop, except explicit in-memory test
  seams that cannot cross connections.
- Capture persistence and extraction are separate outcomes. Refresh/extraction failure cannot
  erase a committed Local capture or any authority's last good readable content; current Server
  save timing remains synchronous until its authoritative response.
- Server transport failure without an authoritative response is an unknown save outcome. Retry is
  duplicate-safe but not state-idempotent under current status/favorite defaults, so it is never
  automatic and requires explicit warning.
- The current Server authority excludes local workspace. Workspace changes cannot invalidate or
  repartition a Reading dataset until the Reading API owns that scope.
- Server feature capabilities are tri-state and authority/version-qualified. Unknown destructive
  or data-creating actions remain disabled.
- Schema versions newer than 3 fail capture reads/writes closed; the implementation never
  downgrades or stamps an unknown schema.
- Legacy recovery is mandatory whenever v1 rows exist and exports one stable coherent snapshot;
  later removal requires a new explicit migration decision.
- Managed offline files enforce private authority roots, symlink/root containment, transactionally
  reserved fixed admission limits, published-staging reconciliation, and bounded resumable
  staging/purge recovery.
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
| Accept current Server count/page responses as coherent | Separate database statements can race; client validation cannot reconstruct a snapshot. |
| Automatically retry an unknown Server save | Current request defaults can reset status and favorite on an existing canonical URL. |
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
- Exact search and paging remain reachable and truthful at scale once Server snapshot coherence is
  positively established.
- Capture saves survive extraction failure and restart interruption.
- Legacy generic Collections remain recoverable without continuing to shape the product.
- The reader feels consistent with Media and other migrated Library destinations.

### Accepted trade-offs

- Local gains a new durable schema and extraction lifecycle to maintain.
- Local and Server capabilities are not identical and require visible capability reporting.
- Server browse depends on a same-endpoint database transaction fix, and unknown feature groups may
  remain disabled until positive capability evidence exists.
- Unknown Server save outcomes require a warning and user-directed recovery rather than automatic
  retry.
- Existing generic Collection tools/routes require a compatibility retirement pass.
- Legacy records are exported, not automatically transformed.
- Managed offline copies add a small file lifecycle beside SQLite metadata.
- A single task crosses UI, service, database, compatibility, and verification seams and may need
  atomic implementation slices while preserving this one product contract.

## Rollback plan

Rollback disables capture browse and capture-service behavior while leaving schema-v3 tables and
managed files intact. It does not hide the Collections destination whenever any v1 row exists;
that destination becomes a recovery-only screen exposing **Legacy Collections data…** and the
labelled inspector/export adapter. Because v1 tables are untouched, recovery remains available
until a later migration/removal ADR. Rollback must not reclassify v1 containers as current
Collections or point capture APIs at Media.

## Links

- [Approved Collections capture reader design](../../Docs/superpowers/specs/2026-08-31-library-collections-capture-reader-design.md)
- [ADR-030: Local Library agent tool boundary](030-local-library-agent-tool-boundary.md)
- [ADR-055: Library destructive-action reversibility](055-library-destructive-action-reversibility-rule.md)
- [ADR-067: Library top-level pagination contracts](067-library-top-level-pagination-contracts.md)
- [ADR-086: Shared adaptive Library reader shell](086-library-adaptive-reader-shell.md)
