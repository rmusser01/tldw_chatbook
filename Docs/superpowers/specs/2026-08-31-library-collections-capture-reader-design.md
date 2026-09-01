# Library Collections Capture Reader Design

Date: 2026-08-31

Status: Approved

Task: TASK-18919

ADR: [ADR-107](../../../backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md)

Related: [Library adaptive reader design](2026-08-24-library-destinations-adaptive-reader-design.md), [ADR-067](../../../backlog/decisions/067-library-top-level-pagination-contracts.md), [ADR-086](../../../backlog/decisions/086-library-adaptive-reader-shell.md)

## Summary

Replace Chatbook's obsolete interpretation of Collections as arbitrary groups of
Library records with the Pocket/Instapaper-style capture and reading domain defined by
`tldw_server`. Collections becomes one focused reading workflow:

1. capture a URL;
2. scan and filter saved captures;
3. read extracted content;
4. highlight, annotate, classify, archive, or return to the original.

The destination uses the established adaptive Library reader structure:

- **Library** — global Library navigation plus contextual capture scopes and saved searches;
- **Items** — one exact, paged capture list and Quick Capture;
- **Work** — the permanent Read, Highlights, Notes, and Info surface.

The active Library source remains explicit. Local uses a new capture-owned repository
inside the existing configurable Collections database file. Server uses the authenticated
Reading List API. Switching source replaces the dataset; it never merges Local and Server
captures. Media is neither the capture store nor the capture identity. A Media reference is
optional provenance only.

The existing `library_collections` and `library_collection_items` tables remain untouched
as read-only legacy data. They are not renamed, imported, or silently reinterpreted.

## Goals

- Make Collections a coherent capture-to-reading destination shaped like Library Media.
- Support the same core browsing and reading loop under one explicit Local or Server source.
- Keep Local captures independent of Media records and Server captures qualified by the selected
  server profile and authenticated principal.
- Keep every matching capture reachable through deterministic 20-row pages with an exact
  active-scope total after the Server snapshot prerequisite below is satisfied.
- Preserve truthful state through slow extraction, rapid traversal, source switches,
  concurrent writes, partial failures, and application restarts.
- Let Library and Items collapse independently, with reclaimed Library width expanding
  Items toward its comfort cap before surplus flows to Work.
- Preserve legacy generic Collections for explicit recovery without continuing their old
  product model.
- Verify Local and Server behavior through one shared contract suite plus production-shaped
  Textual and live reader walkthroughs.

## Non-goals

- Arbitrary folders, reusable source sets, or cross-type Library membership containers.
- Treating a Collection as a Library item type.
- Folding Media Read Later records into Collections or creating a Media record for every
  capture.
- Merging, synchronizing, or deduplicating Local captures against Server captures.
- Automatically converting legacy generic Collections into captures.
- Building the umbrella Collections features for templates, digest schedules, or full
  import/export management in this task.
- Adding a new public `tldw_server` endpoint. One internal same-endpoint snapshot fix is a Server
  enablement prerequisite.
- Claiming Local and Server feature parity where one authority lacks a capability.
- Creating another adaptive-pane framework instead of using the Library-owned shell.

## Authoritative product model

The `tldw_server` Reading List contract is the semantic reference. A capture has a URL and
canonical URL, title, domain, optional summary and readable representations, status,
favorite flag, tags, freeform note, timestamps, optional highlights, optional linked Notes,
and optional Media provenance. Supported reading statuses are `saved`, `reading`, `read`,
and `archived`.

The current server performs fetch/extraction before its database upsert; a handled fetch failure
may still produce a capture with `last_fetch_error`, but it is not a commit-first asynchronous
lifecycle. Local mode deliberately commits first and extracts afterward. The UI presents these
timings truthfully instead of claiming mechanical parity. Under either authority, an extraction
failure that follows an authoritative committed response never removes the bookmark.

The wider server Collections umbrella also includes templates, digest schedules, imports,
and exports. Those workflows remain separate destinations or follow-up work; they do not
inflate this reader into a general Collections dashboard.

## Current Chatbook state and cutover

Chatbook currently has three nearby but incompatible concepts:

1. `LibraryCollectionsDB` schema v1 stores arbitrary named containers and source
   memberships in `library_collections` and `library_collection_items`.
2. `LocalLibraryCollectionsService` and current Library/MCP surfaces expose that generic
   container model.
3. Media reading services expose some server Reading List operations, while the current
   local `save_reading_item` path creates a Media record and cannot preserve the full capture
   contract, including notes and favorite state.

None is the new Local capture authority. The Media normalizer also omits capture-owned fields
needed by this reader. TASK-18919 therefore adds a capture-specific repository, models, and
scope service rather than extending the Media reader seam.

At cutover:

- the Library Collections destination stops presenting generic containers;
- generic-container create, rename, membership, and delete entry points return the explicit
  structured reason `legacy_read_only` during their compatibility window;
- no old tool or route name is redirected to new captures;
- whenever any v1 row exists, **More > Legacy Collections data…** remains reachable from the
  Collections destination and opens a distinctly labelled read-only recovery view;
- that required recovery adapter provides bounded inspection of active and deleted collections
  and every membership plus a complete schema-versioned streaming JSON export;
- export holds one coherent SQLite read snapshot, uses stable collection/member ordering, publishes
  atomically to a user-chosen validated path, never truncates at an internal page boundary, and
  never writes record content into logs;
- the export envelope is `tldw-chatbook-legacy-collections` version 1 with `exported_at`, a
  `collections` array ordered by `collection_id` (including `deleted_at`), and a `memberships`
  array ordered by `collection_id, membership_id`; values preserve stored text without inventing
  capture URLs or types;
- the recovery entry and adapter remain until a later ADR explicitly decides migration or removal;
- tool, RAG, Home, rail-count, and help inventories stop describing generic containers as the
  current Collections product;
- the configured `library_collections_db_path` remains the database location, so users do not
  acquire a second ambiguous Collections setting.

## Authority and identity

### Active authority

Collections follows the active Library source:

- **Local** — the capture tables in the resolved profile's configured Collections database;
- **Server** — the authenticated Reading List dataset for the selected server profile and
  authenticated principal.

The current Reading API has no workspace or dataset parameter. Changing Chatbook's local workspace
therefore does not change Server Collections authority, clear its selection, or hide its retained
receipt. Workspace may join the authority key only after a future Reading API explicitly accepts
and returns that scope.

The source selector never offers a combined view. A source switch clears the prior page,
selected and loaded identities, detail, saved-search snapshot, and exact totals before requesting
the new source. An active archive/Undo receipt remains session-retained under its originating
authority but is hidden and inactionable under every other authority; switching back may restore
that exact receipt. It is never rebound to the newly active dataset.

### Authority-qualified identity

Every list row, detail request, cache entry, selection, mutation, and delayed completion is
qualified by an opaque authority key plus the source-owned item id. The authority key includes:

- Local profile and resolved Collections database identity; or
- Server profile and authenticated principal identity.

Raw local paths and private principal identifiers are not displayed or logged. A compact,
non-reversible authority fingerprint may be used internally. Persisted layout preferences are
source-neutral; persisted or session-restored selection is accepted only when its authority key
matches the current authority.

Every asynchronous application fence contains at least:

`destination + authority + scope + item + mutation/content revision + generation`.

Unmount and source change advance the generation. A late success or failure that does not match
the complete fence cannot update the visible page, Reader, receipts, or actions.

## Local capture storage

### Additive schema v3

`LibraryCollectionsDB` advances additively to schema v3. Schema v2 creates the capture-owned
tables; schema v3 adds persisted extraction-owner and lease-expiry fields so another supported
process can recover only expired work rather than interrupting a live claim. The tables cover:

- reading items and their durable extraction state;
- normalized item tags;
- reading highlights and content anchors;
- saved searches;
- linked-Note references;
- managed offline-copy metadata; and
- FTS5 search content with owned synchronization triggers.

The reading-item record preserves the server-shaped fields required by the reader: local capture
id, submitted and canonical URL, domain, title, summary, freeform note, optional text and clean
HTML, content hash/word count, publication/read timestamps, status, favorite, processing state,
last fetch error, optional Media provenance, created/updated timestamps, and a monotonically
increasing revision. Text and clean HTML are independently optional; neither is fabricated merely
to satisfy a schema shape.

The migrations:

- acquire a bounded `BEGIN IMMEDIATE` migration lock;
- recheck the current schema inside the transaction;
- create all v2 objects and record version 2 atomically, then add the v3 lease fields atomically;
- never rename, delete, or write v1 generic-container rows;
- roll back completely on failure; and
- leave an older process able to read or mutate only the physically separate legacy tables.

The current `schema_version` table's maximum-version convention remains valid. Fresh creation,
real v1-to-current migration, v2-to-v3 lease migration, failed migration rollback,
concurrent-open behavior, and v3 reopen are all tested. If `MAX(schema_version) > 3`, this
implementation refuses capture reads and writes with
`schema_too_new`; only the explicitly compatible v1 legacy inspector/export may remain available
after verifying its expected tables. It never stamps or attempts to repair an unknown future
schema.

### Concurrency and coherent reads

The UI and local MCP runtime may open the same database in separate processes. Local mutations use
the existing immediate write transaction pattern and a capture revision compare-and-swap. A stale
expected revision changes nothing and returns a conflict containing enough current metadata to
offer Reload or Retry without discarding the user's draft.

Exact total and one 20-row page are read inside one SQLite read transaction. Filtering and FTS
matching happen before sorting and slicing. Every allowed sort ends with the stable capture id as
its final tie-breaker. Page envelopes follow ADR-067: undersized non-final pages, oversized pages,
duplicate/missing ids, invalid totals, and inconsistent page metadata fail closed.

### Canonical URL upsert

Local save mirrors the server's idempotent capture semantics:

- canonical URL is the deduplication key inside one Local authority;
- saving an existing canonical URL updates the existing capture rather than creating a duplicate;
- incoming tags merge deterministically;
- absent optional input preserves existing nonempty values;
- an explicit supported value may update title, status, favorite, summary, or note;
- a resave of an archived URL returns that same capture and applies the explicit requested status,
  otherwise preserving its current status; and
- content hash and extraction replacement are deterministic and revisioned.

This uniqueness does not cross Local profiles, database paths, server profiles, or principals.

### Extraction and managed files

Quick Capture writes the capture first with a durable processing state, then schedules extraction
off the Textual event loop. The extractor uses existing network-admission and path/security
boundaries, rejects unsafe URL schemes and SSRF targets, limits redirects and response size, and
sanitizes stored HTML. It does not create a Media record merely to obtain text.

Processing states distinguish queued/processing, ready, failed, and interrupted. On startup, a
stale in-flight state becomes **Interrupted · Retry** rather than remaining busy forever. Each
processing claim stores an opaque owner token and expiry; the live owner renews its lease while
the off-loop extractor runs, and startup interrupts only expired or migrated-unowned claims. A
different process therefore cannot invalidate a fresh same-authority extraction. Extraction
failure records a bounded user-safe reason while preserving the saved capture.

Large offline copies are managed private files, not unbounded SQLite blobs. Local bodies live under
the profile-resolved private data directory at
`collections_archives/<local-authority-fingerprint>/`. The root is owner-only; published files are
owner read/write only. The authority fingerprint is non-reversible and a capture cannot name a
different authority's root.

The database stores normalized relative ownership metadata, content hash, size, media type, and a
two-phase lifecycle state. Every open/create/delete re-resolves beneath the authority root and
rejects absolute paths, `..`, symlinks at any component, or root escape. Initial admission limits
one offline copy to 50 MiB and one Local authority to 1 GiB. Quota admission and the staging
reservation occur atomically in the write transaction and count every ready or staging reservation,
so concurrent saves cannot each admit the same remaining capacity. A limit failure preserves the
capture and reports the required recovery.

Creation records a staging state, writes and synchronizes a temporary sibling, atomically publishes
the file, then marks metadata ready. On restart, a staging row with a published final file is
hash/size-validated and completed to ready; otherwise its temporary/final artifacts are removed and
the row becomes failed. Hard delete first makes the capture and file metadata
inaccessible under a transactional purge tombstone, removes the file, and finally deletes owned
rows. Startup scavenging runs off-loop in bounded batches under a durable resumable cursor; it
removes abandoned temporary files, reconciles staging rows, completes purge tombstones, and marks
ready metadata missing if its file cannot be recovered. A crash can therefore leave recoverable
staging/purge work, never an active row that silently claims a missing copy. Local **Save Offline
Copy** remains disabled until this complete managed-file seam is available.

## Cross-database references

Media provenance and linked Notes live outside the Collections SQLite database, so capture tables
do not declare cross-database foreign keys. They store authority-qualified external references and
validate them through the owning service when opened.

A missing, deleted, moved, or unauthorized target appears as **Media unavailable** or **Linked Note
unavailable**. It never cascade-deletes the capture. Server note links remain qualified by the same
server authority as the capture unless the server API explicitly reports another supported
authority.

## Capture service boundary

TASK-18919 introduces capture-specific summary, detail, page, saved-search, highlight, note-link,
mutation, capability, and error contracts. It composes:

- a synchronous Local repository invoked off-loop for file-backed SQLite and extraction work; and
- an asynchronous Server adapter over the existing authenticated Reading List client methods.

It does not reuse `MediaReadingScopeService` or its capture-losing normalizer. The scope service is
the only UI entry point and normalizes both backends into capture-owned models without flattening
away source-specific capability information.

### Server enablement prerequisite

The current server `list_content_items` implementation executes `COUNT(*)` and its page query as
separate database statements. A concurrent writer can therefore produce rows and `total` from
different snapshots. Client validation cannot reconstruct the coherence required by ADR-067.

Before Server Collections browse is enabled, that existing server operation must read count and
rows inside one database snapshot/transaction and must pass a controlled concurrent-writer test.
This is an internal implementation correction to the existing endpoint, not a new client-facing
endpoint. A build containing the fix advertises exact boolean
`capabilities.hasReadingSnapshotPagesV1: true` through the existing
`/api/v1/config/docs-info` discovery response only after that controlled test passes. Chatbook
enables Server browse only for exact `true` under the current server-profile/principal capability
snapshot. Missing, false, or malformed evidence fails closed as **Server Collections needs a
paging update** with reason `server_page_snapshot_unavailable`; Local remains available. Runtime
heuristics and page-shape validation never substitute for the attestation.

### Page request and envelope

One immutable page request contains:

- authority key;
- search text;
- zero or more statuses;
- favorite filter;
- exact tag filters;
- domain filter;
- inclusive date bounds;
- an allowlisted sort; and
- one-based page with fixed size 20.

The response contains the complete applied scope, exact total, page, size, validated summary rows,
and a source revision/generation marker when available. Local produces count and rows from its own
read transaction. Server accepts the envelope as coherent only after the enablement prerequisite
is positively established; shape validation alone is insufficient. Search, filter, and sort always
apply before paging. A fresh out-of-range page gets one generation-guarded reload of the last valid
page under ADR-067; repeated shrink enters stale recovery rather than looping.

The exact total is shown for the active scope only. Tags and domains are typed filters and may use
suggestions from the current item or already returned rows. They are never advertised as complete
source-wide facets unless a future aggregate endpoint provides that evidence.

### Saved searches

Saved searches belong to the active authority and are listed through a bounded, paged contract.
Stored queries accept only the page request's defined filter keys, value types, and sort values.
Unknown keys, nested expressions, SQL fragments, and invalid sort names fail validation; arbitrary
JSON is never translated into SQL.

### Capabilities

The service returns tri-state per-authority capabilities—`unknown`, `supported`, or
`unsupported(reason)`—for browse, capture, update, highlights, linked Notes, summarize, listen,
archive, offline copy, hard delete, and related recovery actions. A disabled action remains visible
with a source-owned reason. Permission denial, unsupported API, offline state, missing dependency,
stale identity, and undiscovered support remain distinct reasons.

Local capabilities derive from composed services. Server support derives only from positive
evidence: an advertised API version matched by a versioned compatibility table, a documented
non-mutating feature probe, or an authoritative successful response for that exact feature group.
Unknown destructive or data-creating actions remain disabled; a reachable **Check availability**
action may run safe probes. Unknown read-only/idempotent actions may be attempted only after an
explicit user action and downgrade on a feature-route 404. A feature-route 404 changes only that
capability, never ordinary reading or unrelated actions.

Server capability state, including `hasReadingSnapshotPagesV1`, is cached by server profile,
authenticated principal, and advertised API version/capability snapshot. Profile, credential,
principal, version, or capability-snapshot change invalidates it. No Local implementation is added
merely to make the table look symmetric.

## Information architecture

### Library

When Collections is active, the existing Library rail reveals contextual rows beneath the
Collections destination:

- All Captures;
- Saved;
- Reading;
- Read;
- Archived;
- Favorites; and
- bounded Saved Searches with a reachable **More saved searches** continuation.

These rows apply a scope to Items; they are not separate Collections or folders. Only the active
scope receives an exact total. Global Library destinations remain present and retain their normal
ownership.

### Items

Items is a reading list rather than an operations table. Its header contains **Quick Capture**,
**Filter captures**, filter/sort controls, active range/total, and page controls. Each compact
two-line row shows:

- title and selected/loaded relationship;
- domain and compact saved/published date;
- status, favorite, and extraction failure/interruption markers when applicable.

Rows do not expose private URL query strings. Page navigation preserves selection by
authority-qualified identity when the selected row remains in scope. Scope/filter change selects
the first matching row only after the new page is authoritative. Empty results retain the query
and offer clear-filter or Quick Capture recovery.

Selection and loaded Reader identity are separate. Traversal updates row selection immediately but
starts detail loading only after a short injected settle delay; Enter loads immediately. While B
loads, Reader may retain A only beneath explicit copy such as:

> Loading “B”… showing “A” until ready.

Identity-sensitive actions remain disabled until selected and loaded identities match.

### Quick Capture

Quick Capture accepts a validated URL with optional title, tags, and freeform note. Local returns
the committed capture before extraction completes and preserves omitted status/favorite on a
canonical-URL retry.

Current Server save performs extraction before persistence and its request defaults
`status="saved"` and `favorite=false`. The Server UI therefore shows **Saving and extracting…**
until an authoritative response arrives. A transport failure with no response is **Save outcome
unknown**, not a confirmed failure or success. It offers Refresh first and never retries
automatically. An explicit retry requires copy warning that the current server may reapply Saved
status and clear Favorite on an existing canonical URL; canonical-URL upsert prevents a duplicate
but is not state-idempotent. A future server change may remove this warning only after status and
favorite become preserve-on-omission and extraction becomes commit-first.

After a confirmed save, follow-up placement/detail failure does not reclassify the save. Items
reconciles the known capture when safe, marks the page stale, and offers an authoritative refresh.

## Work reader

Work is permanent and reading-first. Its quiet header shows domain/source, title, optional byline
or publication date, reading time when known, status, and authority label. The primary toolbar
keeps reading-status actions, favorite, **Move to Archive**, **Open Original**, and **More**
reachable.
At constrained widths lower-priority actions move into More without disappearing.

Exactly one mode is visible:

- **Read** — sanitized readable content, preferring safe clean HTML when supported and falling back
  to stored text. Remote scripts, active content, and surprise asset fetches are never rendered.
- **Highlights** — active and detached highlights, their notes, anchors, and reattachment/retry
  state. Creation is enabled only when the runtime can provide a stable Reader selection or other
  source-supported anchor.
- **Notes** — the capture's freeform note editor followed by a distinctly labelled Linked Notes
  list. The two models never overwrite each other.
- **Info** — editable tags/status when supported, URLs, timestamps, word count, authority,
  extraction status, offline-copy state, and optional Media provenance.

The active mode persists across item traversal during the Collections session. Missing content in
one mode shows an item-specific empty state rather than silently changing modes.

Summarize, Listen, Save Offline Copy, refresh extraction, linked-Note operations, and hard delete
live in More or the owning mode. **Move to Archive** changes reading status; **Save Offline Copy**
creates an archive snapshot. Their labels, results, and recovery are never conflated.

## Mutations, archive, and deletion

Mutations invalidate older reads before committing. A successful mutation with a failed follow-up
read reconciles only fields proven by the mutation response, retains the last good page as stale,
suppresses unsafe exact-total and identity-sensitive actions, and offers Retry.

Moving to Archive records the prior status and leaves an in-place ADR-055 receipt with **Undo**.
Undo restores the prior status through the same authority seam, reinserts or removes the row under
the current filter truthfully, and explains when the item was restored outside the visible scope.

Hard delete is separate, capability-gated, and title-specific. Confirmation states that the capture,
its capture-owned highlights, and its managed offline copy are permanently removed and cannot be
undone. Local deletion uses the purge-tombstone lifecycle above, so a crash resumes cleanup rather
than exposing a half-deleted capture or leaking an untracked file. Missing external Media or Notes
targets are not part of that deletion.

## Loading, stale, conflict, and recovery states

- Initial list loading affects Items while Work remains mounted with a selection prompt.
- Detail or extraction failure affects Work/current row without disabling list traversal.
- Page refresh failure retains the last good rows with a stale banner and disables totals, paging,
  row mutations, and bulk-like actions that require current identity.
- Extraction pending and extraction failure are distinct from page refresh pending/failure.
- Local revision conflict preserves the user's input and offers Reload or Retry.
- Server authentication or policy failure preserves Local availability and explains how to change
  source or configuration.
- Malformed envelopes and unsafe HTML fail closed with metadata-only diagnostics.
- A source change, route exit, or unmount invalidates pending list, detail, mutation, extraction,
  summary, TTS, highlight, and note-link generations.

Logs may include operation kind, source kind, bounded reason code, page number, row count, and
generation. They exclude article bodies, notes, highlights, titles when unnecessary, raw local
paths, credentials, stable private ids, and URLs containing sensitive query data.

## Adaptive layout and focus

Collections uses `LibraryAdaptiveReaderShell` and the shared pure resolver under ADR-086. It adds
one destination preference section, `[library.collections_reader]`, for `items_open` and
`items_width`; shared Library visibility, width, and custom-width opt-in remain under
`[library.reader]`.

Defaults use the shared fixed 40-column Items target, 32-column minimum, and 56-column automatic
comfort cap. Collections declares a 48-column Work minimum and 56-column Work comfort because its
Notes mode is editable. Custom width remains opt-in and uses the shared normalization. Responsive
state never persists.

Resolution follows the shared contract:

1. Work is permanent and both five-column grips remain reachable.
2. Library and Items use their requested widths when the active Work mode remains usable.
3. Responsive shortfall collapses Library before Items.
4. With Library collapsed and Items open, reclaimed width expands Items toward 56 columns before
   surplus goes to Work.
5. Explicitly opening a pane gives it temporary priority and may responsively collapse the other
   optional pane.
6. Widening restores requested panes and widths without rewriting preferences.

Pure resolver evidence starts in Read mode with requested Library and Items both open, custom
widths off, no explicit priority, and an explicit adaptive-shell content width `W`. It pins these
exact input/output cases independently of terminal chrome:

| Pure resolver input `W` | Effective Library / Items / Work widths |
| --- | --- |
| 160 | 30 / 40 / 80; all three roles open |
| 120 | 0 / 56 / 54; Library collapsed |
| 100 | 0 / 42 / 48; Library collapsed |
| 80 | 0 / 0 / 70; Work priority with both grips reachable |

Mounted evidence is separate. At terminals 160x50, 120x35, 100x30, and 80x24, the test records the
actual settled `shell.content_size.width`, computes exactly one result from that measured `W` and
the declared preferences/mode/profile, and asserts every rendered region and grip matches it. The
canonical Collections walkthrough stores terminal size, measured `W`, requested state, Work mode,
and resulting Library/Items/Work/grip geometry alongside its captures. It never assumes terminal
width equals shell width and never accepts either of two layouts for one measured input.

Focus follows Library, Library grip, Items controls/list, Items grip, Work toolbar/modes/content.
Automatic collapse evacuates focus to the corresponding visible grip. Escape closes transient
state first, then graduates outward through effective panes. In any effective Work-only state, the
next outward Escape invokes the same explicit Items reopen as the labelled restore control and
transfers focus to the retained/first Items row; the resolver gives Items temporary priority and
keeps Work mounted under ADR-086. No screen binding shadows terminal conventions or the
repository's global bindings.

## Verification

### Local repository and migration

- fresh v3 creation, real v1-to-current and v2-to-v3 migration, rollback, reopen,
  concurrent-open, and synthetic
  future-version `schema_too_new` tests;
- v1 tables byte/row unchanged by migration and new capture operations;
- canonical-URL upsert, tag merge, explicit archived resave, content replacement, and revision CAS;
- exact count/page snapshot, stable tie-breaker, FTS query, page shrink, and malformed envelope;
- extraction ready/failed/interrupted transitions, private-root containment, symlink rejection,
  quotas, staging/purge restart recovery, and managed-file atomicity;
- reachable legacy inspection plus validated, coherent-snapshot streaming JSON export containing
  active/deleted collections and every stably ordered membership beyond the first page.

### Shared Local/Server contract

One parameterized suite covers supported list, detail, save, update, favorite, status, tags, notes,
archive/Undo, saved searches, highlights, linked Notes, and capabilities. Source-specific tests
prove unsupported actions remain disabled with the correct reason. More than 40 captures prove the
second and third pages are reachable. Server enablement additionally requires a controlled writer
between count and row evaluation to prove the same endpoint now holds one snapshot; the pre-fix
server must reproduce the mismatch and remain unsupported.

### Session and Textual behavior

- selected versus loaded identity and settle-delay/Enter behavior;
- source/profile/principal/database switches and complete late-response fencing, plus proof that a
  local workspace switch does not change Server Collections authority;
- Local committed-save/follow-up-read failure, unknown Server save outcome with no automatic retry,
  explicit retry warning, stale refresh, and recovery;
- Server capability unknown/supported/unsupported transitions, per-authority cache invalidation,
  safe probes, and feature-route 404 isolation;
- local conflict preservation and archive receipt/Undo placement;
- sanitized HTML/text fallback and no active or remote content execution;
- each mode, capability-disabled copy, missing external references, and narrow toolbar overflow;
- pointer and keyboard operation for both grips in every open/collapsed combination;
- focus evacuation/restoration, custom-width refresh, and idempotent shrink/expand cycles.

### Production-shaped and live evidence

Run the production-shaped cross-reader suites so Collections cannot regress Media, Conversations,
Notes, Prompts, or Skills geometry and source behavior. Perform isolated Local and Server
walkthroughs at 160x50, 120x35, 100x30, and 80x24, covering untouched startup, route activation,
resize restoration, keyboard-only traversal, every collapse combination, Quick Capture,
extraction failure, archive/Undo, and source switch.

Live runs use an isolated configuration and data directory, verify the resolved database path, and
fingerprint the real profile before and after. Server evidence identifies the returned capture by
authoritative content, not merely by absence of an exception. Condition-based waits replace fixed
sleeps.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep arbitrary generic Collections | It contradicts the server product and the user's capture/reading intent. |
| Use Media Read Later as Collections | It makes captures Media records, loses capture fields, and preserves the wrong identity model. |
| Make Collections server-only | It breaks the approved Library source grammar and removes a useful offline/local capture workflow. |
| Merge Local and Server captures | Identical ids or URLs can belong to different principals and authorities; merged totals and mutations would be unsafe. |
| Accept current Server totals as exact | The count and rows can come from different database snapshots; the client cannot validate coherence back into existence. |
| Automatically retry an unknown Server save | Canonical URL prevents duplicates, but current request defaults can reset reading status and favorite. |
| Put captures in a new database file | It adds an ambiguous second Collections setting and recovery location without improving ownership; additive tables provide a clean boundary. |
| Reinterpret legacy memberships as captures | A member may not be a URL capture, and automatic conversion would silently change user data semantics. |
| Reuse Media normalizers and scope service | Their models intentionally omit capture-owned detail and the Local implementation creates Media records. |
| Store offline pages as SQLite blobs | Large content would bloat the shared database and make backup, deletion, and partial-file recovery harder. |
| Show tags/domains as complete rail facets | The server list API does not provide aggregate facet counts, so the UI would imply completeness it cannot prove. |
| Build one generic Library data controller | It would move domain state into the shell and violate ADR-086's destination-ownership boundary. |

## Delivery boundary

Implementation planning may split TASK-18919 into atomic child tasks or ordered commits, but every
slice must leave the Library usable and may not restore the obsolete generic-container product.
The plan must include the required ADR block:

```text
ADR required: yes
ADR path: backlog/decisions/107-collections-capture-authority-and-legacy-boundary.md
Reason: TASK-18919 changes durable Collections storage, source authority, migration, service, and legacy-data boundaries.
```
