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
- Keep Local captures independent of Media records and server captures independent of the
  selected server profile, principal, and workspace of any other session.
- Keep every matching capture reachable through deterministic 20-row pages with an exact
  active-scope total.
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
- Adding or changing `tldw_server` endpoints.
- Claiming Local and Server feature parity where one authority lacks a capability.
- Creating another adaptive-pane framework instead of using the Library-owned shell.

## Authoritative product model

The `tldw_server` Reading List contract is the semantic reference. A capture has a URL and
canonical URL, title, domain, optional summary and readable representations, status,
favorite flag, tags, freeform note, timestamps, optional highlights, optional linked Notes,
and optional Media provenance. Supported reading statuses are `saved`, `reading`, `read`,
and `archived`.

The server may preserve a capture even when fetching or extracting the URL fails. Local
mode follows the same user-visible rule: capture persistence and content extraction are two
separate outcomes. A user never loses the bookmark merely because readable content could not
be produced.

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
- a clearly labelled legacy adapter may perform bounded listing/detail and complete streaming
  JSON export only; export uses a user-chosen validated path, never truncates at an internal page
  boundary, and never writes record content into logs;
- tool, RAG, Home, rail-count, and help inventories stop describing generic containers as the
  current Collections product;
- the configured `library_collections_db_path` remains the database location, so users do not
  acquire a second ambiguous Collections setting.

## Authority and identity

### Active authority

Collections follows the active Library source:

- **Local** — the capture tables in the resolved profile's configured Collections database;
- **Server** — the authenticated Reading List dataset for the selected server profile,
  authenticated principal, and workspace scope when present.

The source selector never offers a combined view. A source switch clears the prior page,
selected and loaded identities, detail, saved-search snapshot, and exact totals before requesting
the new source. An active archive/Undo receipt remains session-retained under its originating
authority but is hidden and inactionable under every other authority; switching back may restore
that exact receipt. It is never rebound to the newly active dataset.

### Authority-qualified identity

Every list row, detail request, cache entry, selection, mutation, and delayed completion is
qualified by an opaque authority key plus the source-owned item id. The authority key includes:

- Local profile and resolved Collections database identity; or
- Server profile, authenticated principal, and workspace/dataset identity.

Raw local paths and private principal identifiers are not displayed or logged. A compact,
non-reversible authority fingerprint may be used internally. Persisted layout preferences are
source-neutral; persisted or session-restored selection is accepted only when its authority key
matches the current authority.

Every asynchronous application fence contains at least:

`destination + authority + scope + item + mutation/content revision + generation`.

Unmount and source change advance the generation. A late success or failure that does not match
the complete fence cannot update the visible page, Reader, receipts, or actions.

## Local capture storage

### Additive schema v2

`LibraryCollectionsDB` advances additively to schema v2. The migration creates capture-owned
tables for:

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

The migration:

- acquires a bounded `BEGIN IMMEDIATE` migration lock;
- rechecks the current schema inside the transaction;
- creates all v2 objects and records version 2 atomically;
- never renames, deletes, or writes v1 generic-container rows;
- rolls back completely on failure; and
- leaves an older process able to read or mutate only the physically separate legacy tables.

The current `schema_version` table's maximum-version convention remains valid. Fresh creation,
real v1-to-v2 migration, failed migration rollback, concurrent-open behavior, and v2 reopen are
all tested.

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

This uniqueness does not cross Local profiles, database paths, server profiles, principals, or
workspaces.

### Extraction and managed files

Quick Capture writes the capture first with a durable processing state, then schedules extraction
off the Textual event loop. The extractor uses existing network-admission and path/security
boundaries, rejects unsafe URL schemes and SSRF targets, limits redirects and response size, and
sanitizes stored HTML. It does not create a Media record merely to obtain text.

Processing states distinguish queued/processing, ready, failed, and interrupted. On startup, a
stale in-flight state becomes **Interrupted · Retry** rather than remaining busy forever. Extraction
failure records a bounded user-safe reason while preserving the saved capture.

Large offline copies are managed private files, not unbounded SQLite blobs. The database stores
only validated relative ownership metadata, content hash, size, media type, and lifecycle state.
File creation uses a temporary sibling plus atomic replace. A failed file operation cannot claim an
offline copy exists. Local **Save Offline Copy** is capability-gated until this managed-file seam is
available.

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
and a source revision/generation marker when available. Search, filter, and sort always apply before
paging. A fresh out-of-range page gets one generation-guarded reload of the last valid page under
ADR-067; repeated shrink enters stale recovery rather than looping.

The exact total is shown for the active scope only. Tags and domains are typed filters and may use
suggestions from the current item or already returned rows. They are never advertised as complete
source-wide facets unless a future aggregate endpoint provides that evidence.

### Saved searches

Saved searches belong to the active authority and are listed through a bounded, paged contract.
Stored queries accept only the page request's defined filter keys, value types, and sort values.
Unknown keys, nested expressions, SQL fragments, and invalid sort names fail validation; arbitrary
JSON is never translated into SQL.

### Capabilities

The service returns explicit per-authority capabilities for capture, update, highlights, linked
Notes, summarize, listen, archive, offline copy, hard delete, and related recovery actions. A
disabled action remains visible with a source-owned reason. Permission denial, unsupported API,
offline state, missing dependency, and stale identity remain distinct reasons.

No Local implementation is added merely to make a capability table look symmetric. Server
endpoint absence or version mismatch downgrades that action truthfully without disabling ordinary
reading.

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
the committed capture before extraction completes. Server reports commitment only after an
authoritative save response. A transport failure with no response is **Save outcome unknown**, not
a confirmed failure or success; Retry is safe because canonical-URL upsert is idempotent.

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
undone. Missing external Media or Notes targets are not part of that deletion.

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
comfort cap. Custom width remains opt-in and uses the shared normalization. Responsive state never
persists.

Resolution follows the shared contract:

1. Work is permanent and both five-column grips remain reachable.
2. Library and Items use their requested widths when the active Work mode remains usable.
3. Responsive shortfall collapses Library before Items.
4. With Library collapsed and Items open, reclaimed width expands Items toward 56 columns before
   surplus goes to Work.
5. Explicitly opening a pane gives it temporary priority and may responsively collapse the other
   optional pane.
6. Widening restores requested panes and widths without rewriting preferences.

Representative outcomes are evidence, not hard-coded breakpoints:

| Terminal | Expected Collections outcome |
| --- | --- |
| 160x50 | Library + Items + Work; wide reading surface |
| 120x35 | Items + Work with Library responsively collapsed; full row detail |
| 100x30 | Deterministic Items + Work or Work-priority state with both grips |
| 80x24 | Focused Work with both restore controls and no horizontal overflow |

Focus follows Library, Library grip, Items controls/list, Items grip, Work toolbar/modes/content.
Automatic collapse evacuates focus to the corresponding visible grip. Escape closes transient
state first, then graduates outward through effective panes. No screen binding shadows terminal
conventions or the repository's global bindings.

## Verification

### Local repository and migration

- fresh v2 creation, real v1-to-v2 migration, rollback, reopen, and concurrent-open tests;
- v1 tables byte/row unchanged by migration and new capture operations;
- canonical-URL upsert, tag merge, explicit archived resave, content replacement, and revision CAS;
- exact count/page snapshot, stable tie-breaker, FTS query, page shrink, and malformed envelope;
- extraction ready/failed/interrupted transitions and managed-file atomicity;
- validated streaming legacy JSON export that includes every row beyond the first page.

### Shared Local/Server contract

One parameterized suite covers supported list, detail, save, update, favorite, status, tags, notes,
archive/Undo, saved searches, highlights, linked Notes, and capabilities. Source-specific tests
prove unsupported actions remain disabled with the correct reason. More than 40 captures prove the
second and third pages are reachable.

### Session and Textual behavior

- selected versus loaded identity and settle-delay/Enter behavior;
- source/profile/principal/database switches and complete late-response fencing;
- committed-save/follow-up-read failure, unknown server save outcome, stale refresh, and Retry;
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
