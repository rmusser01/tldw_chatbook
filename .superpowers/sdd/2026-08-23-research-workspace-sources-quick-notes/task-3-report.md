# Task 3 report — Research source browsing, readiness, and selection

## Status

Complete. Authority-specific source browsing, durable attach-existing,
readiness, desired selection, preview, removal, and Server ordering now have a
Textual-free domain/controller foundation. Task 4 UI work was not started.

ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: Task 3 implements ADR-078's accepted authority, canonical-owner, and
persistent desired-scope boundaries. The backward-compatible RAG-scope payload
seam described below is an implementation of that existing persistence decision,
not a new owner or sync policy.

## Pre-edit identity contract

| Domain field | Local owner and meaning | Server owner and meaning |
| --- | --- | --- |
| qualified workspace | `(local, workspace_id)` | `(server, profile_id, principal_id, workspace_id)` |
| `catalog_item_id` | canonical `Media.id`, stored and transported as a decimal string | canonical server `Media.id`, stored and transported as a decimal string; readiness alone may report `None` for an attached `missing_media` row, never a fabricated `0` identity |
| `catalog_item_version` | canonical Media version/last-modified projection when the Media owner exposes it | canonical Media version projection when the Media catalog seam exposes it; otherwise `None`, never copied from the association |
| `workspace_source_id` | `WorkspaceMembership.membership_id`; identifies only the association row | workspace-source response `id`; identifies only the association row |
| `workspace_source_version` | no independent Local association version; `None` | workspace-source response `version`; changes on update, selection, and reorder |
| desired selected ID | canonical Local `Media.id` in the workspace `RagScope` | server workspace-source `id` through `/sources/selection` |
| removal | unlink membership and remove its canonical Media ID from desired scope | delete workspace-source association and reconcile desired source IDs; canonical Media remains |

Normalized source rows retain all four identity/version fields. Code must never
substitute a Local membership ID for a Local Media ID or a Server Media ID for
a Server workspace-source ID.

## Pre-edit server and WebUI audit

- The real server endpoints are source list/create/update/delete, preview,
  status, selection, reorder, and capabilities. There is no workspace-source
  retry endpoint.
- Selection and reorder return only `{ok}`. Chatbook must validate `ok is true`,
  then list sources and reconcile the updated per-row versions, selection, and
  positions. It must not invent a response revision.
- The server lifecycle vocabulary is exactly `queued`, `ingesting`,
  `extracting`, `chunking`, `indexing`, `queryable`, `partially_queryable`,
  `failed`, `retrying`, `missing_media`, and `blocked_by_permissions`.
- Server selection updates all rows and can increment selected row versions
  twice; reorder increments each ordered row version once.
- The WebUI parity inventory preserved for Task 4 is: Quick URL add; simple
  search plus advanced filters and sort; select all, visible, and clear;
  selected count; Move/Copy, Preview selected, and Remove; folders and Select
  folder; row status/detail, add to folders, Preview & annotate, move up/down,
  and remove; modal tabs in exact order Upload, My Media, URL, Paste, Search
  Server.

## Pre-edit risks and guards

- Every adapter operation revalidates the exact qualified ref, and Server
  re-reads the active profile and principal immediately before dispatch.
- Local catalog and membership/Media SQLite work must stay off the event loop;
  Server must never fall back to Local or blend catalog results.
- Desired selection survives parsing, indexing, temporary unavailability, and
  restart. Effective retrieval is a derived intersection and cannot widen an
  explicit empty Research selection to unscoped retrieval.
- The smallest compatible RAG seam will add an explicit-empty ownership flag to
  the canonical payload. Ordinary Console zero-selection save remains its
  existing clear/unscoped behavior.
- Readiness retry only clears the failed readiness receipt and refreshes the
  already-associated canonical item. Recovery copy may direct the user to
  refresh status, re-add/restore the source, or review permissions; it never
  claims indexing was retriggered.
- Startup readiness work uses its own stage-specific SQL predicate before
  `LIMIT`, coexists with Task 2's association startup bound, and remains
  suppressed during bulk restore.
- Controller caches and visible source state are keyed by qualified identity
  and fenced by monotonic generation, context revision, and capability
  revision so Local→Server→Local ABA results cannot repaint current state.

## Implementation

- Extended the port with the nine approved source operations. Local uses
  `WorkspaceMembership(role=source,item_type=media)` and canonical Media; Server
  uses the audited workspace-source rows and selected profile/principal only.
- Added bounded Local/Server catalog browsing through the real
  `MediaReadingScopeService` with an explicit mode on every call. Local SQLite
  aggregation runs through `asyncio.to_thread`; no adapter fallback or result
  blending exists.
- Added strict Pydantic request/response contracts for every touched source
  path. Opaque Unicode/path IDs use `quote(..., safe="")`; selection and reorder
  validate the real `{ok: true}` response, then GET sources to reconcile row
  versions, order, and selection. DELETE validates the actual empty 204 body.
- Added closed readiness normalization and mode-specific effective retrieval.
  FTS requires FTS, semantic requires vector, and Hybrid requires both. Secret-
  shaped diagnostic text is withheld. Retry only clears the readiness receipt
  and rechecks the existing association.
- Added a stage-specific readiness startup query whose association/readiness
  predicates precede `LIMIT`. One startup worker resumes bounded association
  work first and bounded readiness work second, preserving bulk-restore
  suppression and live scheduling.
- Added controller surface generations for sources, catalog, readiness,
  preview, and selection. Canonical caches are qualified by workspace ref and
  owner ID, and only a current result may enter a source cache or visible state.

## Explicit-empty persistence seam

`RagScope` payload version 2 adds `empty_is_scoped`. Research persists
`items=[]` plus `empty_is_scoped=true`, so restart/corruption recovery remains
fail-closed instead of widening Select none to unscoped retrieval. Version 1
payloads remain readable. Ordinary Console zero-selection continues to clear
the scope because its default flag remains false. No sentinel source ID or
device-overlay owner was introduced.

An existing malformed `workspace_rag_scopes` row now reads as that same
explicit-empty state at the Workspace registry seam. The shared conversation
codec remains unchanged: malformed conversation metadata still reads as
legacy unscoped state.

## Fix round 1 hardening

- Removed the silent `ResearchSourceSummary.catalog_item_id` fallback so the
  catalog and association identity axes cannot collapse.
- Accepted the real unpaged Server source/status projections up to a finite
  10,100-row owner bound (public pages and mutation requests remain at 100).
  Local readiness receipts use a keyed membership lookup, so an association
  after row 100 still converges without widening the public page.
- Server catalog pages now validate page coordinates, row counts, and stable
  totals, and stitch the next 100-row backing page when an offset/limit crosses
  a boundary. Local `updated_*` sorts translate to the canonical Media owner's
  `last_modified_*` vocabulary.
- A non-`None` removal version fails typed before Local storage or Server
  dispatch because neither real delete owner can enforce that precondition.
- Server desired selection is reconciled with the association row's returned
  version before the association receipt succeeds. A failed update preserves
  catalog success and leaves association retryable across restart; the former
  post-terminal adapter selection workaround was removed.
- `missing_media` readiness preserves the workspace-source association ID while
  carrying no canonical ID for `media_id=None/0`. Permission, missing-media,
  and vector-failure recovery copy stays bounded and does not advertise a fake
  retry endpoint.

## Fix round 2 reconciliation

- Added `SourceSelectionResult`, deliberately separate from
  `BoundedPageResult`. It carries exact ordered, unique desired owner IDs up to
  the audited 10,100-row owner cap and at most 100 reconciled selected source
  rows. It does not claim that an owner projection is one visible page.
- Local selection now validates up to 10,100 attached canonical Media IDs,
  persists them in the canonical workspace `RagScope`, and reads that owner
  state back directly. It never lists page 1 to prove selection, so canonical
  item 101 and Select all beyond 100 survive reconciliation.
- Server selection accepts the same finite owner cap, validates every refetched
  row's workspace/association/boolean identity, derives the exact selected ID
  tuple from the post-PUT owner projection, and returns only the first 100
  selected rows as optional cache evidence.
- Controller selection validates the qualified result and exact desired ID set,
  preserves the current visible page, invalidates an older source refresh, and
  updates canonical rows only for the bounded rows the owner returned.
- Server readiness now validates the top-level `workspace_id` and every status
  row's `workspace_id` before normalization. A typed identity mismatch leaves
  the readiness receipt pending and retryable with zero Local calls; nonidentity
  refresh failures still settle terminal as `readiness_refresh_failed`.
- A valid Server `missing_media` preview now retains its required association ID
  with `catalog_item_id=None`. The nullable identity is allowed only for Server
  `missing_media`/unavailable modes, and the controller cache remains keyed by
  qualified workspace plus association ID.
- Updated the older lifecycle-capability test to account for the nine source
  capabilities added by Task 3 while still proving an unimplemented source
  service reports those capabilities unavailable rather than inferred.

## WebUI parity mapping retained for Task 4

| Existing WebUI control | Task 3 foundation |
| --- | --- |
| Search, advanced type filters, sort, pagination | bounded `search_catalog` |
| Select all / visible / clear and selected count | persistent `set_selected_scope` plus normalized selected rows |
| Preview selected; Preview & annotate entry point | bounded `preview_source`; annotation UI remains Task 4 |
| Remove | association-only `remove_source`; never canonical Media deletion |
| Row status/detail | normalized `get_readiness` with exact lifecycle and recovery copy |
| Move up/down | Server `reorder_sources`; Local exposes typed unavailable because no canonical Local order owner exists |
| Quick URL add; Upload, URL, Paste | durable Task 2 ingestion/source-operation route; UI remains Task 4 |
| My Media; Search Server | authority-explicit `search_catalog` |
| Move/Copy, folders, Select folder, add to folders | not falsely advertised by this port; Task 4 must disable with a typed capability until a real owner contract exists |

The audited modal tab order remains exactly: Upload, My Media, URL, Paste,
Search Server.

## Fix round 3 — exact owner pages and write preconditions

- Added `ResearchSourcePage`, a source-specific bounded-page projection that
  carries the exact ordered owner `desired_source_ids` independently of the
  visible rows. Local restart and both Server pages preserve a selection that
  exists only at row 101; the generic bounded-page contract remains unchanged.
- Controller source navigation consumes only the exact owner projection. A
  second source-generation increment after a current selection write prevents
  a refresh started during that write from repainting stale desired state.
- Split malformed readiness projections from exact workspace-identity
  mismatches. Invalid containers, row shapes, and bounds now follow the normal
  terminal `readiness_refresh_failed` path; exact top-level or row workspace
  mismatches alone remain typed and retryable.
- Preserved the actual server's `media_id=0` missing-media transport shape while
  normalizing `None`/`0` to no canonical identity at the adapter boundary.
  Available preview/status rows still reject zero, and domain objects reject a
  fabricated canonical ID `"0"`.
- Server reorder now reads and validates the exact owner before mutation. An
  owner or request that cannot express one exact order within the 100-ID write
  contract fails typed `reorder_precondition_unavailable` with no PUT; bounded
  exact owners retain the existing PUT-then-GET reconciliation.

## Fix round 4 — atomic Local desired-selection ownership

- Replaced the coordinator's split `get_workspace_scope` / mutate /
  `set_workspace_scope` sequence with one
  `LocalWorkspaceRegistryService.reconcile_research_source_selection` owner
  operation. The coordinator calls it off the event loop after the canonical
  Media membership exists.
- A missing scope remains the canonical implicit-all state when the new source
  is desired. Deselecting from that state materializes every other attached
  Media/Note source; an explicit scope changes only the target Media item; a
  malformed row starts from explicit empty and adds the target only when
  desired.
- `WorkspaceDB.transaction` now exposes the same opt-in `immediate` write-lock
  mode used by the repository's other read-then-write SQLite owners. The new
  registry operation reserves that lock before its first membership/scope
  read, so different source operations cannot both commit from one stale
  desired-scope snapshot.

## Verification evidence

- Round 4 complete Research Workspace gate: `243 passed, 1 warning in 2.09s`.
- Round 4 Workspace registry/DB neighbor gate: `98 passed, 1 warning in 1.16s`.
- Round 4 scoped Ruff, changed-file `compileall`, `git diff --check`, and the
  Impeccable detector: pass. The scoped format probe found no new production
  candidate; the remaining three whole-file legacy candidates are also
  candidates at `6a23c6ff1`.
- Round 4's only warning is the accepted environment
  `RequestsDependencyWarning`. Full pytest was not run, per repository policy.

- Round 3 focused owner/consumer gate: `219 passed, 1 warning in 2.79s`.
- Round 3 independent Research neighbor gate: `265 passed, 1 warning in 4.01s`.
- Round 3 exact Task 3 Library-ingest/DB neighbor gate:
  `275 passed, 1 skipped, 1 warning in 19.60s`. The skip is the existing
  Windows spawn/resource-tracker boundary.
- Round 3 scoped Ruff, changed-production `compileall`, `git diff --check`, and
  the Impeccable detector: pass. The scoped format probe identifies the same 10
  whole-file legacy candidates at `67114c7ed`; no formatting churn was added.
- Full pytest was not run, per repository policy.

- Round 2 focused gate (contracts, controller, Local/Server adapters,
  association/readiness/selection, workspace adapters, source client, and Notes
  service): `199 passed, 1 warning in 2.27s`.
- Expanded restored-tree owner/consumer gate excluding the known high-FD
  Library-ingest process suite: `440 passed, 1 warning in 11.21s`.
- Library-ingest runner gate: `145 passed, 1 skipped, 1 warning in 15.95s`.
  The skip is the existing Windows spawn/resource-tracker boundary.
- The split gates contain only the accepted environment
  `RequestsDependencyWarning`. A diagnostic combined run reported FD growth
  +208; isolation with `TLDW_TEST_FD_GROWTH_LIMIT=0` showed the unchanged
  Library ingest test process owns +190 while all six Round 2 focused files own
  +3. This is preserved as baseline fixture evidence, not hidden by raising the
  sentinel. Full pytest was not run.
- Scoped Ruff on all fix-round production and focused tests: pass. The format
  probe still reports 12 existing legacy files as whole-file reformat
  candidates, so they were not mechanically reformatted into this focused fix;
  no whole-file formatting churn was introduced.
- Changed-production `compileall`, `git diff --check`, and the Impeccable
  detector on Research/user-facing recovery copy: pass.

## Inverse mutation evidence

- Corrupt Workspace JSON returning legacy `None` made both the restart and
  registry fail-closed guards red.
- Inventing catalog ID `"0"` for `missing_media` made both nullable cases red.
- Restoring the 100-row owner cap made Server list, readiness, and client
  projection guards red; restoring first-page lookup made the keyed Local
  row-101 adapter and durable receipt guards red.
- Removing Local/Server delete precondition rejection made both storage and
  dispatch guards red.
- Removing duplicate-association selection/version reconciliation let the
  association receipt become terminal before desired selection and made its
  restart test red.
- Disabling the second backing-page fetch truncated offset 90 / limit 25 and
  made the cross-page stitch guard red.
- Passing UI `updated_*` sort terms to the Local owner broke the real SQLite
  order and exact owner-call trace.
- Restoring the silent association-ID fallback for a missing catalog ID made
  the distinct-identity constructor guard red.
- Advertising `Retry indexing` made the exact recovery-copy and closed
  lifecycle guards red because no source retry endpoint exists.

All nine reviewed defect families were reverted before the restored-tree gates.

Round 2 added four inverse checks, all restored before the final gates:

- Rebuilding the visible source page from a selection result replaced the
  existing 101-row owner page with a synthetic one-row page and made the page
  preservation guard red.
- Removing top-level readiness workspace validation made the exact top-level
  mismatch guard red; removing the coordinator's typed identity branch made the
  pending-receipt/no-Local guard red.
- Converting `media_id=None` back to `""` made the actual-shaped missing-media
  preview fixture fail at the domain boundary.

Round 3 added seven independent inverse checks across all four reviewed
families, all restored before the final gates:

- Replacing page-level exact desired IDs with the bounded visible selection
  made the row-101 navigation guard red; removing the post-write source
  generation increment made the selection-versus-refresh race guard red.
- Reclassifying malformed readiness as an identity mismatch made the paired
  adapter and coordinator terminal-state guards red.
- Removing zero-to-null adapter normalization, restoring `ge=1` independently
  on status and preview transport schemas, and accepting domain ID `"0"` each
  made their actual-shape boundary guard red.
- Raising the reorder precondition from 100 to the owner cap allowed a
  101-source PUT and made the exact no-mutation trace guard red.

Round 4 added one owner-boundary inverse, restored before the final gates:

- Replacing only `BEGIN IMMEDIATE` on the atomic Local reconciliation with a
  deferred transaction made the barrier-started real-file SQLite concurrency
  guard fail with `database is locked`; restoring the reservation made both
  source updates persist without loss.

## Files and boundaries

Production changes are limited to the Research contracts/adapters/controller,
source readiness and receipt scheduling, the canonical RAG/Workspace seam,
the existing notes-workspace API client/service schemas, and app composition.
No new database table, retrieval sentinel, parallel title/order store, retry
endpoint, or canonical Media deletion path was added.
