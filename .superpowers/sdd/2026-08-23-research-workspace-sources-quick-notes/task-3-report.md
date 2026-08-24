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
| `catalog_item_id` | canonical `Media.id`, stored and transported as a decimal string | canonical server `Media.id`, stored and transported as a decimal string |
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
  already-associated canonical item. Recovery copy is limited to Refresh
  status or Re-add; it never claims indexing was retriggered.
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

## Verification evidence

- Named Task 3 plus controller gate: `100 passed, 1 warning in 1.38s`.
- Touched-neighbor gate (association, RAG codecs/storage, Workspace registry,
  Media normalizers, legacy client, relevant app restore/wiring):
  `173 passed, 138 deselected, 1 warning in 5.73s`.
- The only warning is the accepted environment `RequestsDependencyWarning`;
  no new warning class appeared. Full pytest was not run.
- Scoped Ruff on changed production and focused tests: pass. The entire legacy
  `Tests/Library/test_library_ingest_runner.py` still reports its two unrelated
  pre-existing Ruff findings outside the changed lines; they were not edited.
- Ruff format check passes for all new Task 3 files. A whole changed-inventory
  format probe reports existing unformatted legacy files, so they were not
  mechanically reformatted into this already-large feature diff.
- Changed-production `compileall`, `git diff --check`, and the Impeccable
  detector on Research/user-facing recovery copy: pass.

## Inverse mutation evidence

- Hybrid `fts and vector` → `fts or vector`: the mode matrix failed. The first
  missing-embeddings probe initially stayed green because it used a Local
  membership ID instead of the canonical Media ID; that false-positive fixture
  was corrected and recorded in `lessons-testing-evidence.md`.
- Removed `{ok: true}` validation: the false-response/no-refetch test failed.
- Removed selection's source-generation invalidation: the late pre-write source
  refresh repainted reconciled selection and its controller test failed.
- Changed path quoting from `safe=""` to `safe="/"`: the Unicode/slash endpoint
  trace test failed.
- Dropped the parsed explicit-empty flag: the real SQLite restart test widened
  the Research scope and failed.

All mutations were reverted before the final gates.

## Files and boundaries

Production changes are limited to the Research contracts/adapters/controller,
source readiness and receipt scheduling, the canonical RAG/Workspace seam,
the existing notes-workspace API client/service schemas, and app composition.
No new database table, retrieval sentinel, parallel title/order store, retry
endpoint, or canonical Media deletion path was added.
