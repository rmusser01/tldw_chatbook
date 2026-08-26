# Library Prompt Enhancement Series Design

**Date:** 2026-08-02
**Status:** Approved decisions; three review passes completed and final findings addressed
**Scope:** `tldw_chatbook` TASK-202, TASK-196, TASK-198, TASK-199,
TASK-197, and TASK-203 as six sequential, merge-gated pull requests

## Goal

Complete the deferred Library Prompt enhancements on top of the merged Console
Prompt Workbench without weakening Prompt/Recipe identity, source authority,
structured-artifact fidelity, or Console application safety.

Success means:

- Library Prompt editor actions remain visible, functional, and clearly grouped.
- Retained prompt versions can be inspected and safely restored as a new current
  version.
- Named collections support complete, paginated browsing and explicit
  membership management.
- `{variable}` placeholders are filled through one shared dialog across Console
  slash insertion, Console picker insertion, and Library insertion.
- Chatbook archives carry modern Prompt and Recipe records without collapsing
  their System/User lanes or structured definitions.
- Library Prompt rows support bounded multi-select, bulk Chatbook export, and
  confirmed bulk delete.
- Each enhancement lands independently and is reviewed, tested, and merged into
  `dev` before the next enhancement begins.

## Approved Product Decisions

The user approved these decisions during design:

- Deliver exactly six sequential PRs in foundation-first order:
  TASK-202, TASK-196, TASK-198, TASK-199, TASK-197, TASK-203.
- Merge-gate the sequence. A later task does not begin implementation until the
  preceding PR is merged and the latest `dev` is confirmed.
- Use one shared variable-fill dialog for System and User lanes so a variable
  such as `{customer}` receives one value everywhere it is active.
- System application remains optional and off by default.
- Library and Console use the same variable grammar and render operation.
- Prompts and Recipes remain first-class Library records. Recipes cannot execute
  directly and still create unsaved Prompt working copies when selected for use.
- Multi-select v1 covers selection, bulk export, and bulk delete. Bulk tagging
  remains outside TASK-203 because it is not part of that task's acceptance
  criteria.
- No visual companion is required for this design; the approved direction is
  text-only and follows the existing Neon Workbench design system.

## Non-goals

- A new Prompt or Recipe table.
- A second Prompt scope facade or direct widget-to-database calls.
- Cross-source collections or merged local/server collection results.
- Collection deletion, collection Chatbook portability, or bulk tagging.
- Persisted variable defaults, variable values, or model-generated variable
  values.
- Recursive template expansion, expressions, format specifications, attribute
  lookup, or arbitrary Python formatting syntax.
- Exporting version history, usage counters, sync state, local database IDs,
  timestamps, or collection membership inside a Prompt record.
- Server API changes in this six-PR client series. Existing server seams are
  used when truthful; absent capabilities remain visibly unavailable.
- Starting later PR work speculatively while an earlier PR is in review or CI.

## Existing Baseline and Problems Found During Review

The series starts from the merged Prompt Workbench on `dev` and is governed by
ADR-011 and ADR-040.

Important baseline findings shape the design:

- `LibraryPromptsListCanvas` renders six flat actions after a tall editor. An
  existing regression records that the action row can sit below the viewport,
  and the current Copy text action has no handler.
- The standalone local prompt adapter can reconstruct history from `sync_log`,
  but the application-wired `PromptScopeService` rejects local history and its
  in-module local adapter does not expose the history methods.
- The current history helper scans the entire sync log even though the database
  already indexes `entity_uuid`.
- The Library Prompt list loads at most 100 rows, then filters and sorts that
  snapshot in memory. That cannot truthfully implement collection filtering or
  complete search for a growing library.
- Local and server collection CRUD seams already exist behind
  `PromptScopeService`, but there is no cross-source atomic membership contract.
- Server template utilities recognize `{{name}}`, whereas the approved user
  grammar is `{name}` with doubled braces as literal escaping. The UI must not
  depend on the server utility.
- Library Prompt insertion currently stages only a bare User string. It cannot
  carry optional System replacement or a guarded shared-variable result.
- Chatbook already defines a Prompt content type, but the local exporter writes
  one collapsed `content` field and drops keywords, artifact type, structured
  definition, and separate lanes. The importer reconstructs that collapsed
  field as a System prompt.
- Existing Library multi-select behavior for notes, media, and conversations
  provides a reusable `RowSelection` pattern, but Prompt rows do not use it.

## Governing ADRs and Required ADR Work

| Task | ADR required | Path | Reason |
| --- | --- | --- | --- |
| TASK-202 | No new ADR | ADR-011 and ADR-040 | UI grouping and bug repair preserve existing boundaries. |
| TASK-196 | Yes | Allocate the next available `backlog/decisions/NNN-local-prompt-retained-version-history.md` when the task starts | Makes retained sync-log payloads a user-visible history source and defines restore, pruning, keyword, and concurrency semantics. |
| TASK-198 | No new ADR | ADR-011 and ADR-040 | Surfaces existing local collection storage through the existing scope; case-fold duplicate prevention is explicitly service-level and makes no schema or migration change. |
| TASK-199 | Yes | Allocate the next available `backlog/decisions/NNN-prompt-variable-grammar-and-guarded-insertion.md` when the task starts | Defines a durable placeholder grammar and a cross-module Console application contract. |
| TASK-197 | Yes | Allocate the next available `backlog/decisions/NNN-chatbook-prompt-record-contract.md` when the task starts | Changes the portable archive record and import/export service contract. |
| TASK-203 | No new ADR | ADR-011, ADR-040, and TASK-197's Chatbook ADR | Adds UI selection and uses already-decided bulk service boundaries. |

ADR numbers are allocated from the latest merged `dev` at the start of the
corresponding task to avoid collisions with concurrently merged decisions. The
ADR is committed before implementation code in that task's branch and linked
from its Backlog implementation plan and notes.

## Cross-cutting Architecture

### Authority boundaries

- `PromptScopeService` is the only Library-facing boundary for Prompt CRUD,
  retained versions, collections, paginated browse, and bulk deletion.
- Local adapters own SQLite queries and transactions. Server adapters own API
  calls. Widgets receive normalized state and emit typed intent.
- `PromptChatbookScopeService` remains a compatibility facade for its current
  call sites; the new Library features do not add another route through it.
- Chatbook serialization remains in `Chatbooks/` and the local Chatbook service.
  The Library builds an export scope but does not construct archive JSON.
- Template extraction and rendering live in one Textual-independent module.
  The dialog does not implement parsing and hosts do not implement rendering.
- The Console composer and session store remain the only owners of composer and
  System-prompt mutation.

### State shape

New UI state is immutable where practical and separates observed state from
intent:

- `PromptBrowseScope`: the fixed local backend, query, collection ID, sort,
  page, and page size. The Library exposes no source selector in this series.
- `PromptBrowseResult`: normalized items, exact total, page metadata, and scope
  fingerprint.
- `PromptHistoryState`: retained version summaries, selected preview, paging,
  loading/error state, and captured current version.
- `PromptVariableSpec`: exact variable name and the active lanes in which it
  appears.
- `PromptVariableApplication`: rendered lanes, lane-application flags,
  destination behavior, session guard, and optional System fingerprint. Raw
  values are not retained after rendering.
- `PromptBulkOutcome`: successful identities, failed identities with bounded
  error categories, and optional export artifact result.

### Async and stale-result handling

- Synchronous local DB work that may exceed 100 ms runs in a thread worker.
- Async server work is awaited in a Textual worker without blocking the event
  loop.
- Browse, history, collection, and export workers use exclusive groups plus
  monotonic request tokens or immutable scope fingerprints.
- A late result may not overwrite state for a newer Prompt, query, collection,
  page, source, or selection.
- Cancellation is distinct from empty results and failure.

### Failure taxonomy

The UI distinguishes loading, empty, no matches, unavailable capability,
policy denial, malformed artifact, validation failure, stale version, partial
batch failure, I/O failure, cancelled, and expired handoff. It never turns a
service exception into a credible-looking empty Prompt Library.

## TASK-202 — Group and Repair the Editor Action Area

### User experience

The editor becomes a two-row layout contract: scrollable content occupies the
remaining height and an auto-height action area occupies the bottom. This is a
grid or equivalent parent layout, not a docked overlay, so the actions never
cover the final field and do not create a nested-scroll dead zone.

Actions are grouped in logical and keyboard order:

1. Primary: Save Prompt, Save Recipe, or Update original.
2. Content: Use in Console, Export, Copy Markdown.
3. Lifecycle: Duplicate and Delete.

At narrow widths the groups stack without changing their order. Save receives
the primary visual treatment. Delete receives the existing danger treatment.
Conflict state replaces the normal actions with Save as new and Reload while
keeping those actions equally visible.

Copy Markdown reads the live editor working copy, including unsaved edits,
renders the same Markdown System/User representation as individual export, and
uses the app's clipboard seam. Success is announced only after the clipboard
call succeeds. Missing or failing clipboard support has an explicit warning or
error.

Single delete uses the same confirmation component later reused by bulk delete.
If the editor is dirty, confirmation states that both the saved artifact and
the unsaved working copy will be discarded.

### Scope correction

TASK-202 absorbs the directly overlapping defects in TASK-2700 and TASK-2701.
All three task records travel in the same PR. Before code changes, TASK-202,
TASK-2700, and TASK-2701 are each put In Progress and receive their own concise
implementation plan and ADR check; the two defect plans link to the TASK-202
series plan instead of creating separate branches. Before the PR is declared
complete, each record receives its own checked acceptance criteria,
verification evidence, implementation notes, and Done transition. This is one
implementation PR, not three PRs and not an exception to Backlog hygiene.
TASK-2702, the broader dirty-navigation feedback issue, remains separate.

### Updated acceptance boundary

- Actions are grouped by primary, content, and lifecycle purpose.
- Save is visually distinguishable.
- No existing action is removed or left inert.
- Actions are visible at 200x50 and scroll-reachable at shorter sizes.
- Copy Markdown succeeds through the clipboard seam and reports unavailable or
  failed clipboard states honestly.
- Single delete is confirmed, including dirty-working-copy consequences.
- A geometry regression test and user-guide correction close TASK-2701.

## TASK-196 — Retained Prompt Version History

### Storage and service contract

Add a bounded database query for Prompt sync-log entries by entity UUID,
operation, and descending change ID. Add a schema migration with a composite
index suitable for that query. History never calls
`get_sync_log_entries(since_change_id=0)` and filters the entire result in
Python.

Each page query reads at most `page_size + 1` retained create/update snapshots.
The extra older snapshot is used only as the predecessor for the last visible
row's changed-field summary. A row is compared only with its immediately
preceding version. Version gaps caused by pruning show `Earlier baseline
unavailable` rather than comparing non-adjacent snapshots and claiming a false
diff. Version 1 shows `Created`.

History pages include create/update entries only. Each normalized snapshot
contains:

- version, change ID, operation, and timestamp;
- name, author, description, System lane, and User lane;
- Prompt format, Prompt schema version, definition, and artifact type;
- keywords when captured by a modern snapshot;
- an explicit flag when keywords were not captured historically.

Future create/update sync payloads include the effective keywords after keyword
membership has settled in the same transaction, without changing keyword-link
ownership. The row, keyword links, and snapshot either commit together or roll
back together. Older consumers must ignore the additive field. Collection
membership, usage counters, and deletion state are not versioned by this UI.

`PromptScopeService.list_prompt_versions` routes both local and server sources
to their adapters and normalizes one version envelope. The Library uses only
the local path in this six-PR series. Server history behavior may remain
available to other consumers through the same seam, but no Library source
selector or server-history UI is introduced.

Restore accepts the target snapshot version and the expected current version.
Local restore calls the ordinary conditional Prompt update with the snapshot's
artifact fields. It creates a new sync-log update and therefore a new current
version. It never changes history in place.

### User experience

The editor contains a collapsed `Retained history (N)` disclosure. The first
page loads only when opened. `Load older versions` retrieves another bounded
page.

A version row shows version, timestamp, artifact type, and changed-field
summary. Selecting it shows normalized read-only metadata and System/User
previews. Malformed, foreign-v1, and unsupported definitions use the existing
compatibility states rather than being interpreted as v2.

Restore is enabled only for a snapshot that normalizes as valid legacy text or
as a valid structured-v2 Prompt/Recipe under the current local capability and
ADR-040 rules. Malformed JSON, definition/compiled-field mismatch,
artifact-type/kind mismatch, unknown format or schema version, unsupported
future artifact type, and foreign structured-v1 snapshots remain preview-only;
Restore is disabled with the exact compatibility reason. Foreign-v1 content is
never reparsed, converted, or written through Restore. Its existing explicit
Save-as-new conversion remains the only path into an editable v2 artifact.

Viewing is allowed while the working copy is dirty. Restore is disabled until
the user saves or discards the working copy. Confirmation states that restore
creates a new current version and calls out a Prompt-to-Recipe or
Recipe-to-Prompt type change. Restoring the current byte-identical content is
reported as no change and creates no new version.

Immediately before restore, the service rechecks the expected current version.
A mismatch produces the normal conflict state and requires Reload. Success
reports, for example, `Restored v3 as current v8.` Older snapshots without
keywords retain the current keywords and disclose that behavior. A modern
snapshot with captured keywords restores them in the same conditional
transaction as the artifact fields. A duplicate-name conflict or keyword
validation failure changes neither the current Prompt nor its history and
leaves the selected version available for correction or retry.

The UI consistently says retained history because sync-log cleanup may limit
the oldest available version.

### Updated acceptance boundary

- The app-wired `PromptScopeService`, not a direct local adapter call, exposes
  bounded retained history.
- History survives ordinary create, edit, and restore operations while its
  backing entries remain retained.
- Restore is conditional, appends a new version, and reports the source and new
  version distinctly.
- Prompt/Recipe identity and v1/v2 compatibility rules survive preview and
  restore.
- History query performance is index-backed and independent of unrelated
  sync-log volume.

## TASK-198 — Collections and Complete Prompt Browsing

### Paginated browse seam

Add a dedicated `PromptScopeService.browse_prompts` contract. Do not overload
the existing bounded `search_prompts` method used by Console command/picker
flows because that method does not expose reliable pagination totals.

The local browse adapter performs query, optional collection membership,
whitelisted sorting, pagination, and exact counting in SQLite. A collection
query joins collection membership before applying search and pagination; it
never filters the current 100-row UI snapshot. The fixed sort whitelist
prevents user-controlled SQL identifiers.

The Library Prompt canvas moves from `_local_source_records`'s sampled Prompt
tuple to its own browse state. Search is debounced, scope-token guarded, and
shows exact page/total information. The existing rail count may continue using
the lightweight local count seam.

The Library Prompt canvas remains local-only throughout this series and calls
the scope with `mode="local"`; it adds no source selector and never presents
server rows as part of an All or collection result. Existing server collection
methods remain routed through `PromptScopeService` for their current callers,
but TASK-198 adds no server collection UI. Cross-source browsing, membership,
and export require a separate end-to-end contract and remain out of scope.

### Collection management

The list toolbar contains:

- collection selector, default `All prompts`;
- `New collection...`;
- search, sort, and page controls.

The collection selector is complete as well: it uses the collection service's
exact total and bounded pages rather than stopping at the existing 200-row
default. A small collection set renders directly; a larger set offers search
and `Load more` within the chooser. Pre-existing case-fold name collisions are
disambiguated as `Name · #id` in chooser and manager labels while writes remain
ID-based.

Search applies within the active collection. Changing collection, query, sort,
or page creates a new browse-scope fingerprint.

Collection create and rename use a compact in-canvas management state. Names
are non-empty. Every successful create or rename through
`PromptScopeService` is checked case-insensitively against other active local
collections inside the same serialized write transaction; rename excludes its
own ID.
This is a service-level user-facing validation contract, not a storage
migration or claim that the existing case-sensitive `UNIQUE` constraint has
changed. Pre-existing case-fold collisions are not silently renamed or
deleted: both remain visible by ID, new conflicting writes are blocked, and
the user can resolve them by renaming. Collection deletion is not exposed.

Membership is a separate action from Prompt Save. The editor shows current
collections and opens the same membership manager, but `Apply membership`
produces its own outcome. Local membership replacement is one transaction.
Collection-centric editing updates one collection at a time. All membership
IDs are validated as active local Prompt/Recipe rows before settlement; a
failed validation rolls back the local membership change.

### Updated acceptance boundary

- Named collections can be created, renamed, browsed, and assigned.
- All and collection views are complete, service-backed, searchable, and
  paginated.
- One artifact may belong to multiple collections.
- Prompt content Save and membership Apply are visibly separate outcomes.
- Library collection calls remain local and route through
  `PromptScopeService`; no server or mixed-source result is implied.

## TASK-199 — Shared Prompt Variables and Guarded Insertion

### Grammar

One pure module defines the durable grammar:

- `{customer}` is a variable when the name matches
  `[A-Za-z_][A-Za-z0-9_]*`.
- Names are exact and case-sensitive.
- `{{` emits a literal `{`; `}}` emits a literal `}`.
- Other braces, including ordinary JSON/XML braces, remain literal.
- Rendering is single-pass. Braces introduced by a value are never reparsed.
- The parser is deterministic, preserves first-occurrence order across the
  active lanes, and accepts at most 64 unique variables with names no longer
  than 64 characters.
- Invalid or unmatched non-variable brace text remains literal; it is not an
  expression error.

A syntactically valid placeholder beyond either limit produces a validation
state rather than silently becoming literal or being truncated. The dialog
offers `Use original placeholders` or Cancel; it does not render a partial
variable set.

The implementation uses a lexer/state machine rather than repeated regular
expression substitution so escaped braces cannot reveal an inner variable.
The ADR includes a table for adjacent, escaped, nested-looking, unmatched, and
triple-brace cases.

### Shared dialog

One `PromptVariablesDialog` serves:

- exact `/prompt name` resolution;
- selection from the Console Prompt picker;
- Library `Use in Console`.

It receives compiled System/User text and destination behavior, extracts the
active variables, and shows each unique name once with a lane-use label. Values
are blank by default and may remain blank. The dialog is scrollable when the
variable count is large.

Destination copy is exact. A resolved `/prompt` command and a Console Prompt
picker choice both replace the entire composer snapshot captured when that
flow opened. If the picker was opened from the command palette over ordinary
draft text, the dialog says `Replace the current Console draft`; it does not
mislabel that draft as a `/prompt` command. Library `Use in Console` says
`Append to the current Console draft`. The captured snapshot is the one later
checked by the stale-composer guard.

When a System lane exists, the checkbox reads:

`Replace the current session System prompt with this System lane`

It is off by default. Toggling it recomputes the active variable list while
preserving ephemeral values for variables that remain or later reappear. The
dialog also states whether User text will replace the `/prompt` command draft
or append to the Console draft from Library.

If System replacement is off and there is no applicable User lane, the primary
Apply action and `Use original placeholders` are both disabled with a short
`Select a lane to apply` explanation. Cancel remains enabled. Neither apply
path may become a no-op or bypass explicit System authorization.

Primary application renders the ephemeral copy. A secondary `Use original
placeholders` action applies the selected lanes without interpolation. This is
the backward-compatible escape hatch for existing prompts whose `{x}` text was
not authored as a variable. Cancel mutates nothing and leaves a slash-command
draft intact.

If a Prompt has no recognized variables and no System lane, insertion retains
the current direct fast path. A System lane alone still opens the dialog so the
user can authorize or decline replacement.

Recipes remain non-executable. Selecting a Recipe keeps the existing unsaved
Prompt-copy flow before any variable application is possible.

### Typed application request

Extend the memory-only `CONSOLE_PROMPT_INSERT` handoff from a string to a
validated, detached `PromptVariableApplication` type. It stores rendered lanes,
application flags, destination behavior, target session, authorization-time
System fingerprint, creation time, and expiry. It does not store the raw
variable map, source Prompt body, or values separately; sensitive fields are
excluded from representation and logs.

The handoff is latest-wins, one-shot, owner-thread-only, and is expired when
monotonic elapsed time is greater than or equal to 120 seconds. Console checks
expiry both when claiming and before applying, then acknowledges and discards
expired or wrong-session requests with a warning. A transient missing composer
releases the claim for retry only while it remains valid.

For Library append, Console captures the composer snapshot when consuming the
handoff so the text appends to the settled active draft. It rechecks the
System fingerprint captured when the user authorized replacement. Slash flow
captures the command-bearing composer snapshot before the dialog and refuses
application if it changes.

The existing Console prompt transaction applies the rendered lanes. In-memory
composer/System changes are coordinated and reversible. Durable conversation
persistence remains a separate outcome: a live System update with a failed
write is reported honestly rather than described as an atomic rollback.

### Updated acceptance boundary

- `{name}` placeholders are detected and filled at insertion time only.
- Literal braces and literal `{name}` text remain expressible.
- Slash, picker, and Library insertion use the same parser and dialog.
- One value is reused across active System and User occurrences.
- System replacement is separately authorized and defaults off.
- Cancel, expiry, stale session, and stale composer/System state apply nothing.
- No variable values or rendered bodies are logged or persisted as defaults.

## TASK-197 — Lossless Chatbook Prompt Records and Prompt Export Scope

### Record contract

Modern `content/prompts/prompt_<id>.json` files add
`chatbook_prompt_record_version: 1`. This field is distinct from the artifact's
`prompt_schema_version`.

This versioned record is produced and consumed by the local Chatbook service in
this client series. The Library's existing server-mode export gate remains in
place, and the UI does not claim that server-created archives have adopted the
new record until a server contract says so.

The canonical record preserves:

- name, author, and description/details;
- separate `system_prompt` and `user_prompt` fields;
- keywords as an ordered string list;
- `artifact_type`;
- `prompt_format`, `prompt_schema_version`, and complete
  `prompt_definition`;
- a compatibility `content` projection for older readers.

The compatibility projection is explicitly lossy and exists only so older
Chatbook importers do not create an empty Prompt. It is constructed exactly by
this formula, where a missing lane is the empty string and lane text is not
trimmed or line-ending-normalized:

```python
content = (
    "### SYSTEM ###\n"
    + system_prompt
    + "\n### USER ###\n"
    + user_prompt
    + "\n"
)
```

The inserted section delimiters use LF. If a preserved lane already ends in a
newline, the formula intentionally produces an empty line before the next
delimiter or record end. No metadata or structured definition is added to
`content`, and no lane is chosen over the other. An older importer will still
flatten this whole projection into its System field; that known loss is
preferable to silently discarding either lane. Modern readers never parse
`content`: they dispatch by the record-version field and use the canonical
fields. A missing record version uses the existing legacy flattened importer.

Semantic round trip does not preserve local row ID/UUID, version number,
sync-log history, usage counters, timestamps, or collection membership. Import
creates a new local identity and current version. Conflict rename or imported
prefix changes the name only when the user selected that policy.

Modern validation occurs before any row write. It validates artifact type,
format/version/definition agreement, structured-v1 preservation, structured-v2
codec rules, keywords, and local size limits. Invalid modern records fail
closed per item and leave no partial Prompt row. Missing `artifact_type` is
defaulted to Prompt only for legacy records, not malformed modern records.

Prompt record lookup uses the matching manifest `ContentItem.file_path`,
resolved through the importer's contained relative-path validator. It does not
construct a path from the untrusted manifest ID. Duplicate Prompt IDs, duplicate
Prompt file paths, type/path mismatches, missing files, and paths outside the
extraction root fail closed before the affected item is parsed or written.

### Export scope

Extend `ExportScope`, count resolution, and selection resolution with
`prompts`. Prompt IDs come from a fresh uncapped local query, never the rendered
Prompt page. `Everything` adds the Prompt/Recipe count and selections.

The Prompt list's `Export...` action opens the existing Library export canvas
pre-scoped to Prompts. TASK-203 later reuses the same scope with explicit IDs.
Records deleted after scope resolution are skipped and included in the outcome
summary. If every Prompt in a Prompts-only scope disappears or becomes invalid
before collection, no empty archive is finalized; the form reports that no
selected Prompt remained. An Everything export may still complete with its
other selected content and reports the skipped Prompt count. If no Prompt or
other selected content remains, it also finalizes no empty archive and reports
that the resolved scope became empty.

Archive creation retains the existing partial-file plus atomic-replace
finalization. Prompt collection runs in the export worker. Export logs metadata,
counts, and identities but not bodies or definitions.

### Import experience

Chatbook preview and selection show a combined Prompt/Recipe content count and
allow the Prompt content type to be included or skipped. The importer reports
imported, renamed, skipped, and failed Prompt records distinctly. Full error
detail remains bounded and content-free.

Prompt name conflicts use the existing conflict policy selected in the import
workflow. `ASK` never chooses overwrite, rename, or skip silently inside the
worker; an unresolved conflict is reported without a write. Prefixing or rename
changes only the imported record's name.

The existing Library Markdown Prompt importer remains separate; Chatbook
archives continue through the Chatbook import workflow.

### Updated acceptance boundary

- The Chatbook format carries versioned modern Prompt and Recipe records.
- Library Export supports Prompts-only and includes Prompts in Everything.
- Export resolves all matching IDs, including libraries larger than the UI
  page.
- A modern Prompt/Recipe round-trips semantically into a fresh database.
- Legacy flattened archives continue to import through their prior fallback.
- Import preview exposes Prompt content selection and partial outcomes.

## TASK-203 — Prompt Multi-select and Bulk Actions

### Selection model

Reuse `Library.row_selection.RowSelection` with normalized Prompt identities.
Selection keys contain backend and source ID, not row index or cached artifact
type. Current records are refetched immediately before destructive execution so
an artifact whose type changed is classified correctly.

Selection is deliberately bounded to visible rows in the current browse scope.
The toolbar says `Select visible`, and the scope copy distinguishes displayed
rows from total matches. Changing query, collection, page, or sort clears
selection. The Library is local-only in this series, so there is no source
transition to retain or clear. This prevents hidden destructive selection and
avoids claiming cross-page selection support.

In select mode:

- every row displays a selection marker;
- native Enter and Space toggle the focused row;
- row activation does not open the editor;
- Escape clears and exits select mode;
- Clear removes the selection without leaving select mode;
- the action strip shows selected count, Select visible, Clear, Export
  selected, and Delete selected.

### Bulk export and delete

Export selected captures the identities plus browse-scope fingerprint and opens
TASK-197's export form with explicit Prompt IDs. Successful export, cancellation,
and export failure all retain selection so the user can retry or perform another
action.

Delete selected uses the shared Prompt deletion confirmation. It states the
Prompt/Recipe count and shows a bounded name preview. Confirmation captures a
selection fingerprint; any selection or scope change invalidates it.

`PromptScopeService.bulk_delete_prompts` owns routing, and this local-only
Library surface calls it with local identities. Local deletion produces per-ID
success/failure outcomes and uses existing soft-delete behavior. Successful
identities clear; failed identities that remain visible after refresh remain
selected for retry. The active browse result and collection count refresh after
settlement. Server and mixed-source bulk actions are not exposed by this
series.

After refresh, the page is clamped to the new last valid page. If no rows
remain, focus moves to the list toolbar and the honest empty state is shown. A
failed identity remains selected only when it is still visible in the refreshed
scope. A concurrently renamed/reordered failure that is no longer visible is
reported in the bounded failure summary and cleared rather than retained as a
hidden selection.

### Updated acceptance boundary

- Prompt rows support visible, keyboard-operable multi-select.
- Selection is clearly scoped, counted, and clearable.
- Bulk export uses the selected explicit IDs and preserves selection.
- Bulk delete is confirmed and reconciles partial outcomes accurately.
- Prompts and Recipes participate in Library management while Recipe execution
  remains prohibited.

## Compatibility and Security

- Legacy Prompts remain editable and executable under their existing rules.
- Server structured-v1 records remain compatibility-only and are never parsed
  or rewritten as Console v2.
- Supported v2 Prompt/Recipe definitions remain canonical and must agree with
  compiled fields and artifact type.
- Unknown future artifact types or modern record versions fail closed.
- Older servers expose only reported capabilities; no destructive probing is
  used.
- Prompt bodies, definitions, history previews, collection names, variable
  values, and import errors render without Rich markup interpretation.
- Template values are plain text. They do not execute format operations or
  recursively expand.
- Chatbook import retains existing archive path validation and adds per-record
  size validation before parsing/writing large structured definitions.
- Error logs use operation, identity, counts, and exception categories. They do
  not log Prompt bodies, definitions, rendered variable results, or values.
- Delete remains soft-delete. Material removal or history pruning is not added
  by this series.

## Verification Matrix

### TASK-202

- Geometry at 80x24, 100x30, 140x40, and 200x50.
- Normal, narrow, dirty, conflict, clipboard-unavailable, and delete-confirm
  states.
- Stable action IDs, logical focus order, live-working-copy Markdown, and
  user-guide correction.

### TASK-196

- Migration/index tests and `EXPLAIN QUERY PLAN` index-use regression.
- Many unrelated sync-log rows with bounded Prompt-specific paging, page-edge
  predecessors, and pruned version gaps.
- Old snapshots without keywords, modern Prompt/Recipe snapshots, malformed
  definitions, no-change restore, stale expected version, and type-changing
  restore.
- Atomic keyword restore plus duplicate-name/keyword-validation rollback.
- App-wired scope integration and UI dirty/preview/confirm/outcome pilots.

### TASK-198

- More than 100 Prompts with exact totals and pagination.
- Search plus collection plus sort, more than 200 collections, case-fold
  collision labels, serialized duplicate prevention, and local transaction
  rollback.
- Debounce/stale token, error/Retry, empty collection, page focus restoration,
  and local-only source behavior.
- Policy routing tests prove widgets never bypass `PromptScopeService`.

### TASK-199

- Property tests for arbitrary braces, deterministic extraction, escapes,
  adjacent and triple braces, case sensitivity, empty values, one-pass
  rendering, and no input mutation.
- Shared System/User variables, System toggle, use-original, cancel, variable
  limit, no-active-lane disablement, and scrollable dialog.
- Exact slash, picker, ordinary-draft replacement disclosure, and Library
  append integrations.
- 120-second boundary, expired/replaced/wrong-session handoff, stale
  composer/System fingerprints, transient composer retry, persistence warning,
  and Recipe refusal.
- Log-capture tests assert values and rendered bodies never appear.

### TASK-197

- Modern record schema, legacy fallback, Prompt and Recipe round trips,
  foreign-v1 preservation, v2 fidelity, invalid-modern rollback, and conflict
  policies.
- Older-reader compatibility projection.
- Prompts-only, Everything, explicit-ID, beyond-page-cap, deleted-mid-export,
  all-items-disappeared for Prompts-only and Everything, and atomic archive-
  finalization tests.
- Validated manifest paths, duplicate ID/path rejection, unresolved ASK
  conflicts, import preview selection, and bounded partial outcome UI.

### TASK-203

- RowSelection scope-clear rules, Select visible, Enter/Space/Escape behavior,
  and targeted UI reconciliation.
- Stale confirmation, changed artifact type, export selection retention, bulk
  delete partial success, last-page clamping, off-view failure reconciliation,
  and collection refresh.
- Integration with TASK-197 explicit-ID export and narrow-terminal captures.

Each PR also runs the affected Prompt Management, Library, Chatbook, Console,
database, API-schema, and UI suites; repository lint/format checks; `git diff
--check`; documentation tests where present; and the full suite before merge.
Exact commands belong in each implementation plan because the available test
targets may change after prior PRs merge.

## Merge-gated Delivery Contract

For each task:

1. Fetch and confirm the latest merged `origin/dev`.
2. Create a fresh ignored `.worktrees/` worktree and `codex/` branch.
3. Mark that Backlog task In Progress and add its implementation plan,
   including the required ADR path/reason. TASK-202 additionally puts its two
   absorbed defect records, TASK-2700 and TASK-2701, In Progress and gives each
   the linked one-PR plan described above.
4. If an ADR is required, write and commit it before implementation code.
5. Add or update acceptance criteria before implementing any newly approved
   behavior.
6. Follow red-green-refactor: observe the focused test fail before production
   code, then make the smallest implementation pass.
7. Run focused, affected integration, UI, static, documentation, and full-suite
   verification proportional to the change.
8. Render and inspect the required terminal sizes and failure states.
9. Perform self-review and request independent code review.
10. Complete Backlog acceptance checkboxes, implementation notes, ADR links,
    docs, and Done status only when the Definition of Done is satisfied.
11. Open one ready PR against `dev`, address every review thread and CI failure,
    and merge it.
12. Confirm the merge commit on `dev`; only then begin the next task.

The sequence is:

1. TASK-202 — editor action grouping and action-row defects.
2. TASK-196 — retained version history.
3. TASK-198 — complete browse and collections.
4. TASK-199 — shared variable grammar/dialog/application.
5. TASK-197 — Chatbook Prompt record and bulk Prompt scope.
6. TASK-203 — multi-select and bulk actions.

TASK-203 depends on TASK-197's explicit-ID export. TASK-198 supplies the
complete paginated browse and stable scope model that TASK-203 selects from.
The remaining steps are intentionally merge-gated even where code could be
developed in parallel, because the user selected review/CI/merge feedback as
the authority for the next PR's baseline.

## Documentation and Backlog Hygiene

Each task PR updates its own Backlog file only after marking it In Progress and
before implementation. Approved acceptance-criteria expansions in this spec
must be copied into the task before the corresponding code is written.

This approved umbrella specification is carried into the first TASK-202 branch
and committed in the TASK-202 PR. It does not receive a separate design-only
PR, so the user-visible delivery remains exactly six PRs.

Expected documentation updates include:

- `Docs/User_Guide/library/prompts.md` for actions, history, collections,
  variables, Prompt export, and multi-select as those features land.
- Chatbook format/import documentation in TASK-197.
- Prompt variable grammar examples and literal-brace guidance in TASK-199.
- ADR links in the task implementation plan and notes.
- Visual QA artifacts when the repository's current PR practice requires them.

TASK-2700 and TASK-2701 are updated and closed by the TASK-202 PR after their
criteria pass. No unrelated Backlog task is silently marked complete.

## Open Questions

None. Product behavior, scope, sequencing, compatibility posture, and merge
gates are approved. Implementation details that do not alter these outcomes may
be refined in the per-task plans after each preceding PR merges.
