# TASK-203: Library Prompt multi-select and bulk actions design

Status: Approved in conversation on 2026-08-12; revised after independent specification review
Task: [TASK-203](../../../backlog/tasks/task-203%20-%20Library-Prompts-multi-select-bulk-actions-in-the-list.md)

## Purpose

The local Library Prompt list already provides exact search, collection scope,
stable 50-row paging, single-item version-checked delete with Undo, and exact
Chatbook Prompt export. It does not let a user curate several Prompts or Recipes
and act on that set. TASK-203 adds one ephemeral selection session spanning
searches and pages, exact selected export, and atomic selected delete/Undo.

The design deliberately does not add bulk tagging. Collections are the current
Prompt organization surface, and keyword mutation would require a separate
contract not present in this task's acceptance criteria.

## Goals

- Select local Prompts and Recipes across searches, pages, sort orders, and
  collection scopes.
- Keep the complete selection visible, understandable, and explicitly
  clearable even when only part of it is on the current page.
- Export exactly the selected active IDs through the existing Chatbook export
  canvas and ADR-057 record format.
- Soft-delete the selected set all-or-nothing against captured versions.
- Restore a completed delete all-or-nothing through one in-place receipt.
- Make single delete a one-item use of the same mutation family.
- Preserve local-only selection/export, exactly-once batch mutation policy,
  privacy, focus, and narrow-TUI boundaries.

## Non-goals

- Bulk keyword/tag editing.
- Selecting server Prompt rows or adding server batch capabilities.
- A generic bulk-action framework for every Library source.
- Persisting a selection across application restarts.
- Adding a Prompt Trash view or changing retained-history semantics.
- Adding another exporter, archive format, delete modal, or scroll owner.

## Existing decisions and reusable seams

- [ADR-055](../../../backlog/decisions/055-library-destructive-action-reversibility-rule.md)
  requires one delete family, an at-point receipt, Undo, and one shared
  mutation interlock for soft-deleted persisted Library data.
- [ADR-057](../../../backlog/decisions/057-portable-chatbook-prompt-records.md)
  makes selected Prompt export local-only and all-or-nothing and already
  supports `ExportScope(kind="prompts", ids=...)` through the existing export
  canvas.
- [ADR-060](../../../backlog/decisions/060-atomic-local-prompt-batch-mutations.md)
  defines the all-or-nothing Prompt delete/restore conflict policy, typed
  pre-commit result boundary, legacy single-API compatibility, and receipt-owner
  navigation admission.
- `PromptsDatabase.soft_delete_prompt` and `restore_deleted_prompt` already own
  versioned tombstones, keyword recovery, FTS maintenance, and sync events.
- `PromptDeleteConfirmationModal` already supports a plural immutable request,
  bounded name preview, Prompt/Recipe counts, literal copy, and a stale-result
  fingerprint.
- `LibraryPromptBrowseController` already owns exact browse loading, stale
  result rejection, search, page, sort, and collection scope.

## ADR check

ADR required: yes
ADR path: `backlog/decisions/060-atomic-local-prompt-batch-mutations.md`
Reason: ADR-055 defines the reversibility pattern and ADR-057 defines selected
export, but neither previously decided the all-or-nothing multi-row Prompt
conflict policy or the new database-to-service typed batch result contract.
ADR-060 records those cross-module boundaries while leaving storage schema,
archive format, and server single-item behavior unchanged.

## Selected-entry model

The Prompt list owns one Textual-free `PromptSelectionBasket`. A selected entry
is immutable and contains:

- positive local Prompt ID;
- positive captured row version;
- captured display name;
- captured artifact type (`prompt` or `recipe`).

The basket is keyed by local ID, preserves the first captured version until the
entry is explicitly deselected, and carries a monotonic generation incremented
on every semantic selection change. Re-visiting a page, settling a new browse
result, or pressing Select page never refreshes an already-selected entry's
version. Toggling it off and on is a new selection and captures the newly
visible version.

Canonical bulk/export order is ascending numeric local ID. This makes archive,
modal, service, and receipt behavior deterministic without treating current
search order as durable state.

The page projection adds presentation-only fields to Prompt rows/state:

- whether each visible row is checked;
- whether select mode is active;
- total selected count;
- selected count on the settled current page.

No selected Prompt content, lane text, details, keyword, definition, collection
membership, or history is retained in the basket.

## Selection lifecycle

Selection persists through:

- debounced search and Enter-flushed search;
- Previous/Next paging;
- sort changes;
- collection changes;
- browse loading, empty, and error states;
- opening, cancelling, failing, or completing the selected Export canvas and
  returning to its Prompt-list origin.

Selection clears through one shared lifecycle helper when the user:

- presses Done;
- presses Clear all;
- successfully deletes the selected batch;
- enters a Prompt editor or Create Prompt;
- switches to another Library content source;
- leaves/unmounts Library.

Done and deliberate navigation announce `Selection discarded · N prompts`
when the basket was nonempty. Clear all is already the user's explicit action
and needs no redundant notification. Application teardown clears silently.
Selection is intentionally absent from `LibraryScreen.save_state()`.

## List interaction

Normal mode uses button-only rows that remain valid at the canvas's 40-column
minimum:

- row 1: Sort and Select;
- row 2: Import and Export.

Select mode keeps Filter, Collection, paging, loading/error Retry, and empty
states available. It renders:

- literal summary: `7 selected · 2 on this page`;
- management row: Select page, Clear all, Done;
- action row: Export selected, Delete selected.

Rows remain one focusable Button each and use the established literal `☑` and
`☐` prefixes. Pressing a row toggles selection rather than opening the editor.
Select page adds every valid row from the currently settled page and never
overwrites an existing entry. During loading/error, Select page is disabled
with a reason while whole-basket Clear all, Export selected, and Delete
selected remain usable when the basket is nonempty.

Disabled bulk actions use the existing non-colour marker and explanatory
tooltip convention only as supplementary help. One literal status line remains
visible and keyboard-readable whenever an action is disabled. Zero selection
takes precedence over page state and reads `Select one or more items to use
bulk actions.` When selection is nonempty and the current page is loading or
failed, it reads `Current page is unavailable; selected items remain available
for Export or Delete.` This explains disabled Select page and zero-selection
bulk actions without depending on focus or colour. During a mutation, every
row toggle, selection action, receipt action, Prompt create/update/delete
action, and Prompt route transition is inert under the shared Prompt mutation
interlock. The list renders fixed literal progress such as `Deleting 7 selected
items…`; no mid-transaction Cancel action is offered.

Selection actions restore the exact row/control focus after any required
recompose. Browse settlement preserves the filter cursor when search owns
focus. When deleted rows remove the former focus target, focus falls back to
the nearest surviving row and then the Select control.

The feature adds no `VerticalScroll`. Real-bundle compositor evidence must
prove both action rows, count, confirmation, progress, and receipt are visible
and keyboard reachable at 64x24 and 120x40.

## Selected export

Export selected snapshots the basket in ascending numeric ID order, then
converts it to the existing string contract as
`tuple(str(entry.local_id) for entry in entries)`. That tuple becomes
`ExportScope(kind="prompts", ids=...)` and opens the existing Chatbook export
canvas. The export canvas owns counts, destination, overwrite, cancellation,
progress, and retry exactly as it does today.

Export resolves the latest active content for each selected identity at export
time. A content edit after selection therefore does not invalidate export.
ADR-057 remains authoritative: if an ID is missing/deleted or any selected row
cannot be read/encoded, the Prompt-bearing archive is not finalized. Returning
from the export canvas leaves the live selection basket unchanged regardless of
success, failure, or cancellation.

## Atomic delete contract

Bulk delete accepts a nonempty immutable tuple of unique `(local_id,
expected_version)` targets. Types are exact: bool is not an integer, IDs fit
SQLite's positive signed range, and versions are positive.

The database batch method:

1. starts `BEGIN IMMEDIATE` before reading any target;
2. resolves targets in canonical ID order;
3. validates that every row is active and at its captured version;
4. captures and validates all exact keyword/tombstone recovery metadata;
5. performs every Prompt update, keyword unlink, FTS removal, and sync event
   through transaction-local primitives;
6. constructs and validates the complete immutable receipt DTO, including every
   field the UI requires, while the transaction is still open;
7. commits once;
8. only after commit reports aggregate success and returns that already-valid
   typed receipt without further normalization.

Any invalid, missing, stale, uniqueness, SQLite, or helper failure rolls back
the entire transaction. The UI never loops the public single-delete service.
The Library's existing single delete becomes a one-target call into this same
strict batch mutation family. Transaction-local helpers prevent nested public
methods from emitting per-item success diagnostics or metrics before the outer
batch commits.

The pre-existing public single database/service APIs remain compatible: ID,
UUID, and name lookup, optional delete version, missing-delete `False`, restored
row return shape, and server single-item routing do not change. Those wrappers
resolve legacy identifiers inside their own `BEGIN IMMEDIATE` transaction and
call the same transaction-local helpers rather than calling the strict batch
API recursively.

The local service exposes the strict batch directly. The scope service accepts
it only for local mode, validates before policy/backend access, enforces the
existing local delete policy once, and offloads through the established Library
service runner. After the database returns, local and scope services are typed
pass-through: they perform no fallible mapping, normalization, coercion, or DTO
construction. No server batch fallback exists.

## Confirmation and stale settlement

Delete selected opens the existing `PromptDeleteConfirmationModal` from an
immutable snapshot of the basket. The title/body name exact Prompt and Recipe
counts, preview at most three bounded literal names, append `and N more` when
needed, and promise the available Prompts-list Undo—not a nonexistent Trash
surface.

The modal request carries an opaque monotonic generation, not concatenated
IDs/names. A duplicate or late result, a changed basket, a changed route, or an
already-admitted mutation is a no-op. External row changes remain possible;
the database version check is authoritative.

On stale or missing targets, nothing is deleted, the complete basket remains,
and the list shows fixed copy:

`Selection changed; nothing was deleted. Clear all and select the items again.`

No invalid item is silently dropped and no version is silently refreshed.

## Receipt and atomic Undo

One plural receipt model represents both one-item and many-item deletes. A
single receipt preserves the current named `✓ deleted · Prompt/Recipe · Name`
shape; a batch renders `✓ deleted · N items`. Both provide Undo and Dismiss.

An existing receipt remains available while a new confirmation or deletion is
attempted and is superseded only after the newer delete commits successfully.
A failed attempt therefore cannot erase an older recovery opportunity.

Undo passes every `(local_id, tombstone_version)` receipt entry to one local
batch restore transaction. It validates all rows, versions, recovery payloads,
and conflicts before mutation, then restores Prompt state, keywords, FTS, and
sync events together. The complete typed restore result is constructed and
validated before commit and is passed through unchanged afterward. Any failure
restores nothing and leaves the complete receipt available. Full success clears
the receipt and refreshes the exact Prompt browse plus rail count.

## Concurrency and route ownership

Single delete, bulk delete, Undo, Prompt create/update, and receipt mutations
share the existing Prompt mutation flag and worker group. No independent bulk
flag or worker group may be added. Every Library rail/source/editor/create and
Export transition is refused while delete/Undo is settling. Ordinary app-level
outgoing navigation is also vetoed: `LibraryScreen.flush_pending_work()` returns
false while the Prompt mutation is admitted, so app navigation cannot destroy
the screen before its off-thread database call settles and publishes the
receipt.

The database transaction is the atomicity authority; UI generation guards are
only stale-presentation protection. Search/browse workers may settle while a
confirmation is open, but they cannot alter the basket or its captured
versions.

## Error and privacy contract

User-facing failures are fixed and content-free:

| Condition | Outcome |
| --- | --- |
| stale/missing target | `Selection changed; nothing was deleted. Clear all and select the items again.` |
| batch service unavailable | `Bulk Prompt actions are unavailable.` |
| generic delete failure | `Could not delete the selected items. Nothing was deleted.` |
| generic Undo failure | `Could not restore the deleted items; Undo is still available.` |

New or modified persistent diagnostics may contain only a fixed operation,
aggregate item count, and exception category. They must never contain Prompt
names, details, lanes, definitions, keywords, IDs, versions, selection or
receipt representations, exception messages, or tracebacks. Success metrics
are emitted only after the outer transaction commits. Any changed logger owner
must be reconciled narrowly with the persistent diagnostic inventory.

## Implementation ownership

- A small pure Prompt-selection model owns entries, generation, canonical
  ordering, and page counts.
- Prompt list state projects that model for rendering.
- `LibraryPromptsListCanvas` owns presentation only.
- `LibraryScreen` retains thin handlers and reuses its existing modal, export,
  focus, navigation, and shared Prompt mutation seams.
- Prompt database/local/scope services own batch validation and atomic
  persistence.
- Existing Chatbook export code remains unchanged except for any tests needed
  to exercise the already-supported selected Prompt scope. Selected export
  retains ADR-057's sanitized Prompt collection/scope diagnostic boundary;
  this task does not broaden its privacy promise to unchanged generic exporter
  diagnostics.

No generic Library bulk controller, new dependency, schema migration, or new
export/delete presentation surface is introduced.

## Verification strategy

### Pure selection and state

- strict selected-entry and basket validation;
- cross-search/page/sort/collection persistence;
- deterministic canonical order and duplicate suppression;
- existing captured version survives page revisit and Select page;
- total and on-page counts, complete clear, and lifecycle clear;
- checked row/list state for loading, error, empty, and settled pages.

### Real file-backed SQLite and service boundaries

- multi-Prompt and mixed Prompt/Recipe delete/restore fidelity;
- exact keyword, FTS, sync event, version, and collection-membership behavior;
- stale, missing, duplicate, invalid, name-conflict, and forced mid-batch
  rollback with zero partial mutation;
- `BEGIN IMMEDIATE`, one commit, and validation-before-first-write evidence;
- receipt/restore DTO construction and validation before commit, with typed
  no-fail service pass-through afterward;
- single-item paths call the same transaction primitives while preserving
  integer/name/UUID, optional-version, missing-delete, restored-row, and server
  compatibility;
- local-only routing, strict validation before policy/DB access, and exactly one
  policy decision per batch;
- adversarial diagnostics proving content, identity, exception text, and
  traceback absence.

### Screen and mounted product path

- select rows across two literal searches and multiple pages/collections;
- Select page does not refresh an existing captured version;
- exact selected Export scope and preserved basket after Back;
- stale modal/generation no-op and stale database selection recovery;
- prior receipt preserved on failed new delete;
- shared single/bulk/Undo interlock and route refusal;
- another Library source and another app screen both remain blocked while a
  mutation is admitted, with the receipt owner preserved;
- full delete, rail/list refresh, batch receipt, full Undo, and focus fallback
  against real file-backed SQLite;
- fixed literal copy and no hidden Rich markup interpretation;
- real CSS bundle compositor captures at 64x24 and 120x40 proving visible,
  unclipped, reachable actions with the existing single scroll owner.

### Mutation and closeout gates

Mutation checks must prove discrimination for:

- removing validation-before-write;
- splitting the transaction/commit;
- moving receipt construction or service normalization after commit;
- refreshing a selected entry's captured version on page settlement;
- clearing selection on search/page/sort/collection or Export admission;
- bypassing selection-generation validation;
- clearing the prior receipt before a new delete commits;
- restoring only a subset of a failed Undo.

Run the affected Prompt database, service, state, controller, Prompt canvas,
Library shell/export, RuntimePolicy, and diagnostic-inventory tests; Ruff,
formatter, typing/compile, CSS source/bundle parity when applicable, and diff
checks; then perform one final Impeccable detector and real compositor review.

## Alternatives rejected

### Loop the existing single-delete API in the screen

Rejected because it cannot provide all-or-nothing behavior and could emit
per-item success diagnostics before a later item fails.

### Preserve only currently rendered selection

Rejected because the approved workflow is cross-search curation. The UI makes
hidden selection explicit through total versus on-page counts and a bounded
confirmation preview.

### Generic Library bulk-action framework

Rejected as speculative. Media, Notes, and Conversations already have mature,
different selection lifecycles; TASK-203 needs only Prompt-specific captured
versions and atomic delete/Undo.

### Partial delete with failed rows left selected

Rejected by the approved all-or-nothing contract. A successful-looking partial
destructive action would make the cross-search basket difficult to reason about
and complicate a truthful Undo receipt.

### Bulk tagging in the same task

Rejected because it lacks an approved batch keyword mutation contract and is
not required by the clarified acceptance criteria. Collections remain the
current organization feature.
