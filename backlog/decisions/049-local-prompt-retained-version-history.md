# ADR-049: Local Prompt Retained Version History

Status: Accepted
Date: 2026-08-08
Related Task: [backlog/tasks/task-196 - Prompt-version-history-UI-in-the-Library-editor.md](../tasks/task-196%20-%20Prompt-version-history-UI-in-the-Library-editor.md)
Supersedes: N/A

## Decision

Expose local Prompt and Recipe history as a bounded, index-backed projection of
retained `sync_log` create and update snapshots. Call the feature **retained
history**, not complete version history: sync-log cleanup may remove older
snapshots, and the UI must never imply that the oldest retained row is the first
version.

The local database query is constrained by Prompt entity UUID, Prompt entity
type, the fixed operations `create` and `update`, and descending `change_id`.
Its composite index must make query cost independent of unrelated sync-log
volume. A request validates a finite positive page size and reads at most
`page_size + 1` matching snapshots. The extra older snapshot is used only as
the immediate predecessor of the last visible row; it is not another visible
row in that page. Paging uses a strict `before_change_id` cursor.

A visible row is compared only with its immediately preceding retained Prompt
version. Version 1 is labelled `Created`. If pruning or malformed retained data
means that immediate predecessor is unavailable, the row is labelled `Earlier
baseline unavailable`; non-adjacent versions are never compared as though the
gap did not exist. Delete, link, and unlink events are excluded. Collection
membership, usage counters, and deletion state are not part of this history.

Future Prompt create and update snapshots add the effective ordered `keywords`
after keyword membership has settled. This is an additive payload field: older
consumers may ignore it, and older snapshots without it remain readable with
`keywords_captured = false`. Prompt fields, keyword links, and the corresponding
snapshot are written in one transaction. A keyword validation or persistence
failure rolls back all three.

`PromptScopeService` remains the application-facing boundary. It routes local
history to the live local adapter, preserves existing server routing for other
callers, and normalizes both through one retained-version envelope. Malformed
payloads remain explicit preview/error records instead of being silently
dropped or failing the whole page.

Restore accepts both the selected snapshot version and the expected current
version. Inside one conditional transaction, the local service re-reads the
current record, rejects a stale expected version, and validates the snapshot
under current source capabilities and ADR-040. Only valid legacy text and valid
structured-v2 Prompt/Recipe snapshots are restorable. Malformed JSON,
definition/compiled-text mismatch, artifact-type/definition-kind mismatch,
unknown format or schema version, unsupported future artifact types, and
foreign structured-v1 snapshots are preview-only with their exact compatibility
reason. Foreign v1 content is not reparsed, converted, or written through
restore.

An eligible restore uses the ordinary conditional Prompt update path and
therefore appends a new current `update` snapshot; it never mutates retained
history in place. A modern snapshot replaces keywords in the same transaction.
An older snapshot with no captured keywords retains the current keywords and
the outcome discloses that choice. If all restorable artifact fields, metadata,
and effective keywords are byte-identical to current state, restore returns
`no_change` and creates no sync row. Duplicate-name conflicts, keyword failures,
or any other write failure roll back the Prompt row, keyword links, and sync-log
append together.

## Context

The standalone local Prompt adapter can currently reconstruct snapshots by
scanning the entire sync log, while the application-wired scope rejects local
history. That behavior is both unbounded and outside the Library's canonical
service route. The existing sync log is still the right retained source because
ordinary Prompt changes already produce versioned snapshots, but exposing it to
users requires a bounded query, honest pruning semantics, normalized
compatibility states, atomic keyword capture, and optimistic concurrency on
restore.

ADR-040 already defines Prompt/Recipe identity, schema-v2 validation, compiled
compatibility text, foreign-v1 handling, conditional update, and byte-identical
`no_change` behavior. This decision extends those rules to retained snapshots;
it does not introduce another artifact schema or a separate history store.

## Required Boundaries

- Widgets and Library state do not query `sync_log` or call a database adapter
  directly; they use `PromptScopeService`.
- History is lazy and paginated. Opening the collapsed disclosure triggers the
  first bounded request, and older pages use the returned cursor.
- Snapshot text is rendered literally in read-only previews. Viewing remains
  available while the editor is dirty, but restore is disabled until the
  working copy is saved or discarded.
- Restore confirmation states that a new current version is created and calls
  out a Prompt-to-Recipe or Recipe-to-Prompt type change.
- The expected current version is checked at the transaction boundary, not only
  in UI state. A mismatch uses the existing conflict/Reload outcome.
- Migration adds the composite history index without rewriting retained rows.
  Additive keyword payloads do not invalidate older rows or older consumers.
- Malformed and compatibility-invalid rows remain observable and preview-only;
  they are never silently normalized into a restorable artifact.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Scan `get_sync_log_entries(since_change_id=0)` and filter in Python | Work grows with every unrelated sync event, bypasses an appropriate index, and decodes an unbounded number of payloads. |
| Present the retained rows as complete version history | Cleanup may prune early rows, so this would make an unverifiable completeness promise and produce false changed-field summaries across gaps. |
| Copy snapshots into a new history table | Duplicates retained payload ownership and introduces a second lifecycle without a product need. |
| Include delete, link, unlink, collections, usage, or deletion state | These events do not represent restorable Prompt/Recipe artifact versions and would blur ownership of independent state. |
| Restore by overwriting the current row or changing history in place | Destroys lineage and bypasses the ordinary version, validation, and sync contract. |
| Restore without `expected_version` | A stale preview could overwrite a concurrent edit. |
| Treat missing historical keywords as an empty list | Older snapshots did not capture keyword state; clearing current keywords would invent history that was never recorded. |
| Convert foreign structured-v1 snapshots during restore | Violates ADR-040 and can lose roles, variables, assembly rules, or future v1 meaning. |

## Consequences

### Benefits

- Local retained history has bounded, index-backed cost and an honest pruning
  model.
- Restore preserves Prompt/Recipe fidelity and concurrent edits while producing
  a new, auditable current version.
- New snapshots restore keyword state atomically, while old snapshots remain
  usable without destructive guesses.
- Compatibility-invalid history is still inspectable without weakening the
  current artifact validator.

### Accepted trade-offs

- Retained history is not an audit log or backup and may begin after version 1.
- A schema migration and an additional sync-log index are required even though
  existing retained rows are not rewritten.
- Historical snapshots without keywords cannot restore their original keyword
  membership because it was never captured.
- One extra matching row is read per page to support a truthful predecessor
  summary.
- Restore may become unavailable when current capabilities no longer accept an
  otherwise readable historical artifact.

## Links

- [TASK-196 implementation plan](../../Docs/superpowers/plans/2026-08-02-task-196-retained-prompt-history.md)
- [Library Prompt Enhancement Series design](../../Docs/superpowers/specs/2026-08-02-library-prompt-enhancement-series-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-040: Versioned Prompt Artifacts and Safe Improvement Transactions](040-versioned-prompt-artifacts-and-safe-improvement-transactions.md)
