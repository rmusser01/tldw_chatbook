# ADR-184: Local Console Prompt Draft Shelf

Status: Accepted
Date: 2026-09-25
Related Task: TASK-18930
Supersedes: N/A

## Decision

Store saved, unsent Console message drafts in a device-local
`LocalPromptDrafts` table inside the existing local Prompts database. Drafts
are not Prompt or Recipe records, do not enter the Prompt sync log, and are
never routed to the server. The existing Prompt Workbench remains the sole
Console surface for browsing Library Prompts, Recipes, and the Draft Shelf.
The table is introduced by the Prompts database v4-to-v5 migration; runtime
services validate and use that schema but do not create it lazily.

Each draft owns a stable integer identity, exact UTF-8 content, creation and
update timestamps, and an optimistic version. Display titles are derived from
the first non-empty content line rather than stored as a second editable
authority. Create, edit, and delete operations run through the local
`PromptScopeService` boundary. A server mode is rejected before storage is
consulted.

The shelf holds at most 100 entries. Creation reserves SQLite's writer slot,
counts entries, and fails with a typed capacity outcome when the limit is
reached. It never evicts an older entry. The recovery path is explicit: the
user deletes one or more shelf entries and retries. Draft deletion is a hard
delete with two-press confirmation in the UI; these local working copies do
not create sync tombstones.

Saving from the composer persists the canonical draft text returned by the
Console composer, including the full content represented by collapsed paste
tokens. The user chooses **Save and keep** or **Save and clear**. Clearing is
allowed only after the database transaction succeeds. Promoting a shelf entry
creates a normal local Library Prompt through the existing Prompt save API and
may then assign one existing local collection through the existing membership
API. Promotion never removes or rewrites the shelf entry.

The keyboard route uses the Console command palette. ADR-031's ban on
terminal-convention bindings means this feature does not claim Ctrl+S.

## Context

Prompt history records accepted sends and its live-draft pseudo-entry only
supports recall. It cannot provide named identities, in-place edits, explicit
deletion, paging, or promotion without changing its lifecycle and recovery
meaning. Library Prompt records, meanwhile, are durable reusable artifacts
that may sync to a server; silently treating every partial thought as a Prompt
would blur that boundary and pollute search and collections.

The local Prompts database already owns Prompt records and local-only
collection metadata. Keeping the Draft Shelf in that database reuses its
private SQLite lifetime, transactions, backup ownership, and configured
profile location without introducing another raw-file participant. A separate
table keeps temporary working copies out of Prompt sync, export, FTS, usage,
and version-history paths.

## Required Boundaries

- Draft rows are device-local working copies, not Prompt or Recipe records.
- Draft operations never write `sync_log`, Prompt FTS tables, collection
  memberships, Chatbooks, chat transcripts, or server APIs.
- The 100-entry limit is checked and inserted within one immediate
  transaction. A full shelf is a refusal, never an eviction.
- Exact draft content is stored; list previews and titles are projections.
- Updates and deletes require the version the user reviewed and reject stale
  writes.
- Save-and-clear clears only after confirmed persistence.
- Inserting a draft uses paste semantics at the current composer caret and
  never replaces unrelated draft text.
- Promotion uses the existing local Prompt and collection contracts and leaves
  the shelf source intact even when all promotion steps succeed.
- Prompt Workbench owns focus restoration, stale-worker rejection, loading,
  empty, error, and capacity recovery states.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Reuse sent prompt history | Its append/recall lifecycle has no stable editable identity or explicit management contract. |
| Save every stash as a Library Prompt | Pollutes reusable artifacts and can expose partial drafts to sync/export behavior. |
| Add a standalone JSON/JSONL file | Creates a second raw-file recovery participant and duplicates transactional behavior already available in the Prompts database. |
| Evict the oldest draft at capacity | Deletes user work without an explicit choice and makes the cap surprising. |
| Build a second stash browser modal | Duplicates Prompt Workbench navigation, focus, Library browsing, and truthful-state behavior. |
| Bind Ctrl+S | Conflicts with ADR-031's terminal-convention key rules. |

## Consequences

### Benefits

- Partial drafts remain private to the selected local profile and are managed
  explicitly.
- The limit cannot silently destroy older work.
- Promotion reuses existing Prompt validation, collection membership, and
  Library discoverability.
- One workbench provides a continuous Draft Shelf to Library workflow.

### Accepted trade-offs

- A full shelf blocks new saves until the user deletes an entry.
- Drafts do not sync between devices.
- Promotion and optional collection assignment are two honest operations; a
  collection failure may leave a successfully created Prompt unfiled.
- Reinserted content is plain composer text; collapsed paste presentation
  metadata is not reconstructed.

## Links

- [ADR-029: Versioned Prompt Artifacts and Safe Improvement Transactions](029-versioned-prompt-artifacts-and-safe-improvement-transactions.md)
- [ADR-031: TUI Keybinding and Footer Hint Conventions](031-tui-keybinding-and-footer-hint-conventions.md)
- [ADR-057: Portable Chatbook Prompt Records](057-portable-chatbook-prompt-records.md)
- [Console Prompt Draft Shelf Design](../../Docs/superpowers/specs/2026-09-25-console-prompt-draft-shelf-design.md)
