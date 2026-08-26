# ADR-060: Atomic local Prompt batch mutations

Status: Accepted
Date: 2026-08-12
Related Task: [TASK-203](../tasks/task-203%20-%20Library-Prompts-multi-select-bulk-actions-in-the-list.md)
Extends: [ADR-055: One reversibility rule for Library destructive actions](055-library-destructive-action-reversibility-rule.md)

Allocation note: ADR-060 was allocated after sweeping every visible local and
remote ref, repository worktree, and open pull request on 2026-08-12. ADR-058
and ADR-059 were already allocated on the visible `kimi-zai-hosted-chat`
worktree/ref; no visible source reserved ADR-060.

## Decision

Local Prompt/Recipe multi-row delete and restore are atomic, version-checked
batch operations owned by `PromptsDatabase` and exposed through the local and
scope Prompt services. Every target succeeds in one SQLite transaction or no
target changes. Library single-item deletion uses the batch operation with one
target, so single and bulk variants share the ADR-055 receipt, Undo, and
interlock family.

Each strict batch target carries a positive local Prompt ID and the positive
version captured when the user selected or opened it. Batch delete starts
`BEGIN IMMEDIATE`, validates the entire target set before the first write, then
performs Prompt tombstoning, keyword unlink, FTS removal, sync events, and
version advancement through transaction-local helpers. Batch restore likewise
validates every exact tombstone version and recovery payload before restoring
any row, then restores Prompt state, keyword membership, FTS, and sync events
together.

The database constructs and validates the complete immutable delete receipt or
restore result inside the transaction, before the transaction context can
commit. After commit, local and scope service layers return that typed value
unchanged. No result mapping, normalization, coercion, or DTO construction may
fail after commit and turn a successful durable mutation into a reported
failure.

The pre-existing public single-item database and service APIs retain their
accepted identifiers and return behavior. `soft_delete_prompt` continues to
accept an integer ID, UUID, or name, keeps optional `expected_version`, and
returns `False` for a missing/already-deleted row. `restore_deleted_prompt`
continues to accept those identifier forms and returns its established restored
row shape. These legacy wrappers resolve their identifier within their own
`BEGIN IMMEDIATE` transaction and use the same transaction-local mutation
helpers as the strict batch APIs. Server single-item routing remains unchanged;
there is no server batch fallback.

## Required boundaries

- Batch inputs are nonempty immutable tuples with unique, canonical positive
  SQLite-range local IDs and positive versions. Python bools are rejected as
  integers. Validation completes before policy or database access at each
  public boundary that receives untrusted input.
- Local batch delete/restore reserves the SQLite writer slot before resolving
  targets. Target validation and mutation happen on one connection and one
  commit. Missing, stale, duplicate, conflicting, malformed, or failed targets
  roll back the complete batch.
- Public single and batch entry points share transaction-local helpers. Public
  methods are not nested to fake atomicity, and transaction-local helpers emit
  no success diagnostics or metrics.
- The database constructs all fields the caller needs in a strictly validated,
  immutable typed result before leaving the transaction context. Service
  layers are typed pass-through after the durable call returns.
- Success diagnostics/metrics occur only after commit and may contain only a
  fixed operation and aggregate count. Failures may additionally contain an
  exception category. Prompt names, details, lanes, definitions, keywords,
  IDs, UUIDs, versions, target/receipt representations, exception messages,
  and tracebacks are forbidden.
- The scope service exposes batch mutation only in local mode, performs exactly
  one RuntimePolicy decision for the batch, and does not loop single-item
  service calls.
- One immutable receipt represents one committed transaction. Its Undo restores
  the complete transaction atomically. A failed newer delete does not replace
  an older receipt; a failed Undo keeps its receipt.
- The screen that owns an admitted delete/Undo also owns settlement and receipt
  publication. Every in-Library route/source/editor/export transition and
  ordinary app-level outgoing navigation is refused while the mutation is in
  flight. `LibraryScreen.flush_pending_work()` participates in that app-level
  veto so a `to_thread` database write cannot commit after navigation destroys
  its receipt owner.
- Selection/presentation generation checks reject stale modal and UI outcomes,
  but SQLite target/version validation remains the mutation authority.

## Context

ADR-055 established the Library-wide reversibility rule: a soft delete leaves
an in-place receipt with Undo, and single/bulk variants share one mutation
family. It did not choose a conflict policy or cross-layer result contract for
multi-row Prompt mutation. TASK-203 adds a selection basket that can contain
rows hidden behind other searches and pages. Partial deletion would be hard to
explain and recover, while looping the existing public single API cannot
provide one commit.

`PromptsDatabase` already stores versioned Prompt tombstones, exact keyword
recovery metadata, FTS state, and sync events. Its transaction helper supports
`BEGIN IMMEDIATE` and nested transaction awareness. That is sufficient for an
atomic batch without a schema change, but only if mutation work is factored
below the public single methods and all fallible result construction completes
before the outer transaction commits.

The Library screen owns the at-point receipt. App navigation currently replaces
screen instances after a pending-work flush, so navigation admission is part of
the durable correctness boundary: permitting navigation while an off-thread
batch is admitted could leave committed deletion with no surviving receipt.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Loop `delete_prompt`/`restore_deleted_prompt` in the UI or scope service | Each call can commit independently, producing partial mutation and per-item success before the batch outcome is known. |
| Wrap the existing public methods in one outer transaction | Nested transaction awareness prevents inner commits, but the public methods still emit success diagnostics/metrics and return legacy shapes before the outer commit; this misreports rolled-back work and leaves post-commit normalization hazards. |
| Permit partial success and keep failed targets selected | A cross-search selection would become difficult to reason about, and one truthful atomic Undo receipt could not describe the outcome. |
| Change the existing single-item APIs to strict ID/version tuples | This breaks accepted name/UUID/optional-version callers and unnecessary server routing. Shared helpers preserve compatibility without widening the batch contract. |
| Introduce a generic Library batch transaction framework | Other Library sources use different stores and recovery contracts. A cross-store abstraction would add flexibility TASK-203 does not need. |

## Consequences

### Benefits

- Users never receive a successful-looking partial Prompt deletion or restore.
- Single and bulk Library delete share the same durable behavior and recovery
  family.
- A committed transaction always has a valid result ready for its receipt
  owner; response parsing cannot manufacture a post-commit failure.
- Existing single-item and server callers remain compatible.
- Diagnostics stay aggregate and privacy-safe.

### Accepted trade-offs

- A stale version on one hidden selected row blocks the entire batch and asks
  the user to reselect.
- Batch validation and mutation hold the SQLite writer reservation for the
  complete selected set.
- Ordinary navigation is briefly vetoed while delete or Undo settles so the
  at-point receipt cannot lose its owner.

## Links

- [TASK-203 design](../../Docs/superpowers/specs/2026-08-12-task-203-library-prompt-multi-select-design.md)
- [ADR-049: Local Prompt Retained Version History](049-local-prompt-retained-version-history.md)
- [ADR-057: Portable Chatbook Prompt Records](057-portable-chatbook-prompt-records.md)
