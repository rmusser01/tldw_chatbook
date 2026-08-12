# Local Todo Task API Design

Status: Approved design; implementation pending
Date: 2026-08-11
Related task: TASK-13216
Related ADR: ADR-032 (`backlog/decisions/032-local-agent-tool-permission-boundary.md`)

## 1. Purpose

Replace the Console's full-list `todo_write` operation with item-oriented task
operations that remain correct when a parent agent and fleet children use the
same session task list concurrently.

The current handler validates a caller-supplied list and assigns it with
`store[:] = items`. Two fleet calls can therefore both report success while the
later full-list replacement silently removes the earlier caller's changes. The
new contract gives every task a stable identity, makes mutations atomic, and
uses an expected-version check so concurrent edits to the same task fail
explicitly instead of overwriting each other.

This follows Claude Code's current move from one full-array `TodoWrite` call to
`TaskCreate`, `TaskUpdate`, `TaskGet`, and `TaskList` operations with stable task
IDs. Chatbook keeps its existing `todo_*` naming and display fields rather than
copying Anthropic's contract byte-for-byte.

Primary references:

- <https://code.claude.com/docs/en/agent-sdk/todo-tracking>
- <https://code.claude.com/docs/en/agent-teams>

## 2. Scope

### In scope

- Remove `local:todo_write` from the Console-local catalog.
- Add `local:todo_create`, `local:todo_update`, `local:todo_get`, and
  `local:todo_list` when a session task store is supplied.
- Preserve their full strict canonical schemas while projecting non-mutating
  Google/Cohere native disclosures compatible with each transport.
- Give each task a stable session-local ID and positive integer version within
  the portable JSON exact-integer domain `1..2**53-1`.
- Make create, update, delete, get, and list operations thread-safe.
- Require compare-and-swap semantics for every update and delete.
- Preserve the existing validation bounds and the global invariant that at most
  one task is `in_progress`.
- Continue rendering a transcript task marker after each successful mutation.
- Preserve the task records and ID high-water mark across normal in-process
  Console screen navigation.
- Update the local-tool design, ADR-032, tool inventories, prompts, and tests.
- Reserve ADR-032's exact external-profile segment `__local__` so an external
  profile cannot project tools under the Console workspace-tool principal
  `local:__local__`.

### Non-goals

- Durable task persistence across application restarts.
- Dependencies, blocking relationships, assignment, claiming, or task owners.
- Sharing tasks across separate Console sessions.
- Exposing session tasks through the standalone MCP server. The standalone
  composition supplies no Console session store, so it registers none of the
  four tools.
- Migrating obsolete persisted permission entries for `local:todo_write`.

### 2.1 Reserved Hub identity

`local:__local__` is the synthetic permission and Hub catalog identity owned
by the Console's workspace-local tools. An external MCP profile's
user-controlled `profile_id` normally projects to `local:<profile_id>`; the
exact profile id `__local__` would therefore alias that reserved principal.

The root cause is incomplete validation at three trust boundaries. The normal
save path trimmed surrounding whitespace and rejected delimiters or embedded
whitespace but accepted a normalized `__local__`; the JSON load path admitted
hand-written profile, discovery-snapshot, and runtime records without the save
validator; and `local_tools_from_record()` trusted a raw profile id while
constructing its Hub key. A discovered external `fs_write` could consequently
project as `local:__local__::fs_write`, the same identity resolved for the real
workspace `fs_write` permission.

The exact `__local__` token is reserved. New profiles are rejected before any
persistence. Existing persisted reserved profiles and their associated
discovery/runtime state are ignored on load. Raw reserved catalog records are
dropped independently as defense in depth. The record is not renamed or
reinterpreted, and all other currently valid ids retain their existing error
and normalization semantics. In particular, surrounding whitespace remains
trimmed and embedded whitespace remains invalid. A space-wrapped `__local__`
therefore normalizes to the exact reserved token and is rejected, while case
variants remain distinct valid ids.

## 3. State model

One `SessionTodoStore` belongs to one mutable `ConsoleChatSession`. It lives in
the dependency-light, stdlib-only module
`tldw_chatbook/Agents/session_todo_store.py`; placing it there avoids making
the Chat session store import provider/config/permission machinery or making
the local provider import Chat services. The same
store object is passed to the `LocalToolProvider` used by the parent and every
fleet child for that turn, and it survives provider reconstruction on later
turns in the same Console session.

The store owns:

- an insertion-ordered mapping from task ID to task record;
- a monotonically increasing integer ID counter (IDs are rendered as decimal
  strings and are never reused during the session); the private counter may
  hold `2**53` only as the exhausted high-water sentinel; and
- one state lock protecting records, ID allocation, reads, invariant checks,
  version comparisons, and defensive snapshots; and
- one mutation-serialization lock held from before state mutation until after
  the corresponding transcript callback returns.

A task record has this public shape:

```json
{
  "id": "1",
  "version": 1,
  "content": "Run the focused tests",
  "status": "pending",
  "activeForm": "Running the focused tests"
}
```

`activeForm` is optional. Public reads return defensive copies in creation
order; callers never receive the mutable internal record or collection.

Limits remain 50 live tasks and 500 characters per `content`. `activeForm`
gains the same 500-character boundary so every model-controlled display field
is bounded. Public task IDs and versions are generated by the store, never
accepted on create, and remain in `1..MAX_TODO_NUMBER`, where
`MAX_TODO_NUMBER = 2**53 - 1`. This is JavaScript's largest safe integer: the
upper ceiling through which every integer is exactly representable and
distinguishable by standard JSON/JavaScript number consumers. Compact tool
results therefore remain portable instead of relying on a Python-specific
wider integer range. A task ID is a canonical positive decimal string whose
numeric value is within that domain. A version is an exact built-in integer in
that domain; booleans and `int` subclasses are invalid. The private/persisted
`next_id` high-water may equal `MAX_TODO_NUMBER + 1` only as an exhaustion
sentinel; no live ID or version may equal that sentinel. `content` and
`activeForm` must also encode as strict UTF-8; lone surrogates are rejected
before any state change.

### 3.1 In-process navigation snapshot

Console screen navigation reconstructs `ConsoleChatSession` instances from the
process-memory-only `ScreenStateStore`. The session's explicit screen-state
projection therefore includes a pure-data task snapshot:

```json
{
  "next_id": 4,
  "tasks": [
    {"id": "1", "version": 2, "content": "A", "status": "completed"},
    {"id": "3", "version": 1, "content": "B", "status": "pending"}
  ]
}
```

The snapshot contains records and the next-ID counter only—never locks or
callbacks. Export takes the state lock and returns defensive pure data. Restore
validates exact types, unique bounded canonical IDs, versions in
`1..MAX_TODO_NUMBER`, task/text bounds, the live-task cap, the one-`in_progress`
invariant, and an exact built-in `next_id` in
`1..MAX_TODO_NUMBER + 1` that is greater than every live numeric ID. The upper
value is accepted only as the exhausted high-water sentinel. Existing
creation-order and deleted-ID-gap semantics are otherwise preserved. A missing
snapshot is the legacy shape and restores an empty store. An invalid snapshot
also restores an empty store and emits one fixed payload-free warning rather
than crashing navigation or partially importing state. A valid navigation
round trip preserves deleted-ID high-water state, so IDs are not reused after
returning to Console.

## 4. Tool contract

All results are compact JSON text. Each handler constructs and UTF-8 measures
its complete response before returning it, so the response is at or below the
provider's existing 32-KiB result cap and `_fit_result` never truncates a valid
task response. Validation failures continue to become bounded failed
`ToolResult` values at the provider boundary.

Handlers validate the raw argument mapping rather than relying on JSON Schema,
because `LocalToolProvider.invoke` passes raw arguments directly to handlers.
Every tool rejects unknown properties. This makes direct Python invocation and
canonical schema-mediated local/MCP/UI validation obey the same contract.
Canonical schemas and raw validators both enforce the `MAX_TODO_NUMBER`
ceiling for IDs, versions, and cursors. A native provider may receive a
capability-compatible disclosure projection, but raw exact handler validation
remains the final enforcement boundary and returns a bounded corrective error
when a model sends a call outside the lowered disclosure.

### Canonical schema and native transport projection

Each local `ToolSchema` remains the authoritative full strict JSON Schema. The
four task schemas retain `additionalProperties: false`, `maxLength` and
`maximum` ceilings, bounded canonical-ID `pattern` constraints, nullable update
semantics, and the delete-only conditional. Local catalog hashing, MCP/UI
validation, and direct schema consumers use that canonical object unchanged.

Native transports project a fresh disclosure copy without aliasing or mutating
the canonical schema:

- Google Gemini `FunctionDeclaration` has mutually exclusive `parameters`
  (its OpenAPI Schema subset) and `parametersJsonSchema` (full JSON Schema).
  The Google converter sends the complete canonical schema under
  `parametersJsonSchema` and omits `parameters` for converted local tools.
- Cohere v2 accepts JSON Schema but strict tools support only a subset. The
  Cohere converter recursively builds a new copy bounded to supported keywords:
  preserve object/property names, types, descriptions, `required`, `enum`,
  supported `anyOf`, and `additionalProperties`; lower a nullable union to the
  supported Cohere nullable shape; and omit unsupported `allOf`, `oneOf`,
  `not`, numeric `minimum`/`maximum` variants, string `minLength`/`maxLength`,
  and `pattern` regex constraints including anchors and lookaheads.

The Cohere projection is disclosure, not authorization or validation. It must
not weaken the canonical definition, definition hash, local validation, or raw
handler checks. Both converters are deterministic pure projections of their
input tool list and require no live provider request.

### `todo_create`

Input:

```json
{
  "content": "Run the focused tests",
  "activeForm": "Running the focused tests"
}
```

- `content` is required, nonblank, and at most 500 characters.
- `activeForm` is optional, must be a string when present, and is at most 500
  characters. `null` is invalid on create.
- Both text fields must be strict-UTF-8 encodable before ID allocation or any
  state change.
- Caller-supplied `id`, `version`, `status`, and every other unknown property
  are rejected.
- New tasks always start as `pending`, version 1.
- Creation fails when 50 live tasks already exist.
- When `next_id == MAX_TODO_NUMBER`, one successful create issues that final ID
  exactly once and commits `next_id == MAX_TODO_NUMBER + 1`. Every later create
  fails atomically with a fixed bounded ID-exhaustion error and performs no
  state change or callback. After complete input validation, ID exhaustion is
  checked before the live-task capacity condition so the sentinel always has
  that fixed outcome.
- Success returns the complete created task record.

### `todo_update`

Input:

```json
{
  "id": "1",
  "expected_version": 1,
  "status": "in_progress"
}
```

- `id` and `expected_version` are required.
- `id` must be a canonical decimal string in `1..MAX_TODO_NUMBER`.
  `expected_version` must be an exact built-in integer in that domain; a
  boolean or `int` subclass is rejected.
- At least one of `content`, `status`, or `activeForm` must be present.
- `content` follows the create bounds.
- `activeForm` may be a bounded string or `null`; `null` removes the field.
- `status` is `pending`, `in_progress`, `completed`, or `deleted`.
- A non-delete update applies only when `expected_version` equals the current
  version, patches only supplied fields, and increments the version once.
- A successful non-delete update returns the complete updated record, including
  its incremented `version`, so it can be used in a subsequent CAS operation.
- `status: "deleted"` applies the same version check and then removes the task.
  It returns `{ "id": "1", "deleted": true, "version": 2 }`.
- `status: "deleted"` must be the only mutation field; combining deletion with
  `content` or `activeForm` is rejected before the version check.
- A valid same-value patch is still an update: it increments the version once,
  emits one callback, and returns the incremented complete record.
- An update after a winning delete returns not-found, regardless of the stale
  caller's `expected_version`.
- Setting a task to `in_progress` fails when another live task is already
  `in_progress`.
- A stale version returns a fixed conflict explaining that the caller must use
  `todo_get` and retry. It does not include the current version: a caller must
  reread the current task rather than blindly resubmit a stale patch. It never
  echoes caller-controlled task content.
- A task at version `MAX_TODO_NUMBER - 1` may be updated or deleted and returns
  version `MAX_TODO_NUMBER`. A task already at `MAX_TODO_NUMBER` cannot be
  updated or deleted because a successful result would require
  `MAX_TODO_NUMBER + 1`, outside the public domain; a matching CAS therefore
  fails atomically with the fixed bounded version-exhaustion error and performs
  no state change or callback.
- Error precedence remains: validate the complete request first; then resolve
  not-found; then compare `expected_version`; then reject version exhaustion;
  then validate the proposed record and global invariant before commit. Thus a
  malformed delete loses to validation, an absent task loses to not-found, and
  a stale CAS loses to conflict even when the live task is at the maximum.
- Unknown properties, including caller-supplied `version`, are rejected.

### `todo_get`

Input: `{ "id": "1" }`. `id` follows the same bounded canonical-decimal rule
as `todo_update`, and unknown properties are rejected.

Success returns the complete task record. An unknown or deleted ID returns a
fixed not-found error.

### `todo_list`

Input: `{}` for the first page, or `{ "cursor": "17" }` to continue after a
previous page. `cursor` must be a canonical decimal task ID in
`1..MAX_TODO_NUMBER` when present; unknown properties are rejected.

Success returns `{ "tasks": [...], "next_cursor": "17" }` or
`{ "tasks": [...], "next_cursor": null }`, with defensive task copies in
creation order. Deleted tasks are absent. The cursor is the last returned
monotonic task ID, so deletion between pages cannot shift an index and cause a
duplicate; tasks created after the first page may appear in a later page.

Pagination is byte-aware rather than count-only. The handler appends the next
task only when the complete compact JSON page, encoded as UTF-8 with
`ensure_ascii=False`, remains within 32 KiB. It returns at least one remaining
task per page; the per-field bounds guarantee a single task always fits. The
cursor advances only through returned tasks. This preserves valid, parseable
JSON for maximum-size ASCII and multibyte task data without weakening the
provider-wide result cap.

A cursor is any canonical positive decimal string in `1..MAX_TODO_NUMBER`
(`"1"`, not `"0"`, `"01"`, `"+1"`, an integer, or a value above the maximum).
It is an exclusive numeric lower bound and does not need to identify a
currently live task. A previously returned cursor therefore remains valid
after that task is deleted. A never-issued or future cursor is also accepted
only within the bounded domain and returns an empty terminal page until tasks
with greater IDs exist. Pagination is live rather than snapshot-isolated:
tasks created after an earlier page may appear once on a later page, updated
tasks keep their position, and deleted tasks disappear. Records that remain
live throughout traversal are neither duplicated nor omitted.

## 5. Concurrency and callback behavior

The mutation-serialization lock covers the complete mutation transaction and
its notification. The state lock is nested only for steps 1–5:

1. find the target task;
2. compare its version;
3. validate the resulting record and the one-`in_progress` invariant;
4. commit the create, patch, or deletion;
5. build a defensive snapshot; and
6. release the state lock; and
7. invoke the transcript callback with that snapshot while retaining the
   mutation-serialization lock.

This topology preserves the relative order of todo mutation snapshots without
holding the state lock across injected/external code. A callback may inspect
the store synchronously or hand a read to another thread without deadlocking;
reads remain available during rendering. Callbacks must not perform reentrant
task mutations. A slow callback delays only later todo mutations; it does not
serialize reads, filesystem, Git, web, MCP, or other fleet tools. No ordering
claim is made relative to unrelated fleet/tool transcript markers, which may
legitimately interleave.

The callback remains a never-raise seam. If it fails, the committed mutation is
not rolled back, the provider returns success, and a fixed payload-free warning
is logged. The callback receives only a defensive snapshot, not the live
mutable task collection.

Concurrency guarantees:

- concurrent creates receive distinct IDs and both remain present when
  capacity and ID space permit; at 49 live tasks, when ID space permits beyond
  the next allocation (`next_id < MAX_TODO_NUMBER`), exactly one of two racing
  creates succeeds and the other receives the fixed capacity error. At the
  terminal ID boundary (`next_id == MAX_TODO_NUMBER`), the final ID is issued
  once and the loser receives fixed ID exhaustion because exhaustion precedes
  capacity;
- updates to different IDs preserve both changes when the requested transitions
  are jointly valid;
- patches to different fields on one task are serialized, but a caller with a
  stale version must reread and retry rather than silently overwrite;
- two agents racing to set different tasks `in_progress` cannot violate the
  global invariant; and
- failed validation, not-found, version conflict, ID exhaustion, or version
  exhaustion performs no mutation and emits no transcript callback.

## 6. Permissions and catalog migration

`todo_create` and `todo_update` carry `tags=("mutates",)` and therefore retain
the ADR-032 approval risk floor. `todo_get` and `todo_list` carry no mutation
tag and follow the normal resolved local-tool permission state.

The four schemas are registered only when `SessionTodoStore` is provided.
Constructions without session state retain no todo capability. Removing
`todo_write` leaves any old permission-store override inert; it is not copied
to the new tools because one broad historical grant must not silently authorize
four new definitions. Definition hashes and the existing permission machinery
handle each new tool normally.

The Console's exact catalog grows by three entries relative to the old single
tool. Discovery and exact-inventory tests must be updated; no direct-disclosure
threshold is changed.

### Atomic rollout boundary

The provider/native-adapter work and Console session/composition work are one
deploy/merge unit. The intermediate provider commit must not be merged,
released, or deployed alone because the current Console still passes the
legacy mutable list into `LocalToolProvider`. The following Console change must
remove that list seam, inject `SessionTodoStore`, and run the reachable
provider, native-projection, integration, and Console suites before combined
review. No temporary `todo_write` or list-to-store compatibility shim is
permitted.

## 7. UI behavior

The transcript marker continues to show task content and status in creation
order. IDs and versions are protocol coordination fields and do not need to be
displayed in the existing marker. Create, patch, and delete each append one
marker based on the committed defensive snapshot. Get/list operations do not
append transcript markers.

At the display boundary, `format_todo_marker` continues flattening line breaks
and additionally replaces C0, DEL, and C1 terminal-control characters with
spaces before rendering. The stored task text and tool-read JSON retain the
validated Unicode content; only the terminal-facing projection is sanitized.

No durable data migration is required because Console task state is
session-lifetime and in memory only. The in-process screen-state projection and
restore described in §3.1 are required so ordinary navigation does not reset a
live session.

## 8. Errors and privacy

Errors are short, deterministic, bounded, and non-reflective where caller
content could contain private data. Required cases include invalid input, task
limit reached, task not found, version conflict, ID exhaustion, version
exhaustion, and the one-`in_progress` invariant. IDs and integer versions may
appear in errors; task content and `activeForm` do not.

Callback diagnostics are fixed and payload-free. No exception object, task
content, workspace path, credential-like string, or raw argument is logged.

## 9. Verification

Tests must be RED before production changes and must cover:

- a real profile save accepting `__local__` before the fix and rejecting it
  without modifying persisted state after the fix;
- a hand-written persisted `__local__` profile plus discovery/runtime state
  entering the catalog before the fix and becoming wholly inert after it;
- raw `local_tools_from_record()` projection producing no tool for
  `__local__`, independently of store filtering;
- a spoofed external `fs_write` demonstrating the pre-fix collision with the
  real workspace tool's `local:__local__::fs_write` permission identity, while
  a legitimate external profile remains present and independently keyed;
- mutation probes removing each save, load, and raw-projection guard and
  proving the corresponding focused assertion turns RED;
- explicit classification of exact `__local__`, a space-wrapped `__local__`
  that trims to the reserved token, embedded-whitespace rejection, and valid
  case variants without broadening the pre-existing profile-id rules;
- a current user-guide contract that names workspace, read-only Git, and web
  Hub tools and rejects wording that presents session todo/task tools as Hub
  inventory; and

- exact removal of `todo_write` and registration/schema projection of all four
  replacement tools;
- unchanged full canonical task schemas, including strict extra-property,
  bound/pattern, nullable-update, and delete-only constraints;
- exact Google native payloads using `parametersJsonSchema` and omitting
  `parameters`, plus proof that conversion does not alias or mutate the
  canonical schema;
- exact Cohere supported-subset payloads preserving allowed disclosure fields,
  lowering nullable unions, and omitting unsupported composition, range,
  length, and regex keywords, plus canonical non-alias/mutation proof;
- raw handler corrective errors for calls that pass a lowered Cohere disclosure
  but violate the canonical contract;
- create/get/list/update/delete happy paths and bounds;
- defensive-copy behavior and creation ordering;
- permission tags and absence when no session store exists;
- deterministic barrier-based concurrent creates retaining both tasks;
- concurrent updates to different tasks retaining both changes;
- same-task stale-version conflict leaving the winner unchanged;
- the one-`in_progress` race allowing at most one winner;
- a 49-live-task plus two-create race allowing exactly one winner without
  exceeding the cap when ID space permits, plus the terminal-ID variant where
  the final ID is issued once and the loser receives fixed ID exhaustion
  because exhaustion precedes capacity;
- callback order, defensive snapshot, no callback on failure, and callback
  exception containment with payload-free logging;
- a callback that synchronously calls get/list directly on the store, including
  a cross-thread read, proving the released-state-lock contract under a bounded
  subprocess that can be terminated if a deadlock survives;
- a callback-blocking barrier proving a second mutation cannot commit or return
  until the first mutation's callback is released; this is the deterministic
  witness that removing the mutation-serialization lock turns the test red;
- byte-aware `todo_list` pagination producing complete parseable JSON for
  maximum-size ASCII and multibyte data without duplicates across cursors;
- exact-boundary, one-over, and very-long values for task ID, version,
  `next_id`, and cursor in store/snapshot tests, plus provider schema and direct
  raw invocation for ID, `expected_version`, and cursor;
- last-ID creation exactly once followed by fixed atomic ID exhaustion, plus
  update/delete from `MAX_TODO_NUMBER - 1` to the maximum and fixed atomic
  version exhaustion at the maximum;
- valid compact JSON at every numeric boundary, with every individual result
  within 32 KiB and every emitted JSON number in the portable exact domain;
- cursor tests that delete the last task from one page before continuing and
  create a new task after the first page;
- strict rejection of lone surrogates before create/update, with no ID
  allocation, state change, callback, or failed-result-after-commit outcome;
- exact delete-only, no-op-update, and update-after-delete semantics;
- valid, missing-legacy, and malformed Console screen-state task snapshots,
  including a navigate-away/back round trip preserving records and next ID;
- transcript projection sanitization for terminal control characters;
- the real parent/fleet shared-provider path; and
- existing Console transcript rendering and local-tool integration flows.

Native projection verification uses exact payload unit tests only; no live
Google or Cohere network call is required. Mutation checks must make Google use
`parameters`, let an unsupported Cohere keyword survive, or alias/mutate a
canonical schema and demonstrate that the corresponding focused test turns
red.

Mutation checks must remove or weaken the mutation-serialization lock, state
lock, version comparison, defensive copy, and one-`in_progress` guard
independently and demonstrate that the corresponding focused test turns red.

## 10. ADR decision

ADR required: yes.

ADR path: `backlog/decisions/032-local-agent-tool-permission-boundary.md`.

Reason: the task changes the public local-tool interface and concurrent state
semantics, so ADR coverage is required. Existing ADR-032 already owns the
local-tool provider boundary and todo permission model; an addendum there is
clearer than creating a competing decision record. The addendum must land
before production implementation. The implementation must also amend the
existing local-agent-tools design to supersede its `todo_write` contract. The
Google/Cohere native-schema projection is a compatibility repair inside the
existing provider adapters, not a new architectural boundary, so it does not
require another ADR.
