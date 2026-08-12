# TASK-197 Bulk Prompt Chatbook Export Design

Status: Approved
Date: 2026-08-12
Task: [TASK-197](../../../backlog/tasks/task-197%20-%20Bulk-export-of-prompts-via-the-chatbook-format.md)
Decision: [ADR-057](../../../backlog/decisions/057-portable-chatbook-prompt-records.md)

## Summary

Add current local Prompts and Recipes to the Library's existing Chatbook bulk
export workflow. The archive gains a strict, versioned Prompt record that
round-trips portable artifact content into a fresh Prompt database while
excluding local identity, retained history, collections, deleted records, and
usage state. The Library Prompt list regains one visible `Export…` action that
opens the shared export canvas already used by Media, Conversations, and Notes.

The design deliberately does not create a second export dialog, a new archive
container version, a second Prompt model, or a dependency on the rendered
Prompt page.

## User Outcome

A user can open Library → Prompts, press `Export…`, choose a destination in the
existing export canvas, and receive one `.zip` Chatbook containing every active
current Prompt and Recipe. `Everything` exports include Prompts too. Importing
that Chatbook into a fresh profile restores names, descriptions, authors,
separate System/User lanes, keywords, Prompt/Recipe type, and structured or
compatibility-only stored definitions.

The import creates destination-owned identity and current version state. It
does not recreate deleted rows, retained history, collection membership, usage,
or source timestamps.

## Goals

1. Define one lossless portable Prompt-record contract inside Chatbook v1.
2. Preserve legacy Chatbook Prompt import behavior.
3. Add uncapped Prompt counts and selections to Library export scope.
4. Reuse the existing local-only export UI and execution pipeline.
5. Fail an archive before finalization rather than silently omitting an
   in-scope Prompt.
6. Prove the behavior through real file-backed SQLite and real Chatbook ZIP
   round trips.

## Non-goals

- Exporting retained Prompt history.
- Exporting Prompt collection definitions or memberships.
- Exporting soft-deleted Prompt or Recipe rows.
- Preserving local row IDs, UUIDs, client IDs, versions, timestamps, or sync
  lineage.
- Bulk Prompt export from server mode.
- Server Chatbook schema/API changes.
- Changing per-Prompt Markdown export/import.
- Adding collection filters, selected-row export, or a Prompt-only import
  wizard in this task.
- Changing same-name Chatbook conflict behavior.
- Adding a new modal, controller, CSS module, or dependency.

## Existing System

The Chatbook container already has:

- `ContentType.PROMPT`;
- `ChatbookManifest.total_prompts`;
- one `content/prompts/prompt_<id>.json` file per selected Prompt;
- Prompt routing in `ChatbookCreator`, `ChatbookImporter`, and
  `LocalChatbookService`.

The existing Prompt JSON contains only `id`, `name`, `description`, one
`content` value, and guessed timestamps. Export chooses System text or User text
instead of preserving both. Import puts `content` into a legacy Prompt's System
lane and drops author, keywords, artifact type, format, schema version, and
definition.

Library bulk export already owns uncapped Media/Conversation/Note scope
resolution, a shared form canvas, background counts, destination normalization,
overwrite confirmation, progress, cancellation, Retry, and durable receipt
recording. The Prompt list intentionally removed an unwired `Export…` button
while awaiting this task.

## Design

### 1. Portable Prompt record

Create a small pure codec at
`tldw_chatbook/Prompt_Management/prompt_chatbook_record.py`. It has no Textual or
database imports and exposes two public functions:

```python
encode_chatbook_prompt_record(detail: Mapping[str, Any]) -> dict[str, Any]
decode_chatbook_prompt_record(payload: Mapping[str, Any]) -> dict[str, Any]
```

The encoded JSON shape is:

```json
{
  "record_schema": "tldw-chatbook-prompt",
  "record_version": 1,
  "name": "Summarize research",
  "author": "Ada",
  "details": "A two-lane Prompt",
  "system_prompt": "You are a careful analyst.",
  "user_prompt": "Summarize {{topic}}.",
  "keywords": ["analysis", "research"],
  "artifact_type": "prompt",
  "prompt_format": "structured",
  "prompt_schema_version": 2,
  "prompt_definition": "{\"kind\":\"block_prompt\",...}"
}
```

Contract details:

- `record_schema` and `record_version` are required and exact.
- `name` is a nonblank string.
- `author`, `details`, `system_prompt`, `user_prompt`, and
  `prompt_definition` are strings or `null`; empty strings remain empty.
- `keywords` is a list of strings in the source database's canonical active
  membership order.
- `artifact_type` is exactly `prompt` or `recipe`.
- `prompt_format` is exactly `legacy` or `structured`.
- `prompt_schema_version` is an exact integer (never bool) or `null`.
- `prompt_definition` is the exact stored database text, not a parsed or
  canonicalized object. This preserves foreign, unsupported, malformed, and
  mismatched compatibility states without pretending to repair them.
- Extra keys are rejected so a misspelled field cannot be silently lost.

The decoder returns only ordinary `PromptsDatabase.add_prompt` content fields.
It never returns source identity or lifecycle values.

### 2. Legacy record compatibility

If both version markers are absent and the payload has the historical
`name`/`description`/`content` shape, decode it using the established behavior.
The exact known legacy key set additionally permits the historically emitted
optional `id`, `created_at`, and `updated_at` fields. When present, `id` must be
an exact integer (not bool) and each timestamp must be a string or `null`; all
three are then ignored:

- `name` → name;
- `description` → details;
- `content` → System lane;
- User lane, author, and definition → absent;
- keywords → empty;
- artifact type → Prompt;
- format → legacy.

A payload with one version marker missing, an unknown version, a mix of new and
legacy-only keys, an unknown extra legacy key, or an invalid field type is not
treated as legacy. It fails closed with bounded recovery copy and no Prompt
mutation.

### 3. Archive-local identity

Source database IDs are input selectors only. While collecting Prompts, the
creator assigns deterministic archive-local ordinals in selection order:

```text
item-000001
item-000002
...
```

The ordinal becomes the manifest `ContentItem.id` and the filename
`content/prompts/prompt_item-000001.json`, matching the unchanged importer rule
`prompt_<manifest-id>.json`. No source ID, UUID, version, client ID, or source
timestamp value is written. The manifest item uses the Prompt name as title,
details as description, and its required v1 `created_at`/`updated_at` slots
serialize as `null`.

The importer continues to locate files through manifest Prompt item IDs, so
legacy numeric IDs and new archive-local IDs both work.

### 4. Export collection and failure semantics

Add `PromptsDatabase.fetch_prompt_chatbook_snapshot(prompt_id)`. It opens one
SQLite read transaction, uses one local cursor to read the active Prompt row
and its active keyword membership, and returns one detached mapping. It does not
route through `transaction`, `execute_query`, `get_prompt_by_id`, or
`fetch_keywords_for_prompt`: the generic transaction helper logs rollback
exception text/tracebacks, the generic query diagnostics include source
parameters, and the existing separate reads are not a coherent snapshot. The
method owns `BEGIN`, commit, and best-effort rollback directly without logging
failure details. SQLite or shape failures become one fixed `DatabaseError` with
no exception chaining, message, source ID, or traceback in this task-owned
seam.

`ChatbookCreator._collect_prompts` fetches each selected row through that
snapshot method. It encodes and writes one record, appends the manifest item,
and advances progress.

Unlike the historical collector, it does not catch-and-continue. A missing
selected row, database error, validation failure, or write error is wrapped as
a repr-safe `PromptChatbookExportError` containing only the archive-local item
ID and exception category. `create_chatbook` catches that type before its
existing broad traceback/message handler, emits only a fixed category-level
diagnostic, and returns fixed recovery copy. Cancellation retains its existing
separate path. The creator's existing temporary-work-directory and atomic
`.partial` → destination replacement boundary ensures no successful partial
archive is finalized.

Diagnostics added or modified for TASK-197 use fixed operation names,
archive-local ordinals, counts, and exception categories only. They never
include Prompt names, descriptions, lanes, keywords, definitions, source IDs,
exception messages, or tracebacks.

The Library screen's count and both inline/worker selection-resolution recovery
branches are tightened at the same boundary. They log only the validated scope
kind and exception category, never `scope!r`, explicit IDs, exception text, or
`exception=True`; user-facing selection failure copy is fixed and does not
interpolate the exception.

### 5. Import behavior

`ChatbookImporter._import_prompts` loads JSON, dispatches through the shared
codec, applies the existing optional `[Imported]` prefix, and writes through
`PromptsDatabase.add_prompt`. Validation completes before the write. Destination
identity, version, timestamps, sync event, FTS rows, keyword links, and the one
ordinary destination `create` history snapshot are created by the normal
database path.

The existing same-name behavior remains unchanged for TASK-197: a conflict is
reported for that item rather than silently overwriting current content.
Chatbook `prefix_imported` remains available to avoid a collision. Generalizing
Prompt conflict policies is separate work because the current Prompt importer
does not implement the passed `ConflictResolution` strategy.

One invalid Prompt record increments the import failure count and leaves that
Prompt absent; it does not partially write lanes before discovering an invalid
definition field. Other independently valid archive items retain the existing
Chatbook import behavior.

### 6. Uncapped Library scope

Add `PromptsDatabase.get_all_active_prompt_ids() -> list[int]`, implemented as
one ordered `SELECT id FROM Prompts WHERE deleted = 0 ORDER BY id`. This mirrors
the existing uncapped Media and ChaChaNotes ID seams and avoids paging through a
UI-oriented 50-row browse contract.

Extend `library_export_scope` with:

- `kind="prompts"`;
- a `PromptIdSource` protocol;
- a fourth stable count key, `prompts`;
- `ContentType.PROMPT` selection mapping;
- Prompt inclusion in `everything`;
- labels such as `Prompts · 207 items` and
  `Everything: … · 207 prompts`.

Explicit IDs remain allowed only for a single source by the generic
`ExportScope` contract, although this task does not add Prompt row selection
UI.

### 7. Library UI and lifecycle

Restore `Export…` to the existing Prompt list toolbar after `Import…`. Pressing
it calls `_open_library_export_canvas(ExportScope(kind="prompts"))`. The export
canvas renders no media-quality controls for this scope.

The screen resolves `app_instance.prompts_db` alongside the existing Media and
ChaChaNotes handles. Counts and final selection resolution receive all three
handles. In-memory fixtures retain the existing inline-count exception;
file-backed production databases use the current exclusive worker.

No new canvas, modal, state model, controller, CSS, keybinding, or footer hint is
needed. The three compact toolbar Buttons remain the only children in the
existing auto-height `ds-toolbar` row.

Server mode continues to show the existing local-export refusal before any
database query. Navigating away or opening another scope retains the current
request-token and cancellation behavior.

## Data Flow

```text
Library Prompt list
  -> ExportScope(kind="prompts")
  -> fresh uncapped active Prompt IDs
  -> existing Library export canvas + worker
  -> LocalChatbookService.export_chatbook
  -> ChatbookCreator
  -> Prompt record codec
  -> atomic Chatbook ZIP

Chatbook ZIP
  -> ChatbookImporter
  -> Prompt record codec (new v1 or legacy payload)
  -> validate before mutation
  -> PromptsDatabase.add_prompt
  -> destination-owned identity/version/history
```

## Error and Recovery States

- **No active Prompts:** count settles to zero, existing empty-scope copy is
  shown, and Export remains disabled.
- **Count failure:** existing quiet all-zero recovery behavior applies and the
  form remains safe; diagnostics carry only scope kind and exception category.
- **Prompt disappears after counting:** export fails with fixed copy; Retry
  re-resolves fresh IDs and counts.
- **Unknown record version:** import reports an unsupported Prompt-record
  version; no Prompt is written.
- **Invalid new record:** import reports an invalid Prompt record; no partial
  row or keywords are written.
- **Legacy record:** import succeeds with its historically available data only.
- **Server mode:** existing warning explains that Library Chatbook export is
  local-only.
- **Cancellation:** existing cancellation state applies; no finalized partial
  destination is left.

## Security and Privacy

- All archive JSON is written with UTF-8 and `ensure_ascii=False`; user text is
  data, never markup, a path component, or a diagnostic.
- Filenames use archive-local ordinals only.
- JSON schema dispatch is explicit; no unsafe deserialization or dynamic type
  construction is introduced.
- Prompt body and definition text are expected archive content but forbidden
  from logs, errors, and persistent diagnostics.
- Import validates types and versions before calling the database.
- Existing destination/path validation and atomic ZIP finalization remain the
  authority.

## Testing Strategy

### Pure codec

- Exact new-record encode/decode for `None`, empty, multiline, Unicode, emoji,
  RTL, and markup-looking values.
- Prompt and Recipe, legacy and structured, supported and compatibility-only
  stored definitions.
- Unknown/missing/bool versions, extra/misspelled fields, invalid enums,
  invalid keywords, and invalid optional values fail closed.
- Legacy payload compatibility remains exact.
- Repr/log capture contains no body or definition sentinels on failure.
- Adversarial explicit-ID and exception-message sentinels are absent from
  count/selection, snapshot, collector, and importer logs/status copy; no task
  path emits a traceback.

### Database and scope

- `get_all_active_prompt_ids` returns more than 100 ordered active IDs and
  excludes deleted rows.
- Prompt-only scope never touches Media or ChaChaNotes.
- Everything returns four exact counts and four selections.
- Explicit Prompt IDs resolve through the generic single-source contract.
- Labels remain truthful for zero, one, and large counts.

### Real Chatbook round trip

Use separate file-backed source and destination `PromptsDatabase` instances.
Export a mixed set containing:

- legacy Prompt with distinct multiline System/User lanes;
- structured-v2 Recipe with Unicode and literal `[bold]` text;
- keywords;
- foreign-v1 or other compatibility-only stored definition;
- more than one 50-row Prompt browse page;
- one deleted control row.

Open the actual ZIP and assert the exact Prompt payload key set, deterministic
archive-local manifest IDs, `prompt_<manifest-id>.json` paths, null manifest
timestamp slots, and absence of source lifecycle keys. Do not use raw substring
absence across arbitrary Prompt text, because a user may legitimately place an
ID-looking value in a lane. Import into the destination and compare every
portable content field. Assert new destination identity/lifecycle fields,
exactly one ordinary destination-created history snapshot, and absence of
source deleted/history/collection state.

Mutate one selected source row away between scope resolution and collection;
the export must fail without a finalized partial archive. Add invalid-version
and legacy-archive fixtures through the real importer.

### Mounted Textual UI

- Prompt `Export…` opens the shared canvas with `kind="prompts"`.
- Server mode refuses before counts or export service calls.
- Empty/counting/ready/error/Retry/cancel states reuse existing controls.
- Prompt scope hides media-quality controls.
- At 64x24 and a normal viewport under the generated app stylesheet, Sort,
  Import, and Export remain painted, focusable, and inside the Prompt toolbar;
  the export canvas actions remain reachable. Add
  `library-prompts-export` to the Prompt focus-identity allowlist and verify the
  rendered compositor/frame in addition to widget regions.
- Button labels and scope/error text render literal Unicode/markup-looking
  content where applicable.

Every new guard is mutation-checked: bypass record-version dispatch, reuse a
rendered Prompt page, include deleted IDs, skip a failed Prompt, or leak source
identity, and confirm the focused regression turns red.

## Documentation

Update `Docs/User_Guide/library/prompts.md` to replace the “No bulk export yet”
note with the current workflow, local-only authority, Everything behavior,
round-tripped fields, legacy compatibility, and explicit exclusions. Update any
Library export guide summary whose three-source count becomes four-source.

## ADR Check

ADR required: yes.

ADR path: `backlog/decisions/057-portable-chatbook-prompt-records.md`.

Reason: TASK-197 establishes a durable portable artifact schema, backward-
compatibility dispatch, identity/privacy exclusions, and an all-or-nothing
cross-module export boundary.
