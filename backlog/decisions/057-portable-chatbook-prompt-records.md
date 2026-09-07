# ADR-057: Portable Chatbook Prompt Records

Status: Accepted
Date: 2026-08-12
Related Task: [TASK-197](../tasks/task-197%20-%20Bulk-export-of-prompts-via-the-chatbook-format.md)
Supersedes: N/A

Allocation note: ADR-057 was allocated after sweeping every visible remote ref
and open pull request on 2026-08-12. ADR-056 was the highest allocated number;
no visible ref or pull request reserved ADR-057.

## Decision

Keep the Chatbook container at version 1.0 and add a versioned Prompt-record
payload inside the existing `ContentType.PROMPT` seam. A new Prompt record uses
`record_schema = "tldw-chatbook-prompt"` and `record_version = 1` and carries
only portable artifact content:

- name, author, and details;
- separate System and User compatibility lanes, preserving `None`, empty, and
  non-empty values;
- the canonical active keyword membership;
- `artifact_type`, `prompt_format`, `prompt_schema_version`, and the exact
  stored `prompt_definition` text.

The payload does not carry the source row ID, UUID, client ID, optimistic or
sync version, source timestamp values, deletion state, retained-history rows,
collection memberships, or usage state. Chatbook manifest entries use
archive-local opaque Prompt IDs such as `item-000001`; the existing v1 manifest
timestamp slots remain present as `null`. Source database IDs are used only
while collecting the archive and never appear in its paths, manifest, or
payloads.

The importer accepts both this strict record and the legacy unversioned
Chatbook Prompt payload (`name`, `description`, and one `content` value, plus
the historically emitted optional `id`, `created_at`, and `updated_at`
metadata). A legacy `content` value retains its historical meaning as a legacy
Prompt's System lane; those optional legacy metadata values are ignored.
Unknown record versions, partial new-format records, invalid field types, and
malformed archive structure fail closed for that record before a database
mutation. Artifact compatibility state is not silently repaired:
legacy, supported Console-v2, foreign-v1, future/unsupported, malformed, and
mismatched stored definitions round-trip as stored data and remain subject to
ADR-040 when a later operation tries to edit or apply them.

Library Export adds a local-only `prompts` scope and includes Prompts in
`everything`. Counts and selections issue fresh uncapped active-ID queries
against `PromptsDatabase`; they never reuse the rendered 50-row Prompt browse
page. Prompt collection is all-or-nothing: if a selected active row disappears,
cannot be read, or cannot be encoded, the Chatbook export fails and its partial
archive is not finalized. Other content collectors retain their existing
behavior.

## Context

Chatbook version 1.0 already declares `ContentType.PROMPT`, tracks
`total_prompts`, writes one JSON file per Prompt, and routes Prompt import. The
existing record is not portable or lossless: export chooses either System or
User text as one `content` value and drops author, keywords, artifact type,
format, schema version, and definition; import always creates a legacy Prompt
with that one value in the System lane.

ADR-040 makes Prompt versus Recipe, schema dispatch, compiled compatibility
lanes, and exact stored definition part of the artifact contract. TASK-196 and
TASK-198 added independent retained-history and collection lifecycles. A bulk
backup that captured either lifecycle implicitly would blur ownership and make
ordinary re-import restore state the user did not select. The portable record
therefore represents the current artifact only.

A whole-Chatbook version bump is unnecessary because the container already
supports Prompt content and its manifest is forward-compatible with additive
per-content payloads. Versioning the Prompt record itself gives readers an
explicit dispatch point without forcing migrations for conversations, notes,
media, characters, or kept briefings.

## Required Boundaries

- Encoding and decoding live in one pure Prompt-record codec shared by
  `ChatbookCreator` and `ChatbookImporter`; UI code does not construct archive
  payloads.
- Prompt export reads each active Prompt row and its active keyword membership
  through one export-specific `PromptsDatabase` snapshot method and one SQLite
  read transaction. That method owns `BEGIN`/commit/rollback directly, uses its
  own cursor rather than the generic transaction or query/logging helpers, and
  converts failures to a fixed, category-only error without logging rollback
  details.
- New-format decoding is strict and pre-mutation. Python bools are not accepted
  as integer schema versions; strings, lists, and optional values are not
  silently coerced.
- `prompt_definition` is exported as its exact stored JSON text or `None` so a
  compatibility-only record is not canonicalized, upgraded, or repaired as a
  side effect of backup and restore.
- Keyword order follows the database's canonical active-membership read. Import
  restores semantic membership through the ordinary Prompt keyword write path.
- New local identity, version, and timestamps are generated by the destination
  database. The archive cannot overwrite those values.
- The TASK-197 Prompt collection path has a dedicated sanitized failure type
  handled before `ChatbookCreator`'s broad exception handler. Diagnostics added
  or modified for this path may report a fixed operation, archive-local item
  ID, count, and exception category. They must not log Prompt names, details,
  lanes, keywords, definitions, source IDs, exception messages, or tracebacks.
- Library export count/selection recovery logs only the validated scope kind and
  exception category. It never renders `ExportScope`, explicit IDs, exception
  text, or a traceback, and user-facing resolution failure copy is fixed.
- Library Prompt export reuses the existing export canvas, worker,
  cancellation, overwrite, progress, and recovery ownership. It does not add a
  second bulk-export dialog.
- Server mode remains refused because Library Chatbook export reads local
  databases only.
- Prompt export is not silently partial. An in-scope Prompt failure aborts the
  archive before final replacement of the destination.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Bump the complete Chatbook container to version 2.0 | Prompt is already a declared v1 content type. A container bump would impose unrelated compatibility work on every other content family. |
| Store each existing per-Prompt Markdown export inside the Chatbook | This duplicates the JSON content seam and couples bulk correctness to a presentation-oriented parser rather than a strict record contract. |
| Reuse the server Prompt payload verbatim | That adapter canonicalizes/deserializes definitions and carries optional local sync identity fields; the Chatbook record needs exact stored-definition preservation and explicit identity exclusion. |
| Export source Prompt IDs in the manifest and filenames | It leaks local identity and gives imports an accidental suggestion that row identity should be restored. Archive-local ordinals are sufficient for selection and lookup. |
| Include retained history and collection memberships | Those are separately owned optional lifecycles from ADR-049 and TASK-198, not content intrinsic to the current Prompt/Recipe artifact. |
| Skip a Prompt that fails during export | A successful-looking partial backup is data loss. The entire Prompt-bearing archive must fail before finalization instead. |
| Reject compatibility-only or malformed stored definitions | Backup must preserve current stored artifacts without claiming they are editable. ADR-040 remains the authority for later apply/edit eligibility. |

## Consequences

### Benefits

- Bulk Chatbook export preserves both Prompt lanes, metadata, keywords, and
  Prompt/Recipe structured content without importing source identity.
- Existing Chatbooks with legacy Prompt files remain importable.
- Prompt-record evolution has its own explicit version without widening the
  container migration surface.
- `Everything` exports become a complete current-artifact backup across the
  Library's four supported source families.
- A Prompt export cannot report success after silently dropping an item.

### Accepted trade-offs

- Re-import creates new UUIDs, versions, and timestamps and does not restore
  history or collections.
- The archive-local Prompt ID (for example `item-000001`) differs from the
  source database ID; the existing importer convention resolves it as
  `content/prompts/prompt_item-000001.json`.
- Legacy Chatbooks remain inherently lossy because information absent from
  their old `content` payload cannot be reconstructed.
- Compatibility-only definitions can be restored as compatibility-only data;
  they remain non-editable/non-applicable until handled under ADR-040.

## Links

- [TASK-197 design](../../Docs/superpowers/specs/2026-08-12-task-197-bulk-prompt-chatbook-export-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-040: Versioned Prompt Artifacts and Safe Improvement Transactions](040-versioned-prompt-artifacts-and-safe-improvement-transactions.md)
- [ADR-049: Local Prompt Retained Version History](049-local-prompt-retained-version-history.md)
