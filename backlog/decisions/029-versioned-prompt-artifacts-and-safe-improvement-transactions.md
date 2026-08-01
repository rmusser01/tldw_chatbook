# ADR-029: Versioned Prompt Artifacts and Safe Improvement Transactions

Status: Accepted
Date: 2026-08-01
Related Task: N/A
Supersedes: ADR-028

## Decision

Store Console-authored block Prompts and Recipes as structured prompt schema
version 2, with a first-class `artifact_type` discriminator and deterministic
compiled System and User compatibility text. Preserve the existing server
structured prompt schema version 1 unchanged. Route prompt-improvement work
through a sensitive auxiliary-completion boundary and apply results through an
exact, segment-aware Console composer transaction.

Every Prompt record gains:

- `artifact_type = "prompt" | "recipe"`, defaulting existing records to
  `"prompt"`;
- `prompt_format = "structured"` for editable block artifacts;
- `prompt_schema_version = 2` for the Console block schema;
- a canonical `prompt_definition` whose `kind` is `block_prompt` or
  `block_recipe` and matches `artifact_type`;
- compiled `system_prompt` and `user_prompt` fields for legacy consumers.

The discriminator is included in local and server brief, detail, search,
create, update, import, and export contracts. Prompt execution and selection
paths reject Recipes. Recipes remain first-class Library records, but selecting
one creates an unsaved Prompt working copy rather than applying it directly.
No parallel Recipes table is introduced.

Structured schema dispatch is explicit. Existing server schema v1 keeps its
flat `blocks`, variables, roles, and assembly configuration. Console schema v2
uses System and User lanes with editable Free-form, Markdown, and XML blocks.
A stored v1 definition is never parsed as v2, silently rewritten, or saved
through the v2 editor. It may be viewed through its compiled compatibility
text and explicitly converted into a new v2 record.

The Prompt scope boundary normalizes local JSON text and server dictionaries
into typed artifact states. Malformed, unsupported, and foreign-v1 definitions
remain distinguishable from legacy text. Source capabilities declare supported
schema versions, artifact types, field limits, server search, and conditional
update. Missing v2 capability disables v2 writes rather than probing by
destructive save.

Prompt-improvement requests use an immutable `ComposerDraftSnapshot`. The
snapshot preserves exact segment text, origin, display state, label, cursor,
and selection. Ordinary typed and pasted prompt text is eligible for
improvement. Inline-file contents and pending attachments are excluded from
the provider request. Inline-file segments are represented only by protected,
opaque placeholders in model-facing text and are rehydrated exactly during
application. Undo restores the exact prior snapshot.

The auxiliary provider request is marked sensitive through the gateway and
adapter boundary. Provider and model resolution are reused, but tools,
streaming, history, RAG, attachments, staged sources, transcript writes, and
prompt-content logging are disabled. Adapters may log metadata and sizes, but
not request or response content.

Auto and Review return only a rewritten prompt envelope. Structured mapping
returns a `recipe_fill` envelope containing the selected Recipe fingerprint and
block-ID/content entries. The application merges fills into the canonical
Recipe locally. The model cannot modify block titles, syntax, tags, ordering,
lane membership, or mapping hints. Unknown, missing, or duplicate IDs fail
closed; unmatched content may populate one locally created Additional-context
block.

## Context

ADR-028 assigned the new Console block document schema version 1. The server
already owns a different structured prompt schema version 1 whose Pydantic
model forbids extra fields. Reusing the number would cause valid Console block
documents to be rejected and make two incompatible document shapes appear to
share one version.

ADR-028 also inferred Prompt versus Recipe from JSON. Prompt list responses do
not load the full definition, while artifact type affects browse labels,
selection, execution, usage accounting, and save behavior. A durable column is
therefore required even though it adds a migration.

The Console composer is segment-based. Its current full-text getter expands
collapsed inline files, while its full-text setter replaces the segment model
with one literal segment. A string-only improvement transaction would risk
sending file contents and would lose provenance, labels, and exact undo state.

Finally, some provider adapters currently log full payloads. A feature-level
promise not to log content is insufficient unless sensitivity is propagated to
the adapters that construct and log the final request.

## Required Boundaries

- Schema v1 and v2 have separate models, validators, compilers, and explicit
  dispatch. Version 1 behavior is not changed by adding v2.
- `artifact_type` and definition `kind` must agree, and stored and definition
  schema versions must match. A mismatch is a corrupt artifact state and
  cannot be applied or saved in place.
- Structured v2 definition plus compiled text is written atomically as one
  prompt-record update.
- Brief and search records include `artifact_type` plus derived System/User
  lane-presence flags; detail normalization parses and validates
  `prompt_definition` once outside the UI.
- Empty Browse queries use paginated listing. Non-empty queries use backend
  search. Unsupported server search is reported without presenting a partial
  client-side filter as complete.
- The composer owns snapshot, model-text projection, apply, and restore. The
  improvement service never reads or mutates private composer segments.
- Opaque inline-file placeholders contain no filename, path, or file content,
  use collision-free request-specific nonces, and must round-trip exactly in
  original segment order before application.
- Pending attachments remain outside the improvement snapshot and unchanged by
  Apply or Undo.
- The auxiliary provider port carries a sensitive-content policy through every
  adapter. Full request and response payload logging is forbidden for this
  operation.
- Structured model output supplies exactly one value for every selected Recipe
  block and no structure. The selected Recipe remains the sole authority for
  titles, syntax, ordering, lane membership, and mapping hints.
- Conditional Update requires a real expected-version check at the source
  boundary. Servers without it support Save as new only.
- Source field and definition limits are validated before Save. Content is
  never silently truncated.
- A byte-identical rewrite returns `no_change` and creates no apply, undo, or
  usage event.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Reuse structured schema version 1 and discriminate by kind | The server already validates a different closed v1 shape; two meanings for one version would make storage and interoperability ambiguous. |
| Keep Recipe type only inside `prompt_definition` | Brief rows cannot label or filter Recipes without loading every definition, and legacy execution paths could treat Recipes as runnable Prompts. |
| Add a separate Recipes table | Duplicates Library ownership, search, versioning, import/export, and local/server policy without an independent lifecycle benefit. |
| Convert existing v1 records automatically | Can lose variables, developer/assistant roles, assembly rules, and future v1 semantics. |
| Improve `draft_text()` and restore with `load_draft()` | Expands inline files into the request and flattens the composer's segment model on apply. |
| Let the model return a complete block document | Allows model output to mutate the Recipe structure and increases validation and token cost. |
| Rely on feature-level logging discipline | Lower provider adapters can still log the final payload after the feature hands it off. |
| Client-filter the currently loaded server page | Presents incomplete server results as a complete search. |

## Consequences

### Benefits

- Existing structured v1 records retain their original semantics.
- Prompt and Recipe identity is cheap to list, search, filter, and enforce.
- Structured mapping follows the selected Recipe exactly.
- Inline-file contents and attachment state cannot leak through draft
  flattening.
- Improvement Apply and Undo preserve the real Console composer artifact.
- Privacy claims are testable at the final provider-adapter boundary.

### Accepted trade-offs

- Local and server Prompt stores require an `artifact_type` migration and v2
  schema support.
- The feature spans `tldw_chatbook` and `tldw_server2` for full server parity;
  older servers remain readable but cannot accept v2 saves.
- Existing v1 structured records require explicit Save-as-new conversion to
  enter the Console v2 editor.
- Composer segments need a public immutable snapshot contract and explicit
  origins.
- Provider adapters that log payloads need redaction work before improvement
  calls can use them safely.
- Server Update remains unavailable until a true conditional version contract
  is exposed.

## Links

- [Console Prompt Workbench and Improvement Design](../../Docs/superpowers/specs/2026-08-01-console-prompt-improvement-design.md)
- [ADR-028: Structured Prompt Artifacts and Auxiliary Improvement Calls](028-structured-prompts-and-auxiliary-improvement-calls.md)
- [ADR-005: Console Workspace Server Readiness](005-console-workspace-server-readiness.md)
- [ADR-006: Provider-Aware Generation Settings](006-provider-aware-generation-settings.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
