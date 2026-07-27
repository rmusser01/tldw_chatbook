# Minimal RAG Citations in Console

Date: 2026-07-27

Status: User-approved direction and final document review; pending final user review

## Goal

For a persisted local RAG answer in Console, show which retrieved chunks the
answer cited and let the user open the corresponding item in Library.

The feature is:

```text
RAG retrieval
  -> existing prompt evidence and [S#] markers
  -> persist the selected answer's marker mappings
  -> Sources (N)
  -> view cited chunks
  -> open the existing Library item
```

## Existing pieces to reuse

Chatbook already has almost everything needed:

- the citation trace stores exact prompt chunks and their source identity;
- the selected assistant message owns its trace;
- `[S#]` parsing and validation already run during citation checking;
- `hydrate_trace()` already reads the stored chunks and identities with the
  current authorization checks;
- Library already opens local RAG results through an exact
  `source_type`/`source_id` method.

This work must reuse those paths. It must not add another provenance model or
storage system.

## Required behavior

### Persist the selected answer's citations

At the existing terminal citation-finalization step, store every eligible
`[S#]` occurrence for the answer body that is actually persisted. Known
ordinals point to entries in the already-stored final prompt evidence set and
use `VALID`; unknown ordinals use `UNKNOWN_MARKER`. This satisfies the existing
trace invariant that stored occurrences exactly match all eligible marker
spans. The footer and panel use only `VALID` occurrences.

This is the only pipeline change. Do not persist a second citation model,
facts, claims, semantic support, or additional repair history for this feature.
Old traces with no stored occurrences remain unchanged and simply have no
Sources footer.

### Console footer

After a cited assistant message is persisted, render one focusable row directly
under it:

```text
Sources (3)
```

The count is the number of distinct prompt evidence entries referenced by valid
stored occurrences. Repeated references to the same entry count once. Invalid
or unknown markers do not count.

Do not show the footer while a message is streaming, before persistence, when
the active message/trace association fails verification, or when there are no
valid cited entries.

Footer metadata is loaded outside Textual `compose()`/`recompose()`. Add one
narrow repository helper that accepts the persisted message ID and current
body, reads the persisted revision, and performs
`get_active_trace_for_message()` using the repository's existing private
fingerprint codec. Console must not read or expose that codec.

On transcript load and after a new assistant message is persisted, a
background worker uses that helper and caches only the footer count for the
message. The worker captures the persisted message ID, current body, and a
request generation. Its result is discarded unless all three still match the
current transcript before application.

### Sources panel

Activating the footer opens a simple Sources panel containing only cited
entries, in first-citation order. Each entry shows:

- source title when available;
- the exact stored chunk sent to the model;
- an `Open in Library` action when the source is a supported local item.

Use one `ModalScreen` at every terminal width. It contains a cited-source list
and one scrollable chunk-detail region. Do not integrate with or replace the
Console right rail, and do not add responsive rehosting.

The modal loads citation data in a worker only when opened. It first obtains an
existing active-trace result through the narrow repository helper, then calls
the existing bounded `hydrate_trace()` path. Immediately after hydration and
before applying the result to the modal, it calls
`verify_active_trace_result()` on the same result and discards the hydration
when verification fails.

For this local Console feature, construct `CitationReadAuthorization` directly
from `repository.identity_context`:

- `authority_scope=LOCAL_PROFILE`;
- `profile_id` and `governance_scope_id` both use the active local profile;
- `allowlisted_authority_ids` contains only the active local authority;
- only `view_snapshot` and `view_source_identity` are enabled.

All other authorization flags remain false. This is a request object for the
existing hydration API, not a new authorization service.

Transcript rendering must not load or cache chunk bodies.

Stored chunks are untrusted text. Render them literally with Rich/Markdown
markup and link interpretation disabled.

### Open the original item

For local media, notes, and conversations, read the stored `source_kind` and
`source_id`, map the kind to Library's existing `source_type`, and pass the
pair through the same exact-ID opening path already used by Library RAG search
results.

Use one small static mapping:

```text
media_db    -> media
notes       -> notes
chat_history -> conversations
```

Console sends the mapped type and ID in a bounded Library navigation context.
Library validates both values and calls its existing
`_open_library_item_by_id()` path. No other source kind receives an open
action.

Do not execute stored paths or URLs, perform fuzzy lookup, call
`tldw_server`, or create a resolver registry. If the item no longer exists or
cannot be opened, Library owns its existing unavailable warning. The
historical chunk remains stored and is visible again when the user reopens the
Sources panel; this feature does not add a preflight lookup or return journey.

## Error handling

- Missing, stale, or body-mismatched trace: hide the footer.
- Hydration unavailable or access denied: show one `Sources unavailable`
  state. Do not add partial-hydration repository APIs for this feature.
- Unsupported source type: show the chunk but no open action.
- Library item missing: let Library show its existing unavailable warning.
- Never treat structural citation validity as proof that the source supports
  the answer.

## Explicitly out of scope

- new tables, migrations, policy versions, or trace formats;
- capability-object systems or revocation event buses;
- source resolver frameworks or plugin systems;
- fact, claim, or semantic-support stores;
- Console right-rail integration or responsive inspector rehosting;
- `tldw_server`, Sync, export/import, or cross-device work;
- current-source comparison, automatic refresh, or source observations;
- inline marker links, bibliographies, citation formatting, or uncited-context
  analysis.

## Verification

Use scoped tests only:

- all eligible selected-answer occurrences persist and survive reload, with
  known ordinals valid and unknown ordinals marked unknown;
- the footer count deduplicates valid cited entries;
- no footer appears for streaming, stale, uncited, or legacy empty traces;
- footer discovery runs outside compose/recompose and opening Sources
  revalidates the active trace;
- the repository helper keeps the fingerprint codec private and reads the
  persisted revision internally;
- stale footer and hydration workers cannot update a changed transcript or
  modal;
- local hydration authorization enables only snapshot and source-identity
  reads;
- opening the footer loads and renders exact stored chunks;
- one modal works at both wide and narrow terminal widths;
- chunks render as literal text without markup or link interpretation;
- failed all-or-nothing hydration shows one bounded unavailable state;
- media, note, and conversation actions use the existing exact-ID Library
  path;
- missing/unsupported items fail safely without hiding historical chunks.

## Delivery

Implement this as one focused feature task and one PR. Split it only if the
code review shows the persistence and Console changes cannot be safely reviewed
together.

ADR required: no

ADR path:
`backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`

Reason: this uses the existing citation trace, authorization, and Library
navigation paths without changing their architecture.
