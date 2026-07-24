# RAG Citation Provenance and Source Inspection Design

Date: 2026-07-23
Status: Pending user document review; conversational design approved and third
adversarial review findings addressed
Scope: Local and server-backed RAG from retrieval through Console, persistence,
source inspection, artifacts, export, import, and eligible Sync v2 transport

## Architecture decision

ADR required: yes
ADR path:
[`backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`](../../../backlog/decisions/024-rag-citation-provenance-and-source-resolution.md)
Reason: This design changes durable storage and migration, Sync v2 payload
ownership, client/server contracts, cross-module provenance interfaces, source
resolution, and security and privacy policy.

No Backlog task is created by this design. Before implementation planning, the
owner must create or select an epic task and split the work into atomic,
dependency-ordered child tasks that can each ship in one PR.

## Purpose

A user who receives a RAG-generated answer must be able to:

1. See which submitted chunks the answer cites.
2. Distinguish cited chunks from other context submitted to the model.
3. Read the exact submitted snapshot that grounded the historical answer.
4. Open or inspect the original document or item when current permissions and
   source capabilities allow it.
5. Understand when the original has changed, moved, disappeared, become
   unavailable, or been revoked.
6. Preserve this provenance through reload, saved artifacts, policy-permitted
   export/import, cache reuse, and compatible synchronization.

This is not a Console-only decoration. The citation contract starts at
retrieval, records the exact final provider request, survives generation repair
and retries, and is consumed by every supported presentation and persistence
surface.

## Relationship to existing citation work

The earlier
[`2026-05-23-citation-snippet-carry-through-epic-design.md`](2026-05-23-citation-snippet-carry-through-epic-design.md)
established `EvidenceReference`, `EvidenceBundle`, `CitationRef`, stable `[S#]`
markers, citation validation, Console staging, and artifact carry-through.
Those contracts remain compatibility inputs.

This design supersedes the earlier document wherever it describes the durable
answer-level representation. New writes use a canonical per-answer
`CitationTrace`, governed evidence snapshots, and typed source locators.
Legacy evidence bundles and validation metadata remain readable and are
adapted into partial, `legacy_inferred` traces.

## Reference review: tldw_server

The design was compared with `tldw_server` at commit
[`d9c245ac14c40df855d1ab6cd19b3c137b16b47b`](https://github.com/rmusser01/tldw_server/commit/d9c245ac14c40df855d1ab6cd19b3c137b16b47b).
At that revision:

- `RAGResult` can carry retrieved documents, chunk citations, academic
  citations, generated answers, claims, and verification results.
- Retrieved and derived evidence are modeled separately.
- Chunk lineage can identify a source document, chunk index, offsets, page,
  section, and paragraph.
- Structural citation mapping and trust/claim-support evaluation are distinct;
  a valid marker does not prove semantic support.
- Guardrails can require sentence and quote citations and check numeric
  fidelity.
- Post-generation verification can perform one bounded repair.
- Adaptive retrieval, reranking, verification, or cache paths can replace the
  documents and answer before the response becomes terminal.
- The server
  [`DataSource` enum](https://github.com/rmusser01/tldw_server/blob/d9c245ac14c40df855d1ab6cd19b3c137b16b47b/tldw_Server_API/app/core/RAG/rag_service/types.py)
  contains `media_db`, `chat_history`, `notes`, `character_cards`,
  `web_content`, `prompts`, `world_books`, `dictionaries`, `sql`, `kanban`,
  and `claims`.

Chatbook adopts those useful invariants without copying the server's complete
pipeline implementation. The trace is sealed only at the terminal response
boundary. Existing server `documents`, `citations`, and `chunk_citations`
responses remain supported. A server may additionally return an optional,
versioned `grounding_trace` representation of the canonical trace.

## Goals

- One immutable, versioned provenance object per final answer.
- Exact preservation of text submitted to the final provider request.
- Honest separation of retrieval, prompt submission, answer citation,
  structural validity, semantic support, and current-source state.
- Stable citation markers and user-friendly source inspection.
- Resolver support for every source family currently indexable locally or by
  the supported server pipeline.
- Policy-aware snapshot retention, revocation, export, import, and sync.
- Backwards-compatible reading of existing citation metadata and server
  responses.
- Bounded storage, latency, retry, and import behavior.

## Non-goals

- Proving that the model internally used submitted context. The system can
  prove only that evidence was submitted and that the answer referenced it.
- Treating marker syntax as proof that a claim is supported.
- Building a fully dynamic plugin system for arbitrary source resolvers.
- Re-fetching arbitrary URLs recorded in imported or historical traces.
- Embedding credentials, system prompts, raw provider telemetry, or unrestricted
  local paths in traces.
- Automatically downloading inaccessible source content during inspection or
  export.
- Replacing academic or bibliographic citation formatting with chunk citation
  markers.
- Re-indexing citation snapshots as Library or RAG content.
- Shipping the architecture as one implementation PR.

## Approaches considered

### Selected: canonical per-answer provenance trace

Retrieval executions, exact submitted evidence, answer attempts, validation,
snapshots, and resolution policy are assembled into one sealed trace. Existing
payloads adapt into or out of this contract.

This gives downstream consumers one stable object without requiring a
server-scale event system.

### Rejected: extend existing payloads independently

Adding more fields to `EvidenceBundle`, citation validation dictionaries,
Console display state, sidecar persistence, artifacts, and server response
adapters would be initially faster. It would leave several competing sources of
truth and would make retries, cache reuse, migration, and permission changes
ambiguous.

### Rejected: full append-only provenance event ledger

A complete event-sourced ledger would provide excellent audit reconstruction,
but it adds replay, compaction, ordering, retention, and sync complexity that a
single-user TUI does not need for this feature. The design retains immutable
bounded stages inside a final aggregate instead.

## Core concepts

### CitationTrace

`CitationTrace` is the immutable final provenance aggregate associated with an
answer. It contains:

- `schema_version`
- opaque `trace_id`
- original `request_id` and `generation_id`
- `origin`: `local`, `server`, `imported`, or `legacy_inferred`
- `completeness_at_seal`: `complete`, `partial`, `redacted`, or `unavailable`
- lifecycle state, which must be `sealed` before a grounded answer is persisted
- ordered references to one or more `EvidenceRun` records
- one or more exact `PromptEvidenceSet` records
- one or more bounded `AnswerAttempt` records
- the final selected attempt
- structural and semantic trust summaries
- seal-time policy decisions and policy version
- creation and sealing timestamps

The trace is immutable after sealing. Current-source observations and imported
trace rebinding are separate append-or-replace observations; they do not rewrite
historical evidence.

The immutable trace stores opaque identities, marker ordinals, validation
results, stage relationships, and references to governed payloads. Submitted
text, source identity, title, lineage, locator, content hashes, comparison
fingerprints, and non-final answer-attempt text do not appear in aggregate JSON.
A hydrated trace view may resolve those payload references for an authorized
caller, but payload redaction or secure purge leaves only the sealed
non-sensitive metadata and opaque reference. Current access, redaction, and
observation state are computed separately from the seal-time record.

### CitationTraceBuilder

`CitationTraceBuilder` is an internal, mutable request-scoped accumulator. It
accepts retrieval runs, prompt sets, answer attempts, validation results, and
repair outcomes. It is not persisted as authoritative provenance and is never
shown as a completed trace.

The builder seals exactly once after all answer mutations have ended:

- retrieval and reranking
- prompt trimming and formatting
- provider generation
- citation validation
- the single citation-repair attempt
- adaptive retrieval or generation reruns
- cache selection

If a pipeline path cannot reach a coherent terminal boundary, the answer may
still be shown, but it is not labeled as grounded.

### EvidenceRun

An `EvidenceRun` represents one retrieval execution, not one answer. It records:

- run and parent-request identities
- query identity, a secret-scoped opaque fingerprint, or no query fingerprint
  where raw query storage is not permitted; never a raw unsalted query hash
- retrieval mode and stage
- source and authority scope
- retrieval, fusion, and reranking metadata
- ordered candidate metadata
- typed scores with an explicit score kind and scale
- lineage to source document and chunk
- start/end offsets with an explicit `offset_basis`
- timestamps and bounded timing summaries

Sensitive query, candidate, source, score, and lineage fields belong to the
governed evidence-run payload rather than immutable trace JSON. Secure purge can
reduce a retained run to an opaque run identity, stage, ordering, and redaction
status.

Scores are not forced into a shared `0..1` range. BM25, vector similarity,
distance, RRF, and reranker scores retain typed semantics.

Candidate retention is bounded. Full text snapshots are retained only for
evidence actually submitted; answer citations select among those submitted
snapshots. Non-submitted candidates retain only the metadata required to explain
ranking.

### PromptEvidenceSet

A `PromptEvidenceSet` records the exact evidence submitted at a provider request
boundary. Each entry contains:

- stable marker such as `[S1]`
- an opaque reference to the governed snapshot containing the exact submitted
  text after truncation, heading injection, formatting, and other
  transformations
- evidence-run reference and ordinal
- seal-time storage-policy class

The governed snapshot descriptor, not trace JSON, contains exact-byte and
normalized-comparison hashes, token and character counts, transformation
details, source lineage, seal-time authority decision, typed locator, and export
policy.

The contract says “submitted to model,” never “used by model.”

Marker numbering is stable within one prompt set. A new provider request that
changes the evidence creates a new prompt set. A rerun may reuse the same
prompt set only when its submitted evidence is byte-identical.

### AnswerAttempt

An `AnswerAttempt` records:

- attempt identity and kind, including initial, citation repair, or pipeline
  rerun
- associated prompt evidence set
- answer integrity reference and an opaque reference to governed attempt text
  when retained
- extracted marker and answer-span mappings
- structural validation results
- semantic verification results when available
- repair reasons and outcome

The selected attempt binds to the owning message body by fingerprint rather than
duplicating that body in trace JSON. The binding uses a secret-scoped body
fingerprint stored on the governed owner association, not a portable raw content
hash. Earlier attempt bodies use separate governed, bounded payload records so
repair behavior can be inspected and tested when policy permits. Purging an
attempt body does not rewrite the sealed attempt metadata. Existing pipeline
retry limits bound the number of attempts; citation repair adds at most one
attempt. Fetching additional evidence always creates a new retrieval run and
prompt set.

### EvidenceSnapshot

An `EvidenceSnapshot` is the exact submitted or cited text required to explain a
historical answer. It is distinct from:

- the current source item
- a search-result preview
- a longer source chunk not submitted to the provider
- an academic or bibliographic citation

Content-addressed deduplication is scoped by profile or tenant, authority,
confidentiality policy, and an opaque `revocation_scope_id` supplied by the
source policy. Independently revocable items have different revocation scopes;
workspace-wide material may share one only when workspace policy revokes it as
a unit. Exact-byte hashes support integrity; normalized hashes support
comparison. Both live in governed payload storage and are removed on purge.

Snapshot storage modes are:

- `embedded`
- `server_reference`
- `ephemeral`
- `redacted`

Embedding is allowed only when local storage does not weaken the source's
confidentiality policy. Otherwise the trace retains a governed reference or a
redacted record.

### Storage, completeness, and trust matrix

`completeness_at_seal` is immutable. Current payload access is mutable and does
not rewrite it.

| Storage and access state | Entry/trace effect | Persisted trust presentation | Snapshot action |
| --- | --- | --- | --- |
| Embedded and allowed at seal | Entry is `complete` | Determined by final-set reduction | Read exact snapshot |
| Validated durable server reference and allowed at seal | Entry is `complete` | Determined by final-set reduction | Resolve exact snapshot through the authoritative server |
| Ephemeral payload | Entry is `partial` after the live request ends | Legacy/partial after reduction | Available only while the governed live payload exists |
| Redacted at seal | Entry is `redacted` | Legacy/partial after reduction | Forbidden |
| Missing payload while other reliable final-set provenance remains | Entry is `partial` | Legacy/partial after reduction | Forbidden |
| Invalid authority, inconsistent references, or no trustworthy final prompt set | Whole trace is `unavailable` | Ungrounded | Forbidden |
| Durable server reference temporarily offline or authentication-required | Unchanged from seal | If seal was complete: Grounded with an availability warning; otherwise retain prior label | Disabled until access recovers |
| Embedded or referenced payload later revoked | Unchanged from seal | If seal was complete: Grounded with an evidence-revoked warning; otherwise retain prior label | Forbidden |

A valid marker can resolve structurally even when current snapshot access is
forbidden. The UI then shows the marker mapping and permitted opaque status, not
the source text or identity.

### Completeness reduction

Active answer trust is derived only from the selected `AnswerAttempt` and its
referenced final `PromptEvidenceSet`. Non-final attempts and their prompt sets
remain diagnostic history and do not downgrade the selected answer.

Each entry in the selected prompt set is classified at seal:

- `complete`: embedded exact payload, or an authoritative durable server
  reference that attests to the exact payload
- `partial`: ephemeral or missing payload where some reliable prompt provenance
  remains
- `redacted`: policy prohibited retaining the payload at seal
- `unavailable`: no trustworthy final prompt set, invalid authority, or
  internally inconsistent references

The trace reduction is deterministic:

1. Any `unavailable` final-set condition makes the trace `unavailable`.
2. Otherwise, any `redacted` entry makes the trace `redacted`.
3. Otherwise, any `partial` entry makes the trace `partial`.
4. Only an all-`complete` final prompt set makes the trace `complete`.

Fully **Grounded** and **Grounded with warnings** presentations require
`completeness_at_seal=complete`. A trace sealed as `partial` or `redacted`
remains **Legacy/partial**, even if every marker present in the answer happens
to cite one retained entry. Later offline, authentication, or revocation state
does not rewrite completeness at seal; it changes a complete trace to
**Grounded with warnings** at presentation time.

### Validation and trust

The trace keeps independent states for:

- `marker_valid`: the marker refers to submitted evidence
- `span_valid`: the marker maps to a syntactically valid answer span
- `claim_supported`: semantic support is `supported`, `unsupported`,
  `insufficient`, `unknown`, or `not_checked`
- `source_current`: a separately timestamped current-source observation

A structurally grounded answer may still contain unsupported claims. The UI
must not collapse these states into one “verified” badge.

Semantic trust in the sealed trace is the result available at the terminal
generation boundary. A future post-seal evaluator may store a separate
versioned assessment, but it cannot rewrite historical trace trust.

Academic and bibliographic citations remain separate typed records. They may
link to evidence where appropriate, but they are not rewritten into chunk
markers.

### Citation occurrence contract

Complete local and server traces use marker namespace `chatbook_s_v1` with the
literal grammar `[S<positive-decimal-ordinal>]`; zero, leading-zero, comma-grouped,
and unknown ordinals are invalid. Multiple citations for one claim are emitted
as adjacent complete markers such as `[S1][S3]`, optionally separated by
whitespace.

Each parsed `CitationOccurrence` records:

- attempt-local occurrence ordinal
- raw marker and marker namespace
- evidence ordinal or `null` for an unknown marker
- raw marker start and end offsets in the exact unrendered answer
- associated claim-span start and end offsets when deterministically available
- `offset_basis=unicode_codepoint_v1`
- structural validation state

Offsets are calculated against the exact stored Python/Unicode answer before
Markdown rendering; integrity fingerprints use its UTF-8 bytes. Repeated
markers produce distinct occurrences that reference the same evidence. Grouped
markers produce distinct occurrences sharing one claim span. Reordered markers
are valid when their ordinals exist. Markers inside code or escaped literals are
not occurrences. Repair recalculates occurrences against the repaired body.

Pinned legacy server
[`citations.py`](https://github.com/rmusser01/tldw_server/blob/d9c245ac14c40df855d1ab6cd19b3c137b16b47b/tldw_Server_API/app/core/RAG/rag_service/citations.py)
uses numeric markers such as `[1][3]`. Chatbook records those under
`legacy_numeric_v1`, keeps the answer unchanged, and exposes them only as
partial legacy provenance. A new server response is complete only when it
declares and supplies occurrence mappings for `chatbook_s_v1`. Chatbook
requests that marker namespace explicitly; other server clients may continue
requesting their own style.

## Pipeline lifecycle

The canonical lifecycle is:

```text
request
  -> EvidenceRun 1..N
  -> PromptEvidenceSet 1..N
  -> AnswerAttempt 1..N
  -> optional single citation repair
  -> optional bounded pipeline rerun
  -> terminal answer selection
  -> CitationTrace seal
  -> atomic message and provenance persistence
```

Authority is checked at retrieval and again when prompt evidence is assembled.
A source becoming unauthorized between those boundaries is excluded from the
prompt set.

Cached answers retain the original generation ID and sealed trace. A cache hit
gets a new request ID but does not pretend the answer was regenerated or the
source was freshly resolved.

Local generation owns local trace construction. For server generation, the
server owns authoritative trace construction when it returns a supported
`grounding_trace`. Chatbook does not reconstruct a complete trace from server
response fragments; it adapts those fragments into a clearly partial legacy
trace.

### Streaming and repair presentation

A streaming assistant response remains one `provisional` message until citation
validation and any repair finish. Inline markers may be visible during the
stream, but they are not interactive and the message is not labeled grounded.

When the initial stream ends:

- If validation passes, the initial attempt becomes selected and is sealed.
- If repair is required, the widget keeps the initial text visible with
  `Checking citations…`; the repair runs without appending a second assistant
  message.
- If repair succeeds, the same provisional widget replaces its body with the
  repaired result and shows `Citations repaired · View original attempt`. The
  replacement is explicit rather than a silent overwrite.
- If repair fails, the initial body becomes selected with citation warnings.
- If the user cancels repair, the initial body becomes selected with
  `Citation repair canceled` and warnings.
- If generation itself is canceled before a complete answer exists, no grounded
  trace is sealed.

The selected attempt and final visible body fingerprint are calculated only
after this state resolves. A recovered partial-stream draft may remain visible,
but it is ungrounded and cannot inherit an unfinished builder.

## Citation rendering and inspection

### Answer presentation

Answers retain inline `[S#]` markers and receive a compact source footer:

```text
Sources 3 cited, 2 additionally provided · Support not checked
```

The footer is present for grounded answers even when every marker is valid. It
communicates citation count, supplied-but-uncited evidence, and trust state
without placing multiple badges on every source row.

Citation rendering is Markdown-aware. Markers in fenced code, inline code,
escaped text, and literal examples are ignored. Linkification is a presentation
layer and does not rewrite the stored answer.

A keyboard-focusable “View sources” action always exists because inline
Markdown links are not guaranteed to be focusable in every terminal.

### CitationEvidenceInspector

`CitationEvidenceInspector` is a reusable content component, not a modal by
default:

- Console hosts it in the existing right Inspector rail.
- Library and Search use their existing result/detail area.
- Narrow terminals and artifact views without a rail use a full-screen
  inspector screen.

It contains two primary groups:

- **Cited**
- **Provided to model, not cited**

For `embedded` storage, the selected row shows the submitted snapshot
immediately. For `server_reference` storage, it shows a non-blocking loading or
unavailable state while the governed snapshot is resolved asynchronously.
Current-source resolution is a separate operation that starts as “not checked”
and runs lazily in a background worker. Network or source access never blocks
initial answer or inspector rendering.

Current observations are keyed by trace and evidence identity. Results from a
stale async request are discarded when the selection changes.

The inspector preserves a return target containing conversation, message,
trace, and evidence identities so closing or native navigation can restore the
user's location and focus.

Explicit copy operations are:

- Copy source reference
- Copy cited excerpt
- Copy bibliographic citation

### Current-source comparison

The historical submitted snapshot is always primary. Current resolution may
offer “Compare” only when a unique, policy-permitted current item is available.
Comparison distinguishes:

- content state: unchanged, changed, unknown
- location state: same, relocated, ambiguous, unknown
- availability: available, missing, offline, error
- permission: allowed, revoked, authentication required, unknown
- capability: inspect, open, compare, refresh, external navigation

Relocation requires a unique compatible source identity, surrounding-context
fingerprint, and confidence above the resolver's threshold. Ambiguous matches
are never shown as a definitive diff.

## Source locator and resolver contract

### SourceLocatorEnvelope

Native location data uses a versioned envelope:

- envelope schema version
- canonical source kind
- authority identity
- resolver payload version
- typed resolver-owned payload

The core trace does not accept free-form native metadata.

### Resolver registry

Resolvers are registered in a static allowlisted registry. Trace data cannot
name an arbitrary class, command, path handler, or URL handler. Each resolver
declares supported capabilities:

- inspect current
- open native
- open external
- jump to location
- compare
- refresh observation

The supported inventory is derived from retriever registries and versioned
contract fixtures, not maintained only as UI copy. At the reviewed baseline,
Chatbook's local semantic index emits media, notes, and conversations. The
pinned server enum additionally covers character cards, web content, prompts,
world books, dictionaries, SQL, kanban, and claims.

| Canonical source kind | Baseline producer | Default resolution contract |
| --- | --- | --- |
| `media_db` | Local and server | Open Library media item; jump to chunk, time, page, or section when lineage permits |
| `notes` | Local and server | Open current Notes item; file navigation requires a currently authorized file-backed note |
| `chat_history` | Local and server | Open conversation and message |
| `character_cards` | Server | Open current Character/Persona item when the configured server exposes an authorized item route |
| `kanban` | Server | Open authorized board/card when a durable item identity exists |
| `prompts` | Server | Open current Prompt item |
| `world_books` | Server | Open current world-book/lore item and entry |
| `dictionaries` | Server | Open current dictionary and entry |
| `web_content` | Server | Inspect the indexed web item; external URL open is a separate explicit capability |
| `claims` | Server-derived | Open the authorized parent media/chunk when lineage exists; otherwise snapshot-only |
| `sql` | Server-derived | Structured snapshot-only evidence; never replay SQL, open a database path, or claim a document destination |

“Supported” means the resolver can honestly represent the source, snapshot, and
capabilities. A native-open capability is required only when the producer
supplies a durable authorized item identity. Derived SQL and claims without
parent lineage must expose a clear snapshot-only state rather than inventing a
link.
An implementation-time inventory fixture records, per producer and source kind,
its identity fields, authority and tenant binding, locator version, and allowed
capabilities. A default-on release cannot omit any source kind emitted by its
enabled retriever registries.

No general third-party resolver plugin system is included.

### Governance

Storage and capabilities are separate decisions. A trace records a storage mode
and policy-derived capabilities:

- view snapshot
- view source identity
- resolve current
- open native
- open external
- export

Personal/local ordinary deletion preserves historical snapshots by default.
Governed workspace or server policy may revoke snapshot text. Explicit secure
purge removes or redacts derived snapshots according to the affected governance
boundary.

Snapshot revocation does not silently rewrite the assistant message, because
the generated answer is independently owned conversation content. A governance
policy that also requires derived answer removal must invoke an explicit secure
purge or quarantine operation and report that wider effect before execution.

After access loss, the UI reveals only an opaque evidence identity and permitted
status. It does not reveal snapshot text, source title, locator, hash, or other
restricted metadata.

Revocation and secure purge leave a durable, non-content
`redacted_tombstone`. The tombstone retains only opaque payload/origin identity,
opaque revocation scope, reason code, policy version, and timestamp. Text,
title, lineage, locator, hashes, and comparison fingerprints are nulled. Cache
hydration, import, and Sync replay consult tombstones before writing payloads,
so a previously purged origin payload cannot be resurrected. Tombstones remain
for the policy and sync-retention period even after all trace owners disappear.

Imported locators are inert until their schema and authority are validated and
the user explicitly rebinds or trusts the destination. Rebinding creates a new
current observation and never rewrites the imported historical trace.

### Files and URLs

Local file locators store a configured source-root ID and relative path, not an
unrestricted absolute path. Each open operation canonicalizes the current path
and rechecks existence, root containment, and symlink behavior.

The client never refetches an arbitrary URL merely because it appears in a
trace. External browser navigation is an explicit capability and action.
Current URL resolution uses a trusted indexed item or a server endpoint that
enforces its own SSRF and egress policy.

Server authorization is authoritative. Client-side visibility is not a
security boundary. Temporary offline or authentication failure remains
distinct from revocation.

Resolvers run asynchronously, never auto-open a destination, sanitize logs, and
discard stale responses.

## Persistence and migration

### Storage location and logical tables

Local answer provenance is stored in the same SQLite ownership boundary as chat
messages so the final message and provenance can commit atomically. The logical
schema contains:

- `rag_evidence_runs`: stable retrieval-run identity and stage metadata with
  governed bounded candidate payloads
- `rag_citation_traces`: indexed trace identity, lifecycle, origin,
  completeness at seal, final attempt, validation summary, timestamps, and
  bounded aggregate JSON
- `rag_evidence_snapshots`: governed authority-scoped snapshot text, hashes,
  source identity, lineage, locator, and transformation metadata
- `rag_answer_attempt_payloads`: governed bounded non-final attempt bodies
- `rag_trace_evidence_refs`: ordered trace/prompt evidence references to
  snapshots or governed references
- `rag_source_observations`: the latest bounded current-source observation per
  trace evidence item and resolver
- `rag_payload_tombstones`: durable non-content revocation and secure-purge
  barriers
- trace owner associations for messages and saved artifacts

Owner associations use real foreign keys when the owner shares the database.
Cross-database artifact ownership uses a validated stable reference and an
explicit cleanup path rather than pretending SQLite can enforce a cross-file
foreign key.

Prompt-set and answer-attempt metadata remain bounded JSON in the trace for the
first version. Raw submitted text and non-final attempt bodies live only in
their governed payload tables. Fields needed for lookup, lifecycle, governance,
migration, or operational reporting are projected into indexed columns.

Citation snapshots and trace JSON are explicitly excluded from FTS, Library
indexing, and RAG ingestion.

Current-source observations are not part of the immutable trace. The observation
store replaces the latest value for a trace/evidence/resolver key and records
`observed_at`; it does not retain an unbounded polling history. Export uses this
last-known observation unless the user explicitly refreshes first.

### Identity and transactions

All Chatbook-owned primary IDs are opaque random 128-bit values. External
server IDs are bounded validated opaque strings. Their namespace and
idempotency rules are:

- A local trace key is `(profile_id, trace_id)`. The builder allocates
  `trace_id` once and retains it through uncertain persistence retries.
- A server trace key is `(connection_authority_id, tenant_id, server_trace_id,
  wire_schema_version)`.
- An imported trace receives a new local `trace_id`, retains external authority
  and external trace ID as inert origin metadata, and is deduplicated by
  `(profile_id, secret_scoped_import_package_fingerprint, external_trace_id)`.
  Import never rewrites its origin into `local` or `server`.
- A snapshot deduplication key is `(profile_or_tenant, authority_id,
  confidentiality_policy_id, revocation_scope_id, exact_payload_hash)`.
- An owner link is unique by owner kind, owner ID, owner revision, and trace ID.
- Cache reuse creates a new message owner link to the original trace; it does
  not clone or rename the trace.
- Sync retries use the sync operation ID plus canonical trace namespace.
  Conflict forks create new owner links, not new provenance identities.

`CitationTraceBuilder` and every retrieval/attempt payload remain in memory
until a terminal trace is sealed. One local transaction writes or idempotently
reuses all of the following:

- final message
- trace summary and aggregate metadata
- evidence-run rows and governed candidate payloads
- snapshot and non-final attempt payload rows
- trace-to-payload references
- applicable tombstone checks
- message owner association and body fingerprint

No canonical incomplete retrieval rows are written before sealing, so the
design does not require an orphan-run subsystem. If the transaction outcome is
uncertain, retrying with the same namespaced IDs returns the committed aggregate
or completes it without duplicates. Failure leaves the streamed answer visible
but ungrounded and offers one idempotent message-plus-provenance retry when the
builder is still available.

### Message binding and mutation

A trace is actively grounded only while the secret-scoped owning-message body
fingerprint matches the selected answer-attempt integrity reference.

- Editing or replacing the assistant body marks the association
  `body_mismatch`; the immutable trace remains available as historical
  provenance but the edited message loses its grounded presentation.
- Import activates an owner association only after schema, authority, and body
  fingerprint validation. Otherwise the trace remains inert or historical.
- A Sync v2 overwrite without matching provenance invalidates the active
  association rather than reusing the old trace.
- A Sync v2 duplicate may retain the trace only when its body fingerprint
  matches.
- Conflict-resolution forks receive distinct message ownership associations.
- A message tombstone tombstones its owner association; the trace remains while
  another owner or retention policy requires it.

No operation may attach a trace from one generated answer to materially
different message text.

### Cross-database artifact ownership

When an artifact and trace share a database, their owner link uses the same
transaction and a real foreign key. When they do not, artifact save/delete uses
a small durable outbox handshake:

1. The artifact store writes the artifact plus a pending provenance-link or
   unlink operation.
2. The trace store idempotently adds or tombstones an artifact owner lease.
3. The artifact store marks the outbox operation complete.

A startup/background reconciler retries pending operations. Garbage collection
treats pending links and live leases as owners and does not collect a trace
during an unresolved unlink. Portable exports remain independent files and do
not create owner leases.

### Retention and deletion

Trace owner associations drive retention:

- Soft-deleting a conversation retains provenance under the existing recovery
  policy.
- Deleting one owner removes only that association.
- Saved artifacts keep an immutable trace alive independently of their source
  conversation.
- Portable exports do not count as database owners.
- Secure purge applies the source or workspace governance policy across the
  affected references.
- An unreferenced snapshot is collected only after its retention window and
  relevant sync tombstones permit collection.
- Revoked referenced payloads retain non-content tombstones; ordinary
  unreferenced payload collection may delete the payload entirely only when no
  revocation barrier is required.

### Legacy migration

Readers accept current `EvidenceBundle`, `CitationRef`, `citation_validation`,
and `chat_rag_context.json` sidecar data. They synthesize a partial,
`legacy_inferred` trace and never upgrade it to complete provenance.

Legacy source IDs may be mapped to a current typed locator only by the static
resolver registry after a fresh authority lookup. Free-form legacy paths, URLs,
and content references are inert data: they cannot open, refresh, compare, or
export as native locators. If no safe current mapping exists, the legacy trace
is snapshot-only or unavailable. Legacy migration tests apply the same path,
URL, tenant, and rebinding attacks used for imported traces.

New writes use only the database trace contract. Migration is transactional and
backgrounded per conversation, with a migration journal and retry state.
Legacy data remains readable until cutover succeeds. The sidecar is not
automatically deleted.

If an older client modifies legacy data after cutover, the new client records a
divergence and does not silently merge two provenance histories.

## Sync v2 and server transport

Citation provenance is a message-owned adjunct, not a new independent sync
authority. Sync follows
[`ADR-008`](../../../backlog/decisions/008-sync-v2-client-m1-contract-alignment.md):
the server's advertised contract remains canonical.

A peer must advertise support for the trace schema and snapshot transport mode
before provenance sync is enabled. Until then:

- the message may sync under the existing contract
- its remote provenance is marked absent or partial
- the local authoritative trace is not deleted or overwritten
- server references remain references

Snapshot text is included only when source policy and the negotiated sync
contract allow it. Each field follows the existing tenant and transport
security boundary; citation work does not claim client-private encryption where
the active Sync v2 contract provides only server-trusted transport.

Delivery is idempotent. Unknown trace fields are preserved only where the peer
contract promises round-trip preservation. A peer that cannot safely preserve
the trace must not become authoritative for it.

### `grounding_trace/v1` wire ownership

For RAG answer APIs, `tldw_server` owns the canonical OpenAPI/JSON wire schema
and producer semantics for `grounding_trace/v1`. Chatbook owns its internal
`CitationTrace` model and a bounded adapter for that wire schema; the two repos
do not share or import Python implementation code.

The server repository publishes:

- versioned schema and compatibility rules
- valid, partial, malformed, authority-mismatch, grouped-marker, and redaction
  fixtures
- terminal-boundary trace construction
- tenant/authority enforcement and snapshot policy
- preservation of existing document and citation response fields

Chatbook pins those fixtures in consumer contract tests. Schema changes are
backwards-compatible additions within v1 or require a new advertised version.
The Chatbook consumer may land disabled before the producer, but the complete
server path and default-on release require a compatible server capability.

Server production is a separate `tldw_server` task and PR. Chatbook validation,
mapping, and UX are separate Chatbook tasks. Provenance Sync v2 likewise
requires a separately advertised server capability; a client-only task cannot
declare that path complete.

Existing document and citation arrays remain supported. Cached server responses
retain their original trace and generation identity.

An authenticated server trace is accepted only after:

- schema version and total/count/depth/text limits pass
- marker, snapshot, and attempt references are internally consistent
- tenant, workspace, and authority identities match the authenticated response
  context
- locator kinds and payload versions are allowlisted
- server references bind to the same authenticated authority

A valid supported `grounding_trace` is authoritative. Existing citation arrays
may be compared for sanitized diagnostics but are not merged into it. If the
trace version is unsupported, Chatbook ignores it and adapts safe legacy arrays
into a partial trace. If a nominally supported trace is malformed or
authority-mismatched, Chatbook records `trace_invalid` and may use safe legacy
arrays only as explicitly partial provenance. It never silently promotes that
fallback to complete grounding or executes locators from the rejected trace.

Server-provided embedded payloads are persisted locally only when Chatbook's
current storage policy permits an equivalent confidentiality boundary.
Otherwise the adapter retains a validated durable server reference or reduces
the local view to partial/redacted according to the completeness rules; server
metadata cannot weaken local storage policy.

## Export and import

Default export contains:

- answer text with markers
- human-readable Sources section
- cited snapshots when policy allows
- machine-readable trace manifest
- last-known validation and freshness observations with timestamps
- explicit redaction reasons

An explicit option adds all evidence submitted to the model, including
uncited context.

Export never resolves or refreshes a source automatically. “Refresh before
export” is a separate, visible user action. The export manifest records:

- schema version
- trace and evidence counts
- embedded, referenced, and redacted counts
- export policy selection
- export timestamp
- completeness

Exports exclude credentials, system prompts, unrestricted absolute paths, raw
observability data, restricted identities, and evidence prohibited by policy.

Portable packages use an export-local reference map. They never expose
database primary keys as executable identities. The manifest preserves inert
origin namespace and external IDs, maps them to package-local evidence IDs, and
integrity-checks included payloads. Import allocates new local IDs, validates
the full reference graph before any owner link is activated, and uses the
package fingerprint rules above to make retry idempotent.

Imported traces use `origin=imported`, receive hard limits for total bytes,
nesting depth, text size, locator size, and record count, and have no native
authority. Unsupported or invalid schemas fail closed. Imported locators remain
inert until explicit validation and rebinding.

## Failure handling

- Invalid or missing markers receive one automatic citation-repair attempt.
- Remaining invalid markers stay visible as warnings; they are not dropped or
  treated as trusted citations.
- A changed or deleted personal source retains its historical snapshot and
  reports the current state.
- Governed revocation hides or redacts snapshot and identity fields according
  to policy.
- Offline, authentication-required, missing, revoked, and ambiguous states use
  distinct copy and capabilities.
- Unsupported server traces become partial legacy traces.
- Unknown, oversized, or malformed stored/imported schemas fail closed and do
  not receive partial native resolution.
- Trace persistence failure cannot leave a false grounded badge.

## Testing and evaluation

### Contract and persistence tests

Unit and property-based tests cover:

- trace and snapshot round-trip serialization
- schema upgrades and legacy adapters
- stable marker assignment
- canonical and legacy marker grammars; repeated, grouped, reordered, unknown,
  code-literal, and Unicode-offset occurrences
- exact prompt-boundary text and hashes
- typed scores and offset bases
- one-repair enforcement
- deterministic mixed-storage completeness reduction using only the selected
  attempt
- uncertain transaction retry across every aggregate row
- deduplication, retention, revocation-scope isolation, purge tombstones, and
  cache/import/sync anti-resurrection
- cross-database artifact outbox and garbage-collection races
- migration restart and post-cutover divergence
- message edits, body-fingerprint mismatch, import rebinding, sync replay, conflict
  overwrite/fork/duplicate behavior, tombstones, and incompatible peers
- hostile imported schemas and locator payloads
- exclusion from FTS and RAG indexing

### End-to-end paths

The integration path is:

```text
retrieval -> reranking -> prompt assembly -> generation
-> validation/repair/rerun -> trace sealing -> persistence
-> reload -> inspector -> source resolution -> export/import
```

Golden paths cover local and server RAG, streaming and non-streaming,
cache hits, adaptive reruns, citation repair, restart, saved artifacts, and
policy-permitted export/import.

Every source resolver passes a conformance suite for inspect, open, permission,
unavailable, and export behavior. Deeper changed, moved, deleted, revoked,
offline, and attack scenarios run once per resolver capability class rather
than as an unmaintainable full cross-product.

The versioned source-inventory fixture proves every enabled local and server
retriever source kind has an identity, authority, tenant, locator, and
capability classification. Server contract fixtures cover numeric legacy
arrays and every required `grounding_trace/v1` fixture published by the server.

### UI tests

Mounted Textual tests and real-terminal QA verify:

- wide Console inspector rail and narrow full-screen fallback
- Markdown-aware marker activation
- keyboard equivalence and focus restoration
- immediate snapshot rendering
- non-blocking current resolution
- stale async result rejection
- provisional stream, visible repair transition, successful body replacement,
  cancellation, failure, and recovered-draft behavior
- honest status and capability copy
- terminal escape and Markdown/link-injection handling
- copy and export policy enforcement

### Security tests

Security coverage includes:

- path traversal and symlink swaps
- unsafe URL schemes and SSRF attempts
- malicious imported locators
- cross-tenant snapshot references
- revoked metadata and hash leakage
- terminal control sequences
- clipboard and external-open policy
- sanitized logs and diagnostic exports

### RAG evaluation

Evaluation begins with representative end-to-end traces, not a predefined
failure taxonomy. The initial target is approximately 100 traces, stopping when
the last 20 reveal no new failure categories. CI uses curated and synthetic
fixtures. Real user traces remain local or require explicit opt-in and
sanitization. Evaluation corpora are never production-indexed.

This saturation review discovers failure categories; it is not itself a
release gate. Before default-on release, an evaluation task publishes a
versioned fixed dataset manifest, human labels, exact evaluation commands,
metric implementations, and deterministic pass/fail thresholds. Synthetic
examples supplement but do not replace the fixed human-reviewed set.

Retrieval and generation are evaluated separately:

- first-pass retrieval: Recall@k
- reranking: MRR, NDCG@k, or Precision@k
- synthesis: multi-hop Recall@k
- citation structure: marker resolution, span validity, and snapshot fidelity
- citation correctness: support for the associated claim
- citation completeness: coverage of claims requiring evidence
- attribution correctness: selection of the appropriate supporting evidence
- generation: answer faithfulness and relevance

Deterministic properties use code-based checks. A semantic evaluator becomes a
release gate only after calibration against human labels with recorded
false-positive and false-negative acceptance thresholds.

Citation repair is evaluated as a separate attempt and must not change evidence
numbering, introduce unsupported claims, or erase the original attempt.

### Performance qualification

A prerequisite benchmark task runs before storage-contract implementation. It
publishes a versioned corpus/fixture manifest, supported hardware and provider
envelope, exact measurement commands, sample-count and warmup rules, and numeric
pass/fail thresholds for:

- first-token latency regression
- post-stream trace finalization
- inspector opening from stored data
- maximum trace and snapshot size
- database growth per grounded answer
- migration throughput and interruption recovery

The storage and pipeline plans must cite that benchmark artifact. A separate
late qualification task enforces the same thresholds against the completed
feature. Source refresh and external network latency are measured separately
and never block answer or inspector rendering.

## Rollout

The architecture lands through dependency-ordered, internally guarded stages:

1. Versioned benchmark baseline and numeric budgets.
2. Versioned contracts, governance, security boundaries, and storage caps.
3. Database persistence, migration, retention, and recovery.
4. End-to-end local RAG trace capture and internal inspector.
5. Resolver conformance for every enabled source family.
6. Console, Library/Search, artifacts, and policy-aware export/import.
7. Server-owned `grounding_trace/v1` producer plus Chatbook consumer
   compatibility, released in that dependency order.
8. Negotiated Sync v2 server capability and client transport, privacy
   qualification, performance qualification, and RAG evaluation.
9. Default-on release after the complete source inventory passes.

The schema is not down-migrated during rollback. A recovery switch may stop new
trace writes while supported stored traces remain readable. Unsupported schemas
fail closed. Rollback never deletes provenance or the legacy sidecar.

Operational metrics contain counts, sizes, timings, and status codes only. They
exclude queries, answers, source titles, snapshots, locators, and resolver
payloads.

## User-facing trust states

- **Grounded:** complete sealed trace, exact submitted evidence, and valid
  structural mappings.
- **Grounded with warnings:** a complete-at-seal canonical prompt record with
  unresolved markers, failed repair, evidence revoked after sealing, or
  unavailable current sources.
- **Legacy/partial:** useful older evidence metadata without an exact canonical
  prompt-boundary trace.
- **Ungrounded:** no reliable provenance.

“Grounded” does not imply that every claim is semantically supported. Claim
support is shown separately when checked.

## Success criteria

The feature is complete when:

- Every successful canonical RAG branch seals and atomically persists its trace.
- Mixed prompt storage modes reduce to the same trust state on every reader,
  based only on the selected attempt.
- Every valid citation with an embedded, policy-allowed payload resolves to the
  exact submitted snapshot after restart; validated server references resolve
  it when the authoritative server is reachable and authorization still
  permits. Other storage modes produce the exact matrix-defined warning rather
  than a false resolution claim.
- Users can distinguish cited evidence from additional submitted context.
- Users can inspect snapshots and navigate to policy-permitted current sources
  across every supported source family.
- Changed, moved, missing, offline, revoked, and partial states remain honest
  and distinct.
- One bounded repair occurs for malformed citation output and unresolved
  markers remain visible afterward.
- Saved artifacts and policy-permitted exports retain provenance.
- Persistence retries, cache reuse, imports, artifact links, and Sync conflict
  forks cannot duplicate, misattach, resurrect, or prematurely collect a trace
  or governed payload.
- Legacy conversations and old server responses remain readable without false
  complete-grounding claims.
- No restricted snapshot, identity, locator, or hash leaks through UI, sync,
  logs, clipboard, or export.
- Migration is restart-safe and does not delete legacy data.
- The resolver conformance, security, RAG evaluation, and recorded performance
  gates pass.

## Implementation workstreams and task-decomposition rules

This is an epic architecture. Before code changes, create or select a Backlog
epic. The following are workstreams, not task files. Each bullet below is the
maximum scope of one atomic child task; split further when its acceptance
criteria cannot be completed in one PR:

1. Publish the prerequisite performance/storage benchmark corpus, commands,
   environment, and numeric budgets.
2. Define trace, snapshot, attempt-payload, trust, and serialization contracts
   with pure compatibility adapters.
3. Define locator envelopes, capability policy, and the static resolver
   registry without implementing source families.
4. Add the database schema and sealed-aggregate repository transaction.
5. Add namespaced identity, idempotent retry, and cache-owner reuse.
6. Add payload revocation scopes, tombstones, retention, and garbage
   collection.
7. Add bounded current-source observation storage.
8. Add cross-database artifact owner outbox/reconciliation.
9. Add legacy sidecar migration, journaling, divergence detection, and
   fail-closed legacy locator handling.
10. Capture local retrieval runs and exact prompt evidence sets.
11. Capture answer attempts and terminal trace sealing.
12. Add provisional streaming and the single visible citation-repair
    transition.
13. Add canonical occurrence parsing and legacy numeric marker mapping.
14. Validate and adapt current server citation arrays into partial traces.
15. In `tldw_server`, publish `grounding_trace/v1` schema and fixtures.
16. In `tldw_server`, construct and authorize the terminal server trace while
    preserving existing response fields.
17. In Chatbook, add bounded validation and mapping for
    `grounding_trace/v1`.
18. Add Markdown-aware answer markers and the compact Sources footer.
19. Add the read-only shared snapshot inspector and wide/narrow hosts.
20. Add resolver navigation infrastructure, versioned source inventory, and
    return-state handling.
21. Add one resolver-family child task per media, notes, conversations,
    characters, kanban, prompts, world books, dictionaries, web content,
    claims, and SQL, unless two families demonstrably share the same existing
    identity, authority, and opener contract.
22. Add saved-artifact ownership carry-through.
23. Add policy-aware human and machine export.
24. Add bounded inert import, package identity mapping, and explicit rebinding.
25. In the server repository, advertise provenance Sync v2 capability and
    contract fixtures.
26. In Chatbook, add negotiated Sync v2 trace transport and conflict
    lifecycle.
27. Add resolver conformance and actual-use UI QA.
28. Add security qualification.
29. Enforce the prerequisite performance/storage budgets against the completed
    feature.
30. Publish the fixed RAG evaluation dataset, human labels, commands, and
    deterministic retrieval/generation gates.

Do not combine export, import, artifact ownership, and sync into one task. Do not
combine all resolver families into one task. Do not combine a server producer
change with a Chatbook consumer change. Security, performance, and RAG quality
qualification have distinct acceptance criteria and remain distinct closeout
tasks.

Cross-repository work uses separate Backlog tasks and plans in Chatbook and
`tldw_server`, linked by the concrete wire-schema version and fixtures after
those tasks exist.

Each implementation plan must repeat:

```text
ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Implements the canonical provenance, persistence, sync, resolver, or
security decision.
```
