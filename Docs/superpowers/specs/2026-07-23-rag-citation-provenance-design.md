# RAG Citation Provenance and Source Inspection Design

Date: 2026-07-23
Status: Approved design; independent spec review passed; pending user document
review
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

The immutable trace stores identities, hashes, lengths, validation results, and
opaque references to governed payloads. It does not duplicate submitted chunk
text or non-final answer-attempt text inside aggregate JSON. A hydrated trace
view may resolve those payload references for an authorized caller, but payload
redaction or secure purge leaves the sealed metadata and opaque reference intact.
Current access, redaction, and observation state are computed separately from
the seal-time record.

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
- exact-byte and normalized-comparison hashes
- token and character counts
- evidence-run and source lineage
- authority decision at submission time
- typed source locator
- storage and export policy

The contract says “submitted to model,” never “used by model.”

Marker numbering is stable within one prompt set. A new provider request that
changes the evidence creates a new prompt set. A rerun may reuse the same
prompt set only when its submitted evidence is byte-identical.

### AnswerAttempt

An `AnswerAttempt` records:

- attempt identity and kind, including initial, citation repair, or pipeline
  rerun
- associated prompt evidence set
- answer hash and an opaque reference to governed attempt text when retained
- extracted marker and answer-span mappings
- structural validation results
- semantic verification results when available
- repair reasons and outcome

The selected attempt binds to the owning message body by hash rather than
duplicating that body in trace JSON. Earlier attempt bodies use separate
governed, bounded payload records so repair behavior can be inspected and tested
when policy permits. Purging an attempt body does not rewrite the sealed attempt
metadata. Existing pipeline retry limits bound the number of attempts; citation
repair adds at most one attempt. Fetching additional evidence always creates a
new retrieval run and prompt set.

### EvidenceSnapshot

An `EvidenceSnapshot` is the exact submitted or cited text required to explain a
historical answer. It is distinct from:

- the current source item
- a search-result preview
- a longer source chunk not submitted to the provider
- an academic or bibliographic citation

Content-addressed deduplication is scoped by profile or tenant, authority,
confidentiality policy, and source-governance boundary. Exact-byte hashes
support integrity; normalized hashes support comparison. Hashes are internal
and are not revealed after access loss.

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

| Storage and access state | Completeness at seal | Persisted trust presentation | Snapshot action |
| --- | --- | --- | --- |
| Embedded and currently allowed | `complete` when every submitted entry is present | Grounded, or grounded with validation warnings | Read exact snapshot |
| Validated durable server reference and currently allowed | `complete` when the authoritative server attests to every submitted entry | Grounded, or grounded with validation warnings | Resolve exact snapshot through the authoritative server |
| Durable server reference temporarily offline or authentication-required | Unchanged from seal | Grounded with an availability warning | Disabled until access recovers |
| Embedded or referenced payload later revoked | Unchanged from seal | Grounded with an explicit evidence-revoked warning | Forbidden |
| Ephemeral payload | `partial` after the live request ends | Legacy/partial; never persisted as fully grounded | Available only while the governed live payload exists |
| Redacted at seal | `redacted` | Legacy/partial or ungrounded according to remaining evidence; never fully grounded | Forbidden |
| Missing, malformed, or authority-mismatched payload | `partial` or `unavailable` | Legacy/partial or ungrounded | Forbidden |

A valid marker can resolve structurally even when current snapshot access is
forbidden. The UI then shows the marker mapping and permitted opaque status, not
the source text or identity.

### Validation and trust

The trace keeps independent states for:

- `marker_valid`: the marker refers to submitted evidence
- `span_valid`: the marker maps to a syntactically valid answer span
- `claim_supported`: semantic support is `supported`, `unsupported`,
  `insufficient`, `unknown`, or `not_checked`
- `source_current`: a separately timestamped current-source observation

A structurally grounded answer may still contain unsupported claims. The UI
must not collapse these states into one “verified” badge.

Academic and bibliographic citations remain separate typed records. They may
link to evidence where appropriate, but they are not rewritten into chunk
markers.

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

The first complete release covers every currently indexable source family:

- media
- notes
- conversations
- characters
- kanban items
- prompts
- world books
- dictionaries
- indexed URLs

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

- `rag_evidence_runs`: immutable retrieval-run records and bounded candidate
  metadata
- `rag_citation_traces`: indexed trace identity, lifecycle, origin,
  completeness at seal, final attempt, validation summary, timestamps, and
  bounded aggregate JSON
- `rag_evidence_snapshots`: authority-scoped snapshot content and hashes
- `rag_answer_attempt_payloads`: governed bounded non-final attempt bodies
- `rag_trace_evidence_refs`: ordered trace/prompt evidence references to
  snapshots or governed references
- `rag_source_observations`: the latest bounded current-source observation per
  trace evidence item and resolver
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

IDs are immutable and writes are idempotent. Snapshot identity includes its
deduplication scope and exact hash. The final local message, sealed trace,
snapshot references, and message-owner association commit in one transaction.
The owner association records the bound message revision and body hash.

Incomplete retrieval runs receive bounded orphan retention and cleanup.
Persistence failure leaves the answer visible but removes the grounded
presentation state and offers provenance retry when safe.

### Message binding and mutation

A trace is actively grounded only while the owning message body hash matches the
selected answer-attempt hash.

- Editing or replacing the assistant body marks the association
  `body_mismatch`; the immutable trace remains available as historical
  provenance but the edited message loses its grounded presentation.
- Import activates an owner association only after schema, authority, and body
  hash validation. Otherwise the trace remains inert or historical.
- A Sync v2 overwrite without matching provenance invalidates the active
  association rather than reusing the old trace.
- A Sync v2 duplicate may retain the trace only when its body hash matches.
- Conflict-resolution forks receive distinct message ownership associations.
- A message tombstone tombstones its owner association; the trace remains while
  another owner or retention policy requires it.

No operation may attach a trace from one generated answer to materially
different message text.

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

For RAG answer APIs, `grounding_trace` is an optional versioned response field.
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
- exact prompt-boundary text and hashes
- typed scores and offset bases
- one-repair enforcement
- transaction interruption and orphan cleanup
- deduplication, retention, revocation, and purge
- migration restart and post-cutover divergence
- message edits, body-hash mismatch, import rebinding, sync replay, conflict
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

Server contract fixtures cover existing citation arrays and the optional
canonical trace.

### UI tests

Mounted Textual tests and real-terminal QA verify:

- wide Console inspector rail and narrow full-screen fallback
- Markdown-aware marker activation
- keyboard equivalence and focus restoration
- immediate snapshot rendering
- non-blocking current resolution
- stale async result rejection
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

The implementation plan begins with a reproducible baseline and records numeric
thresholds for:

- first-token latency regression
- post-stream trace finalization
- inspector opening from stored data
- maximum trace and snapshot size
- database growth per grounded answer
- migration throughput and interruption recovery

Those recorded thresholds become release gates. Source refresh and external
network latency are measured separately and never block answer or inspector
rendering.

## Rollout

The architecture lands through dependency-ordered, internally guarded stages:

1. Versioned contracts, governance, security boundaries, and storage caps.
2. Database persistence, migration, retention, and recovery.
3. End-to-end local RAG trace capture and internal inspector.
4. Resolver conformance for every supported source family.
5. Console, Library/Search, artifacts, and policy-aware export/import.
6. Optional server trace contract and legacy response compatibility.
7. Negotiated Sync v2 transport, privacy qualification, performance
   qualification, and RAG evaluation.
8. Default-on release after the complete source matrix passes.

The schema is not down-migrated during rollback. A recovery switch may stop new
trace writes while supported stored traces remain readable. Unsupported schemas
fail closed. Rollback never deletes provenance or the legacy sidecar.

Operational metrics contain counts, sizes, timings, and status codes only. They
exclude queries, answers, source titles, snapshots, locators, and resolver
payloads.

## User-facing trust states

- **Grounded:** complete sealed trace, exact submitted evidence, and valid
  structural mappings.
- **Grounded with warnings:** a sealed canonical prompt record with unresolved
  markers, failed repair, evidence revoked after sealing, or unavailable
  current sources.
- **Legacy/partial:** useful older evidence metadata without an exact canonical
  prompt-boundary trace.
- **Ungrounded:** no reliable provenance.

“Grounded” does not imply that every claim is semantically supported. Claim
support is shown separately when checked.

## Success criteria

The feature is complete when:

- Every successful canonical RAG branch seals and atomically persists its trace.
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

1. Define trace, snapshot, attempt-payload, trust, and serialization contracts
   with pure compatibility adapters.
2. Define locator envelopes, capability policy, and the static resolver
   registry without implementing source families.
3. Add database schema, repositories, atomic message ownership, and bounded
   observation storage.
4. Add legacy sidecar migration, journaling, divergence detection, and
   fail-closed legacy locator handling.
5. Capture local retrieval runs and exact prompt evidence sets.
6. Capture answer attempts, one citation repair, terminal sealing, and cache
   identity.
7. Validate and adapt current server citation arrays into partial traces.
8. Add bounded validation for the optional server `grounding_trace`.
9. Add Markdown-aware answer markers and the compact Sources footer.
10. Add the read-only shared snapshot inspector and wide/narrow hosts.
11. Add resolver navigation infrastructure and return-state handling.
12. Add one resolver-family child task per media, notes, conversations,
    characters, kanban, prompts, world books, dictionaries, and indexed URLs,
    unless two families demonstrably share the same existing identity,
    authority, and opener contract.
13. Add saved-artifact ownership carry-through.
14. Add policy-aware human and machine export.
15. Add bounded inert import and explicit rebinding.
16. Add negotiated Sync v2 trace transport and conflict lifecycle.
17. Add resolver conformance and actual-use UI QA.
18. Add security qualification.
19. Establish and enforce performance/storage budgets.
20. Establish the RAG error-analysis dataset and retrieval/generation
    evaluation gates.

Do not combine export, import, artifact ownership, and sync into one task. Do not
combine all resolver families into one task. Security, performance, and RAG
quality qualification have distinct acceptance criteria and remain distinct
closeout tasks.

Each implementation plan must repeat:

```text
ADR required: yes
ADR path: backlog/decisions/024-rag-citation-provenance-and-source-resolution.md
Reason: Implements the canonical provenance, persistence, sync, resolver, or
security decision.
```
