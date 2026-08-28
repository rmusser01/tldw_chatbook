# Console Reference-Backed Semantic Trace Ledger Design

**Date:** 2026-08-28

**Status:** Amended after final adversarial review; awaiting owner re-approval

**Originating task:** [TASK-23026](../../../backlog/tasks/task-23026%20-%20Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md), completed by the now-superseded bounded-retention implementation

**Decision:** [ADR-097](../../../backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md)

**Supersedes:** [ADR-096](../../../backlog/decisions/096-console-safe-capture-retention.md) and its bounded-excerpt design

**Related follow-up:** [TASK-23112](../../../backlog/tasks/task-23112%20-%20Add-lossless-chunk-row-encoding-for-streamed-trace-events.md)

## Summary

Console exchange capture currently serializes and compresses the complete accumulated
`messages_payload` for every provider call. A production-shaped 200-turn conversation retained
15.40 MB because each later exchange copied all earlier messages again. Capture is enabled by
default, soft deletion retains the blobs, and no automatic reclamation path exists.

The fix is not a smaller repeated transcript. The saved conversation becomes the source of truth
for ordinary message content. A new local-only, append-only semantic trace ledger records durable
references to immutable message revisions, stores provider-only semantic material once in a
content-addressed artifact store, and gives each provider call a boundary into that ledger plus a
reference to its effective request header. Reconstructing a call folds the ledger through that
boundary. No call stores a list or blob proportional to the conversation's age.

Edits append semantic replacements without rewriting historical calls. Forks share an immutable
trace prefix and append their own suffix instead of copying trace history. Hard deletion
materializes any still-needed live message revision once before its source row disappears. Safe
and Full become disclosure profiles over the same stored trace; mandatory credential filtering
and optional irreversible PII masking remain capture-time policies.

Existing capture blobs are normalized automatically in bounded, resumable background batches.
Ordinary rows become message-revision references, provider-only rows become deduplicated
artifacts, and a legacy blob is deleted only after the normalized projection reproduces its
sanitized legacy meaning. Logical reclamation and physical SQLite compaction are reported
separately.

This design adapts DeepSeek Harness's useful invariant—model-visible content must be durably
explainable—and its append-only surface projection, changed-only request headers, stable sequence
boundaries, and canonical-log/telemetry separation. It deliberately does not copy DeepSeek's
fork seed, persist ordinary message bodies a second time, retain token chunks in this task, or
claim literal provider HTTP replay.

## Problem and current ownership

`ConsoleProviderGateway` currently builds an `ExchangeCapture` around every provider call.
`build_request_capture()` allowlists selected `chat_api_call()` kwargs, including the complete
`messages_payload`, stubs large binary-like strings, records `omitted_keys`, and places the
compressed JSON blob in local-only `message_exchanges`. Rows are owned by the eventual assistant
message and keyed by `(message_id, run_tag, seq)`.

That ownership has five defects:

1. Each call copies the complete conversation accumulated so far, producing quadratic cumulative
   storage.
2. A provider run may contain retries, tool-loop calls, failures, or abandoned calls before an
   assistant message exists, so an assistant message is not the natural call owner.
3. Editing a message destroys the exact body used by an earlier call unless the capture copied it.
4. Forking either loses source trace context or requires another physical copy of it.
5. The Safe/Full storage distinction encourages either incomplete diagnostics or repeated
   sensitive content, while `omitted_keys` currently provides only a narrow top-level inventory.

The existing `PreparedConsoleRequest` is the correct provider-neutral semantic assembly seam. It
already separates system framing, memory, mandatory context, compactable history, active request,
and tools. It does not yet carry durable source provenance, and the gateway currently flattens it
to provider kwargs before capture. The new design adds capture-only provenance descriptors beside
those semantic values. Descriptors never enter a provider request and never grant tool or file
authority.

## Goals

- Store ordinary conversation content once and reference immutable semantic revisions from trace.
- Reconstruct the sanitized semantic request and response—or an explicit incomplete-call record—for
  every provider call admitted with Capture On, including retries, tool loops, failures, stopped
  calls, and abandoned generations.
- Preserve historical call meaning across message edits, regeneration, forks, soft deletion, and
  source hard deletion.
- Persist provider-only material exactly once per sanitized semantic identity.
- Make cumulative normalized trace growth linear for an append-only conversation.
- Normalize and logically reclaim existing oversized captures without requiring a manual purge.
- Keep post-dispatch settlement best-effort: trace failure must not roll back a successful provider
  result or saved conversation message, while reservation and destructive-message preservation keep
  their explicit fail-closed contracts.
- Remove recognized credentials from every durable trace owner and offer optional PII masking with
  built-in and user-authored rules.
- Keep Safe and Full labels as honest viewer/export disclosure profiles rather than competing
  durable histories.
- State precisely what can and cannot be reconstructed.

## Non-goals

- Capturing literal provider HTTP headers, authentication material, TLS traffic, or SDK-internal
  request framing. The llama.cpp literal payload remains capturable only where Chatbook already
  owns it.
- Token-level stream replay or raw `assistant/chunk` persistence. Lossless chunk-row encoding is
  separately filed as TASK-23112.
- Synchronizing traces, indexing hidden Full bodies, or adding trace data to transcript FTS.
- Encrypting the trace database beyond the application's existing storage guarantees.
- Rewriting or sanitizing the user's canonical conversation when trace PII redaction is enabled.
- Retroactively applying optional PII rules to legacy captures without a separate explicit user
  action.
- Forensic erasure from filesystem snapshots, backups, exports, or already copied databases.
- One-action lineage-wide purge. Each conversation owner is detached explicitly.
- Replacing all conversation persistence with a new event-sourced application model.
- Recording calls deliberately sent with Capture Off.

## Fidelity contract

The trace must not use “exact” without qualifying intentional omissions. Its fidelity class is:

> **Semantic with disclosed omissions:** the trace reconstructs the exact provider-neutral values
> accepted at Chatbook's provider-call boundary except for content deliberately transformed or
> withheld by the frozen credential, PII, binary, size, or failure policy recorded for that call.

The boundary is the final semantic kwargs handed to `chat_api_call()` after Chatbook-owned request
construction and provider routing. Adapter-internal HTTP serialization remains outside the
contract. A provider-specific literal object may be retained only where Chatbook constructs and
owns that literal object, such as the existing llama.cpp seam.

Each reconstructed component identifies one of:

- an immutable message semantic revision;
- a sanitized content-addressed artifact;
- a deterministic structural value recorded in a request header or trace event; or
- an explicit unavailable/omitted marker with policy reason.

If credential filtering, PII masking, binary stubbing, truncation, legacy loss, corruption, or a
sanitizer failure changes a component, the viewer and export disclose that fact. They never claim
byte-for-byte or complete semantic reconstruction for that call.

## Core invariants

1. **Model-visible implies durably referenced or explicitly omitted.** Nothing in a captured
   semantic call can exist only in transient capture memory.
2. **Ordinary message bodies are not copied on capture.** A current semantic revision is metadata
   pointing to the canonical message row until copy-on-write materialization becomes necessary.
3. **One logical trace representation.** Storage packing, compression, and later chunk-row
   encoding are physical codecs, never a parallel event vocabulary.
4. **Calls point to boundaries, not history arrays.** No normalized call row contains a list of all
   preceding message or event references.
5. **Historical trace is immutable.** Edits and compaction append replacement events; the viewer
   never updates historical request or response bodies.
6. **Fork prefixes are shared.** A fork owns a reference to one immutable source boundary and only
   its new suffix.
7. **Sanitization fails closed.** A component that cannot be safely transformed is omitted with a
   content-free marker; raw fallback persistence is forbidden.
8. **Conversation success outranks post-dispatch trace settlement.** Once dispatch starts, trace
   sealing is independently idempotent and cannot roll back a provider result or a successfully
   saved assistant message. Pre-dispatch reservation remains mandatory.
9. **Destructive semantic mutation preserves reachable history transactionally.** Required
   materialization, projection binding, semantic revision transition, canonical mutation, and
   surface replacement either commit together or all abort.
10. **Every destructive sweep rechecks the complete reachability graph.** A global trace-graph
    epoch advances on every root or edge mutation; sweep aborts if that epoch changes.
11. **Capture On reserves and marks dispatch before provider entry.** Every dispatched Capture On
    call has a durable content-free reservation and a durable `dispatch_started` transition. Crash
    recovery distinguishes calls known not to have dispatched from calls whose provider receipt is
    unknowable.

## Conceptual storage model

Final SQL names may follow repository conventions, but the following ownership boundaries are
required.

### Semantic message revisions

A semantic revision is distinct from the existing optimistic-lock `messages.version`, which can
change for reasons unrelated to provider-visible content.

Each semantic revision records:

- stable revision identity;
- source conversation and message identities;
- normalized structural role/content kind without a persisted content digest;
- creation reason and time;
- a live canonical-source locator while that source remains valid; and
- the source revision it replaces, when any.

Creating a trace reference to the current message creates metadata only. Before an edit or hard
deletion overwrites/removes referenced content, every required frozen disclosure projection is
materialized once and bound through a separate immutable `(revision, policy) -> artifact-or-omission`
relation. The revision itself does not have one ambiguous materialized locator. Repeated calls under
the same frozen policy reuse the same binding; different policies may resolve to different artifacts
or omissions. The canonical locator is retired only in the same transaction that proves all
reachable policy bindings are durable.

Ordinary conversation text is never assigned a persisted raw, salted, or keyed content fingerprint
for trace identity. Revision equality is its opaque revision identity maintained transactionally by
the message write path. Capture-time comparison with provider values may use an ephemeral in-memory
digest, but that value is discarded before persistence. This prevents a deleted or masked message
from leaving a durable guess-verification oracle in trace metadata.

#### Model-visible envelope and mandatory mutation choke point

The semantic revision covers the complete provider-visible message envelope: role, optional name,
text/content, ordered multimodal blocks, tool-call identifier and tool-call structures, replayed
reasoning/thinking fields, attachment identities and selected variants, and every provider-neutral
sidecar field that can change final semantic kwargs. Inclusion, ordering, or context selection is a
surface-head change; changing one of those message-owned values is a semantic revision.

All writes capable of changing that envelope—including ordinary edits, generation settlement or
replacement, imports, sync, attachment/variant rewrites, regeneration, and hard deletion—must pass
through one `SemanticRevisionCoordinator` database boundary. Implementation first inventories every
current mutation route. A transaction-scoped mutation grant plus database constraints or triggers
reject direct mutation/deletion of a referenced semantic source unless the same transaction creates
the required revision transition, projection bindings, and surface replacement. Negative tests must
prove that direct SQL and every inventoried bypass fail closed. Soft deletion that retains the row
changes visibility/ownership, not its semantic bytes; later hard deletion still uses this boundary.

### Trace lineage and segments

A conversation owns one trace head. A durable fork creates a child lineage segment with:

- its own immutable identity;
- a reference to the parent segment;
- the inclusive parent boundary inherited by the child; and
- a suffix sequence beginning after that boundary.

The logical event stream is the parent prefix followed by the child suffix. A segment has one
parent, so ordinary forks form a tree; shared immutable artifacts and revisions may create a DAG
of references without creating mutable multiple-parent event streams.

Temporary forks use the same shape in memory. Saving a temporary fork with a durable prefix
persists the prefix reference. Saving one whose source prefix exists only in memory materializes
that prefix once under the saved fork without persisting or mutating the temporary source.

### Typed trace events and the model-visible surface

Trace events are append-only and carry monotonically ordered segment-local sequence identities.
The minimum logical vocabulary is:

- turn and call boundaries;
- model-surface append;
- model-surface replacement carrying one predecessor surface head, one bounded contiguous range
  reference, and the replacement node;
- tool call and model-facing tool result;
- request-header selection;
- provider route/overlay selection where it affects captured semantics;
- response selection;
- call outcome and usage; and
- trace gap or unavailable component.

A surface event stores a small structural shell and references content slots. It does not embed an
ordinary conversation body. The surface is a persistent sequence/operation chain addressed by an
immutable head. A replacement never stores a variable-length list of shadowed events: predecessor
head plus validated start/end nodes identifies the replaced range. Folding append and replacement
events yields the ordered semantic
surface visible to the model at any boundary. A human transcript remains a separate projection of
canonical messages; compaction or replacement may shadow model context without hiding the
original human transcript.

### Request headers

A request header is the complete logical non-history envelope for one model-message series:

- provider and model;
- effective generation parameters and adapter-default provenance;
- rendered system framing references;
- tool-schema references;
- response-format and reasoning/thinking controls;
- endpoint's canonical credential-free identity; and
- captured provider-owned overlays required to explain the semantic boundary.

The first request records a header. Later calls reuse it while the effective value is unchanged.
A real change or surface-series reset records a new immutable header. Large components are
content-addressed artifact references, so even a complete logical header does not repeat system
prompt or tool-schema bodies.

### Provider calls

A provider call is owned by conversation, trace lineage, turn, run, and call sequence—not by the
eventual assistant message. It records:

- immutable call identity and idempotency key;
- lifecycle state (`reserved`, `not_dispatched`, `dispatch_started`, `dispatch_unknown`,
  `response_started`, or a terminal outcome);
- trace surface boundary consumed by the call;
- effective request-header identity;
- provider/model/route identity;
- response revision or artifact identity;
- complete/stopped/error/interrupted/abandoned outcome;
- normalized usage when available;
- frozen capture and redaction policy provenance; and
- disclosed omission and integrity state.

Retries and tool-loop calls are distinct calls. A call may later be linked to the assistant message
that presents its result, but that link is presentation metadata rather than ownership.

The lifecycle is monotonic. `reserved` can become `not_dispatched` or `dispatch_started`;
`dispatch_started` can become `response_started`, `dispatch_unknown`, or a terminal provider error;
`response_started` can become `complete`, `stopped`, `error`, or `interrupted`. `abandoned` is a
terminal user/run decision only when durable evidence proves no provider operation remains live.
Recovery never moves a call backward or converts uncertainty into `not_dispatched`.

### Content-addressed artifacts

Provider-only semantic bodies are stored after mandatory sanitization and optional capture-time
PII transformation. Identity includes the sanitized bytes, structural media type, and relevant
normalization version. Artifacts include rendered system text, injected instructions, RAG/memory
context not represented by a canonical message, tool schemas, provider overlays, unmatched legacy
components, and provider responses that differ from the saved assistant revision.

Binary bodies remain external or stubbed according to the existing attachment boundary. A digest
may verify an existing immutable attachment, but trace capture does not duplicate its bytes.

Artifact content identities are computed only over the sanitized body that the artifact itself
stores. They are never computed over a withheld credential, masked PII value, canonical conversation
text, or omission marker. Because anyone who can read the artifact already has its sanitized body,
its internal deduplication identity reveals no additional omitted content.
Reuse compares the stored sanitized bytes and structural identity rather than trusting a digest
alone. A mismatch allocates a separate opaque artifact identity instead of aliasing two bodies.

### Redaction projections

Redaction projections are keyed by opaque source revision/artifact identity plus an immutable policy
identity containing credential-detector version, PII-enabled state, and an opaque PII ruleset
revision identity.
They store a structured field path plus normalized Unicode codepoint ranges, category, stable rule
IDs, detector versions, and outcomes. They never store matched values, value hashes, captured substrings, exception text, or
user-authored regex source. Start/end ranges necessarily disclose the matched codepoint count and
position; that limited leakage is accepted because exact ranges are required to mask a referenced
canonical message without copying it. No separate value length or surrounding text is stored.

The same revision referenced repeatedly under the same frozen policy reuses one projection. Both
Safe and Full trace reads resolve canonical references through the call's frozen mandatory and
optional masks. The ordinary conversation UI remains an unredacted view of canonical messages.

## Provenance-aware request preparation

`PreparedConsoleRequest` keeps its existing event-neutral semantic sections. A parallel immutable
capture descriptor travels beside each semantic unit and identifies its source:

- message semantic revision;
- conversation setting or rendered system artifact;
- automatic project instruction or workspace context;
- RAG/memory artifact;
- tool definition, call, or result;
- provider transformation/overlay; or
- active user request.

Descriptors are capture-only. They are not serialized into provider messages, cannot grant local
filesystem authority, and do not alter token counting or context selection.

Provider adaptation must preserve a binding between every final semantic value and its descriptor.
Immediately before dispatch, the gateway independently verifies that resolving the descriptors
reconstructs the final captured kwargs. A mismatch marks the affected component unavailable on the
already-reserved call; it never falls back to persisting the raw kwargs wholesale.

## Runtime flow

1. **Admit the run.** Resolve Capture On/Off, PII redaction, provider settings, and applicable
   conversation/next-send overrides. Freeze them on the provider run before any call begins.
2. **Prepare semantics and provenance.** Build the semantic request and parallel descriptors.
3. **Adapt to the provider boundary.** Apply provider transformations while carrying content-slot
   provenance into the final kwargs.
4. **Reserve before dispatch.** Allocate conversation/turn/run/call identity and synchronously
   commit a minimal content-free `reserved` call carrying lineage identity and frozen capture-policy
   provenance. If this reservation fails, Capture On does not dispatch automatically: the UI offers
   Retry or an explicit one-shot **Send without capture** action that admits a new Capture Off call.
5. **Open and advance the surface.** Append only new semantic items or explicit replacements. Reuse existing
   revision/artifact references.
6. **Select the header.** Reuse the prior header if value-identical; otherwise record a new complete
   logical header containing content references.
7. **Bind and mark dispatch.** Store the surface boundary and header identity on the reservation.
   Immediately before entering the provider adapter, durably transition the call to
   `dispatch_started`. If binding,
   sanitization, or descriptor verification fails after reservation, retain the reservation as
   incomplete with content-free status when writable; a manually admitted call may proceed because
   the durable row already proves that it existed.
8. **Observe the response.** Accumulate the bounded semantic response required by the current
   Inspector. Raw token chunks remain out of scope.
9. **Seal independently.** Commit response reference/artifact, outcome, usage, omissions, and policy
   provenance in an idempotent trace transaction independent of conversation-message settlement.
10. **Settle failures honestly.** A failed post-dispatch trace write enters a bounded best-effort
    settlement queue. Process death leaves the durable reservation open or incomplete; cold recovery
    maps `reserved` to `not_dispatched`, maps `dispatch_started` without provider evidence to
    `dispatch_unknown`, and maps `response_started` to `interrupted`. It never claims whether an
    unknown dispatch reached the remote provider.

Reservation retries use the immutable call idempotency key: after an ambiguous local commit result,
the gateway queries that identity before creating another row. For the initial manually initiated
call, reservation/binding failure offers Retry or explicit Send without capture. During a retry,
tool loop, or other already-running interactive sequence, the run pauses before the next provider
entry and requires Resume with capture or an explicit capture-off continuation. An autonomous or
scheduled run fails safely instead of silently disabling capture. A capture-off bypass exists only
in live run/UI state because Capture Off deliberately creates no durable trace row.

Temporary conversations remain non-durable. They cannot enter a provider adapter with durable
Capture On while still temporary. The UI offers **Save & Send**, which promotes the conversation and
persists its in-memory lineage before reservation, or an explicit **Send without capture**. Existing
temporary trace/fork state can remain coherent in memory but is not restart-durable until saved.

After a provider response, conversation settlement first saves the canonical assistant message and
its semantic revision. Trace sealing then re-reads that immutable revision and links it only if its
sanitized envelope exactly equals the assembled provider-facing response. If trace sealing loses the
race or cannot verify equality, it stores the sanitized response artifact rather than a questionable
revision link. Failure to save the conversation message may still leave the provider response in the
trace artifact; failure to seal trace never rolls back a saved message. If display filtering, reasoning separation, tool payloads, synthetic
fallback copy, or another transformation makes it differ, the trace stores the provider-facing
response as one sanitized artifact and labels the relationship.

## Message edits and replacements

Every destructive semantic mutation transaction must:

1. identify the current semantic revision;
2. enumerate every reachable frozen policy binding that would lose its source content;
3. materialize each required sanitized projection or explicit omission once;
4. append a new semantic revision for the edited message;
5. append a bounded predecessor-head/range surface replacement; and
6. commit projection bindings, revision transition, canonical mutation, and replacement atomically.

Historical calls retain their earlier boundary and resolve the old revision. New calls fold the
replacement. Regeneration and context compaction use the same append/replace semantics rather than
editing trace events. This fail-closed mutation contract is intentionally different from
post-dispatch response settlement: an edit, attachment rewrite, sync update, generation replacement,
or hard delete that cannot preserve reachable trace history does not commit.

## Fork behavior

A fork captures the immutable trace head at the same source snapshot fence used for the message
lineage. The inherited prefix includes calls associated with copied active-lineage messages through
the boundary, including failed or abandoned attempts that happened before that boundary. It
excludes calls admitted after the snapshot and calls belonging solely to excluded branches.

The source remains untouched. The child reads the shared prefix as immutable history and appends a
new suffix. Viewer projection interleaves the child's copied transcript with the inherited trace by
source lineage and call boundaries, so the full conversation remains coherent.

Hard-deleting the source conversation detaches its owner but cannot remove a prefix still owned by
a fork. Before source message rows disappear, required trace projections materialize in the same
transaction. If any projection cannot be preserved safely, deletion aborts.

Purging trace from one conversation detaches that conversation only and lists forks that still own
shared history. This task provides no lineage-wide one-click purge.

## Capture and privacy controls

Three controls remain deliberately separate:

- **Capture:** On or Off.
- **PII redaction:** On or Off, default Off.
- **Viewer profile:** Safe or Full.

Capture and PII policies support global default, sparse conversation override, and eligible
next-send override with explicit precedence. The resolved values freeze at run admission and apply
to retries and tool-loop calls. Forks inherit the source's future-capture configuration, but every
historical call retains its original policy provenance. Viewer profile is a local presentation
preference and does not mutate capture policy or historical data.

The existing capture-detail Safe/Full setting is retired for future writes. Historical
`capture_detail` remains immutable provenance. Existing capture-enabled and per-conversation
enable/disable choices migrate to Capture On/Off, but an old Full choice never silently enables Full
disclosure: after upgrade, every profile starts in Safe viewer mode and receives a one-time
explanation. Persisted old Full bodies remain available only after the user explicitly switches the
viewer to Full. Obsolete next-send capture-detail overrides are discarded at restart rather than
being reinterpreted; new next-send controls are Capture and PII only.

Safe and Full use the same stored trace:

- **Safe** shows ordinary transcript context plus structural trace facts while collapsing or
  masking sensitive provider-only bodies, tool arguments/results, automatic instruction bodies,
  and other high-disclosure sections. Safe search, copy, and export operate only on this projection.
- **Full** may reveal all persisted non-credential content that survived capture-time PII policy.
  Expanding a sensitive section and copying/exporting Full require explicit confirmation.

Canonical message references are masked through each call's frozen credential/PII projections in
both profiles. Safe/Full changes provider-only disclosure, not whether frozen capture-time masks are
honored. The normal conversation transcript remains unchanged and may therefore contain PII that
the trace masks.

Existing users receive a one-time disclosure that Safe is a viewer/export profile, not a reduced
at-rest trace. Status copy reports stored fidelity, credential filtering, PII state, and current
viewer profile separately.

## Mandatory credential filtering

Known credential kwargs, credential references, URL userinfo/query/fragment, and known
credential-bearing nested fields are structurally excluded. A versioned detector also scans
provider-only free text for recognized secret formats. Arbitrary secrets embedded in prose cannot
be guaranteed detectable; consent and help text state this limitation.

Filtering occurs before artifact identity or persistence. For message-revision references, the
trace stores a mandatory credential redaction projection and applies it on every trace read. If a
referenced live revision must be materialized, only that sanitized trace projection enters the
artifact store.

A credential sanitizer error yields an unavailable component marker. No raw fallback is allowed.
Bodies, values, matches, and exception strings never enter logs or migration diagnostics.

## Optional PII redaction

PII redaction is irreversible for captured provider-only artifacts and applies to both Safe and
Full trace projections. For canonical message references it stores immutable masks; it does not
rewrite the user's conversation. The settings UI must say this explicitly.

Version 1 includes built-in detectors and user-authored regex rules. Each user rule has a stable ID,
label/category, enabled state, pattern, limited documented flags, and deterministic priority.
Patterns are length-bounded, compile-validated, and screened for unsupported constructs when saved.

Because CPython's `re` cannot be safely interrupted, custom rules execute as one bounded batch in a
stdlib-only disposable subprocess. Input bytes, field count, pattern count, match count, and wall
time are capped. The parent kills the process on timeout and treats crash, malformed output, excess
matches, or deadline as a fail-closed redaction failure for affected components. A worker is never
reused across unrelated captures.

Candidate spans use Unicode codepoint offsets over the exact source content bound to the recorded
revision or artifact identity. They are sorted deterministically and overlapping ranges are unioned. A union with multiple categories
uses `mixed`; credential filtering has already occurred and cannot be weakened by a PII rule.
Changing or deleting a rule never changes historical masks. Trace rows retain stable rule IDs and
an opaque ruleset revision identity, not a hash or the pattern source.

## Viewer, search, copy, export, and purge

Historical trace is not editable. The viewer supports inspect, expand/collapse, filter, permitted
search, copy, export, and owner-scoped purge.

The viewer reconstructs one call by:

1. resolving its lineage prefix and suffix through the recorded boundary;
2. folding surface append/replacement events;
3. resolving the selected request header;
4. resolving source revisions and artifact slots through the call's disclosure projections; and
5. rendering explicit omissions and integrity status beside the result.

Safe operations never materialize hidden Full bodies. Full search, when explicitly active, scans a
bounded in-memory projection; hidden Full text is never written to FTS, persistent previews,
clipboard history owned by Chatbook, logs, or metadata. Full expansion and copy/export confirmations
name PII state and the possibility of sensitive prose that detectors missed.

Purge removes one conversation's trace ownership under capture quiescence. Shared lineage nodes or
artifacts remain while another fork owns them. The action reports those remaining owners and does
not imply forensic erasure.

## Failure, integrity, and concurrency

- Reservation, lifecycle transition, binding, and sealing operations are idempotent under immutable
  call identity.
- Every Capture On provider dispatch requires a committed content-free call reservation. Reservation
  failure blocks automatic dispatch until Retry or an explicit one-shot Capture Off confirmation.
- Cold recovery maps a committed `reserved` row with no dispatch transition to `not_dispatched`, a
  `dispatch_started` row without provider evidence to `dispatch_unknown`, and a `response_started`
  row without a terminal seal to `interrupted`; already durable events remain immutable.
- Provider success and assistant-message persistence are not transactional with post-dispatch trace
  settlement. Destructive semantic message mutations are transactionally coupled to required trace
  preservation and fail closed.
- A trace component resolution mismatch fails closed for that component and marks the call
  incomplete; it never captures raw kwargs as a fallback.
- Fork boundaries and edit replacements validate immutable revision and lineage heads before commit.
- Hard deletion stages sanitized materialization and removal in one transaction.
- Every mutation of a reachability root or edge—including events, surface heads, owner/fork links,
  projection bindings, revision locators, headers, artifacts, and migration state—advances one
  global trace-graph epoch transactionally. Garbage collection records it with the mark snapshot,
  obtains the maintenance exclusion for sweep, rechecks the epoch in the sweep transaction, and
  aborts on change.
- Multiple independent app processes writing one profile remain unsupported unless an existing
  database-wide owner already provides the required exclusion.

## Existing-data normalization

Schema installation is a fast DDL migration that enables new normalized writes and dual reads.
Content conversion does not block first UI readiness. A `TraceMaintenanceCoordinator` runs only
when no provider run or other database maintenance is active, processes batches bounded by both row
count and decoded bytes, and yields between batches.

For each legacy exchange:

1. Decode through the bounded production decoder.
2. Apply the current mandatory credential filter. Optional PII is not retroactively applied.
3. Split request history into independently classifiable semantic components.
4. Match ordinary rows to unique historical message revisions using role, within-call order,
   structural shape, and an ephemeral sanitized semantic fingerprint discarded before commit.
5. Store exact matches as revision references, but do not infer a cross-call parent, fork, edit, or
   replacement chronology the legacy blob never recorded.
6. Store unmatched provider-only rows as independently content-addressed artifacts.
7. Preserve ambiguous rows as individual sanitized legacy artifacts with an ambiguity marker; never
   drop them or retain the entire accumulated list merely because one row is ambiguous.
8. Deduplicate system framing, tool schemas, provider overlays, and responses.
9. Represent each legacy call as an isolated immutable `legacy_snapshot` surface whose ordered
   components use persistent sequence nodes `(parent_node, component_ref)`. Equal prefixes therefore
   deduplicate structurally without storing a per-call history array. Call owner, run tag, sequence,
   and timestamp provide only their recorded partial order; live normalized lineage starts from an
   explicit import/legacy boundary and never invents a global predecessor.
10. Write and read back the normalized call.
11. Reconstruct the expected sanitized legacy projection and compare it structurally.
12. Delete the legacy blob only after equivalence succeeds in the same transaction.

ADR-096's earlier bounded-excerpt migration may already have irreversibly removed rows. A valid
aggregate omission marker becomes an explicit component node `legacy_omission`; the new system does not
pretend to recover its missing content. Existing Safe project-instruction omissions and corrupt or
truncated blobs receive the same honest treatment.

Dual reads prefer a verified normalized call and otherwise decode legacy. Migration state is
visible as:

- new-capture protection active;
- legacy normalization pending;
- logical reclamation complete; and
- physical compaction pending or complete.

Recognized corrupt/unavailable rows remain isolated and counted without content-bearing logs.
Unexpected code or SQLite errors roll back the current batch and leave its checkpoint retryable.

## Garbage collection and physical compaction

Deleting a conversation or migrated blob drops an ownership root and advances the trace-graph epoch;
it does not synchronously walk
the complete shared graph. Every other edge mutation advances the same epoch. A background
mark/sweep later reclaims unreachable events, artifacts,
headers, revisions, and redaction projections. Soft-deleted conversations and migration-pending
legacy rows remain roots.

SQLite logical deletion frees pages but does not necessarily shrink the database file. After
logical normalization and garbage collection, automatic physical compaction enters a visible
maintenance pause only when:

- no provider run or other maintenance is active;
- all application connections can be closed;
- WAL is checkpointed;
- available disk space passes preflight; and
- the maintenance lease remains current.

If admission fails, compaction remains visibly pending and retries at a later eligible idle window.
Logical live bytes, freelist bytes, WAL bytes, and allocated database-file bytes are measured and
reported separately. Neither logical purge nor VACUUM claims erasure from backups or prior exports.
Graph mutation is excluded during the final sweep transaction; inability to obtain the exclusion is
a normal retry condition, not permission to sweep against a stale mark.

## Performance and growth verification

The acceptance benchmark must use the real provider gateway and production kwargs shape, not a
hand-built capture. It creates 200 turns with semi-incompressible content and records at turns 1,
50, 100, 150, and 200:

- live database bytes;
- SQLite allocated bytes and freelist pages;
- WAL bytes;
- normalized call/event/header rows;
- unique artifact and materialized-revision bytes;
- legacy bytes remaining; and
- encode/decode and call-settlement time.

For an append-only conversation with unchanged header and no provider-only body changes:

- each ordinary turn adds a bounded number of trace rows and content references;
- no provider call stores an array or blob proportional to prior transcript length;
- cumulative normalized bytes grow linearly with new semantic content and call count; and
- the measured growth from turns 100 to 200 is reported against growth from turns 1 to 100 rather
  than hidden behind compression ratios.

The report must separately show logical reclamation after legacy normalization and physical file
size after the eligible compaction step.

The benchmark fixture is versioned in the repository with fixed per-turn input/output lengths,
provider kwargs shape, randomness seed, and content checksum. The benchmark is a release gate, not
an observational report. Using five fresh-database runs and the median (with the reference machine
and SQLite settings recorded), it passes only when:

- normalized live trace-owned bytes added by turns 101–200 are at most 1.25 times those added by
  turns 1–100, excluding canonical conversation bytes and SQLite freelist/WAL allocation;
- normalized trace-owned rows added by turns 101–200 are at most 1.25 times those added by turns
  1–100;
- normalized live trace-owned bytes at turn 200 are no more than 2.0 MiB for the pinned fixture;
- pre-dispatch reservation/`dispatch_started` persistence has p95 latency at most 10 ms and no
  reference-run sample exceeds 50 ms;
- post-dispatch settlement runs off the UI thread and has p95 transaction latency at most 25 ms;
- each legacy batch holds its write transaction for at most 100 ms, decodes at most 4 MiB and 100
  rows, and yields before the next batch; the pinned 200-turn legacy fixture normalizes within 5 s;
  and
- every result includes the raw per-run measurements so a median cannot hide a pathological run.

A second 200-turn compaction-heavy fixture replaces the oldest 75 percent of the active model
surface every 20 turns. It must meet the same 1.25 second-half row/byte growth ratios, and schema
inspection must prove no replacement row contains a variable-length shadowed-source list.

Physical compaction of the pinned fixture must complete within 5 s on the recorded reference
machine. Arbitrary user databases have no dishonest universal time bound: they must receive a
preflight estimate and visible progress, the UI event loop must remain responsive, and provider
dispatch stays paused only for the admitted maintenance interval.

## Rollout and implementation decomposition

TASK-23026 is the originating finding but is already Done on `origin/dev` under ADR-096's
superseded bounded-retention implementation. Its checked acceptance criteria and implementation
notes remain historical evidence and must not be rewritten. Before implementation, the writing
plan must create a new umbrella task for ADR-097 and dependency-ordered, atomic Backlog work
packages suitable for single pull requests. Core ledger capture, mandatory filtering, and logical
legacy normalization land and prove their gates before custom-regex execution and physical
compaction are enabled:

1. schema, semantic revisions, artifact identity, trace lineage, and dual-read foundation;
2. provenance-aware request preparation and normalized new-call capture;
3. edit, regeneration, fork, hard-delete, recovery, and ownership integration;
4. mandatory credential policy, Safe/Full viewer projection, search/copy/export, settings migration,
   and built-in PII masking;
5. legacy snapshot normalization, garbage collection, linear-growth benchmark, and rollout docs;
6. bounded custom-regex subprocess and policy-management UX after core ledger/privacy gates pass;
   and
7. physical compaction admission/progress after logical reclamation and GC are proven.

Exact task IDs must be created before they are referenced from Backlog task files. TASK-23112 stays
a separate later feature and is not a prerequisite for the forthcoming ADR-097 implementation
umbrella.

## Verification contract

Implementation follows test-driven development. Targeted coverage must include:

- direct calls, multiple retries, tool loops, fallback calls, stops, errors, interruptions, and
  abandoned regenerations;
- exact boundary/header selection under unchanged and changed provider configuration;
- Anthropic system separation, provider transformations, and llama.cpp literal ownership;
- response equality/ref selection versus provider-only response artifacts;
- repeated message edits, surface replacements, compaction, and copy-on-write materialization;
- mutation-route census plus rejection of direct SQL and every edit/sync/import/attachment bypass;
- durable-to-durable, durable-to-temporary, temporary-to-temporary, and later-saved fork prefixes;
- temporary-chat Save & Send, explicit capture-off send, and restart-durability boundaries;
- source soft deletion, successful hard deletion, and injected materialization failure rollback;
- shared-owner purge disclosure and trace-graph-epoch garbage-collection races;
- credential fields, URLs, nested values, recognized free-text formats, sanitizer failure, and
  absence from trace-owned tables/artifacts and every trace-derived log, exception, preview,
  clipboard, and export path; canonical conversation rows are explicitly outside this absence
  assertion;
- durable pre-dispatch reservation and `dispatch_started` transition; ambiguous reservation retry;
  manual Retry/Send-without-capture behavior; interactive loop pause/resume; autonomous fail-safe;
  post-reservation binding failure; seal failure; and crash recovery to `not_dispatched`,
  `dispatch_unknown`, or `interrupted` as evidence permits;
- built-in and custom PII rules, Unicode spans, overlaps, `mixed` classification, invalid patterns,
  subprocess timeout/crash/malformed output, ruleset changes, and Safe/Full behavior;
- dual reads, all legacy capture variants, ambiguous matches, old aggregate markers, corruption,
  isolated persistent legacy snapshots without invented lineage, idempotent resume, equivalence
  failure, and batch rollback;
- idle-maintenance admission, WAL checkpoint failure, disk preflight failure, deferred retry, and
  physical size reporting; and
- the append-only and compaction-heavy 200-turn release gates described above.

Per repository policy, implementation verification begins with tests touching the modified
functionality. A full suite requires explicit user opt-in.

## Documentation changes

The Console user guide and Inspector help must explain:

- ordinary transcript bodies are referenced rather than copied into each exchange;
- provider-only context is stored once when capture is enabled;
- Safe and Full are viewer/export disclosure profiles over the same stored trace;
- credential filtering is mandatory but cannot guarantee arbitrary prose secrets;
- PII redaction protects traces and does not rewrite the conversation;
- existing Safe/Full capture-detail choices migrate to a Safe-default viewer without deleting old
  Full bodies;
- temporary chats require Save & Send for durable Capture On, or an explicit capture-off send;
- edits and forks retain coherent historical traces through immutable revisions/shared prefixes;
- owner-scoped purge may leave history retained by forks;
- automatic legacy normalization may contain explicit irrecoverable omissions from older policies;
  and
- logical deletion, physical SQLite compaction, backups, and exports have different erasure
  semantics.

## Alternatives considered

### Keep a bounded Safe excerpt

Rejected. It makes Safe less useful precisely when older context, injected instructions, or an
earlier tool result caused the response. It also leaves Full with quadratic duplication and makes
diagnostic completeness depend on a setting selected before the failure.

### Keep the last N rows plus one fingerprint per omitted row

Rejected. Per-row metadata still produces quadratic cumulative growth, and fingerprints can become
guess-verification oracles for private text.

### Keep one complete request blob per call and deduplicate only compression blocks

Rejected. Compression hides but does not remove repeated logical ownership, complicates deletion
and edits, and retains a second transcript as the operative trace truth.

### Store arbitrary diffs between serialized request JSON values

Rejected. A custom patch language creates diff/apply/fallback machinery, brittle provider-specific
paths, and corruption propagation. Typed surface events and immutable component references preserve
one logical model.

### Replace the complete application conversation model with a DeepSeek-style session log

Rejected. Chatbook already has canonical message trees, persistence, variants, sync boundaries, and
conversation ownership. The trace ledger should reference that source rather than migrate unrelated
application behavior.

### Copy a fork's trace prefix into the child

Rejected. It recreates the storage defect across forks and makes source deletion/purge semantics
ambiguous. Immutable prefix sharing preserves coherent history without physical duplication.

### Let Safe and Full store different content

Rejected. A reduced Safe store cannot explain failures caused by omitted context, while automatic
Full duplication recreates the privacy and storage problem. One sanitized store plus explicit
disclosure profiles is both useful and honest.

### Run custom PII regex in the application process

Rejected. Python's standard regex engine has no portable hard timeout; a pathological user rule can
freeze the TUI. Validation remains useful, but only a killable subprocess provides the required
runtime bound without a new mandatory regex dependency.

### Delete all trace when the source conversation is deleted

Rejected. A fork that inherited the source boundary would become incoherent. Copy-on-write
materialization and shared ownership retain only history that remains reachable.

## ADR check

ADR required: **yes**

ADR path: `backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md`

Reason: the design changes durable schema and migration, semantic data ownership, fork lineage,
edit/delete behavior, provider/capture contracts, privacy policy, redaction guarantees, and
long-lived Inspector UX. It supersedes ADR-096 and amends the existing Full semantic-capture and
fork decisions.

## External reference

- [DeepSeek Harness session model](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.md)
- [DeepSeek Harness persistence model](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/persistence.md)
- [DeepSeek reconstructable-request decision](https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-07-05-reconstructable-requests.md)
- [DeepSeek telemetry/redaction separation](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session-telemetry.md)
- [DeepSeek chunk-row codec reference for TASK-23112](https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/session/src/chunk-rows.ts)
