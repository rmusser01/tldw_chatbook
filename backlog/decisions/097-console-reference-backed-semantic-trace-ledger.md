# ADR-097: Use a reference-backed semantic trace ledger

Status: Accepted

Date: 2026-08-28

Originating Task: [TASK-23026](../tasks/task-23026%20-%20Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md), completed by the superseded bounded-retention implementation

Related Spec: [Console Reference-Backed Semantic Trace Ledger](../../Docs/superpowers/specs/2026-08-28-console-reference-backed-semantic-trace-ledger-design.md)

Supersedes: [ADR-096](096-console-safe-capture-retention.md), which bounded Safe
captures by discarding older semantic history.

Amends: the Console Full semantic-capture policy and
[ADR-092 Console chat fork](092-console-chat-fork-copy-and-authority-boundary.md).

## Context

Default Console exchange capture persists the complete accumulated provider message
list in every call's compressed `message_exchanges.capture_blob`. A production-shaped
200-turn conversation retained 15.40 MB because each later call copied all previous
messages. Soft deletion keeps the blobs and no automatic reclamation path exists.

ADR-096 chose a bounded Safe excerpt and one aggregate omission marker. That makes
future Safe storage linear, but it deliberately discards the historical context needed
to explain some provider behavior and leaves exact Full capture with the same repeated-
history architecture. The owner rejected that trade-off: ordinary conversation content
already has a durable owner and must not be copied into each exchange. Provider-only
semantic context should be stored once or explicitly omitted.

Fixing this requires new durable ownership for semantic revisions, provider-only
artifacts, call boundaries, fork prefixes, redaction projections, legacy normalization,
and deletion. It also changes Safe/Full meaning and the earlier rule that automatic
project-instruction bodies never enter default durable capture.

## Decision

1. **Make the saved conversation the ordinary-content source of truth.** A captured
   call references immutable semantic message revisions. Capturing a current revision
   creates metadata pointing to the live message row and does not copy its body.

2. **Use copy-on-write historical revisions.** Before an edit or hard deletion would
   destroy referenced content, the required sanitized trace projection is materialized
   once per unique disclosure policy and bound through an immutable
   `(revision, policy) -> artifact-or-omission` relation in the same transaction. A
   revision does not have one ambiguous materialized locator. Failure aborts the edit
   or deletion.

   Semantic revision identity is an opaque transactional identity, not a persisted
   raw, salted, or keyed digest of canonical conversation text. Ephemeral comparison
   fingerprints are discarded before commit.

3. **Add one append-only semantic trace ledger.** Typed events record turn/call
   boundaries, model-surface append/replacement, tool traffic, request-header
   selection, provider overlays, response selection, outcomes, usage, and explicit
   gaps. Events contain structural shells and content references, not repeated ordinary
   bodies.

4. **Give each provider call a trace boundary and header reference.** Calls are owned
   by conversation, lineage, turn, run, and call sequence rather than the eventual
   assistant message. Retries, tool-loop calls, failures, stops, interruptions, and
   abandoned generations remain distinct calls. No call stores a history array
   proportional to the conversation's age.

5. **Store request headers only when their effective value changes.** A complete
   logical header records provider/model configuration, rendered system references,
   tool-schema references, response/reasoning controls, endpoint's credential-free
   identity, and required provider overlays. Large components are content-addressed
   artifacts, so a new header does not duplicate their bodies.

6. **Store provider-only semantic material once.** Rendered automatic instructions,
   RAG/memory context, tool schemas, provider overlays, unmatched legacy rows, and
   responses not equal to a saved assistant revision enter a sanitized content-
   addressed artifact store. Binary bodies remain external or stubbed under existing
   attachment rules. Artifact reuse compares sanitized stored bytes and structure after
   digest lookup; a mismatch receives a separate opaque identity rather than aliasing.

7. **Define fidelity as semantic with disclosed omissions.** Reconstruction covers the
   final semantic kwargs handed to Chatbook's provider-call boundary, not provider-
   internal HTTP. Credential filtering, optional PII masking, binary stubbing,
   truncation, corruption, legacy loss, and sanitizer failure are explicitly disclosed;
   affected calls never claim byte-exact or complete semantics.

8. **Carry provenance through provider-neutral preparation.** `PreparedConsoleRequest`
   keeps semantic sections while parallel capture-only descriptors identify message
   revisions, settings, automatic context, tool values, and provider transforms.
   Descriptors never reach the provider or grant authority. The gateway binds them to
   the exact final semantic values before dispatch and fails capture closed on mismatch
   instead of persisting raw kwargs.

9. **Share immutable trace prefixes across forks.** A fork records the source trace
   boundary alongside its message snapshot fence and appends only its own suffix.
   Durable and temporary forks retain coherent inherited history without physical
   prefix copying. Source deletion cannot remove a prefix still owned by a fork.

10. **Keep historical trace immutable and non-editable.** Message edits, regeneration,
    and context compaction append model-surface replacements using one predecessor
    surface head plus a bounded contiguous range, never a variable list of shadowed
    events. The viewer may inspect, filter, search permitted projections, copy, export,
    or purge ownership; it never edits historical trace.

    The semantic revision covers the complete provider-visible message envelope,
    including ordered multimodal/tool/reasoning/attachment sidecars. Every mutation
    route passes through one coordinator enforced by transaction-scoped database
    guards; direct mutation of referenced semantic content fails closed.

11. **Separate capture, PII, and viewer controls.** Capture On/Off and optional PII
    redaction resolve at global, conversation, and eligible next-send scope and freeze
    for a run. PII defaults Off. Safe/Full is a local viewer/export disclosure profile
    over the same stored trace, not a different at-rest history. Forks inherit future
    settings while historical calls retain their frozen provenance. Both viewer modes
    apply each call's frozen credential/PII masks to canonical references; the ordinary
    conversation transcript remains unchanged.

12. **Filter credentials mandatorily and fail closed.** Known credential fields,
    credential-bearing URL components, nested credential fields, and recognized secret
    formats in provider-only text are removed before trace persistence. Arbitrary prose
    secrets cannot be guaranteed detectable and the UI says so. Sanitizer failure
    creates a content-free unavailable marker; no raw fallback is stored.

13. **Offer irreversible trace PII masking.** Built-in detectors and validated user-
    authored regex rules produce immutable structured-field-path plus Unicode-span
    projections. Custom regex runs in a bounded killable subprocess because CPython
    `re` has no portable hard timeout.
    Historical projections retain source identity, start/end codepoint ranges,
    detector/rule IDs, and an opaque ruleset revision identity, never matched values,
    value hashes, surrounding text, or regex source. Ranges necessarily reveal matched position and
    codepoint count; this is accepted so referenced canonical messages can be masked
    without copying them. PII masking protects traces but does not silently rewrite
    canonical conversation messages.

14. **Normalize legacy captures automatically and resumably.** A fast schema step
    enables normalized writes and dual reads. After UI readiness, idle bounded batches
    split legacy blobs into revision references and deduplicated provider-only
    artifacts. A blob is deleted only after reading back the normalized call and
    reproducing its sanitized legacy projection. Legacy calls become isolated immutable
    snapshot surfaces backed by persistent prefix sequence nodes; migration does not
    invent cross-call edit/fork chronology the old rows never recorded. Ambiguous rows
    remain individual legacy artifacts. ADR-096 aggregate markers become explicit
    irreversible legacy omissions.

15. **Reserve Capture On calls before provider dispatch.** A minimal content-free call
    reservation containing identity, lineage, and frozen capture policy commits before
    each Capture On dispatch, then durably becomes `dispatch_started` immediately before
    provider-adapter entry. Reservation failure blocks automatic dispatch and offers
    Retry or an explicit one-shot Send without capture action. Interactive tool/retry
    loops pause for that choice; autonomous runs fail safely. Cold recovery maps an
    untouched reservation to `not_dispatched`, an uncertain started call to
    `dispatch_unknown`, and a response-bearing open call to `interrupted`, but only
    after a bounded inactivity grace period so another live app process cannot have
    its newly active provider call terminated by startup recovery.

    Temporary conversations cannot make a durable Capture On call until Save & Send
    promotes their in-memory lineage. Before dispatch, a component sanitization or
    descriptor-verification failure may proceed only after a content-free
    omission/incomplete marker and the remaining boundary/header state commit durably.
    Inability to persist the boundary, header, or incomplete state blocks dispatch and
    requires Retry or explicit Capture Off. After dispatch, component capture and sealing are
    best-effort and independently idempotent: they cannot roll back a provider result or
    saved assistant message. Destructive semantic edits/deletes are different: required
    preservation and canonical mutation commit together or all abort.

    A failed post-dispatch handoff remains explicitly owned by the store after its
    worker returns. App teardown waits for the worker and makes a final idempotent
    settlement attempt; any still-unsettled handoff remains visible in the definitive
    teardown diagnostic rather than being silently discarded.

16. **Use shared-owner garbage collection and honest physical maintenance.** Deletion
    detaches one conversation root. Database guards reject ordinary/direct deletion from
    append-only trace tables; a sweep receives a connection-local deletion grant only
    after its maintenance lease and exact marked epoch are validated in the sweep
    transaction. Background mark/sweep reclaims unreachable objects only after a global
    trace-graph epoch recheck. Every root or reachability-edge
    mutation advances that epoch, and sweep holds maintenance exclusion. SQLite
   physical compaction uses SQLite same-file `VACUUM` automatically at a later
   eligible visible idle pause. An app-wide ChaChaNotes connection registry first
   rejects new acquisitions, waits for all thread-owned connections to return, closes
   them, pauses provider dispatch, checkpoints WAL with `TRUNCATE`, verifies free disk
   for SQLite's temporary rewrite plus a safety margin, and retains the maintenance
   lease through reopen and integrity verification. `VACUUM` runs in a dedicated
   maintenance worker with bounded progress/cancellation checks where the Python
   SQLite API supports them. Every failure path reopens the database and resumes
   connection acquisition/provider dispatch from a `finally` boundary; incomplete
   admission or compaction remains visibly pending and retryable. Logical, freelist,
   WAL, and allocated bytes are reported separately; no action claims forensic erasure
   from backups or exports.

17. **Prove linear growth with the real gateway.** A semi-incompressible 200-turn
    production-shaped benchmark records normalized rows and bytes, legacy bytes,
    database/freelist/WAL size, and settlement costs. A second fixture repeatedly
    replaces 75 percent of the surface. Second-half trace bytes and rows may be at most
    1.25 times first-half growth; the pinned append-only fixture is capped at 2.0 MiB of
    trace-owned live bytes at 200 turns. Reservation p95 is capped at 10 ms, settlement
    p95 at 25 ms, and migration write batches at 100 ms. No normalized call or
    replacement may contain a list or blob proportional to prior transcript length.

    Latency-critical reservation, `dispatch_started`, and `response_started` writes
    temporarily disable automatic WAL checkpoints on their thread-local connection and
    restore the caller's exact setting in `finally`. This does not change SQLite
    durability or the connection default. Terminal settlement remains on the bounded
    off-UI persistence worker under the caller/default checkpoint policy, so it, a
    later ordinary commit, or connection close remains an explicit checkpoint owner.
    The reference benchmark reports phase WAL allocation and close cost as well as the
    timed samples; long-reader coverage proves the scoped policy does not retune another
    connection or hide terminal checkpoint behavior.

18. **Migrate disclosure settings conservatively and stage expensive features.** Old
   capture enablement maps to Capture On/Off, but old Safe/Full capture detail remains
   historical provenance and every upgraded profile starts with the Safe viewer. Full
   requires a new explicit viewer choice. Core ledger capture, mandatory filtering, and
   logical normalization must prove their gates before custom-regex execution or physical
   compaction is enabled.

19. **Keep token-chunk packing separate.** Raw token-level event capture and lossless
    chunk-row encoding are deferred to [TASK-24206](../tasks/task-24206%20-%20Add-lossless-chunk-row-encoding-for-streamed-trace-events.md)
    and are not required by the forthcoming ADR-097 implementation umbrella.

## Consequences

- Safe remains diagnostically useful because the stored trace can explain provider-
  visible context without repeating the transcript. Safe and Full differ in disclosure,
  not historical availability.
- Automatic project-instruction and injected-context bodies may be stored once under
  the default capture path after mandatory filtering and optional PII masking. They
  remain excluded from ordinary transcripts, metadata, logs, permission decisions, and
  authority grants.
- The normalized data model is larger than ADR-096's bounded-excerpt patch, but it
  resolves future Full growth, message-edit fidelity, fork coherence, and legacy
  reclamation with one ownership system instead of parallel exceptions.
- Current messages remain unduplicated until an edit or deletion makes one historical
  copy necessary. Repeated calls and forks reuse that copy.
- Capture On adds one small synchronous reservation write before each provider call.
  Users may explicitly bypass a reservation failure for one send; that choice is shown
  in live run/UI state but, consistently with Capture Off, is not guaranteed a durable
  trace record.
- A trace with intentional masking or legacy omission is structurally reconstructable
  but not content-complete, and the UI must say so.
- Source hard deletion may fail safely when required historical materialization cannot
  complete.
- Shared fork prefixes mean purging one conversation may not reclaim bytes retained by
  another owner; the UI reports remaining forks.
- Optional PII masking cannot remove PII from the canonical conversation and does not
  retroactively rewrite legacy traces without explicit later work.
- Legacy logical reclamation occurs in the background; allocated database bytes may
  remain until automatic physical compaction is admitted.
- TASK-23026 remains the completed historical record of ADR-096's implementation.
  Implementation requires a new umbrella task and multiple dependency-ordered Backlog
  work packages. The design spec defines their boundaries; exact IDs are created
  during implementation planning.

## Alternatives considered

### Retain a fixed Safe excerpt

Rejected. It discards causal context, leaves Full quadratic, and makes debugging depend
on choosing Full before the failure occurs.

### Store one fingerprint per omitted message

Rejected. Reference metadata still grows quadratically across calls and hashes can
confirm guesses about private content.

### Keep compressed full-request blobs

Rejected. Compression reduces some bytes but preserves repeated logical ownership and
does not solve edits, forks, deletion, or exact component reuse.

### Store arbitrary JSON deltas

Rejected. A custom patch language duplicates representation and introduces brittle
diff/apply/fallback behavior. Typed surface events and immutable references retain one
logical model.

### Replace Chatbook conversations with a wholly event-sourced session system

Rejected. Existing messages, variants, sync, and conversation ownership remain useful.
The trace ledger references them and event-sources only provider-visible historical
projection.

### Copy the trace prefix on fork

Rejected. It recreates the storage defect for fork-heavy workflows and complicates
deletion. Immutable shared prefixes provide coherent reads without copying.

### Run user regex in-process

Rejected. Validation cannot eliminate every catastrophic-backtracking pattern and
CPython `re` cannot be portably interrupted. A killable subprocess provides a hard
runtime boundary without making a transitive regex engine a core dependency.

## Links

- [Approved design](../../Docs/superpowers/specs/2026-08-28-console-reference-backed-semantic-trace-ledger-design.md)
- [TASK-23026](../tasks/task-23026%20-%20Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md)
- [Superseded ADR-096](096-console-safe-capture-retention.md)
- [Console chat fork ADR](092-console-chat-fork-copy-and-authority-boundary.md)
- [TASK-24206](../tasks/task-24206%20-%20Add-lossless-chunk-row-encoding-for-streamed-trace-events.md)
- [DeepSeek Harness session model](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.md)
- [DeepSeek reconstructable requests](https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-07-05-reconstructable-requests.md)
