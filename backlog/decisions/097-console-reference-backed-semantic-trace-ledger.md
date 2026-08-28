# ADR-097: Use a reference-backed semantic trace ledger

Status: Proposed

Date: 2026-08-28

Related Task: [TASK-23026](../tasks/task-23026%20-%20Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md)

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
   once per unique disclosure policy in the same transaction. Failure aborts the edit
   or deletion.

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
   attachment rules.

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
    and context compaction append model-surface replacements citing shadowed source
    events. The viewer may inspect, filter, search permitted projections, copy, export,
    or purge ownership; it never edits historical trace.

11. **Separate capture, PII, and viewer controls.** Capture On/Off and optional PII
    redaction resolve at global, conversation, and eligible next-send scope and freeze
    for a run. PII defaults Off. Safe/Full is a local viewer/export disclosure profile
    over the same stored trace, not a different at-rest history. Forks inherit future
    settings while historical calls retain their frozen provenance.

12. **Filter credentials mandatorily and fail closed.** Known credential fields,
    credential-bearing URL components, nested credential fields, and recognized secret
    formats in provider-only text are removed before trace persistence. Arbitrary prose
    secrets cannot be guaranteed detectable and the UI says so. Sanitizer failure
    creates a content-free unavailable marker; no raw fallback is stored.

13. **Offer irreversible trace PII masking.** Built-in detectors and validated user-
    authored regex rules produce immutable Unicode-span projections. Custom regex runs
    in a bounded killable subprocess because CPython `re` has no portable hard timeout.
    Historical projections retain detector/rule IDs and a ruleset fingerprint, never
    matched values or regex source. PII masking protects traces but does not silently
    rewrite canonical conversation messages.

14. **Normalize legacy captures automatically and resumably.** A fast schema step
    enables normalized writes and dual reads. After UI readiness, idle bounded batches
    split legacy blobs into revision references and deduplicated provider-only
    artifacts. A blob is deleted only after reading back the normalized call and
    reproducing its sanitized legacy projection. Ambiguous rows remain individual
    legacy artifacts. ADR-096 aggregate markers become explicit irreversible legacy
    omissions.

15. **Keep capture best-effort and independently idempotent.** Trace settlement cannot
    roll back a provider result or assistant message. Failed writes enter a bounded
    best-effort settlement queue; process death may leave an honest gap or interrupted
    call rather than invented content.

16. **Use shared-owner garbage collection and honest physical maintenance.** Deletion
    detaches one conversation root. Background mark/sweep reclaims unreachable objects
    only after an ownership-epoch recheck. SQLite physical compaction runs automatically
    at a later eligible visible idle pause with connection closure, WAL checkpoint, disk
    preflight, and maintenance lease. Logical, freelist, WAL, and allocated bytes are
    reported separately; no action claims forensic erasure from backups or exports.

17. **Prove linear growth with the real gateway.** A semi-incompressible 200-turn
    production-shaped benchmark records normalized rows and bytes, legacy bytes,
    database/freelist/WAL size, and settlement costs. No normalized call may contain a
    list or blob proportional to prior transcript length.

18. **Keep token-chunk packing separate.** Raw token-level event capture and lossless
    chunk-row encoding are deferred to [TASK-23112](../tasks/task-23112%20-%20Add-lossless-chunk-row-encoding-for-streamed-trace-events.md)
    and are not required to close TASK-23026.

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
- Implementation requires multiple dependency-ordered Backlog work packages under
  TASK-23026. The approved spec defines their boundaries; exact IDs are created during
  implementation planning.

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
- [TASK-23112](../tasks/task-23112%20-%20Add-lossless-chunk-row-encoding-for-streamed-trace-events.md)
- [DeepSeek Harness session model](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.md)
- [DeepSeek reconstructable requests](https://github.com/deepseek-ai/deepseek-harness/blob/master/.agents/notes/implemented/architecture/2026-07-05-reconstructable-requests.md)
