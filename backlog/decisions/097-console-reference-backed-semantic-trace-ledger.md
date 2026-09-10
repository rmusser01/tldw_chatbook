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

   Agent run identity binds the stable opaque actor/chain pair carried by route
   provenance, not the chain alone (PR2433 review, 2026-09-05). The existing
   `run_id` stores `actor_uuid:chain_uuid`; both inputs are canonical UUIDv4
   values, so this is an unambiguous content-free identity. No new schema,
   content hash, or process-local ownership registry is introduced. Historical
   chain-only runs remain readable, but cannot authorize a new continuation.

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

### Clarification recorded 2026-09-07: automatic project context

TASK-31976.1 applies decisions 6 and 8 to automatic `AGENTS.md` context: Capture
On retains the actual provider-visible context as a `project_instruction`
artifact under the frozen credential/PII policy. The owner explicitly requested
this trajectory fidelity, superseding ADR-069's earlier durable-capture
exclusion; its other ephemerality and authority rules remain in force. Capture
Off retains no such trace artifact. Context rows remain attached to their user
turn for windowing and cannot replace the admitted saved message as call owner.

When a completed turn is followed by another send, its bounded project/tool
suffix may be replaced by the saved assistant, new saved user and current project
context. This requires a witnessed terminal response link, unchanged prefix,
same disclosure policy and durable attribution of every removed artifact to
that completed run; it is revalidated when binding and works from durable
references after a restart. A matching context suffix is still renewed with the
new turn. Prior call reconstruction remains unchanged. Other replacement shapes
remain rejected. A completed llama.cpp fallback can witness that response only
when it immediately follows the failed streaming call with the same owner,
turn, disclosure policy, surface, provider and model; removed artifacts must
still belong to the verified source stream run. Artifact equality compares the frozen credential/PII projection
without modifying the actual provider input.

### Amendment recorded 2026-09-08: per-call rendered system rows

TASK-32048 reproduces a llama.cpp discovery run that reaches two provider calls,
then fails with `unsupported_surface_change` after `load_tools`. Its fenced tool
protocol changes the leading system row while new tool traffic is appended.
The following saved send can also change that row while replacing the completed
tool suffix. Treating the system row as ordinary shared history cannot represent
both legitimate changes with the existing bounded surface operation.

Retain the immutable leading `rendered_system` artifact slot and record its
current value in the call header as a `rendered_system_row` component. This is
limited to the same first message slot with direct rendered-system artifact
provenance and a system-role value. Saved system revisions, moved slots and
other history remain subject to exact reference/value matching. Preparation and
atomic dispatch binding verify slot identity, incoming frozen policy and the
exact final sanitized row; raw provider values remain disposable. Reconstruction
applies the header component only to that eligible slot. Older headers keep
their existing reads, and earlier calls and artifact bytes never change.

This uses the existing header artifact ownership, deduplication and collection
paths. There is no schema migration, new retention policy, transcript-sized
header or additional surface replacement permission. The provider receives the
original message sequence and content.

### Clarification recorded 2026-09-05: retained soft-delete envelopes

The accepted design's mutation-boundary section and TASK-23113.2 AC8 specify
that soft deletion changes visibility/ownership while retaining semantic bytes.
For that operation, this supersedes ADR-090's older deletion-sidecar clearing
amendment: thinking and continuation remain with the retained message envelope,
not as active transcript or replay content. Hard deletion and semantic edits
still use the preservation/mutation boundary above. This clarification records
the already-implemented contract; it does not add retention or disclosure rights.
It resolves the stale tombstone expectation discovered during TASK-31232's
baseline repair. See the [accepted mutation boundary](../../Docs/superpowers/specs/2026-08-28-console-reference-backed-semantic-trace-ledger-design.md#semantic-message-revisions)
and [TASK-23113.2](../tasks/task-23113.2%20-%20Enforce-semantic-revisions-for-every-model-visible-mutation.md).

### Amendment recorded 2026-09-05: completed tool-turn surface transition

TASK-31742's real production-factory integration probes exposed a missing composed
operation: a completed run's bounded tool suffix must become the exact saved
assistant revision while the next saved user is appended. The owner approved the
focused repair and its review corrections before implementation.

Permit one explicitly typed replacement-plus-append for eligible `AGENT_FIRST`
and `FRESH` next sends, not arbitrary multi-item replacement. The same attached
owner, lineage, frozen policy, completed prior run, exact response-revision link,
unchanged prefix and bounded active tool suffix must be proven. Response linkage
alone is never authority. Recheck the predecessor and call witness when persisting
both operations, header and dispatch binding in the existing atomic transaction;
historical call heads remain unchanged. Keep the 256-node range limit, existing
schema, disclosure rules, and growth/latency gates.

Across saved turns, policy agreement compares every persisted disclosure setting
(credential-filter version, PII enabled state, and exact ruleset revision), with
both records required. Accepted turns allocate fresh opaque policy IDs even for
identical settings. Exact prior-run policy identity and exact incoming reservation
and retry identity remain required; historical artifacts are never relabeled.
This clarification follows the real-controller regression and does not admit
changed disclosure settings or replace final-value verification.

Pre-dispatch Retry must prove and reuse its exact still-unbound reservation via
accepted-turn recovery ownership. Never skip unrelated reservations or revive a
terminal call. A write exception is not proof of rollback: reconcile the exact
call and expected head/header to committed, rolled-back or unknown. Preserve
committed dispatch state, invalidate stale capabilities, and allow no automatic
redispatch on uncertainty. Only the original live invocation retaining its exact
unconsumed gateway entry grant may proceed after a proven commit.

The [approved repair contract](../../Docs/superpowers/specs/2026-09-05-console-tool-turn-surface-transition-design.md)
defines exclusions and recovery proofs; the
[implementation plan](../../Docs/superpowers/plans/2026-09-05-console-tool-turn-surface-transition.md)
defines staged verification. This amendment authorizes the contract, not a claim
that implementation or merge verification is complete. It adds no Canvas
privileges, dependency, synchronization contract or new persistence registry.

### Amendment recorded 2026-09-08: current-turn transformed request ownership

TASK-32032 reproduced a Capture-On dictionary send failing before provider entry:
the admitted current user text was transformed after persistence, so its request
row became an unowned artifact. Preserve that row as a typed current-turn
transform with exactly one admitted saved-revision source and one policy-owned
`active_request` artifact. The artifact retains the transformed provider value;
the source remains the exact saved value. Only the accepted current user owner
from the pre-transform snapshot may receive this descriptor. Changed historical
rows, unknown owners, and changes outside the current text field gain no authority.

Record the factory's exact current revision on the existing
`call_boundary.semantic_revision_id` foreign key, atomically with reservation.
Schema v69 permits that reference on call boundaries and verifies that its source
message and conversation own the call. Older NULL boundaries stay readable and
do not gain a guessed source. Existing event-reference reachability rules govern
copy-on-write, archival and collection; no new ownership registry is introduced.
A tool-loop continuation must retain its origin's exact pinned revision; the
same message identity and equal artifact bytes cannot substitute a newer source
revision. Legacy NULL boundaries grant no transformed-source continuation proof.

Extend the existing completed-turn witness with an optional exact transformed
source revision. An eligible next fresh/agent send may replace the prior bounded
current-request suffix with its exact saved source and append the verified saved
assistant plus the next admitted user. A next user may itself carry the same
typed transform. The source must equal the original call-boundary pin, and the
removed source artifact must belong to that origin call. Any removed tool suffix
must satisfy the existing same-run tool proof. Require the latest eligible terminal call,
unchanged prefix, exact tail, attached owner, bounded range, verified assistant
response link, exact saved values, and equal persisted disclosure settings.
Recheck the full witness during final persistence and owned pre-dispatch recovery.
The replacement, appends, header and dispatch binding remain one transaction;
historical call heads retain their original transformed values.
An unchanged continuation suffix may follow the current user in physical node
order. Locate the original last active message at its exact call head, and replace
only the proven contiguous message range; preserve the continuation domain and
its values. An unchanged continuation inside a proposed replacement range still
makes that range ineligible.

If the preceding transformed run ended ERROR, STOPPED or INTERRUPTED and the
next request has no assistant row for it, the same bounded witness may restore
only the exact pinned user source and append the next user. Require a settled
terminal call and the same owner, tail, range, source, policy and tool lineage
proofs. Do not invent a response link or an assistant row. Pending, reserved,
dispatch-open, unknown-outcome and superseded calls remain ineligible. A real
controller regression establishes the durable ERROR outcome before testing its
successor; provider-shaped error text is not evidence of terminal failure.
When a stopped/failed run has a saved partial assistant in the next request, that
row requires the same exact verified response-revision link and value comparison
as a completed response. Terminal state alone never authorizes assistant content.
A real streaming-controller Stop regression verifies the saved partial answer,
durable STOPPED outcome and exact response link before a cold-factory next send.

The source pin retains revision identity only. It creates no new policy binding
or retained source body. If a later operation needs a retired source projection
that existing policy retention cannot supply, that operation remains unavailable.

The current-turn source proof and automatic-project-context proof compose over
one bounded replacement. The exact pinned user source precedes its verified saved
assistant (when present), the next admitted user and only the declared current
project-context rows. Each removed source/context/tool artifact must still belong
to the proven origin or tool chain; changed or disabled project context never
replaces the saved call owner. An unchanged continuation suffix remains outside
that message range. Warm/cold controller tests cover ordinary, tool and llama.cpp
fallback predecessors with current-user dictionary transforms and context renewal.
This integration preserves both proofs and adds no new ownership authority.

Selected saved continuation values cross the trace service as canonical V1 JSON.
The service issues an immutable tuple of strictly parsed checkpoint objects for
provider dispatch; verification still requires that exact issued tuple and checks
the canonical value against its saved source. This preserves the existing provider
contract without asking the JSON sanitizer to accept runtime dataclasses.
Retained saved continuation artifacts are rehydrated from that exact live saved
source for dispatch, then compared under their frozen artifact masks. Compare
the raw supplied checkpoint before filtering: equal redacted bytes cannot grant
source ownership. Provider-only attachment owners retain exact artifact-value
comparison and receive no saved-source normalization authority.
When selected displayable thinking actually changes a serialized message, its
existing MESSAGE_REWRITE descriptor carries a policy-owned thinking artifact of
the exact final provider message, plus the original owner and typed attachments.
An unchanged message keeps its original descriptor, including continuation-only
attachment cases. A rendered thinking message is not an exact raw saved revision.
These artifacts use existing disclosure and retention policy; no reader-side
guessing, new retained canonical bodies or new ownership storage is introduced.

For newly captured supported typed-thinking responses, freeze an explicit
versioned response projection in the existing request-header defaults. Settlement
and the native reader use that same profile to project the exact saved visible
answer and strict terminal ThinkingEnvelope. Displayable blocks project their
exact text and provenance; proprietary blocks project only their content-free
provider evidence. Only consecutive displayable fragments
with identical provider, model, protocol and source-format metadata coalesce;
literal tags are never parsed by the trace ledger. Old headers without this
profile retain their previous visible-message response projection. Unsupported,
malformed, differently owned or mismatched thinking remains artifact-backed.

Before filtering the response, the existing ephemeral settlement handoff records
a process-keyed HMAC of its canonical semantic value, hidden from representations.
The coordinator compares this proof to the exact saved revision before creating
a new thinking-response revision link. The proof and process key never enter the
ledger, diagnostics or exported captures; the existing bounded queue retains
credential-filtered response bytes and this opaque process-local proof, not raw response
bodies. Equal credential masks alone cannot establish response-source equality.
Frozen PII masks use the existing source-span records. Typed-response paths have
an explicit response-profile prefix, alongside the ordinary visible-message
paths, so each projection consumes only its own mask domain. Copy-on-write retains
the same canonical envelope through the response link's existing policy reachability.
A new handoff cannot establish raw equality from a retired masked body; that
terminal retry remains unavailable while the original linked capture stays readable.

Every newly stored response applies its frozen PII policy, including verified
visible-message links and artifact fallbacks without a response profile. If the
detector is unavailable, persist a bounded content-free omission. Terminal artifact
retries compare the same policy projection; existing stored captures are not rewritten.

Rejected alternatives: treating transformed text as an exact saved revision,
disabling capture after failure, accepting arbitrary historical replacements,
or inferring the source from the current message version or timestamp. A call's
turn identifies a message but not its historical revision; response links identify
the assistant, and artifact nodes do not persist transient derived inputs. A
creation-version-only restriction would exclude supported edited/retry inputs.
The existing nullable event FK is therefore the smallest precise durable source
contract. This amendment does not relax other routes or artifact transforms.

### Amendment recorded 2026-09-05: owner-approved latency reference replacement

For TASK-31742, the owner explicitly approved replacing the latency reference
machine with the current host after a direct hardware check: Mac17,6, Apple M5
Max, 18 logical CPUs, 137438953472 bytes (128 GiB). Fixture version 5 identifies
this host as `tldw-mac17-6-m5-max-18c-128gb-apfs-v5`, replacing the M4 Pro
14-core/48-GiB version 4 reference. Keep arm64/APFS, CPython 3.12.11, SQLite
3.49.1, sample counts/order, SQLite settings, checkpoint policy and all numerical
thresholds unchanged. Non-reference hosts still fail release qualification;
correctness-only opt-in is not release evidence.

This is an explicit reference-platform change, not proof of equivalent performance
on the old hardware. Retain the new raw benchmark artifact and report its exact
environment and whether thresholds were applied. The previous M5 diagnostic run
against version 4 remains non-reference evidence and is not retroactively promoted.
Retaining the old reference would require another machine; silently bypassing
identity checks or raising thresholds was rejected. This changes no runtime or
storage contract and does not waive current-head integration/PR gates.

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

### Amendment recorded 2026-09-08: explicit discard after a tool-run capture failure

TASK-32075 reproduces a third-request construction failure leaving previous calls
at RESPONSE_STARTED. Reopening the conversation and explicitly discarding its
pending assistant commits `assistant_generation_state=discarded` and removes the
active dispatch checkpoint, but the previous tool/context surface remains.
A following captured send may replace that bounded suffix using an optional exact
discarded-assistant owner in the existing completed-turn witness. Recheck that the
current saved user is a direct child of that live discarded assistant, the assistant
is a direct child of the prior call's saved user in the attached conversation, and
no active dispatch checkpoint remains for that prior user. Multiple assistant
children (including soft-deleted siblings) make run ownership ambiguous and remain
ineligible. The prior user's source
must match the exact origin call and incoming retained history. The latest prior
call must be response-bearing (RESPONSE_STARTED or a settled terminal outcome);
dispatch-open/unknown calls remain ineligible.

This explicit durable discard substitutes only for the completed assistant response
proof. It does not synthesize an answer, a response link, or a successful trace
outcome. Retain all existing owner, policy, tail, source, bounded tool/context range,
and same-run lineage checks, and repeat the discard proof during final persistence
and owned reservation recovery. Transformed sources retain their exact call-boundary
pin. Historical call records and request heads remain unchanged. No schema or new
ownership registry is needed. An absent, deleted, changed, unrelated or still-active
response owner grants no replacement authority.

Already-failed follow-up sends can have committed additional user messages
without a provider call. Extend the same witness with a bounded ordered tuple of
(saved user revision, discarded assistant message ID) pairs. Each pair must be an
exact saved-user descriptor in the incoming replacement, and the direct-parent
and unique-assistant proof must link the original traced user through every pair
to the current admitted user. Validate all saved values and recheck the full chain
at final binding. Missing, active, changed, cyclic, ambiguous or over-limit chains
remain ineligible. This adds only bounded revision/owner identities to the existing
in-memory witness; it does not store transcript copies or modify historical calls.

### Amendment recorded 2026-09-09: closed unanswered runs and untraced follow-ups

TASK-32197 reproduces a request-construction failure after successful tool calls:
the individual trace calls settle COMPLETE while the saved assistant settles FAILED
with an empty semantic envelope and no dispatch checkpoint. That failed owner has
no pending response to discard. Permit this exact durable closure as an alternative
to explicit discard in the existing bounded tool-turn replacement witness. Require
one undeleted assistant child, empty text and no image, attachment, thinking or
provider-continuation sidecars, no active checkpoint, and a settled response-bearing
prior call. Failed state alone never supplies an answer or successful run outcome.

The user may have continued with Capture Off before returning to Capture On. The
same witness may carry a bounded exact saved-parent chain through intervening closed
unanswered turns and complete untraced user/assistant pairs. Require live exact saved
revisions for every provider-visible intervening row, unique undeleted parent owners,
no intervening captured calls or active dispatch checkpoints, and the same strict
sidecar exclusions for the newly admitted closure/history cases. These rows become
ordinary saved history of the incoming request; do not fabricate captures, outcomes
or response links for their earlier untraced delivery. Retain the prior explicit
Discard contract and its existing descriptor representation. When a completed
uncaptured follow-up requires the closed representation, re-read the original
owner's durable Discard state at every validation, including final binding, before
allowing its existing RESPONSE_STARTED evidence. Failed-empty owners still require
settled response-bearing calls.

Recheck the full closure and chain at final dispatch binding, along with the existing
attached owner, latest settled call, exact source, unchanged prefix and tail, matching
disclosure settings, same-run tool/project artifact ownership and 256-node bound.
Partial failed answers, active or uncertain delivery, changed values, ambiguous or
deleted siblings, gaps, unknown owners and over-limit chains remain ineligible.
Historical calls, request heads and reconstructed requests stay immutable. This adds
only bounded transient identity evidence to the existing witness; no schema, durable
registry, retention change or general history-replacement permission is introduced.
