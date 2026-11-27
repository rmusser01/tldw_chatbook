# ADR-337: Use a reference-backed semantic trace ledger

Status: Accepted

Date: 2026-08-28

Originating Task: [TASK-23026](../tasks/task-23026%20-%20Exchange-capture-stores-the-whole-conversation-on-every-send-forever.md), completed by the superseded bounded-retention implementation

Related Spec: [Console Reference-Backed Semantic Trace Ledger](../../Docs/superpowers/specs/2026-08-28-console-reference-backed-semantic-trace-ledger-design.md)

Supersedes: [ADR-096](096-console-safe-capture-retention.md), which bounded Safe
captures by discarding older semantic history.

Amends: the Console Full semantic-capture policy and
[ADR-334 Console chat fork](334-console-chat-fork-copy-and-authority-boundary.md).

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
    promotes their in-memory lineage. (Amended 2026-10-29, TASK-33621.17, owner
    decision 2026-10-27: a temporary conversation resolves Capture Off unless the user
    turns Capture On for it, so its sends never pause by default and Save & Send is
    that opt-in's choice; saving the chat hands Capture back to the conversation's
    own Capture choice, else the global setting, and a pending one-time next-send
    choice still decides that one send (TASK-37811).
    The default is resolved, never stored with the saved conversation.) Before
    dispatch, a component sanitization or
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
    and are not required by the forthcoming ADR-337 implementation umbrella.

### Clarification recorded 2026-09-07: automatic project context

TASK-31976.1 applies decisions 6 and 8 to automatic `AGENTS.md` context: Capture
On retains the actual provider-visible context as a `project_instruction`
artifact under the frozen credential/PII policy. The owner explicitly requested
this trajectory fidelity, superseding ADR-317's earlier durable-capture
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
- [Console chat fork ADR](334-console-chat-fork-copy-and-authority-boundary.md)
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
settled response-bearing calls. (Relaxed 2026-11-24 by TASK-42534, owner-approved:
the terminal call may instead be a settled ERROR that never started a response;
every removed tool-loop node must then follow a predecessor in the same run that
bore a response (RESPONSE_STARTED, COMPLETE, INTERRUPTED with a response start,
or a failed stream whose llama.cpp resend bore one); nothing is relabeled; and
the failed turn is cut from the next request, not sent. See "Amendment recorded
2026-11-24 (owner-approved): a closed run whose calls failed before any
response".)

Recheck the full closure and chain at final dispatch binding, along with the existing
attached owner, latest settled call, exact source, unchanged prefix and tail, matching
disclosure settings, same-run tool/project artifact ownership and 256-node bound.
Partial failed answers, active or uncertain delivery, changed values, ambiguous or
deleted siblings, gaps, unknown owners and over-limit chains remain ineligible.
Historical calls, request heads and reconstructed requests stay immutable. This adds
only bounded transient identity evidence to the existing witness; no schema, durable
registry, retention change or general history-replacement permission is introduced.

### Integration recorded 2026-09-10: local terminal receipts

The closed-history proof composes with ADR-336 terminal publication. In addition
to its existing empty metadata forms, it permits exactly the serialized
`MessageMetadata` defaults plus one canonical, validated terminal receipt ID.
Compare JSON field types as well as values; numeric zero is not a false boolean.
The receipt is local control metadata, not provider content or delivery evidence.
Nonempty provider, engine, model, media, unknown fields and malformed receipts
remain ineligible. All parent, revision, sidecar, checkpoint and final-binding
revalidation requirements above remain unchanged.

### Integration recorded 2026-10-04: failed intermediate model calls

A provider-call failure with no captured response is settled as a trace-owned
terminal ERROR before the agent can retry. It cannot await the overall assistant
message's terminal save: that message also owns the later successful retry.
Successful and partial response handoffs retain canonical message linking.

If sealing the failed attempt cannot complete, the sanitized handoff remains
owned through cancellation and enters the existing store settlement custody.
Its canonical assistant ID remains absent even when the overall assistant is
later saved: changing that ID would change the pending settlement fingerprint
and would incorrectly attach the successful retry's answer to the failed call.

The existing TOOL_LOOP route includes subsequent model attempts within a run.
After a durable ERROR it admits only an unchanged request surface, under the
same attached owner, actor/chain, turn, source pin, frozen policy and latest
call-boundary proof. The failed call must be settled and provider-inactive.
Compare rendered system content against the immediately preceding call in
both distinct-role and single-preamble wire formats. The final header comparison
runs in the dispatch-binding transaction before capability promotion; changed
system content cannot bypass unchanged-message checks. Provider, model, endpoint,
generation parameters, response format, reasoning controls, tool schemas and
literal provider envelope must also match the preceding failed attempt. The
AGENT_FIRST to TOOL_LOOP route transition is allowed; it grants no target change.
(A provider fallback is not such a retry: it starts a new chain, amendment
2026-11-24, TASK-42525.)
An appended or replaced surface, unknown/open dispatch, stale or foreign chain,
or unsaved settlement remains ineligible. Each retry reserves a distinct ordered
call; it never rewrites the failed call or relabels it as a tool response.


### Amendment recorded 2026-11-24 (owner-approved): a closed run whose calls failed before any response (TASK-42534)

The owner approved this relaxation on 2026-11-24.

The 2026-10-04 integration lets a run retry a call that failed with no
captured response. When the retries fail too, the run's assistant settles
Failed and empty, and its terminal call is a settled ERROR that never started a
response. The 2026-09-09 closed-run witness required a response-bearing
terminal, so the next captured send was refused (`completed_tool_turn_unavailable`)
whenever its request replaced the run's suffix: project context, or tool
history from earlier in the run. A failed call authorizes an unchanged retry,
never tool history, so it adds nothing the witness must vouch for, and the
witness may close the turn on it. In exchange, the witness now checks directly
that every removed tool-loop node was appended by a call whose predecessor in
the run bore a response: RESPONSE_STARTED, COMPLETE, or INTERRUPTED with a
response start (startup recovery settles an open response as INTERRUPTED, the
only way an existing call reaches that state), or a failed stream whose resend
bore one. A closed terminal that completed or stopped without a response, and
every check above, are unchanged. No call is relabeled as response-bearing and
no response link is created. The failed turn's empty assistant is cut from the
next request, as for any closed turn; it is not sent.

### Integration recorded 2026-11-24: stream resends inside a run (TASK-42530, TASK-42532)

A llama.cpp stream that returns no content is resent without streaming on
its own `llama_fallback` run (task-19324). That resend is compared at dispatch
binding like a retry, with the call whose boundary directly precedes its own
(same owner, segment, turn, policy and surface, in ERROR): provider, model,
endpoint, generation settings, response format, reasoning controls, system
composition, rendered system parts, tool schemas and literal envelope must
match, except the overlay-proven streaming flip, which the envelope's `stream`
key carries too (the stream's envelope must read `"stream": true` and equal the
resend's with that turned false; a missing envelope refuses). TASK-42472 let a failed resend stand aside for the agent's
unchanged retry of the streamed call. A resend that bore a response is that
call's answer instead: the next TOOL_LOOP call of the run is admitted after it
as the step that carries its reply and tool results, and is refused if it
appends nothing. A resend of a tool-loop call takes the loop's turn from the
request, and is admitted only directly after the settled failed tool-loop call
it repeats, with nothing appended. The completed-turn witness counts a failed
or answering resend directly after a call of the run as part of that run, and
a resend that ends the turn is witnessed from the run origin of the call it
resent, with or without project context.

A self-hosted or Custom endpoint's refused stream, and a llama.cpp stream
with tools in the request, are resent as the run's next TOOL_LOOP call. When
that resend fails, an in-place retry that streams again is compared with the
streamed call the resend repeated, not with the resend, when that call failed
before any response and its stored settings equal the resend's apart from the
streaming flag (the resend was admitted only as that verified True -> False
flip). Every other target, setting and surface change remains refused.

### Amendment recorded 2026-11-24 (owner-approved): a provider fallback starts a new chain (TASK-42525)

The owner approved this on 2026-11-24 ("2 ok", approving the task's
recommendation). An agent's provider fallback (ADR-110) goes to another
provider, model or endpoint, and may re-encode the run's history for that
provider's tool protocol. It is not a retry, so the 2026-10-04 rule ("a failed
call authorizes an unchanged retry") refused it at dispatch, and a captured run
planned no fallback at all. Under Capture On a fallback now starts a new chain
of the same run. The retry rule inside a chain is unchanged.

1. **Identity.** A run's chains share its actor. The fallback mints a new
   chain, so its calls are run `actor:new_chain`. Its first call is that
   chain's sequence 0, on the TOOL_LOOP route: it is the run's next model call
   and carries the run's history. That shared actor, plus the call boundary
   order below, is the link to the failed chain. It is content-free and needs
   no schema.
2. **Who mints a chain.** Only the Console agent adapter of a primary
   Capture-On run. It does so when the runtime's call names a target other
   than its chain's, and only if that target's endpoint is an entry of the
   run's configured fallback list after the last entry a fallback used. Any
   such later entry qualifies, in list order, not only the next one: the
   runtime skips an entry that is not ready, and an entry passed over is spent.
   Each entry is used at most once, and the list is never walked back. Any
   other target change is refused before preparation, so nothing is reserved.
   A spawned child's preset fallback follows the 2026-11-26 amendment
   below. (Reworded
   2026-11-25 to match the adapter; the rule is unchanged.)
3. **Admitting the chain's first call** (in the reservation transaction, so a
   refusal rolls back and leaves no reserved call). The call whose boundary
   directly precedes it must be a call of another chain of the same actor. If
   that call is a failed llama.cpp resend, the streamed call it resent counts
   instead. That failed call must:
   - be a settled ERROR that never started a response, with no answering
     resend;
   - have the same owner, segment, turn and frozen policy;
   - still hold the surface tail.

   The new call must append and replace nothing: the run's history on the
   surface is what its earlier chains earned. Its provider or model must differ
   from the failed call's, because a call to the same target is a retry and
   stays in its chain. Its rendered system row, tool schemas and generation
   settings belong to its own header and are not compared with the failed
   call's.
4. **The run's origin** is the AGENT_FIRST call. A fallback chain reaches it
   through these links, followed back through earlier fallbacks. Every TOOL_LOOP
   origin check (owner, segment, turn, policy, source pin) applies to it as
   before. Later calls in the new chain are ordinary TOOL_LOOP calls of that
   chain: an in-place retry after one of them fails is compared with it, under
   the unchanged retry rule.
5. **Completed-turn witness.** A run is its AGENT_FIRST chain plus the fallback
   chains linked from it. The witness's origin is the AGENT_FIRST call. Every
   removed node must have been appended by a tool-loop call (sequence 1 or
   later) of one of those chains, and that call's predecessor in its own chain
   must have borne a response (TASK-42534). Every call boundary between origin
   and terminal must belong to one of those chains, or be a llama.cpp resend of
   one of their calls. A chain's first call appends nothing, so it never vouches
   for tool history.

Still refused:
- A fallback whose history projection rewrites rows already on the surface
  (native and fenced tool protocols, after tool activity). It needs a
  replacement, which this amendment does not grant. (Granted by the next
  amendment.)
- A fallback between providers that place the system prompt differently
  (llama.cpp sends it as the first message, a hosted provider does not). The
  new call does not continue the surface even before any tool call. (Granted
  by the next amendment.)
- A spawned child's preset fallback. (Granted by the 2026-11-26 amendment
  "a spawned child's preset fallback starts a new chain" below.)

### Amendment recorded 2026-11-26 (owner-approved): a spawned child's preset fallback starts a new chain (TASK-42564)

The owner approved this on 2026-11-26 ("Yes, new chain (Recommended)"):
"New ADR-337 amendment. The sub-agent's fallback is sent and traced as a new
chain. It only happens before any tool call, so no history rewrite is
needed." A spawned child's preset fallback (ADR-200) switches to the
preset's next `fallback_models` target before the child's first tool batch.
The amendment above refused it, and TASK-42556 made that refusal a visible
skip before any call. Under Capture On it now starts a new chain of the
child's run. The child amendment (2026-11-17) and the primary rules above
apply, with these differences:

1. **Identity and segment.** A child's chains share its actor. The fallback
   mints a new chain, so its calls are run `actor:new_chain`. The schema
   (v87) ties a child segment to the calls of one run identity, so the new
   chain opens a child segment of its own. It is linked, like the first
   one, to the same owner and spawning turn. The shared actor is the link
   between the chains. No schema change.
2. **Who mints a chain.** Only the Console agent adapter, for a child's call
   that names a target other than its chain's. The child's run must have
   durably selected that target as its active preset fallback: ADR-200
   persists the selected index before dispatch, and the index only rises.
   The index must also be later than the one the child's current chain uses.
   So only the preset's `fallback_models` entries qualify, once each, in list
   order. Any other target change is refused before preparation, so nothing
   is reserved. The new chain's first call is its segment's AGENT_FIRST, at
   sequence 0.
3. **Admitting the chain's first call** (in the reservation transaction, so
   a refusal rolls back and nothing is reserved). Every other chain of the
   child's actor must:
   - belong to the same owner and turn;
   - have no call that started a response (so no tool call came first, and
     no rewrite is needed).

   The latest of those calls, in insertion order, must be a settled ERROR
   with the same frozen policy, to another provider or model than the new
   call. Only that call is compared with the new target, as for a primary
   fallback: an earlier chain may have used it, so a preset `[B, A]` on
   primary A goes A, then B, then A. (Clarified 2026-11-26 after review.)
4. **Rechecked at dispatch binding.** The binding transaction repeats point
   3 for a child chain's AGENT_FIRST call. A failure there is an ordinary
   binding refusal.
5. **Later calls** of the new chain are ordinary TOOL_LOOP calls of its own
   segment, under the unchanged retry rule.
6. **Fail closed, visibly.** Under Capture On, a preset alternate that would
   not change the (execution provider, model) target is a visible skip
   before any call, with TASK-42556's copy (an unready alternate keeps its
   own reason). ADR-200 already keeps the switch before the first tool batch
   and walks the list once, so a fallback after a tool call or beyond the
   list is never attempted.
7. **The parent is unaffected.** The child's chains stay in child segments.
   They never append to, replace in, or get judged against the parent's
   surface, and the parent's next message is witnessed as before.

The Trace view numbers a child's calls by actor, so its fallback chain reads
as the same sub-agent (`sub-agent 1 · call 2 · agent first`).

### Amendment recorded 2026-11-24 (owner-approved): a fallback chain's first call may rewrite the run's tail (TASK-42525)

The owner approved this on 2026-11-24 ("Allow it, do it now"). It grants the
replacement the previous amendment withheld. It applies to one call only: the
first call of a provider fallback's new chain, admitted as that amendment
describes. That call may record one bounded rewrite of the surface.

1. **What it may remove.**
   - The surface's leading rendered-system slot. This is allowed only when
     that slot is the first active entry, the failed call's own request
     header carried it (a `rendered_system_row` component), and the new
     call's request has no leading system row: a provider that takes the
     system prompt outside the message list.
   - The run's own tool rows at the surface's tail, replaced by the new
     call's rows in the same places. This is ADR-110's re-encoding for the new
     provider's tool-call format.

   Nothing else is removed. Every other row must match the request unchanged,
   before and after the removal. That includes user and assistant rows,
   project context and provider continuations.
2. **Checks.** The removed tool rows are one contiguous run at the tail.
   - Every one of them is a tool-call or tool-result artifact.
   - Each was appended by a tool-loop call of one of the run's chains (same
     owner, segment, turn and policy) whose predecessor in its own chain bore
     a response. This is the completed-turn witness's TASK-42534 rule. A row
     may instead have been written by an earlier fallback rewrite of the run
     (a second fallback re-encoding the first one's rows), in which case the
     rows that rewrite removed must pass this check too.
   - The new rows match them one for one, kind for kind (tool call for tool
     call, tool result for tool result).
3. **Bound.** At most 256 rows are replaced, in one replacement with no
   chained links; more is refused. The removed system slot is one more row.
4. **Recording.** The slot is removed by a replacement whose node is a
   content-free removal marker. The service's surface projection, from which
   every planner, checkpoint and reader takes the active rows, drops it.
   Readers of raw nodes by sequence (the completed-turn witness's lineage
   query and the rewrite's own checks) see the marker; they count it as one
   of the run's dead rows, like the rows the rewrite replaced, and only when a
   fallback chain's first call of that run wrote it. The tool rows are
   replaced by an ordinary bounded replacement: the first new row takes the
   range's place and the rest follow at the end. Historical calls, request
   heads and the removed rows' bytes stay as they were. (Point 4 reworded
   2026-11-25: it said "every surface projection drops" the marker, which
   was not true of the raw readers.)
5. **Recheck.** The rewrite is rechecked in the dispatch-binding transaction.
   A later completed-turn witness accepts the rewritten rows only by
   rechecking, from the ledger, the rows they replaced, wherever those rows
   are. Its range starts at the run's first row and reaches down over the
   run's dead rows.

Anything outside this is refused, as before. That covers a rewrite on any
call other than a fallback chain's first, a removed row of another run,
actor or turn, a removed row that is not the leading system slot or a tool
row, and more than 256 rows. Refusal happens before the reservation commits.

Not granted here: putting a leading system row back. After a llama.cpp run
falls back to a hosted provider, the next captured send to llama.cpp was
refused, as any captured switch from a hosted provider to llama.cpp already
was. (Granted by the next amendment.)

### Amendment recorded 2026-11-25 (owner-approved): a new turn's first call may put the leading system slot back (TASK-42525)

The owner approved this on 2026-11-25 ("Allow it, do it now"): "A new turn's
first call may put the leading system-prompt slot back when the recorded call
had no system slot and the new request starts with a system prompt; it is the
mirror of the removal you approved." It covers the send that went back to
llama.cpp after a llama.cpp run fell back to a hosted provider, and a captured
manual switch from a hosted provider to llama.cpp. Both were refused
(`unsupported_surface_change`): llama.cpp sends the system prompt as the first
message, and the surface had no slot for it.

1. **Which call.** Only a new turn's first call, on the AGENT_FIRST or FRESH
   route. Never a retry, a fallback chain's call or a tool-loop call; those
   keep their refusals.
2. **Which shape.** The surface has no active rendered-system row: it never
   had one (the turn's earlier calls went to a provider that takes the system
   prompt outside the message list), or a fallback's removal marker removed
   it. The request's first row is a rendered-system row. Every other row is
   planned by the existing rules, unchanged, as if the slot were already in
   place: retained rows must match, and an append, a completed-turn
   replacement or project context follows its own rules. So a second system
   row is a changed row those rules refuse, and the request builder carries
   none after a message. Nothing else is relaxed.
3. **Recording.** Before its other writes, the call appends one
   rendered-system node directly after the surface tail, holding the
   request's system row as a first call's slot does. After its replacements,
   the surface projection moves the active rendered-system rows to the front
   in sequence order, so the slot leads again (and two leading system rows
   keep their order). The call's header carries the row as every later slot
   call's header does (`rendered_system_row`), so the reader rebuilds the
   request exactly. Nothing is removed; earlier calls, heads and bytes stay as
   they were. (Wording corrected 2026-11-25: it said "at most one is ever
   active", which a first call with two leading system rows contradicts.)
4. **Recheck.** The planner proves the shape in the reservation transaction
   and checks the slot like any leading slot (a system row, under the call's
   frozen policy). The dispatch-binding transaction checks the shape again
   before writing: the call is still its turn's first and the surface still
   has no slot. From then on the slot is an ordinary leading slot, which the
   run's tool-loop calls and the next send match under the existing rules.
   Its sequence sits mid-history, so a later Compact now or Edit & resend
   whose rewritten range starts before it stores that range as links that
   skip it: the slot stays where it is, and the dispatch-binding check of the
   chain accepts that one skipped row only as the projection's leading
   rendered-system row. (Added 2026-11-25 after the round-3 re-review: such
   a send was refused for a sequence gap.)

Still refused: a slot put back by any other call; a system row anywhere but
first; a request with two system rows; a second slot over a surface that
already has one (a different system prompt in an existing slot is the
existing header-owned rule, unchanged); and a change to any other row that
the existing rules refuse. Out of scope: a captured manual switch from
llama.cpp to a hosted provider with project instructions, which needs a slot
removed outside a fallback (a follow-up). (Granted by the next amendment.)

### Amendment recorded 2026-11-26 (owner-approved): a new turn's first call may drop the leading system slot (TASK-42550)

The owner approved this on 2026-11-26. The question asked was: "Should a new
turn's first call be allowed to drop that leading system slot (the mirror of
the 2026-11-25 slot-restore amendment, with the same checks and the existing
removal marker)?" The owner answered "Allow it (Recommended)". It covers a
captured manual switch from llama.cpp to a hosted provider. With project
instructions (`AGENTS.md`) the next message was refused
(`unsupported_surface_change`): llama.cpp sends the system prompt as the
first message, so the surface leads with a rendered-system slot, and the
hosted request has no row for it. Only a fallback chain's first call could
remove the slot (the second 2026-11-24 amendment).

1. **Which call.** Only a new turn's first call, on the AGENT_FIRST or FRESH
   route, as for putting the slot back. Never a retry, a fallback chain's
   call or a tool-loop call. A fallback chain's first call keeps its own
   rule (the second 2026-11-24 amendment); this one does not apply to it.
2. **Which shape.** The surface's leading active entry is the rendered-system
   slot, and the request has no leading system row: its first row is neither
   a rendered-system row nor a system-role message. Only the slot is removed.
   Every other row is planned by the existing rules, unchanged, as if the
   slot were already gone: retained rows must match, and an append, a
   completed-turn replacement, a rewritten tail or project context follows
   its own rules. Nothing else is relaxed.
3. **Recording.** Before its other writes, the call removes the slot with the
   existing content-free removal marker (`surface_removal`,
   `TraceOmissionReason.SURFACE_SLOT_REMOVED`): a removal-only replacement of
   the slot's sequence whose node is the marker, directly after the surface
   tail. The call's other rows follow the marker. The surface projection drops
   the marker, so nothing takes the slot's place. No schema changes. Earlier
   calls, their heads and the slot's bytes stay as they were.
4. **Recheck.** The planner proves the shape in the reservation transaction.
   The dispatch-binding transaction checks it again before writing: the call
   is still its turn's first, the slot still leads the surface, and the
   request still has no leading system row. A refusal there writes nothing
   and leaves the call reserved, like any binding refusal.
5. **After a drop.** The surface has no slot, so a later switch back to
   llama.cpp puts it back under the 2026-11-25 amendment. The marker sits
   mid-history: a later Compact now or Edit & resend whose rewritten range
   starts before it spans it as a dead sequence, which counts toward the
   range's 256-sequence bound like the rows an earlier replacement left
   dead. Because the marker comes before the call's own rows, the next
   send's completed-turn replacement (project context, the run's tool rows)
   starts after it, and the raw-node readers that count a fallback's marker
   as one of its run's dead rows (`rewritten_sequences`) are unchanged.

Still refused: dropping the slot on any other call; dropping a leading row
that is not the rendered-system slot; dropping the slot from a request that
still leads with a system row; and a change to any other row that the
existing rules refuse.

### Amendment recorded 2026-11-26 (owner-approved): Edit & resend, Regenerate and Continue are traced (TASK-42560)

The owner decided this on 2026-11-26. The question asked was: "TASK-42560: in
a captured chat with AGENTS.md, after Edit & resend, Continue or Regenerate
(and Retry, the same family), the next message is refused with 'Trace capture
could not start'. It happens even when the provider never changes. Those
replies go out with capture off (per the 2026-10-27 ruling), and the rule that
admits the next send after them (the rewritten-tail rule) refuses
project-instruction rows. Without AGENTS.md the next send works. What should
happen?" The owner chose "Capture those actions": "Reverse the 2026-10-27
ruling: Edit & resend, Continue and Regenerate are traced themselves, as Retry
will be (TASK-42555). This is larger work, and it overlaps #3717."

This supersedes the 2026-10-27 sentence "Edit & resend and Regenerate
themselves remain uncaptured". Retry, Impersonate and citation repair are
unchanged: Retry is TASK-42555's; Impersonate never adds to the transcript.

1. **Capture mode.** Each action freezes the Capture-On request a manual
   send of the chat would: the chat's effective capture policy (global,
   conversation and the one-time next-send choice), the saved rows it read
   before any per-send transform, and its prompt mix-ins composed into the
   request. Under Capture Off it sends as before. Three kinds of action
   still send with Capture Off whatever the policy, as before, each a
   follow-up (corrected 2026-11-26 after review):
   - one that sends the pinned response prefill: Edit & resend, Regenerate,
     and Resend (a plain Continue sends none and is traced). A captured send
     with a pinned prefill already stops at the chat's second message,
     `surface_prefix_mismatch` (TASK-42583);
   - any action in a chat whose provider replays its own continuation
     checkpoints (Kimi reasoning), whose agent requests carry the saved
     replies as provider artifacts that the proofs below cannot name
     (TASK-42584); when that lookup fails, the action sends this way too and
     the stream refuses the conflict itself;
   - a Continue of a message with later rows after it (other than
     transcript-only system rows): its request runs only through the
     selected message, but its reply is appended at the end of the chat, so
     its call would re-record that turn and drop the later ones (TASK-42586).

   An action that follows the policy (traced, or Capture Off) uses up the
   one-time choice as a send does; one of the three above leaves a one-time
   Capture On for the next send (corrected 2026-11-26: the first build left
   every choice armed for a second send). A temporary chat with Capture On
   refuses an action that would be traced before any row is created
   ("Capture On needs a saved chat. Save the chat or turn Capture off, then
   try again."), as decision 15 allows no durable call before Save; the
   three above send as before. Resend of a broken turn that re-runs from its
   user message uses Continue's body and route, so it is traced the same
   way; its in-place Retry of a failed reply is Retry's.

2. **Which call is the action's.** On the plain provider path the call's
   route names it (EDIT, REGENERATE, CONTINUE). On the agent path the run's
   first call is AGENT_FIRST, and every call of the run carries the action in
   its route descriptor, copied from the request the action froze; a
   llama.cpp resend of one of its calls keeps it. The descriptor is
   content-free, is not stored, and grants nothing by itself: only the rules
   below read it. A plain-path action call counts as its turn's first call
   for the leading system slot (the 2026-11-25 and 2026-11-26 slot
   amendments), and an unresolved earlier call of its turn refuses it, as
   for a FRESH call.

3. **The call's turn.** Edit & resend: the edited user, a new turn.
   Regenerate and Continue: the saved user that the replaced or continued
   reply answers. A Continue's request ends with a provider-only
   continuation instruction (a user-role `active_request` artifact) after the
   saved reply it extends, then the turn's project context. The call belongs
   to the saved user before that reply: the planner steps back over the
   instruction and one or more saved assistant replies, which must lead
   directly to a saved user row. A Regenerate of a continued reply has the
   same shape. Only a call naming Continue or Regenerate steps back, and
   only when the trailing row's text ends with Console's instruction
   ("Continue and extend the selected message.", after any prompt mix-in
   body); an Edit & resend's call, or one naming no action, never does
   (2026-11-26, after review). A later call of the run (a tool loop, a resend) takes its turn
   from its last saved user row, and the existing origin checks then apply.

4. **The action's call re-records its turn.** The call replaces the recorded
   surface's tail from its turn's user row, even when that row matches the
   recorded one, so that afterwards every row of the turn was written by this
   call, the shape a normal send of that turn leaves. Nothing after the user
   row is retained; the completed-turn transition, the both-domains append and
   a retained suffix do not apply to this call. The replacement rows are, in
   order:
   - saved history before the user, only when the matched prefix stops
     earlier, under the rewritten tail's rules (2026-10-27);
   - the user (its saved revision or its admitted current-turn transform);
   - for a Continue, its saved replies and then its instruction;
   - the turn's project context (`project_instruction` artifacts).

   The proof is the rewritten tail's: every saved row's message is the direct
   parent of the next; the user is the call's own turn; a Continue's replies
   are saved assistant revisions; the oldest row hangs off the last retained
   saved row, an active memory's boundary, or the root; without it the call
   is refused, even for a one-row replacement. The instruction and the
   project rows are provider artifacts under the call's frozen policy.
   The replaced span keeps the 256-sequence and 256-link bounds (2026-11-06),
   and incoming rows count toward the 256-row bound. The planner proves it
   in the reservation transaction; the dispatch-binding transaction rechecks
   the chain and anchor, as for any rewritten tail.

5. **The send after an action's run.** A normal FRESH or AGENT_FIRST send
   whose segment's latest call belongs to a run that re-recorded its turn
   is planned by the rewritten tail instead of the completed-turn transition.
   The run is recognised from the ledger: its first call (sequence 0, AGENT_FIRST
   or an action route) wrote its pinned user row as the first row of a
   replacement that started at a user row (or at a chain's placeholders),
   which a normal send's own call never does; every call after it belongs to
   the run (or is a llama.cpp resend of one of its calls); the latest call is
   settled. Engineering clarification (2026-11-26, after review): a later
   call of the run the trace refused and marked not dispatched before it
   bound any row (point 7) is passed over, and a latest call whose response
   the trace holds only as a provider artifact (a tool-call response after
   which the run ended: the emergency stop holding its tool turns, a
   breaker, the turn budget) bore no saved reply, so the run counts as
   bearing none; a saved response the trace cannot verify still refuses.
   The completed-turn transition finds a turn's last call the same way,
   passing over a call settled not dispatched before it bound any row: it
   reached no provider and changed no surface. A canceled trace pause
   leaves one too; on the agent path its next send was refused before
   (`unsupported_surface_change`), and now it is planned as after the
   turn before it.
   The replacement then:
   - removes only rows the run wrote after its user row, or from that row
     when it is a transform, whose saved source then leads the replacement;
   - carries, before the new user, the latest call's verified saved
     response when the run bore one (complete or stopped); when the run bore
     none (it failed before any response), nothing, or the reply an earlier
     call of the same turn bore with a verified link (a failed Regenerate
     leaves the original reply on the path);
   - ends with the new user and its project context;
   - may hang off the run's own failed reply when that reply is empty and its
     parent is the last retained row (a failed Continue or Edit & resend;
     the next request skips it), and only when the run bore no response.
   A reply the run did not bear, such as one a later Capture Off Retry wrote,
   is refused here: Retry is TASK-42555's. The dispatch-binding transaction
   recomputes the run from the call directly before the reserved one and
   rechecks every point. A replacement that starts before the run's user row
   (after Compact now, say), or that starts at it but carries another row
   there (a Capture Off Edit & resend of that message, or a switch back to
   the branch the action replaced; 2026-11-26, after review), is an ordinary
   rewritten tail, as before, with no project context.

6. **What the trace shows.** Historical calls and heads stay as they were
   (decision 10): a Regenerate's or Edit & resend's earlier call keeps its
   request and response, and the action's call is a new call of its turn
   with the new reply, so the turn lists both. A Continue's continued reply
   is found through its response link. A Regenerate on a provider other than
   the replaced reply's is admitted under the rules for a turn's first call,
   like any send.

7. **Refusal.** A shape outside these rules is refused before the provider is
   contacted. When the action's prompt mix-ins cannot be composed, it is
   refused as the reply path refuses them ("Mix-in attempt could not be
   saved; response was not sent."), and a one-time next-send choice stays
   armed. Amended 2026-11-26 by the owner's decision below: when the trace
   refuses the action's own call (its reservation or dispatch binding)
   before anything is sent, or the action's trace request cannot be frozen,
   on the agent path and the plain path alike, the action is undone:
   - a refused binding's established reservation is first marked not
     dispatched, as a canceled trace pause does. A refusal that holds no
     reservation is undone only when no call of the action's turn was
     reserved since it began (a factory that fails after its own commit
     leaves one, whose state is unknown);
   - the active branch is set back to the leaf before the action, then its
     empty reply (for an Edit & resend, the edited message with it) is
     removed, then any skipped-skill note it added (a note that cannot be
     removed stays off the branch; the action is still undone);
   - a one-time next-send choice stays armed; the citation-repair
     "original attempt" previews that Regenerate and Edit & resend clear
     before streaming are restored; Edit & resend records its "message
     edited" event once its stream ends (or raises) and the action was not
     undone, so the event follows the reply's own events (corrected
     2026-11-27 after review);
   - the status line and a notification say "Not sent: trace capture could
     not start. Choose Capture off for the next send, or turn Capture off
     for this chat." (with " Your edit was not kept." for an Edit & resend).

   What stays (corrected 2026-11-26 after review): on the agent path the
   failed agent run is still listed in the Run log; the mix-in attempt
   record saved for the removed reply stays; the undo's own writes (the
   reply's tombstone, the branch selection) are recorded as for any delete
   or branch switch.

   The reply is kept, and fails with "Trace capture could not be saved, so
   this reply stopped." (for a Regenerate, whose earlier reply is shown
   again: "Trace capture could not be saved, so the regenerated reply
   stopped. The earlier reply is shown."), when it holds any content (text,
   an image, an attachment, a continuation, thinking, tool output or a diff),
   when a call of the action's own turn reserved since it began was
   dispatched (a later tool-loop call refused; calls of other turns or chats
   never count), when the refused call's dispatch is unknown, its reservation
   could not be marked not dispatched or is unknown, or when an undo write
   before the reply's removal fails (the branch is then set back to the
   reply). The run is settled either way, and a one-time choice is used up
   only when a call of the action's turn was dispatched. After a kept reply
   whose run was sent, the next captured send is planned by point 5 (its
   refused later call is passed over). A kept reply whose request was never
   sent is one no rule names: each later captured message stops at "Trace
   capture could not start", and Send without capture sends it (corrected
   2026-11-27 after review; with an unknown reservation this lasts until
   startup recovery settles the reserved call). Its Retry is Retry's
   (TASK-42555).

   Owner decision (2026-11-26). The question was: "When the trace refuses
   one of these actions' own call (rare), the empty reply it already saved
   stays as a failed reply. Every later captured send in that chat then
   stops at 'Trace capture could not start' until the user picks Send
   without capture. Which fix should go in?" The answer was "Drop the empty
   reply (Recommended)": "When the trace refuses before anything is sent,
   remove the action's empty reply and say why. The chat is left as it was
   before the action, so the user can try again or turn Capture off. Small,
   and stays clear of PR #3717." The copy's "Try again" was replaced after
   review: retrying repeats a structural refusal, while the one-time Capture
   off for the next send applies to actions too.

8. **A skipped failed reply** (2026-11-26, after review). After an action's
   run that bore no response, when the run's user was a transform (prompt
   mix-ins), the replacement starts at the run's user row and restores the
   user's saved source as its one history row; the new user then hangs off
   the run's failed reply, which holds nothing (no content, image,
   attachment, continuation or thinking) and whose parent is that source.
   The request skips that reply. The witness names it, and the
   dispatch-binding transaction rechecks that it is still failed and empty
   and still the user's parent. No other rewritten tail skips a reply: a
   wider rule (any tail) admitted a legacy wrong-owner ledger that
   `test_legacy_artifact_history_remains_frozen_and_has_explicit_recovery`
   keeps frozen.

Still refused: an action call over provider continuations in the replaced
range (as for any rewritten tail); a re-record whose matched prefix stops
before history the rewritten tail cannot prove; project context in a
rewritten tail of any other send; and every shape the rules above do not
name. An action the trace refused before dispatch no longer leaves a failed,
empty reply on the path (point 7, 2026-11-26), so the sends after it are
planned as after no action; a kept reply's next send is planned by point 5.
Out of scope: citation repair (TASK-42585).

## Current-owner integration amendment (2026-10-13)

Canonical message/revision identity and predecessor ancestry remain reachable independently of archived payload. The existing admitted checkpoint roots and graph-epoch invalidation remain authoritative before reservation; metadata roots cannot retain deleted payload through another call sharing a policy. Retry capture compares canonical JSON bytes (including boolean/integer/float identity), and frozen arrays thaw only when forming PersistRequest fields.


## Live native continuation advance amendment (TASK-35582, 2026-10-27)

Actual native CaptureOn tool loops exposed a missing admission shape after the
custom-hosted endpoint projection was repaired: a completed new native batch
appends message rows while extending one live continuation surface value. The
existing one-value replacement and completed-saved-turn witness cannot authorize
that live mixed-domain operation. This amendment is adopted before implementation.
Detailed design and controller boundary: [focused design](../../Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/native-trace-delta-design/design.md),
[controller adoption](../../Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/native-trace-delta-design/controller-adoption.md).

- Add one separately typed, host-issued live-native witness to existing capture
  admission/capability machinery. It is restricted to the exact tool-loop run,
  originating source/turn, segment, previous eligible call and predecessor head.
  Completed-turn authorization and ordinary append/replacement/retry rules retain
  their existing meaning; witness construction alone grants no authority.
- Revalidate the current assistant generation, active path, acknowledged message/
  checkpoint version, raw selected-provider identity and canonical protocol/model/
  full-base binding under existing custody/CAS. Reuse host projection/signing
  provenance where raw and durable checkpoints differ. Broad ignored-field or
  redacted-byte equality cannot establish raw custody.
- Admit exactly one changed current continuation: old rounds are byte/structure
  exact, one new finished native batch has exact IDs/names/raw arguments/results,
  no pending/executing calls remain, and its attachment is the newest exact batch.
  The prior message domain is an unchanged prefix; the appended suffix and any
  steering retain host provenance/order. Other continuation groups retain exact
  value/order/descriptor. Bind suffix/span to existing limits and frozen policy.
- Replace only that continuation node (span1) and append the message suffix using
  existing surface_replace/surface_append events in the same transaction as header
  and call dispatch binding. Preserve per-domain ordinals and old call heads.
  Recheck predecessor and owner/version at final binding; rollback/ambiguous
  commit uses existing owned reconciliation, with no automatic redispatch.
- Preserve linear trace-owned storage: after the first full base, a distinct
  versioned trace-only continuation-advance projection stores bounded top-level
  metadata, prior round count and the single new sanitized round. Its predecessor
  comes from the exact existing one-node replacement edge, never a caller-supplied
  JSON reference. Do not store full growing checkpoints or growing prior-ID arrays.
  Provider kwargs and current assistant V3 remain their full original formats.
- Sanitize the full candidate under frozen policy and verify the old projected
  prefix before compacting it. A projection change, missing/corrupt/cyclic/wrong-
  lineage/policy edge, count/depth/byte overflow or failed custody is an explicit
  refusal/unavailable outcome. Encoded deltas never become execution/Resume state.
- Use a shared node-aware resolver in durable comparisons and native readers.
  Legacy full values stay readable without reinterpretation. Fork/archive/copy
  and GC must preserve actual graph dependencies, and final-owner deletion must
  reclaim them. Verify raw artifact/metadata growth; reconstruction CPU and other
  checkpoint owners are separate costs. No new mutable owner/table is planned.
- Cold live advance requires existing explicit host custody evidence or refuses
  before provider entry. Matching old bytes or a historical trace reference cannot
  recreate execution authority. This does not add durable review-job recovery.

Rejected alternatives: moving checkpoints into headers would lose their surface
lineage; raising replacement counts admits unauthenticated history changes; using
completed-turn proof mislabels a live chain; full-checkpoint snapshots duplicate
old rounds; mutable-latest references lose old calls; dropping capture/private
groups changes fidelity; a general patch engine or new registry is unnecessary.

Existing ADR063/179 owns provider/current-checkpoint custody and is unchanged.
TASK-35582/ADR281 links this amendment. No SQL migration is asserted until artifact
codec/enum/archive constraints are audited; any required DDL uses normal versioned
migration and a revised plan before source work. Required evidence includes actual
normal Console2/3-round/multicall CaptureOn, strict owner/prefix/policy negatives,
atomicity/races/privacy, legacy/corrupt graph reads, retention/forks and linear
artifact-plus-metadata growth. No live provider allowance follows from this ADR.


Runtime custody prerequisite (Ruling61, before this runtime edit): the existing
continuation persistence callback receives only projected checkpoints/events and
returns None. Exact raw-to-projected provenance is currently discarded. Add one
optional capture-only host acknowledgment after that callback succeeds; carry
exact invocation raw/projected relation to bounded current adapter custody, then
existing store projection/sealing and final owner/version/CAS validation. Keep
ordinary/CaptureOff behavior, durable persistence barriers and transition semantics.
Raw values never enter generic event serialization, provider kwargs, DB bodies,
logs, diagnostics or a history registry. Failed/missing/stale acknowledgment grants
no capture witness or execution power. Cover private-body projection, managed
sealing, persistence callback failure and no duplicated side effects explicitly.
No SQL/checkpoint-format or review-job owner change is introduced.


### Canonical completed-history replay amendment (Ruling64, before code)

Adopt QA/ac24/review-wake-repair-design/completed-replay-correction.md and
completed-replay-controller-assessment.md. Preserve the corrected existing
transcript SYSTEM/payload USER policy; carrier-only work is removed. The real
accepted-wake failure is retained complete checkpoint forwarding without its
canonical native owner rows, alongside the independently proved capacity/tool loss.

After existing host owner/target admission, derive canonical recorded assistant/
tool history at final preparation from the exact complete persisted checkpoint.
Reuse the existing translator; keep one collapsed semantic owner plus attachment
as the atomic budget/eviction unit. Preserve exact IDs/args/results and one final
content, private reasoning sidecar, all retained complete groups alongside current
active groups, child isolation and explicit active Resume gates. No re-execution,
fuzzy owner match, missing group tolerance or engine relaxation. Count the same
projection before fit and avoid duplicate completed-expanded body charge.

A narrowly deterministic one-to-many trace derivation may reuse existing completed
owner/revision, private attachment and terminal lineage/range machinery. Recompute
all output rows under frozen policy; derived rows are not fabricated saved ordinary
revisions. Preserve count/span/source/actor/turn/CAS and final shadow gates. No
arbitrary tool span acceptance, active-custody resurrection, growing per-call
history arrays, codec/GC change, registry or new schema. Any needed durable shape
requires concrete assessment before adding it. Existing provider/checkpointV3 and
legacy readers remain. Actual handler-backed [2,1] plus CaptureOff/On, replay/live
coexistence, counting/eviction and strict negatives are required before closure.


Completed replay compound-transition clarification (Ruling67): the actual later
MANUAL CaptureOn request changes both the old native tool span and its same-owner
active checkpoint attachment. Permit only this exact compound replacement under
the existing saved-response/terminal witness, canonical replay derivation and
frozen-policy recorded-round/target proof. Preserve the whole unchanged prefix,
other attachments, source/run/turn/policy lineage,256 span/count and CAS/final
shadow checks. Changed/foreign checkpoints, altered recorded rounds and pending/
executing ownership cannot pass. Use the existing graph transaction and retention;
no generic mixed span, codec/GC/schema or active custody extension. The required
negative controls and independent review must establish this boundary.


Completed-history plus live-group append clarification (Ruling68): when old message
and continuation domains each remain exact prefixes, append only the new native
message rows/current checkpoint through existing surface_append and domain
ordinals. The flat request concatenation may otherwise misclassify the unchanged
historical checkpoint as a mixed replacement. Retain every historical node,
exact descriptor/value/policy and the separate current host live-custody proof.
Preserve final reconstruction/count/CAS, quotas/atomic rollback and all target/
permission gates; reject changed history, insertion/reordering or foreign/changed
attachments. No new durable shape, generic replacement, codec/GC/schema change or
historical-to-live authority transfer.


Recorded-value/privacy clarification (Ruling69): exact replay follows the admitted
persisted checkpoint after existing tool-record projection. In particular the
unchanged CodeReviewToolProvider intentionally persists schema_version/tool_name
and body-free results rather than original ephemeral review args/source/report
bodies. Do not restore those raw values or equate saved replay to old volatile
wire bytes. Compare exact canonical saved strings/IDs/results; apply existing
frozen credential/PII projection only when comparing disclosure trace artifacts.
Raw provider/recompute/live-custody checks remain separate and strict. Masked
byte equality cannot create custody; no ignored fields, disclosure or policy
change. The final oracle and focused mask regression must state this distinction.


Implementation evidence through c3f99ec: the corrected accepted-wake/completed
replay contract passes scoped independent review after transient-credential and
incoming-owner fixes. Canonical persisted replay, eligible MANUAL Safe/Full,
strict negatives and final binding are covered in
Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/review-wake-repair-fix/.
One nonblocking precedence-fixture coverage gap and retained pending-refusal
shutdown limitation remain explicit. No automatic capture-policy, durable format,
codec/GC, execution authority or recovery expansion follows. Live qualification
remains incomplete.
### Clarification recorded 2026-10-18: thinking-serialized tool-call surface identity

[TASK-33164](../tasks/task-33164%20-%20Preserve-captured-archive-recovery-follow-up-ownership.md) reproduced a completed native tool run whose compatible thinking serialization correctly produced MESSAGE_REWRITE provenance, with a THINKING artifact over its original TOOL_CALL artifact and matching THINKING_ATTACHMENT. Surface storage classified the whole rewritten tool-call row as thinking, so the existing completed-tool range validator refused the next captured user turn. The typed rewrite and the strict validator are both required; the surface classification must retain the causal tool-call identity.

For new surface references, only an exact DerivedTraceProvenance MESSAGE_REWRITE with an outer THINKING artifact and direct ProviderArtifactTraceProvenance TOOL_CALL first input retains the tool_call component kind. Existing typed attachment validation remains mandatory. The complete serialized artifact bytes, causal inputs, capture policy, credential/PII projection and thinking/privacy semantics do not change. Ordinary assistant-thinking rewrites, standalone thinking artifacts, unknown transforms and other inputs retain their existing classification. Reference creation, lookup, durable comparison and node matching use the same classification.

Completed-tool source, response-link, exact-value, bounded-range, run/owner/turn, route and policy checks remain unchanged. This does not add thinking to the generic permitted replacement kinds. Existing stored thinking nodes lack per-node causal descriptor proof; their bytes or labels alone cannot authorize retrospective promotion. They remain immutable and subject to the existing refusals. No schema migration or historical trace rewrite is introduced.

This corrects the existing provenance handoff under [ADR090](090-console-thinking-block-ownership-and-replay.md), rather than creating a new authority or provider contract. [Implementation plan](../../Docs/superpowers/plans/2026-09-27-agent-archive-recovery-capture-on.md) records the bounded correction and qualification limits.

### Amendment recorded 2026-10-18: version2 local-thinking response formats

The owner approved a prospective versioned response-profile repair after native Qwen reasoning_content responses failed exact version1 tagged-thinking projection. New displayable local profiles use version2 and source_formats instead of scalar source_format, with the other existing profile fields unchanged. The closed adapter keys are llama_cpp, local_llamacpp, vllm and local_vllm, with a nonblank model. Canonical formats are start_anchored_think followed by reasoning_content, except a model name containing kimi case-insensitively permits only start_anchored_think because the existing structured parser excludes it. Producer and canonical projector share one closed rule; the profile must match its complete ordered result exactly. This declares supported parser formats, not template/native-tool/replay authority.

The exact terminal block retains its concrete source_format and provider/model/protocol/status/text. Version2 validates the exact shape and integer versions, unchanged terminal-round/type/single-block evidence and process-keyed raw equality proof. Credential/PII projection and completed-tool revision/range/lineage/policy checks are unchanged. A scalar version1 profile remains strict; hosted proprietary profiles and legacy visible-only responses retain their meanings. Old headers, links, artifacts, revisions and reconstructions remain immutable; no relinking, migration or retrospective source inference is permitted. Fresh captures establish the new contract.

The contract uses the existing immutable request-header defaults JSON; no schema/storage expansion or new retained raw bodies. Rejected alternatives are widening v1, rewriting old artifact links, relabeling actual parser provenance, weakening equality/completed-tool checks, or disabling capture/thinking. Design: Docs/superpowers/specs/2026-10-18-versioned-thinking-response-profile-design.md; recovery task TASK-33164 and its existing correction plan link this amendment.

### Amendment recorded 2026-10-27: rewritten surface tail after Compact now or Edit & resend

TASK-33621.29 and TASK-33621.7 (Console UX review, 2026-09-29) reproduced two
ordinary actions that blocked capture for the rest of a chat. Compact now commits a
conversation memory. The next request drops the compacted turns from
`messages_payload` and carries the memory as a system message outside the history
rows. Edit &
resend dispatches its own call with Capture Off, because it has no turn
preparation, so the edited sibling branch never reaches the surface. In both cases
the next captured FRESH or AGENT_FIRST send carried a replacement of the recorded
surface's tail plus the new turn. The planner admits only an append or a one-item
replacement, so it refused the send with `count_or_span`, and every later send
stopped at "Trace capture blocked". Measured live on dev `f9baac8706`: after Compact
now, prefix 0, 13 recorded entries changed, 3 incoming; after Edit & resend, prefix
2, 1 recorded entry changed, 3 incoming.

Permit one more typed replacement-plus-append, the rewritten tail, for FRESH and
AGENT_FIRST sends only, when no completed tool-turn transition applies. That
transition's applicability requires matching descriptors and the permitted start
sequence relative to its origin call, not merely an equal descriptor count. A
selected completed transition still undergoes every durable ownership and range
check; validation errors are not caught to retry with another proof. The replaced
range is the recorded surface's whole tail after the matched prefix (no retained
suffix), messages only, within the 256-node range limit. The limit is measured in
stored sequences, the way the replacement is persisted and read back: sequences an
earlier replacement left dead inside the range count, so the planner bounds the
range's span, not its live row count. Every replacement row
except the current user must be an exact saved revision; a saved response may carry
its typed thinking replay. The current user may carry its admitted current-turn
transform. Each row's live message must be an undeleted message of the attached
conversation and the direct parent of the next row's message, and the last row must
be the call's own turn. The oldest row's parent must be one proven anchor:

- the message of the last retained saved row (an edited or switched branch);
- the boundary of an active conversation memory (compaction). The memory reaches the
  provider as a system message outside the history rows, so the active memory's
  boundary is the durable proof of the dropped rows. Retained saved rows may precede
  the gap, the shape a manual range memory sends; no link from them to the memory's
  range is proven yet, and the anchor is any active memory of the conversation rather
  than the one in effect on the branch (tightening both is TASK-37601); or
- no parent, when no saved row is retained (a branch from the conversation root).

Recheck the chain, the anchor and the memory's active state in the dispatch-binding
transaction; any change fails capture closed. Removed rows contribute no disclosure
policy or structural metadata to the composed request.

This is not a general history-replacement permission. Retained suffixes, tool or
project artifacts in the replacement, provider continuations, unsaved or foreign
rows, gaps, deleted rows, inactive memories and unmatched anchors stay refused.
Historical calls and heads stay immutable. No schema, retention change or durable
registry is added; the proof is a bounded transient witness, like the completed
tool-turn witness. Edit & resend and Regenerate themselves remain uncaptured: the
edited branch enters the trace at the next captured send, not at the edit.
(Superseded on 2026-11-26 by TASK-42560, owner-approved: Edit & resend,
Regenerate and Continue are traced themselves. See "Amendment recorded
2026-11-26 (owner-approved): Edit & resend, Regenerate and Continue are
traced". The rewritten tail itself, for Compact now and for a Capture Off
edit followed by a captured send, is unchanged.)

Known limit (review, 2026-10-29). A rewritten tail wider than 256 stored sequences
stays refused, so Compact now in a chat of more than about 128 recorded exchanges, or
Edit & resend on an early message of one, still blocks every later captured send.
Admitting it would reach the over-span fallback, which records an
`unsupported_replacement_span` omission in place of the first new row and keeps
every replaced row in the surface, so the trace would stop matching the request.
Recording a wider replacement changes the durable format (`SurfaceReplacement`
rejects a wider span when it is read back), so it needs its own owner decision
(TASK-38298). Superseded on 2026-11-12: the owner accepted the chained replacement
recorded in the next amendment, which lifts this limit up to 256 links.

The owner ruled on 2026-10-27 that TASK-33621.7 fixes the edited-branch trace
defect only, with no remembered per-chat capture bypass. This amendment records
the mechanism that fix and TASK-33621.29 share. It is recorded with the
implementing change and is subject to the owner's review of that change.

### Amendment 2026-11-06, accepted 2026-11-12: a rewritten tail wider than 256 stored sequences

Status: **Accepted.** Owner decision 2026-11-12 (TASK-38298 AC#1): "Approve the
chain". The owner was asked: "record a rewritten history wider than 256 stored
rows as a chain of ordinary replacement rows, each at most 256; content-free
placeholders collapsed by a final link; cap 256 links (about 32,000 exchanges); no
schema change, today's reader already reconstructs it exactly, old traces
unaffected". TASK-38298 implements the decision below. It supersedes the known
limit of the 2026-10-27 amendment.

**Measured on dev `8ae979d81b`** (scratch probes, deleted; their source and output
are kept with the TASK-38298 evidence):

- Through the real controller, store, gateway preparation, compaction service and
  trace runtime (agent path; only the provider adapter is a recorder): 130
  exchanges (259 recorded rows), then Compact now ("Conversation memory updated"),
  then a send. The send stops at the card: run BLOCKED, no reservation, no provider
  call, `count_or_span` with prefix 0, 259 recorded entries changed, 3 incoming,
  span 259. Edit & resend on the chat's first message, then a send, gives the same
  refusal and the same numbers.
- One replacement row wider than 256 sequences, written directly: every read of
  that trace segment fails, because `SurfaceReplacement` rejects the row when the
  segment's replacements are read. That includes the calls recorded before the
  wide row and the planning of every later send in the chat. So a wider bound,
  read by this or any earlier version, loses the conversation's whole trace and
  blocks its captured sends, not just one call.
- A chain of ordinary replacement rows, each spanning at most 256 sequences,
  written with today's repository and read with today's reader (the fold is
  unchanged since `c78f641adf`): the 259-row compaction and the 131-live-rows-over-
  261-sequences shape both reconstruct to exactly the new request. The head
  recorded before the chain still reconstructs its 259 rows. The next send plans
  as a plain append. Trace GC keeps every link and sweeps nothing. A 1,999-row
  history needs 8 links plus the final link (11 nodes, 9 replacement rows and 11
  events), and its cold projection reads in 0.02 s.

**Decision: record it as a chained replacement.** This applies only where the
rewritten tail is already admitted: FRESH or AGENT_FIRST, the whole tail after the
matched prefix, messages only, and the same proofs and dispatch-binding recheck.
The one difference is that the replaced range's stored span exceeds 256.

1. The planner groups the replaced live rows, in stored-sequence order, so that each
   group spans at most 256 stored sequences. Dead sequences inside a group count;
   a run of dead sequences between groups is not replaced again. The existing
   check that no unchanged live row lies inside the replaced range still applies
   to the whole range.
2. Each group becomes one ordinary replacement row, a link: one predecessor head,
   one contiguous range and one replacement node. That node is a content-free
   placeholder (`reference_kind` omission, component `surface_replacement`, a new
   reason code `replacement_chain_link`). It carries no conversation content, and
   no gap event is written for it.
3. A final link replaces the contiguous placeholder sequences with the first new
   row. The remaining new rows append as they do today. The links, placeholders,
   appends, header and dispatch binding commit in the one dispatch-binding
   transaction. Every link's range is revalidated there, and the witness is
   rechecked as today.
4. The final link spans the placeholders, so at most 256 groups are allowed: about
   65,536 stored sequences, roughly 32,000 exchanges. A wider rewrite is refused
   as today.

In this amendment a link is one group's replacement row. The final link is
counted separately, so a chain of N links writes N + 1 replacement rows.

Truthfulness: the call's reconstructed request is exactly what the provider
received. No call's head lies between links, so no call's surface contains a
placeholder. Historical heads are unchanged, and no omission is recorded in place
of real content. Compatibility: no schema change. Every reader since the ledger
landed folds replacement rows in replacement-node sequence order and accepts
links of at most 256 sequences, so older versions read these chats normally.

What this changes in the ADR: decision 10 and the 2026-10-27 amendment allow one
bounded replacement per surface change. A chain keeps that shape for each row,
but one send writes up to 257 replacement rows plus their placeholder nodes, about
one per 256 replaced sequences. The count is proportional to that send's replaced
span, not to every call, so decision 17's per-call growth bound is untouched. The
implementation must still re-measure the reservation p95 with a chain.

**Alternatives and their measured costs.**

- *Raise the bound for this shape behind a format guard.* Per the second
  measurement, any version that enforces 256 loses the whole conversation's
  trace when it reads one wider row. The guard would have to be a schema-version
  bump, so that an older app refuses the whole database instead. That blocks
  downgrades for every user to admit one shape.
- *Keep the bound and name the cause on the card.* This fixes the unexplained
  card but nothing else. Every later send in a long chat still costs a click,
  because the owner ruled out a remembered per-chat bypass on 2026-10-27.
- *The over-span omission (declined in the 2026-10-29 review).* It records an
  omission in place of the first new row and keeps every replaced row, so the
  trace stops matching the request.
- *Start a new trace segment when the whole tail is replaced.* That changes owner
  and segment lineage, fork inheritance and GC roots. It is a far larger change
  for the same outcome.

The rewritten-tail witness separately bounds the rows sent after the rewrite point
to 256: 255 saved rows plus the current user. That bound is transient, with no
durable format, and it did not bind in either measured shape (3 incoming rows).
This decision keeps it. A compaction that keeps more than 255 rows after the
memory boundary still stops at the card.

**Question put to the owner (2026-11-06).** Do you approve the chained replacement
above: links of at most 256 stored sequences, content-free placeholders collapsed
by a final link carrying the first new row, capped at 256 links (about 65,536
stored sequences), with no schema change? Or do you choose a format-guarded wider
bound, or keeping the 256 bound with a cause-specific card? The owner approved the
chain on 2026-11-12.

**Implementation (TASK-38298).** The planner cuts the replaced live rows into links
by stored sequence and refuses more than `MAX_SURFACE_REPLACEMENT_CHAIN_LINKS`
(256) links as `count_or_span`. The admission and the verified delta carry the
links on the replacement range; only a rewritten tail may have them. The writer
validates every link's range, that no live row lies between links, and that each
link holds at least one live row (TASK-40577: the reader refuses a link with none,
so committing one would leave the new head and every later head unreadable, and,
after a restart, every later send unable to plan; heads recorded before it still
read), in preparation and again in the dispatch-binding transaction, then writes
the placeholders, their links and the final link in that transaction. Placeholder
sequences precede the new rows, and the warm projection checkpoint records them
in its lineage as the cold reader does.

Reservation re-measured 2026-11-12 (reserve plus dispatch binding, preparation
excluded, as the reference gate times it; 20 interleaved rounds per shape on an M4
Pro under a shared load average of 28 to 48, so wall times are noisy and thread
CPU is the steadier figure):

| Shape | Links (plus the final link) | Wall p50 / p95 | Thread CPU p50 / p95 |
| --- | --- | --- | --- |
| Append, 259 recorded rows | 0 | 3.2 / 10.0 ms | 3.2 / 4.9 ms |
| One replacement, 256 rows | 0 | 7.8 / 20.4 ms | 6.5 / 10.1 ms |
| Chain, 259 rows | 2 | 11.9 / 55.5 ms | 9.1 / 12.6 ms |
| Chain, 1,999 rows | 8 | 109 / 146 ms | 73 / 89 ms |

A chain costs about one replacement's time per 256 replaced rows, mostly the
per-link range reads and the replacement rows' lineage trigger. (Superseded by the
2026-11-20 re-measure below: each link's cost grows with the segment's stored depth,
so a chain costs about links x depth, not a fixed amount per link.) It is paid once,
by the send after the rewrite. Ordinary sends, which the 10 ms reservation gate
measures, take no new work.

Dispatch binding re-measured 2026-11-20 (TASK-40578; thread CPU, load1 4 to 7). The
lineage trigger, not the range reads, is most of a chain's cost. That trigger walks the
node chain toward the segment root once per replacement row. At 1,999 rows (8 links)
binding takes 36 ms, of which the two validation passes are 18% and the trigger about
64%. At the 256-link cap (65,536 stored rows) binding takes about 38 s, so it holds
the write lock for that long. The validation passes are 0.6% of that and the trigger
99%; TASK-42130 tracks the trigger cost.

The review measured the sizes between those two points. At 5,000 rows (20 links)
binding takes 164 ms of CPU. At 10,000 rows (40 links) it takes 1.03 s of CPU, and
7.0 s of wall time at load1 82. The cost grows about as links x stored depth. A
captured request carries at most about 20,000 messages, so about 80 links.
`Tests/Benchmarks/test_console_trace_chain_binding.py` reruns these shapes on
request.

### Amendment recorded 2026-11-13: append in each domain when a send grows both domains

Status: approved by the owner on 2026-11-13. It was proposed on 2026-11-12 with
its implementing change (TASK-34100.3, folding in TASK-34100.3.1), under the
title "append in each domain for replayed reasoning", and the branch was held
from merging until the owner decided. The pre-merge review of 2026-11-13
tightened the rule and this text (a side-effect-free probe, the precedence and
scope below, and what the owner check can and cannot prove).

TASK-34100.3 made saved-chat Moonshot (Kimi) replies persist, which made a third
turn reachable. Every model Moonshot lists returns reasoning on each reply, and
Console replays each earlier reply's saved reasoning checkpoint on the next turn.
With capture on (the default), turns 1 and 2 traced normally. Every later
FRESH or AGENT_FIRST send stopped before the provider with "Trace capture blocked".
Measured on dev `26b2e677dd`: `count_or_span`, prefix 3, suffix 1, 3 incoming
entries changed, 0 recorded entries changed, domains `messages_payload` and
`provider_continuations`. Turn N's request is the recorded messages, then the
previous reply and the new user message, then every recorded checkpoint, then
the previous reply's checkpoint. Nothing recorded changes. Flattened, though, the
new messages sit before the old checkpoints and the new checkpoint sits after
them, so the planner saw two change points. It already admits growth in one
domain: a plain append, and new messages before an unchanged continuation
suffix. Dropping the optional replay is no workaround, because removing a
recorded continuation is itself refused (TASK-34100.3 review round 2 probe).

Admit one more append shape, for FRESH and AGENT_FIRST sends without live
native custody and outside a completed project-context transition. The rule is
not specific to Kimi: any provider's send of that kind qualifies when it grows
exactly in both domains. Replayed reasoning is the case that produced it and,
today, the only one known. Every condition must hold:

- Every recorded message equals, through the planner's existing durable
  comparison, the incoming message at the same index. The planner's flat
  prefix already compares them and stops at the first recorded continuation,
  so it must cover exactly the recorded messages; the rule does not compare
  them again.
- Every recorded continuation equals the incoming continuation at the same
  index, through the same comparison.
- At least one message and at least one continuation are new.
- Every incoming continuation is a typed `CONTINUATION_ATTACHMENT`. One in a
  recorded position must be owned by a message equal to a recorded message,
  and a new one by a message equal to a new message.

What the owner check proves: only which side of the history a checkpoint
belongs to, by descriptor equality. For a recorded slot it checks that the
owner is some recorded message, and the slot's bytes must equal the recorded
checkpoint. The guarantee that no recorded reply gains, loses or swaps a
checkpoint therefore comes from content, not from the owner reference. A Kimi
checkpoint carries its own reply's content, so two recorded checkpoints can
only trade places if they are byte-identical, and then the request carries the
same bytes either way. That is the strength of the existing rule for new
messages before an unchanged continuation suffix, which also matches
continuations by content.

The owner reference cannot be made an identity proof. The pre-merge review
asked that each owner occur exactly once among the incoming messages (or be a
saved revision). The mounted Console carries every assistant reply as the same
`ProviderArtifactTraceProvenance(ACTIVE_REQUEST, policy)` descriptor, so every
Kimi checkpoint's owner occurs once per reply, and that requirement refused
every turn 3+ again (all Kimi pilot rows, 2026-11-13); it was reverted. Only an
owner that carries an identity (a saved revision) lets the rule refuse a new
checkpoint attached to an earlier reply. With the Console's reply artifacts
such a checkpoint is appended like any new one; the ledger stores
continuations without owners and still records exactly the request sent.

Precedence: the rule is tried first. It takes precedence over the
completed-tool-turn replacement for a non-project witness, and when it admits a
request the compound replacement is not attempted (a witnessed turn whose
recorded entries all match needs no replacement). A completed project-context
turn is outside the rule and keeps the existing rules.

The probe has no side effects. The planner's comparison records each matched
artifact's original value for the dispatch boundary, so when the rule refuses
it restores those retained values to what they were before it ran. It never
compares incoming index 0, so it never touches the rendered-system slot.

Defensive invariants, not gates: the rule also refuses domains out of provider
order (all messages, then all continuations) on either side, interleaved
domains, and a recorded surface with no continuation. None of these can reach
it through the planner, which builds both sequences domain-sorted and calls the
rule only when its flat prefix stopped short of the recorded surface.

The new messages and continuations append at the end of their own domains, through
the existing surface append, per-domain ordinals and final binding. This is the
delta shape cold bootstrap (`_bootstrap_surface_parent`) and dispatch binding
already accept. No recorded node is replaced, no history is rewritten, and no
schema, retention, codec, registry or witness is added. The delta's tool-loop
overlay names each marked message by its position in the admitted indices. An
edited, removed or reordered recorded entry, an untyped continuation, or a new
checkpoint whose owner is not a new message (detectable only for an
identity-bearing owner, above) falls through to the existing rules, which
refuse it as before. The rule lives in
`Chat/console_trace_domain_append.py`.
Its rule-level negative controls are in
`Tests/Chat/test_console_trace_domain_append.py`, including a refused probe
that must leave the retained values unchanged and the Console's own reply
descriptors (equal artifacts), which pin both the admission and the limitation
above.
`Tests/Chat/test_console_trace_domain_append_planner.py` drives the real
planner (`prepare_current_surface_delta`) over a real ledger, warm and cold:
it admits exactly the new entries of each domain and reads the ledger back as
the request sent, and it still refuses an edited saved message the ledger
never recorded, new bytes for a recorded revision, changed checkpoint bytes,
reordered checkpoints, a second checkpoint on an earlier reply (its fixture's
replies are saved revisions) and a dropped replay, plus the tool-loop, Continue and Regenerate routes, live native
custody and a completed project-context turn. It also checks that a refused
probe leaves the planner's retained values as it found them, and that the
tool-loop overlay survives the non-contiguous delta.
`Tests/UI/test_console_trace_retry_refusal.py` drives a really dropped replay
through the mounted Console and records that it is still refused.

The same change drains the reply's pending trace settlement when the final
continuation's atomic dispatch settlement completes it. That path set the reply
complete without the drain the ordinary terminal write performs, so every saved
Kimi call stayed `response_started` and the Trace view showed none. This adds no
admission authority. Evidence: `Tests/UI/test_console_moonshot_reasoning_reply.py`
runs five saved turns per Kimi model with capture on and reconstructs each call's
request from the ledger.

### Amendment 2026-11-17 (owner decision, TASK-41310): a spawned child's calls have their own segment

Decision 3 of ADR-092 captures surviving fleet calls, but this ledger had no
owner for them. A child's request carries no saved revision of its own (its
system prompt and task are new rows), so the runtime refused it as
`trace_owner_unavailable`; a task byte-equal to a saved user message borrowed
that message's reference and the child was then judged against the parent's turn
and surface. Every sub-agent failed under Capture On, the shipped default.

1. A child's provider calls belong to the parent's owner and to the spawning
   turn, under the child's own run identity (its actor and chain).
2. Each child run has one root-less segment of its own, linked to that owner and
   turn by its run identity (`console_trace_child_segments`, schema v87). (A
   child's preset fallback chain opens one more, under the same actor:
   amendment 2026-11-26, TASK-42564.) The
   child's surface fence compares only against that segment. Nothing a child
   sends can append to, replace in, or be judged against the parent's surface,
   and the parent's own refusals are unchanged.
3. The child's request names the spawning turn in its route provenance
   (`parent_revision_id`); its rows never reuse the parent's saved references.
   The turn comes from the admitted request the parent run was built from, not
   from anything the child produced.
4. Ownership guards treat a linked child segment as owned exactly like its
   owner's root. A child segment is never a fork parent, and a fork does not
   carry the source's child segments.
5. Retention follows the owner: GC keeps a child segment while its owner is
   attached or retained, and collects it after detachment.
6. An unresolved child call never blocks the parent's turn
   (`has_unresolved_call_for_turn` counts only the owner's own segments); it is
   abandoned on restart like any fleet child.

Readers that walk the owner's root lineage do not show child calls yet; that is
a follow-up, not a change to this contract.

Note 2026-11-18 (TASK-41650): the Trace view now lists them. It reads the
calls of every child segment linked to the conversation's attached owner
(`ConsoleTraceRepository.read_child_calls`) and shows each as a `Child call`
row of the spawning turn. The root-lineage readers above are unchanged, so the
Conversation Inspector's per-message reconstruction still omits child calls.

## Renumbering provenance

Renumbered from ADR-097 to ADR-337 on 2026-11-03 under TASK-36943.3 (owner decision 2026-10-25: the younger file of a shared number is renumbered). ADR-097 now refers to 097-boot-budget-ratchets.md.
