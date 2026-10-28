# ADR-281: Primary-agent multi-model code review recipe

- Status: Accepted; implemented and reviewed; live AC24 qualification pending (2026-10-28)
- Date: 2026-10-18
- Related task: [TASK-35582](../tasks/task-35582%20-%20Implement-primary-agent-multi-model-code-review-recipe.md)
- Design spec: [Primary-agent multi-model code review](../../Docs/superpowers/specs/2026-10-18-multi-model-code-review-design.md)
- Implementation plan: [Detailed plan for TASK-35582](../../Docs/superpowers/plans/2026-10-18-multi-model-code-review-implementation.md) (implementation in progress; not a completion claim)

Extends the guide pattern in [ADR-149](149-offer-first-canvas-guide-and-inline-skill.md).
Reuses [ADR-147](147-agent-provider-routing.md) routing and preserves
[ADR-009](009-local-skill-trust-boundary.md),
[ADR-032](032-local-agent-tool-permission-boundary.md), and
[Console file authority](../../Docs/Architecture/console-file-authority.md).
Amends [ADR-135](135-fleet-completion-delivery-and-crash-recovery.md) only to add
typed review-boundary continuation eligibility; preserves its accepted-attempt
fence and [ADR-134](134-fleet-admission-and-automatic-work-budgets.md) causal-chain,
manual-priority and resource limits. The implemented schema-25 extension is described below; live qualification remains
unverified in the [qualification record](../../Docs/superpowers/qa/2026-10-18-multi-model-code-review/README.md).

## Context

Users want independent code reviews from models/APIs they choose, reusable defaults
for “do multi-model code review,” separately readable reports, and an optional
unified report that verifies or refutes every finding.

Chatbook already has provider/model-aware agent presets, gated routing overrides,
concurrent fleet execution, permission enforcement, and an editable Agents settings
surface. Ordinary model-invoked skill tools run as children; children cannot spawn
another generation. A skill file that tells its child to launch several reviewers
cannot implement this workflow without changing the delegation boundary.

A prompt-only guide also cannot establish immutable source identity or complete
report retention. Fleet results can be shortened, run-row summaries are bounded,
and chat scratch is temporary. The design must distinguish model instructions
from host-owned evidence and session lifetimes.

Review found that Console's per-turn child allowance is fixed at two, so two
reviewers leave no same-turn verifier slot and three reviewers cannot launch in
one turn. Participant-qualified IDs also contradict a provenance-free verifier
view. The revised design therefore settles continuation admission, concrete host
interfaces and schemas, verifier-ID projection, accumulated context bounds, and
aggregate storage limits before implementation planning.

## Decision

1. **Keep orchestration in the primary run.** Add a discoverable, packaged,
   primary-only `code_review_guide(stage)` with closed `review`/`verify` stages and
   a bounded result. The primary coordinates existing spawn/wait execution. An optional
   trusted inline skill aliases the same guide; it is user-invocable and disabled
   as a child-executing model skill. Natural-language use does not require it.

2. **Reuse presets and destination authority.** One device-local default team in
   `[agents.code_review]` holds ordered stable reviewer-preset references, optional
   focuses, and a verifier-preset reference. Existing presets own API/model and
   sampling choices; existing routing, allowlists, readiness, destination consent,
   and permission floors remain authoritative. Per-run overrides do not save
   themselves. Admit provider/model overrides in preparation's run choices and
   freeze them before scoped spawning; launch cannot replace the captured target.
   No provider/model is silently chosen for an unconfigured team.

3. **Give every reviewer the same general brief.** Specialist focuses supplement
   it. Start reviewers with self-contained tasks without peer reports or the
   primary's suspected findings. Narrow source-inspection capabilities at both
   catalog and runtime gates; presets alone do not exhaust runtime restrictions.

4. **Bind review evidence to a captured target.** PR/range targets use exact
   revisions and explicit comparison semantics; patch/working-tree context uses
   captured bytes and source hashes. A frozen read-only review view preserves
   workspace admission, exclusions, path validation, and instruction handling.
   Drift or missing context is a blocker or explicit coverage limit, not permission
   to inspect newer code or bypass source authority through raw Git/shell access.
   Bounded source list/literal-search/path-resolution operations discover callers
   and tests at the same captured view. Committed context may load lazily;
   uncaptured mutable context requires a new capture or an explicit gap.
   Explicit bounded line-range reads reach discovered evidence directly; cursors
   retain the captured document/range and cannot widen authority.
   Implement the existing path-aware provider seam so captured refs/cursors map
   to original admitted project scopes. The shared instruction ledger delivers
   nested guidance before source evidence; mapping performs no instruction read,
   capture or cursor advancement and cannot widen consent or permission.

5. **Keep preparation and reports within Console session ownership.** Narrow
   support at the existing Console/agent boundary captures source context, applies
   child role scopes, captures full policy-permitted terminal reports before fleet
   rendering caps, and permits bounded report readback. The existing per-tab
   scratch manager owns the review bundle and leases. No new scheduler, review
   database, sync owner, or durable-job recovery contract is introduced. Expired,
   missing, or storage-truncated reports remain explicitly incomplete; normal
   conversation persistence does not make transient report references durable.
   The spec defines `code_review_prepare`, a mutually exclusive review-scoped
   `spawn_subagent` form, role-checked `code_review_read`, revision/idempotency-
   checked `code_review_submit`, and explicit `code_review_release`. A versioned
   `ReviewContext` binds source/preset snapshots, stage grants, child receipts,
   quota reservations, original finding index, private ID map and dispositions.
   Model arguments cannot mint role, stage, source or continuation authority.

6. **Attribute actual execution.** The host assigns review/participant identity,
   records preset/guide versions and requested versus actual destinations, and
   preserves fallback switches. Failed, cancelled, malformed, or budget-exhausted
   reports never appear as successful empty reviews. Model self-identification
   and bounded run summaries do not establish what actually ran or what it found.

7. **Present originals before optional verification.** An ordinary review request
   runs the independent stage. Verification requires a request or acceptance for
   that stage; existing explicit authorization does not need a repeated question.
   Preserve each individual report and create the unified report separately.

8. **Use a fresh configurable verifier.** Start a new subagent using the verifier
   preset and original target view. Build a host-projected claim view with unrelated
   opaque verification IDs; participant-qualified original IDs stay private.
   Coalesce only exact projected repeats, keep the reversible original mapping,
   and omit report ordering, models, original severity votes, group proposals and
   multiplicity. Different triggers/evidence remain distinct units. Source can
   identify an author and paraphrases can imply related claims; perfect anonymity
   is not promised. Each unit gets a confirmed/refuted/unresolved disposition with
   evidence; duplicate proposals do not transfer verdicts between nonidentical
   claims. Failure to reproduce alone is not refutation; agreement is not proof.
   Executable checks use existing authorized process paths and disposable locations;
   host check receipts return to the verifier before a final disposition, or the
   claim remains unresolved. Suggested commands are not execution receipts.
   A threaded verifier's pending proposal yields the primary's fleet wait without
   collecting the live verifier. The primary explicitly claims or declines; a claim admits at most one
   actual authorized execution, never permission or dispatch by itself. The same
   verifier awaits host evidence within its existing deadline/cancellation fences.
   Unavailable execution or parent termination signals an unresolved result before
   fleet draining; late settlement cannot rewrite a closed verdict. This handshake
   adds no child, automatic turn, model polling or budget allowance.
   Preserve synchronous spawning at `max_live_subagents=1`: bind feedback mode
   from the actual launch branch, keep source-based verification available and
   resolve executable-check proposals unavailable immediately. Never wait for the
   blocked inline parent, force a fleet or spend a replacement-child allowance.

9. **Validate coverage without claiming an oracle.** Mechanical checks require
   each verification unit exactly once, valid canonical references and no invented
   units, then expand the private map to cover each original finding exactly once.
   Missing results stay unresolved. The unified
   report retains minority findings, refutations, uncertainty, executed checks,
   actual target/participants, and coverage gaps. Such checks establish accounting,
   not the semantic correctness of the verifier's judgment.

10. **Qualify observable behavior.** Test final provider dispatch under concurrent
    review, pinned source reads and authority, complete report retrieval, optional
    stage acceptance, failure/truncation, and finding coverage. Add a bounded real
    Console/provider qualification, distinguishing live-model results from scripted
    controls. A matching prompt or valid output envelope alone is insufficient.

11. **Preserve child ceilings through bounded continuations.** Preflight the
    current remaining total child slots and finite automatic-chain allowance.
    Keep the default two per turn. Two reviewers plus pre-authorized verification
    use one later primary turn for the verifier; three reviewers use a later turn
    for reviewer three and, when already authorized, its remaining slot for the
    verifier after presenting originals. Larger teams require further admitted
    batches. A host-sealed boundary receipt identifies the exact next slot/stage,
    settled predecessors, source epoch, accepted stage grant and original chain.
    Reuse ADR-134/135 reservations/acceptance and automatic-work accounting, without
    fabricating survivor results, resetting counters or creating a fresh allowance.
    Kill switch, exhausted/uncertain chains, manual priority and context/close/stop
    fences block continuation visibly. Later explicit user acceptance of stage two
    is normal new user work; no redundant stage confirmation is required.
    Extend the new boundary link through claim/read/accept/abort/completion/recovery:
    only a proven pre-acceptance abort releases its active claim; accepted,
    completed or uncertain links retain consumed evidence and cannot replay.

12. **Make structured submission the authoritative report.** Schema version 1
    defines findings, coverage, host envelopes, verification dispositions and check
    references. Stable local keys receive host original IDs; chunks are committed
    under slot-attempt submission revisions and scoped submission-ID checks.
    Primary check actions use a separate host-bound primary-run/review/verifier-attempt
    stream with the same revision/idempotency discipline.
    Context-control revisions and raw capture cannot invalidate a different
    participant's append. Refuse invalid references, versions,
    enums and conflicting replays without changing accepted records. Seal only
    after validated coverage and the actual terminal run receipt agree. Preserve
    raw output separately; do not reconstruct a purportedly complete finding list
    from prose. One record is at most 8 KiB and one submission at most eight records,
    narrowed by actual tool/model capacity and aggregate quotas.

13. **Bound accumulated model input, not only transport pages.** The primary
    coordinates from the host finding index, not concatenated original reports.
    One fresh verifier consumes at most eight units and 4,096 estimated claim-input
    tokens per batch, further limited to half its remaining admissible input space.
    Source-evidence pages are at most 1,024 estimated tokens and respect actual
    remaining space. Final request admission counts fixed/history/tool/guidance
    input and response reserve using the selected model's existing capacity facts.
    Unknown capacity or an oversized claim is explicit; no silent model switch or
    semantic truncation. Commit each batch outside prompt history before advancing
    its cursor. Verifier children need explicit checkpoint-backed history reduction,
    independent of primary-only run-log eviction, preserving live tool/continuation
    state. Irreducible protocol state produces a request-level gap rather than a
    falsely oversized claim. Reduction cannot erase progress; unfinished units stay unresolved.
    Batching grants no extra child, call, time or token budget.
    Complete model-bound response envelopes also fit the smaller positive runtime
    result-character cap and the existing 16,000-character native ceiling, applied
    conservatively in both protocols. Pack before generic truncation; preserve
    whole cursors/revisions and preflight mutation acknowledgments. A record that
    cannot fit alone yields an explicit transport gap with its original retained;
    no empty-page loop, semantic clipping or global result-budget increase.

14. **Reserve aggregate session storage and release it explicitly.** V1 constants
    cap source/verification content at 64 MiB, all review files at 2,048, combined
    report/raw/manifest/index/ledger content at 16 MiB, and each participant's raw
    plus submitted output at 1 MiB. Total bytes including staging are at most
    80 MiB per review and 160 MiB retained/reserved per tab, with at most four
    retained review contexts. Existing narrower limits apply. Reserve before
    capture/writes, count physical copies and temporary outputs, release unused
    reservations after sealing/settlement, and retain occupied bytes until actual
    cleanup. Refuse oversized input or stop affected publication honestly on
    overflow/disk error. Full tabs require explicit keep/export/release or scope
    choice; no silent eviction. Tombstone before cleanup and drain native leases;
    cancellation/status changes alone cannot refund quota or admit late writes.

## Alternatives considered

- **Ordinary model-invoked skill as coordinator:** incompatible with the existing
  child delegation limit; widening nesting is unnecessary for this feature.
- **Explicit inline skill only:** a useful alias, but natural-language agent
  invocation should work without installing/trusting a personal skill.
- **General workflow-engine extension:** introduces branching/parallel execution
  and lifecycle machinery beyond the existing primary/fleet path the request needs.
- **Raw direct calls to each API:** duplicates existing provider integration,
  credentials, cancellation, and accounting instead of using routed agent presets.
- **Guide instructions as the only mechanism:** cannot enforce source identity,
  actual destination attribution, role isolation, or retention before truncation.
- **Majority voting or automatic author review of its own claim:** agreement is
  not evidence; a fresh verifier reduces shared conversation context while source
  checks and explicit uncertainty remain necessary.
- **New universal report/artifact database:** unnecessary for the initial
  session-local workflow; explicit keep/export can use existing destinations.
- **Raise global child defaults to fit the review:** changes unrelated work and
  hides the difference between total child allowance and concurrency; use bounded
  same-chain review continuations instead.
- **Strip model names but retain participant IDs/group counts:** still exposes
  provenance; build a separate verifier projection and keep its map host-side.
- **Read all report pages into one prompt:** transport paging does not bound
  accumulated model input; indexed batches and an incremental ledger do.
- **Bound only individual file/tool operations:** repeated captures and output
  still accumulate; aggregate reservations and cleanup-aware release are required.

## Consequences

The guide is small; source capture, role-bound dispatch, structured reporting,
continuation acceptance and quota accounting are substantive implementation work.
They retain existing fleet/provider/session owners and require focused Agents
settings controls, packaged guidance, tests, and user/architecture documentation.
The new typed continuation source must be qualified against ADR-134/135 rather
than assumed to be an existing wake API. Schema changes are needed only if an
existing owner actually changes; any such change follows its versioned migration.

The first version supports one default team and per-run overrides. It does not
automatically fix source, publish a PR review, merge changes, recover a review job
after a crash, or promise report retention beyond its session owner. Users must
explicitly keep/export reports for longer retention.

The guide influences agent behavior rather than acting as a deterministic stage
executor. Host checks protect target/identity/scope/report accounting; verification
remains fallible and unresolved findings remain visible. The linked specification
defines the detailed reporting, failure, retention, and qualification requirements.


### Native acceptance and continuation implementation amendment (Task35582, slice 7)

The native Console owns a session-local `ConsoleCodeReviewCoordinator`. An actual
accepted USER message (owning active path, no automatic origin) registers stage
intent before tool parameters exist. Direct review authorizes reviewers; explicit
review-and-verify authorizes both. A saved verifier authorizes neither a later stage
nor a short affirmative. Quoted, conditional and negated controls fail closed.
A native, exact-review verification offer may accept a later USER “yes”; a decline,
source fence or session fence invalidates it. Model prose cannot install an offer.

A host prepare-admission callback narrows that intent to exact target, team and
run-override fingerprints. Tool JSON proposes scope inside existing intent; it
cannot create a grant, select an unadmitted binding, waive destination consent or
add verification. The native guide exposes a default working-tree target only for
the already selected admitted local binding. Missing/ambiguous source or review
selection returns an explicit needs-attention outcome. No blanket second stage
question is added to an already explicit review-and-verify instruction.

Native source authorization reuses the composed `ToolCatalogRegistry` run-policy
counter and `LocalToolProvider.authorize_review_source` / `_verdict_for` owner,
including current permission state, persona floors, native approval, admitted
binding guard and revocation. Capture and later source access require this gate;
review intent and project-instruction consent are not file permissions. Destination
admission reuses `confirm_project_dispatch` and
`commit_project_instruction_dispatch_decision` for the actual resolved target,
current selected binding and instruction state. The existing consent owner retains
its provider/endpoint scope; no parallel destination-consent store is introduced.

Each accepted automatic continuation or later explicit manual stage binds the
actual new primary actor, exact live grants and existing source authority to the
retained context. Context metadata consumes its existing quota. Original
preparation, old streams and terminal receipts remain immutable; retries receive a
new slot attempt. Later manual acceptance uses fresh current turn capacity and
consent. Its default verifier retains the admitted definition/route; an explicit
new choice is routed and frozen through current admission. Serialized metadata or
same-session membership cannot authorize takeover. Retired contexts and controller
shutdown discard their ephemeral authority registries.

The existing `ConsoleFleetWakeCoordinator` remains the only delivery owner. AgentRuns
schema 24→25 adds the body-free `automatic_review_boundary_claims` link and nullable
boundary identity on existing wake attempts. Ordinary cause/attempt-kind constraints
remain unchanged. Claim validates the live host receipt and actual terminal SQLite
predecessor steps, then reserves one normal fleet-wake generation atomically.
Acceptance revalidates live authority before provider preparation. Proven prepared
abort deletes only its exact active link and releases that reservation; its aborted
attempt ID remains audit-only. Accepted, completed and uncertain links/charges remain
consumed. Restart enters existing review-required recovery; it reconstructs no bundle
or executable grant. Notifications occur outside review-store/SQLite transaction
locks, avoiding a store→DB / DB→store lock inversion.

Native settlement mounts or reuses `ConsoleCodeReviewModal` and displays each
reviewer’s original page before acknowledging presentation. Full originals, raw
output and captured evidence remain paged in the native reader; `u` reopens it in
the owning Console chat. The transcript separately records the session-local
reference. The nonzero Console journeys exercise that handoff before fresh
verification. Normal persistence of content actually presented
is permitted; automatic source/raw/report bundle bodies remain outside DB/log metadata
and later model history. Disabled, exhausted, fenced or uncertain continuation
projects `continuation_blocked` and retains available progress without promising an
automatic retry. Explicit new work never resets the old automatic chain.


Task 7 review corrections: verification authority accepts only a complete,
unqualified native stage clause; conditional, negated or unknown qualifications
leave verification ungranted until a later accepted instruction or exact native
offer acceptance. `Retry reviewer R1` binds R1 from the accepted USER message;
caller/tool slot proposals cannot substitute another slot. `Resume review` binds
only never-started reviewer slots in the single retained review to fresh current
manual admission, preserving the original capture, frozen routes, earlier streams
and terminal receipts. It revalidates destination consent and current child source
permission and assigns fresh attempts; it does not carry old verification intent
into a reviewer-only request. Ambiguous reviewer/review selection fails explicitly.
Old live wake eligibility is superseded and removed from the existing wake owner
outside review locks; stale callbacks cannot overwrite the new revision's status.
Prepared ownership is released only by the existing proven-abort lifecycle;
accepted/uncertain evidence is never refunded or replayed. A still-accepted missing
slot blocks manual resume until its outcome is known. The old automatic chain and
its used/reserved budget remain unchanged.

Later manual-stage grants record accepted stage intent against the retained context;
they are composed before the new primary's local permission callback is installed.
They never substitute for file permission: native child claims and source
read/materialization fail closed if that current callback is missing or denies.
The initial preparation continues to authorize source access before capture and
exact preparation-grant issuance. Fixture-only direct issuer/preparation helpers
remain distinct from the actual Console admission evidence.


Task 7 acceptance-race correction: the registered native boundary authority also
supplies an independent acceptance scope. Existing ledger acceptance acquires that
scope before its SQLite admission transaction and retains it through the
post-commit native accepted transition. Final manual-stage publication acquires
the same scope before the review-store lock, then revalidates exact selected-slot
claims and accepted/uncertain boundary state before issuing or retiring authority.
Destination confirmation holds neither scope nor publication locks. This gives
old acceptance versus manual supersession one winner, including the interval
after validation and the interval after commit but before native notification.
Ordinary claim-time validation never acquires the acceptance scope: its existing
progress-lock→DB→store/registry order cannot wait on a scope held by acceptance
while acceptance waits on DB. Manual publication makes no AgentRuns ledger calls. Accepted
links/charges remain consumed; manual wins leave the old preparation available
only to the existing proven-abort cleanup. No schema or durable-body changes.

### Task 8 implementation amendment: final prepared request admission

The existing gateway adapter is `_StreamingModelAdapter`; its new optional host-only
`review_request_admission` context binds a reject-only callback using the existing
thread-local call scope. It captures that callback before crossing async/thread
boundaries and invokes it after actual `PreparedProviderRequest` continuation/thinking
preparation, before dispatch. It never travels through model/tool kwargs or mutates the
request. The callback consumes existing prepared accounting/capacity for the effective
selected destination, including fallback. Direct AgentService requests use the same
admission contract and existing estimators; unknown capacity is refused. Review-only
preparation disables generic safety-window truncation, since unacknowledged protocol
history may not be silently discarded. Ordinary preparation is unchanged.

Direct private-continuation calls lacking this safe prepared seam fail closed with a
request-level `review_context_reduction_blocked` gap. They retain owner groups and
accepted dispositions, finalize missing units unresolved, and never reset continuation
or launch another verifier. This narrow bridge extension avoids a second tokenizer or
private request shaper. It is required because pre-continuation estimates cannot
establish the final billed request's capacity. The host-only original map, immutable
batch pages and accepted disposition checkpoint remain in charged session scratch;
generic durable events and live callbacks do not receive original attribution.


### Slice 9 implementation note

The existing Raw CLI executor has no aggregate generated-byte/file reservation
boundary. Stock executable review checks therefore resolve unavailable before
copying or launching. A narrow optional capability belongs only to the actual
external executor; the focused `Agents/code_review_checks.py` state module adds
no executor or scheduler. Tests use a fault-controlled finite external boundary
through native RawCliRuntime admission and must not be cited as stock sandbox
proof. Proposal unit IDs denote related claims: executed evidence requires an
actual receipt read by the same live verifier, while independently supported
source-only decisions remain permitted. Native report presentation uses a mounted
session reader before acknowledgement; saved transcript references are not durable
report bundles. The existing explicit verification grant needs no extra prompt.

The existing RawCliRuntime final-admission seam accepts an optional host-only
review scope, absent from ordinary execution and model arguments. It serializes
the review close fence with native commit/accepted receipt while retaining existing
runtime authority checks. The review lock is not held across approval or process
lifetime. Prepared invocation cleanup stays with its context; accepted/uncertain
cleanup stays with the actual native owner.


### Final review correction: close and verifier completion

The immediate close fence and exact scratch revocation remain independent of the
publication lock (ruling 40 in the QA/controller ledgers). Final native admission
already entered before that fence retains in-flight ownership through native
commit/settlement; only its actual accepted callback creates acceptance evidence.
No admission entered after the fence is accepted. An uncertain in-flight process
keeps its reservation until the native owner proves cleanup, without automatic
retry or refunded budget. This refines the slice-9 serialization description.

Verifier completion intent requires its own fully delivered/disposed unit ledger
and no request gaps. Successful terminal and capture receipts still govern sealing.
A reasoned unresolved disposition accounts for its unit; host-filled missing work
does not manufacture intent or a complete seal. Reviewer coverage remains separate.


### AC23-B qualification dependency: fixed worktree-root preparation

The ordinary isolated-fleet regression gate also qualifies the ADR-126/ADR-228
amendment for independently preparing the fixed profile `agent_worktrees` root
while retaining its exact parent AgentRuns operation. It changes no review role,
source, budget or acceptance authority. AC23 still requires the complete focused
regression/resource evidence; AC24 live-provider qualification remains separate.


### Native qualification identity repair (2026-10-27)

TASK-35582's custom native coordinator exposed the existing raw-selection versus
canonical-execution/origin-versus-base handoff mismatch. ADR-063's selected
destination continuity amendment and ADR-147's durable primary selection amendment
govern its repair under ADR-179's engine boundary. This evolves existing provider
continuation JSON and frozen destination JSON; it adds no durable review-job owner
or recovery, route registry, permission or child budget. Code-review source/role
leases and expiry still apply independently of any generic provider checkpoint.


### Native Capture On compatibility amendment (2026-10-27)

TASK-35582 also applies the [ADR097 live-native trace amendment](097-console-reference-backed-semantic-trace-ledger.md)
for exact captured native tool-loop evolution. That existing-owner trace repair
adds no model/tool authority, provider substitution or durable review-job recovery.
Its focused design is adopted before source implementation, with strict custody,
atomicity, historical reconstruction, retention and linear-storage evidence.


### Accepted review-wake provider/catalog preflight amendment (2026-10-27)

The latest real run sealed/presented both reviewers, but V1's accepted wake failed
locally before HTTP: strict hosted ownership rejected split SYSTEM carriers, and
unverified first-request planning independently removed coordinator tools. The
[focused design](../../Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/review-wake-repair-design/design.md)
and [controller adoption](../../Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/review-wake-repair-design/controller-adoption.md)
are adopted before implementation.

Only the exact accepted host review-boundary authorization may obtain bounded
current capacity from its actual pinned resolved destination before normal schema
planning. Existing cache/singleflight and outer preparation/automatic-chain limits
apply. Revalidate receipt identity, accepted ledger context, grant, source/epoch,
context/session/Stop and priority after await. Hold no native/publication lock
across metadata I/O; source checks use their existing proper owner. No config
re-resolution, stale capacity injection, fallback route or new permission.

After normal planning, require actual scoped spawn plus required admitted-stage
review operations, directly disclosed or reachable through existing discovery.
Unknown/failed capacity, genuine small window or current permission floors may
refuse visibly before generation/child dispatch. Never silently send a mailbox-only
continuation, force tools, renew grants or refund/replay an accepted attempt.
Ordinary primary/progress/Goal/queued Send keeps existing nonblocking resolution
and unverified bootstrap behavior; R57's QA correction remains valid separately.

SYSTEM automatic notice identity/order is preserved through the existing supported
ordered carrier, under ADR063/179 clarification. ADR230 owns bounded preparation;
ADR134/135/147 finite acceptance/routing remains unchanged. No SQL/persisted JSON
version, durable review-job owner, registry, scheduler or extra live allowance.
Required proof includes configured Console→normal handlers→fake physical HTTP,
expired/cold pinned metadata, sealed/presented reviewers, one fresh verifier and
[2,1] schedule, plus strict lifecycle/target/permission/carrier/capture negatives.


### Source correction to accepted-wake carrier diagnosis (Ruling63)

Exact BASE7952422 controller source deliberately projects automatic SYSTEM
transcript notices as payload-only trailing USER while preserving AGENT_WAKE
authority and order. The prior reconstructed SYSTEM-carrier diagnosis was not
the actual wire path. Its carrier conversion/Safe-capture clauses are superseded;
ordinary serialization and the strict engine stay unchanged. Accepted-review-only
pinned capacity/revalidation and normal schema reachability requirements remain.
The full handler-backed fixture instead exposes completed checkpoint forwarding
without native replay rows. A focused replay contract is pending before source
changes at that extra boundary. Original diagnostic evidence is retained with
QA/ac24/review-wake-repair-design/diagnosis-controller-correction.md. No authority,
schema or additional live allowance follows from this correction.


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


Implementation evidence through c3f99ec: the corrected accepted-wake/completed
replay contract passes scoped independent review after transient-credential and
incoming-owner fixes. Canonical persisted replay, eligible MANUAL Safe/Full,
strict negatives and final binding are covered in
Docs/superpowers/qa/2026-10-18-multi-model-code-review/ac24/review-wake-repair-fix/.
One nonblocking precedence-fixture coverage gap and retained pending-refusal
shutdown limitation remain explicit. No automatic capture-policy, durable format,
codec/GC, execution authority or recovery expansion follows. Live qualification
remains incomplete.
