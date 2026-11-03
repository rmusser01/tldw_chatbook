# ADR-285: Native phased agent workflows in a shared worktree

- Status: Accepted 2026-11-01 (implementation in progress; runtime qualification pending)
- Date: 2026-11-01
- Task: [TASK-38493](../tasks/task-38493%20-%20Design-native-phased-agent-workflows-in-a-shared-worktree.md)
- Spec: [Shared-worktree workflow design](../../Docs/superpowers/specs/2026-11-01-native-phased-agent-workflows-design.md)
- Plan: [Native shared-worktree workflows](../../Docs/superpowers/plans/2026-11-01-native-phased-agent-workflows-implementation.md) (implementation in progress)
- Implementation: Tasks 1–7 independently reviewed; remaining delivery and qualification are tracked in the linked plan.
- Governing contracts: [project instructions](069-console-project-instruction-local-state-and-preflight.md),
  [Change Review consent](084-change-review-consent-and-asynchronous-finalization.md),
  [per-turn review ownership](089-console-per-turn-change-review-ownership.md),
  [turn lifetime](094-console-turn-lifetime-and-navigation-boundary.md),
  [CLI boundaries](094-raw-and-virtual-cli-execution-boundaries.md),
  [backup/recovery](126-complete-local-backup-and-recovery.md),
  [physical admission/budgets](134-fleet-admission-and-automatic-work-budgets.md),
  [fleet delivery/recovery](135-fleet-completion-delivery-and-crash-recovery.md),
  [portable Workflows](138-portable-workflow-definitions-and-local-execution.md),
  [provider routing](147-agent-provider-routing.md),
  [fork/spawn](150-agent-chat-fork-and-spawn.md),
  [design language](150-design-token-system-and-design-language.md),
  [component patterns](161-component-pattern-library.md) and
  [Goal runs](227-native-console-goal-runs.md).

## Context

The user requested native dynamic workflows: a lead plans phases, agents perform
parallel work and later phases use their results. Coding is first, research next,
then general tasks. The user requires one shared worktree, approval of the
initial plan and every subsequent phase, saved progress and explicit restart
resume. Existing Console owns provider execution, fleet tasks, permissions,
physical operations, accounting, change review and recovery.

Source review found six gaps in the first design: instruction-discovery paths
were mistaken for task-scope enforcement; disjoint outputs were mistaken for
independent tasks; typed workflow admission was incomplete; existing absolute
deadlines consumed approval waits; exclusivity omitted other native activity
using aliases of the worktree; and required results depended on optional logs.
Exact evidence freshness and a bounded plan format also needed explicit rules.

## Decision

1. Add app-owned phase coordination under ConsoleRuntime. Keep execution in the
   existing controller/bridge, AgentService and FleetCoordinator. Share the
   established child launch path while preserving its captured routing, hooks,
   permissions, accounting and cleanup; do not introduce a second executor.
2. Plans are bounded structured data: ordered phases, tasks, dependencies,
   declared input/output footprints, tools, model targets, checks and finite
   limits. No generated orchestration code or general DAG language. Planning is
   a durably accepted read-only attempt charged to the workflow.
3. Add explicit workflow origin, private phase authorization, stable task launch
   claims and policy selection through the existing automatic-work ledger and
   native admission path. FULL acceptance precedes billable work/effects.
   Repeated approvals/launch delivery are idempotent. Workflow-owned children
   cannot independently trigger ordinary fleet-wake synthesis.
4. Task scope narrows current permissions at discovery, runtime guard and final
   invocation. Path-awareness is not authority. File writes reuse existing
   atomic preconditions, including edit/patch observed digests; blind workflow
   writes and alternate writer/recursive/runtime escapes refuse. Opaque tools
   require planned exclusive mode and existing permission/trust decisions;
   serialization does not constrain their actual OS or remote authority.
   Effectful command/MCP hooks additionally require proven native ownership
   through physical drain; current unowned notification workers make workflow
   preparation refuse those hooks, including read-only planning. Existing hook
   configuration and ordinary execution remain unchanged. Workflow skill scripts
   also refuse, even in exclusive mode, until typed process/reader cleanup proof
   and owned lifetime are available; exit status or warning text is not proof.
5. One runtime-owned worktree activity owner coordinates full read/write
   footprints before launch, including aliases, nested roots, ordinary turns and
   survivor children. Read/write dependencies serialize. Acquire complete
   footprints and capacity without partial waits; retain claims through actual
   physical ownership. External writers remain an attribution/freshness limit,
   not a claim of filesystem isolation or multi-process scheduling.
   Opaque-capable native work acquires a runtime-wide opaque-effect reservation
   before root-specific claims, since its cwd does not bound its effects.
   Restriction state is run/task-bound even when providers are shared.
6. Workflow time policy charges the union of active execution intervals and
   excludes only durably quiescent approval/capacity waits. Alive/unknown cleanup
   remains charged. Counters and uncertainty survive restart; resume/replanning
   never refills them. Existing fleet and Goal absolute deadline policies and
   Goal's no-child contract remain unchanged. Use workflow-specific finite
   settings and immutable captured policy through the existing accounting owner.
7. Store revision/control state, decisions, phase/task attempts, active intervals
   and bounded private required results in AgentRunsDB. Reserve result capacity
   before dispatch and persist required handoffs independently of diagnostic
   logs. Reuse Goal payload-admission/retention patterns and exact-manifest checks
   without relabeling workflow execution as a Goal. Keep detailed native logs
   optional. Retain accounting/idempotency tombstones after settled removal.
8. Bind phase decisions to exact plan/control revision, predecessor results,
   current relevant inputs and verifier evidence. Progress cursors are separate.
   Revalidate after waits and restart; stale/missing evidence is not a pass.
   Use actual typed process outcomes and positive drain, not model success text
   or a logical terminal run row, to settle work and enable downstream checks.
9. Reuse Change Review consent, bounded root-ordered windows, actual snapshot
   references and disclosure of exclusions/failure/concurrent attribution.
   No automatic capture enablement, merged fictional per-task diff, destructive
   Stop or automatic Git rollback. Existing Undo remains separately guarded and
   confirmed against live writer occupancy.
10. Reuse the one Console startup recovery audit. Opening services/history never
    starts a global sweep. All restored workflows require explicit Resume;
    unknown effects retain conservative charges and cannot replay automatically.
    Update exact installed schema catalogs, backup participants and recovered
    activation with any new storage. ADR-244's planned host fence is not treated
    as implemented or as a hidden dependency.
11. Use existing Console command, form/modal, status/subscription, fleet and
    Changes patterns. Widgets hold no execution authority. Initial-plan and
    phase approval, planning revision, Pause, Stop, final review and recovery
    remain distinct controls. Navigation detaches views; shutdown closes
    admission and drains owners. The user approved the final UI, failure and
    verification contracts in spec sections 10–12 on 2026-11-01 and authorized
    implementation planning.

## Alternatives

| Alternative | Reason not selected |
| --- | --- |
| Per-task worktrees and an integration checkout | User requires the same worktree; introduces unwanted merge/apply work. |
| Extend portable Workflows first | Expands ADR-138's sequential executor before the requested Console experience. |
| Reuse Goal iterations directly | Goals currently prohibit children and retain absolute elapsed deadlines. |
| Submit ordinary user turns or invent survivor wakes | Misstates origin, budget, draft and result-claim ownership. |
| Scope only in prompts or path-discovery hints | Does not enforce actual mutation authority or prevent alternate tools. |
| Treat native logs as required result storage | Logs may be disabled, truncated or fail independently. |
| Hold a whole-worktree lock while waiting for approval | Blocks legitimate manual work and makes overnight review monopolize resources. |
| New orchestration runtime, event bus or arbitrary DAG language | Duplicates existing owners and increases first-release scope. |

## Consequences and qualification

### Finite native snapshot qualification (Task 8D1)

App-owned coordinated Change Review jobs acquire finite runtime-wide opaque
activity before discovery, configuration validation and Git work. Inherited Git
configuration, includes and ignore files extend beyond selected roots. A covering
original opaque tree claim may be borrowed; bounded claims cannot be upgraded.
The actual native tool operation survives worker return until the original
publisher callback returns. Baseline timeout, cancellation and shutdown deadlines
are not physical drain. Discover/baseline handoff is conclusive observational
completion; it does not certify child effects or phase completion. If both final
publication attempts fail, physical observational work can retire with a distinct
unresolved disposition. Task/phase durable settlement remains separately required.

Native capture requires Git >=2.36 and refuses configured clean/process/smudge
filters and promisor/partial-clone remotes before helper-capable entry (including
implicit fetch from object lookup). Each invocation independently disables
hooks, signing, fsmonitor, automatic GC/maintenance and external diff/textconv.
Private built-in version/config discovery and the actual command share the same
finite opaque job. Global ignore rules, historical contents and restore semantics
remain unchanged; no persistent attribute override or user-config edit is used.
This is coordination of known native work, not an OS sandbox against external
processes editing configuration. Standalone legacy tracker behavior is unchanged.
Capture-capable ordinary tree admission and phase wave/window integration belong
to Task 8D2; capture-enabled workflow preparation remains refused until then.
Consent's separate `_initialize_root` / `initialize_shadow_root` snapshot uses
the original feature source and native qualification described below.

### Native Git physical completion (Task 8D2 Git custody)

Selected Git may be a legitimate delegating wrapper; successful version/config
output and direct-child exit do not prove descendant completion. All three native
launches (version, config, command) use the existing low-level process-tree
controller on the original synchronous worker or publisher thread through a local
asyncio Runner. The same initially opaque ticket/source/operation remains active;
no executor, replacement owner, queue or reaper is introduced. Guarded storage and
SQLite work stays on its original synchronous activation, outside the transport
coroutine. Existing argv, environment, inherited stdin, byte/text behavior,
timeouts and helper suppression/refusal remain unchanged.

Completion joins actual stream collection, containment-empty proof and handle
closure, original worker return, original host acceptance/discard, and the return
of operation.finish before root completion and claim retirement. Borrowed tickets
never settle the covering parent's claim. Its future producer must keep that
parent root/source/storage live until the original ticket.retired; premature
parent finish is outside this producer's contract. Timeout or exceptional command return
requests bounded TERM/KILL; a still-observable tree then uses the same controller's
late wait under the original worker, source/storage scopes and stopping operation.
Delay alone is not irreversible cleanup uncertainty. If proof itself becomes
unavailable, cleanup is marked unproven and the original worker/scopes/operation
remain retained. This can hold disposal indefinitely; leader exit, readiness,
publication or a cancelled shutdown waiter cannot manufacture proof.

Only initial native Change Review attachment uses the existing
native_workflow_qualified platform gate. Unsupported platforms retain ordinary
consent/coordinator/Git behavior, while qualified source loss cannot downgrade.
Managed groups are the proof boundary, not resistance to OS-level escape; no new
Windows native qualification is claimed. Costs are process-group/Runner overhead
and retained finite capacity/storage while cleanup remains unresolved. Historical
Undo checkout/filter semantics are unchanged and remain separate work.

### Consent initializer ownership (Task 8D2A)

The app-owned consent initializer uses its existing bounded queue and workers,
with a separate original feature source: configuration, registry/database,
binding, consent revision/generation, initializer and shadow storage. Source
capture uses the existing metadata-only data-path selector; only the admitted
worker verifies and secures that same path, refusing profile or selector drift.
Native dispatch takes finite opaque activity and one actual tool operation from the
original RuntimeCapacity; empty conversation metadata denotes a non-Chat feature,
without a model, child or AgentRun. The same qualified Git boundary and finite
worker/handoff retirement apply; D1's Chat-source checks remain strict.

Settings enable/retry is manual; lazy configuration and binding observers are
automatic and preserve reserved manual capacity. Busy admission produces existing
failed/retry readiness. Concurrent opaque initializers may require Settings retry;
there is no new waiting queue or polling retry. Source reads/callbacks and native
admission run outside the consent bookkeeping lock. Shutdown seals/discards
unstarted work before capacity closes, retains already admitted callbacks and
original readiness/discard storage until actual return, and never treats a
bounded join as drain. READY requires current original consent/binding and a
successful initializer; failed or stale outcomes are not capture success.

This dependency stage preserves lazy initialization but does not establish early
ordinary-tree admission. The next ordinary producer must pass its exact covering
opaque claim and reserve initializer operations before enqueue and parent finish;
independent reacquisition behind that tree would self-conflict. Ordinary hooks,
references, child lifetimes and workflow phase/window integration remain pending.

This is a substantive extension to admission and shared-worktree coordination,
not merely a plan prompt. Existing captured authority, physical retirement and
database durability remain load-bearing. Resource bounds may pause useful work;
the UI must preserve results and explain the exhausted limit. Shared execution
does not guarantee consistent views against external programs, exactly-once
remote effects, full Git rollback or execution while the app is closed.

The spec maps each review correction to real-file/SQLite/native-dispatch tests,
duplicate-decision and held-worker cases, active-time/restart controls, optional
logging controls, snapshot/privacy failures, mounted interactions and a real
failed-command → correction → passing-command journey. Runtime qualification is tracked by the implementation plan and is not implied
by this accepted decision. Link this
ADR from the implementation plan and task notes; select the
next actual schema version only when implementing.

### Finite legacy hook event custody (Task 8D2B1)

The app's legacy hook engine admits one complete finite opaque event before its
existing queues accept any target. Its genuine non-child native owner consumes
no coordinator slot; every actual target reserves its own tool operation. The
original runtime, engine, permissions/configuration and session source stay
attached through physical command cleanup and conclusive host outcome delivery
or discard. First-deny may return while siblings retain their operations.
Process pipe closure, timeout, caller cancellation and bounded shutdown are not
positive process-exit and managed-group-absence evidence. Unknown cleanup keeps
the original opaque protection. Admission follows existing event failure policy.

Settlement satisfies every applicable existing required write, plus the original
host handoff; an event with no required durable payload has an explicit empty
obligation set. Optional hook transcript persistence stays optional. Hooks remain
arbitrary effects; no hook journal, AgentRun kind, fabricated Chat, crash sweep or
per-handler durable write is introduced. Runtime-only coordination does not
exclude unrelated ordinary orphaned OS/remote effects across an app crash.
Existing workflow task/result/checkpoint/accounting durability remains mandatory.

Shutdown seals legacy ingress before closing capacity and retains actual worker
and host handoff drain through cancelled waiters. Standalone engines retain their
existing API and policy. V2 planned events, MCP nested inheritance and SessionEnd
teardown are the mandatory next D2B2 slice, not implemented by D2B1. Subsequent
ordinary-tree admission must replace independent event acquisition with original
covering-tree borrowing before enqueue, alongside D2A initializer borrowing.
Workflow capture/planning/coordinator admission remains closed.

Native workflow execution requires the presently qualified POSIX process-group
custody capability. One metadata-only qualification predicate governs native hook
attachment and the process-local phase authority factory, which refuses before
phase acceptance or attributable work on unqualified platforms. Enabling the
workflow setting cannot bypass this boundary. Ordinary Windows hooks retain their
existing granted execution path and its limits, without new native custody or an
invented cleanup guarantee; passive workflow storage/history remains portable.
Future public planning, Start and Resume must use the same qualification before
billable preparation or acceptance. Task 9 will surface the refusal reason. This
is a platform capability boundary, not a new user setting or permission grant.

### Native plugin hook preparation qualification (Task 8D2B2a)

Plugin hook configuration is not a pure ceiling read: Console preparation calls
the existing PluginService admission/storage callbacks before a hook engine or
lifecycle exists. The original preparation source and worker-published selected
maximum therefore own a finite opaque native operation before those callbacks
are queued, through original NativeHooks acceptance or conclusive discard.
Cancellation of an awaiting Future is not original callback retirement. Runtime
and session close seal further preparation and retain actual callback and host
return, including native operation-release callbacks, before storage disposal.
Origin comes from the original trusted submission, never event/session/run text
or the absence of automatic context. Empty plugin selections acquire no worker.

This finite request reuses the existing plugin worker and admission policy; its
completion does not retire or replace PluginRuntimeOwner's persistent storage
infrastructure. No new journal, queue, executor, poller or permission framework
is added. The same native platform gate preserves ordinary unqualified behavior.
Planned v2 handler/MCP/nested/host/SessionEnd ownership remains mandatory D2B2b;
ordinary-tree borrowing and workflow capture/planning/coordinator stay closed.


### Native command-only v2 event qualification (Task 8D2B2b1)

The next finite event boundary covers complete command-only v2 events, including
planned, individual, notification, lifecycle, tool and compaction consumers.
Before effects, the original host freezes the complete potential handler maximum.
An event with any reachable MCP handler keeps its entire existing execution path;
it has no partial native claim. This temporary coverage decision is private and
immutable: event IDs, source loss or later matcher changes cannot downgrade a
covered event or widen its maximum. Existing platform qualification still applies.

Each covered event retains one opaque non-child native owner and the original
operation for each actual command target before existing queue acceptance.
NativeHooks' published selection identifies plugin targets; their original
admission and process-owner callbacks execute under that custody. Original
process exit, group absence and required plugin settlement must join actual host
publish/checkpoint/transform/compaction acceptance or conclusive discard. The
original Stop proposal consumer retains this hold through its existing queue
acceptance/discard; later ordinary durable admission remains its original owner. A scope
close, timeout or cancelled waiter is not that join. Empty event journal
obligations remain explicit; no ordinary journal or new execution framework exists.

Authorized SessionEnd work retains its operations while capacity is open, then
ordinary ingress seals and shutdown drains the original workers and host handoffs.
Unknown cleanup or failed required writes retain protection. D2B2b2 must close
the remaining MCP event/nested preselection boundary, including exact nested holds
before capacity closure; there is no post-close reserve exception. This extra gate
temporarily leaves MCP-containing events without native coordination. Full ordinary
borrowing and workflow capture/coordinator execution remain incomplete and closed.

## Task 7 producer and storage contracts

Task 7 uses additive AgentRuns schema v27 for immutable required checkpoint
sidecars and inert Resume command identities. Empty verifier definitions and
absent decision evidence are omitted narrowly, preserving shipped v25/v26 JSON
bytes and hashes, including records at the 128 KiB bound. Native observations,
required handoff and result share the existing per-attempt reservation; optional
run logs cannot substitute for them. Checkpoint, result, task retirement and
active accounting commit in one existing FULL transaction before releasing the
original physical claim.

Approved `VerificationSpec` entries name a selected-root executable wrapper,
exact arguments, source SHA and input paths. The host derives its shell command
with `shlex.join`; only the actual selected native raw-shell provider may supply
proof. Existing raw-shell arming, permissions and exclusive task scope still
apply. Required sanitized output is captured before preview/optional-log bounds;
normal native stdout and stderr EOF, native cleanup and the exact typed outcome
are independent requirements. Overflow or missing tail cannot certify a check.
The existing 1 MiB wrapper digest limit remains; exact resolved shell provenance
uses the existing bounded 16 MiB manifest reader and is checked against the
actual native result. Skill scripts remain unsupported.

The original native child tail supplies typed failure receipts for known
no-worker setup/thread-start failures, including durable rows created before
scope binding. Receipts retain the actual original owner and activity claim;
unknown row/cleanup state retains protection. Saved IDs and a caller's drained
boolean cannot manufacture physical retirement. File inputs supply approved
path/manifest references, not eager bodies; actual reads use current `fs_read`
permissions. Complete persisted predecessor handoffs are injected task-only.
Same-phase output references bind the approved producer definition at approval
and require fresh completed native producer checkpoints before consumer launch.

Upcoming phase approval, completed-phase quality review and execution recovery
have separate meanings. Passive service Resume validates clean saved state and
returns awaiting approval; it grants no execution. The later public coordinator
must freshly prepare real provider/tool/permission custody before acceptance.
The one startup owner may close prepared unused or positively quiescent old
custody, retaining spent counters and completed results. Positive quiescence is
persisted only after actual coordinator drain and is reset on interval reopen.
Unmarked accepted, open intervals, corrupt/missing native lineage and unknown
effects remain recovery-required. A fresh later phase attempt may continue
remaining tasks; no old capability is adopted or accepted generation refunded.

### Native MCP event and nested teardown custody (Task 8D2B2b2)

Extend the original complete event owner to its metadata-selected reachable MCP
workers and conditional nested PreToolUse/PostToolUse/PostToolUseFailure workers.
Retain exact original context, registry/provider, schema, permission and causal
selection before queue publication; arguments/results cannot widen this maximum.
Each distinct reachable worker reserves its existing native operation while
capacity remains open; the same worker and sequential validator share one. The
existing causal/dependency limits and native capacity bound whole admission,
including conditional candidates. Disabled, impossible and selected DENY targets
create no holds; ASK remains subject to normal approval. Nested consumers borrow
only the exact original live branch, never a fresh opaque claim or returned hold.

Original MCP worker return, actual dispatch bridge finally, original plugin request
terminal/reaper and required settlement return join the existing host handoffs and
native operation.finish return. Persistent plugin infrastructure retains its own
owner; one request cannot prove it drained. Never-issued descendants settle only
after original refusal/discard or producer return. Unknown cleanup and failed
required writes retain stopping capacity/source/activity protection. No journal,
executor, queue, poller, permission store or budget ledger is introduced.

SessionEnd retains the full authorized prospective tree before capacity close.
Later nested execution uses those holds without post-close reservation. Existing
configured runtime disposal still revokes MCP currentness; custody supplies no new
terminal permission. Positive live-lifecycle execution after capacity close is
qualified separately from actual disposed-runtime refusal and physical drain.
Conservative conditional reservations may refuse events that previously ran
without native coordination. Standalone/unqualified behavior and shared platform
qualification remain unchanged; no Windows physical proof is claimed. Outer
ordinary-tree borrowing and public workflow capture/coordinator stay incomplete.


Accepted-result lifetime qualification (Task 8D2B2b2): exact original live context
registration remains mandatory for capture, admission, worker start/currentness
and nested borrowing. After actual host return, operation release and positive
native owner drain have conclusively retired the original custody, passive
accepted-context revalidation may use its retained original selection without
requiring the normally spent per-input registration. A retired custody flag alone
is insufficient: the existing owner drain observation must be positive. Provider,
schema/catalog maximum, plugin/permission/runtime/store and activation checks stay
in force. Closed scopes and returned operations cannot dispatch or be borrowed;
no caller flag or replacement context grants authority. This preserves legitimate
later Send reuse of accepted runtime context without reviving launch permission.


I1 permission qualification (Task 8D2B2b2 fix1): passive context retains the actual
MCP provider's original tool, arguments, grant decision, profile and exact service/
permission-store sources. Refresh only that original permission policy before
passive input preparation, outside bookkeeping locks; never reactivate spent
work, dispatch, prompt or consume a grant. The existing resolved-path permission
store fence publishes one shared quiescent token across same-path instances;
policy mutation invalidates it before effects, and uncertain writes remain
unavailable. Fresh successful reads may preserve unrelated permission edits;
deny, kill switch, profile/argument/session-grant narrowing or read/source
uncertainty refuses. Final context publication rechecks only original identities
and the positively observed token. No new permission store/ledger or passive-mode
flag. Cost is bounded original-policy reads per passive preparation and
conservative refusal during concurrent mutation. ADR required: no new ADR;
existing ADR-285/163 source-revocation qualification governs this correction.

### Finite explicit-reference custody (Task 8D2C)

The original Console expansion worker freezes its existing explicit-reference
readable roots, launch cwd, registry/workspace/run-scope selection and merged
sensitive/exclusion policy once. A missing configured sandbox is not created and
contributes no root; independently authorized existing folders remain. Captured
absence never falls back to a newly selected workspace. Root/source/selection
changes refuse; exclusions can narrow but removals cannot widen the capture.
Stock file/folder reads claim complete bounded READ roots. Reachable Git and
unknown effectful callbacks use initially OPAQUE coverage. The single finite
non-child tool operation owns the real preparation worker and original host
acceptance/discard through operation.finish return. No model or durable event
row is introduced. Later ordinary producers must consume their original covering
claim before enqueue; no unused borrowing API is added at this stage.

Reference Git shares the gated native qualification and managed-tree transport
over its actual executable/repository prefix, without constructing a shadow repo.
Both captured cwd and actual canonical repository top-level must fit the same
readable maximum before body output. Whole-repo diff and existing literal
exclusion pathspecs run at that top-level. Version/config/command processes all
retain the same original worker/source/storage through tree and stream proof,
host return and operation.finish return. Unsupported platforms preserve ordinary
behavior; qualified source loss never downgrades. No proof retains original
custody, potentially indefinitely. Costs are source/metadata checks, qualification
and containment overhead and coarse Git contention. Ordinary-tree borrowing,
Undo, RawCLI and phase/wave/window/coordinator integration remain separate work.

### Direct-user native RawCLI custody (Task 8D2)

Trusted Console submission captures original app/runtime/store/session/invocation before scratch allocation or worker enqueue. It admits one manual non-child OPAQUE owner and one tool operation, then binds only the actual local_command run. Temporary/nonpersistable refusal and exact draft/attachment restoration remain. Original source loss refuses, never reselecting another runtime, profile, DB or scratch root.

Local-command row and terminal step writes are required. Successful native writes or exact original row/status/canonical-step readback establish settlement; False, a prior terminal status or finalized latch alone does not. Partial create/initial-step failure observes the same preselected run identity and required error outcome. No new retry/replay producer exists. RunLogWriter stays best effort. Unobservable required outcomes retain original storage, worker and activity custody.

Qualified direct/model/Task7 execution joins actual containment and existing admission/queue-relay helper returns using the same already-reserved operation and captured storage sources. Model handoff comes from exact catalog/concrete provider/original runtime and actual worker operation, never names or IDs alone. Standalone/unqualified execution remains bounded and reports cleanup-unproven when helper return is absent. Returned cleanup proof and required-output sealing follow helper proof. Original synchronous host acceptance or source-validated discard and operation.finish return precede root/claim retirement.

Disposal seals admission before capacity close and scratch tombstone, retaining original storage/scratch leases until real drain despite repeated cancellation. Costs are coarse OPAQUE contention, source/DB readback checks, a small private RawCLI handoff, and potentially indefinite native worker/storage/disposal retention when process/helper/required-write proof is unavailable. No second executor, queue, journal, permission grant, model/child owner or public workflow entry is introduced.


### Confirmed historical Undo custody (Task 8D2)

The original app Runtime binds a separate Undo issuer at bridge/provider composition. Actual Review or Console-card confirmation consumes the exact captured rows, paths and fingerprints; saved IDs and direct provider calls grant no authority. Finite OPAQUE preflight reads retire before the human modal. Confirmed single-file and multi-root Undo acquire one complete OPAQUE claim and one manual nonchild tool operation before original effect entry. Native activity/recovery/capacity conflicts refuse without a queue; active-session refusal remains additional. Original DB/conversation/row/root/source and disk fingerprints are revalidated under custody. Qualified source loss cannot downgrade to unowned work.

Historical checkout/filter and snapshot semantics use the existing native Git process-tree transport independently of D1 qualification. Once mutation is attempted, even partial per-path failure requires the original post-revert snapshot. Successful paths require exact original reverted-path-union DB readback; a committed-then-raised update can be proved this way. A snapshot exception has no supported replay or arbitrary-HEAD proof. Unknown required outcome delivers honest partial results then retains the same worker/source/storage/OPAQUE protection indefinitely; no new journal, retry producer or poller is introduced.

The original host accepts results or conclusively discards stale presentation before operation.finish RETURN; original worker activation, DB handles, shadow storage and any actual scratch-generation lease survive that return. Existing scratch lookup is metadata-only and cannot allocate: tombstoned/lost previously recognized generations refuse instead of becoming unmanaged. Admission/start fences seal unstarted work during disposal; already-started original obligations may finish on their original retained source, without new confirmation, replacement-source permission or liveness inferred from a cancelled waiter. Costs are coarse contention, exact source/readback checks and potentially indefinite disposal retention after unprovable cleanup or settlement. This qualifies only the existing confirmed Undo feature; ordinary tree, wave and public workflows remain pending.

A started thread whose original activation cannot enter may retire as unused only after all partial scopes close normally and the same thread observes operation.finish return. The effect callback has not entered; no required mutation exists. The same distinction covers sealed unstarted tickets. Callback-body failures, uncertain scope unwind and unknown finish return retain custody. No replacement-source retry is permitted. The existing preparation-read hop performs only source/DB/capacity admission before original effect-worker enqueue; its extra pool handoff is an explicit cost.

### Prospective ordinary agent storage sources (Task 8D2 partition A)

The original issuing submit selects its agent storage sources before SessionStart
effects. A private live carrier retains original runtime, bridge, DB, permission,
registry and scratch identities; copied configuration carries no live custody.
The existing preparation and native worker handoffs reacquire the same source
tuple in their actual task/thread, including the selected run-log confinement
root. The bridge consumes that root instead of selecting it again. A recognized
stock writer observes the same source before and after bind, and validates its
actual bound descendant against the original writer/root/bind identity. A changed
writer, root or late source refuses; path containment alone does not qualify it.

Retained MCP context still requires exact original source equality, fresh original
permission checks, and positive original event drain for passive use. No guard
reset, tuple-subset fallback or spent execution revival is allowed. This repairs
the complete Console-to-child-model source path; it does not provide early
physical admission before references/initializers, ordinary-tree borrowing,
child write settlement, SSH ownership or late-event retirement. Those remain
required. Cost is prospective metadata selection, retained original storage and
conservative refusal when original sources change; no new owner or queue exists.


Task8D2 A effective project-authority qualification: for the original stock
capture, hook context identity is workspace, enabled state, the full selected
binding snapshot and complete options. Keep every binding/root/locator identity,
write flag and exclusion. Persisted choice/fingerprint and consent-notice
bookkeeping are redundant with that authority and do not invalidate unchanged
accepted context when normal first-folder adoption and independently validated
project consent record them. Custom/unknown readers retain their complete values.
Changed selection, disable and retarget still revoke; activation storage tuple
matching remains strict. This is not a new consent grant or ordinary physical
admission.


Task8D2 partition A review fix1 clarifies the same original-source contract:
MCP completion is published only by the actual guarded coroutine's outer
finally, including source-admission entry and unwind; conclusively unlaunched
refusals may retire after real return, while unknown physical/required settlement
still retains custody. Stock run-log qualification covers the concrete segment
path, append/write, manifest and close callbacks before their receiving calls;
a same-root writer with changed callbacks is not stock. Detailed logs remain
supplementary, including the existing disabled-log and standalone paths.


### Native SSH request custody (Task8D2 prerequisite S)

An actual AgentService model fs_* call retains its original native tool operation
through the selected stock provider/RemoteRoot/adapter/executor, every ping/request,
original cache/result host return and operation.finish return. Exact original
request identity, validated remote terminal cleanup and local request/IO retirement
are separate required facts. The same original worker/storage stays retained on
unknown cleanup; timeout, local SSH death and result text are not proof.

Finite native ssh -G and ssh -O checks reuse ProcessTreeController on the original
worker, preserving argv/configuration/timeout semantics. The existing ControlMaster
manager and session registry own healthy shared infrastructure: a private acceptance
observation binds original invocation/source and actual launcher/IO return to real
publication or verified exact adoption. Registration, path existence or zero exit
alone does not establish acceptance. Unlearned/ambiguous setup and unpublished
session starts require real cleanup or retain native uncertainty. Healthy shared
connections need not exit for a request to finish. No new owner, protocol, poller,
permission, binary restriction or master-lifetime service is introduced.

This qualifies existing operation/source retirement, not early ordinary worktree
coverage. Ordinary covering claims, separately scheduled recovery probes, remote
guidance and complete child/late integration remain mandatory. Task7 currently
accepts local roots only and is not SSH positive evidence. Native platform limits
remain; uncertainty may retain worker/storage/capacity and disposal indefinitely.


### Complete ordinary physical tree (Task8D2 integration contract)

Ordinary capture separates original selected metadata from initializer, plugin,
reference and SSH effects. Before input consumption or effect enqueue, the
original turn acquires its complete READ/WRITE/OPAQUE maximum and one primary
native owner. Immutable configuration carries no live authority. Actual callable
reachability uses stock implementations and the selected profile, persona, Goal
and feature gates; ASK remains reachable, DENY does not. Composition may narrow
that captured maximum but cannot silently expand it.

All real preparation, initializer, reference, hook and Git workers borrow this
original tree before enqueue. The bridge consumes that owner at its real primary
row; children attach before row/client/model/thread admission and retain their
originating tree as survivors. Canonical required SQLite row/step outcomes and
physical/source/host/operation.finish return are independent settlement facts.
Unknown cleanup or writes retain original resources; optional logs and terminal
UI/Fleet state cannot establish completion. No new journal or executor exists.

The launch conversation key stays the original persisted ID or original session
ID. Only the original store's committed first-persistence publication may map it
to the genuine durable conversation; explicit rebind/source replacement refuses.
Rows use their real durable IDs, capacity keeps the original launch key, and no
early row, second owner, mutable launch identity or capacity API is introduced.

Reserve genuine finite late E/Stop/child/tool/lifecycle obligations at their
causing producer before effects or capacity closure. Required bundles admit
completely or refuse before effects; original optional notification omission and
dependency/event-control policies remain authoritative. Do not reserve every
hypothetical turn event or allow new reservations after closure. Costs include
real occupied slots and busy refusal at small capacities, coarse OPAQUE
serialization for reachable unknown effects, and potentially indefinite original
storage/disposal retention. Synchronous accepted-turn intake keeps metadata/input
custody; actual attachment transfer waits for original asynchronous admission.
Public workflow coordination and whole-Task8 qualification remain separate.

The ordinary maximum consumes worktree-effect metadata from existing native
provider declarations. LocalToolSpec defaults unknown/custom worktree effects to
opaque; stock declarations distinguish confined paths from private control-plane
effects. Selected-profile DENY may be projected by the existing by-key resolver
(ASK remains reachable). Real final composition must fit that captured maximum;
declaration metadata grants neither tool permission nor dispatch authority.

### Pure declaration correction (controller accepted)

Pre-admission must not call `_default_specs`: it constructs fallback Watchlists services, binds methods and resolves `_deep_search_settings`; those are not inert data. Use one immutable native effect table consumed both by existing LocalToolSpec declarations and the early maximum. No provider, handler, fallback service or description/settings helper is constructed for early projection. Existing frozen configuration and original permission payload supply feature/profile/persona/Goal reachability. The eventual factory still constructs the one real catalog. Unknown factory/spec/handler/service identities require OPAQUE; a caller-set effect field never proves boundedness.

Closed local effects: fs_* and git_* route through the selected original WorkspaceToolExecutor and admitted root descriptors; spills are under original scratch. web_fetch uses in-memory bounded HTTP/PDF/image/ZIP decoding (BytesIO/pymupdf stream), web_search and web_crawl return bounded text with in-memory caches; web_deep_search joins its existing synchronous research pipeline and returns bounded text. These four declarations accept no output locator. Watchlists reads use the original private DB; create/update commands use original service transactions and check/briefing tools accept their existing durable coordinator handoff, not arbitrary replacement callbacks. Todos/characters use their original private stores; optional ask_user is the original UI handoff. The current BuiltinToolProvider only constructs calculator/datetime plus _GATEABLE_BUILTINS: file tools use the explicit Console sandbox, notes use the original private DB; no download/media/artifact tool is currently registered by that provider. Other providers' output/artifact and runtime/script/SSH paths must be classified separately from their actual declarations and roots.

WHY: metadata construction must not execute arbitrary replacements before admission. COST: every native declaration change maintains its shared effect table, source qualification and final subset check; configuration/source widening refuses the held attempt. No second effective catalog or new permission grant.

### Raw and builtin source qualification refinement

Raw CLI projection consumes the exact original app constructor closure, its original helper/code/defaults/captured app, the original RawCliRuntime policy descriptors, app_config identity and literal configured unlock/armed state. It never invokes permitted/_read_permitted for classification; the actual runtime still performs its permission check. Unknown callbacks/runtime produce OPAQUE; a changed source or newly armed/unlocked provider cannot widen the captured attempt. The defining records live with app.py/app_service_wiring.py/console_raw_cli.py, not a caller authority tag. Cost is finite original-source records and a pure recheck before composition/dispatch.

Builtin projection consumes the existing _GATEABLE_BUILTINS table with a worktree-effect field; no tool/provider construction. Enabled native modules only declare classes/functions: file_operation_tools defers sandbox allocation, note_management_tools explicitly defers its directory/DB service creation, and document_expansion_tool fetches bounded original private DB documents. Actual class/factory/descriptor/function identities are retained before the real constructor. Unknown declaration/factory must become OPAQUE without executing a caller-selected module import; final actual tool/root containment and live execution gates remain separate. File reads/writes use actual admitted local roots and original scratch; expand_document and native notes are private DB operations, not arbitrary output locators. Maintenance cost: update declaration/source records alongside a changed native implementation; no duplicate catalog/provider construction.

### Existing personal-context preparation survivor

Controller _personal_context_service enqueues its actual app getter with asyncio.to_thread under the existing setup wait_for deadline. On expiry it returns None while the getter may still run and publish the app's cached service; the existing in-flight Event prevents a second competing getter. Profile provider composition is a second original to_thread producer. Both consume the same ordinary preparation operation/source API before enqueue. A strongly retained original preparation task is shielded under the existing deadline, preserving prompt deadline/no-result behavior without canceling or declaring the worker finished. Existing worker-owned in-flight clearing and the original getter/cache publication remain authoritative. Native getter/app/bootstrap-function source is pinned before dispatch; its final original cache assignment checks the same source. Unknown callbacks contribute OPAQUE without early invocation. No new executor/timer/queue or capacity allowance. Cost: a deadline-surviving original worker occupies its operation and covering claim/source until actual return/finish, potentially indefinitely.


Ordinary MAX Library continuation: original `_library_provider_for_app` selects direct descriptor tools or the RAG fallback from the captured Library authority. Projection must read that original selection and selected app service identities without invoking its factory. Reachable media rechunk with `reindex=true` enters shared RAG/indexing without a closed destination contract and remains OPAQUE. Native note save records Notes organization/publication intents; Sync v2 later copies them to its outbox, while the independent NotesSync observer can derive automatic file updates from note state. No closed original destination/handoff is provided here, so reachable note save remains OPAQUE; this does not classify native read/list the same way. Unknown provider factories/backends/callbacks remain OPAQUE before original construction. Native read/list uses its concrete original class/method/source identity and actual private DB or selected skills-root effect. Final actual provider and callable/source identity must be contained by the original selection; newly reachable services refuse. COST: finite native source records beside the original definitions and original selected service retention; no second catalog, backend inference, NotesSync redesign or new executor.


Ordinary virtual CLI remote consumer (controller accepted): actual VirtualCliProvider per-alias VirtualCliRegistry maps read commands onto the original remote adapter/executor. Retain exact original catalog owner/provider, selected registry/parser/projector/authority/adapter sources before the existing model enqueue; bind the actual selected registry before its execute. Reuse the same original model operation and S request-ID/typed-terminal/local-IO/helper/shared-infrastructure/finish-return proof; no fabricated LocalToolSpec, authority, owner or extra operation. Normal virtual permission/task/alias checks and remote-Git refusal remain. WHY: capture_model_ssh previously selected only fs_* LocalToolProvider despite real virtual remote reads. COST: one finite original virtual source association and receiving equality checks, with unknown cleanup retaining the same original worker/storage. Qualification uses raw source identity without invoking changed descriptors.


Ordinary primary host return (controller accepted): register the exact existing _run_maintenance_agent_call native-host obligation before its original enqueue. Outer submit cancellation/return seals new ingress and requests completion; it does not settle canonical writes while the shielded original native call still runs. Only both original native callback return and outer return permit final canonical readback/primary finish_root, once under either callback order. Existing member/tool/source holds still control physical claim retirement. Cost: one finite native-host receipt and original return callback; no new execution owner, operation, retry, journal or status-based proof. Failed required writes remain retained.


Original ordinary disposal completion (Task8D2): retain only actual admitted _OrdinaryTurn instances in the original runtime, published under its existing execution bookkeeping lock after original scratch admission; a disposal race conclusively retires the unused original admission outside the lock. One private completion observes the existing whole-tree lifetime. Its producer is _retire_scratch AFTER original lease exit and every settlement callback RETURN; the earlier election flag is not completion proof. Runtime disposal waits on these exact completions after original preparation drains and before store/gateway teardown, shielding repeated cancellation beyond grace. Unknown physical return or failed required canonical proof stays pending. Remove a registry entry only after completion. WHY: zero-grace held original operation.finish control showed bounded outer/controller return can precede original tool/child/snapshot/source returns; maintenance-task-only joining is insufficient. COST: one original completion and observer entry per admitted tree, no new owner/operation/executor/queue/capacity API, retry, permission or fresh source authority.


Admission cancellation publication (Task8D2): run_preparation_read delivers cancellation only after the original capture callback physically returns. A genuine owner may therefore already be registered before the awaiting submit observes cancellation. Publish that exact admitted tree into the original submission's finish custody in the await-finally path, including failed return, then let the existing outer finally finish only after original issuer scope unwinds. No unadmitted tree is published; actual source replacement still fails closed. This closes the orphan found by admission-return-red1 without a new owner, replay, return shortcut or early claim release.


MCP detached identity projection (Task8D2): add immutable mcp_definition_sources rows (opaque tool_id, original server_key, raw name, projected LLM name) alongside definition hashes. One private result from the existing original ordered HubTool capture carries both; the public dict-return API remains a wrapper. Use original llm_tool_name/dedupe_names, selected profile/persona and captured source, with no second source read/catalog composition. Native selected-profile DENY filtering already existed; this change closes Goal/final name identity and synchronous fallback selected semantics. Missing/custom structured metadata stays OPAQUE and proves no bounded Goal exclusion. Final native collision wrapper/provider entry must match original fields/hash/alias and captured selected maximum; ambiguous IDs or aliases refuse, never split IDs or remap authority. Goal/config revocation may narrow; changed naming/order requires fresh capture/refusal. COST: detached immutable source rows and one private capture result, no live capability, new owner/operation or alternate catalog.

MCP selection custody (Task8D2): the original received-turn source privately witnesses its exact native capture result and original MCP source, retained in the existing same-task received binding. Ordinary admission may use this witness only with the original receipt, runtime, session and unchanged detached MCP fields; caller-supplied values alone never prove exclusion. Capture failure and unqualified/custom sources remain OPAQUE, while exact native absent or killed sources may be bounded. This is a narrowing observation, not a permission or a live capability in the snapshot; actual provider gates and final identity checks remain authoritative. No second capture, registry or owner is introduced. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: implements the existing private ordinary-custody contract.

Common MCP capture witness (Task8D2): the original qualified native capture publishes one private task-context witness, inherited by its actual submit, for exact controller, session, runtime, MCP source and detached selected metadata. Admission verifies the original source and unchanged MCP definitions/names/profile/persona before using it to prove a bounded exclusion; received intake separately retains its original claim. Unknown or foreign snapshots remain OPAQUE. The witness retains at most one last source identity per live task and no active lease, permission or registry entry. This supersedes the receipt-only association above and preserves no authority in the public snapshot. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: original-source narrowing under the existing ordinary custody contract.

Deferred MCP source selection (Task8D2): custom or unknown manifest/governance callbacks are never invoked for early classification. A detached pending marker plus the one private original capture witness carries their exact source to the same admitted OPAQUE primary; its borrowed operation runs the single original selection through callback, worker and finish return before SessionStart or input transfer. Missing, changed or lost witness visibly refuses, never silently omits tools or reselects. Existing in-memory queued prompt/claim may retain the same original witness through return-to-head; serialization and restoration do not preserve it. Direct, received, queued, Goal, wake and voice use their original custody and source checks; no second capture, owner, registry or live public-snapshot authority is added. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md. Reason: completes existing ordinary admission and source custody.

Cold deferred MCP permission publication (Task8D2): the original installed local-store binding supplies the prospective canonical permission path before selection; the actual native permission-owner publisher prepares exact candidate, installed binding, method and path proofs before its existing publication lock. At that fence, pure identity/slot checks publish that same candidate to the original service, capture and ordinary source. No callback, source I/O or other owner lock runs under the fence; unproven or replaced sources refuse. The original capture witness consumes only that exact publication proof after worker return. Existing ADR285 and source-custody contracts apply; no new ADR or owner.

Task 8D2 original queued receipt handoff: while the exact received owner is still live, seal only its already-completed selected metadata and original source/publication proof into the existing in-memory queued prompt witness. The same actual entry and coordinator-issued current claim must be checked at dispatch before that metadata can narrow the new ordinary tree; the new tree supplies its own real execution owner. Draft-only edits and return-to-head retain that original witness; changed configuration, drop, consume and restoration do not create or revive it. Source and trust-winner changes refuse, and no old execution capability, second catalog, registry or serialized witness is introduced. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md; reason: direct implementation of original-source custody.

Task 8D2 cold mounted fallback uses the existing original cold-skill preparation as private pending metadata until the actual ordinary custody record acquires its own OPAQUE tree and primary. The same original catalog and conditional trust producer then run once under that tree before input transfer or MODEL, with exact source and physical-return checks. Existing UI Guardian text, captured draft revision/commit, handoff, staged evidence, RAG and queue paths remain their original consumers. The existing in-memory queued witness may retain this pending preparation without claiming an unrun catalog returned. Missing or changed original source refuses; detached snapshots grant no authority. ADR required: no new ADR; ADR path: backlog/decisions/285-native-phased-agent-workflows.md; reason: direct implementation of the existing original-custody contract.


**Original skill trust publication handoff (ADR-285 implementation).** A cold original skill preparation records the app/local trust slots at noninvoking capture. Its existing private ordinary witness carries that exact preparation to the original admitted producer. The original pinned synchronous bridge getter or existing asynchronous catalog initializer may publish one latest private local-service metadata receipt only after its real return, original source/factory/path checks, exact initial-slot match and a pure identity publication fence. Queue sealing and exact-claim consumption accept unchanged original slots or that exact native publication proof; an observed or caller-inserted receiver is not proof, and the receipt is neither detached configuration authority nor an execution owner. Existing trust-bearing prepared receipts keep their exact original winner checks. Draft-only edits and return-to-head retain the same private queue carrier; config replacement, drop and restore cannot mint the receipt. WHY: a legitimate first turn may publish the original cold trust service while a separately captured next send waits, but merely adopting a same-type receiver at queue seal is unproven. COST: two original captured slot identities and one latest private source-bound publication receipt; unknown or replaced sources refuse conservatively. The injected-receiver control is a focused 1F/5.08s RED; this record precedes behavior.

The original trust publisher association must survive the before/after race: a task-private scope around the exact original native operation records the candidate at the actual synchronous app getter assignment or asynchronous native build-lock publication, not from an observed slot. Only after the original local getter physically returns, validate the same captured source/factory/path outside bookkeeping locks and publish the one local-service metadata receipt through a pure identity fence. The final app/local winners must be that recorded actual candidate, or the exact already-published app winner captured at original preparation. A concurrent foreign same-type insertion with no recorded original publisher cannot mint the receipt. Cost: a bounded private in-flight candidate marker in addition to the one latest receipt; no producer replay, new store, execution owner or public snapshot authority.


**Original first-save Library provenance handoff (ADR-285 implementation).** Once the original Q1 identity-publication effect physically returns and its existing fingerprinted postcommit ledger marks that exact effect complete, the same controller verifies the exact original live store/session/holder, first-persist commit and binding, and its original Q1 frozen `new_session`/None Library policy against the actual published `durable`/revision-1 snapshot using the existing hook-parent full-value comparison. Only then may it retain one private immutable narrowing receipt on that existing live session; it holds the original commit/source/policy identities, never an execution owner or capability. At a later ordinary consumer, both queue-before-publication and queue-after-publication cases require the exact original common capture witness, actual current queue claim when queued, the unchanged originally captured ceiling, the same live session/store/holder/binding and the exact current published durable policy. Only source/revision provenance can then advance; both policy values still intersect with the frozen maximum, and the original durable transaction rechecks its revision. Detached/restored/no-witness input, unrelated commits, rebinds, changed policy/holder or equality without the original Q1 ledger cannot mint proof. Reroute reuses the same original witness. The receipt is one session-lifetime cell, not a persisted journal/map or general authority. WHY: original Q1 inserts the same captured values but advances their durable provenance; retaining the old source makes a normal queued first-save fail at durable acceptance. COST: one private first-publication receipt and a strict witness/claim/current-policy check. Genuine unsaved first-session RED is `new-session-queue-red1` 1F/7.55s.

Q1 original-result transport refinement: the native store publication retains its existing public `None` return. A private invocation-local carrier receives only the actual original repository read result/session/holder at the exact native holder assignment. After that store method physically RETURNS, the existing controller callback may place this exact value as a provisional session cell. Only after the original fingerprinted `identity_publication` effect has completed may the same cell become verified, with source/commit/session/holder/original snapshot checks repeated. If cancellation lands after ledger completion, an already-completed retry may verify the same retained provisional cell; failed/incomplete effects and a later observed holder pointer cannot authorize. This uses the one session cell, not an additional journal/map, and source checks occur before and after original methods.


### First-send Stop storage-only retirement (Task 8D2; approved before implementation)

WHAT: retain private paired native raw/normalized published configuration observations on the original agent source, and accept only the exact original successful onboarding transaction as the one-flag semantic successor. Bind this proof to the original runtime, transaction, cache identities and exact operation/event; all host and target returns and required writes must be proven before release. Use the original storage/lease only around closed native journal work, with notifications outside that storage context; no ordinary tree, execution owner, inherited waiver, new source, or new effect authority is installed. Source checks for admission, dispatch, borrow, worker and effects stay strict. Preserve the original claim and storage until the actual finish call returns. Unknown callback/source or any semantic/source mismatch remains retained/refused.

WHY: `stop-firstsend-red8` (1F/20.33s) shows the actual first-send onboarding publication increments the generation between physical Stop return and original operation retirement, causing strict source activation to retain the event indefinitely. Same-path equality alone cannot prove semantic continuity, and generic finish callbacks include activity notifications and possible later E admission.

COST: private full in-memory type-sensitive raw and normalized snapshots plus exact original publisher/cache proof; these may contain secrets and are neither logged nor serialized. The single native onboarding mutation must match both normalized flag projections, locked pre-write effective raw values, successful transaction and exact successor/cache identities. Failed writes, unrelated configuration/environment changes, unknown callbacks and new dispatches gain no exception. ADR required: no new ADR; existing ADR-285 supplies the original source/physical custody contract.

Native-view correction before implementation: `stop-source-observation2` confirmed both native caches and sources were correctly installed but the normalized view failed the plain-value predicate. `stop-source-observation3` (1P/5.08s, stable onboarding) isolated the sole unsupported concrete type to `pathlib.PosixPath`; native `config.load_settings` derives PROJECT_ROOT/API_COMPONENT_ROOT as Path at lines 2297–2300 and publishes them at 2757–2758. Permit only exact immutable native PosixPath values in the otherwise type-sensitive full-tree comparison; preserve exact type and lexical value, no path resolution/I/O or general custom-value equality. This is bounded to the observed native representation and no current-value fallback.


Native fresh-bootstrap retirement qualification (before behavior): the exact original native bootstrap can install `_first_run is True` in the raw cache only after its actual config-file creation returns; its locked persisted/effective snapshot omits this transient key. A proven same-path native first-send replacement may therefore compare the original FULL views to expected post-reload views with precisely raw `_first_run` and normalized `COMPREHENSIVE_CONFIG_RAW._first_run` absent. Require exact original bootstrap source/cache/path and creation provenance, True type/value, locked disk absence, and type-sensitive full comparison of every other field; a persisted marker, false marker or any additional change is not this transition. The full original views remain unchanged and new dispatch remains strict. The original app object and all source/permission/path/DB/scratch/lease guards remain exact. Do not require equality of the entire unrelated process environment: full paired raw/normalized publication views and the existing selected/direct source guards govern the original storage-only callback. The observed TIKTOKEN_CACHE_DIR change comes from the native lazy tokenizer and grants no authority. WHY: observations 4–6 show a real freshly created profile loses only these two ephemeral cache projections during native reload; COST: one exact original creation-provenance check and a finite expected-view transition, with no general key exclusion, registry, repin or effect authorization.


Original Stop receipt notification retry (before correction): trace4 shows the exact receipt notification reached the same native event while an earlier reconcile held its latch; the earlier reconcile then refused at original source entry and discarded its pending signal. Permit one same-event retry only for that recorded initial `check_memory` refusal before any source stack/body was entered, when the already-pending notification and current exact semantic receipt and complete original physical/member/required gates all still hold. No retry for activation unwind, body, finish, write uncertainty or other errors, and no new owner/operation/poller. WHY: otherwise the legitimate original receipt cannot wake the existing returned Stop. COST: one private same-call entry-refusal marker and one finite original-event reschedule. Trace5 1P/5.48s used an observer and a too-broad preliminary retry; it is diagnostic only and cannot qualify the narrowed implementation.


Source-entry clarification (before correction): passive trace1 identified the original pre-yield FINAL source check, after original ExitStack entry, as the refused point. Treat the whole original `activate` context entry through its yield as source entry. A same-event pending-receipt retry is eligible only if that exact final pre-yield exception object propagates after the original ExitStack actually and successfully unwinds, with no event body or finish entered; then recheck the current exact receipt and every original source/storage/member/physical/required gate before one retry. An unwind failure/replacement, body entry or any other exception never mints this marker or retry. WHY: the first-check-only marker conservatively misses a real in-flight original source-entry race. COST: an exact private same-call exception identity and positive unwind marker; no effect authority, new owner or generic retry.


Original Stop post-finish fence (before behavior): recheck the same original source, semantic receipt and physical/member/required gates after the actual operation.finish CALL RETURNS and before marking the target released. If a notification changes a pinned source during finish or any check/unwind fails, keep the original source-scope count, scratch lease and claim retained; do not treat finish return as authority to release against a replaced source. Native notification callbacks run without journal DB context and future dispatch remains strict. WHY: capacity notification occurs inside finish and can synchronously change selected state before its return. COST: one source recheck and conservative sticky retention on an unproven return, with no reaper or replay.


Original Stop retirement receiver/caller fence (before correction): before ANY operation pop, callback, DB context, source-scope or claim action, raw-inspect the exact original source instance/class and definition-time native retirement/check/activation/receipt methods including code/default/closure state, refusing instance overrides or replacement helpers. The private finish tuple is evidence only; require the immediate original native `_EventCustody._reconcile_active` frame, exact event/source/target/operation/owner relationship, and original membership/physical/required proofs. Unknown/foreign callers and replaced methods raise without invoking them or changing capacity. WHY: `stop-seam-red1` genuinely called the actual returned Stop operation from a foreign caller after replacing its source checker; current code returned instead of refusing. COST: finite original function records and noninvoking receiver/frame checks at the existing boundary; no registry, new owner or caller-issued authority.


Stop publication producer pin (before correction): the private initial-view capture and original fixed-key publication helper are native producer functions, so record their original function/code/default/closure state and raw-check them before and after the actual call, as well as at retirement consumption. A replaced producer cannot mint a receipt by restoring the consumer later. WHY: the receiving fence alone would not prove which function originally created its metadata. COST: two additional finite original function records and no added source read, capture, executor or authority.


Ruling: WHAT: apply the same noninvoking original receiver/function fence in the pending-notification exception path before calling any retirement method. WHY: an earlier strict source entry can begin before the receipt exists, so the later exception branch cannot assume its initial entry validated a newly available retirement receiver. COST: one repeated pure native identity check on that finite retry; changed receivers refuse before invocation and no new authority is introduced.


### Ruling: first-send publication after original source retirement (FIX1)

WHAT: Keep the app-owned onboarding flag immediately visible after a successful manual parent, but defer its one native persistence until all already accepted ordinary source lifetimes, queued source handoffs, physical tree members and required settlement have actually retired. One runtime-owned pending intent/task and short metadata-only admission/publication fence serialize the final quiescence election with new original source captures; no IO or callbacks run under bookkeeping locks. New captures wait before acquiring a lifetime, or synchronous intake refuses before input transfer. Existing source, permission, DB, root, lease and required-write checks remain strict. Parent response does not await children or late hooks. The original save runs outside locks and disposal waits for its actual return; failed or uncertain proof cannot establish retirement.

WHY: FIX_BASE actual runtime/received survivor RED (1F/20.08s) and native late SubagentStop RED (1F/11.46s) show the onboarding generation change occurs while original admitted physical work still needs its original config identity. Stop-only semantic retirement does not cover those other original consumers. The exact producer/carrier/consumer and cancellation table is recorded in `task-8d2-ordinary-integration-fix1-source.md`.

COST: one private runtime pending intent/task, finite capture/queue/tree bookkeeping and potentially delayed disk persistence; an abrupt exit before the deferred save can show onboarding again. ADR required: no new ADR; this corrects the lifecycle under existing ADR-285. Prior Stop successor text records the earlier approach; it does not grant a general generation waiver or new dispatch authority.

Ruling: WHAT: use the existing one-key `apply_settings_mutation_to_cli_config` transaction with its native locked identity precondition and existing original-function source pins; retain the in-flight metadata fence through actual worker RETURN, including cancellation. WHY: checking config identity before calling the ordinary save alone leaves a race with a concurrent settings transaction. COST: one pure original-source check in the existing config transaction; changed sources refuse and no generation is repinned.

Ruling: WHAT: retain the same original tree source until native activation DB/execution scopes and the original fleet worker's actual source DB cleanup return, checking original thread-local handle identities; uncertain cleanup retains custody, while conclusively unwound body exceptions preserve existing settlement. WHY: the native FIX1 controls exposed live original fleet-thread DB leases after the former tree retirement signal. COST: finite original-scope lifetime and handle proof only; no new owner, lease, reaper, generation waiver or change to non-tree execution.

FIX1 supersedes the earlier Stop semantic-successor implementation: source-closed deferred persistence makes its sole receipt producer unused. Remove paired snapshots, bootstrap receipt metadata and the private Stop finish/journal exception together; keep strict original activation and native config-function pins for the locked one-key mutation. Actual hook/worktree callbacks, storage checks and cleanup semantics remain. Replacement ordering is qualified by actual runtime survivor and held SubagentStop controls; the final native child tail beyond retained DB/execute scope is source-only implicit return/thread bookkeeping, not a universal OS-thread-join claim. ADR required: no new ADR; existing ADR-285.

FIX2 ruling: WHAT — extend the same first-send metadata counter to exact original RawCLI and confirmed-Undo pending membership. Gate native capture before selecting generation, atomically overlap capture/admission with pending count, and release/wake only after original physical/required/host/operation-finish/storage/scratch/root/claim return. Gate only trusted RAW classification through the SAME captured draft commit/start; normal slash/unknown/escaped routes retain their existing availability. WHY — actual unchanged FIX1 native transaction hold admitted a RawCLI call with the old generation (RED1F12.63s, normal original cleanup), and the directly connected Undo source has the same generation-sensitive pending lifetime outside ordinary trees. COST — reuse one existing per-runtime metadata count; no new owner/claim/registry/poller/waiver or source adoption. Lost original membership keeps its count conservatively. Unadmitted Undo confirmation metadata may become stale between user actions and then refuses before effects. Model/Task7 RawCLI keeps its existing single operation/tree. ADR required: no new ADR; existing ADR-285. Source/caller/physical table: task-8d2-ordinary-integration-fix2-source.md.
