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
