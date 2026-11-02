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
