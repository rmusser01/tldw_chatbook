# Native phased agent workflows in a shared worktree

- Date: 2026-11-01
- Status: Approved 2026-11-01; implementation planning authorized, execution not started
- Design task: [TASK-38493](../../../backlog/tasks/task-38493%20-%20Design-native-phased-agent-workflows-in-a-shared-worktree.md)
- ADR required: yes
- ADR path: [ADR-285](../../../backlog/decisions/285-native-phased-agent-workflows.md) (Accepted; design only)
- Implementation plan: [Native shared-worktree workflows](../plans/2026-11-01-native-phased-agent-workflows-implementation.md) (planning only)
- Reason: Introduces durable phase/task authority, shared-worktree coordination,
  workflow timing and result ownership across Console and AgentRunsDB.
- Approval state: the user selected native Console orchestration, coding before
  research before general tasks, one shared worktree, initial-plan and per-phase
  approval, explicit restart resume, and correction of all six review gaps.
  The user approved the completed controls, failure and verification sections
  and explicitly requested finalization of this spec/ADR and an implementation
  plan on 2026-11-01. This approval does not start product implementation.

## 1. Outcome and boundaries

A user gives Console a coding goal and selects an existing writable local
Workspace folder binding. The lead inspects permitted inputs and proposes a
bounded phase plan. Approval starts its first execution phase. Tasks work in
the same selected worktree, including its existing uncommitted content. Each
later phase requires a new human decision. Progress and results survive restart;
execution resumes only through explicit, freshly validated authority.

The default flow is planning → implementation → verification → independent
review → final report. Delivery is the terminal report, not an extra merge or
apply phase. A proposed repair becomes a new approved phase attempt.

There are no child worktrees, integration checkout, automatic commits, resets,
stashes, pushes or PR publication. Stop preserves edits. This milestone adds no
daemon, external orchestration dependency, workflow editor, generated Python/JS
runner, recursive delegation, remote-file execution or machine-wide quota.
Research and general tasks are later deliveries, not implementation work hidden
inside this milestone. ADR-138's portable Workflows surface remains unchanged.

## 2. Existing owners and bounded additions

| Concern | Reuse | Required extension |
| --- | --- | --- |
| App lifetime | [ConsoleRuntime](../../../tldw_chatbook/Chat/console_runtime.py) | App-owned workflow coordinator; views do not own execution. |
| Received intent and accepted dispatch | [controller](../../../tldw_chatbook/Chat/console_chat_controller.py), [automatic submissions](../../../tldw_chatbook/Chat/console_automatic_submission.py) | Typed workflow origin and private coordinator-issued phase authorization. |
| Child execution | [AgentService](../../../tldw_chatbook/Agents/agent_service.py), [FleetCoordinator](../../../tldw_chatbook/Agents/fleet_coordinator.py) | Share the existing child admission/launch path with host-scheduled tasks. |
| Capacity and actual retirement | [RuntimeCapacity / ExecutionOwner](../../../tldw_chatbook/Agents/execution_capacity.py) | Bind task footprint reservations to the actual owner and positive drain outcome. |
| Tool restrictions | [runtime guard](../../../tldw_chatbook/Agents/agent_runtime.py), [registry](../../../tldw_chatbook/Agents/tool_catalog.py), [LocalToolProvider](../../../tldw_chatbook/Agents/local_tool_provider.py) | Enforce task-bound effects and paths before permission review and at invocation. |
| File mutation | [local tools](../../../tldw_chatbook/Tools/local_tool_impls.py), [patch tools](../../../tldw_chatbook/Tools/patch_tool_impls.py), [ReadLedger](../../../tldw_chatbook/Agents/fs_read_ledger.py) | Require observed-content/absence preconditions for workflow writes; carry edit/patch read digests through the existing atomic writer. |
| Durable admission and accounting | [AutomaticWorkLedger](../../../tldw_chatbook/DB/automatic_work.py), [AutomaticWorkContext](../../../tldw_chatbook/Agents/automatic_work_runtime.py) | Workflow policy, phase acceptance, task launch claims, cumulative active-time accounting. |
| Saved state | [AgentRunsDB](../../../tldw_chatbook/DB/AgentRuns_DB.py), [Goal store patterns](../../../tldw_chatbook/DB/goal_runs.py) | Workflow revisions, phases, decisions, attempts and bounded private results in this database. |
| Evidence | [Goal manifests and evidence refresh](../../../tldw_chatbook/Agents/goal_iteration.py) | Reuse bounded hashing and freshness rules without invoking Goal execution policy. |
| File review | [consent](../../../tldw_chatbook/Workspaces/change_review_consent.py), [finalization](../../../tldw_chatbook/Workspaces/change_review_finalization.py), [tracker](../../../tldw_chatbook/Workspaces/change_turn_tracker.py) | Link phase review to exact native review windows; preserve consent, limits and attribution warnings. |
| UI patterns | [Goal status](../../../tldw_chatbook/Widgets/Console/console_goal_status.py), [Goal UI controller](../../../tldw_chatbook/UI/Console_Modules/goals.py) | Follow view/subscription/offload patterns; do not relabel Goal state as workflow state. |

The current child launcher is nested inside one run and captures substantial
turn state. Its extraction is a compatibility-sensitive change, not a promised
small wrapper. Preserve routing, frozen presets/fallbacks, permission ownership,
hooks, project instructions, cancellation, run identities, capacity, accounting,
client cleanup and inline/threaded behavior. Never call a bare child thread or
persisted run row as an alternative execution path. Public ordinary spawning
keeps its existing behavior and default shared-worktree semantics.

## 3. Plan and task contracts

Use strict structured data with ordered phases and explicit earlier-task
dependencies. No executable expressions, arbitrary graphs or plugin language.
Reject duplicate keys/IDs, cycles, forward phase dependencies, unknown fields,
invalid references, unavailable roles and scopes outside admitted authority.

A revision contains the goal, selected binding and captured root identity,
provider/preset references and approved fallback targets, finite limits, phases,
and expected deliverables. Each task contains instructions, dependencies, input
and output references, declared read/write footprints, allowed tool subset,
execution mode, model target and expected checks. Tool access narrows existing
permissions; plan prose and task results grant no authority.

First-release bounds are 128 KiB per complete plan, at most six execution phases
and twelve task definitions, and 64 declared footprint entries per task. Paths
are validated binding-relative exact paths or subtrees; there is no new glob
language. A declared directory read conflicts with creation/removal under that
directory. Discovery requiring a wider footprint stops for replanning instead
of quietly expanding an active task's scope.

Initial inspection is a durably accepted, budgeted, read-only planning attempt.
It cannot mutate the worktree before plan approval. Revised plans likewise use
bounded planning attempts requested by the user. Revisions preserve prior
attempts and results as history; they never rewrite their ownership or charges.

## 4. Enforced scope and shared-worktree scheduling

`PathAwareToolProvider.path_targets` supplies instruction-discovery hints, not
task authorization. Add a host-owned restriction bound to workflow, revision,
phase, task attempt, run and live binding. It covers schema discovery, runtime
callbacks, the pre-review guard and final provider invocation. A permission
Allow cannot override this restriction. Use the registry's actual owning
provider and existing path/exclusion validation, not model-supplied tool labels.

File tasks use admitted `fs_*` tools and existing read-only Git/virtual tools.
Alternate builtin writers, runtime spawning/new-chat escape paths, scripts,
shell and unknown-effect MCP calls cannot bypass a file task's scope. Tools
with opaque effects require an explicitly planned exclusive task mode and
their normal exact permission/trust/arming requirements. Their actual OS or
remote authority remains unchanged; exclusivity is not a sandbox.

Effectful command/MCP hooks also require actual native ownership through physical
drain. Until that lifetime is supported, workflow preparation explicitly refuses
configured effectful hooks; read-only planning never silently runs them. Existing
hooks remain configured and ordinary execution keeps its ordering and permissions.
Workflow skill scripts likewise refuse in scoped and exclusive modes until typed
process and reader cleanup proof and owned lifetime are available. An exit code or
warning string cannot establish that proof. Later launch/coordinator integration
must preserve these refusals unless it supplies the actual native ownership seam.

Restriction state is keyed by actual native run/task attempt, as existing
approval stamps and ReadLedger are, because siblings share provider instances.
Never install one mutable provider-wide current task or trust a task ID supplied
in tool arguments. Runtime spawn/skill callbacks and progressive loading must
retain the same narrowing; tasks cannot create unregistered grandchildren.

Workflow writes require a valid current observation or explicit absent-file
precondition. Reuse the atomic writer's preconditions for writes, edits and
patches, including the bytes actually read by edit/patch computation. ReadLedger
remains helpful feedback but a missing/evicted stamp does not permit a blind
workflow overwrite. A refused stale operation requires a fresh read/replan as
appropriate; never automatically resubmit the same destructive request.

Add one small worktree-activity owner under ConsoleRuntime, using existing
admitted root/path identities and ExecutionOwner lifetimes. It coordinates
access, not execution or a second allowance ledger. Canonical aliases, nested
bindings and case/path normalization must agree with existing path resolution.
Binding names alone are not distinct worktrees.

Before launching any worker, atomically acquire its complete declared read/write
footprint, then attempt native capacity admission without waiting. If capacity
refuses, release that unlaunched task's footprint before returning to pending.
Read/read overlap is allowed;
write/write and read/write overlap are ordered. Incompatible tasks remain in the
bounded durable phase task list, not as started workers holding partial locks.
Do not hold database transactions or coordinator locks across waits, callbacks,
Git, model/tool execution or human decisions. Refused admission releases only
its own unconsumed reservations.

Opaque tasks claim the entire selected worktree exclusively. Ordinary native
turns without a workflow footprint conservatively claim their admitted roots;
accepted survivor children retain that claim until they actually drain. Acquire
multi-root claims together to avoid lock-order deadlocks. When a workflow is
active, conflicting manual work receives a visible busy/refusal outcome under
existing queue rules; no hidden persistent queue for manual prompt bodies is
introduced. Manual priority remains intact at admission.

An opaque effect may target paths outside its cwd or admitted root list. For
both workflow tasks and ordinary native turns that can invoke such effects,
use a runtime-wide opaque-effect reservation conflicting with other owned
worktree activity. Acquire that broader reservation before root-specific
claims; never upgrade while holding a partial footprint. This coordinates
known native owners only and does not restrict external OS/remote effects.

At most one execution phase per workflow and one active workflow execution
phase per canonical worktree are admitted at a time. Approval waits release
activity claims only after positive drain and durable settlement, allowing
ordinary manual work. Reapproval rechecks affected inputs afterward.

Known task claims are released through `ExecutionOwner.on_drained` only after a
positive outcome and durable task settlement. Logical terminal status or a
callback carrying cleanup-unproven is insufficient. External editors and other
processes remain outside this coordination guarantee; observed drift invalidates
affected evidence and approval inputs.

## 5. Durable authority and dispatch

Introduce workflow-specific automatic submission/attempt types through the
existing controller, automatic-submission resolver, context policy selection,
registry guards and ledger recovery. Ordinary MANUAL, wake, chat-start and Goal
origins retain their current policies. A phase authorization is process-local,
coordinator-issued and bound to the exact live owner/session/conversation,
workflow revision, phase and accepted attempt. Stored IDs are lookup references,
not capabilities. Restore, import, UI selection or a transcript row cannot
reconstitute an old execution capability.

Plan approval starts the first execution phase. A later approval names the
exact plan revision, phase, predecessor result digest, checked input manifest
and control revision. Stable decision IDs make identical repeated commands
idempotent; conflicting replay or stale targets refuse. Progress publication
has a separate cursor and cannot invalidate an unrelated pending control merely
by appending a task event. Stop/replan advances control authority and revokes
unused dispatch claims before further work can start.

Reserve required budget and private result capacity before dispatch. Commit
phase acceptance with FULL durability before attributable billable preparation
or effects. Each task has a stable attempt identity and an atomic launch claim
before starting its actual run. Duplicate delivery finds that claim and existing
run; it never starts a second worker. Proven pre-dispatch refusal may release
unconsumed reservations; accepted or uncertain work never automatically retries.

Host scheduling uses actual task settlement events to launch the next eligible
approved task, not repeated lead-model polling. A workflow phase respects the
existing per-session primary slot and global manual reserves. It does not allow
concurrent ordinary turns in that same chat. Existing task steering/Stop remains
available; a new goal-changing prompt pauses/replans rather than silently
modifying an approved phase.

Workflow-owned child membership is persisted and recognized by completion
delivery. Their events go to the phase owner and do not also authorize ordinary
fleet autowake synthesis. Preserve normal fleet behavior for unrelated children.
The phase stays accepted until every task and owned operation has settled.
Task result, evidence, terminal state and accounting settlement commit together
in AgentRunsDB before the next phase becomes approvable. A persistence failure
after an effect retains uncertainty and blocks replacement execution.

## 6. Finite policy and approval time

Use a distinct workflow policy in the existing automatic accounting machinery;
never identify workflow work as a manual turn, fake survivor wake or Goal.
The `[agents]` settings use the existing finite-validation and canonical
Settings patterns: `workflow_runs_enabled` defaults false and
`max_workflow_*` supplies policy ceilings. Defaults are eight accepted
planning/execution phase attempts, twelve task launches, 32 model calls,
500,000 budget tokens, 8,192 output tokens per call and 900 seconds of active
execution. Task run bounds intersect the captured native bounds and remaining
workflow allowance. Setup shows the ceilings and refuses a plan whose known
minimum phase/task counts cannot fit. These conservative values are admission
limits, not a promise that every approved plan will finish within them.

Snapshot ceilings at creation; live reductions and the workflow kill switch
may narrow admission. Raising settings, approving another phase, replanning,
continuing a task or restarting never replenishes that workflow's counters.
Provider fallback and planning/synthesis calls spend the same allowance.

Extend accounting only for workflow active-time semantics. Charge elapsed time
over the union of active workflow execution intervals, not the sum of parallel
task durations. An interval begins before accepted planning/execution dispatch
and remains charged while any local effect/model cleanup is alive or uncertain.
Ordinary capacity waits after drain and committed human-review waits do not
consume this active-time allowance. Within-phase permission waits retain their
existing behavior and bounds; they are not assumed quiescent automatically.

Close an interval only on positive drain and durable settlement. Persist charged
time and interval ownership. On restart, preserve all charges; a foreign open
interval or unexplained clock state requires recovery and cannot be rounded to
zero. An uncertain interval remains conservatively charged through reconciliation
or exhausts the remaining allowance; operator acknowledgment cannot invent a
drain timestamp. Goal and fleet absolute deadlines remain unchanged. A clean,
committed approval wait can therefore last overnight without spending active
execution time or granting any additional calls/tokens.

## 7. Required results, evidence and change review

Save bounded authoritative task results, outcome/status, required handoff data,
typed process outcomes and verification manifests in private AgentRunsDB-owned
payloads. Follow Goal payload reservation/ownership patterns: 128 KiB per
record, 4 MiB per workflow and 128 MiB aggregate workflow payload capacity,
reserved before dispatch. Refuse mandatory results that cannot fit; do not
truncate them into success. Detailed native run logs remain supplementary and
may be disabled or truncated without losing required handoffs.

Retain results needed by active, paused and uncertain workflows. Explicit
removal is permitted only for settled payloads and retains accounting and
idempotency tombstones. No active evidence is held solely by a volatile message
or a source/log path that can be pruned. Private payloads retain existing
storage admission, backup/restore and recovered-content activation requirements;
no secrets, automatic AGENTS.md bodies or grants are copied into metadata.

Adapt Goal's `capture_manifest` / `refresh_evidence` contracts to exact task
inputs, output bytes, verifier identity/arguments and current authority. Retain
the existing bounded hashing limits; unrepresentable or unstable input scopes
cannot be labeled checked. Existing complete owned Change Review tree references
may represent their actual disclosed scope; they cannot certify excluded paths.
Source/hash unavailability is explicit, not an empty manifest meaning unchanged.

Recheck freshness before accepting a phase decision, dispatching dependent work,
publishing verified completion and resuming. A command exit code is typed runtime
evidence, not `ToolResult.ok` or the model saying tests passed. Failed, skipped,
stale, missing and passed checks remain distinct. Changes to providers, fallback
targets, tools, binding access or relevant inputs trigger fresh preparation and
approval where the captured contract changed; current permissions are always
resolved again and old approval stamps are not restored.

Reuse Change Review consent and root-ordered reservations. Link each phase to
its actual review-window records, including survivor windows; never flatten
ambiguous windows into an invented per-task diff. Request existing capture
consent through its current owner when needed. Disabled tracking, capture
failure, exclusions, size limits and concurrent attribution remain visible.
No workflow feature silently enables capture or turns snapshots into a general
rollback guarantee. Stop leaves current files in place. Existing guarded,
confirmed Undo remains a separate operation and must refuse conflicting live
workflow writers as it does other active work.

Task 7 concretizes required verifier provenance with bounded existing
`VerificationSpec` definitions: an approved executable wrapper within the
selected root, exact arguments, SHA-256 and input paths. The host derives
`shlex.join((verifier_path, *arguments))`; the actual native raw-shell provider,
resolved bash identity, typed result, physical cleanup and both stream EOFs
supply evidence. Required sanitized output is collected before preview/log
truncation, bounded to 96 KiB total per task and still subject to the combined
128 KiB result/checkpoint reservation. Overflow or missing EOF is incomplete.
Wrapper hashing retains the existing 1 MiB limit; shell provenance uses the
existing bounded 16 MiB manifest primitive. No scripts/new command language.

Task inputs contain approved source path/manifest references and complete saved
predecessor handoffs. Literal file bodies are fetched through current `fs_read`
permissions after native binding. Same-phase references bind exact producer
definitions at phase approval; fresh completed native producer checkpoints are
mandatory before dependent task preparation and launch. Upcoming phase approval
never accepts that phase's eventual result; completed-phase quality review is
an explicit separate operation with a tagged current evidence record.

## 8. Persistence and restart recovery

Use new versioned AgentRunsDB tables for immutable plan revisions, workflow
control state, phase/task attempts, decisions, private payloads/reservations and
active-time intervals. Reuse agent run IDs, automatic chains/reservations and
physical operation ownership rather than duplicating them. The next migration
number is chosen from the actual integrated schema at implementation time;
do not reserve or renumber a shipped migration in this design.

The existing runtime startup owner performs one recovery audit for fleet, Goals
and workflows. Opening history or constructing the workflow service never runs
a second global sweep. Preserve accepted, prepared, uncertain and charged
attempts conservatively. Clean pending review becomes awaiting explicit Resume;
unknown execution becomes recovery-required. A terminal legacy run row alone
does not prove physical cleanup or completion of a phase.

Resume validates live profile/runtime ownership, exact chat and binding/root,
plan/control revisions, current input/output evidence, tool/provider references,
remaining allowances and required saved results. It allocates fresh capabilities
only for work conclusively safe to start. Completed results remain history but
cannot satisfy new dependent work when their evidence is stale. Partial success
is retained; failed tasks and their dependents wait for a proposed repair.
Unknown effects are never replayed automatically, and closing uncertainty is
not permission to rerun it or proof that a remote side effect stopped.

Task 7 uses schema v27 additive checkpoint/resume-command tables and preserves
absent-new-field v25/v26 canonical payload bytes. A positive quiescent boundary
is durable only after existing unsettled checks and actual coordinator drain;
every reopened native interval resets it. Startup closes only prepared unused
or explicitly quiescent old custody with exact launched result/checkpoint
lineage and no unresolved effect. It preserves spent generations/calls/tokens,
closes unused no-run task attempts and releases only unconsumed reservations.
Unmarked accepted/open/unknown boundaries remain recovery-required.

The storage service's `resume` is an inert transition to awaiting approval.
The public coordinator must perform genuine fresh asynchronous native
provider/tool/permission preparation before accepting any new phase attempt.
Same-owner capacity continuation keeps its original acceptance; post-restart
remaining work needs a new phase attempt and generation without adopting old
capabilities or discarding fresh successful-result references.

Update installed recovery schema catalogs, participant coverage, backup
inventory/restore activation and targeted migration/retention tests with the new
storage. ADR-244's future shared-host fence is not assumed implemented. Initial
support retains the current single executing Console runtime per profile
assumption; no multi-process scheduling or mobile-host capability is claimed.

## 9. Review correction coverage

| Review finding | Contract | Acceptance scenarios in section 12 |
| --- | --- | --- |
| 1. Missing scope enforcement | Section 4: invocation-bound narrowing, alternate-writer refusal, guarded atomic mutation | A2, A3 |
| 2. Disjoint writes are not independence | Sections 3–4: explicit inputs/dependencies and complete footprint admission | A3, A4 |
| 3. Incomplete workflow admission | Section 5: typed phase capability, FULL acceptance, task launch claims, no duplicate wake | A5, A6 |
| 4. Approval waits exhaust elapsed budget | Section 6: workflow-only cumulative active time and unchanged existing policies | A7 |
| 5. Exclusivity lacks a runtime-wide root boundary | Section 4: alias-aware claims including ordinary owned activity | A4, A8 |
| 6. Required results rely on optional logs | Section 7: pre-reserved private results with explicit retention | A9 |
| Evidence freshness and truthful snapshots | Sections 7–8: exact manifests, consent, exclusions and conservative resume | A8, A10, A11 |
| Small plan and code reuse | Sections 2–3: bounded ordered phases and shared existing execution path | A1, A12 |

## 10. Console interaction

Add a native `/workflow` command family to existing Console command discovery:
`/workflow new` opens goal/workspace/limits setup and `/workflow` opens this
chat's saved workflow history. It does not consume or clear an unsent composer
draft implicitly. A workflow belongs to one existing chat; initial delivery
allows one nonterminal workflow per chat. Selecting another tab changes only
the view. This is separate from the saved Workflows destination and `/goal`.

Use existing form, modal, status and subscription patterns, following the
[design language](../../../backlog/docs/design-language.md) and
[component catalog](../../../backlog/docs/component-patterns.md). Reuse the
native fleet view for individual agents and the existing Changes entry for file
review. Add one compact workflow status line and one detail modal; no new screen,
sidebar, DAG canvas or repeating workflow poller. Views subscribe to bounded
runtime snapshots and offload history/payload reads through finite DB callbacks.
Keep task lists paginated; a snapshot contains at most 50 rows per page.

Setup displays selected working folder, current uncommitted content as input,
provider/preset choices, finite limits and actual Change Review readiness. The
prepared-plan view displays phases, dependencies, task footprints, expected
outputs/checks and captured limits. `Approve plan and start` identifies precisely
the displayed revision; default focus is not the approving action.

The detail view shows phase state, task status/progress, limits spent/remaining,
required results and links to exact change/check evidence. At a successful
boundary offer `Approve next phase`, `Revise remaining plan` and `Stop`.
`Revise remaining plan` consumes a bounded read-only planning attempt and never
rewrites completed history. Final review offers `Accept result` or a requested
repair plan. Quality acceptance cannot clear uncertain execution or stale checks.

During execution offer `Pause`, `Stop` and existing task inspection/
steering. Pause finishes only already admitted tasks; tasks not yet launched
stay pending and the workflow pauses after admitted owners drain. If a phase is
partly finished, resuming its remaining tasks requires a new explicit decision
against current inputs. Stop revokes further launch authority immediately and
reports `Stopping` until physical retirement is known; edits are retained.

Closing a modal or navigating away detaches the view, not the workflow. Pending
decisions remain in the owner; normal background attention points back to the
exact chat/workflow. App shutdown closes admission and drains existing owners.
On restart show `Resume required` or `Recovery required`, never auto-run. Stale
buttons refuse and refresh; duplicate clicks display the existing receipt.
Use existing token-backed focus/status/disabled patterns, keyboard traversal and
Escape-to-close. No new terminal-convention/global hotkeys are introduced.

## 11. Failure and completion

| Situation | Required visible outcome |
| --- | --- |
| Provider/task failure | Preserve successful siblings; block dependents and phase advancement; offer a bounded repair plan. |
| Scope violation or newly needed dependency | Refuse the effect; show which task needs revised scope/ordering; no automatic scope widening. |
| Capacity unavailable | Keep unlaunched tasks visibly pending without consuming launch allowance or starting blocked workers. |
| Permission unanswered/denied | Park/refuse through the existing owner; phase approval never supplies tool consent. |
| Writer timed out or cleanup unproven | `Stopping`/`Recovery required`; retain ownership and conservative charges; no conflicting replacement task. |
| Budget exhausted | Pause with the exhausted resource and saved results; no refill through Resume or another phase approval. |
| Required payload persistence fails | No verified completion or next-phase approval; preserve uncertain effect/charge and recovery diagnostics. |
| Snapshot unavailable | Explicit missing/change-attribution evidence; no fabricated clean diff or automatic capture enablement. |
| Inputs/verifier changed at review | Mark affected checks stale and require fresh preparation/checks; preserve historical evidence. |
| Binding/provider authority changed | Pause before further dispatch; refresh or revise the plan under current permission ownership. |
| Stop or restart | Preserve edited files, results and history; no automatic rollback, continuation replay or new allowance. |

A finished agent turn is not a completed task objective. Task results distinguish
execution status, delivered outputs and actual checks. A phase completes only
after required tasks and mandatory checks succeed, outputs persist and owners
drain. A result requiring judgment remains awaiting human review. Approved
advancement, quality acceptance and physical recovery are separate decisions.

## 12. Verification and delivery

Use targeted regressions with real temporary files/Git repositories and SQLite,
actual native admission/dispatch owners and deterministic held-worker barriers.
Test model reports alone are not proof of tool effects. Required scenarios:

- **A1 — Shared execution and launch compatibility:** two eligible agents edit
  distinct files in the same selected worktree, retaining pre-existing dirty
  content. No worktree/branch/index mutation is introduced. Ordinary inline and
  threaded spawn, routing/fallback, hooks, permissions and cleanup retain behavior.
- **A2 — Scope at actual invocation:** wrong path/alias, readonly binding,
  revoked root, alternate builtin writer, runtime escape, approved-session shell,
  progressive loading and unknown MCP cannot bypass the task's captured mode.
  Exercise read-before-write/absent-file and simultaneous edit/patch races.
- **A3 — Dependencies:** disjoint writers can run concurrently only when inputs
  permit it; write/read and directory create/list conflicts serialize. Invalid,
  cyclic, oversized or outside-root plans refuse before dispatch.
- **A4 — Coordination:** alias/nested bindings share the appropriate claims;
  conflicting workflows, ordinary turns and survivor children cannot overlap
  exclusive work. Denied admissions unwind without partial-lock deadlock.
- **A5 — Approval and dispatch:** duplicate/reordered decisions, old revisions,
  Stop-before-thread-start, task event replay and result-commit failure create
  at most one physical launch per accepted attempt. Record effects with barriers,
  not only row counts. Workflow children do not also trigger fleet autowake.
- **A6 — Retirement:** a logical terminal row with a held worker cannot release
  claims, start a dependent check, declare phase completion or bypass capacity.
  Positive/negative drain callbacks and failed cleanup remain distinguishable.
- **A7 — Time/accounting:** overnight drained approval and capacity waits do not
  consume active time; overlapping tasks charge union duration; alive cleanup is
  charged; restart/clock drift/unknown intervals remain conservative. Limits,
  Stop and live reductions hold, with unchanged Goal/fleet deadline controls.
- **A8 — Freshness:** edit relevant files, verifier bytes, tool/provider refs or
  binding access while the review view is open or during restart. Old approval
  and evidence cannot authorize changed dependent work. Unrelated metadata
  progress must not invalidate the correct control revision.
- **A9 — Durable results:** disable/truncate/fail native logs and reopen with a
  fresh DB/process. Required handoffs survive; payload admission, missing records,
  pruning/removal, duplicate results and full-capacity refusal are honest.
- **A10 — Review truth:** consent off, ignored/oversize files, overlapping review
  windows, external edits and baseline failure remain disclosed. Stop preserves
  files; existing Undo refuses active writers and never undoes another actor's
  later edits under a workflow shortcut.
- **A11 — Recovery/storage:** exercise real schema upgrade/reopen, one startup
  audit, restart at prepared/accepted/effect/result-commit boundaries, uncertain
  attempts, private payload retention and exact backup/restore schema catalogs.
  Restored history stays inert until current-owner admission and explicit Resume.
- **A12 — Mounted and live flow:** real keyboard/click start, plan approval,
  phase review, revision, pause, Stop, navigation and explicit restart Resume;
  narrow/wide token-backed UI; a real provider makes a file correction between
  a failed and passing targeted command. Retain exact source/check manifests,
  process outcomes and diffs. No full-suite claim substitutes for this journey.

Implement only after final design and subsequent implementation-plan approval.
The implementation plan should separate: durable admission/results/time;
guarded same-worktree dispatch/coordination; Console controls/recovery/evidence;
and final live qualification. Each slice has its own atomic Backlog tasks when
planning begins; no uncreated implementation tasks are referenced here.

Before coding, reproduce and attribute the earlier focused-check failures:
seven read-ledger tests passed, while five provider/fleet tests and one teardown
failed; the isolated provider case hit `raw_source_selection_changed` before
useful execution. These historical results are not current fleet qualification.
Do not suppress storage admission or modify unrelated work to obtain green.

After implementation, update living Console/agent-runtime/tool-catalog/database
architecture docs, canonical Settings help, recovery catalogs and user guidance
with actual delivered behavior. This draft does not describe those extensions
as already available. Targeted tests only unless a full sweep is requested.

## 13. Alternatives and review status

Separate child worktrees were rejected by the user. Extending the portable
Workflows executor first expands the wrong delivery surface. Reusing Goal
iterations directly would violate their no-child and elapsed-deadline policy.
Normal user messages, fabricated fleet wakes and generated orchestration code
would bypass or duplicate existing ownership. A general DAG engine, new log
store, duplicate permission service and automatic Git rollback are unnecessary.

This approved document addresses the six review gaps and both improvements,
including Console interaction, failure handling, verification and delivery.
The user authorized implementation planning after finalizing the written design.
Planning is separate from execution: no product implementation, schema migration,
provider execution or full test sweep is represented as complete by this spec.
# Ordinary source composition qualification (Task8D2)

Before original SessionStart effects, the actual Console submit must capture the
complete prospective agent storage tuple, including original DB/permission
sources and selected run-log confinement root. Private custody stays with the
original issuing submit and actual worker handoffs, never in detached serialized
configuration or an ID lookup. The real bridge consumes that root; stock writer
bind validates the exact original descendant without changing the source tuple.
Custom/replaced writers, root changes and late unrepresented sources refuse.
Retained MCP context must reach a genuine child model call with unchanged strict
source and permission checks. This source partition alone does not implement
before-reference/initializer physical admission, ordinary borrowing, canonical
child settlement, SSH proof or late-event lifetime; the full ordinary integration
remains mandatory under ADR285/163.



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


### Concurrent phase-wave capture (Task8D2; approved before behavior)

[ADR-285's wave contract](../../../backlog/decisions/285-native-phased-agent-workflows.md#concurrent-phase-wave-capture-task8d2-approved-before-behavior)
amends sections 4, 7 and 8: one actual B/E per concurrently admitted task wave,
with same-mutex Activity ordering, original process-local membership, earlier and
ready manual priority, and finite OPAQUE snapshot distinction. Finite native
preparation owns callbacks before catalog effects; actual callable/permission
qualification must prove bounded tasks. No capability pruning manufactures overlap.
Human decisions precede opening; B physical return precedes child writes. The
inert original reservation is durably recorded before scheduling, and only the
first real committed member anchors it. Close E at every genuine gap; reopen
with fresh B after manual effects. A no-commit race closes an honest unused wave.

Schema v28 adds one workflow window table and ONE normalized reference table:
64 windows/workflow, 64 roots/window, 4096 UTF-8 bytes/path, 32768 root JSON bytes,
64 refs/window, 65536 reserved bytes/window within existing 4 MiB/128 MiB budgets.
Zero-row/no-run/late E is durable without rewriting task checkpoints. Both failed
publisher attempts cannot acknowledge phase success; conclusive observational
return is separate from required durable window/phase settlement. Preserve B
provenance on E failure. Active/unresolved protection survives review closure,
pruning, archive/restore and startup recovery; restored rows grant no custody.

Use current original source capture/publication counting through actual native,
required host, storage and finish return. Stale pre-consent metadata refuses
without repinning. Required evidence includes two actual bounded native tasks
overlapping in one wave, before-first-child and partial-gap manual races, fresh-B
attribution, no-commit race, finite cross-root jobs, held return, publication
failures/empty/late E, immutable checkpoints and actual archive/restore/recovery.


Task8 D2 close refinement (source-confirmed): workflow preparation must not allocate the unused primary model lifeline. Only `run_primary` starts it, and that entry refuses workflow dispatch; actual children retain their unchanged native child model lifelines. Launched-task native frame teardown returns under a pre-reserved operation on that same original child before its physical drain. A no-launch frame may retire only the source-verified empty display/maps and native provider unbind metadata; any unexpected effectful callback or changed cleanup source refuses, retaining its source lifetime. There is no fresh cleanup owner or OPAQUE claim against the wave, and no generic cleanup exemption.


Task8 D2 hook construction refinement: the unchanged native workflow hook policy refuses configured effectful hooks and any existing legacy engine. `ConsoleAgentBridge._prepare_native_reply` must not invoke the ordinary lazy `ensure_run_hooks` constructor for an accepted workflow preparation after that check: it creates an otherwise unused engine even when unconfigured, invalidating its own preparation. Workflow child callbacks keep the captured absent engine; ordinary hooks and the original workflow refusal checks remain unchanged. This creates no newly supported hook execution.


Task8 D2 existing-empty-hook refinement: an earlier ordinary send creates the original app-owned legacy engine even with no configured handlers. Its mere presence is not an effectful hook. Workflow preparation may recognize only that exact RunHooksEngine and its existing _NativeLegacyHookSource attachment, original runtime/permissions/callback identities, no pending custody, and no published targets, while the saved hook configuration has no enabled handlers. Reuse the issuer capture/check; never fire engine callbacks. Configured handlers, custom/replaced engines or changed sources remain refused. This enables fresh workflow B after an ordinary gap without enabling workflow hooks.


Task8 D2 required-reference count refinement: the v28 window records a bounded snapshot_count alongside the ONE normalized ref table: NULL while required E is pending/unresolved, 0..64 atomically with conclusive E rows/refs, and zero for unused/removed windows. This is the necessary count witness distinguishing a genuine successful zero-row E from missing required references during restore validation; it carries no second mapping or authority. It remains inside the existing 65536-byte reservation and does not change v25–v27 catalogs.


Task8 D2 original native failure/host return refinement: a known failed native create/setup/thread start must finalize its original log and return its original frame under the pre-held close operation when no child thread was started. Unknown creation retains custody. Reserve one additional finite launch-host operation on the SAME child owner before dispatch callbacks, and finish it only after the original launch source/storage/execution scopes return; it prevents an early child or known failure root finish from releasing the original claim before host return. It adds one existing operation per admitted child, not an owner, scheduler or claim. Tool-worker automatic budget/authority checks must enter the existing owned_goal_storage scope before their first database read; the actual child model-loop coroutine must activate its exact original workflow source and owned_goal_storage before gateway checks. The unchanged original lifeline and pre-held child close/join retain protection through these real thread/source/storage returns. Live native archive evidence found AgentRunsDB leases on departed tool/model threads when those checks preceded the finite scopes; no cache reset or cross-thread close substitutes for their original return.


Task8 D2 approved owning-loop callback correction: the existing project-authority wrapper may invoke the SAME synchronous callback directly inside the original native Textual App context only when the captured dispatcher/context are original, current thread equals App._thread_id, and the actual running loop is App._loop. Worker callers continue through original App.call_from_thread; missing/mismatched loop or custom dispatcher/context preserves the existing route/refusal. This corrects the actual ordinary gap send reaching its authority callback on the UI thread without changing permission, source, recovery or activation checks; one positive native identity/thread/loop check, no replacement dispatcher or authority repin.


### Wave review fix1: receiving source, unused publication and native callers

Task8 D2 wave review fix1 (FIX_BASE c441672e4cffeb0310df9c133bd527ef883438f8): retain the actual native local provider base specs, admitted roots, alias-specific receiving specs and native handler function records after composition using the existing ordinary capture/check helpers. Revalidate through the existing post-permission/final task provider checks; replacement refuses before effects, without a fabricated ordinary tree, pruning, permission change or post-admission widening.

An original active-window body refusal is conclusive only when the same recorded body exception escapes the existing FULL transaction after successful rollback, PRAGMA restoration and original connection/storage return. A window-local private outcome binds that proof to the exact store, phase and inert reservation. Consume it only after the original launch source/storage scopes return. The same coordinator lock verifies original registering/no-operation state and removes only that inert reservation; original unused-source retirement executes outside the lock and must return, then each native no-launch frame must close before fence/configuration release. Commit/ack/unwind/source/cleanup uncertainty retains original protection; no inferred absent row, generic exception release, phantom unused row, schema change or ordinary required-write weakening.

The existing native-only task route remains supported without a Console frame. Its original bound owner, allowance, path/tool checks, native child preparation, fleet, worker/model scopes and cleanup remain authoritative. Only an actual original Console preparation adds wave anchoring, original-source wrappers and pre-held frame-close/launch-host operations; never create those operations for an absent Console caller. Audit bind, tool, launch, model/finalizer and no-thread failure together. ADR required: existing ADR285 and ADR084 apply; no new ADR. Cost: retained concrete instance records, one private no-commit outcome, exact inert cleanup and optional-frame checks. Native negative controls and existing native-only launch/cleanup tests qualify the changes; Task38531 remains In Progress, with Task8E/F/G/WHOLE outstanding.


### Wave review fix2: original unused storage return

Task8 D2 wave review fix2 (FIX_BASE c8dd34ef29255b153fcde9593acba098191d757b), N1: the original FULL no-commit outcome does not prove later source/storage retirement. Only that exact original _WindowUnpublished/store/phase/reservation/wave/preparation/source association selects an additional private close-worker guard. Before the outer owned_goal_storage/execution and frame/source scopes, capture each original native database thread-local object and incoming handle without acquiring a connection. After ALL those original scopes return on the same worker, require the SAME original local/handle values: borrowed handles stay borrowed; newly owned handles must retire. An unreturned handle, changed local, body/unwind/cleanup uncertainty retains the original wave/fence/runtime/config/source protection regardless of early _closed/_native_closed flags. No retry, replay, fresh source/owner or current-pointer adoption certifies an uncertain earlier return. This applies only to the new unpublished-window cleanup; ordinary pinned helpers and global DB close behavior stay unchanged. WHAT: actual original unused storage return before release. WHY: native WorkspaceDB close can suppress failure while its cache/lease/participant remain live. COST: a small worker-local original-handle tuple and conservative retention. ADR required: existing ADR285/084 apply; no new ADR/schema/public flag/registry/framework. A genuine reached original WorkspaceDB/native-close negative and actual rollback→ordinary-send positive qualify this rule; isolated retained exit is not drain. Task38531 remains In Progress, all Task8E/F/G/WHOLE requirements remain pending.


### Task8E original Pause/Stop and drained remaining acceptance (approved before behavior)

WHAT: serialize original `_run_owned` child reservation COMMIT and Pause/Stop CAS under one finite original-phase control fence; publish exact original task/owner/activity/service membership before releasing it. Resolve current config/limits/source and open the exact native connection beforehand; the invocation-local sealed fence supplies that same activated connection to existing FULL transaction SQL without redundant nested source entry. Preserve every native SQL authorizer, lease/quiescence/semantic/commit hook and default caller; no raw base connection or config bypass. Only exact original committed-task continuation at the exact persisted Pause revision may pass bind/model/token/tool receivers. Stop revokes first and requests original cancellation outside locks. Required original checkpoints/frame/source/B/E obligations remain exact terminal settlement, never a phase-wide paused grant.

A positively drained partial attempt preserves spent charges/results and closes only strict never-run tombstones. Fresh explicit remaining approval retains exact fresh successful siblings and creates only their complete never-run complement; actual failed/cancelled-run/stale/check-failed tasks require repair. Preserve caller command ID verbatim, link the complete immutable decision at control R to its atomically prepared attempt at R+1, and compare full identity on duplicate delivery. Real remaining acceptance uses the existing tracked preparation reader, actual OPAQUE finite owner/tool/claim and original source/storage; the same worker retains custody through actual original host acceptance/discard and source/finish RETURN before capability delivery. No reconstructed authority or new scheduler/registry/queue.

WHY: durable control currently revokes committed children and no source-closed same-phase remaining acceptance exists. COST: finite private commit/control serialization and source checks, one bounded same-invocation host handshake, conservative retained custody on unknown physical/storage/publication return, and normal new generation cost for a new attempt. ADR required: no new ADR; existing backlog/decisions/285-native-phased-agent-workflows.md and ADR084 apply. Spec10–11 and original Task8 step6 remain binding. Native RED/GREEN, named affected verification, exact BASE lint/formatter/preservation evidence and independent full E review are required; Task38531 stays In Progress/all AC unchecked and F/G/WHOLE remain outstanding.


Task8E retirement reconciliation (WHAT/WHY/COST): publish the existing native launch marker under the original TaskAuth lock followed by the same finite phase control mutex, preserving launch-versus-close atomicity. Settlement latches retired under that mutex, releases it, then rechecks the captured original launch/dispatch markers before any unused SQL tombstone. A marker is a retirement exclusion only; the actual child allowance COMMIT remains the Pause continuation cutoff. No source/DB/effect/drain callback or task/map/service lock is acquired from the control mutex. Cost: a short additional in-memory arbitration and conservative retirement refusal if native launch changed. This closes late precreated-row creation after unused settlement, without adding a claim or registry.


Task8E finite return ordering (WHAT/WHY/COST): the original worker retains source/storage/operation/OPAQUE ownership through the same-invocation host acceptance or discard and operation.finish RETURN. It records successful original scope unwind, then actually returns through `run_preparation_read`. Only that original async invocation, after observing physical worker return (including cancellation/error), retires the original root/claim and delivers or settles the same accepted cell. Unknown source/finish return keeps the original owner/cell. The existing first-send source scope spans the complete invocation. Cost: one bounded in-memory completion flag coupled to actual native read return, not a caller drain assertion or additional worker.


Task8E fence issuer closure (WHAT/WHY/COST): the private invocation-local native policy/connection fence is issued only while the original `_commit_task_launch` or `_control` producer frame is entering it. Direct entry from a phase holder is refused before source acquisition or the mutex, so the preopened-handle optimization cannot become an alternate native admission API. Cost: one bounded frame/identity check at the existing issuer; no registry or new authority.


### Task8E review FIX1 (approved before behavior)

WHAT: distinguish saved PREPARED remaining commands from inert accepted duplicates; refuse prepared retry after exact original command/decision comparison. Prove SAME-worker original DB local/handle return around ALL remaining-reader and terminal-settlement scopes. Keep the exact accepted-unused cell/root/OPAQUE claim through actual host, worker, source/storage, operation-finish and required terminal settlement; closing admits only this exact cancelled-unused terminal receiver through the existing tracked finite reader. Pin the finite original producer/receiver definitions at module definition time and bind native connection/policy consumption to the exact child-commit or control operation.

WHY: full E review found native PREPARED retry can use stale evidence, suppressed WorkspaceDB close can publish custody, actual controller shutdown loses the accepted cell before strict settlement, and mutable producer comparisons admit replacement functions. Four genuine unchanged-product controls reproduced these exact paths (4F100.93s; identical pre/post maps).

COST: bounded original-state/handle/function records and checks, plus replacing the already-required terminal offload with the existing tracked read seam while retaining SAME owner/cell. No new worker engine/owner/registry/schema/retry, generic disposed/revision bypass, raw SQLite API, or broadened default caller. Failed/unknown/cancelled terminal return retains custody and charges. ADR required: no new ADR; existing ADR285/084 apply. Source: ignored task-8e-fix1-source.md; final report will retain exact warning/receipt limits. Task38531 remains In Progress/all acceptance criteria unchecked; E review, F/G/WHOLE remain.


FIX1 actual shutdown return refinement (WHAT/WHY/COST, before edit): native shutdown observation1F32.86 proves original terminal service returned awaiting_resume, SQL native completed/phase closed, and terminal callback returned after all original scopes; runtime's repeated begin_shutdown then re-delivered cancellation from the tracked reader. Retain a private same-cell terminal_scopes_returned fact assigned ONLY after every original terminal source/storage/handle scope returns. After the existing reader has consumed the actual worker RETURN, a CancelledError with this exact positive fact and still-current original source may retire the proven unused cell/root/claim, then re-raise cancellation. Other errors, missing return fact, changed source or native scope/handle uncertainty retain original protection. No retry/reset/new authority; cost one exact callback-return fact and a separate cancellation branch. This refines the earlier blanket cancelled-terminal retention rule: known positive return is not unknown cleanup.


### Native read-only planning custody (Task 8F; approved before behavior)

WHAT: Planning uses its distinct original typed purpose and the existing native
Console preparation, one catalog/first request/AgentService, actual primary,
provider client, lifeline and finalizer. Original per-session intake/primary token
and source span a finite OPAQUE preparation owner, its actual retirement before
project-instruction consent, and then one fresh primary owner before client or
MODEL. Initial unsaved Chat request identity comes only after the same native
store's original committed publication returns. Saved immutable requests are
never rewritten. Forbidden tool-provider factories are suppressed before their
callbacks, including configuration, local service wiring and builtin factories.
Callable tools are only original physically bounded fs_list/fs_read and pure
calculator/time, rechecked at the final native invocation with exact source,
registry, provider, schema, permission, binding/root/exclusions and budget.

Mandatory objective and exact repair/checkpoint/handoff evidence must fit the
actual native input budget or refuse. The actual final_text is parsed once by
strict WorkflowPlan validation; immutable request/target identities and native
metadata are exact. Host verifier hashes come from original bounded native reads,
never model-invented bytes. Missing future wrapper checks remain blocked until
explicit revision after actual creation. No TaskAuth, execution phase, child
allowance, second planner/catalog, auxiliary model, queue, poller or executor.

The original primary root remains admitted and unretired through terminal work,
without an extra scarce tool/model operation. Dispatch is closed. A private
original receiver requires actual native model/tool/generator/lifeline client and
driver-loop RETURN plus all host/source/storage/handle returns, and verifies
empty native operations under a short pure capacity lock. The SAME live root and
source then own the canonical FULL plan write. Its actual required worker/host/
storage/source/handle RETURN precedes finish_root/on_drained and claim/source/
intake/token release. Unknown proof retains the originals and charges. No SQL or
callback runs under the capacity lock; root retention grants no new dispatch.

WHY: existing task-only preparation cannot authorize planning, unbounded native
observation/content reads cannot support physical read-only qualification, and
root retirement before required storage would create a drain/write cycle. COST:
exact process-local planning custody and receiving checks, bounded native read
loops, conservative OPAQUE serialization and retained resources on uncertainty.
Ordinary and execution/Goal source contracts remain unchanged. ADR required: no
new ADR; existing backlog/decisions/285-native-phased-agent-workflows.md applies.
Public Task8G journeys, WHOLE and root9/10 remain mandatory; parent AC stays open.


Task 8F conditional future verifier metadata amendment: reuse the existing original `_raw_shell_metadata` projection on the captured native App, require the already-published original runtime/raw source association, and narrow by witnessed current permission, local-tools, kill, persona and profile policy. Factor one pure `raw_shell_schema` descriptor from the original provider; its ordinary name/live gates still run first. The same planning catalog may retain that source-qualified host descriptor for prospective tasks, never as a callable planning tool or a constructed shell provider. Original arming/source publication must precede planning; unavailable metadata remains explicit. Revalidate original descriptor/schema/eligibility and actual bounded wrapper SHA at plan acceptance; future task execution still requires all native arming/Ask/source/Task7 guards. Cost: conditional availability, extra source checks, no automatic arming or enrichment runtime. Original store persistence/publish producer records qualify the actual return before immutable request creation; ordinary bodies are unchanged.

Original persistence extension: planning-only records cover ChatPersistenceService.persist_console_conversation_with_policy, validate_workspace_target and create_conversation, with original repository/DB binding and native store persist/stage/publish return. This closes replacement of the actual pre-request producer; ordinary method bodies and transaction semantics stay unchanged.


Task 8F final receiving closure: suppress optional micro-compaction before its scheduler only for the current original planning preparation/controller/owning Chat, keeping SAME native finalizer and persistence. Definition-time finite native prepare/lifeline/primary/finalizer and consent-helper records guard the original receiving path; disallowed direct tool names are refused before schema/coercion reads as well as at final provider invocation. No general registry, alternative finalizer or ordinary behavior change. Earlier storage-retirement failures remain explicit until their own source proof.


Task 8F native provider RETURN amendment: the original generic stream queues done before its guarded worker exits, so planning retains and awaits the SAME uncancelled worker_task and original off-loop close_completion under shield before native stream return. Preserve cancellation intent through actual returns. Planning propagates late worker/storage and response-close failure onto those same futures and retains the original prep/root/claim/token/charges; no plan can settle from a translated failure. Extend the existing planning storage-return check only with current-thread/source-participant lease, operation, pending-acquisition and retiring-thread before/after facts under a short pure lock, preserving borrowed originals. Exact original planning lifeline cleanup gathers pending native work without cancelling these futures; its existing bounded shutdown join remains and unknown return remains retained. Ordinary/execution behavior is unchanged. No new executor, registry, return framework or refill. Genuine held original worker RETURN RED: task-8f-worker-tail-red 1F21.92s, pre/post unchanged, planning incorrectly retired before the original worker return; this demonstrates this gap, not attribution of earlier intermittent storage failures.


Task 8F final physical listing and negative receiving correction: Python 3.12 Path.iterdir materializes os.listdir before yielding, so replace only the existing native _list_relative_directory scan with context-managed os.scandir, consume at most MAX_SCAN_ENTRIES+1 native entries and retain existing exclusion/symlink/sort/output semantics. The iterator closes on cap/error/return. Genuine observation RED1F1.31s saw all20 names materialized despite cap3 (unchanged pre/post pins); pytest also emitted unrelated old temporary-directory cleanup warnings. No glob/grep expansion/new reader. Existing sticky planning provider-failure also fences require_current and actual model seed/stream receiving before schedule/effects; cleanup remains return-only, no new authority flag or retry/refill.


Task 8F final original tracked-read proof: capture the SAME issued ConsolePreparationRead only inside its original invoke/source binding, checking exact creator/source/callback/issuing Task and active process/thread/task. Definition-time source records cover native read/binding classes, field descriptors, ContextVar, run_preparation_read and _retire; ordinary bodies remain unchanged. Before root finish require the SAME original producer Future physically done, uncancelled and exception-free, plus the SAME positively returned retired Future. Caller cancellation can mask a producer exception in the existing helper, so callback-local terminal completion alone cannot certify retirement. No new read/capability/map/E-cell. Genuine late-source-return plus concurrent cancellation RED1F16.45s demonstrated root wrongly retired; required canonical plan row alone remains distinct from live retirement.


### Task8F FIX1 — original receivers and required inputs (before behavior)

WHAT: close the exact original filesystem factory/concrete executor instance/reader receiving chain before callbacks and final invocation, retaining existing finite records and raw receiver checks. Capture the selected native session system prompt after original provider-selection return and preserve it through the SAME first-request composer, complete mandatory seed and fit-or-refuse budget. Carry distinct exact repair instruction on original prep/seed while leaving immutable request unchanged; bind command/workflow/revision/body equality with nullable immutable SHA256 on the SAME workflow_phase_attempts row. Actual integrated native/file/recovery schema28 advances to29 with matching additive catalogs and passive validation; old NULL remains unknown, exact duplicate returns original inert receipt with no new spend/capability, changed body conflicts. Count retained digest bytes under existing storage capacity.

WHY: FULL F review demonstrated a changed executor instance writing through fs_list and found omitted selected system context and explicit repair intent; existing identifiers alone cannot prove instruction equality. I2 current active ChaChaNotes operation remains unattributed. One exact original runtime maintenance callback/outer operation/Future hold will distinguish the concrete disposal return gap; no neighboring fix or passing observer establishes historical cause. Any I2 behavior change requires its own source ruling.

COST: finite original receiver/input checks and one64-byte equality column with native v29 migration/recovery catalog support; no new reader/planner/journal/table/registry/queue/owner. Preserve ordinary/execution/SSH/virtual/CAS behavior and original source/required-return refusal. Prior raw failures, PyAudio/frontmatter/temp-warning qualifications and E timeout limits remain. Task38531 stays In Progress/all AC unchecked; scoped FIX1 review then publicG/WHOLE/root9/10 remain.


### Task8F FIX1 — original runtime maintenance return (before behavior)

WHAT: retain and shield the SAME original run_owned_db_call coroutine at all four existing legacy-maintenance callbacks through repeated cancellation, consuming actual callback, operation and handle return before redelivering cancellation. Runtime disposal consumes the SAME maintenance task even when already done, preserves original errors/unknown return and caller cancellation; ordinary restart-safe retry remains outside disposal/cancellation. WHY: identity-bound original native control failed1F20.99s with31 equal pre/post pins: runtime.dispose returned while its pre-planning maintenance operation remained live; exact worker release/return retired that operation and original creators. This proves this separate runtime gap, not historical intermittent-tail attribution. COST: one temporary asyncio Task per existing callback and local cancellation handling; no global DB-helper change, new worker/owner/queue/registry or global-idle predicate. Existing shutdown deadline governs other cleanup; unknown physical return cannot become successful disposal. ADR required: existing ADR285. Source: task-8f-fix1-source.md; exact held-worker and minimum cancellation/error/default verification, static/protection/report and scoped review remain required.


Task8F FIX1 installed recovery authorizer amendment (before behavior): permit only SQLite internal quick_check(workflow_phase_attempts) during installed db.agent_runs migration and exact CREATE_TRIGGER(workflow_repair_input_immutable,workflow_phase_attempts). The real staged v24→v29 refusal exposed these missing native allowlist entries. Keep all unrelated actions denied, exact post-step catalog comparison/passive validation, and historical NULL as unknown input association. Cost two exact allowlist entries in the existing authorizer, no new migration framework or general PRAGMA/DDL permission. Existing ADR285 applies. Receiver-negative controls preserve actual failed/no-plan/no-effect requirements and independently proven original return; their earlier erroneous exception expectation remains failed evidence, not a passing refusal claim.


Final finite receiver amendment (before behavior): the existing retained-executor dict can invoke inherited hash/equality, which are absent from the class-defined original-method tuple. Require exact object.__hash__/object.__eq__ before any executor dict access, alongside original lookup/new and absent bool/len. Two pure identity comparisons, no new registry/callback/authority. Targeted changed-hash/no-effect and actual native bounded-tools controls qualify this final source; prior41P35.33s/44-pin cohort stays pre-amendment evidence.


Task8F FIX2 residual I1 (before behavior; FIX_BASE baf4d26d8435eea6774cbb8acd1801ea24049224): in the existing planning finite reader check, bind the actual workspace_tool_executor.resolve_workspace_path imported receiver to its retained defining original in local_tool_impls._NATIVE_PLANNING_READERS, preserving defining alias/code/default/closure checks. The same original guard runs before executor construction/factory callbacks and final native dispatch. WHY: scoped review proved four marker writes through this changed alias while original two-MODEL/plan/terminal/creator assertions passed (46 identical pins); earlier instance/reader controls do not cover it. COST: one pure identity predicate and one native no-effect mode; no executor body, registry, owner, read, callback or authority addition. Promote the exact write-then-original-delegate negative, require canonical failed/no-plan/one-MODEL/zero-child plus actual physical/terminal/claim/creator return, and qualify native tools positive plus bounded ordinary read/CAS/default controls. Existing ADR285 applies, no new ADR. Preserve prior evidence, protection and ordinary/execution/SSH/virtual/CAS behavior. Scoped FIX2 review then publicG/WHOLE/root9/10 remain; Task38531 stays In Progress/all AC unchecked.


### Task8G public native coordinator (approved before behavior)

WHAT: ConsoleRuntime lazily owns the explicit-request/decision workflow coordinator. Preserve complete immutable request intent and all execution targets while retaining the original selected planner separately; native readiness/source/target equality precedes first-save, acceptance and billing. Share the existing original tracked acceptance cell for truthful initial and remaining decisions, with atomic FULL decision/command identity, duplicate inert receipts and exact host/worker/storage return. Drive only bounded eligible native fixed waves from original completion/availability events, retaining the existing primary/intake slot through settlement and excluding typed workflow children from ordinary wake delivery. Retain SAME unused preparation only after the native no-dispatch wait returns normally through both inner and outer source/storage scopes, original operations/owner/claim and old B/E/fence. Closure, fencing, receipts, failed drains or uncertain returns never authorize retry.

Pause preserves E's committed-task cutoff; accepted read-only planning may finish before paused CAS. Stop uses the original accepted phase control fence/FULL revocation and original cancellation, retaining stopping until physical/required return. Resume requires a new explicit fresh remaining decision; repair preserves history/spend and exact instruction. Distinct bounded WorkflowRuntimeSnapshot subscriptions carry metadata only; private full history remains an offloaded service read. Shutdown retains original pending acceptance/planning/wave tasks until their actual return, with uncertainty visible and protected.

WHY: existing F/E/native wave producers lack their public coordinator; singleton request reconstruction, initial acceptance lifetime, transient capacity races and ordinary fleet wake synthesis require exact integration. COST: bounded process-local command/preparation metadata, source checks, native target resolutions and event subscriptions; no additional scheduler, poller, permission/budget ledger or execution engine. Direct F defaults and existing E remaining semantics remain compatible. Genuine native public/race/cancellation/unwind/refusal and focused affected evidence precede exact-path commit; headless proof is not UI/live/performance qualification. Existing ADR285 applies; no new ADR. Task38531 remains In Progress/AC unchecked until independent G and WHOLE gates.

Task8G native control return amendment: public Pause/Stop offload the original phase control fence through the existing tracked preparation reader and stock worker guard. The exact issued read, callback, issuing task, source phase, producer and retirement futures are retained; acknowledgement and overlays require positive original wrapper/source/storage/handle return. A same-run pending-control fence prevents physical driver release while this return is unresolved. Cancellation never replaces the native result proof, retries the FULL CAS, borrows a foreign async activation context or clears uncertainty. This adds finite return bookkeeping to the existing producer, with no new execution authority or UI-thread blocking transaction.

Task8G capacity-return refinement: only after the exact eligible no-dispatch wait and its original undispatched cleanup return, the same prepared child records a negative in-progress marker under its authorization lock. The local candidate then returns through the original inner operation and source scopes. Only both normal returns replace the negative marker with that exact positive candidate. Any inner exit error, including the same exception object, leaves negative metadata; the wave retains original operation/preparation/fence and never closes, queues, or retries it. Non-wait paths do not install this marker. Wave pending eligibility additionally requires all original outer scopes, operation, owner/claim and executor returns.

Task8G projection correction (source-backed): `runtime_snapshot` reads the latest native phase attempt's plan revision, phase ID, purpose, status, retirement and control revision, and the latest saved plan revision. Only an execution attempt with status=completed and retirement=drained for that latest plan can project awaiting_result_review, and only while the durable floor is running/awaiting_approval. A committed quality-complete decision must match that workflow+plan+phase and have control_revision at or after that phase's accepted revision, with payload action=complete/evidence.kind=quality; it projects the durable awaiting_approval floor as next-phase review instead. A later/new phase wins by latest ordinal, a newer plan excludes old completion, and paused/stopped/recovery/completed/removed floors remain unchanged. Metadata projection grants no execution capability. Frozen settlement-timing showed a completed native phase incorrectly projected awaiting_phase_review2.30s after provider release, before unchanged5s timeout; this is a projection defect, not a timing waiver.

Task8G original pending-acceptance amendment (controller-approved before behavior): retain the existing E acceptance cell on the exact coordinator-owned pending run before its first tracked read/root/claim/ledger operation. Association checks original coordinator/controller/runtime/service, session, command, full decision and issuing Task. No missing-cell/read/owner marker proves retirement. Original body refusals are carried locally through normal source/storage/read-wrapper exits, then re-raised only after the same producer/retired futures, post-source checks and original owner/claim retirement. The terminal reader separately retains its exact original returned settlement snapshot and read/future proof. Unknown returns keep the same phase/cell/intake/token and recovery status, with no CAS retry or execution from a cell/ID. Pending Pause/Stop cancel only the original issuing task; before cancellation require the user's expected current revision and exact initial decision/attempt relation. Accepted revision is decision.rev+1; returned unused terminal snapshot must be same workflow at decision.rev+2, with original command/decision/attempt identity and unchanged current revision. Pause retains the resulting awaiting_resume floor; Stop uses only that exact terminal receipt for passive stopped CAS. PREPARED leftovers and competing revisions remain refused/recovery, never adopted.

Task8G E return-only wrapper refinement (controller-approved before behavior): the existing tracked reader uses original source identity checks on wrapper entry/exit, while the worker still checks current admission before every prepare/accept/effect and after host handoff, and the host checks current after read return. Closing after acceptance therefore reaches the same unused-cell terminal path after a positively returned reader; changed sources or producer/retirement failure still retain custody. This does not authorize post-close admission. Direct shutdown controls cover both COMMIT sides, original returned settlement and owner/claim drain.

Task8G command cancellation amendment (ordering correction): after caller cancellation, await the same original command task to physical completion, retain its original exception on that task and the original read producer/cell, consume it for diagnostics, then redeliver caller cancellation. Ordinary uncancelled calls and duplicates propagate that same original command exception. Unknown acceptance retains original run/owner/claim/intake/token and recovery; no retry or cleanup inference. Chronology: the four-line drain-loop exception catch preceded this required source/governance ruling, was reported to the controller, and its frozen failed evidence was preserved; this amendment does not claim prior authorization.

Task8G accepted Resume/routing/close refinements (before behavior): planning-paused Resume requires a current saved plan and the actual latest phase attempt (not latest matching planning row) positively completed/drained for that exact plan publication and Pause revision relation, rechecked inside the original FULL transaction together with original _active coverage for phases/tasks/intervals/windows and all existing freshness/binding/accounting/allowance/CAS checks. Existing awaiting_resume behavior remains. Target resolution first considers actual native session selection, including configured default models, and only then independently configuration-derived selection; full ready/reference/source equality is mandatory, never routing from a requested endpoint reference. Runtime close may release pending admission only after its SAME command and original accepted-unused cell/terminal reader/owner/claim positively return; unknown or PREPARED retains original custody and cannot be described as successful drain.

Task8G F/G control-return release amendment (accepted before behavior): original G scheduling validates its definition-time planning driver and retains the exact Task/run; original F receives that run as an invocation-local private argument, validates exact original coordinator/driver/coroutine/source/preparation/phase at entry and final release, and cannot fall back to direct-caller behavior after a mutable association disappears. Genuine direct F calls remain unchanged. F awaits the same pending control Event through repeated cancellation, with no orphan task, then rechecks exact Event/return/source/physical proof before synchronously releasing original root/claim/intake/token and redelivering cancellation. Unknown or replaced sources retain custody. Concurrent controls cannot replace a pending or unknown Event, including planning Pause overlay. Native controls must cover held F terminal plus control return, mutated/missing association, repeated cancellation, late failure and actual positive creator retirement.

Task8G sticky recovery amendment (accepted before behavior): pending, native and late control overlays preserve existing recovery_required. Planning completion rechecks recovery before/after passive status CAS; both original admission release paths refuse recovery or control_unknown before token/intake/pop. F itself also requires same original run not in recovery at its final source/control/physical check before root/claim/slots release. Positive unrelated control may record revocation/action but cannot clear another unproven source/return. No automatic recovery transition exists; original custody remains. This corrects the native replaced-driver/late-positive-control case without weakening its retained root/claim/status assertions.

Task8G truthful shutdown amendment (accepted before behavior): after awaiting the same original commands/drivers and exact positive pending-cell releases, close refuses any retained G run before runtime enters downstream _dispose_owned/capacity/store teardown. Workflow admission stays sealed; repeated close/dispose keeps the same unresolved originals and never re-CASes or replaces them; existing independent finally drains remain. Caller cancellation is retained while original waits complete and redelivered only after safe disposition. Known F cancellation is classified only when raised by the original run_planning await, before any later G status operation: prove SAME original producer/retired/actual terminal snapshot, current original return-source, positive owner.on_drained plus !admitted, exact claim gone and intake/token None. Settled/closed flags only support this proof. Do not call F's pre-release physical proof after root release. Existing recovery/control_unknown stays sticky; all other cancellation remains unresolved.

Task8G conclusive pre-prepare refusal amendment (accepted before behavior): only the same original E invocation body error, with preparation_entered still False and exact unchanged producer/callback/source/cell/issuer, positive original read producer/retired and source/storage scopes, and original owner/claim return, permits releasing G admission while re-raising that exact error. Set preparation_entered immediately before the original ledger prepare call. Any entered prepare, missing/mutated handle/source/owner or ambiguous return retains the original run/intake/token; no retry, refund, or capability is inferred from a missing local attempt. Direct initial and remaining stale decisions must leave billing/phase rows unchanged and permit a separately explicit valid approval; existing PREPARED/no-reader uncertainty remains retained.

Task8G metadata publication lifetime amendment (controller-approved before behavior). WHAT: G metadata get/publication reuse the original tracked preparation reader, same stock worker/source/storage scopes and exact read/producer/retired proof. Each actual admitted invocation retains its original issuing Task, source, run/phase or absence and separate host-disposition Future. Existing issuing Tasks await the prior original disposition before their own finite metadata read; no new task, thread, shared lock, actor or execution authority. This is not a bound on caller count; positive disposed cells retire, unresolved originals remain. Keep SAME final driver/run reachable through actual worker and synchronous host callback/discard, then release/pop without an intervening await. Closing seals external metadata admission, permits exact already-owned terminal callers, drains admitted host dispositions without joining unrelated UI Tasks or self-waiting, and refuses unknown return before downstream teardown.

WHY: previous run removal preceded final publication, and bare to_thread cancellation did not prove worker/handle return. Runtime projection also read multiple autocommit statements, and G get combined separate snapshots. Reuse the existing nested-safe BEGIN/read/finally rollback pattern for runtime projection and one shared worker-only read transaction for full history plus projection; no migration or transaction across await/callback. Native revision/progress do not cover every accounting/retirement mutation: serialize actual read+host disposition, allow equal counters, reject componentwise regression only against positively observed native receipts. Capture original run/phase before waiting, recheck after waiting and worker; superseded publication discards and superseded get explicitly refuses with workflow_snapshot_superseded, never borrowing replacement overlay or rewriting revisions. Check closing/token identity per synchronous subscriber, and isolate callback exceptions.

COST/LIMITS: finite passive per-invocation return bookkeeping and serialization of metadata reads/host dispositions, retaining only actual admitted cells. Caller cancellation and original producer failure are separately retained; positive source/outcome permits discard then cancellation redelivery, unknown source/producer/handle retains recovery and refuses close. Conclusive pre-read cancellation and original read body refusal are distinguished from missing return after reader entry; no automatic retry. Genuine held-worker/close/repeated-cancellation/late-error, coherent concurrent snapshot/equal-counter/source replacement/external host-tail controls required. Earlier raw failures and 60s kills remain unattributed and are not qualified by this source fix.

Task8G final metadata original-helper closure (controller-approved before behavior). WHAT: definition-time descriptor/function/code/default/closure records seal G publication, metadata wrapper/body, get and lower-bound methods. Check original records before invocation, after worker/host disposition and immediately before final release. Require returned carrier IS SAME original run.final_metadata cell with exact type/issuer/run/phase, positive original host Future, same native reader/producer/retired proof and no body/producer/source errors before any pop or slot release. Only an original unchanged conclusively cancelled pre-read wait with explicit no-entry/proof-absence may discard without a native read. WHY: dynamic helper replacement or a missing/forged returned carrier must not release original custody, including planning's prior pop-before-dereference order. COST: finite native source/identity checks and pre-call/held-worker replacement controls; no registry, retry, capability, owner or source adoption. Unknown retains same run/cell and close refuses. Existing metadata transaction/serialization/cancellation rules and historical evidence limits remain. ADR required: no new ADR; existing ADR285.

Task8G positive runtime cancellation handoff (controller-approved before behavior). WHAT: only the explicit private original-runtime disposition of coordinator.close returns its saved caller cancellation after exact positive G commands/drivers/read/host/run disposition. Independent direct close retains ordinary cancellation redelivery. The original runtime.dispose caller passes its SAME runtime identity, continues the existing planning/wave/_dispose_owned path, and preserves native post-disposal cancellation plus independent finally drains. Unknown/source/producer/host errors still raise before downstream teardown. WHY: immediately redelivering proven G close cancellation skipped native downstream disposal; empty sets or caught arbitrary cancellation cannot prove original close returned. COST: one private passive runtime-identity option and saved cancellation local, no new task/lock/owner/authority or cleanup bypass. Strengthen the actual held-worker repeated-cancel test to observe native _disposed/capacity closure before first dispose cancellation; original creator return remains separately proved, and uncertain controls still forbid those destructive effects. Existing ADR285 applies; no new ADR.


### Task8G FIX1 reviewed predecessor, pending planning and affected-native corrections (before behavior)

ADR required: no new ADR; existing ADR285 applies. FIX_BASE bb983093456803f006774435f7384924d923fc34. Independent review found five Important issues; Task38531 stays In Progress with all three AC unchecked. I2 active-clock accounting and I5 real survivor notification remain separately gated/pending; this amendment does not certify either or erase original155P28F5E and historical killed/unknown evidence. Exact source, prospective hashes, rulings and diagnostic receipts: ignored task-8g-fix1-source.md/task-8g-fix1-pre-edit.json.

WHAT (I1): map only the SAME accepted decision's exact prior-phase result references to positively completed/drained latest task attempts and native checkpoint/run/result identities for that plan, retaining same-phase remaining partition checks. Use those IDs for dependency eligibility; original native preparation/launch repeats actual binding, dependency/handoff and freshness checks. WHY: valid cross-phase dependencies currently never enter G's done set and strand a charged phase. COST: one bounded passive native mapping read, no new evidence cache or authority. Genuine dependency/handoff, stale/failed predecessor and same-phase remaining controls must prove real charges and creator return.

WHAT (I3): retain the SAME passive pending G planning invocation, original command Task/issuer and native F prepared cell before its first read/admission. Associate original unsaved identity through its actual host handoff, preserving exact request/repair and direct F callers. Stop/Pause/close reach only that original operation, retain read/producer/retired/source/storage/owner/claim proof through repeated cancellation, and never reconstruct authority from an ID. For positively returned but unaccepted original planning, one existing-ledger FULL abort transaction compares exact receipt/native/workflow/revision/repair/generation identities, requires still-PREPARED planning with no model effect/open interval/task/window linkage, then changes only reserved generation to released, native prepared to aborted, and exact phase prepared/pending to closed/drained/result_bytes0, with exact rowcounts. Accepted generation is never refunded. Unknown COMMIT/wrapper/source/retirement keeps original custody, forbids Stop CAS/retry/release; an inert aborted row is not drain proof. WHY: G currently registers after F returns, so accepted pending planning escapes Stop, and a proven pre-acceptance cancellation otherwise lacks safe reserved-attempt retirement. COST: finite original invocation/return bookkeeping and one narrow terminal receiver using existing schema states, no migration, second planner, owner or capability. Actual pre/post COMMIT initial+repair, unsaved identity, duplicate/foreign revision, missing/source/late-return, accepted-no-refund and current recovery-validator controls remain required.

WHAT (I4): change only transferred consuming test setup/hunks to use the actual bootstrap profile and SAME runtime/store/controller/source attachment. Preserve seven-path ownership and original cached c5. The actual diagnostic records raw_source_selection_changed, missing _hooks_v2_runtime, missing original hook source identities, and separate hook_activity_busy from retained opaque custody (4F1E/15.14s;8316 equal pins), not a compatibility pass. Early archive cancellation precedes actual attachment handoff, so prove exact staged/no-loss ownership there and separately exercise genuine post-transfer recovery; no fabricated argument or hand-seeded custody. While the original injected hook terminal refusal holds, require actual other-session command refusal and same opaque claim; after original reap/close positively returns the owner/process/claim, separately explicit other-session command succeeds. Nonrefusing behavior stays unchanged. WHY: existing tests do not reach their intended native boundaries or contradict actual custody order. COST: narrow authentic fixture/oracle corrections and focused original native checks; no guard reset, fake owner, full sweep, timeout waiver or assumption these failures are regressions.


### Task8G FIX1 I2 original quiescence accounting and cleanup (approved before behavior)

ADR required: no new ADR or migration; existing ADR285. WHAT: implement the exact return contract and cleanup constraints read in root-i2-accounting-ruling.md and root-i2-cleanup-accounting-ruling.md, with the complete receiver inventory in task-8g-fix1-source.md. G may close the existing active interval only after SAME prior finite preparation/source/storage/owner/claim/worker and actual wave/B/E/receipt return proof. Its original FULL quiesce transaction leaves the accepted phase retirement=unknown. No committed drained marker or second acknowledgment transaction proves outer return. One bounded passive record on SAME original phase/run identifies the interval/transaction, issuer/callback/read/producer/retired and prior physical records; it becomes positive only after the original whole wrapper and all source/handle returns are positive. No new global registry, actor, execution owner or ID-derived capability.

Receiving inventory: ledger _workflow_active_seconds, _snapshot, _check_admission, check_active, reserve/claim/pre-reopen admission, admit_call and native commit consume only SAME already-held native phase/TaskAuth/dispatch/current AutomaticWorkContext, or the exact original metadata cell which captured that live G run/phase. Pin original context getter/class/slot/method and receiver definitions; retain actual object/reverse/source links, never select a new phase by ID. Pass this distinction through WorkflowRunService.get/runtime_snapshot and WorkflowRunsStore.get/_snapshot, _prepare_review and actual task-input preparation/revalidation. Metadata context must still be an admitted original cell with same issuer/read/source/run/phase/disposition, checked before/after worker and host; captured absence or supersession cannot borrow current proof. Coherent transaction, original host lifetime and all current source checks remain.

A valid same-original positive record satisfies ONLY its one durable unknown-quiescence contribution. Closed prior charge then remains exact and original native work may reopen without refill. Without that record, active-time/admission reads raise existing clock_unknown; readable history reports uncertain accounting, preserving actual counts and never substituting zero or an invented measured ceiling. Chain review_required, uncertain reservations, other phases/intervals, source/control changes and sticky recovery continue to block. Never globally clear snapshot.uncertain. Plain DB-only readers and restored rows have no witness; startup must not clean-retire this accepted unknown phase, even when the old process actually acknowledged idle. The consequent conservative recovery after crash is an explicit cost.

Before actual B/capture, finite preparation, provider/tool/child work or cleanup resumes, consume/invalidate idle disposition so concurrent observation cannot keep claiming idle. Ordinary work reuses original resume transaction with all current owner, revision, budget, binding/freshness, source and dispatch gates; no effect follows an unproven resume wrapper. Reserve, poll, acknowledgment and quality acceptance do not reopen or retire this record. An original resume COMMIT followed by wrapper failure leaves its opened interval charging and original run/record retained, never reset to idle. Only original native dispatch or positively proven terminal settlement retires the record.

Necessary cleanup after Stop/revocation or exhausted budget is distinct: one private branch in the SAME native resume receiver accepts only SAME already-owning phase/preparation, positive original wait/interval record, current runtime owner and definition-time original close/terminal caller/source. It may bypass dispatch revision/budget admission ONLY to charge already-owned cleanup, opening the existing interval/retirement transition in the original FULL off-loop transaction. No workflow/control status, reservation, resource counter/limit change, MODEL/tool/child launch or returned dispatch capability. Preparation/capture can never use this branch. Accounting failure must NOT prevent physical cancellation/reap/close already permitted by the original native custody: retain exact accounting/source/return errors and unknown projection, continue only that originally authorized cleanup where its own contract permits, and never release unproven resources or manufacture a new cleanup path. Physical cleanup and proof of successful retirement remain separate.

WHY: current G charges a positively idle capacity waiter, while a naive quiesce commit could silently undercharge unknown cleanup and look clean on restart. COST: finite per-phase original return records, narrow native/captured-metadata context plumbing, a guarded negative bookkeeping cleanup branch and conservative passive/restart uncertainty. No authority or accounting ledger replacement. Required evidence separately covers positive stable real wait/no delta; resumed B/preparation/held provider and cleanup charging; repeated cycles/no refill; alive/unknown pre-quiesce refusal; post-COMMIT source/producer/retirement failure; passive clock/snapshot/review; copied/mismatched/superseded proof; revoked/expired cleanup/no new effects/counts/status; direct misuse; restart after positive-live and failed-return unknown. Existing direct/ordinary contracts and all earlier raw failures remain. This source/governance approval is not a test result; all-five independent review remains required.


### Task8G FIX1 original finite and wave return association (approved refinement)

WHAT: the same original finite preparation producer Future carries one immutable tuple binding its original preparation, issuing task, callback, issued read, owner, claim, operation, body result/error and finite record identity. Validate the actual positive producer/retired/source, unchanged definition-time native owner methods, actual scope and owner/claim return and every tuple association before unpacking the original public result or re-raising its original error/cancellation. A producer, source or retirement failure remains unknown; a mutable local record cannot substitute another owner. Preserve direct F callers and existing exception precedence.

At the actual original wave return retain the same wave/driver/B/E Futures/source/reservation/receipts and original returned snapshot-ticket owner/claim/operation/retired references. Check definition-time native wave/runner and ExecutionOwner descriptors/functions before use and after return. Require the genuine original worker/handoff/ticket/claim returns, not just mutable completed flags. An unused baseline has its native cancellation disposition and actual B return, with no fabricated E. These tuples are passive evidence only: no new read, worker, owner, registry, execution authority, stored-ID reconstruction or accounting exemption.

WHY: mutable finite owner/claim fields and later dereferencing of mutable wave fields cannot alone establish which original scope actually returned. COST: finite identity/source checks and immutable return tuples; direct changed-owner/claim/wave/source and copied-cell negatives plus native positive controls. Earlier clock1P is scoped to its preceding bytes; no retrospective qualification. Unknown custody, all accounting gates and conservative restart behavior remain. Existing ADR285 applies; no new ADR or migration.


### Task8G FIX1 return-source shadows and restart accounting (approved before behavior)

WHAT: complete the existing finite return checks with the original OwnedOperation module alias and SAME returned operation/ticket/source/reservation instance method descriptors against their already-captured original class records. No broader registry, owner or authority. WHY: class-only records can overlook a replaced module alias or instance shadow when consuming the original evidence. COST: finite identity checks and conservative refusal; direct replacement-instance negatives and native positives remain required.

WHAT: the existing unknown-clock reader includes native workflow attempts in accepted OR review_required state when phase retirement is unknown. A review_required row always contributes clock_unknown, with no live-witness exemption. Only the SAME still-accepted native phase/owner/attempt can consume its exact positive return witness after the original current runtime-owner/source checks. Preserve every other unknown row, chain/reservation uncertainty and admission gate; no state rewrite, schema, recovery acknowledgment or new authority. WHY: actual recover changes accepted to review_required and the phase to uncertain/unknown; the accepted-only query otherwise loses that uncertainty and reports a closed sum as exact after restart. COST: conservative clock refusal for unreconciled recovery rows. Genuine positive-live/restart and late committed failure/restart controls must prove actual native recovery transitions and unchanged retained charges, never reconstruct live custody from IDs. Existing ADR285 applies; no new ADR/migration. All prior failed/killed/qualified evidence remains unchanged.


### Task8G FIX1 original fleet-attention storage return (approved before behavior)

ADR required: no new ADR or migration; this amends existing ADR285's native survivor/physical-return contract. WHAT: the existing synchronous fleet attention callback owns only newly acquired exact native file-backed AgentRunsDB/CharactersRAGDB handles through existing operation_owned_connection. On the receipt branch, read only exact native receipt/marks objects' already-stored database fields, deduplicate actual identities, and scope complete receipt/fallback work. Preserve original app/custom/lazy getters: fallback marks lookup remains at _write_mark entry; title database/getter remains at actual inline or queued-host _conversation_title, then scope only that actual native query. No new getter or initialization on ignored/queued events. Borrow existing registered handles/transactions; leave custom and memory owners unchanged. Existing fanout/fallback/title exception handling remains, including observable retained leases when close fails; a caught error or returned Thread/owner never proves storage drain. No cache force-close, new actor, authority, or wake-order change.

WHY: the approved original opener diagnostic proved publish_fleet_drain→clear_fleet_fallback→marks.clear_mark opened a SAME surviving child's ChaChaNotes handle that remained live after that exact native child target/Thread and owner returned. The direct native parent has no ordinary-tree storage stack. The finite receiving callback must retire its own new handle on its original thread; fabricating a tree, borrowing G custody, suppressing attention or closing a foreign cache would hide the defect. COST: narrow native scopes on one consumer and its existing fallback/title queries; close errors can conservatively retain storage, and no broader service lifetime is changed. Native same-thread connection/lease return, borrowed active transaction, lazy/custom/memory, fallback/inline/close-error and unchanged consumer controls are required, followed by one materially changed-source original G setup with saved actual Threads and creator return. The prior1F diagnostic is preserved; fleet-notification causality and enabled mixed-wake still need separate genuine controls. Parent remains In Progress/AC unchecked.


Task8G FIX1 native lazy receipt refinement (approved before behavior). WHAT: include the actual _LazyConsoleActivityReceiptService alongside exact ConsoleActivityReceiptService when selecting the already-owned receipt databases. Read only stored vars(_runs_db/_marks) for lazy and vars(_db/_marks) for eager; original publication access remains the first lazy delegate initialization. Do not call _get_delegate or custom/app/title getters during discovery, and retain the same native file-only, borrowed/custom/memory and close-error boundaries. WHY: the real runtime installs this exact lazy wrapper; changed-source native observation proved the eager-only branch skipped it and left the same original child connection live. Existing native hydration already handles both exact types. COST: one precise existing native-type branch and direct lazy/ignored-event/native handle controls, then one changed-source saved-thread G setup with actual close/creator proof. Preserve prior failed runs and all original bounds; setup alone proves neither unique fleet wake causality nor enabled mixed-wake behavior.


### Task8G FIX1 known open cleanup charge clarification (approved before test)

ADR required: no new ADR/migration; existing ADR285 applies. WHAT: preserve the existing native distinction between continuously charged original open cleanup and unproven physical return. WHY: a late producer error must retain recovery/custody without fabricating an unrelated token/time uncertainty. COST: conservative continuing charge and separate direct physical-return/refusal evidence. The complete binding clarification follows; no product behavior is authorized by this amendment.

**Root ruling — known open cleanup charge and unproven retirement are distinct**

Ruling: the proposed late-cleanup negative does not have to force AutomaticWorkSnapshot.uncertain=True or clock_unknown solely because the outer cleanup accounting producer fails AFTER the original native cleanup COMMIT has opened the attributable interval. A positively verified, current-clock-owned OPEN interval that continues charging its full duration is the existing conservative accounting disposition. Explicit recovery/refusal and retained original custody must carry the unproven return; a false accounting flag is never proof of physical retirement, successful cleanup, approval readiness or final known total.

Root read BOTH full existing I2 ruling bodies, the complete current writer source appendix, actual AutomaticWorkLedger._snapshot/_check_admission/_workflow_current/_workflow_clock_unknown/_workflow_active_seconds/resume_workflow_phase, WorkflowRunsStore._active, workflow service runtime/review projections, and G _accounting/_execute/_retain_final_metadata/get/native-control paths. This clarifies the existing contract, not a passing native control or new admission/cleanup authority. The first ruling explicitly separates exact time from phase retirement; the cleanup addendum already requires an opened interval to continue charging after a later COMMIT/wrapper failure.

Required boundaries:

- The ORIGINAL cleanup transaction must actually have reopened the SAME interval under the existing native current clock and changed its accepted phase retirement to pending. Observe that actual durable state and its increasing charge through the real receiver; a reported COMMIT/Boolean or an in-memory mode alone cannot prove it. Keep accumulated prior charge, counters, limits and every unrelated uncertainty unchanged. Do not close/reset the interval or restore the idle witness after the error.
- For this specific known-open case, recording the ACTUAL accounting.uncertain flag is correct. Do not broaden the meaning of that flag into a generic execution-lifetime marker or change otherwise known token/time accounting merely to satisfy an invented test oracle. Preserve chain review_required, uncertain token reservations, other unknown phases and actual clock/source errors; those still require their existing uncertainty/refusal.
- Require the SAME original saved producer error/accounting cell/run and real recovery_required projection. Original preparation close must continue only under its existing native custody despite accounting failure, and actual close/producer/retired returns must be checked separately. The accounting failure must retain original run/intake/automatic-primary token and prevent normal release; repeated close/admission and quality/continuation controls must refuse. An open interval or accounting.uncertain=False cannot clear that failure or authorize replacement work.
- This ruling does NOT permit an absent/uncommitted/foreign-clock/changed-source interval, a closed unknown interval without its exact live witness, or a lost-return restart to yield exact time. If those conditions cannot be verified, keep original clock_unknown/uncertain accounting and refused admission. On restart, use genuine native recovery and preserve its conservative unretired-interval handling; do not reconstruct live proof from rows. The earlier positive-live and failed-return CLOSED-unknown restart controls remain separate evidence.
- The native test must assert the actual increasing OPEN charge, no restored idle disposition, exact error identity, permitted physical cleanup continuing, original uncertainty/custody retained, no new provider/task/reservation/refill and repeated-close/admission refusal. Record the actual flag/status/interval/phase independently. Its deliberate retained process exit is NOT creator drain. Preserve every earlier failed raw; no deadline change, DB-row edit, guard override or new display-only exception.

This is a scoped contract clarification/test oracle ruling; no product behavior change is authorized by it. If actual current source fails one of these requirements, surface that observed gap before proposing a fix. Record the complete clarification and reason/cost in canonical governance/readback before the late variant, without rewriting either earlier full I2 ruling or historical receipt. Independent scoped review still judges complete I1–I5 FIX1; root has run no Task8 test or probe.

Cost if wrong: callers could mistake known ongoing charge for proven cleanup and enable unsafe continuation, or a falsely claimed open current-clock interval could undercharge failure. Keeping separate accounting and retirement signals requires honest recovery/status UI and direct native refusal/retention controls. Conservative continuing charge can exceed actual physical cleanup time until native recovery reconciles it; that overcharge is preferable to falsely crediting the unresolved invocation as idle and never supplies a final measured total.


### Task8G FIX1 independent ordinary delivery context (approved before behavior)

ADR required: existing ADR285 amended; no new ADR/schema. WHAT: in the one original ConsoleFleetWakeCoordinator._attempt task creation, use loop.create_task(coroutine, context=contextvars.Context()). The same original independently authorized ordinary delivery Task begins with an empty context. No other scheduling or Task context changes, especially none at common ConsoleRuntime custody: the newly issued AutomaticWorkContext must flow unchanged from original _deliver into original received custody and source preparation. Keep all native wake claim/auth/ledger/source/fence/admission and physical return gates. No original child/host guard is reset, mutated, adopted or treated as returned by this change; fresh context supplies no authority or retirement proof.

WHY: pre-fix native task-8g-fix1-mixed-context-entry failed1F29.34s with8316equalpins and directly observed SAME original child guard4722025648 via actual retry Handle(s)/coalescing TimerHandle into deliveryTask4730403328, custodyTask4722693824 and tracked source producer4729270576. Actual origin(pid84432,thread6431371264) differed from delivery host8461753728/task and producerworker6263107584; unchanged native guard refused at execution169. Same child guard remained unchanged at retry return; child/owner/lease and creator genuinely retired afterwards. That proves this accidental context inheritance, not ordinary delivery success. Earlier failed/timeout/unknown receipts remain unchanged.

COST/alternatives: ambient tracing/custom ContextVars are intentionally not inherited into this new independent delivery Task; custom task-factory compatibility requires direct testing. A fresh context at common custody would incorrectly discard freshly issued native wake authority, and is rejected. Global guard reset, borrowing child execution, changing source.prepare/shared reader isolation or a new actor/context registry are rejected. Test only the two existing raising/recording scheduling double signatures for the new context keyword, preserving all existing failure/closed-coroutine/bookkeeping assertions. Require native fresh-delivery guard/Handle/Task/custody context identity, real enabled mixed G exclusion plus unrelated delivery/known usage/counts/no refill, original child and creator returns. Ordinary queued-before-G-release liveness remains a separate diagnostic after ordinary admission succeeds; no inference from this task boundary or from setup alone.


### Task8G FIX1 execution release notifies native pending wake; permanent close fences first

ADR required: existing ADR285 amended; no new ADR/schema. WHAT: only after the SAME original G execution phase, child/wave/source/worker and final metadata producer/retired/host disposition positively return, require the exact original received-slot and automatic-primary releases to return True and SAME run removal before requesting one existing native wake.retry_soon. Bind the original issuing Task/loop/runtime/controller/store/wake/registry-lock/intake/token/phase at execution entry; revalidate finite definition-time receiver/method/code/default/closure sources before release and immediately before notification. Wrong, missing, replaced, copied or uncertain originals cannot release or notify. A successor is never adopted. This changes no ordinary priority, claim, admission, provider, ledger, source, context, charge or result gate. Q1 covers this execution release only; F/E/chat-start/Goal release liveness remains unqualified and the broader Q2 common-release notification is rejected because it can immediately loop after native delivery-task creation failure.

WHY: the genuine queued-before-G-release native diagnostic failed1F48.17s (PID/PGID93129, actual wait1,8316 unchanged source/test hashes). Both actual unrelated child IDs reached a natural timer and SAME G-token refusal, then original G terminal read/cell/phase/driver/token release and creator cleanup positively returned. At original20s delivery failure, both IDs were still pending, there was no retry timer and no ordinary submission/delivery Task; every observed retry occurred before G release. This is distinct from the passing ordinary-completion-after-G-retirement case and earlier observer failures. Notify actual availability after return rather than synthesize a wake, bypass priority, poll idle, reset guards or refill a chain. The changed-source native journey and direct negatives remain required; this amendment is not a gate pass.

Keep notification OUTSIDE the existing physical-failure try/catch/finally. Save original publication.caller_error before release. A later receiver/source/scheduling error is logged and propagated as the SAME post-release exception, never reinserted run/cell, poisoned disposed metadata, invented recovery custody, restored token/slot or repeated CAS/charge. If original caller cancellation coexists, redeliver SAME cancellation with actual notification failure as cause. Native retry's existing disposed/no-loop/closed-loop and RuntimeError policy remains; scheduling return is not delivery or physical-return proof.

For permanent ConsoleRuntime.dispose only, move the SAME existing stock-field/exact-native wake.dispose before the first G.close await, immediately after _workflow_closing=True. This seals native _disposed before already-posted retry Handles can attempt new delivery; _workflow_closing alone is not an ordinary ingress guard. Retain SAME recovery_owner through original finally/drain_recovery. Do not invoke custom facade getters, controller.begin_shutdown early, unrelated teardown, a new cancel owner or a replacement source. If native fence presentation/cancellation throws, save exact error and still perform original already-owned G/planning/wave physical cleanup and independent finally drains; redeliver at the former fence-error boundary before _dispose_owned. Preserve original cleanup uncertainty as primary and actual caller-cancellation precedence with separate fence error observable. Without fence failure, positive downstream disposal still precedes saved cancellation.

A pre-invocation source failure is NOT a sealed fence or successful disposal. Immediately before the new Q1 notification, on SAME original host Task/loop, verify current original receiver/source/registry-lock; under that saved native lock read runtime._workflow_closing and wake._disposed. If closing and disposed is not True, refuse ONLY the added notification as a separate post-release error. Otherwise release the lock and invoke SAME retry synchronously, with no await between checks/call. Native disposed is monotonic; no different host Task can interleave those synchronous observations. This does not prevent already-posted callbacks after a failed pre-call fence and makes no successful-shutdown/no-admission claim for that partial failure. Actual disposed=True keeps native retry inert. Explicit G.close in a still-live runtime remains distinct.

COST/validation: finite original receiver/source records and one existing availability request; pending work can remain after an unexpected notification error until another genuine trigger/recovery. Earlier fencing may expose existing presentation failure sooner; it must not skip required physical cleanup. Require actual queued delivery with exact two unrelated IDs/no G/no refill/known usage and original tree/ticket/B/E/child/creator return; wrong-token/replaced-source/unknown-metadata controls; an already-posted Handle after actual permanent fence while G.close is held; native fence plus buddy failure with original cleanup/finally; pre-call fence refusal preventing only new Q1 notification; exact notification failure and original cancellation precedence after positive release. Keep original10/5/20/60 bounds, all failed raw/pins and isolated retained-exit limitations. No new actor, authority, schema, generic wake registry, replay, timeout waiver or global idle. Direct launch own-return is a separate still-gated proposal.


Q1 literal compatibility clarification before behavior: preserve native retry_soon's unbound-loop policy. Capture wake._loop separately from the original G host loop; require its exact unchanged identity at release/notification, including a genuinely captured None. Original current Task/host loop still must be SAME G driver/coordinator loop. Do not require an unbound native wake to acquire a loop or turn its documented inert return into a new refusal. No replacement loop is adopted; actual scheduling/delivery remains separately proved by the native queued journey. Implementation uses only finite definition-time native wake source metadata, not a generic actor or scheduling change.

## FIX1 direct native return and cooperative wave custody — accepted source amendment

WHAT: preserve the existing bridge-prepared capture-disabled direct call. Its SAME original service invocation owns only its own launch operation, after original core and activation/guard scopes return normally. The genuine wave remains responsible for its additional item source, phase, execution, owned storage and actual to_thread return. Neither a direct receipt nor an owner callback proves an external caller's storage/phase/worker, required checkpoint, complete phase or creator return. Preserve no-bridge native callers, original duplicate receipt identity and ordinary fleet/continuation behavior.

Use one private service result computation with the existing public receipt/raising adapter, and one private native result computation with the existing ordinary/no-bridge tuple/raising adapter. Preserve ONE shared native launch/fallback/finally body. The original native failure producer constructs its actual receipt/error once; an eligible known no-child result crosses its required native cleanup, service core and guard NORMAL returns as a local value. The public adapter then raises SAME error. W carries that same original value through actual outer/worker normal return before its receiver finishes SAME actual locally reserved operation. Successful W results, eligible waits/refusals and known terminal results carry that invocation's operation; duplicate results carry None and cannot consume another invocation's mutable field. Actual source/cleanup/unwind failure, even SAME error object, stays unknown. Unknown receipts cannot be mistaken for positive settleable children while a retained launch operation prevents drain.

The original entry classifies once from unchanged real auth/native/bridge/caller source: N genuinely has no bridge, D has capture disabled and no wave, W has SAME originally enrolled wave and original driver/gather/child coroutine/worker relationships. No later wave=None, copied tuple, ID, caller flag or mutable current operation converts W to D. Retain the existing original gather and actual to_thread coroutine before await; worker entry may precede issuer.cr_await suspension. Original frame/closure/value and actual receiving return are required. CPython/private Task-group observations are source qualification for the actual runtime, not authority or universal eager/custom-factory support. Missing association retains/refuses; no extra Task, worker or registry.

Cooperative writers share each original authorization lock for short association transitions. The original None launch marker remains from selection through actual own/wider return, including no-owner waits. Reserve/refresh/finish/root/drain and source callbacks stay OUTSIDE auth/capacity/activity locks. Save operations when their original calls return, install them against SAME entry, and leave the exact field installed across original finish callbacks. Check original sources/owner/membership/associations before finish and again after actual return; then clear only that exact field. Physical removal followed by a callback/error is unresolved return, never restored capacity. A failed or missing reserve cannot donate an operation found later in a mutable owner collection.

Original waves pre-enroll their fixed distinct members under those same locks before driver/B/source effects. Existing later wave assignments become validations. Before create_task has even been attempted, a multi-member refusal may roll back ONLY exact locally claimed unused associations with actual no-effect facts; it never closes or discards configuration/allowance. After attempted task creation, missing driver/empty fields cannot prove no entry. Bind_native captures SAME original preparation/wave in its existing critical section, preserves ledger binding, invokes only the saved original wave outside the lock and rechecks association. No successor-wave reread. Preparation close refuses installed launch operations, unresolved markers/waits/start and foreign wave-only enrollment even if a drain callback has already set tracker.drain. Only original wave _open/run or SAME native settlement Task/gather may close its enrolled original member. Keep wave association through original unpublished-storage and closing scopes; actual closing selection is atomic under auth and failure retains it. Child close operation remains a separate original obligation. Wait unlink occurs only after original wave driver return and exact positive wait/source proof.

Private validator/freshness computations preserve original stock goal_storage_operation, public raising adapters, evaluation order, short circuits, all three validator and six freshness consumers, original partial=True None and service.call whitelist. A refusal is created once at its actual eligible predicate and crosses NORMAL original storage return. Positively returned non-None manifest inequality may qualify; None/unavailable/swallowed collector/source-close failures and nested exceptions do not. No repeated scan, decorator bypass, optional transport flag or generic exception classifier. Non-wait refusal never becomes pending/unlaunched/retry, new acceptance, refund or fabricated receipt. Required no-dispatch owner/activity/cleanup and service/wave scopes must all return; SAME-error unwind is still unknown.

Bridge-only start ambiguity explicitly refines the earlier known-terminal compatibility language: if Thread.start was invoked but did not return and genuine supported no-start proof is absent, strict D/W is UNKNOWN BEFORE the old known-no-child fleet/status/worktree/cancel/refund effects. Constructor/pre-call setup failures retain actual known no-start eligibility after required cleanup. N/no-bridge and ordinary fleet keep their original Thread.start/status/refund/error route, including the existing reserved-launch refund regression test. Actual started-worker admission refusal remains its existing separate path. Do not infer no-start from false started flag, error type, absent map entry or later dead Thread; no OS-exhaustion or cross-platform guarantee.

Retain one bounded repr-hidden passive `uncertain_start` tuple on SAME existing _PreparedWorkflowChild: actual constructed Thread, original target, original child_kwargs and source error. Save target/input before Thread.run may delete its attributes. Existing auth/preparation/receipt/owner/claim/op links remain the only custody; no duplicate ownership graph. Never serialize/log/display or dispatch from this holder, never put a possibly unstarted Thread into the positively launched threads map. Strict ambiguous Exception retains existing unknown receipt/error behavior; strict BaseException preserves uncertainty then redelivers SAME BaseException through the ONE original finally. A later target return cannot certify missing native/own/wider return. Preserve accepted-versus-reserved accounting and existing optional log error isolation; normal optional helper return is not universal writer retirement.

Definition-time finite function/code/default/closure and class/instance descriptor records cover original new producer/receiver/adapter, actual core wrapper/body, activation's plain delegator plus SAME guard wrapper/body, operation/owner methods and module aliases, validator/freshness/storage sources, preparation close/return, wave and bind receivers. Check the checker before invoking it, actual callback/issuer/frame associations, and originals before dispatch/consuming return/release. A substituted method/core/finisher cannot supply proof. Cooperative exclusion does not promise atomicity against arbitrary raw Python mutation ignoring native locks; detected change retains uncertainty, never fabricates restoration.

WHY: the actual direct dependency fixture retained SAME launch operation after real child return; mutable W finishing could instead consume an operation from another invocation. Raised error identity cannot prove required scopes returned, and an entered-but-unreturned Thread.start may already have started work. Existing normal-return local values and original locks preserve actual ownership without a second engine or receipt capability. A-only blanket retention of every genuine clean failure breaks supported cleanup; mutable wave=None release and generic acknowledgments cannot prove return. The selected single-tail/private-value protocol preserves real known cleanup while retaining uncertainty.

COST and validation: finite local result/source checks, all existing writer cooperation and one passive per-native-child uncertainty slot. Conservative ambiguity can retain reservation/capacity/claim until recovery; false positive return could release live work. Exact protected transfers cover only task_scope imports/bind_native/source records and specified native test observation hunks; the native preparation fixture remains read-only. Meaningful focused controls cover actual D result and independent caller return; N/ordinary reserved/committed/error compatibility; same-W duplicate; known versus invoked-start unknown; non-wait/None/same-error exits; real enrollment/no-owner/callback-close/bind/multi-member races; actual W B/E/wait retry and source/operation/caller replacement. Preserve original bounds, frozen pre/post/raw, old failures/kills and isolated retained-exit no-drain limits. No full suite, manual finish, global idle, source swap, guard reset, new owner/worker/registry/lock, schema or permission. Q1 remains separate; no all-five FIX1/WHOLE gate or parent completion is claimed.

This amendment adopts the controller-accepted exact literal source proposal SHA60787f3be34958bb5411aec4147d47510ffd6a2231b9bda2828c5ee2be9b25eb and full root267/67af/ffcd/13fc contracts, including the explicit seventh refinement. Full bodies and exact prospective bytes are retained in the task's source/evidence archive. Task38531 stays In Progress with acceptance criteria unchecked until the complete required review/gate.

### FIX1 capacity refusal: exact current body origin, separate scope returns

WHAT: at ONLY the immediate native-call catches INSIDE original activity.prelaunch and (for tool reservation) owner.activate, qualify the existing four CapacityRefused saturation reasons by the COMPLETE CURRENT traceback. Its head is SAME currently active original launch frame at that exact native admission CALL site, followed only by current begin_execution, or current reserve_tool then _reserve. All f_back edges, actual receivers, owner/capacity/arguments and aliases must match this invocation; terminal tb.tb_lasti is the definition-time original RAISE for child_capacity, automatic_child_capacity, tool_capacity or automatic_tool_capacity. No extra/old frames, searching for code names, stale chain, mutable reason or unwound frame.f_lasti. Runtime closure, finished execution and previous-tool-still-running remain excluded under existing policy. Check original class/instance descriptors, CapacityRefused alias/stock constructors, SAME original capacity lock AND exact stock primitive lock type. A foreign/proxy lock cannot substitute its __exit__. Owner operations are checked only against SAME actual returned local obligations and SAME set, never global idle or a reconstructed operation.

This identifies BODY origin only. Keep candidate as a local value through original owner/activity NORMAL exits, original undispatched cleanup, core and guard NORMAL exits, then W original source/storage/phase and actual worker NORMAL return. Never run this predicate after even one such scope unwound: contextlib can restore the old traceback when a generator exit raises SAME RuntimeError/BaseException. Any such exit/error stays unknown and cannot be reclassified at an outer catch. Actual finish/drain/claim/live/unfenced proof remains mandatory before wait publication/marker clear. No traceback claim certifies physical return.

WHY: _refresh_limits precedes native admission and can call observers; innermost code/reason alone accepts a re-raised old refusal. Current contiguous frame/call/RAISE association separates the actual native branch without changing the capacity API. Native availability observers deliberately isolate Exception; a swallowed callback exception followed by a real later saturation is not proof all callbacks succeeded. Escaping source/profile/cleanup failures remain separate. No general exception classifier or traceback-authenticity guarantee against arbitrary same-process traceback surgery.

COST/qualification: a finite original code/bytecode association specific to the actual qualified CPython path; unsupported/missing association refuses as unknown. Preserve actual declaration/source and opcode mapping; no universal Python/custom-executor guarantee. No capacity-source edit, new owner/operation/lock/registry, result API, retry or authority. Required native controls distinguish real child/tool saturation from a stale SAME-error rethrow and SAME-error owner/activity exits, changed constructor/source/lock and an inserted but unreturned reservation. All original returns/bounds/isolated-retention limits apply. This adopts fixed source4fde4a6ed4a0fe0605e596ab3b7462a52c473c6c9b9b316c3e82a7e2c677a63a with controller's additional exact CURRENT CALL-head requirement, after full read of the independent capacity source report. No native passing claim is made.

### FIX1 pre-enrolled wave refusals and actual captured-source return

WHAT: original wave pre-enrollment keeps classification/close exclusion before effects. Carry ONLY the four explicit original `_open` BODY refusals through its normal return to the SAME registered run driver: real live child capacity, unfenced fleet saturation, actual footprint conflict, and original `_open_wave` returning None. Preserve original retain policy (footprint conflict remains terminal even with retain_waits). No caught source/capture/context exception becomes a known refusal. Non-retained originals close through their exact driver-owned native P.close/read/producer/retired path; all original required release callouts must return before registry pop/unlink and SAME error redelivery. No post-create_task missing-field rollback, new dispatch/retry/refund or fabricated B/E.

Capture is not automatically unused: actual native capture can borrow an ordinary tree and reserve three operation tickets. Original nested capture therefore returns SAME local (source, footprint) through its existing finite reader, whose immutable original producer result, callback/issuer and positive producer/retired/owner/claim return are verified. Pin original issuer/capture/constructor and actual source/receiver. At the no-fence branch ONLY, actual unborrowed tree/owner/claim None plus original stock holds lock, empty holds/unclosed source, unchanged actual source capture and no coordinator registration/effects qualifies. Foreign/borrowed/unknown source stays retained with its original tree/tickets/wave/P; never close another tree or synthesize physical ticket return. Branches1–3 genuinely have source=None. The original wave remains current through unused-storage close and every release callback; callback failure retains honest uncertainty even if an earlier resource was physically removed.

WHY: required pre-enrollment moved ahead of an old early-raise path, and an absent fence or window cannot prove the capture operation never acquired borrowed custody. Explicit local branch/error/result and actual original read return preserve the distinction without treating IDs/empty fields as authority.

COST: one finite private open-result carrier and definition-time source/receiver checks within existing Tasks/readers/locks, conservative retention when source or close return is unknown. No source actor, global registry/lock, worker, capacity API or schema change. Direct controls must cover normal unborrowed busy/known unused close, original borrowed holds retained, original capture/exit error and unchanged conflict policy. Governs accepted source4e31ddca857f173f7fb6a207394a92570ebb5ff069cc8a9b48f43ae47e978fbf; no passing test claim.

### FIX1 caller cancellation after a positively returned retained wait

WHAT: `run_workflow_wave` currently obtains SAME original driver.result and checks original source, then redelivers caller cancellation before unlinking retained waits. Move ONLY cancellation redelivery after the existing exact retained-wait unlink. A positive closed/no-window wave must not leave its SAME unused P attached to the already-removed wave. Require SAME actual driver positive noncancelled result, current source, original closed wave/P association and no launch operation/native marker/dispatch/negative wait; only that existing positive branch may unlink. If driver/source/worker return is unknown or raises, it never reaches unlink and retains original custody. Preserve original cancellation/error precedence and never cancel/detach the original worker or infer return from cancellation.

WHY: the registered original wave protects pre-enrollment, but its real positive closed wait must hand the unused preparation back before the original caller observes cancellation. Otherwise later native Stop/unused-close sees an obsolete wave and refuses. This is a source-confirmed ordering gap, not an executed prior failure or waiver of physical proof.

COST: one local ordering correction, no await between positive unlink and original cancellation redelivery, no new Task/lock/registry/authority or retry. Direct control cancels ONLY the exact original waiting caller at SAME original `_release` NORMAL RETURN after native closed/pop, then proves actual driver positive return and P unlink before native unused close/read/creator return; unknown cases remain separate. Original bounds and all earlier raws preserved.

### FIX1 first actual caller cancellation and ordering chronology

WHAT: in the original run_workflow_wave shield/drain loop, retain the FIRST actual caught CancelledError object. Repeated caller cancellation keeps that original object and message. Original driver.result and current-source checks still precede any positive retained-wait unlink; only afterward raise SAME saved cancellation. Failed/cancelled/unknown original driver or source errors keep their original precedence and never gain unlink/retirement from caller cancellation. No await separates final positive unlink and redelivery.

WHY: HEAD and the first ordering correction stored a Boolean and raised a fresh exception class; the earlier 1P3.71s control proved ordering, not caught/propagated identity. The ordering correction had four-file readback but OMITTED the separately fixed controller source/Ruling gate: writer interpreted root's “not a new gate” direction as authorization, while it meant no additional root/user contract approval. Preserve that explicit deviation and all earlier bytes/raw; do not retroactively claim controller approval before that edit. The subsequent fixed identity proposal a6121d8551b1c9fc6d618a333f855a6bf6efec80b8b72e3f764871f0a58a8f03 received controller Ruling before this identity correction.

COST: one original local exception reference, no new Task/lock/registry/authority/retry/uncancel policy. Strengthen the genuine positive no-window wait with SAME actual caught/propagated exception and message. Separately hold original non-retain unused-close finite callback normal return, deliver two actual cancellations, prove the first stays saved, then require SAME actual original body/driver error ahead of cancellation after original physical close/producer/retired/creator return. This separate control does not prove repeated-cancellation positive wait. Original5s hold/60s process bounds, source/owner/claim checks and unknown retention remain.


### FIX1 original direct storage admission and explicit freshness recovery

WHAT: retain one passive, repr-hidden original preparation observation `(P, source, sources)` only after the actual original `_AgentSources` constructor and `prepare()` return normally from the original P `_configure`. Pin the original class/constructor/prepare/receiver, source methods, module aliases, original guard/captured-sources/execution source before invocation and after return. Retain the exact built-in source tuple and immutable row/owner/path references; store the SAME observation object in the existing frozen native child record through the checked original issuer. N has no observation. A copied or later equal tuple, replaced source/method, changed native/auth/phase/list or missing original receipt refuses; the observation grants no storage permission and is never reconstructed from an ID.

For an actual entry-classified bridge direct launch (D), after original claim/marker checks, supply ONLY that original tuple to the existing `execution(self, sources=...)` on the actual launching worker BEFORE its own `_core_operation(self.db)`. Keep the ONE original native cleanup/finally/close/refund/root sequence and final log in place. Require the new service-owned guard scope's NORMAL EXIT, original core return and exact source recheck before the original invocation's local launch operation is finished or known outcome delivered. Admission, callbacks and storage remain outside auth/capacity locks; only short identity writes/checks use the existing lock. N/no-bridge, W's existing outer source scope and ordinary routes keep their current behavior. Do not add full P.activate_source, preparation ContextVars, scratch leases, handle stacks, source discovery, transfer/resolve overrides, guard reset or outer-core widening. A genuine caller-owned narrow core/foreign context remains subject to native refusal; caller outer return remains independent. Any admission/exit/cleanup/source/callback failure—even SAME body error—retains original custody, never a refund/retry/positive return shortcut.

WHY: actual direct known-constructor cleanup encountered native storage admission while its own narrow runs-DB operation was active. Source supports admitting the original complete P observations before that operation, as W already does through its wider scope. The actual diagnostic identifies operation_path_outside_scope at a member equal to current source tuple[2], but does not identify which of four predicates failed or a particular DB owner. Preserve both original constructor1F1E and diagnostic1F1E; no historical cause/exoneration claim.

COST: one passive existing native-record field, finite source/identity checks, and additional service-owned original guard leases whose full return must now be proved. Forged observations could widen sources; early finish could abandon leases; unsupported external nested scopes may still refuse. Existing native path/lease/permission/current-binding gates remain decisive. No source-class/activation/Backup_Recovery edits, registry, second engine or authority.

WHAT: reuse the existing frozen `_TaskInputResult` in three finite local cores for read-scope, dependency-scope and task-freshness derivation. Keep the original public helpers as unchanged-signature raising adapters; preparation and prepare_decision remain their consumers, while the original validator consumes actual private values through its ORIGINAL stock storage wrapper. Preserve original input/dependency/DB/spec/shell order and short-circuiting, one resolution/collection and SAME branch-created error. Only a completed original explicit local predicate may return a conclusive refusal. Never catch or relabel DB/pathlib/resolver/collector/source/context exceptions, unavailable manifest/None or failed scope exits. Extend finite original helper/result/alias/receiver records before invocation and after actual normal return. Alias/shell known refusals must still complete original unused cleanup and creator return; a same-object wrapper/core/guard/W exit error remains unknown.

WHY: the original alias and shell predicates threw across the inner storage-wrapper boundary, so their errors could not prove normal return. Returning an actual local predicate value preserves refusal and permits independent proof of every original cleanup/scope return. Earlier affected3P3F3E and diagnostic3F3E remain separate evidence, not passing controls.

WHAT: explicitly choose protected-path alternative B. Keep native Tools.resolve_workspace_path, security choke point, fs/git/workspace/agent callers and generated remote source unchanged. A real exclusion/protection change after preparation that raises in that external resolver has no positive inner-helper/storage return. Strict bridge D/W therefore retains SAME marker/source/claim/owner/reservation/error and refuses close/continuation/quality pending recovery. Preserve no new provider/run row/committed child/refill and all actual reserved/accepted charges. This intentionally differs from the former automatic unused-source/creator-drain expectation; it is not clean cancellation or legacy drained parity. N/ordinary/shared Tools permission behavior is not globally rewritten. Reject a new path traceback classifier, error whitelist, duplicate validation or larger shared resolver extraction for this task.

COST: local value seams add finite wrappers/source checks; conservative protected-path recovery can retain a reserved attempt/capacity/claim despite no new provider effect. This user-visible recovery requirement must remain explicit. Convert only the original excluded scenario to a separate guarded private retained negative, using the SAME original native/dependency fixture generators and actual exclusion mutation/no-effect/accounting/custody assertions. Unique marker plus explicit exit17 only after all assertions; no creator/drain claim, exclusion reset or forced cleanup. Other four mutation parameters and all existing diagnostic/user bytes remain unchanged. On unexpected earlier failure attempt only the original generator cleanup and preserve primary/cleanup errors.

Validation: finite native constructor D positive; normal/held D, original N failed-start refund and genuine W; original observation equal-copy and source replacement negatives before replacement execution, SAME error from new own-guard exit and genuine external-narrow-core refusal. Original alias/shell positives, public preparation/decision adapters, copied/replaced helper and same-error unwind negatives; actual protected-path isolated retention/charges/repeated-close. No full suite, historical BASE execution or blanket factory/performance claim. Exact pre-edit path/index/c5 and outside-hunk proof, frozen actual source/test maps, original process limits and raw failures remain mandatory.

Governance: FULL root-direct-original-storage-admission-ruling (6971ebf7…923c6) and FULL root-freshness-refusal-boundary-ruling (4d297377…66063), fixed literal source ed5376455217fe6c2d92083c98324286285b3927b9dc084f3f0658c7261272d3 and controller source/hunk Ruling precede this amendment. Both FULL root bodies were archived byte-exact beside all prior seven full contracts and both full reviews. This amendment/readback must precede product/oracle edits; it does not cure or rewrite the earlier cancellation ordering-gate omission, certify unknown return or declare Task8/FIX1/WHOLE complete.


### FIX1 original execution-close storage receiver before effects

WHAT: before original execution P.close selects state/worker/source/lifetime effects, and again in its actual finite callback and original native-owner cleanup receiver, perform a pure original method/issued observation/receiver association check. Extend only existing finite storage-source checker: definition-time P/native/source class aliases and descriptors/functions remain exact; SAME configured observation binds actual P/source/sources tuple and original frozen native observation where present. Short original auth lock protects association data reads only; no callback/storage/admission under it. Genuine initial source-less cleanup is eligible only while source/observation/native frame/native source record are actually absent. Planning/direct F and N/ordinary routes remain unchanged; W keeps its original close association and positive physical return requirements. Missing/changed partial source stays retained, never restored or adopted.

WHY: actual current storage negatives2F1P10.59s showed launch pre-effect source refusal, then close later invoked a replaced check_memory method before its post-return source proof. Equal copied source tuple also reached actual _closingTrue before rejection. Those actual failures and unsupported earlier False oracle remain historical receipts; no rollback or retrospective success. FULL6971 original method-before-invocation requires closing receivers to check their own source before using it, not merely inherit earlier launch validation. Controller accepted fixed source362efc63062820fd51af646458723c32ad10628aa763d2d38b4430073fbf526f before this amendment/behavior.

COST: a finite pure check at existing receiver boundaries and conservative retention on mismatch. This does NOT add full require_current/check_memory/freshness/policy admission before cleanup; I2 already-owned physical cleanup after budget exhaustion/revocation must retain original behavior. Existing source.activate/check_memory/guard and accounting gates remain in their original locations, and all post-source/producer/retired/operation/claim proofs remain mandatory. No new actor/lock/registry/capability/schema/refund/force-close or reset. Validate exact copy/method negatives without replacement execution, original D/W positive cleanup, real exhausted/revoked cleanup, and genuine source-less initial cleanup; preserve all previous failed/killed and isolated-no-creator evidence.


### FIX1: original live native binding receiver before invocation (2026-11-04)

WHAT: At the existing non-None workflow authorization branch of the original AgentService._run_one, check the exact definition-time workflow task-scope module/class and its existing bind method/helper/checker records before selecting executable evidence. Bootstrap the checker with direct original function/code/default/keyword-default/closure identity checks, require static original receiver/class descriptors, validate original helper/method, then invoke SAME saved unbound bind_native on SAME actual authorization/owner/activity claim outside native locks. Recheck after original helper return; never perform another mutable method lookup at invocation. Preserve original bind_native internal source/continuation/owner/claim/committed-ledger/wave checks and all shared native child/finally/fallback/error order. Bind AutomaticWorkRefused before the new earlier use because _run_one already imports it later. Original absent-authorization ordinary path remains unchanged; no-bridge N does not acquire any new P/wave/marker requirement.

WHY: Independent pinned source analysis9a318 and sole-writer adoption06b587 identify the single product dynamic bind_native call BEFORE its receiving method's own source check. A changed descriptor can otherwise execute before rejection. This is a concrete source ordering gap, not an already-executed regression or an all-caller atomicity claim. Controller accepted exact bounded source/hunk06b587 before this amendment/behavior. The eleven existing direct task-scope test calls keep the public original method; they do not gain service-caller coverage by relabeling.

COST AND LIMITS: finite original aliases/checks at the existing late bind boundary, no new authority/registry/lock/worker/return type or schema. Child reservation is already committed, actual row and owner.run_id exist, and parent receipt may have returned; refusal is NOT a no-dispatch capacity wait, refund or retry. Original already-owned cleanup must still run, with actual errors/uncertainty retained and no manually finished operation. This bounded repair assumes the actual original _run_one receiving path: existing named records do not pin its entire enclosing wrapper/body, and arbitrary simultaneous mutation of the SAME function's code/defaults or coordinated records is not globally serialized. A saved original callable prevents a changed descriptor from redirecting that call; it supplies no execution/retirement authority and is not universal source immutability.

Validation: exact actual committed live before-lookup and post-helper descriptor controls require separately reviewed source/harness that distinguishes real physical return from retained uncertainty; no claim before execution. Qualify current genuine D/W/N and unchanged public binding routes with the smallest source-relevant named nodes, preserving prior receipts as byte-scoped and all failed/killed/retained-exit limits. No broad suite, source swap, guard reset or other runtime boundary. FOUR-file byte-exact append/readback receipt task-8g-fix1-live-bind-readback.json precedes product edit. Task38531 remains In Progress.


### FIX2: original finite callback values through every owning return (2026-11-04)

ADR required: yes; existing ADR285 governs the same runtime/source/custody boundary. This is an amendment to that contract, not a new authority or schema. Task38531 stays In Progress and both Important FIX1 review findings remain open until current bounded controls and independent scoped review return.

WHAT: Controller accepted the complete fixed source proposal `task-8g-fix2-source.md` SHA256 1d361ff6ea0434abbc09c4a1582593943047abd10edee72727c01bf080948d4a, including the notifier-context appendix, after FULL root source review140bee71744bd92c78c5c517ffe615a73ea5002b316af35d8ea28f6027fbc562. Product changes are restricted to original console_workflow_preparation finite-read/return/close receivers and their existing source records. Existing native-return test hunks and precise private-probe routes qualify both findings; native reader, capacity, activity, source/activation/storage implementations and fixtures remain read-only. Original c6389 HEAD, staged c5/61 and279 foreign worktree bytes are preserved. Thirteen complete root support bodies are byte-archived and indexed in `task-8g-fix2-full-supports.json`, extending the twelve unchanged FIX1 bodies with FULL140bee. The fixed source and exact original records govern implementation; a shorthand reference never replaces those bodies.

1. Append the existing original per-read record before fallible issue. In execution, catch the actual synchronous callback only INSIDE all its original owner/source/activity/phase scopes and carry its SAME error as a local body-error value. For execution AND planning awaitables, catch only actual `await pending` inside the existing async owned-storage/source scopes. The SAME existing async worker returns a local immutable pending/result/error tuple through actual inner contexts and original async-worker guard; consume it only after the SAME Future.result returns normally. Retain the actual pending/coroutine/Future on the original passive record for source/return diagnostics. Submission, guard/context exit, Future.result or tuple/source failure remains unknown, even when it raises the SAME callback error object. No traceback/type/identity classifier or new worker/Task may certify an owning scope.

2. Execution crosses original owner/source/activity/phase AND outer execution/owned-storage/unpublished-storage exits normally before its single original finite finish/root/claim tail. Remove exceptional finish retry and premature _finite clearing. Invoke SAME actual operation.finish once, observe SAME owner positive drain after original root return, then original claim settlement; clear finite custody and emit unchanged immutable ten-item tuple only after this actual tail returns. A finish/callback BaseException after physical removal retains actual uncertainty and original references, never retries to obtain an idempotent return and never claims capacity membership was restored. Actual genuine no-owner prelaunch busy branches carry their original error through normal exits with no invented owner/operation/callback. Missing/ambiguous admission return is not this branch.

3. Add a pure execution prior-read physical-return check before subsequent finite reads can overwrite custody and before execution close first source/lifetime/mutation. Require actual original producer/retired/tuple/source/owner/claim return for each issued read and finite=None. Positive callback-error or caller-cancelled reads with proven producer return are distinguished from unknown; no blanket historical-error block. Initial source-less cleanup retains its actual no-prior-read/native-source limits. Closing-body errors do not authorize configuration continuation, and failed closing scope/producer prevents a later close from clearing original source/configuration. This check adds no policy, wall-budget, control revision, CAS, DB admission or dispatch authority. I2 already-owned physical cancellation/reap/close after exhaustion/revocation remains permitted by its existing custody even if separate accounting failed; actual unproven physical resources remain retained.

4. The separate known-before-issue alternative is execution ONLY. Preserve planning's existing conservative early-issue/close behavior with no new planning-read exception. Bind SAME original run_preparation_read coroutine before awaiting, its original frame/code, issuing Task/thread, exact native arguments/local checker, saved original validator, P/runtime/controller/service/source/configuration-scope/set. Definition-time pure source/helper/static descriptor/function checks precede selecting and calling the saved original validator; closing retains its original pinned no-op. Only its first actual original host validator failure, before native read/submit/producer construction and with exact current frame/caller association, can record a passive no-issue tuple. Cleanup requires SAME coroutine actually CLOSED/cr_frameNone/not running, retained original frame with no native read/submission/producer locals, SAME saved wrapper error and unchanged original bindings/source. No validator re-execution in cleanup, no absent-proof shortcut and no fabricated positive result. Later validator/constructor/submission/worker failure or unsupported/drifting association remains unknown. This CPython-qualified frame/coroutine proof permits only the original physical cleanup continuation and proves no arbitrary callback-private return or execution authority.

5. Unknown preparation retains P/configuration in the ORIGINAL runtime._workflow_preparations registration created by _retain_configuration BEFORE preparation reads. G's run.preparations is populated only after preparation returns; never claim earlier coordinator association or fabricate owner/registry custody. Direct pre-coordinator unknown remains a retained recovery cost; bare runtime disposal is not newly certified to drain it. Actual local owner/claim return does not certify the later common tracked-reader handoff, binding reset, source/host checks or producer publication. If those later checks fail after local release, retain P/configuration and report actual earlier absence honestly rather than saying the owner remains admitted.

6. Preserve original error and cancellation ordering. Execution keeps SAME original wrapper error/caller cancellation if the native reader raises, separately retaining producer/body outcomes; on actual positive return it proves the tuple before body-error redelivery. Planning keeps original proof-before-wrapper/body ordering. A callback CancelledError is a body value, distinct from issuing-task cancellation. Synchronous callback StopIteration retains native public RuntimeError('coroutine raised StopIteration') with SAME original StopIteration as cause; async coroutine conversion stays native. Do not change common reader repeated-cancellation semantics, no-refill/accounting, F/no-bridge public behavior or unknown producer precedence.

NOTIFIER CONTEXT WHAT/WHY/COST: OwnedOperation.finish mutates under its original capacity lock and invokes drain/availability callbacks outside it; they isolate Exception but not BaseException. Bookkeeping needs no active source, but callbacks can copy ContextVars. Moving finite finish after original normal scope return intentionally changes its synchronous notification context to SAME worker's RESTORED AMBIENT Context, not necessarily empty. No fresh Context, guard reset, copied retired ExecutionState or new activation is introduced. Existing root/claim drain/activity callbacks already occur outside the original P contexts; preserve native notification/settlement ordering and actual errors. G's actual released callback only posts SAME run.wake.set through its captured loop, whose native Handle copies that restored ambient Context. Existing waiting Tasks resume through their own registered continuation Context, not Event setter authority. The other production capacity subscriber, change-review _availability, only advances a release version/queues an existing retry reason; its original worker later admits/activates its retained source. Both already accept native limit-change notifications outside owner activation. Custom external subscribers which depended on temporary P/phase/owner/guard bindings will no longer receive those temporary bindings at this finite notification: this is an explicit compatibility cost, not a context-free refactor or promise of arbitrary subscriber compatibility. Native callbacks still provide availability only. Re-entering activate_source would acquire fresh guards/DB/scratch scopes and fresh failure exits, potentially blocking already-owned cleanup; that alternative is rejected. Notification return is not proof every isolated observer succeeded, physical retirement or new admission.

WHY: Current independent review813924bda92e6477885f520380f6cc03313b8dcd5a748499276107e113d53933 reports0Critical/2Important, SPEC FAIL/QUALITY REQUEST CHANGES. Its genuine isolated same-error probe returned owner_admittedFalse/claim_liveFalse while the SAME original storage lease remained live. Identity equality did not prove exit. Source also shows planning Future.result catches outside the async worker's owning scopes and execution auto-close before G registration. The callback test independently failed1F1E before its requested callback boundary. These actual failures, all earlier failed/killed/setup raws and both recorded governance deviations remain unchanged; no retrospective exoneration or all-green claim.

SECOND FINDING WHAT/WHY/COST: Correct only the existing successful-child callback test's source-proven setup overlap. One shared setup deadline up to30s inside the unchanged private-process total60 covers original phase/launch readiness and SAME child's actual target return/join; no restart/extra allowance. Critical own-result and callback holds remain5s. Use the existing finite code-entry profile idiom for every actual handler and capture the SAME child target before native Thread.start. Require actual successful child target/Thread return before parent finisher removes its last operation, original post-removal callback with P exclusion still installed, inert duplicate, W/unrelated-close refusal, SAME callback and finisher normal returns, original checkpoint and fixture return. Passive observers record errors without throwing into native callbacks. Finally releases only original test holds, awaits SAME launch, attempts original child cleanup join before fixture scratch teardown, restores original profiles and preserves primary/cleanup failures separately. Later finally return cannot rescue a failed setup assertion. No skip/xfail, assertion deletion, fabricated child/finish, equivalent automatic startup repeat or historical timing/cause/performance inference.

VALIDATION/DELIVERY: Run finite current-source positive synchronous/async callback errors and special cancellation/StopIteration controls; exact inner owner/source/activity/async-guard and outer execution/storage SAME-error negatives; original no-owner wait, source-less cleanup, execution first-validator versus entered/failed-read distinctions, unknown repeated-close/no-second-finish; actual restored-ambient notification/drain and original issuing-Task context observations. Source-relevant unchanged capacity callback/notification and change-review real-slot retry tests qualify native consumers without replaying completed I1/I3/I5 gates. Retain actual I2 exhausted/revoked cleanup and direct F/W controls. One changed successful-child callback node uses setup30/hold5/total60. Preserve actual PID/PGID/command/wait and frozen maps; isolated17 negatives are not drain/creator successes. Native_host proves fixture/runtime/gateway/DB return; real WorkflowRig proves unmounted App-constructor retirement, not reopen/restart. Actual restart evidence stays separate. No fullsuite, global idle, registry, schema, fake proof, source swap, reset, manual finish, retry/refund or new actor. Both review findings require actual results and independent scoped review before closure.


## Task 8G FIX2 amendment — original terminal configuration return and inert repeated close

**WHAT.** Adopt only the exact reviewed source `task-8g-fix2-repeat-close-v2-source.md` (SHA256 `7aafe9d051163167bb3a833fee7bfd69003ebf128e83ab23a4df50d8a466ad4b`). The existing `_release_configuration` returns a passive `(SAME preparation, SAME original scope, SAME original runtime registry)` tuple only after its actual original scope `__exit__` and registry discard both return normally. Existing planning and nonterminal consent callers ignore that private return value. The existing direct execution close and the existing W release call site receive the SAME actual original result and bind it to the actual positive last finite closing record and exact finite-read sequence. These are the two original result receivers; no generic helper/registry or new cleanup pass is added.

An execution-only historical first-validator no-issue record may pass its existing gate after terminal release only with its original non-None scope, same original current registry, exact received terminal carrier, unchanged finite-record sequence and SAME positive last original closing read. That closing read must re-prove original producer/retired/immutable result and original `close` nested `finish` callback/receiver through the original invoke closure, with no body or wrapper error. All original terminal CPython coroutine/frame/no-entry/validator/source checks stay mandatory. Ordinary live-scope recognition is unchanged. `_closed`, absent scope, absent registry membership, an allocated/copied tuple, or a positive unrelated read alone never supplies return proof. Every other finite read and current source remains independently checked before the existing already-closed no-op.

The W receiving hunk records only the individual preparation's actual original configuration return; later W callouts/source checks and original wave closed/pop ordering remain mandatory for whole-wave return. A later failure never restores physically removed capacity or claims W returned. Missing/failed original scope or release-method return yields no terminal carrier and cannot use this alternative. No retry of scope exit, finite operation finish, owner return, or prior callback is authorized.

**WHY.** Approved test-only source38d012 added a second SAME-P close after an actual original first-validator refusal and proven finite terminal close. The real RED was1F/3.70s, wrapper6.269988, PID=PGID29797 wait1,8317 equal source pins: first close's producer/retired/scope/accounting assertions passed, while second close refused the historical record because successful cleanup deliberately removed the live scope registration. Keeping the original live predicate alone breaks idempotence; bypassing on `_closed` would instead launder unproven cleanup. The positive original terminal result plus the actual closing producer supplies the bounded distinction.

**Limits and compatibility cost.** One private per-preparation passive tuple retains existing source/read identities; the private release method returns a three-item result that its legacy planning/consent callers ignore. Two existing receivers perform finite identity checks. There is no actor, lock, owner, registry, admission, schema, refund, source replacement, callback suppression, or new execution authority. No configuration scope is reacquired to satisfy proof, and actual disposed-source guards are unchanged. The original consent release/reacquire lifecycle and runtime-set behavior remain unchanged. FULL consent source report656e confirms G's run→phase→authorization→P strong root after original `_enter` and separate settlement guards; set absence is not evidence of lost G custody. Configured consent `scope=None` first-validator behavior remains a separate conservative source question and cannot borrow this terminal non-None-scope exception.

**Verification and chronology.** Preserve the actual RED and all older failed/killed receipts. After this exact four-file append/readback, implement only the approved preparation hunks and clean W receiving hunk. Run the same strengthened first-validator test GREEN, the source-relevant original unused-W positive, and an exact original configuration-scope-exit retained negative after its separate fixed test-source ruling. Keep old positives scoped to their actual maps; no duplicate completed I1/I3/I5 gates or full suite. Native fixture return, unmounted App constructor retirement, and actual restart remain distinct. All prior FIX2 no-issue/StopIteration/caller-cancellation/body-versus-wrapper/notifier context costs and the thirteen FULL support bodies remain in force. No retrospective approval or failed-evidence rewrite.


## Task 8G FIX3 amendment — execution configuration return and genuine consent no-issue cleanup

**WHAT.** Apply the controller-approved complete source `task-8g-fix3-source.md` frozen at b7e80fb9ba581205539e0ddf49318a8f56ec1397db7d4def02a719c11fd87402, with its pre-behavior original-W-scope audit. This is an execution-only correction to the two actual independent FIX2 review findings. Every execution already-closed no-op must prove the SAME original terminal configuration return, actual last closing read/producer/retired result and exact original source/receiver/record sequence, regardless of whether any prior read used the no-issue alternative. A flag, empty scope/set/capacity, closed generator, copied result or unrelated successful read never substitutes for actual return. A failed original scope exit/release receiving return retains uncertainty and cannot be rescued by another close, scope exit, finite finish or callback.

Record the exact original configuration entry/exit invocation and scope before its fallible effects, and consume the actual normal returned release tuple only at the original direct close, consent or W receiver. Keep the original field-clear, original scope exit and set-discard order. The actual original tuple binds P/scope/runtime registry; original method/descriptor/code/default/closure and native scope-producer/context-manager checks precede invocation/consumption. Unknown transition/error stays retained, without new admission or repeated validation intended to manufacture success. Add only finite passive P/read/W references and pure identity/source checks; no new class of authority, owner, actor, thread, Task, lock, registry, schema or durable capability.

**Unknown INITIAL entry custody.** The original G preparation assignment happens only after prepare_workflow_reply returns, and initial acquisition precedes bind_catalog/auth/P association. P-to-scope/error/self or an error traceback is not an external lifetime root. For an actually invoked original native configuration producer/enter/install sequence which fails to return, retain SAME pending invocation/scope/error and add SAME P to the SAME already-existing exact builtin runtime._workflow_preparations set before re-raising SAME error. The original runtime still must own that saved set. This is exception-only membership, not fabricated configuration entry or uniform consent retention. Later close/read refuses before new source/lifetime/read/release/finish work; existing same-phase settlement already refuses that set member. A source failure before invocation is distinct. No replacement registry is adopted. Native publishing refusal and entered-but-unreturned scope are not classified as never-acquired from exception type or field absence. The retained availability/recovery cost is intentional; bare runtime disposal is not thereby certified to drain the unknown execution P.

**Genuine consent.** Original execution consent intentionally releases its acquired configuration scope before requesting confirmation. Receive that actual normal nonterminal release at the original `_finish_native_preparation` call site and bind it to SAME phase/auth/P/source/native/lifetime/snapshot, original finite-read prefix, callback and issuing Task. G's existing phase→task authorization→P root and separate settlement guard remain intact; absence from the configuration set is not loss of that G root. Do not keep configuration held across consent or add uniform set membership. An actual scopeNone consent first-validator failure may use ONLY that exact original received release plus all existing CPython first-call/frame/coroutine-closed/no-reader/no-submit/no-producer/validator/source/error checks. None by itself is never proof. Preserve the original source/error/cancellation and no-refill behavior.

Consent decline, positive callback error or exact first-validator refusal still invokes the original same-P unused close. Require its actual new positive closing read and consume the already-proven original consent release; do not call an absent scope again or synthesize a return from None. Normal proceed reacquires its actual distinct original scope at the unchanged location, invalidating the temporary consent proof before new entry; later terminal return uses that second scope. A failed consent issue cannot reach original reacquisition because `_finish_native_preparation` has no catch/retry around that awaited read. Proceed's positive record has no before_issue stamp. Do not invent historical no-issue migration or a failed-issue-then-proceed path.

**Initial source-less and F limits.** Original execution construction immediately enters `_finish_native_preparation`, whose first statement retains configuration before any read. The genuine source-less busy control has an actual acquired configuration scope and one positive no-owner read; preserve its cleanup and idempotence. No repository-native execution factory produces a successful never-acquired empty P close/repeat lifecycle. Do not manufacture a None-scope receipt or an external-private-constructor exemption. The existing empty-list read gate remains; this qualification concerns the actual terminal receiver. Planning has distinct genuine pre-retain token/intake refusals; F's established early-issue, best-effort close and retained-unknown policy remains unchanged. This execution repair does not generalize into F.

**W return separation.** Original member close is an intermediate physical return while W still owns later E/configuration/all-callout obligations. W's existing `_release` returns a passive actual disposition only after all original callouts/source checks/closed/pop complete; receive it at all five exact original `_open`/`run` call sites. A later P no-op needs its own terminal receipt and this actual W release disposition. All five calls are outside owning With/AsyncWith/Try scopes in the unchanged original driver: registration/publication execution-storage scopes and native workers/child/member closes have already returned. A failure of an earlier scope, even the SAME error, cannot reach the receiver. The release tuple proves only those actual callouts. Driver.done supplies liveness only; driver exception/type/identity never supplies scope-return proof. Intentional post-release refusals keep their original error. A failed release-method return before the receiver stores its value stays unknown even after physical closed/pop. A later driver/public-caller failure is not whole-W success. Preserve existing stricter `_workflow_wave_return`, G settlement/release, cancellation, B/E, no-refund and external-caller return rules; no generic driver-success workaround or error adapter.

**WHY.** Independent review8a917921e1b6162cd452c075b7acd299d077961b50fed14a6bdea4a66abad91b reported0Critical/2Important, SPEC FAIL/QUALITY REQUEST CHANGES. Its actual normal-P terminal probe observed the first exact original scope-exit error, P still registered/no terminal receipt, then false successful second close because normal prior reads bypassed the old no-issue-specific receipt gate. Its actual consent probe observed a genuinely returned nonterminal scope, first-validator cancellation before consent callback/reader construction, and cleanup masking the original error because scopeNone was rejected without considering the actual release. Preserve both probes/raws and their setup-only/PID/map limitations, all earlier failed/killed receipts, false test predictions and governance deviations. No retrospective exoneration.

**COST / compatibility / verification.** Missing original return conservatively retains availability and source objects; exception-only existing-set retention is a deliberate new failure disposition. Finite passive original identities/current-source checks may refuse changed/custom source. No fresh policy/admission/DB/CAS check, guard relaxation, retry, refill, refund or force-close is added. I2's already-owned physical cleanup after genuine fixed-policy expiry/revocation remains permitted, with separate accounting errors and unproven resources retained. FIX2 SAME-error inner/outer/body transport, StopIteration/caller-cancellation distinctions and restored-worker-ambient notifier compatibility remain unchanged. All thirteen FULL support bodies under manifestcf36575b1b611e2aeecc113764df2313b6c095b79a0240da56bdb66d7f36ca20 remain binding.

Use meaningful native review-reproduction RED/GREEN, real normal/idempotent/source-less and prior no-issue compatibility; original initial-entry unknown retained control; actual consent proceed/refuse/body-error/exit-or-issued-unknown; exact W release-receiver failure and positive intentional refusal; focused F/I2/current affected/static/protection checks. Keep original private60 bounds and each existing narrower bound; no equivalent automatic rerun/fullsuite. Require actual original sources/physical producer/retired/owner/claim/callback/creator facts, not empty capacity or return flags. Isolated child17 retained cases are not drain successes. Native fixture return, unmounted App-constructor retirement and actual restart remain distinct. Exact eight owned paths/hunks only, original staged c5/61 and279 foreign bytes preserved. Append/read back this identical block in ADR285/spec/plan/Task38531 before behavior, then fresh independent scoped review; Task remains In Progress with parent AC unchanged.


<!-- WHOLE-FIX1-GOVERNANCE-BEGIN -->
## Task 8 WHOLE fix1 — original planning release wakes queued ordinary results

**WHAT.** Extend release-driven availability only to the positively returned original F planning invocation. Capture its original direct-or-G association, Task/frame, phase/source, runtime/controller/store/session, native wake/host and wake loops/locks, received claim/token and original configuration scope/set before native work. Retain the existing original provider/lifeline/tool/storage/read/producer/retired/control/root/claim return sequence. Before release, verify finite original method/descriptor/code/default/closure records for its read, scope and slot receivers and current exact original associations. Consume the actual native configuration return and require actual received/token releases to return True, clearing only the resource which returned. Partial errors remain honestly partial; no complete receipt or new notification follows an unknown return. Unused/early F close, common token release, ordinary wake scheduling/admission and execution Q1 remain unchanged.

A passive original F return tuple is evidence only at its exact original receiving invocation, never authority from an ID, flag, copied tuple, completed row, absent registry, finished Task or future exception. Direct F requests one existing native wake.retry_soon only after its own complete release, and keeps its lazy coordinator absent. G-associated F never converts to direct on a missing association: original G consumes SAME F return or SAME saved original F cancellation, preserves its existing control/metadata proof and requires actual same-run pop before one notification. Native pending/coalescing/priority/admission/authorization/context/ledger/owner/physical pipeline remains responsible for delivery and charging. No new Task, worker, timer, actor, registry, durable capability, refill or authority.

**Error/fence contract.** Revalidate original notification receiver/helper/source/loop/lock after actual release without replaying live source.check_memory, policy, admission or DB work. Never require successor slots empty or release/adopt them. Under the original wake lock, runtime closing with native wake not actually disposed refuses the new notification; actual native disposed/no-loop/closed-loop behavior stays inert, including retry_soon's documented RuntimeError handling. No Runtime.dispose reorder is added. Unexpected post-release validation/scheduling errors are logged and propagated outside physical/metadata-retention catches, with SAME saved caller cancellation taking precedence and separate error chained. They cannot recreate a recovery owner, poison disposed metadata, restore slots/configuration, replay acceptance or refund. Earlier genuine source/physical/metadata uncertainty retains its original behavior; no global source guard is relaxed.

**WHY.** Original bounded native run5 on BASE0f7c4bd2 produced1F29.68s,wrapper31.934s,PID=PGID63984,actual wait1,no timeout/signal,8317 equal pins. Its exact original finalizer Handle refused SAME F token while its original read/binding/worker/producer/retired were active; native priority=True and send-refusal=None. All required original F releases and G metadata/pop then returned, but both unrelated native child results stayed pending through delivery5 with no timer/no delivery and only the four observed prior retries. Original creators returned. Earlier setup/observer failures and run3 unresolved cleanup remain separate adverse evidence; no past cause or globally-last fairness claim is inferred. Q1 explicitly covered execution only and required this separate F ruling.

**COST, scope and verification.** Finite passive source/return references and exact original receiving checks may conservatively refuse changed/custom source. A failed new post-release notification can leave pending results until a later genuine trigger/recovery; it is visible failure, not physical custody. Original direct-F result shape and native inactive-loop/disposal semantics remain. Genuine successor ownership is independent. Verify the same native RED journey GREEN with actual post-release retry/Handle/delivery/usage, direct F without G, source/association refusal before release, actual original worker/metadata ordering, unexpected notification error plus original cancellation, and original permanent shutdown with no new admission. Keep original10/4/5/60 bounds and primary/cleanup distinctions; do not fabricate retry/completion/owner/source, reset guards, rerun equivalent failures or broaden to a full sweep. Four-file append/readback precedes behavior; exact owned hunks preserve staged c5/61 and279 foreign bytes. Existing thirteen FULL contracts, scoped re-review and whole Task8 gate remain mandatory. Task38531 stays In Progress with parent AC unchanged.
<!-- WHOLE-FIX1-GOVERNANCE-END -->


<!-- TASK9-ORIGINAL-PLANNING-GENERATOR-BEGIN -->
## Task 9 prerequisite — original planning configuration generator

**WHAT.** At original native planning configuration acquisition, including the pre-phase preparation, retain the actual manager and its originally created generator with the issuing runtime and registry. Before terminal effects require that same manager still selects that live original body on that same runtime. At the existing configuration release receive its original native exit normally, verify the originally entered generator completed, and carry this association through the original F receiving frame before issuing a complete receipt. A different generator of the same code/runtime, a closed generator by itself, absent fields, counts or a copied receipt cannot certify return. On uncertain exit preserve the actual partial disposition. Never close a substitute to recover the original, replay ownership release, or require every other legitimate configuration scope to have returned.

**WHY.** The independent 7dc71ac1 native direct probe deliberately replaced only the planning manager's gen with a second already-entered stock generator on the same runtime. F returned and notified with its original generator still suspended, the second closed and one first-send source active. Only a later fixture exit returned the original generator. This demonstrated a false positive receipt, without evidence of spontaneous substitution, a coordinated run, pending children, extra charges or an actually blocked writer.

**COST AND SCOPE.** One passive planning acquisition association and finite source/state checks can reduce availability on a changed or unproven receiver. Existing worker, root, claim, slot, cancellation, G metadata/pop, post-release error and shutdown-fence obligations still apply. The execution-only configuration transition remains separately owned. Preserve existing wake, accounting, consent and public coordinator contracts. Verify original acquisition and terminal substitution refusals and positive direct/coordinated delivery under the original private bounds. Task 38531 and Task 9 UI stay gated until independent prerequisite review reconciles the runtime signoff.
<!-- TASK9-ORIGINAL-PLANNING-GENERATOR-END -->

## Task 10 qualification amendment — exclusive native shell, 2026-11-04

The mounted Console publishes an armed shell descriptor to the read-only planner. A public phase accepted that descriptor but failed before its first child because execution preparation had omitted the matching provider. At the existing finite task entry, an approved exclusive task explicitly naming `shell_exec` may compose the original Console raw-shell provider with the current session, configuration and captured local binding. The matching raw-shell review hook and local review hook both run. The published original app/runtime shell source, current permission, arm state, exact provider and session must still match. The verifier uses the existing native process, current workspace authority, original executable bytes and actual exit/manifest records. Planning, scoped tasks and explicit private producer inputs keep their existing paths. A plan, approval or restored record grants no shell authority. Qualification includes a revoked arm, kill switch and changed source before launch. This amends the existing ADR-285 implementation; the feature stays disabled by default.

## Task 10 qualification amendment — failed verifier and passive restart, 2026-11-04

A native child reached Done after its required verifier actually exited 7. Its immutable check correctly said failed, but the workflow task inherited the model's successful completion. The coordinator then tried to settle success and retained Recovery instead of releasing a physically drained phase for explicit repair. When the original required evidence is current, complete, cleaned up and has an integer exit different from the approved expected exit, record a submitted completed workflow task as failed. Preserve its original native Done status, submitted digest, exact verifier, both manifests and process exit. Other evidence states retain their own meanings. Existing physical retirement and FULL settlement must finish before repair.

A separate fresh-process crash revealed that passive accounting imported Console and RAG startup inside a history DB read even when every unknown native attempt was already marked review_required by the original startup audit. For that all-review-required case, return conservative unknown before the Console import. Empty, mixed and accepted-row handling, charges and live witness checks keep their existing rules. This change introduces no new startup audit or recovery capability.


## Task 10 qualification amendment — concurrent hook-source validation, 2026-11-04

Actual whole-mounted local read5 recorded two original workflow source checks overlapping on the same native legacy hook issuer. One check's original capture held admissions1 with events0 and no published target. A different workflow thread treated that source-only admission as unfinished hook work and cancelled the child before its read. The original capture then returned normally with admissions0. The native cancelled-row guard correctly refused the subsequent tool call. The separate original-App RED2 regression reproduced that refusal while a real no-target ApprovalRequested admission remained correctly blocked.

Serialize only workflow source validators with an issuer-local reentrant lock spanning the existing cleanup checks and the same original capture's actual return. After waiting, obtain a fresh original hook configuration snapshot and recheck exact current runtime engine and issuer identity, attachment, cleanup and published targets. Preserve configured effectful-definition refusal, v2 checks, original capture and physical hook retirement. Actual event admissions retain their existing conservative refusal. No source/permission owner, runtime engine, cleanup bypass or ordinary hook route is introduced. Original source loss and cancelled native rows continue to refuse.

| Lock | Acquisition and return |
| --- | --- |
| Existing config snapshot lock | Initial policy read releases before the validation lock. A second original policy read occurs after acquiring the validation lock and releases before capture. |
| Issuer workflow validation RLock | Only workflow source validators acquire it. It spans current identity/policy/cleanup checks and original capture return. Event producers never acquire it. |
| Existing issuer Condition | Original capture and cleanup readers acquire briefly under the validation lock. The original capture releases it before source/recovery reads and reacquires only for its actual finally decrement. |
| Runtime hooks/v2 map lock | This correction adds no acquisition and preserves existing checkpoint ordering. |

The validation lock has no reverse acquisition from event production, config snapshots, source reads or retirement. It may wait for an actual finite validator return. It does not reinterpret an unresolved admission as returned. Verify overlapping original checks, held real event capture, changed original source, fresh effectful configuration after waiting, original callback/creator boundaries and bounded whole-mounted real read before returning to the full correction journey. Existing ADR285 and root independent review govern this narrow ordering repair. Preserve the original adverse receipts, including read5's separate MediaDB Creator refusal.


## Task 10 qualification amendment — scheduled retention cache ownership, 2026-11-04

The actual mounted local-provider workflow completed its failed-check, authorized correction, fresh passing-check and independent review journey. Its later Creator retirement refused a MediaDB cache after original App shutdown and executor join. A separate completed-history reopen waited for the original65-second sync-log retention timer. The original retention callback returned, but its exact new MediaDB connection survived on a dead executor thread with a live lease and zero active operations. The shorter8.50-second history case ended before that timer and therefore covered a different lifecycle interval.

Apply the existing ADR126 finite-worker helper at this exact original callback boundary. Submit `run_finite_local_worker`, the same `db.delete_sync_log_entries_older_than` bound method and the same cutoff to the existing `asyncio.to_thread`. The helper uses original owner APIs on the producing worker after the real callback returns. Its existing exclusions preserve borrowed/preexisting caches, foreign threads, custom/in-memory owners, active operations, transactions and uncertain closes. The helper itself, retention semantics, timer schedule and existing App shutdown ownership retain their current implementations. This is an application of existing finite-worker custody, with no new global drain or storage owner.

Qualification retains the original two-store cache regression before correction, current preservation/cancellation controls and the same whole-mounted completed-history/original-timer observation after correction. Main-thread handles may legitimately remain until their original owner closes them. Creator return requires their final actual retirement. Keep restart9's functional proof and its separate failed Creator on its exact earlier source, without replaying the full provider flow for this cleanup-only change.


## Task 10 qualification addendum — available failed-check evidence, 2026-11-04

The Console check detail displays each stored check reason. A complete fresh native nonzero exit previously retained the correct failed status and exit but inherited native_evidence_unavailable. For newly published checks, derive native_evidence from the same strict verified_exit predicate already used for conclusive native evidence. This includes both expected and unexpected integer exits with complete output, proven cleanup and current exact manifests/executable/shell. Other evidence states retain native_evidence_unavailable. Preserve native and workflow status handling, full proof, original submitted_digest, exact replay and FULL transaction. Existing historical records retain their old reason label. This narrowly supersedes the earlier keep-check-fields ruling only for future reason publication, with no history rewrite or additional execution authority.
