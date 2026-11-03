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
