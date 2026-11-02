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
