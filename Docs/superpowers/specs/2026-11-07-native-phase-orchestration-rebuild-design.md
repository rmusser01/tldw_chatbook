# PR3411 native phase orchestration rebuild

- Date: 2026-11-07
- Direction: authorized by the user; implemented and qualified inline
- Task: [TASK-39147](../../../backlog/tasks/task-39147%20-%20Rebuild-PR3411-native-phase-orchestration-on-existing-runtime.md)
- ADR required: yes
- ADR path: [ADR-356](../../../backlog/decisions/356-native-phase-orchestration-runtime-ownership.md)
- Reason: replace orchestration, runtime and recovery contracts while preserving
  [ADR-285](../../../backlog/decisions/285-native-phased-agent-workflows.md)'s product requirements.
- Plan: [ordered implementation](../plans/2026-11-07-native-phase-orchestration-rebuild.md)

## Verified starting point

GitHub read on 2026-11-07: PR3411 is OPEN, head
`c52accf6354a902385cb483aed085d012d94ca2c`, base `dev` at
`a0ae30b7ed9c313fe19cb3a0da5d95ebde182385`. The local PR branch matches.
The working checkout is `dev` at `e08627ac38392fb890984169376b1c242c3a2f3e`;
it already contains an earlier workflow implementation, not the integrated PR
tree. There are substantial unrelated staged/unstaged/untracked files. Neither
the branch nor index is to be reset or replaced.

The old run, **Explore linked functionality**
(`01a18b6f-f7ba-7241-a7c0-704eb5ba9672`), was reported interrupted/notLoaded by
the app before source editing. A process snapshot found no PR3411 test process.
Other active chats exist; the mobile work's recent commands use its separate
worktree. Recheck ownership if these source paths change during this task. Do not
message/control the old run's agents or edit its source exports.

The earlier integrated source export is read-only salvage evidence. It is never
the implementation destination. Compare exact PR and checkout hunks before
salvaging a fix. Measurements from PR head are not checkout line counts.

## Product contract

A Console coding goal runs read-only planning, approved implementation,
verification, independent review and a final report. Initial plan and each phase
need explicit approval of current bounded inputs. Repair is a new approved
attempt. Restart never auto-resumes; resume refreshes permissions, selected root,
plan/control revision, predecessor results, file evidence and remaining budgets.

All agents inherit one selected writable local Workspace binding and use the same
worktree, preserving existing dirty content. No automatic Git mutation or PR
publication. Keep the existing strict plan bounds: 128 KiB plan, at most six
execution phases, twelve task definitions and 64 footprint entries per task.
Reject duplicate/unknown fields and IDs, cycles, forward dependencies and invalid
paths. Finite cumulative workflow settings remain intersections with live limits;
approval waits do not refill counters or consume active-execution time.

## Owners and flow

| Concern | Owner and boundary |
| --- | --- |
| Phase order, approvals, checkpoint state | One ConsoleRuntime-owned coordinator, disposed by runtime, not a view |
| Child composition, accepted launch and cancellation | Existing AgentService native child path and FleetCoordinator |
| Physical resources and capacity | Existing ExecutionOwner and actual producer/resource cleanup |
| Permissions and tool execution | Existing permission store, registry and provider dispatch |
| Task restrictions | Injected LoopDeps validation plus service/provider final invocation |
| Shared-file claims | Runtime worktree activity, attached to actual execution lifetime |
| Durable plan/results/charges | Existing AgentRunsDB workflow/automatic-work stores |
| Display and commands | Existing Console setup/status/UI controller, passive snapshots |

Coordinator flow: persist bounded draft → read-only planning → await current plan
approval → accept phase and reserve task capacity/results → launch eligible tasks
through native service → collect typed completions and actual drain → persist
results/checkpoints → await next approval → final report. Dependencies use completed
and current predecessor results. An independent review uses a separate agent
context from implementation and cannot inherit a self-approval result.

Native acceptance has a clear commit point. Before it, the native launch owner
returns every acquired provisional resource on refusal/cancellation. After it,
runtime owns cancellation and eventual completion even if the coordinator or UI
goes away. A typed outcome distinguishes refused-before-start, accepted,
completed-and-drained and cleanup-unproven. Model success is separate from verifier
success. Completion notification is best effort; cleanup cannot depend on delivery.

No Python source/code/caller inspection participates in these decisions. Ordinary
chats use ordinary native lifecycle and participate in active shared-tree conflicts
without authenticating workflow implementation details.

## Tool and file boundaries

Use an injected repeatable `LoopDeps.validate_tool_call` restriction before
preparation and approval, after hook mutation, and before runtime-intercepted tools
dispatch. Service composition binds the actual run's task scope to that callback.
Keep `guard_tool_calls` at its existing once-per-batch point: the existing Console
guard executes legacy PreToolUse hooks and must not be replayed as a freshness
check. This distinction comes from tracing the actual bridge composition.
The real registry/provider call
also revalidates current permissions, control revision, root and scope after waits.
The pure loop must not import workflow policy. Guard exceptions refuse; hook
identity changes refuse; permission Allow and exemptions cannot override refusal.

Scoped file tasks expose only known bounded effects and cannot escape via spawning,
skills, alternate writers or opaque tools. Planned exclusive work still requires
normal shell/MCP trust and permission. Preserve existing refusals for effectful
hooks/scripts until actual runtime process ownership supports their drain.
Do not remove those refusals merely because attestation is removed.

Acquire complete claims atomically before launch. Read/read may overlap;
write/read and write/write serialize, including aliased/nested paths. Capacity
refusal releases unlaunched claims. Actual worker cleanup retains accepted claims.
No coordinator lock or DB transaction spans a model, tool, callback or human wait.
Writes/edits/patches keep observed-content or observed-absence atomic preconditions;
stale input refuses without automatic destructive retry.

## Restart, history and cleanup

Keep two facts separate: (1) native worker/resource liveness, including surviving
or unproven subprocesses; (2) uncertain prior effects that require review before
dependent work resumes. Old workflow rows alone cannot prove live workers.
Startup uses the existing native startup audit. A real survivor remains protected.
Historical unknown effects hold the affected workflow for reconciliation and
invalidate its evidence, but do not monopolize unrelated scratch authority.

Explicit reconciliation records the current audit/evidence and reviewed disposition,
closes only nonlive historical uncertainty, and removes its own obsolete recovery
claim. It refuses while actual resource cleanup remains unproven. Fresh explicit
resume is separate. Persist known charges and conservatively unknown usage without
continuously charging a nonexistent execution interval after reconciliation.

Preserve plan/result/check history with privacy and size bounds; detailed logs stay
optional. Supported result and change-window removal frees actual quota once and
retains useful history/accounting tombstones. New schema work uses the existing
transactional migration owner and a fresh allocated version, never a rewritten
shipped migration. Old records remain readable but cannot mint live approval.

## Salvage/removal manifest

| Area | Treatment | Required evidence before retaining |
| --- | --- | --- |
| `Chat/console_workflow_preparation.py`, `console_workflow_runs.py`, `console_workflow_wave.py` (9,962 lines at PR head) | Replace orchestration with one coordinator and native service operations | Ordered approvals, duplicate delivery, stop/dispose, setup refusal and resource drain |
| `Chat/console_ordinary_tree.py` | Remove attestation; retain only actual shared-file coordination needed by ordinary turns/children | Inactive ordinary chat and conflicting/live-child cases |
| `Agents/workflow_run_service.py` | Reduce to bounded history/checkpoint/reconciliation operations | Privacy, quota, current revision and supported recovery release |
| `Agents/workflow_task_scope.py` | Retain validated narrowing; remove source/caller inspection | Hook mutation, permission wait, path/exclusion and final invocation controls |
| `Agents/agent_service.py` workflow hunks | Disentangle into native typed child lifecycle; preserve original engine | Inline/threaded children, failed start, cancellation, hooks, capacity and actual cleanup |
| `Agents/agent_runtime.py` | Restore injected policy boundary; remove direct workflow imports | Guard denial, late mutation/revocation and runtime-intercepted tool refusal |
| `Agents/workflow_models.py`, `DB/workflow_runs.py` | Selective retention, not certification | Strict parsing, transactional upgrade, result/window quota and legacy history |
| `UI/Console_Modules/workflows.py`, setup/status widgets and config | Selective retention with current coordinator commands | Existing token language, callable worker admission, approvals, restart and markup safety |
| Execution/resource, local-write and worktree-activity changes | Preserve actual protections; simplify ownership at their native boundary | Stale writes, dirty content, conflict and liveness tests |
| Governance/tests requiring identity-check counts or permanent closed-recovery blocking | Replace with observable contract tests | No reduced permission, budget, revision or actual drain coverage |
| Old QA exports and chronological rulings | Historical evidence only | No copying generated artifacts into production or treating old passes as current |

The six main replacement/simplification files total 16,039 lines at the reviewed
PR head. This is review scope, not a deletion count. Trace callers/config/tests
before deleting any file; the original engine is never wholesale replaced.

Independent fixes need exact hunk checks: child signal-setup failure; parked
pre-tool refusal; sqlite_stat1/optional schema handling; quota overcharge; hook
process-group cleanup after leader exit; onboarding persistence race; terminal
workflow reopening; import-time schema work; Rich markup; Undo first-send copy;
transactional migration. Keep each genuinely useful minimal fix and its behavior
test. Their presence in the PR is not current acceptance evidence.

## Acceptance evidence

Run targeted private-profile tests only. Prove refusal before any provider/tool
side effect, hook mutation outside scope, permission/revision revocation after a
wait, launch setup failure, cancellation and late physical drain, conflict ordering,
stale/absent writes, duplicate approval, quota removal and explicit restart resume.
Exercise one complete native Console planning → approved execution → failed check
→ approved repair → independent review → final report → restart/history journey.
Check ordinary chat with workflows disabled and with unrelated recovered history.
Record exact source, commands, results and limits; do not imply full-provider or
full-suite certification from isolated probes. Keep TASK-39147 In Progress until
all acceptance criteria and repository DoD are met.

## Exact independent-fix disposition in this checkout

The pinned PR has later unrelated runtime, schema and UI lineages; it is not the
source tree for this replacement. Decisions after comparing the actual hunks:

- **Native child setup/start:** retain the existing launch path, move guard factory
  and entry inside its unstarted-child teardown, and preserve custody when a
  thread actually started before `start()` raised. Native failure tests observe
  actual owner drain, claims and charges rather than a function signature.
- **Pre-tool refusal:** inject repeatable validation before preparation and after
  mutation/approval. Keep the effectful batch guard once. No parked preparation
  runs for a call already refused by policy.
- **SQLite statistics:** salvage only exact engine stat1/stat4 metadata recognition
  for AgentRuns archives and deletion during the installed workflow rebuild.
  Application catalogs and column metadata stay exact. No general schema-profile
  framework or unrelated owner migration was copied.
- **Quota/history:** keep the current bounded store, release task/window and
  interrupted planning result reservations on supported removal, and keep explicit
  recovery review outside payload quota. The PR's larger portfolio-accounting
  rewrite was not copied or certified.
- **Terminal controls:** keep a completed/stopped/closed history from becoming
  active again through generic control or plan publication. Supported result
  removal remains valid. Explicit revision-checked recovery review can close
  older terminal uncertainty without reopening execution.
- **Migration/import:** add the new review field to this checkout's v29 repair-digest
  schema as v30. Use FULL transactional upgrade and static catalog declarations;
  never rewrite a shipped migration. This does not certify the pinned PR's other
  physical schema lineages or its differently numbered v30 archive.
- **Hook process groups:** preserve this checkout's existing native process-group
  kill and positive terminal observation, including after the leader exits. The
  pinned executor also changes vendor protocols/owner composition; those unrelated
  hunks were not copied. Workflow effectful-hook refusals remain in place.
- **Onboarding and Undo first-send:** preserve existing independent code and dirty
  changes. Newer onboarding/readiness and first-send work in the PR is outside this
  task; no whole-file salvage or certification is claimed.
- **Rich/Textual UI:** keep user-derived status/setup content literal, confirmed
  history actions revision-checked and workers admitted as callables. Refused setup
  worker admission must leave Prepare plan retryable without creating a coroutine
  or starting native planning. No new stylesheet or dependency is required.

Failed callback delivery after an actual positive native return is separate from
failed resource cleanup. Returned planning with uncertain usage records bounded
recovery, retains charges, releases its native owner and supports explicit review.
Reversed/unknown accounting clocks preserve the unknown interval instead of
inventing a timestamp or retaining returned workers. Real failed source/storage or
operation exits still retain their actual owners.
