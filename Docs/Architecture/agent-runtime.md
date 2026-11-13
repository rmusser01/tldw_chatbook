# Agent runtime

This document describes the agent execution loop: the pure control loop in `Agents/agent_runtime.py`, the impure service in `Agents/agent_service.py`, budgets and stop conditions, native vs fenced tool calls, sub-agents (fleet), retries and provider fallback, and the run log.

## Authoritative files

| File | Role |
| --- | --- |
| `Agents/agent_runtime.py` | The PURE control loop — no Textual, app, DB, or I/O imports. `run_agent_loop()`, `LoopDeps` |
| `Agents/agent_service.py` | The loop composition/service layer: provider calls, permission gate, sub-agent spawning, `AgentRunsDB` persistence. `AgentService.run_turn()` |
| `Agents/agent_models.py` | Step-kind and outcome constants (`RUN_DONE`/`RUN_STUCK`/`RUN_CANCELLED`/`RUN_ERROR`/`RUN_SUPERSEDED`, `STEP_*`, `TOOL_OUTCOME_*`), `AgentConfig`, `RunBudget`, `ToolCall`, `ToolResult` |
| `Agents/fleet_coordinator.py` | `FleetCoordinator` / `FleetHandle` — concurrent sub-agents |
| `Agents/run_context.py` | ContextVars (`current_run_id`, `use_run_actor`, `use_tool_call_id`) — how run identity reaches providers without a run parameter |
| `Agents/run_log.py` + `run_log_eviction.py` + `run_log_search.py` | Full-fidelity run log (`RunLogWriter`), per run TREE, latched `bind()` |
| `Agents/model_retry.py`, `fallback_chain.py` | Transient-error retry classification; cross-provider fallback (ADR-110) |

## The seam

`run_agent_loop(config, initial_messages, active_schemas, deps)` drives think → (tool) → observe until done/stuck/cancelled. Everything impure is injected through `LoopDeps`:

- `call_model`, `invoke_tool`, `spawn`, `find_tools`, `load_schemas`, `should_cancel`, `clock`, `sleep`;
- `review_tool_calls` (the batch approval seam) and `guard_tool_calls` (restriction-only, runs **before** approval exemptions; exceptions deny the whole batch);
- `validate_tool_call` — repeatable fail-closed restriction before preparation,
  after hook mutation and at final dispatch, including runtime-intercepted tools.
  AgentService injects the current task/registry check. The pure loop imports no
  workflow policy. Keep it separate from the once-per-batch guard, which can run
  effectful legacy PreToolUse hooks; refused candidates do not reach those hooks.
- `is_tool_call_preauthorized`, `before_tool_dispatch`, `post_tool_call`;
- `prepare_tool_calls` — project-instruction whole-batch preparation (see [console-file-authority.md](./console-file-authority.md));
- skill runtime tools (`read_skill_file` / `install_skill` / `run_skill_script`);
- `fork_chat` / `new_chat` (ADR-348, primary-only);
- fleet tools (`wait_agents`, `check_agents`, `merge_agent_worktree`, `discard_agent_worktree`, `send_to_agent`, `report_to_supervisor`, `read_agent_messages`, `drain_mailbox`) — dispatched **in-loop**, deliberately not through `invoke_tool`, because waits must not be killed by the per-call timeout;
- `on_step` / `on_trace_step` / `on_record` — observation hooks, never load-bearing (wrapped never-raise);
- `persist_provider_continuation` / `expand_provider_continuation` — durable provider continuation.

`AgentService` implements these closures. Its constructor takes the `AgentRunsDB`, a `ToolCatalogRegistry`, the `chat_call` provider function, the review/guard hooks, skill runners, fleet coordinator, and capacity inputs. `run_turn()` documents two contracts: the run-log writer is per run TREE and rebuilt fresh per call; and sub-agent rows may still be `running` when `run_turn` returns (`subagents_outlive_turn`) — readers must wait on `on_child_settled`, never on scope exit.

## One turn with a tool call (dataflow)

1. The Console bridge composes the per-run registry and allow-list, builds `AgentService`, and calls `run_turn` on a worker thread (see [console.md](./console.md)).
2. **Loop top**: check cancel; budget checks in order — step → model-turn → wall-clock → token. Each exhaustion records a `STEP_ERROR`, makes one tools-stripped wrap-up call, and ends honestly `RUN_STUCK` (never a fake success).
3. **Fleet steering drain** at the protocol-coherent point: mailbox entries append as labeled user messages; the retained-transcript boundary is set here.
4. **Model call**. On a transient exception: in-place retry (bounded, sliced sleeps honoring cancel), then the ADR-110 fallback chain (sticky provider switch with history projected per protocol; refused mid-continuation); otherwise trace `STEP_MODEL_ERROR` and raise.
5. **Stop conditions**: text with no tool calls → `RUN_DONE`; two consecutive empty responses → `RUN_STUCK` (empty-turn limit).
6. **Tool-call extraction**: native `turn.tool_calls` when present, else the fence protocol (```` ```tool_call ```` fences parsed with strict line-boundary checks so look-alike tags never parse). Results are appended keyed on `tool_call_id` (native) or as user-role "Tool result for {name}" text (fence).
7. **Preparation** (optional): `prepare_tool_calls` may return retry-with-context — deferral rows plus ephemeral rows are appended and the model is re-asked, with no review or dispatch of the deferred batch. Native deferrals carry only role, call ID and content; ID-less fenced deferrals retain the name for text rendering. Internal instruction-delivery row keys stay in runtime state and are removed from prepared provider copies.
8. **Guard**: hook/restriction seam runs first; an exception denies the whole batch.
9. **Review**: pre-authorized exemptions (e.g. Canvas nominal case) pass; otherwise `review_tool_calls` runs once per batch — the approval card round trip. Verdict lookup is per-call-id first, with a name fallback that is load-bearing for fence calls (which carry no call id).
10. **Dispatch** per call: cancel → approval-revoked + `RUN_CANCELLED`; loop detection (repeated identical calls, period-aware) short-circuits; special in-loop tools (spawn, fleet, fork/new-chat, find/load tools, skill tools, run-log tools) execute directly, looked up by name in `SPECIAL_TOOL_HANDLERS` (each handler takes the call and the run's `LoopState`); everything else, including a special tool whose `LoopDeps` callable is not wired (its handler returns `ORDINARY_TOOL`), goes through `deps.invoke_tool`.
11. **Result capture**: the `tool_call` run-log record is written **before** dispatch (a crash still leaves a durable trace); the `tool_result` after; content is capped with a recovery pointer into the run log. Consecutive same-tool failures (≥3) → `RUN_STUCK`.

Catalog dispatch checks the run's allow-list before checking schema disclosure.
An excluded or unknown name keeps the generic `Tool not permitted` refusal.
An allowed tool whose schema is absent instead returns `Tool schema not loaded`
with a `load_tools` instruction that names the tool and reminds the model to
include every catalog tool it wants to retain (ADR-104 working-set replacement).
Ordinary and skill dispatch share `_tool_availability_refusal`; neither refusal
dispatches a tool or invents an authoritative approval denial. With hooks v2,
the service performs this check before hook preparation and returns a typed
blocked/not-started refusal through the existing preparation seam (ADR-252).
The runtime preserves that error while unrelated preparation exceptions remain
generic and fail closed. Runtime tools retain their independent gates.

Ordinary INFO logs retain content-free disclosure breadcrumbs even when the
handler formats only the message: `agent schema request` reports active/runtime
counts, discovery availability and protocol; `agent tool discovery` reports its
match count; `agent schema load` reports requested/accepted/omitted/invalid counts
and detail-budget exhaustion. Accepted names are JSON-escaped and limited to eight
names of 64 characters. Terminal `agent run step` messages include kind, outcome
and a JSON-escaped tool name capped at 128 characters. Opaque run ids correlate
the records. Prompts, search queries, arguments, schema/result bodies and project
contents are excluded from these message strings. These host-side observations
identify the recovery path; they do not prove the provider's response or the
reporter's original incident. See TASK-37234.

Local file-worker execution errors (including crashes, timeouts, spawn/protocol
and cleanup failures) are ordinary failures in both `LocalToolProvider` and
`VirtualCliProvider`, so the same-tool limit applies even when paths change.
Without a conclusive worker receipt their effects remain `uncertain`, which
stops automatic goals immediately rather than permitting another write.
Workspace-root and scratch-authority refusals remain blocked; only authoritative
permission denials count toward the separate denial circuit breaker (ADR-154).

## Tool execution bounds

`AgentService._call_with_timeout` runs each call on a daemon thread with chunked joins; on timeout the worker is **abandoned** — it may still complete side effects, so retrying an abandoned call risks running it twice (documented trade-off). The per-call ceiling is clamped to the run's remaining wall budget; a `None` ceiling means "do not dispatch" rather than start-and-abandon. Tools with `ToolExecutionPolicy.DEFINITIVE_AFTER_START` skip the timeout wrapper entirely (cancel-before-start, scrubbed result, terminal notify); the default policy is `BOUNDED_ABANDONABLE`. ADR-314: approval waits re-arm the deadline while human input is pending — waiting on a human never consumes tool budget.

## Sub-agents and the fleet

`spawn_subagent` is gated by the run's allowed tools and `max_subagents`; child budgets are contained (`contain_child_budget` zeroes the child's own subagent quota — depth-1 only). The fleet switch `[agents] max_live_subagents` (default 3; a cap of 1 falls back to inline synchronous children) runs children on their own threads with model-scope lifelines. Parents steer via `send_to_agent` mailboxes, bounded `wait_agents`, and `drain_mailbox` probes; `redirect_primary`/`steer_primary` cut an in-flight stream while keeping partial output as context. Worktree creation (`isolation="worktree"`) uses the accepted writable named repository binding and fresh identity checks; merge/discard require explicit confirmation (ADR-155). The fixed profile worktree directory uses `Backup_Recovery/agent_worktree_storage.py::prepare_agent_worktrees_base` for independently admitted native preparation while the parent's exact AgentRuns operation remains counted through child handoff or refused-launch cleanup. A new directory admission refusal becomes `storage_unavailable`; it grants no wider source authority and does not silently launch a child.

## Persistence and memory

Steps land in `AgentRunsDB`; the full-fidelity run log is a per-run-tree on-disk TREE with latched `bind()`, eviction, and search (`search_run_log`, `run_log_stats`, `run_log_slice` — surfaced to the agent as read-only tools). Run state is per-run; cross-session memory is owned elsewhere: the pinned personal-context profile block is appended to the system prompt by the bridge (see [console.md](./console.md)), and long-lived "Agent Lessons" cross sessions only through the human-reviewed promotion path (`Agents/agent_lesson_promotion.py` — lesson text stays untrusted evidence and grants no authority by itself).


`Agents/activation.py::guarded` passively qualifies an already loaded stock
injected writer and includes its actual explicit root in the initial fresh agent
admission. Implicit, custom or changed writers keep the admitted global-root
fallback. Conversation-history search in `AgentService` uses
`activation.history_execution` to retain fresh source admission through root
selection, directory probes and segment loading; only the recorded canonical
reader receives that admitted root explicitly. Custom readers retain their
one-argument contract. No admission result is cached (ADR-126/ADR-230,
TASK-34406.2).

For an original Console agent submit, `Chat/console_agent_sources.py` observes
the original bridge, Chat and Workspace databases, MCP permission store and
authorized log root before SessionStart. Its private submit carrier is captured
before the existing bridge worker enqueue; workers reacquire storage admission
in their own task/thread. The stock writer retains that confinement root as its
source after binding, with a receipt for the exact bound directory identity.
Changed writers, roots, databases and late permission sources cannot inherit
the original admission. Native hook admission workers and the MCP dispatch
coroutine preserve the same tuple through their existing worker guards.
The MCP completion observer surrounds guard entry and unwind, so a returned
no-dispatch admission refusal can retire without weakening pending settlement.
Stock writer qualification includes the actual segment-path, write, manifest
and close callbacks, checked before invocation at the AgentService receiving
boundaries as well as the original writer methods.

Retained context checks original references without I/O under bookkeeping
locks; passive permission refresh and real writer operations validate filesystem
identities outside those locks. Strict event source equality and registration
checks remain in force. Detailed logging is still optional: disabling it leaves
the required parent/child rows and accepted context intact. This source repair
uses the full effective project authority for hook identity: normal validated
folder adoption and consent bookkeeping do not change the selected binding or
options. Custom readers retain their complete values. This source repair does
not supply the earlier ordinary physical execution owner or its borrowers
(ADR-285/163, Task8D2 partition A).

## Failure behavior summary

| Case | Behavior |
| --- | --- |
| Transient model error | Bounded in-place retry, then provider fallback chain; refused mid-continuation |
| Continuation contract violation | `STEP_ERROR` + `RUN_ERROR` |
| Guard hook exception | Whole batch denied (fail-closed) |
| Review hook exception | Fail-open for legacy batches; fail-closed when a provider continuation is in flight |
| Budget exhausted | One wrap-up call without tools, then honest `RUN_STUCK` |
| Tool timeout | Worker abandoned; result reported failed; side effects may still land |
| Observation hook failure | Never raises; wrapped |

## Config keys (`[agents]`)

`max_live_subagents` (default 3), `subagents_outlive_turn`, `prune_stale_tool_results` and related prune keys, run-log eviction keys. Run ceilings come from `[console] agent_max_*` (see [console.md](./console.md)).

## Related docs

- [tool-catalog.md](./tool-catalog.md) — how tools become visible and dispatchable
- [console.md](./console.md) — the surface that drives `run_turn`
- [llm-providers.md](./llm-providers.md) — the provider calls underneath `call_model`

## Verified gotchas

1. Fence look-alike tags (```` ```tool_calls ````, ```` ```tool_call_schema ````) must not parse as calls — enforced by line-boundary checks in the fence parser.
2. Name-keyed review verdicts are load-bearing: fence-path tool calls have no call id, so name-keyed batch decisions must still stop every matching call or the gate silently opens.
3. Abandoned tool threads can still complete side effects after a timeout is reported.
4. A sub-agent's DB row may still be `running` when `run_turn` returns — wait on `on_child_settled`.
5. The `tool_call` run-log record is emitted before dispatch, so a blocked or crashed tool still leaves a durable trace.


When a full first-request tool surface does not fit, ADR-253 tries a compact
surface containing catalog discovery and the run's gated progress tools before
falling back to plain chat. The compact surface uses the session/plain prompt,
concise discovery guidance, and the same whole-request budget. Optional fleet,
skill-management and run-log schemas do not crowd out find_tools/load_tools.
A plain plan stays plain through runtime setup, including when a progress inbox
exists. Small or unverified local windows are not enlarged by this fallback.

The built-in read_file/list_directory/write_file family uses file_path or
directory_path; relative paths address private scratch. Passing the fs_* family's
root_alias/path arguments is refused with discovery guidance instead of silently
using scratch. list_directory requires its declared directory_path. To inspect an
admitted project alias, discover/load fs_list or fs_read and use their own schemas.

First-request schema composition loads on demand from `Agents/first_request_schemas.py`;
the frozen plan type lives in `agent_models.py`.
`agent_service.py` retains request sizing and dispatch. Compact plans carry their
input ceiling through schema loading, project startup/nested admission, preview
and final request checks. Late inboxes cannot expand their frozen schema set.
Terminal step metadata logging is in `Agents/tool_disclosure_diagnostics.py`.

### Native retained-review association (ADR281)

The runtime binds a native review service only after minting the actual primary
`CurrentRunActor`; ordinary child spawns cannot replace that primary adapter or
mutate its remaining-child counter. Each retained context records exact later
primary/grant associations under its quota. Review-bound children still use private
capability catalogs and immutable slot attempts. Terminal capture/settlement exits
the review-store lock before notifying the native coordinator, whose transcript
receipt and exact next-slot boundary feed the existing automatic-work ledger.

### Blinded verifier request checkpoints (ADR-281, Task 8)

`code_review_verification.py` projects substantive source claims into unrelated opaque
unit IDs. Exact canonical projections share one unit; the charged scratch ledger
alone holds the original-to-unit map. Each nonidentical unit needs its own disposition.
`CodeReviewService.next_verification_batch` admits at most eight whole units against
both actual remaining input and the existing tool-result ceiling (including the
native 16,000-character ceiling). Source pages use the same model estimator and a
maximum 1,024-token evidence budget. Empty originals produce explicit zero coverage.

`verifier_checkpoint` exposes accepted revision/cursors, pending opaque references
and bounded same-attempt verdict counts. `code_review_history.project_verifier_history` uses existing protocol round
boundaries and continuation ownership; only acknowledged complete rounds can be
replaced. System/task/guidance, unfinished tool pairs and private owner groups remain.
This send-only projection works for one native or fenced child with run logs disabled;
it changes neither the primary eviction gate nor the stored transcript. Settlement
fills missing decisions with `verification_incomplete`; scratch failures still close
the native stream and expansion retains an explicit unresolved fallback.

Primary recipes contain only coordinator operations. Fresh primary `report` pages
include each original's accepted `disposition` and `verifier_report_status` after
V1 starts; pending decisions remain unresolved. The existing source authorization,
whole-record paging and immutable cached pages still apply. The coordinator reads
fresh post-settlement reports rather than replacing decisions from raw child prose
or using verifier-only tools. Reduced verifier history explicitly attributes its
accepted verdict counts to the same authenticated attempt; absence of removed
tool rounds does not mean the verifier never performed that work.

`code_review_read(view="checkpoint")` exposes current verifier protocol state
through that same owner after each accepted disposition batch or history reduction. Cached batch pages retain their
original bytes/revisions; the live view supplies current revision, next cursor,
pending delivered IDs and whether every unit has a disposition. Empty pending IDs
alone do not imply completion. Rejected submissions require correction under the
same ID; completion refuses outstanding rejections. For disposition_conflict only,
empty dispositions at the current revision explicitly withdraw that rejected ID
through the ordinary atomic publish path. Accepted decisions remain immutable;
failed retries retain the conflict's recovery eligibility. Fresh IDs, other
rejection reasons and accepted-ID changes cannot use this path. The runtime supplies bounded
replay guidance for `submission_conflict`: the same actor can replay the original
accepted chunk unchanged to clear only that ID's rejection. The response includes
the stored retry payload only when it fits existing result/input limits; otherwise
it reports `original_replay_exceeds_capacity` without truncating records. Retained
text readback reuses report source authorization, including current permissions and
exclusions, with callbacks outside publication locks. The original acknowledgment,
revision and decisions remain unchanged, and a charged host receipt records the
repair. Unresolved rejections pin verifier history so replaying an old acknowledgment
cannot remove another rejection's repair instructions. The runtime preserves bounded
`code_review_submit` and `code_review_read` error details for the model while
ordinary error rendering and body-free logs remain unchanged. Review identity
omissions on bound child reads/submissions are resolved by
`CodeReviewService.bound_review_id` from the live host binding, with the same
adapter resolution before project-instruction path selection. Explicit IDs remain
strict, primary calls still require an ID, and stale/revoked/unbound children gain
no authority. Child schemas advertise the omission; the underlying wire models
remain strict after resolution. Accepted records and receipts are unchanged.
Invalid source
ranges still refuse; after authority/reference checks their bounded feedback gives
the captured line count and a same-document retry without a guessed range. Verifiers
also receive a live-checkpoint hint. Child source-query schemas advertise only the
captured target's comparison sides (working tree: base/staged/worktree; patch and
committed comparisons: base/head). Invalid sides remain refused with bounded valid
choices after source authority checks; no side is silently substituted. Once every
unit has a disposition and pending checks are clear, guidance directs
verifier_complete at that checkpoint revision,
then return after acceptance. Failure limits, sealing and accounting are unchanged.

Reviewer completion also validates `empty_findings` against that participant's
accepted findings before publishing intent. Contradictions return `invalid_record`
with bounded field feedback and the expected boolean; correction uses the same
submission ID without changing accepted findings. Intermediate coverage can precede
findings, and accepted submission replays remain immutable. Final sealing still
independently checks coverage, successful termination and complete capture.

Final review admission is host-only. `AgentService.review_input_capacity` reuses the
existing message/schema estimator and response reservation for direct requests.
`_StreamingModelAdapter.review_request_admission` captures a thread-scoped reject-only
callback before async dispatch, then invokes it on the actual `PreparedProviderRequest`
after continuation preparation and before streaming. Review preparation disables the
generic safety-window truncator. Actual prepared accounting and effective destination
(including fallback) determine capacity; unverified numeric defaults are unknown.
A direct request with private continuation but no prepared adapter fails closed as
`review_context_reduction_blocked`. No private state is stripped or conversation reset.
Request-level gaps preserve accepted decisions and do not label remaining claims too
large. Ordinary calls have no callback and preserve their existing preparation path.

Task 8 review correction: final prepared admission also retains whether the actual
request still owns indivisible continuation groups. When that retained state leaves
a positive remainder too small for a unit within the 4,096-token absolute claim cap,
batch admission records `review_context_reduction_blocked` and consumes no unit/cursor.
The host conservatively refuses because owner-free fit is unproven; it does not subtract
speculative private-state costs. A unit above the absolute cap remains individually
oversized. The flag is reset for each bound run and refreshed from each actual prepared
request, never inferred merely from native protocol or historical transcript rows.
Public batch reads release their store lock before entering source permission callbacks;
the batch owner retains short authority snapshots and revalidation around approval.


Task 9 review correction: `code_review_source._access` pins the retained context,
snapshots binding/source-state/revision/epoch under the publication lock, invokes
source authority outside it, and revalidates before yielding protected source work.
`CodeReviewService._source_context` composes that ownership for submit/claim, check
preparation/wait and native report reads. Public source reads/instruction-target
resolution must not wrap that helper in another publication lock. Native page reads
reuse `ReviewStore.read_file(offset=..., limit=...)`, preserving no-follow and exact
inode/size checks. Original report records are not truncated to the UI page size;
reference-backed evidence pages prevent repeated citation amplification.


Task 9 action-order correction: an accepted decline and its host-generated terminal
receipt share the existing immutable action chunk/manifest commit. Failed chunk or
manifest publication leaves the action revision, pending receipt and native owner
unchanged. `_submit` applies claim preparation or unstarted cleanup inside its
already-authorized publication scope; it does not re-enter source/UI guards there.
Exact action replay reuses the stored acknowledgment/receipt and existing claim.
A claim whose preparation raises returns no success acknowledgment; its accepted
action identity remains replayable, and existing reservation rollback retains only
proven ownership. A native-accepted invocation rejects later decline and retains
its charges until the actual executor proves cleanup.


### Review continuation and qualification

`ConsoleCodeReviewCoordinator` binds accepted user stage intent and exact live
source/team grants; `ConsoleFleetWakeCoordinator` delivers the exact next slot
through the existing automatic-work ledger. AgentRuns schema 25 stores body-free
boundary/attempt links. Prepared abort can release an unconsumed claim; accepted,
completed and uncertain attempts remain non-replayable. Two reviewers plus
pre-authorized verification therefore launch `[2, 1]` across primary turns on one
finite chain. Later explicit verification is new user work referencing the same
frozen bundle, with a fresh verifier and current permission checks.

The [qualification record](../superpowers/qa/2026-10-18-multi-model-code-review/README.md)
contains native nonzero journeys and scoped regression evidence. Live qualification
is blocked by missing saved destinations. Existing profile-guard failures and
retained Workspace thread-owned handles remain explicit limitations; passing
scripted journeys do not close those gates.

### Capture-only native continuation acknowledgment (ADR337)

`LoopDeps.acknowledge_provider_continuation` is optional and runs only after the existing projected persistence callback returns successfully. It receives the exact original/projected event pair and, for a finished call, its capped model-facing result. The Console adapter retains only current-batch projection provenance. The callback does not replace persistence, signing, transitions or tool-side-effect barriers; a failed or missing acknowledgment issues no usable trace custody. Ordinary and Capture Off callers leave it unset. Managed envelopes remain the store finalizer's responsibility, and raw event bodies never enter generic event serialization or provider kwargs through this seam.

## Native phase workflows (ADR-285 / ADR-356)

The rebuild is tracked by [TASK-39147](../../backlog/tasks/task-39147%20-%20Rebuild-PR3411-native-phase-orchestration-on-existing-runtime.md)
and [ADR-356](../../backlog/decisions/356-native-phase-orchestration-runtime-ownership.md).
The approved product remains read-only planning, explicit initial and each-phase
approval, implementation, verification, independent review and final report in one
selected writable local Workspace binding. Workflows default to disabled.

`ConsoleRuntime.ensure_workflow_coordinator` owns one lazy
`Chat/console_workflow_runs.py::ConsoleWorkflowCoordinator`. Its commands compose
native preparation, AgentService execution and owned storage; it has no separate
agent executor. `start` preserves the immutable request and actual planner/target
selection. `approve`, `pause`, `stop`, `resume` and `revise` check the displayed
control revision. Command identities deduplicate delivery. Generic Console submit
cannot impersonate `ConsoleSubmissionOrigin.WORKFLOW_PHASE`.

`AutomaticWorkLimits.from_settings("workflow_phase")` uses independent `[agents]`
ceilings: eight accepted phases including planning, twelve child launches, 32 model
calls, 500000 budget tokens, 8192 output tokens per call and 900 cumulative active
seconds. Saved ceilings intersect current settings without replenishing counters.
Zero refuses admission. Canonical Settings owns Save/Revert of these values.

### Native admission and return

`_accept_workflow_phase` checks the actual startup owner, session, conversation,
database and FULL acceptance. Saved IDs and history never recreate live custody.
`workflow_task_scope.py` narrows the existing catalog and runtime tools. The
injected `LoopDeps.validate_tool_call` and final provider invocation repeat current
phase/task/run/owner/claim, revision, tool, root and exclusion checks after hook
mutations and permission waits. An Allow does not widen the approved task.

`AgentService._compose_native_child` and `_launch_prepared_child_result` are the
shared native child lifecycle. Ordinary inline/threaded spawn retains its existing
behavior. Workflow preparation registers the actual service, first-request plan,
task authorization and wave. Launch retains the same task reservation and requested
run identity through the native allowance commit. Complete file claims precede
child capacity. Known no-dispatch waits return transient resources and retry the
same preparation; uncertain start or cleanup never replays automatically.
A `Thread.start` error after actual worker entry keeps capacity and charges.

`console_workflow_preparation.py` retains native clients, source/storage scopes
and finite preparation reads. `ConsolePreparationRead` exposes the actual read,
producer future, retirement future and callback outcome. Cancellation waits for
that producer and scope return before delivery. Resource registrations, current
Task/thread, actual owners and positive drain establish return; no function,
code/default/closure identity, caller-frame or descriptor attestation grants it.
A callback exception can return resources normally. A failed scope exit or finish
keeps the owner/claim protected even if the callback itself returned.

Planning owns one received-turn slot and native primary token. It reads the
selected system/project context and task-only required input seed without
constructing optional tool services. It never borrows the parent conversation or
silently truncates mandatory input. An unused transactional refusal releases its
slots only after the finite read returned and actual admission rows are unchanged.
A changed or unreadable admission return remains protected. Successful planning
publishes the bounded plan/accounting under its control-revision CAS before
retirement; it launches no execution tasks.

`console_workflow_wave.py` enrolls actual preparations and snapshot tickets, then
launches dependency-ready tasks through AgentService. A wave retains its own
baseline/ending snapshot dispositions until native clients/tools and required
publication return. Its explicit membership excludes duplicate or detached close
and launch. A cancellation before native entry closes a known unused wave; a
started worker keeps its actual owner. Capacity/activity notifications are
subscribed before readiness is rechecked. Active-time accounting quiesces only
after actual finite reads, owners, claims and wave return, and resumes at real work
admission, rather than passive reads or task reservations.

Pause prevents new launch commits while already committed work settles. Stop
revokes continuation and requests native cancellation, preserving workspace edits.
Resume revalidates clean saved state and requests fresh approval; it does not
reconstruct an execution capability. Failed or stale checks require explicit
repair, keeping the immutable objective, prior results and cumulative spend.

### Required evidence and bounded history

`Agents/workflow_run_service.py` owns history/checkpoint/recovery operations. Its
`call` dispatches supported operation names on the actual registered service under
existing finite owned-storage helpers. It cannot execute a supplied callback in
place of the service operation. Reads and runtime projections share coherent
worker-side SQLite snapshots; no transaction crosses a host await. Superseded
reads refuse instead of adopting a later live run. Issuing Tasks sequence actual
host delivery/discard; equal counters do not hide accounting changes.

Only a registered `WorkflowChildReceipt`, its actual owner drain and required
canonical result/checkpoint can settle a task and release that claim. A model's
check label, terminal run row or returned model text alone is insufficient.
`prepare_task_inputs` supplies file references and complete predecessor handoffs;
preparation and launch recheck content/absence, binding identity and dependency
freshness. Shared scoped writes keep observed-content/absence preconditions.

Approved verification wrappers use the existing raw-shell permission/runtime and
executor. Their bounded private capture precedes optional log truncation, requires
both EOFs and cleanup, and records the actual shell, arguments, wrapper digest and
input/output manifests. Other shell commands do not satisfy approved checks.
Effectful workflow hooks and skill scripts remain refused where native notification
or process/reader cleanup is not owned. Ordinary hook/script behavior is unchanged.

AgentRuns v27 stores immutable checkpoints and inert resume-command history; v28
stores bounded Change Review windows and normalized snapshot references; v29 stores
repair-instruction digests. Windows distinguish unresolved ending publication from
an exact zero-row ending. Each workflow allows at most 64 windows, 64 KiB per
window, and 64 roots. Required references survive pruning until explicit permitted
payload removal. Existing immutable plan/result/checkpoint JSON remains byte-stable.

AgentRuns v30 adds one fixed-size `recovery_reviewed_revision` field. The additive
FULL migration, standalone SQL and installed backup catalog agree. Old histories
upgrade with zero, never inferred review. A positive marker is immutable, belongs
to a closed/removed workflow and cannot exceed its control revision. It consumes
no result-payload quota, so a full quota cannot prevent recovery review and cleanup.

Startup projects the existing native audit into folder-scoped historical recovery
protection. A saved plan's directory-chain identity must match the current root;
renamed/replaced directories and changed symlink targets inherit no fence.
Planning-only history without a saved plan retains accounting review but has no
persisted folder ownership or mutating task effects to project. Installed overlays
recheck their captured directory chain before admission and wave readiness; stale
scopes retire independently of other current roots or live native claims. Checks
and release notifications run outside the activity lock.
Transient read/metadata errors preserve an established scope. A fallback root stat
can prove replacement or absence; inability to read it retains the existing fence.
Missing/replaced historical roots do not create a global opaque claim that blocks
unrelated scratch. Legacy closed history with a zero review marker remains
protected while unsettled. `resolve_recovery` requires that current audit,
current control revision and explicit typed review, and refuses actual native
workers, unfinished preparations/waves or prepared/accepted admissions. It records
review and releases only that workflow's recovery overlay. Live claims and unknown
charges/retirement rows remain unchanged.

When planning returns physically but final accounting or delivery fails, the
preparation hands its already-durable admission/charges to a typed resource-return
receipt and captured folder protection before releasing its owner, configuration,
claim and slots. It then joins recovery recording. If that recording fails, explicit
review retries it with the displayed revision through the same returned resources;
no synthetic native completion is recorded. A committed recovery recording is
idempotent even if its delivery failed. An uncommitted settlement becomes
review-required when storage returns; committed plans/completion and known token
charges remain intact. Shutdown may discard the proven nonlive coordinator entry
without reviewing history, including while recording remains unavailable. Existing
native rows and charges survive for the startup audit. Unknown/reversed clocks keep
honest unknown interval evidence; they do not keep returned workers admitted.
Actual failed source/storage scopes or unfinished tool threads still retain custody.
Planning captures the admitted system-prompt text independently of editable session
settings. Saving the same prompt or a future prompt during execution/Stop cannot
invalidate native unwind; required system-prefix and seed checks remain at actual
provider dispatch.
Generic controls cannot reopen completed/stopped/closed history; recovery review
uses its own revision-checked closing transaction.

`remove_payloads` refuses live work and unresolved history. Terminal clean history,
or explicitly reviewed closed uncertainty, can remove results/checkpoints and
window references exactly once, releasing their reservations. Plans, accounting,
agent-run history and workspace files remain. Console detail provides confirmed
**Review prior effects** and **Remove stored results** commands; stale confirmations
refuse after a fresh revision read. These controls grant no resume authority.
If closing review committed but its response was lost, a fresh-revision retry
acknowledges the immutable review marker without changing already closed/removed
history. Known and unknown accounting share this idempotent path; live native work
still refuses review, and removed payloads are not reopened.

### Ordinary Console coordination

`console_ordinary_tree.py` retains the originating native owner and file footprint
for primary, child and surviving work. It checks actual sessions, configuration,
provider/spec/tool maps, roots, permissions and current claims; transparent wrappers
do not add an implementation-attestation gate when workflows are inactive or
disabled. Custom effects remain opaque until admitted. Source scopes and canonical
row/step/terminal persistence return before the originating tree settles. A late
child, tool or lifecycle effect keeps its actual owner and claim through physical
return. Ordinary automatic wake uses its own empty Context and native authorization.
Runtime disposal fences the registered wake owner before draining workflows and
retiring downstream stores and scratch.

Stop cancels current session/app preparation reads and continues through native
settlement of an already streaming response. A concurrent read does not consume
that visible Stop action. Cancelled read handles stay owned until their physical
callbacks return; reads belonging only to another session remain unaffected.

Cold MCP permission publication retains its actual service, store, permission
candidate, profile binding and path under the originating opaque claim, including
ordinary sends with agent runtime disabled. The capture acknowledges that native
resource receipt once, retaining path/binding and other-source checks without
permission-store implementation attestation. A replaced source or lost original
capture context still refuses before effects.

Native Library construction uses the resource registered by that Console runtime.
Bounded admission captures it and rechecks its runtime/app ownership before
construction; replacement callbacks cannot run under that bounded claim. Unknown
factories, including partials bound to the app, acquire opaque protection before
their initialization effects. Registered read-only Library tools stay bounded,
including through transparent callable wrappers, without implementation identity.

Current rebuild evidence and limits are recorded in the
[implementation plan](../superpowers/plans/2026-11-07-native-phase-orchestration-rebuild.md).
Historical [qualification receipts](../superpowers/qa/2026-11-01-native-phased-agent-workflows/README.md)
refer to earlier source and are not current acceptance evidence.
