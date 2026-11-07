# Agent runtime

This document describes the agent execution loop: the pure control loop in `Agents/agent_runtime.py`, the impure service in `Agents/agent_service.py`, budgets and stop conditions, native vs fenced tool calls, sub-agents (fleet), retries and provider fallback, and the run log.

## Authoritative files

| File | Role |
| --- | --- |
| `Agents/agent_runtime.py` | The PURE control loop — no Textual, app, DB, or I/O imports. `run_agent_loop()`, `LoopDeps` |
| `Agents/agent_service.py` | Loop composition, sub-agent spawning and `AgentRunsDB` persistence. `AgentService.run_turn()` |
| `Agents/agent_request_execution.py` | Actual provider/tool closures (`make_call_model`, `make_invoke_tool`) and native review settlement |
| `Agents/agent_models.py` | Step-kind and outcome constants (`RUN_DONE`/`RUN_STUCK`/`RUN_CANCELLED`/`RUN_ERROR`/`RUN_SUPERSEDED`, `STEP_*`, `TOOL_OUTCOME_*`), `AgentConfig`, `RunBudget`, `ToolCall`, `ToolResult` |
| `Agents/fleet_coordinator.py` | `FleetCoordinator` / `FleetHandle` — concurrent sub-agents |
| `Agents/run_context.py` | ContextVars (`current_run_id`, `use_run_actor`, `use_tool_call_id`) — how run identity reaches providers without a run parameter |
| `Agents/run_log.py` + `run_log_eviction.py` + `run_log_search.py` | Full-fidelity run log (`RunLogWriter`), per run TREE, latched `bind()` |
| `Agents/model_retry.py`, `fallback_chain.py` | Transient-error retry classification; cross-provider fallback (ADR-110) |

## The seam

`run_agent_loop(config, initial_messages, active_schemas, deps)` drives think → (tool) → observe until done/stuck/cancelled. Everything impure is injected through `LoopDeps`:

- `call_model`, `invoke_tool`, `spawn`, `find_tools`, `load_schemas`, `should_cancel`, `clock`, `sleep`;
- `review_tool_calls` (the batch approval seam) and `guard_tool_calls` (restriction-only, runs **before** approval exemptions; exceptions deny the whole batch);
- `is_tool_call_preauthorized`, `before_tool_dispatch`, `post_tool_call`;
- `prepare_tool_calls` — project-instruction whole-batch preparation (see [console-file-authority.md](./console-file-authority.md));
- skill runtime tools (`read_skill_file` / `install_skill` / `run_skill_script`);
- `fork_chat` / `new_chat` (ADR-348, primary-only);
- fleet tools (`wait_agents`, `check_agents`, `merge_agent_worktree`, `discard_agent_worktree`, `send_to_agent`, `report_to_supervisor`, `read_agent_messages`, `drain_mailbox`) — dispatched **in-loop**, deliberately not through `invoke_tool`, because waits must not be killed by the per-call timeout;
- `on_step` / `on_trace_step` / `on_record` — observation hooks, never load-bearing (wrapped never-raise);
- `persist_provider_continuation` / `expand_provider_continuation` — durable provider continuation.

`AgentService` composes these closures; `agent_request_execution.py` owns the actual model/tool execution closures. Its constructor takes the `AgentRunsDB`, a `ToolCatalogRegistry`, the `chat_call` provider function, the review/guard hooks, skill runners, fleet coordinator, and capacity inputs. `run_turn()` documents two contracts: the run-log writer is per run TREE and rebuilt fresh per call; and sub-agent rows may still be `running` when `run_turn` returns (`subagents_outlive_turn`) — readers must wait on `on_child_settled`, never on scope exit.

Definition-time source records support consistency and availability checks at the original producer and receiver; they are not isolation from code running inside the interpreter. Rebinding or wrapping a selected callable, including a call-through wrapper, can trigger its existing refusal, opaque fallback or retained-unknown outcome. Passive in-place profiling that leaves the bindings and executable objects intact is supported. A source match grants no permission and does not prove execution or physical return: original workers, operations, storage scopes, claims and process groups keep their own return obligations. See [ADR-285](../../backlog/decisions/285-native-phased-agent-workflows.md).

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
10. **Dispatch** per call: cancel → approval-revoked + `RUN_CANCELLED`; loop detection (repeated identical calls, period-aware) short-circuits; special in-loop tools (spawn, fleet, fork/new-chat, find/load tools, skill tools, run-log tools) execute directly; everything else goes through `deps.invoke_tool`.
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

`verifier_checkpoint` exposes accepted revision/cursors and pending opaque references
only. `code_review_history.project_verifier_history` uses existing protocol round
boundaries and continuation ownership; only acknowledged complete rounds can be
replaced. System/task/guidance, unfinished tool pairs and private owner groups remain.
This send-only projection works for one native or fenced child with run logs disabled;
it changes neither the primary eviction gate nor the stored transcript. Settlement
fills missing decisions with `verification_incomplete`; scratch failures still close
the native stream and expansion retains an explicit unresolved fallback.

`code_review_read(view="checkpoint")` exposes current verifier protocol state
through that same owner after each accepted disposition batch or history reduction. Cached batch pages retain their
original bytes/revisions; the live view supplies current revision, next cursor,
pending delivered IDs and whether every unit has a disposition. Empty pending IDs
alone do not imply completion. Rejected submissions require correction under the
same ID; completion refuses outstanding rejections. For disposition_conflict only,
empty dispositions at the current revision explicitly withdraw that rejected ID
through the ordinary atomic publish path. Accepted decisions remain immutable;
failed retries retain the conflict's recovery eligibility. Fresh IDs, other
rejection reasons and accepted-ID changes cannot use this path. The runtime preserves bounded
`code_review_submit` and `code_review_read` error details for the model while
ordinary error rendering and body-free logs remain unchanged. Invalid source
ranges still refuse; after authority/reference checks their bounded feedback gives
the captured line count and a same-document retry without a guessed range. Verifiers
also receive a live-checkpoint hint. Once every unit has a disposition and pending
checks are clear, guidance directs verifier_complete at that checkpoint revision,
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

## Native workflow admission (ADR-285)

`AutomaticWorkLimits.from_settings("workflow_phase")` uses independent `[agents]`
ceilings: eight accepted phases (including planning), twelve task launches,
32 model calls, 500000 budget tokens, 8192 output tokens per call and 900 cumulative
active seconds. `workflow_runs_enabled` defaults false. Canonical Settings exposes
these through Console behavior drafts and atomic Save/Revert. Zero refuses admission;
saved ceilings intersect live settings without replenishing counters.

`Chat/console_workflow_runs.py::_accept_workflow_phase` validates native owner,
session, conversation, database and recovered-content activation, then issues
private custody only to the caller that wins FULL acceptance. Stored IDs never
reconstruct custody. `AutomaticWorkContext` and the final tool registry invocation
recheck that custody/control revision. `ConsoleSubmissionOrigin.WORKFLOW_PHASE`
refuses generic controller submission; the app-owned
`ConsoleWorkflowCoordinator` uses the native planning and phase-wave producers.
Workflow work never falls through to manual or fleet policy.

`claim_workflow_task` returns a stable `reservation_id` and `launch_run_id`.
The native AgentService launch must pass the reservation as
`automatic_child_reservation` and the run identity as `requested_run_id` (or bind
the same precreated run). Its existing commit is the exactly-once launch fence.
`bind_workflow_task_run` records that exact native run; repeated claims are lookups,
not permission to launch again. `ConsoleWorkflowCoordinator` and the native
phase-wave producer compose these claims with the existing physical owners;
the passive reservation APIs alone grant no scheduling authority.

Task preparation is now exposed by
`Agents/workflow_task_scope.py::_prepare_workflow_task`. It retains one private
authorization per claimed task on the original phase. `AgentService._run_owned`
accepts that authorization and the already acquired activity claim alongside the
existing child reservation/requested run ID. `_run_one` binds it at the native
owner/run-row seam and filters both catalog and runtime schemas. The final tool
thread carries the same custody; actual dispatch checks the live phase, task,
run, owner and activity claim after permission waits. No workflow scheduling is
enabled by these preparation/binding APIs.

`AgentService._compose_native_child` and `_launch_prepared_child` now hold the
existing native child composition and fleet launch/settlement tail. Ordinary
spawn retains its quota and inline branch; retained continuation enters the same
launch tail. Startup instruction snapshots, writer, routing, cancellation and
fleet bookkeeping are captured for the child rather than borrowed from a later
turn.

Child admission includes original retained-source and operation-owned DB setup.
Every setup outcome signals the waiting parent. Public spawn/resume reports a
failed launch rather than a running handle; workflow returns its failed native
receipt. Refusal teardown elects one owner before refund/notification/finish so
an uncertain callback is never retried by fallback cleanup. A false refund needs
an exact already-released reservation readback. Refused routing and terminal
persistence/readback must return positively before settlement notification and
child finish; retained checkouts remain untouched. A no-body setup error travels
as data through the original source return and into the existing refused-child
teardown, including terminal SQL, fleet error, unused reservation/worktree
cleanup and notification. The handshake alone proves neither admission nor
physical return; an interrupted source/DB unwind retains the child owner. The
private one-use child record accepts only an exact nonce-bound original
pre-acquisition pause or normally returned stock denial, after guard/isolation
return. Both actual receivers must be genuine bound MethodType objects before
reading function/instance attributes; forwarded attributes do not qualify them.
The service wrapper code snapshot protects later continuity, not initial origin.
A DB setup failure needs its own exact original passive boundary and
returned bookkeeping; raw admission and DB observations only veto return.

The tool timeout worker separately records decorated-body and selected-source
entry. A pre-body check/admission error travels as data through the original
covering planning/Goal storage, automatic/workflow scopes and retained-source
return before its unused operation finishes. Recovery independently retains a
failed admission lease and still blocks conflicting recovery. No exception class
certifies return: attempted source, covering storage/hold or operation-finish
uncertainty keeps the original sticky capacity protection. A private annotation
on explicit automatic-policy rejection sites preserves their termination reason
only before body/source entry; unmarked collaborator errors remain generic.
This is outcome metadata, not authority or cleanup evidence (ADR285).
The second automatic check's original DB scope returns before dispatch. The
selected source owns its existing DB managers; extra/standalone DBs remain
operation-owned, and exact worker slots/participants are checked after return.
Callback failures cross exits as data, so an identical cleanup exception still
keeps custody. The original SSH activate scope stays lexically inside dispatch guard/source/DB
ownership, so an unknown request/master/session keeps that exact storage live.
Only its operation finish is deferred until native requests/readers and all
covering contexts and work holds return; a never-activated call is not acknowledged.
The original worker acquires its finalization primary-host token (or a live
sealed survivor's separate retained source) under its own outcome handler, before
returning any work scope. This keeps the tree through the same finish RETURN
and callbacks. A pre-holder setup refusal stays with that original worker and
returns an uncertain public result. Once start is invoked, an exception is
uncertain even before ident or target entry publication: the same original
operation retains its worker/reference, and only that target's positive return
can finish it. The caller captures the Thread class once, qualifies its original
class/new/init implementation, then constructs that same class with the target.
Clean caller cleanup is limited to that stock construction failure before start.
The exact worker enters the preparation's existing worker list before possible
handoff. Registration failure never lets the caller release its references;
pre-ident join is unknown, not an already-returned thread. Original planning and
workflow closure join that same thread through finish callbacks after slot removal. Finish-generated DB
return is checked while the last finalization holder is still retained. The
receiver's own source scopes perform finalization DB work. Planning retains its
original preparation/primary anchor and actual worker list; no new operation or
permission grant is created.

Isolated child launch retains its exact original provider/run/authority and
stock definition-recorded admit/retire receivers before attempting installation.
Successful opt-in admission returns the alias dictionary created by that exact
invocation after its lock exits; current-map reads only veto this witness.
An interrupted installation keeps its original reservation and child owner while
the cancellation propagates. If retirement fails after the child runs, its fleet
and durable result stand, but notification and physical finish are withheld.
Refused settlement retires on that provider even after catalog reselection.
Expected retirement compares both keys, exact authority/cache, and alias ownership
before mutating either map under the existing lock. Partial/unpublished admission,
replacement or orphan state keeps custody and checkout bytes; it cannot retire a
later re-admission. Defaults remain unchanged (ADR285).

The private `_prepare_workflow_child` consumes the bridge's actual
`ConsoleFirstRequestPlan`, target reference and fully wired service, checking
the original task/phase, database, registry and runtime. Its frozen preparation
lives on that original task authorization. Each workflow task owns one native
service; another preparation or later `run_turn` cannot replace its tree state.
The earlier direct `_run_owned` workflow seam also seals this ownership and
retains its supported no-log path. Task8B's private Console preparation consumer
now supplies the real bridge result (see [console.md](./console.md#shared-native-preparation-adr-285));
the finite phase-wave consumer supplies snapshot ownership, and the public
`ConsoleWorkflowCoordinator` owns scheduling through those original producers.
The preparation seams alone grant no scheduling authority.
Preparation detaches the task-only native message seed, including bounded
required inputs/handoffs, and requires the approved instruction row exactly
once. It never borrows the parent chat or silently truncates mandatory context;
The shared producer constructs that seed from Task7-validated results.

`launch_workflow_task` acquires the complete activity footprint before native
child capacity and passes the **same** task reservation and run ID through the
shared launcher. The original authorization captures separate host-only reads
from the actual freshness collector's metadata: task inputs and direct dependency
inputs, outputs, verifier inputs/wrappers and saved native shell references.
Directory sources cover recursive collection. Shell references must still match
the native resolved bash executable; wrappers remain inside the selected binding.
These scopes never expand model tools or paths. Final freshness validation runs
under the complete claim and actual child owner, after checking current source
identities and exclusions, before any child row or allowance commit. A known
predispatch refusal drains that finite owner and settles its claim while retaining
the original unconsumed reservation. `_run_owned` still owns the final allowance commit and
`_run_one` still binds the actual owner/run. Concurrent duplicate launch claims
refuse before touching rows, capacity or clients; completed duplicate delivery
returns the same `WorkflowChildReceipt`. Capacity/activity waits keep original
preparation and reserved allowance available; uncertain dispatch never replays
automatically. The internal host caller catches `_WorkflowChildWait` (a
`PermissionError` subtype, with `worktree_activity` or `fleet_capacity` reason)
and native `CapacityRefused` separately from terminal launch refusal. A fleet
wait with no reserved handle unwinds transient owner/activity without finalizing
the pending task log; retry uses the same authorization and native request. Physical
drain alone does not settle the required task-result journal. The receipt keeps
the original activity claim, including a client-setup failure before run binding,
so the coordinator can settle that exact claim after durable result capture.
A parentless task uses its actual child identity for optional native logs and creates no invented
supervisor or peer sender. Its task-owned writer reuses native manifest/final
flush after model scope and row settlement, once, outside the DB-only launch
operation; a genuine shared parent writer remains parent-owned. Its phase
cancellation probe remains live regardless of ordinary survivor settings.

Workflow tasks with effectful hooks refuse while their notification lifetime is
unowned. Workflow skill scripts refuse in both modes until typed process/reader
cleanup proof exists. Ordinary hooks, scripts and Goal policies keep their native
behavior. These are explicit unsupported paths, not hidden permission overrides.

`quiesce_workflow_phase(..., drained=True)` durably closes a capacity wait without
completing the accepted phase or discarding its claim receipts. Unclaimed and
reserved-but-unlaunched children do not keep the interval open; dispatched tasks
need their durable result and positive retirement, and unknown usage refuses closure.
The coordinator must supply a real native physical-drain observation including tools
and owned effects. `resume_workflow_phase` rechecks acceptance/control/live ceilings
before explicit work; native `admit_call` and child-reservation `commit` also reopen
inside their own FULL dispatch transaction. The tool registry resumes after its live
custody check and immediately before provider invocation. Polls and task reservations
do not resume time. No new generation is consumed by resumption.

Planning settlement accepts `outcome="failed"` or `"cancelled"` with no plan after
positive drain, including after a pause advances control authority. Native attempt
state `completed` means settled; `WorkflowPhaseAttempt.outcome` distinguishes success,
failure and cancellation. Existing planning phase statuses map `failed` to failure
and `closed` to cancellation. This path preserves charges, releases the unused
planning-result capacity, and changes neither current plan nor workflow control
revision. Unknown usage keeps the attempt open. Successful planning still requires
atomic plan publication with the original control-revision CAS.

Workflow hook-source validations serialize on the original legacy issuer's
private validation lock. Each waiter rereads saved hook policy and the current
runtime/engine/issuer identity before checking pending cleanup and entering the
original source capture. That capture returns before the next validator checks
cleanup. Actual event admission and required physical retirement retain their
existing gates.

## Native phased workflow evidence

Ordinary callers read workflow currentness through the resident
`console_ordinary_tree.py` carrier. The heavy preparation module reexports the
same readers and ContextVar; an empty lookup does not load workflow owners.
Runtime and controller retain independent original source records before lazy
consumers load. Nonempty lookups qualify the original module, receiver and
function sources; phase-specific admission, consent and physical retirement
remain with the original preparation owners (ADR285).

`Agents/workflow_run_service.py` is the passive history/checkpoint service for
ADR-285. Its finite `call` owns the complete callback through the existing
`run_owned_db_call` and Goal storage lifetime helpers, including bounded source
preflight. `prepare_task_inputs` supplies source references and complete saved
handoffs; it does not authorize file-body reads. Native preparation and launch
revalidate the exact seed, current binding and dependency evidence.
Metadata-only preparation refuses missing/failed predecessor checkpoints before
issuing task custody or reserving a child launch; phase approval still defers
future same-phase output references. The private Console preparation callback
claims its selected root plus external native shell reads from all prior-phase
review sources and the selected task's direct dependencies before reading bodies.
It rechecks source identities after acquisition and releases only after callback
return. Human review and phase settlement still need coordinator-owned admission.

Only the original `WorkflowChildReceipt` retained on the private task
authorization can bind settlement. The service observes its actual owner's drain
callback and publishes required result, native checkpoint, retirement and active
accounting in one FULL transaction before releasing that receipt's original
activity claim. Known no-worker native failures retain original typed receipts;
ambiguous creation/cleanup retains custody. A model's check labels or a terminal
row alone cannot certify completion.

Approved `VerificationSpec` wrappers are observed through the existing raw-shell
provider/runtime/executor. The bounded sanitized capture precedes diagnostic
preview/log truncation, and completeness requires both actual EOFs and cleanup.
Wrapper source/arguments, selected cwd, resolved shell and input/output manifests
are rechecked. Other shell commands do not satisfy expected checks. Skill
scripts remain unsupported, and ordinary raw-shell behavior is unchanged.

Schema v28 adds immutable `workflow_checkpoints` and inert
`workflow_resume_commands`. Historical absent-field JSON remains byte-stable.
Schema v29 adds workflow-only `workflow_change_windows` and one normalized
`workflow_change_window_refs` table. Windows record the original phase, bounded
roots, once-bound actual member anchor and separate B/E dispositions. The expected
E reference count is NULL while unresolved, and 0–64 after conclusive publication;
zero-row E and unused B therefore remain distinguishable from missing rows.
Each workflow admits at most 64 windows, each reserves 64 KiB in the existing
payload budget, and each carries at most 64 roots (4096 UTF-8 bytes per root,
32768 bytes serialized). Native E rows, refs, count and disposition commit in one
FULL transaction. Late E never rewrites immutable task checkpoints. Canonical
archive validation checks the exact phase/anchor/ref joins and required count;
restoration creates no live membership. Active/unresolved windows prevent payload
removal; settled required refs survive pruning until permitted payload removal
leaves bounded tombstones and releases their snapshot refs.
Required Change Review references protect their native snapshots from pruning;
settled explicit private-payload removal retains accounting tombstones.
`prepare_decision` reviews upcoming execution; `prepare_quality_review` and
`review_quality` handle completed results. Recovery carries a different tag.
The existing fleet startup owner passes its single audit to workflow projection.
Only prepared unused or explicit closed/drained quiescence is clean; accepted
unmarked/open/unknown intervals stay recovery-required.

`resume` validates saved clean state and returns awaiting approval without an
execution capability. Private `WorkflowPhaseAuthorization._control` serializes
Pause/Stop with the original `_run_owned` child allowance COMMIT. Only the exact
committed task, owner, activity claim and service can continue at that persisted
Pause revision, including the prebind interval. New preparation and commits remain
strict. Stop revokes this in-memory continuation and requests original cancellation
outside the finite mutex. The fence preloads live policy and the activated SQLite
handle; the original FULL SQL transaction and storage guards remain authoritative.
Definition-time function records bind the finite producer and receiver chain to
its original control or child-commit operation; current replacement methods cannot
become native issuers.

`WorkflowRunService.settle_execution_phase` consumes original returned task, wave,
frame and required checkpoint evidence. Only conclusively never-run claims may
close without results; precreated rows and committed allowances are not unused.
Fresh successful siblings keep their exact result/checkpoint references and spend.
Failed, stale or unverified outcomes require repair. `_accept_remaining_workflow_phase`
validates the complete remaining decision under an original current OPAQUE read
claim and native nonchild operation. Its worker retains source/storage custody
through host acceptance/discard and actual finish RETURN; new custody is delivered
only after physical retirement. Repeated-phase ID acceptance cannot bypass that
receiver. A saved PREPARED retry refuses new custody after comparing the complete
original command and decision; an accepted exact duplicate remains inert. Each
read proves its original worker-local DB handles returned after every native
scope, retaining the same owner/claim/cell if native close is uncertain. During
shutdown, the same accepted but unused cell performs only its required cancelled
settlement through the existing tracked reader. It keeps its original root/claim
until terminal SQL, source/storage/handle and physical worker return are proven.
Known post-return cancellation preserves that proof and is re-raised after
retirement; missing proof never releases custody.
Caller command identity is stored verbatim; the unique decision at R and
attempt at R+1 are committed together. Only the never-run complement gets new
ordinals/charges, and final quality checks retained plus new results. The public
Console coordinator composes these native producers; Console setup, status and
history views call that coordinator without owning execution. Restored history
does not adopt an old owner or grant scheduling authority.

### Public native workflow coordination

`ConsoleRuntime.ensure_workflow_coordinator` lazily owns the
`Chat/console_workflow_runs.py::ConsoleWorkflowCoordinator`. `start` preserves the
complete immutable `WorkflowRequest`, resolves every execution target through the
actual session/configured gateway selection, and retains the original selected
planner separately. The planner need not be an execution target. Native readiness,
full target equality, source, binding and admission checks precede acceptance and
billing. Original `prepare_workflow_planning` and `run_planning` retain planning
custody through required plan/accounting and physical return; planning does not
launch approved tasks. Before initial or repair preparation returns, the public
coordinator retains the actual pending invocation, issuing Task and prepared
object. Stop cancels that same issuer and waits for its original finite reads and
unused cleanup. Only an exact returned planning receipt can abort a still-PREPARED
attempt and release its reserved generation; accepted charges are never refunded.
Missing read/receipt or uncertain source return retains the original pending run.

`approve` uses the same original E acceptance cell for truthful initial or
remaining decisions. Exact command duplicates return their original outcome;
stored workflow/decision/attempt IDs never supply execution authority. An original
pending cell survives cancellation, COMMIT, host handoff and shutdown until its
same read/producer/retired/source/storage/owner/claim returns are proven. A
conclusive pre-prepare refusal releases admission only after those positive
returns; a missing reader or entered PREPARED attempt retains custody.

The accepted execution driver prepares finite native tasks and composes
`console_workflow_wave.run_workflow_wave`. Earlier-phase dependencies become
eligible only through the accepted decision's exact completed, drained and fresh
result/checkpoint references; original native preparation and launch repeat
freshness checks. Capacity/activity/fleet notifications
are subscribed before readiness recheck. A genuine no-dispatch wait retries the
same preparation/TaskAuth only after both original scope/operation returns and
wave B/E settlement; negative or uncertain return markers retain the original
preparation, operation and fence. Workflow child membership excludes these tasks
from ordinary fleet automatic-wake synthesis without changing unrelated fanout.

Direct and wave launches retain an invocation-local native result through their
original storage and execution scopes. Only that invocation's created operation
can be finished; a duplicate receipt cannot finish another launch. Enrollment,
launch and close cooperate under the existing task-authorization lock, with
reservation and finish callbacks outside the lock. Unknown scope return or an
invoked-but-unreturned bridge Thread.start retains original custody. Ordinary and
no-bridge launch cleanup keep their existing contracts. Native binding and close
check their original source receivers before invoking them.

A capacity wait stops active-time charging only after the same finite preparation
read, producer, storage scopes, owner/claim and actual returned wave B/E are
positively proven. Its closed interval retains durable `retirement=unknown`; only
the same live phase's exact idle-return witness satisfies that specific accounting
uncertainty. Database-only readers and restart remain conservative, including
native attempts recovered to `review_required`. Work consumes the idle witness
before the original admission reopens charging. A private original cleanup
receiver can charge already-owned cleanup after revocation or budget exhaustion;
it grants no new work and cannot prevent physically authorized cleanup if
accounting fails. A proven current-clock OPEN interval continues charging even
when a later wrapper fails: an exact running clock is separate from final total,
physical return and retained recovery.

Ordinary automatic wake delivery starts its independently authorized Task with
an empty `contextvars.Context`; child-thread execution guards cannot flow through
retry Handles into that Task. The original delivery still issues its own native
AutomaticWorkContext, which propagates through unchanged runtime custody. Native
fleet-attention callbacks own only newly opened same-thread file-backed storage
handles; borrowed handles, active transactions and lazy/custom getter timing keep
their original contracts.

After exact execution return and release of its original intake and primary
token, G notifies the existing pending-wake receiver. Notification failure stays
separate from physical-return failure and cannot restore released custody.

Planning also retries pending fleet results after its worker, root, worktree
claim, configuration, received turn and primary token have all returned. A direct
`_PreparedWorkflowReply.run_planning` calls the existing wake receiver itself.
For a coordinated plan, `ConsoleWorkflowCoordinator._plan` first receives that
same return, completes final metadata publication and removes the original run.
An uncertain return leaves the run in recovery without notifying. A notification
error after release stays visible; if the caller was already cancelled, the
original cancellation carries the notification error as its cause.

Runtime disposal permanently fences the original native wake receiver before
closing G, including callbacks already posted on the host loop; fence failures
and caller cancellation retain their separate error and cleanup semantics.

Explicit workflow-ID Pause stops new launches while committed native work settles.
Read-only planning Pause waits the original plan return before its paused CAS.
Stop revokes through the original native control fence and drains original work;
unknown accounting/required return remains recovery-required. Controls themselves
use the original tracked offloaded reader and a same-run return Event, which also
fences planning root release. Resume requires fresh review and either original
clean awaiting-resume state or the actual latest positively drained planning
attempt with its saved latest plan; a later execution attempt cannot masquerade
as that planning state. A positively verified failed command produces a failed workflow task even when
the model finished normally. Its immutable native exit, manifests and original
submitted digest survive the explicit repair. Failed/stale results require explicit `revise`, preserving
request, history, spend and exact repair instruction. Completed execution remains
awaiting result review until a matching native quality decision accepts it.

`subscribe` delivers the distinct bounded `WorkflowRuntimeSnapshot`; full private
history remains `get`. Both reuse original tracked passive reads. Full history
and runtime projection share one worker-only SQLite read transaction; no SQL
transaction spans a host await or callback. Existing issuing Tasks sequence each
read and synchronous host delivery/discard through the previous invocation's
actual disposition Future. Equal revision/progress counters remain deliverable
because native accounting and retirement can change without advancing them.
Captured run/phase (or absence), definition-time producer records and source
identity are rechecked: superseded publication discards, while `get` explicitly
raises `workflow_snapshot_superseded`. No replacement overlay is adopted.

Final drivers retain their original run through actual metadata worker and host
return. Only the same final cell's positive original proof permits synchronous
release; source/producer failures preserve both caller cancellation and original
error, keep recovery sticky, and prevent downstream disposal. Close seals
subscriptions/external reads, drains original commands/drivers and admitted host
dispositions without joining unrelated caller work, and refuses unresolved
custody. The original runtime-only cancellation handoff continues its existing
disposal before redelivery after positive G return; independent direct coordinator
close retains normal cancellation semantics. Headless native tests qualify these
specific paths, not mounted UI, remote providers, performance or whole-system
reliability.


### Ordinary tree integration and qualification boundary

AgentService consumes one original early Console owner at the real primary
row, attach actual ordinary children immediately after native reservation and
before precreated rows/clients/models/threads, and report original canonical
row/step write outcomes to private activity settlement. Existing retries and
readback remain; False, RunOutcome, Fleet terminal state and optional logging
do not prove durable settlement. Survivors retain the originating tree. Genuine
late child/tool/lifecycle holds follow original required/optional policy before
causing effects. The gated storage-source and SSH request contracts above remain
governed by ADR285. The Console coordinator and mounted controls are present.
[Qualification receipts](../superpowers/qa/2026-11-01-native-phased-agent-workflows/README.md)
separate deterministic native, fresh-process and real-provider results.


Provider-direct Console turns intentionally omit `_AgentSources`, while their original ordinary admission still owns deferred MCP selection. `MCP/console_snapshot.py::_CapturedSources.permission_owner` retains its exact native permission-publication tuple on that same live `_OrdinaryTurn` when no agent source exists. The existing `console_configuration_preparation.consume_native_mcp_permission_publication` checks the current admitted turn and its controller/app/store/session witness, then the unchanged candidate/method/path/binding/source comparisons before advancing the original witness. Agent/workflow source publication keeps its existing path. A completed selection publishes `mcp_definition_capture="captured"`; incomplete metadata remains conservative. See [ADR285](../../backlog/decisions/285-native-phased-agent-workflows.md).


### PR3411 integrated definition owners — Task13, 2026-11-07

| Names | Producer / old entry / actual defining globals | Receiving/capture ownership |
| --- | --- | --- |
| `_substitute_skills`, `_skill_candidates_from_context`, `_skill_blocked_candidates_from_context`, `_render_skill_bundle_block` | Controller full-signature lazy entries; plugin private bodies with plugin globals and explicit live controller reads | Existing skill invocation callers; original splitter/planning/plugin receiver records remain separately captured at their existing ends. No new trust enrollment. |
| `request_initial_hook_review`, `_hook_review_presentation_current`, `_await_hook_review_work`, `_execute_hook_review_operation`, `_start_hook_review_operation`, `apply_hook_review_action`, `resolve_initial_hook_review`, `_drain_hook_review_operations` | Hook-review functions bound directly to old runtime class slots; hook-review globals, late named runtime reads | Modal/wiring/received-dispatch and three drain callers; original runtime/host physical-operation custody. Selected-close four-method source tuple stays runtime-owned. |
| Native selected-close controller/runtime tuples | Original classes, constructors, subjects, pending type, ticket descriptors and exact lists unchanged | Original `_legacy_close_ready` and runtime finalizer loop check tuple/slot/function/checker before lock/fence/witness/physical-return consumption. |
| Skill publication | One approved `Skills_Interop` ContextVar with original stock assigner/observer frames and just-assigned field | Configuration preparation marker/thread/task/masking/reset and finish receiver remain unchanged. No substitution helper intervenes. |
| `_catalog`, `_metadata`, two helper limits | DB common helper owns executing bodies/globals; Backup names are aliases | Backup local calls/prechecks remain live Backup slots; DB/workflow sites use DB slots. Same passed restrictions and caller-owned transaction. |
| Media SQL tuple | New no-import data leaf; old service alias before class; no function moves | Original service global lookup/transaction, original class/descriptors/full collectors. No record issued by leaf. |
| Personal SQL constants/tuple | Existing models data; old repository aliases | Original repository code/globals/schema/encryption/transaction/capture remain owners. |
| `ConfigLoadFailure` | Existing startup diagnostic leaf constructs one class with config module set before metaclass work; old config alias | Config retains failure state/getter/bootstrap; app warning reads the same value; old-path pickle and class/constructor annotations separately qualified. |
| Generated-video factory | Old app wrapper's reader has actual app or `__main__` globals; factory body belongs to existing video store module | Exactly one selected store/retention; logger is read only after retention raises; no app reimport or authority object. |

The runtime selected-close record keeps `__init__`, `ensure_run_hooks`, `close_session` and `_close_session_after_voice_drain`. The controller record keeps `__init__`, `begin_session_close`, `_legacy_close_ready` and `finalize_session_close`, plus the original class, subject, pending type and ticket descriptors. Their capture points, receiving qualifiers, lock/fence/witness checks and physical-return owners remain unchanged.

The plugin module's deferred-annotation import can change compiler flags of existing functions. Their actual source/capture behavior and any flag consumer must be qualified; the move does not promise byte-identical executables. Old controller entries retain their complete annotation/signature contract; the four new private implementation entries have unannotated outer signatures and truthful plugin globals.


### PR3411 final trace and Notes owner decision — Task13, 2026-11-07

The post-readiness legacy trace maintenance receiver remains wholly defined and captured in `Chat.console_trace_maintenance_runtime`, including its original nested retained/shielded DB await, unknown-return and cancellation decisions, parking/source/callback/physical-return obligations. One exact dated 228-line size pin retains those existing boundaries after the initial 193-line extraction target proved insufficient for the inspected readable timing-preserving alternatives. The file is byte-identical to private base `81b767712da20cc9e2a5746a21e86c3d1de9c283` (SHA-256 `d3be9e8f104b137f395a4d0709a9e45b542f69e04df226c8a68d9e0201f0edfb`). This deliberately records 35 lines of size-ratchet debt, grants no spare lines, executable trust, permission or return proof, and changes no runtime bytes or cold/performance/timeout/concurrency/census cap. No other upward row or layer exception follows. Reconsider only with a separately reviewed coherent extraction preserving every original receiving, timing, cancellation, callback, physical-return and cold contract, then lower the pin to its measured genuine result. Further growth requires its own decision.

The exact one-row size decision preserves the original definition-time receiver record and both actual consumers (`console_agent_sources._check_original_bindings` and `_PreparedWorkflowReply._planning_native_sources`). It introduces no helper, factory, proof or worker owner. The initial193/red measurement remains part of the QA record; a successful228pin does not rewrite that evidence.

### PR3411 native test observer lifetime

The three private wave controls acquire a stdlib monitoring resource during cooperative trusted-startup quiescence, before test/product imports, and retain the same inactive ID until the original arming points. This is a test resource, not production source or permission authority. Existing adapters and callback bodies remain; actual producer return precedes callback admission closure/drain and final alias restoration. The public API cannot read callbacks without exchange, so arbitrary live foreign observers/C threads/audit hooks are outside this qualification. Known conflicts refuse before exchange, failed setup releases only owned inactive state, and unknown armed return remains invalid. ADR285 records the scope and costs; no timeout or numeric gate changes.
