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
- `is_tool_call_preauthorized`, `before_tool_dispatch`, `post_tool_call`;
- `prepare_tool_calls` — project-instruction whole-batch preparation (see [console-file-authority.md](./console-file-authority.md));
- skill runtime tools (`read_skill_file` / `install_skill` / `run_skill_script`);
- `fork_chat` / `new_chat` (ADR-150, primary-only);
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

`AgentService._call_with_timeout` runs each call on a daemon thread with chunked joins; on timeout the worker is **abandoned** — it may still complete side effects, so retrying an abandoned call risks running it twice (documented trade-off). The per-call ceiling is clamped to the run's remaining wall budget; a `None` ceiling means "do not dispatch" rather than start-and-abandon. Tools with `ToolExecutionPolicy.DEFINITIVE_AFTER_START` skip the timeout wrapper entirely (cancel-before-start, scrubbed result, terminal notify); the default policy is `BOUNDED_ABANDONABLE`. ADR-067: approval waits re-arm the deadline while human input is pending — waiting on a human never consumes tool budget.

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
through that same owner after history reduction. Cached batch pages retain their
original bytes/revisions; the live view supplies current revision, next cursor,
pending delivered IDs and whether every unit has a disposition. Empty pending IDs
alone do not imply completion. Rejected submissions require correction under the
same ID; completion refuses outstanding rejections. For disposition_conflict only,
empty dispositions at the current revision explicitly withdraw that rejected ID
through the ordinary atomic publish path. Accepted decisions remain immutable;
failed retries retain the conflict's recovery eligibility. Fresh IDs, other
rejection reasons and accepted-ID changes cannot use this path. The runtime preserves bounded
`code_review_submit` error details for the model while ordinary error rendering and
body-free logs remain unchanged.

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

### Capture-only native continuation acknowledgment (ADR097)

`LoopDeps.acknowledge_provider_continuation` is optional and runs only after the existing projected persistence callback returns successfully. It receives the exact original/projected event pair and, for a finished call, its capped model-facing result. The Console adapter retains only current-batch projection provenance. The callback does not replace persistence, signing, transitions or tool-side-effect barriers; a failed or missing acknowledgment issues no usable trace custody. Ordinary and Capture Off callers leave it unset. Managed envelopes remain the store finalizer's responsibility, and raw event bodies never enter generic event serialization or provider kwargs through this seam.
