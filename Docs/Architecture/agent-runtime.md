# Agent runtime

This document describes the agent execution loop: the pure control loop in `Agents/agent_runtime.py`, the impure service in `Agents/agent_service.py`, budgets and stop conditions, native vs fenced tool calls, sub-agents (fleet), retries and provider fallback, and the run log.

## Authoritative files

| File | Role |
| --- | --- |
| `Agents/agent_runtime.py` | The PURE control loop — no Textual, app, DB, or I/O imports. `run_agent_loop()`, `LoopDeps` |
| `Agents/agent_service.py` | The only impure `Agents/` module: provider calls, permission gate, sub-agent spawning, `AgentRunsDB` persistence. `AgentService.run_turn()` |
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
7. **Preparation** (optional): `prepare_tool_calls` may return retry-with-context — deferral rows plus ephemeral rows are appended and the model is re-asked, with no review or dispatch of the deferred batch.
8. **Guard**: hook/restriction seam runs first; an exception denies the whole batch.
9. **Review**: pre-authorized exemptions (e.g. Canvas nominal case) pass; otherwise `review_tool_calls` runs once per batch — the approval card round trip. Verdict lookup is per-call-id first, with a name fallback that is load-bearing for fence calls (which carry no call id).
10. **Dispatch** per call: cancel → approval-revoked + `RUN_CANCELLED`; loop detection (repeated identical calls, period-aware) short-circuits; special in-loop tools (spawn, fleet, fork/new-chat, find/load tools, skill tools, run-log tools) execute directly; everything else goes through `deps.invoke_tool`.
11. **Result capture**: the `tool_call` run-log record is written **before** dispatch (a crash still leaves a durable trace); the `tool_result` after; content is capped with a recovery pointer into the run log. Consecutive same-tool failures (≥3) → `RUN_STUCK`.

## Tool execution bounds

`AgentService._call_with_timeout` runs each call on a daemon thread with chunked joins; on timeout the worker is **abandoned** — it may still complete side effects, so retrying an abandoned call risks running it twice (documented trade-off). The per-call ceiling is clamped to the run's remaining wall budget; a `None` ceiling means "do not dispatch" rather than start-and-abandon. Tools with `ToolExecutionPolicy.DEFINITIVE_AFTER_START` skip the timeout wrapper entirely (cancel-before-start, scrubbed result, terminal notify); the default policy is `BOUNDED_ABANDONABLE`. ADR-067: approval waits re-arm the deadline while human input is pending — waiting on a human never consumes tool budget.

## Sub-agents and the fleet

`spawn_subagent` is gated by the run's allowed tools and `max_subagents`; child budgets are contained (`contain_child_budget` zeroes the child's own subagent quota — depth-1 only). The fleet switch `[agents] max_live_subagents` (default 3; a cap of 1 falls back to inline synchronous children) runs children on their own threads with model-scope lifelines. Parents steer via `send_to_agent` mailboxes, bounded `wait_agents`, and `drain_mailbox` probes; `redirect_primary`/`steer_primary` cut an in-flight stream while keeping partial output as context. Worktree isolation (`isolation="worktree"` + merge/discard worktree tools) is user-confirm gated.

## Persistence and memory

Steps land in `AgentRunsDB`; the full-fidelity run log is a per-run-tree on-disk TREE with latched `bind()`, eviction, and search (`search_run_log`, `run_log_stats`, `run_log_slice` — surfaced to the agent as read-only tools). Run state is per-run; cross-session memory is owned elsewhere: the pinned personal-context profile block is appended to the system prompt by the bridge (see [console.md](./console.md)), and long-lived "Agent Lessons" cross sessions only through the human-reviewed promotion path (`Agents/agent_lesson_promotion.py` — lesson text stays untrusted evidence and grants no authority by itself).

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
